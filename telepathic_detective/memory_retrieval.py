"""Embedding retrieval over a suspect's memory corpus.

Embeddings are mean-pooled hidden states from a middle layer of the same
Qwen model that plays the suspect, so retrieval adds no dependencies and no
extra model load. Per-text embeddings are cached by content hash, so a stable
memory bank is embedded once per server lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any


FUNCTION_WORDS = {
    "about", "after", "all", "also", "am", "an", "and", "any", "are", "as",
    "at", "be", "been", "before", "being", "but", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers",
    "him", "his", "how", "if", "in", "into", "is", "it", "its", "me", "my",
    "not", "of", "on", "or", "our", "ours", "she", "so", "tell", "than",
    "that", "the", "their", "them", "then", "there", "they", "this", "to",
    "us", "was", "we", "were", "what", "when", "where", "which", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
}


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    text: str
    privacy: str
    tags: tuple[str, ...]
    # "episode" (a specific remembered event) or "habit" (a general routine).
    kind: str = "episode"


@dataclass(frozen=True)
class RetrievedMemory:
    entry: MemoryEntry
    score: float


def memory_is_relevant(
    *,
    semantic_score: float,
    lexical_bonus: float,
    min_semantic_score: float = 0.4,
    min_lexical_bonus: float = 0.2,
) -> bool:
    """Per-entry relevance: the same bar the null gate applies globally.

    Channel slots are limits, not quotas — an entry that is neither
    semantically nor lexically evoked by the question must not surface just
    because a slot is open.
    """
    return (
        semantic_score >= min_semantic_score
        or lexical_bonus >= min_lexical_bonus
    )


def retrieval_passes_null_gate(
    *,
    max_semantic_score: float,
    max_lexical_bonus: float,
    min_semantic_score: float = 0.4,
    min_lexical_bonus: float = 0.2,
) -> bool:
    """Return whether this question meaningfully evokes the memory bank."""
    return (
        max_semantic_score >= min_semantic_score
        or max_lexical_bonus >= min_lexical_bonus
    )


def content_words(text: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[a-z][a-z'-]+", text.lower())
    } - FUNCTION_WORDS


# Questions anchored to the incident window should prefer episodic memories
# over habits: a routine cannot answer "where were you when the alarm sounded."
INCIDENT_ANCHOR_WORDS = {
    "night", "alarm", "breach", "incident", "0300", "moment",
    "pressure", "failure", "failed", "exactly",
}


def question_is_incident_anchored(question: str) -> bool:
    return bool(content_words(question) & INCIDENT_ANCHOR_WORDS)


def extract_entities(text: str, cap: int = 8) -> list[str]:
    """Capitalized, non-sentence-initial tokens — the referents a follow-up
    question might point back at ("Ilya", "Sector", "Checkpoint")."""
    entities: list[str] = []
    for token in re.findall(r"[A-Za-z][a-z'-]+", text):
        if not token[0].isupper():
            continue
        if token.lower() in FUNCTION_WORDS or token in entities:
            continue
        entities.append(token)
        if len(entities) >= cap:
            return entities
    return entities


def build_retrieval_query(
    question: str,
    *,
    prev_user_question: str = "",
    prev_answer: str = "",
    max_answer_entities: int = 8,
) -> tuple[str, set[str], set[str]]:
    """Three-tier query composition (2026-08-01 consensus).

    1. The bare question, when contentful — generated prose must never steer
       retrieval toward a stale topic.
    2. The previous *player question* for content-light follow-ups ("So you
       did see him?") — the referent lives in what the player asked.
    3. A capped entity list from the previous answer, only when the referent
       still cannot resolve ("Why were you there?" where "there" was named
       only in the answer). Never the full answer text.

    Returns (query_text, question_words, carried_words).
    """
    question_words = content_words(question)
    parts = [question]
    carried: set[str] = set()
    if len(question_words) < 2 and prev_user_question:
        parts.insert(0, prev_user_question)
        carried |= content_words(prev_user_question)
        if len(question_words | carried) < 3 and prev_answer:
            entities = extract_entities(prev_answer, cap=max_answer_entities)
            if entities:
                parts.insert(0, " ".join(entities))
                carried |= {entity.lower() for entity in entities}
    return "\n".join(parts), question_words, carried - question_words


def parse_memory_bank(raw: Any) -> list[MemoryEntry]:
    entries: list[MemoryEntry] = []
    if not isinstance(raw, list):
        return entries
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        entries.append(
            MemoryEntry(
                id=str(item.get("id", f"mem-{len(entries)}")),
                text=text,
                privacy=str(item.get("privacy", "private")),
                tags=tuple(str(tag) for tag in item.get("tags", []) or []),
                kind=str(item.get("kind", "episode")),
            )
        )
    return entries


def apply_channel_slots(
    scored: list[RetrievedMemory],
    *,
    public_slots: int,
    private_slots: int,
    prefer_episodes: bool = False,
) -> list[RetrievedMemory]:
    """Take the best entries per privacy channel, never padding an empty one.

    `scored` must be sorted best-first. Channels are filled independently so
    terse private memories are not outcompeted by chatty public ones, then the
    merged result is re-sorted by score. With `prefer_episodes`, episodic
    public memories outrank habits regardless of score — a soft preference for
    incident-anchored questions (habits still surface if no episode qualifies).
    """

    def take(want_public: bool, limit: int) -> list[RetrievedMemory]:
        channel = [
            item
            for item in scored
            if (item.entry.privacy.strip().lower() == "public") == want_public
        ]
        if want_public and prefer_episodes:
            channel.sort(
                key=lambda item: (
                    item.entry.kind.strip().lower() == "habit",
                    -item.score,
                    item.entry.id,
                )
            )
        return channel[:limit]

    merged = take(True, public_slots) + take(False, private_slots)
    merged.sort(key=lambda item: (-item.score, item.entry.id))
    return merged


# bge-style retrieval models want queries (not passages) prefixed.
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class MemoryRetriever:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str,
        pool_layer_fraction: float = 0.5,
        embedder_model: Any = None,
        embedder_tokenizer: Any = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pool_layer_fraction = pool_layer_fraction
        # Dedicated retrieval embedder (e.g. bge-small-en-v1.5): CLS pooling,
        # trained for cosine — no anisotropy centering needed. Falls back to
        # mean-pooled suspect-model hidden states when absent.
        self.embedder_model = embedder_model
        self.embedder_tokenizer = embedder_tokenizer
        self._cache: dict[str, Any] = {}

    @property
    def has_dedicated_embedder(self) -> bool:
        return self.embedder_model is not None and self.embedder_tokenizer is not None

    def _embed(self, text: str, *, is_query: bool = False) -> Any:
        import torch

        if self.has_dedicated_embedder:
            prefixed = (EMBED_QUERY_PREFIX + text) if is_query else text
            key = hashlib.sha256(f"bge:{prefixed}".encode("utf-8")).hexdigest()
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            encoded = self.embedder_tokenizer(
                prefixed,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                output = self.embedder_model(**encoded)
            pooled = output.last_hidden_state[0, 0].float()
            embedding = (pooled / pooled.norm().clamp_min(1e-8)).cpu()
            self._cache[key] = embedding
            return embedding

        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        input_ids = encoded["input_ids"].to(self.device)
        with torch.inference_mode():
            output = self.model(
                input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        layer_count = len(output.hidden_states)
        layer = max(1, min(layer_count - 1, round(layer_count * self.pool_layer_fraction)))
        pooled = output.hidden_states[layer][0].float().mean(dim=0)
        embedding = (pooled / pooled.norm().clamp_min(1e-8)).cpu()
        self._cache[key] = embedding
        return embedding

    def retrieve(
        self,
        *,
        question: str,
        memory_bank: list[MemoryEntry],
        top_k: int = 4,
        lexical_weight: float = 0.6,
        min_semantic_score: float = 0.4,
        min_lexical_bonus: float = 0.2,
        prev_user_question: str = "",
        prev_answer: str = "",
        public_slots: int | None = 3,
        private_slots: int | None = 2,
    ) -> list[RetrievedMemory]:
        """Retrieve memories for a question.

        Query composition is three-tier (see build_retrieval_query): the bare
        question dominates; prior dialogue is consulted only for content-light
        follow-ups, and generated answer text contributes at most a capped
        entity list. When channel slots are set, up to `public_slots` public
        and `private_slots` private memories are taken independently, so terse
        incriminating memories are not outcompeted by chatty autobiographical
        ones. Slots never pad: a channel with no qualifying entries surfaces
        nothing.
        """
        import torch

        if not memory_bank or not question.strip():
            return []
        query_text, question_words, carried_words = build_retrieval_query(
            question,
            prev_user_question=prev_user_question,
            prev_answer=prev_answer,
        )
        question_embedding = self._embed(query_text, is_query=True)
        entry_embeddings = [self._embed(entry.text) for entry in memory_bank]
        if self.has_dedicated_embedder:
            # Retrieval-trained embeddings are cosine-calibrated; no centering.
            def centered(vector: Any) -> Any:
                return vector
        else:
            # Causal-LM mean-pooled embeddings are anisotropic: a few "hub"
            # memories score high against everything. Centering on the corpus
            # mean before cosine restores discrimination.
            corpus_mean = torch.stack(entry_embeddings).mean(dim=0)

            def centered(vector: Any) -> Any:
                shifted = vector - corpus_mean
                return shifted / shifted.norm().clamp_min(1e-8)

        question_centered = centered(question_embedding)
        # Only contentful question words count toward the lexical bonus;
        # carried words (from the composition tiers) count at a discount so
        # "him" can inherit "Ilya" without stale topics dominating.
        context_words = carried_words

        def lexical_bonus(entry: MemoryEntry) -> float:
            entry_words = {
                word.lower()
                for word in re.findall(
                    r"[a-z][a-z'-]+", f"{entry.text} {' '.join(entry.tags)}".lower()
                )
            }
            current = (
                len(question_words & entry_words) / len(question_words)
                if question_words
                else 0.0
            )
            carried = (
                len(context_words & entry_words) / len(context_words)
                if context_words
                else 0.0
            )
            return max(current, 0.6 * carried)

        scored: list[tuple[RetrievedMemory, float, float]] = []
        for entry, embedding in zip(memory_bank, entry_embeddings):
            semantic_score = float(question_centered @ centered(embedding))
            entry_lexical_bonus = lexical_bonus(entry)
            scored.append(
                (
                    RetrievedMemory(
                        entry=entry,
                        score=semantic_score + lexical_weight * entry_lexical_bonus,
                    ),
                    semantic_score,
                    entry_lexical_bonus,
                )
            )
        scored.sort(key=lambda item: (-item[0].score, item[0].entry.id))
        if not retrieval_passes_null_gate(
            max_semantic_score=max(item[1] for item in scored),
            max_lexical_bonus=max(item[2] for item in scored),
            min_semantic_score=min_semantic_score,
            min_lexical_bonus=min_lexical_bonus,
        ):
            return []
        if public_slots is None and private_slots is None:
            return [item[0] for item in scored[:top_k]]
        relevant = [
            item[0]
            for item in scored
            if memory_is_relevant(
                semantic_score=item[1],
                lexical_bonus=item[2],
                min_semantic_score=min_semantic_score,
                min_lexical_bonus=min_lexical_bonus,
            )
        ]
        return apply_channel_slots(
            relevant,
            public_slots=public_slots or 0,
            private_slots=private_slots or 0,
            prefer_episodes=question_is_incident_anchored(question),
        )
