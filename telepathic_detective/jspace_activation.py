from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable

from telepathic_detective.models import RawFeatureHit


WORD_RE = re.compile(r"[a-z][a-z'-]{1,}", re.IGNORECASE)

READOUT_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "answer",
    "assistant",
    "because",
    "before",
    "being",
    "could",
    "couldn",
    "didn",
    "don",
    "does",
    "doesn",
    "endoftext",
    "from",
    "have",
    "haven",
    "hasn",
    "imend",
    "imstart",
    "into",
    "isn",
    "just",
    "me",
    "more",
    "most",
    "only",
    "other",
    "question",
    "response",
    "should",
    "shouldn",
    "some",
    "system",
    "than",
    "that",
    "their",
    "them",
    "then",
    "think",
    "there",
    "these",
    "they",
    "this",
    "those",
    "user",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "wouldn",
    "wasn",
    "weren",
    "we",
    "you",
    "your",
    "aren",
}

TRACE_STOPWORDS = READOUT_STOPWORDS | {
    # Discourse glue that keeps surfacing as low-information chips
    # (2026-08-17 quality pass): these words rank high in the lens but tell
    # the player nothing about what the mind is doing.
    "according",
    "accordingly",
    "actually",
    "additionally",
    "nonetheless",
    "particularly",
    "regarding",
    "regardless",
    "relevant",
    "respectively",
    "saying",
    "says",
    "stating",
    "all",
    "and",
    "any",
    "anything",
    "anyone",
    "both",
    "but",
    "days",
    "during",
    "else",
    "even",
    "ever",
    "everything",
    "fortunately",
    "had",
    "has",
    "having",
    "did",
    "doing",
    "done",
    "prior",
    "since",
    "today",
    "tonight",
    "her",
    "hers",
    "herself",
    "him",
    "his",
    "himself",
    "including",
    "she",
    "hours",
    "inaccur",
    "itself",
    "myself",
    "never",
    "nobody",
    "nor",
    "not",
    "nothing",
    "once",
    "or",
    "ourselves",
    "perhaps",
    "priorit",
    "rather",
    "regarding",
    "simply",
    "so",
    "something",
    "thanks",
    "themselves",
    "ultimately",
    "unfore",
    "unfortunately",
    "whom",
    "who",
    "yet",
}

# Motif -> canonical lexical family -> prefix roots. A family is one piece of
# evidence: all morphological variants matched by its roots score as a single
# unit (max member, never a sum). Only words that match a family may score or
# be displayed; nothing is ever inherited by clustering, so blocked words and
# antonyms cannot ride in on a neighbor.
GUILT_MOTIF_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "GUILT": {
        "guilt": ("guilt",),
        "remorse": ("remorse",),
        "shame": ("shame", "asham"),
        "culpability": ("culpab",),
        "regret": ("regret",),
    },
    "FEAR OF DISCOVERY": {
        "afraid": ("afraid",),
        "fear": ("fear",),
        "scared": ("scared",),
        "caught": ("caught",),
        "arrest": ("arrest",),
        "incriminate": ("incrimin",),
        "panic": ("panic", "panick"),
    },
    "COVER-UP": {
        "conceal": ("conceal",),
        "cover-up": ("coverup", "cover-up"),
        "hide": ("hidden", "hiding"),
        "secret": ("secret", "secrec"),
        "suppress": ("suppress",),
        "withhold": ("withheld", "withhold"),
    },
    "DECEPTION": {
        "deceive": ("deceit", "deceiv", "decept"),
        "dishonest": ("dishon",),
        "false": ("false", "falsif", "falsehood"),
        "fabricate": ("fabricat",),
        "fraud": ("fraud",),
        "lie": ("lied", "liar", "lying"),
        "mislead": ("mislead", "misled"),
    },
    "SABOTAGE": {
        "sabotage": ("sabotag",),
        "tamper": ("tamper",),
        "murder": ("murder",),
        "premeditate": ("premeditat",),
    },
}

GUILT_MOTIF_ROOTS = {
    motif: tuple(
        root for roots in families.values() for root in roots
    )
    for motif, families in GUILT_MOTIF_FAMILIES.items()
}

# Investigation-normal vocabulary that shares a prefix with a motif root but
# carries no incriminating meaning. Checked before root matching.
MOTIF_BLOCKLIST = {
    "culprit",
    "fearless",
    "fearlessly",
    "regrettable",
    "regrettably",
    "secretarial",
    "secretariat",
    "secretaries",
    "secretary",
    "secrete",
    "secreted",
    "secretes",
    "secretion",
    "secretions",
    "shameless",
    "shamelessly",
}

# Words in a player question that already put crime semantics into both
# replays. Reads on such turns must clear stricter gates.
PROVOCATIVE_QUESTION_WORDS = {
    "admit",
    "confess",
    "cover",
    "hide",
    "hid",
    "kill",
    "killed",
    "killer",
    "killing",
    "kills",
    "lie",
    "lied",
    "lies",
    "sabotaged",
}


@dataclass(frozen=True)
class RankedToken:
    token_id: int
    text: str
    layer: int
    rank: int
    logit: float
    position_index: int = 0


@dataclass(frozen=True)
class JSpaceTraceEvent:
    label: str
    kind: str
    start: int
    end: int
    peak: float
    positions: tuple[int, ...]
    layers: tuple[int, ...]
    best_rank: int


@dataclass(frozen=True)
class JSpaceResponseTrace:
    tokens: tuple[str, ...]
    events: tuple[JSpaceTraceEvent, ...]
    layers: tuple[int, ...]


@dataclass(frozen=True)
class MindStreamConcept:
    label: str
    score: float
    glow: float
    positions: tuple[int, ...]
    layers: tuple[int, ...]
    best_rank: int
    # Channel split: glow_public is content drawn from PUBLIC recollections
    # (grounding/recall); glow_private is content pressing in from PRIVATE
    # recollections (guarded stir). glow remains their total.
    glow_public: float = 0.0
    glow_private: float = 0.0


@dataclass(frozen=True)
class MindStream:
    tokens: tuple[str, ...]
    concepts: tuple[MindStreamConcept, ...]
    layers: tuple[int, ...]


@dataclass(frozen=True)
class JSpaceContrastLeak:
    label: str
    supporting_concepts: tuple[str, ...]
    actual_score: float
    counterfactual_score: float
    delta: float
    positions: tuple[int, ...]
    layers: tuple[int, ...]
    best_rank: int
    # The weakest controlled margin: min over all alternate conditions of
    # (source score - alternate score). Display strength must use this, not
    # the counterfactual-only delta.
    min_delta: float = 0.0
    neutral_score: float | None = None


@dataclass(frozen=True)
class JSpaceContrastTrace:
    leaks: tuple[JSpaceContrastLeak, ...]
    counterfactual_leaks: tuple[JSpaceContrastLeak, ...]
    actual_prompt_tokens: int
    counterfactual_prompt_tokens: int
    gate_mode: str = "standard"


@dataclass
class _ConceptAggregate:
    label: str
    rank_score: float = 0.0
    boundary_rank_score: float = 0.0
    layers: set[int] | None = None
    cells: set[tuple[int, int]] | None = None
    boundary_layers: set[int] | None = None
    best_rank: int = 10_000
    strongest_layer: int = -1
    strongest_logit: float = -math.inf
    positions: set[int] | None = None
    position_scores: dict[int, float] | None = None

    def __post_init__(self) -> None:
        if self.layers is None:
            self.layers = set()
        if self.cells is None:
            self.cells = set()
        if self.boundary_layers is None:
            self.boundary_layers = set()
        if self.positions is None:
            self.positions = set()
        if self.position_scores is None:
            self.position_scores = {}


def ranked_tokens_from_logits(
    lens_logits: dict[int, Any],
    *,
    tokenizer: Any,
    top_k_per_layer: int = 32,
) -> list[RankedToken]:
    ranked: list[RankedToken] = []
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for layer, logits in sorted(lens_logits.items()):
        for position_index, row in enumerate(logits):
            values, indices = row.topk(min(top_k_per_layer, row.shape[-1]))
            for rank, (token_id, value) in enumerate(
                zip(indices.tolist(), values.tolist(), strict=True),
                start=1,
            ):
                if int(token_id) in special_ids:
                    continue
                ranked.append(
                    RankedToken(
                        token_id=int(token_id),
                        text=str(tokenizer.decode([int(token_id)])),
                        layer=int(layer),
                        rank=rank,
                        logit=float(value),
                        position_index=position_index,
                    )
                )
    return ranked


def select_jspace_concepts(
    ranked_tokens: Iterable[RankedToken],
    *,
    visible_text: str,
    total_layers_read: int,
    top_k: int = 6,
) -> list[RawFeatureHit]:
    ranked_tokens = list(ranked_tokens)
    boundary_position_index = max(
        (item.position_index for item in ranked_tokens),
        default=0,
    )
    visible_words = {word.lower() for word in WORD_RE.findall(visible_text)}
    aggregates: dict[str, _ConceptAggregate] = defaultdict(
        lambda: _ConceptAggregate(label="")
    )

    for item in ranked_tokens:
        label = normalize_concept_token(item.text)
        if (
            not label
            or is_lexical_echo(label, visible_words)
            or label in READOUT_STOPWORDS
        ):
            continue
        aggregate = aggregates[label]
        aggregate.label = label
        aggregate.rank_score += 1.0 / math.sqrt(item.rank)
        aggregate.layers.add(item.layer)
        aggregate.cells.add((item.layer, item.position_index))
        if item.position_index == boundary_position_index:
            aggregate.boundary_layers.add(item.layer)
            aggregate.boundary_rank_score += 1.0 / math.sqrt(item.rank)
        if item.rank < aggregate.best_rank:
            aggregate.best_rank = item.rank
            aggregate.strongest_layer = item.layer
            aggregate.strongest_logit = item.logit

    ordered = sorted(
        aggregates.values(),
        key=lambda item: (
            -len(item.boundary_layers),
            -item.boundary_rank_score,
            -len(item.layers),
            -len(item.cells),
            -item.rank_score,
            item.best_rank,
            item.label,
        ),
    )

    results: list[RawFeatureHit] = []
    for aggregate in ordered:
        if is_near_duplicate_concept(
            aggregate.label,
            {result.label_hint or "" for result in results},
        ):
            continue
        support = len(aggregate.layers)
        boundary_support = len(aggregate.boundary_layers)
        support_fraction = support / max(1, total_layers_read)
        cell_support = len(aggregate.cells)
        intensity = min(
            0.99,
            0.46
            + 0.24 * support_fraction
            + 0.14 * (boundary_support / max(1, total_layers_read))
            + 0.12 / math.sqrt(max(1, aggregate.best_rank)),
        )
        layer_list = ", ".join(str(layer) for layer in sorted(aggregate.layers))
        results.append(
            RawFeatureHit(
                feature_id=f"jspace:{aggregate.label}",
                source="jspace",
                label_hint=aggregate.label,
                explanation_hint=(
                    "Silent J-Lens concept at the response boundary; "
                    f"ranked in {cell_support} sampled question/boundary cells "
                    f"across {support}/{max(1, total_layers_read)} workspace "
                    f"layers ({layer_list}), including {boundary_support} "
                    f"boundary layers. Best rank: "
                    f"{aggregate.best_rank} at layer {aggregate.strongest_layer}. "
                    "The word was absent from the detective's visible question."
                ),
                intensity_hint=intensity,
                raw_activation_hint=aggregate.strongest_logit,
            )
        )
        if len(results) >= top_k:
            break
    return results


def build_response_trace(
    ranked_tokens: Iterable[RankedToken],
    *,
    response_tokens: list[str],
    visible_question: str,
    total_layers_read: int,
    top_events: int = 7,
    response_position_offset: int = 1,
) -> JSpaceResponseTrace:
    ranked_tokens = list(ranked_tokens)
    visible_words = {word.lower() for word in WORD_RE.findall(visible_question)}
    prefix_words: list[set[str]] = []
    prefix_text = ""
    for token in response_tokens:
        prefix_text += token
        prefix_words.append({word.lower() for word in WORD_RE.findall(prefix_text)})

    aggregates: dict[str, _ConceptAggregate] = defaultdict(
        lambda: _ConceptAggregate(label="")
    )
    for item in ranked_tokens:
        position = item.position_index - response_position_offset
        if position < 0 or position >= len(response_tokens):
            continue
        label = normalize_concept_token(item.text)
        if (
            not label
            or label in TRACE_STOPWORDS
            or is_lexical_echo(label, visible_words)
            or is_lexical_echo(label, prefix_words[position])
        ):
            continue

        aggregate = aggregates[label]
        aggregate.label = label
        aggregate.rank_score += 1.0 / math.sqrt(item.rank)
        aggregate.layers.add(item.layer)
        aggregate.cells.add((item.layer, position))
        aggregate.positions.add(position)
        if item.rank < aggregate.best_rank:
            aggregate.best_rank = item.rank
            aggregate.strongest_layer = item.layer
            aggregate.strongest_logit = item.logit

    ordered = sorted(
        aggregates.values(),
        key=lambda item: (
            -len(item.positions),
            -len(item.layers),
            -len(item.cells),
            -item.rank_score,
            item.best_rank,
            item.label,
        ),
    )

    selected: list[_ConceptAggregate] = []
    for aggregate in ordered:
        if is_near_duplicate_concept(
            aggregate.label,
            {item.label for item in selected},
        ):
            continue
        selected.append(aggregate)
        if len(selected) >= top_events:
            break

    events: list[JSpaceTraceEvent] = []
    full_response_words = [
        word.lower() for word in WORD_RE.findall("".join(response_tokens))
    ]
    for aggregate in selected:
        positions = tuple(sorted(aggregate.positions))
        layers = tuple(sorted(aggregate.layers))
        start = positions[0]
        end = positions[-1]
        support_fraction = len(layers) / max(1, total_layers_read)
        persistence = min(1.0, len(positions) / 4)
        peak = min(
            0.99,
            0.42
            + 0.23 * support_fraction
            + 0.2 * persistence
            + 0.14 / math.sqrt(max(1, aggregate.best_rank)),
        )
        kind = classify_trace_event(
            label=aggregate.label,
            positions=positions,
            response_tokens=response_tokens,
            response_words=full_response_words,
        )
        events.append(
            JSpaceTraceEvent(
                label=aggregate.label,
                kind=kind,
                start=start,
                end=end,
                peak=peak,
                positions=positions,
                layers=layers,
                best_rank=aggregate.best_rank,
            )
        )

    return JSpaceResponseTrace(
        tokens=tuple(response_tokens),
        events=tuple(events),
        layers=tuple(
            sorted(
                {
                    item.layer
                    for item in ranked_tokens
                    if item.position_index >= response_position_offset
                }
            )
        ),
    )


def build_contrast_trace(
    actual_ranked_tokens: Iterable[RankedToken],
    counterfactual_ranked_tokens: Iterable[RankedToken],
    *,
    response_tokens: list[str],
    visible_question: str,
    shared_template_text: str,
    actual_prompt_tokens: int,
    counterfactual_prompt_tokens: int,
    neutral_ranked_tokens: Iterable[RankedToken] | None = None,
    strict_gates: bool = False,
    gate_mode: str | None = None,
    delta_gate: float = 1.2,
    ratio_gate: float = 1.25,
    top_leaks: int = 3,
    top_positions: int = 8,
    response_position_offset: int = 1,
) -> JSpaceContrastTrace:
    visible_words = {word.lower() for word in WORD_RE.findall(visible_question)}
    shared_words = {word.lower() for word in WORD_RE.findall(shared_template_text)}
    response_words = {
        word.lower() for word in WORD_RE.findall("".join(response_tokens))
    }
    if strict_gates:
        delta_gate = delta_gate * 2.0
        ratio_gate = 1.0 + 2.0 * (ratio_gate - 1.0)

    def aggregate(
        ranked_tokens: Iterable[RankedToken],
    ) -> dict[str, _ConceptAggregate]:
        aggregates: dict[str, _ConceptAggregate] = defaultdict(
            lambda: _ConceptAggregate(label="")
        )
        for item in ranked_tokens:
            position = item.position_index - response_position_offset
            if position < 0 or position >= len(response_tokens):
                continue
            label = normalize_concept_token(item.text)
            if (
                not label
                or label in TRACE_STOPWORDS
                or is_lexical_echo(label, visible_words)
                or is_lexical_echo(label, shared_words)
                or is_lexical_echo(label, response_words)
            ):
                continue
            concept = aggregates[label]
            concept.label = label
            concept.rank_score += 1.0 / math.sqrt(item.rank)
            concept.layers.add(item.layer)
            concept.cells.add((item.layer, position))
            concept.positions.add(position)
            concept.position_scores[position] = (
                concept.position_scores.get(position, 0.0)
                + 1.0 / math.sqrt(item.rank)
            )
            if item.rank < concept.best_rank:
                concept.best_rank = item.rank
                concept.strongest_layer = item.layer
                concept.strongest_logit = item.logit
        return aggregates

    conditions = {
        "actual": aggregate(actual_ranked_tokens),
        "counterfactual": aggregate(counterfactual_ranked_tokens),
    }
    has_neutral = neutral_ranked_tokens is not None
    if has_neutral:
        conditions["neutral"] = aggregate(neutral_ranked_tokens)

    # Per condition, fold motif-eligible concepts into canonical families.
    # Words with no (motif, family) assignment — including blocklisted words —
    # never score and never display. A family scores as its strongest member.
    class _Family:
        __slots__ = ("score", "label", "layers", "position_scores", "best_rank")

        def __init__(self) -> None:
            self.score = 0.0
            self.label = ""
            self.layers: set[int] = set()
            self.position_scores: dict[int, float] = {}
            self.best_rank = 10_000

    def family_fold(
        concepts: dict[str, _ConceptAggregate],
    ) -> dict[tuple[str, str], _Family]:
        families: dict[tuple[str, str], _Family] = defaultdict(_Family)
        for label, concept in concepts.items():
            assignment = motif_family_for_concept(label)
            if assignment is None:
                continue
            family = families[assignment]
            if concept.rank_score > family.score:
                family.score = concept.rank_score
                family.label = label
            family.layers.update(concept.layers)
            family.best_rank = min(family.best_rank, concept.best_rank)
            for position, score in concept.position_scores.items():
                family.position_scores[position] = (
                    family.position_scores.get(position, 0.0) + score
                )
        return dict(families)

    families_by_condition = {
        name: family_fold(concepts) for name, concepts in conditions.items()
    }
    all_family_keys = {
        key for families in families_by_condition.values() for key in families
    }
    motif_families: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(all_family_keys):
        motif_families[key[0]].append(key)

    def directed_leaks(
        source: str,
        alternates: list[str],
    ) -> list[JSpaceContrastLeak]:
        source_families = families_by_condition[source]
        candidates: list[JSpaceContrastLeak] = []
        for motif, family_keys in motif_families.items():

            def condition_score(condition: str, key: tuple[str, str]) -> float:
                family = families_by_condition[condition].get(key)
                return family.score if family else 0.0

            motif_source = sum(condition_score(source, key) for key in family_keys)
            alternate_totals = {
                alternate: sum(
                    condition_score(alternate, key) for key in family_keys
                )
                for alternate in alternates
            }
            deltas = {
                alternate: motif_source - total
                for alternate, total in alternate_totals.items()
            }
            ratios = {
                alternate: motif_source / max(total, 1.0)
                for alternate, total in alternate_totals.items()
            }

            # Families that individually favor the source against every
            # alternate; these are the displayable evidence.
            supporting: list[tuple[float, _Family, tuple[str, str]]] = []
            for key in family_keys:
                family = source_families.get(key)
                if family is None or family.score <= 0.0:
                    continue
                family_min_delta = min(
                    family.score - condition_score(alternate, key)
                    for alternate in alternates
                )
                if family_min_delta <= 0.0:
                    continue
                supporting.append((family_min_delta, family, key))
            if not supporting:
                continue

            layers: set[int] = set()
            best_rank = 10_000
            contrast_positions: dict[int, float] = {}
            for _, family, key in supporting:
                layers.update(family.layers)
                best_rank = min(best_rank, family.best_rank)
                # Rank positions by contrast, not source activity: each
                # position contributes its weakest source-minus-alternate
                # margin, floored at zero.
                for position, score in family.position_scores.items():
                    margin = min(
                        score
                        - (
                            families_by_condition[alternate]
                            .get(key, _Family())
                            .position_scores.get(position, 0.0)
                        )
                        for alternate in alternates
                    )
                    if margin > 0.0:
                        contrast_positions[position] = (
                            contrast_positions.get(position, 0.0) + margin
                        )

            gate_delta = delta_gate
            if len(supporting) < 2:
                # A single lexical family must be decisively enriched to count.
                gate_delta = delta_gate * 2.0
            min_delta = min(deltas.values())
            if min_delta < gate_delta:
                continue
            if min(ratios.values()) < ratio_gate:
                continue
            if len(layers) < 3:
                continue

            supporting.sort(key=lambda item: (-item[0], item[2]))
            top_position_list = tuple(
                sorted(
                    position
                    for position, _ in sorted(
                        contrast_positions.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:top_positions]
                )
            )
            opposite = "counterfactual" if source == "actual" else "actual"
            opposite_total = alternate_totals.get(opposite, 0.0)
            neutral_total = (
                alternate_totals.get("neutral") if has_neutral else None
            )
            candidates.append(
                JSpaceContrastLeak(
                    label=motif,
                    supporting_concepts=tuple(
                        family.label for _, family, _ in supporting[:4]
                    ),
                    actual_score=(
                        motif_source if source == "actual" else opposite_total
                    ),
                    counterfactual_score=(
                        opposite_total if source == "actual" else motif_source
                    ),
                    delta=(
                        motif_source - opposite_total
                        if source == "actual"
                        else opposite_total - motif_source
                    ),
                    positions=top_position_list,
                    layers=tuple(sorted(layers)),
                    best_rank=best_rank,
                    min_delta=min_delta,
                    neutral_score=neutral_total,
                )
            )
        candidates.sort(
            key=lambda leak: (
                -leak.min_delta,
                -len(leak.layers),
                leak.best_rank,
                leak.label,
            )
        )
        return candidates[:top_leaks]

    actual_alternates = ["counterfactual", *(["neutral"] if has_neutral else [])]
    counterfactual_alternates = ["actual", *(["neutral"] if has_neutral else [])]
    return JSpaceContrastTrace(
        leaks=tuple(directed_leaks("actual", actual_alternates)),
        counterfactual_leaks=tuple(
            directed_leaks("counterfactual", counterfactual_alternates)
        ),
        actual_prompt_tokens=actual_prompt_tokens,
        counterfactual_prompt_tokens=counterfactual_prompt_tokens,
        gate_mode=gate_mode or ("strict" if strict_gates else "standard"),
    )


def build_mind_stream(
    full_ranked_tokens: Iterable[RankedToken],
    private_ablated_ranked_tokens: Iterable[RankedToken],
    all_ablated_ranked_tokens: Iterable[RankedToken],
    *,
    response_tokens: list[str],
    visible_question: str,
    top_concepts: int = 10,
    top_positions: int = 6,
    response_position_offset: int = 1,
) -> MindStream:
    """The suspect's whole unspoken mind, with two-channel memory glow.

    The stream comes from the full-context replay alone — normal reactive
    thoughts included. Two ablated replays of the same tokens split each
    concept's glow into channels:
      glow_private = full − private_ablated   (PRIVATE content pressing in)
      glow_public  = private_ablated − all_ablated (PUBLIC content grounding)
    Nothing is gated on glow; it only marks.

    Echo policy: words echoing the question or the spoken answer are excluded
    from the ambient layer (trivial presence), but are admitted when they
    carry mark-level channel glow — both replay conditions contain the
    question and the response, so channel deltas cannot be caused by echo.
    An echoed word that glows is the most informative evidence there is
    ("Ilya" burning guarded beneath a denial about Ilya).
    """
    visible_words = {word.lower() for word in WORD_RE.findall(visible_question)}
    response_words = {
        word.lower() for word in WORD_RE.findall("".join(response_tokens))
    }

    def aggregate(
        ranked_tokens: Iterable[RankedToken],
    ) -> dict[str, _ConceptAggregate]:
        aggregates: dict[str, _ConceptAggregate] = defaultdict(
            lambda: _ConceptAggregate(label="")
        )
        for item in ranked_tokens:
            position = item.position_index - response_position_offset
            if position < 0 or position >= len(response_tokens):
                continue
            label = normalize_concept_token(item.text)
            if not label or label in TRACE_STOPWORDS:
                continue
            concept = aggregates[label]
            concept.label = label
            concept.rank_score += 1.0 / math.sqrt(item.rank)
            concept.layers.add(item.layer)
            concept.cells.add((item.layer, position))
            concept.positions.add(position)
            concept.position_scores[position] = (
                concept.position_scores.get(position, 0.0)
                + 1.0 / math.sqrt(item.rank)
            )
            if item.rank < concept.best_rank:
                concept.best_rank = item.rank
                concept.strongest_layer = item.layer
                concept.strongest_logit = item.logit
        return aggregates

    full = aggregate(full_ranked_tokens)
    private_ablated = aggregate(private_ablated_ranked_tokens)
    all_ablated = aggregate(all_ablated_ranked_tokens)

    def score_in(condition: dict[str, _ConceptAggregate], label: str) -> float:
        return condition[label].rank_score if label in condition else 0.0

    candidates: list[MindStreamConcept] = []
    for concept in full.values():
        if len(concept.layers) < 2:
            continue
        positions = tuple(
            sorted(
                position
                for position, _ in sorted(
                    concept.position_scores.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:top_positions]
            )
        )
        no_private = score_in(private_ablated, concept.label)
        no_all = score_in(all_ablated, concept.label)
        glow_private = max(0.0, concept.rank_score - no_private)
        glow_public = max(0.0, no_private - no_all)
        is_echo = is_lexical_echo(concept.label, visible_words) or is_lexical_echo(
            concept.label, response_words
        )
        if is_echo:
            # Mirrors the client mark thresholds: an echoed word must qualify
            # for a mark to enter the stream at all.
            privately_marked = (
                glow_private >= 2.5
                and concept.rank_score > 0
                and glow_private / concept.rank_score >= 0.25
            )
            publicly_marked = (
                glow_public >= 1.5
                and concept.rank_score > 0
                and glow_public / concept.rank_score >= 0.45
            )
            if not (privately_marked or publicly_marked):
                continue
        candidates.append(
            MindStreamConcept(
                label=concept.label,
                score=concept.rank_score,
                glow=max(0.0, concept.rank_score - no_all),
                glow_private=glow_private,
                glow_public=glow_public,
                positions=positions,
                layers=tuple(sorted(concept.layers)),
                best_rank=concept.best_rank,
            )
        )

    # Memory-sensitive concepts are often weaker in absolute full-context rank
    # than generic reactive thoughts. Reserve display capacity for meaningful
    # positive deltas, then backfill with the strongest ambient concepts. The
    # threshold means the reservation never manufactures a fixed recall count:
    # a turn with no real delta remains entirely ambient.
    memory_slots = min(top_concepts, math.ceil(top_concepts * 0.4))

    def is_memory_candidate(candidate: MindStreamConcept) -> bool:
        return (
            candidate.glow >= 1.5
            and candidate.score > 0
            and candidate.glow / candidate.score >= 0.3
        )

    memory_candidates = sorted(
        (candidate for candidate in candidates if is_memory_candidate(candidate)),
        key=lambda candidate: (
            -candidate.glow,
            -(candidate.glow / candidate.score if candidate.score else 0.0),
            -len(candidate.layers),
            candidate.best_rank,
            -candidate.score,
            candidate.label,
        ),
    )
    ambient_candidates = sorted(
        (
            candidate
            for candidate in candidates
            if not is_memory_candidate(candidate)
        ),
        key=lambda candidate: (
            -candidate.score,
            candidate.best_rank,
            -len(candidate.layers),
            candidate.label,
        ),
    )

    concepts: list[MindStreamConcept] = []

    def add_candidate(candidate: MindStreamConcept) -> bool:
        if len(concepts) >= top_concepts:
            return False
        if is_near_duplicate_concept(
            candidate.label,
            {selected.label for selected in concepts},
        ):
            return False
        concepts.append(candidate)
        return True

    memory_added = 0
    for candidate in memory_candidates:
        if memory_added >= memory_slots:
            break
        if add_candidate(candidate):
            memory_added += 1

    for candidate in ambient_candidates:
        if len(concepts) >= top_concepts:
            break
        add_candidate(candidate)

    return MindStream(
        tokens=tuple(response_tokens),
        concepts=tuple(concepts),
        layers=tuple(
            sorted({layer for concept in concepts for layer in concept.layers})
        ),
    )


def guilt_motif_for_concept(label: str) -> str | None:
    assignment = motif_family_for_concept(label)
    return assignment[0] if assignment else None


def motif_family_for_concept(label: str) -> tuple[str, str] | None:
    """Map a concept word to its (motif, canonical family), or None."""
    if label in MOTIF_BLOCKLIST:
        return None
    for motif, families in GUILT_MOTIF_FAMILIES.items():
        for family, roots in families.items():
            if any(label.startswith(root) for root in roots):
                return motif, family
    return None


def question_is_provocative(question: str) -> bool:
    for word in WORD_RE.findall(question.lower()):
        if word in PROVOCATIVE_QUESTION_WORDS:
            return True
        if guilt_motif_for_concept(word) is not None:
            return True
    return False


def question_repeats_history(question: str, prior_user_texts: Iterable[str]) -> bool:
    current = {word.lower() for word in WORD_RE.findall(question)}
    if not current:
        return False
    for prior in prior_user_texts:
        previous = {word.lower() for word in WORD_RE.findall(prior)}
        if not previous:
            continue
        overlap = len(current & previous) / len(current | previous)
        if overlap >= 0.8:
            return True
    return False


def classify_trace_event(
    *,
    label: str,
    positions: tuple[int, ...],
    response_tokens: list[str],
    response_words: list[str],
) -> str:
    start = positions[0]
    first_spoken_position: int | None = None
    for position in range(start + 1, len(response_tokens)):
        spoken_words = {
            word.lower()
            for word in WORD_RE.findall("".join(response_tokens[: position + 1]))
        }
        if is_lexical_echo(label, spoken_words):
            first_spoken_position = position
            break
    if (
        first_spoken_position is not None
        and first_spoken_position - start >= 3
    ):
        return "premonition"
    span = positions[-1] - positions[0] + 1
    density = len(positions) / max(span, 1)
    if len(positions) >= 3 and density >= 0.32:
        return "fixation"
    if start <= 2:
        return "recoil"
    if len(positions) == 1:
        return "intrusion"
    return "undertow"


def normalize_concept_token(text: str) -> str:
    text = text.replace("Ġ", " ").replace("▁", " ").strip().lower()
    matches = WORD_RE.findall(text)
    if len(matches) != 1:
        return ""
    value = matches[0]
    if len(value) < 3 or len(value) > 28:
        return ""
    if not value.isascii():
        return ""
    return value


def is_lexical_echo(label: str, visible_words: set[str]) -> bool:
    for visible in visible_words:
        if label == visible:
            return True
        shortest = min(len(label), len(visible))
        if shortest < 4:
            continue
        common_prefix = 0
        for left, right in zip(label, visible, strict=False):
            if left != right:
                break
            common_prefix += 1
        if common_prefix >= 4 and common_prefix / shortest >= 0.8:
            return True
    return False


def is_near_duplicate_concept(label: str, selected_labels: set[str]) -> bool:
    for selected in selected_labels:
        if not selected:
            continue
        if label == selected:
            return True
        shorter, longer = sorted((label, selected), key=len)
        if len(shorter) >= 5 and shorter in longer:
            return True

        common_prefix = 0
        for left, right in zip(shorter, longer, strict=False):
            if left != right:
                break
            common_prefix += 1
        if common_prefix >= 5 and common_prefix / len(shorter) >= 0.75:
            return True
    return False


class JSpaceActivationClient:
    def __init__(
        self,
        *,
        model: Any,
        lens_model: Any,
        lens: Any,
        tokenizer: Any,
        sampled_layers: tuple[int, ...] | None = None,
        top_k_per_layer: int = 128,
        final_top_k: int = 6,
    ) -> None:
        self.model = model
        self.lens_model = lens_model
        self.lens = lens
        self.tokenizer = tokenizer
        self.sampled_layers = sampled_layers or choose_workspace_layers(
            tuple(lens.source_layers)
        )
        self.top_k_per_layer = top_k_per_layer
        self.final_top_k = final_top_k

    def concepts_at_response_boundary(
        self,
        *,
        prompt_text: str,
        visible_question: str,
    ) -> list[RawFeatureHit]:
        read_positions = choose_read_positions(
            lens_model=self.lens_model,
            tokenizer=self.tokenizer,
            prompt_text=prompt_text,
            visible_question=visible_question,
        )
        lens_logits, _, _ = self.lens.apply(
            self.lens_model,
            prompt_text,
            layers=self.sampled_layers,
            positions=read_positions,
            max_seq_len=1024,
        )
        ranked = ranked_tokens_from_logits(
            lens_logits,
            tokenizer=self.tokenizer,
            top_k_per_layer=self.top_k_per_layer,
        )
        return select_jspace_concepts(
            ranked,
            visible_text=visible_question,
            total_layers_read=len(self.sampled_layers),
            top_k=self.final_top_k,
        )

    def read_generated_response(
        self,
        *,
        prompt_token_ids: Iterable[int],
        visible_question: str,
        response_token_ids: Iterable[int],
    ) -> tuple[list[RawFeatureHit], JSpaceResponseTrace]:
        prompt_ids = tuple(int(token_id) for token_id in prompt_token_ids)
        response_ids = tuple(int(token_id) for token_id in response_token_ids)
        ranked, response_tokens = self._rank_response_ids(
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
        )
        if not response_ids:
            return [], JSpaceResponseTrace(
                tokens=tuple(),
                events=tuple(),
                layers=self.sampled_layers,
            )
        boundary_ranked = [item for item in ranked if item.position_index == 0]
        boundary_hits = select_jspace_concepts(
            boundary_ranked,
            visible_text=visible_question,
            total_layers_read=len(self.sampled_layers),
            top_k=self.final_top_k,
        )
        trace = build_response_trace(
            ranked,
            response_tokens=response_tokens,
            visible_question=visible_question,
            total_layers_read=len(self.sampled_layers),
        )
        return boundary_hits, trace

    def read_mind_stream(
        self,
        *,
        full_prompt_token_ids: Iterable[int],
        private_ablated_prompt_token_ids: Iterable[int],
        all_ablated_prompt_token_ids: Iterable[int],
        visible_question: str,
        response_token_ids: Iterable[int],
        top_concepts: int = 10,
    ) -> tuple[MindStream, JSpaceResponseTrace]:
        full_prompt_ids = tuple(int(token_id) for token_id in full_prompt_token_ids)
        private_ablated_ids = tuple(
            int(token_id) for token_id in private_ablated_prompt_token_ids
        )
        all_ablated_ids = tuple(
            int(token_id) for token_id in all_ablated_prompt_token_ids
        )
        response_ids = tuple(int(token_id) for token_id in response_token_ids)
        full_ranked, response_tokens = self._rank_response_ids(
            prompt_token_ids=full_prompt_ids,
            response_token_ids=response_ids,
        )
        private_ablated_ranked, private_tokens = self._rank_response_ids(
            prompt_token_ids=private_ablated_ids,
            response_token_ids=response_ids,
        )
        all_ablated_ranked, all_tokens = self._rank_response_ids(
            prompt_token_ids=all_ablated_ids,
            response_token_ids=response_ids,
        )
        if response_tokens != private_tokens or response_tokens != all_tokens:
            raise ValueError(
                "Memory-ablated replay decoded different response tokens."
            )
        stream = build_mind_stream(
            full_ranked,
            private_ablated_ranked,
            all_ablated_ranked,
            response_tokens=response_tokens,
            visible_question=visible_question,
            top_concepts=top_concepts,
        )
        trace = build_response_trace(
            full_ranked,
            response_tokens=response_tokens,
            visible_question=visible_question,
            total_layers_read=len(self.sampled_layers),
        )
        return stream, trace

    def read_generated_response_with_contrast(
        self,
        *,
        actual_prompt_token_ids: Iterable[int],
        counterfactual_prompt_token_ids: Iterable[int],
        visible_question: str,
        response_token_ids: Iterable[int],
        shared_template_text: str,
        neutral_prompt_token_ids: Iterable[int] | None = None,
        strict_gates: bool = False,
        gate_mode: str | None = None,
    ) -> tuple[list[RawFeatureHit], JSpaceResponseTrace, JSpaceContrastTrace]:
        actual_prompt_ids = tuple(int(token_id) for token_id in actual_prompt_token_ids)
        counterfactual_prompt_ids = tuple(
            int(token_id) for token_id in counterfactual_prompt_token_ids
        )
        response_ids = tuple(int(token_id) for token_id in response_token_ids)
        actual_ranked, response_tokens = self._rank_response_ids(
            prompt_token_ids=actual_prompt_ids,
            response_token_ids=response_ids,
        )
        counterfactual_ranked, counterfactual_response_tokens = (
            self._rank_response_ids(
                prompt_token_ids=counterfactual_prompt_ids,
                response_token_ids=response_ids,
            )
        )
        if response_tokens != counterfactual_response_tokens:
            raise ValueError("Fixed-response replay decoded different response tokens.")
        neutral_ranked = None
        if neutral_prompt_token_ids is not None:
            neutral_prompt_ids = tuple(
                int(token_id) for token_id in neutral_prompt_token_ids
            )
            neutral_ranked, neutral_response_tokens = self._rank_response_ids(
                prompt_token_ids=neutral_prompt_ids,
                response_token_ids=response_ids,
            )
            if response_tokens != neutral_response_tokens:
                raise ValueError(
                    "Fixed-response neutral replay decoded different response tokens."
                )
        boundary_ranked = [
            item for item in actual_ranked if item.position_index == 0
        ]
        boundary_hits = select_jspace_concepts(
            boundary_ranked,
            visible_text=visible_question,
            total_layers_read=len(self.sampled_layers),
            top_k=self.final_top_k,
        )
        response_trace = build_response_trace(
            actual_ranked,
            response_tokens=response_tokens,
            visible_question=visible_question,
            total_layers_read=len(self.sampled_layers),
        )
        contrast_trace = build_contrast_trace(
            actual_ranked,
            counterfactual_ranked,
            response_tokens=response_tokens,
            visible_question=visible_question,
            shared_template_text=shared_template_text,
            actual_prompt_tokens=len(actual_prompt_ids),
            counterfactual_prompt_tokens=len(counterfactual_prompt_ids),
            neutral_ranked_tokens=neutral_ranked,
            strict_gates=strict_gates,
            gate_mode=gate_mode,
        )
        return boundary_hits, response_trace, contrast_trace

    def _rank_response_ids(
        self,
        *,
        prompt_token_ids: tuple[int, ...],
        response_token_ids: tuple[int, ...],
    ) -> tuple[list[RankedToken], list[str]]:
        import torch
        from jlens.hooks import ActivationRecorder

        if not prompt_token_ids:
            raise ValueError("The replay prompt is empty.")
        replay_ids = (*prompt_token_ids, *response_token_ids)
        if len(replay_ids) > 2048:
            raise ValueError(
                f"J-Space replay is {len(replay_ids)} tokens; the limit is 2048."
            )
        prompt_length = len(prompt_token_ids)
        response_positions = list(range(prompt_length, len(replay_ids)))
        positions = [prompt_length - 1, *response_positions]
        input_ids = torch.tensor(
            [replay_ids],
            dtype=torch.long,
            device=self.lens_model.input_device,
        )
        final_layer = self.lens_model.n_layers - 1
        record_at = sorted(set(self.sampled_layers) | {final_layer})
        with torch.no_grad(), ActivationRecorder(
            self.lens_model.layers,
            at=record_at,
        ) as recorder:
            self.lens_model.forward(input_ids)

        def select(layer: int):
            return recorder.activations[layer][0][positions].float()

        lens_logits: dict[int, Any] = {}
        for layer in self.sampled_layers:
            residual = self.lens.transport(select(layer), layer)
            lens_logits[layer] = self.lens_model.unembed(residual).float().cpu()
        ranked = ranked_tokens_from_logits(
            lens_logits,
            tokenizer=self.tokenizer,
            top_k_per_layer=self.top_k_per_layer,
        )
        response_tokens = [
            str(
                self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            for token_id in response_token_ids
        ]
        return ranked, response_tokens


def choose_workspace_layers(source_layers: tuple[int, ...]) -> tuple[int, ...]:
    if not source_layers:
        raise ValueError("The Jacobian lens has no fitted source layers.")
    count = len(source_layers)
    # The paper finds coherent workspace content after roughly the first third
    # and before the final motor-like layers. Sample this band without assuming
    # a specific transformer depth.
    indexes = {
        min(count - 1, max(0, round(count * fraction)))
        for fraction in (0.36, 0.48, 0.60, 0.72, 0.82)
    }
    return tuple(source_layers[index] for index in sorted(indexes))


def choose_read_positions(
    *,
    lens_model: Any,
    tokenizer: Any,
    prompt_text: str,
    visible_question: str,
    max_question_positions: int = 4,
) -> list[int]:
    prompt_ids = lens_model.encode(prompt_text, max_length=1024)[0].tolist()
    question_ids = tokenizer(
        visible_question,
        add_special_tokens=False,
        return_attention_mask=False,
    ).input_ids
    start = find_last_subsequence(prompt_ids, list(question_ids))
    if start < 0 or not question_ids:
        return [-1]

    meaningful_offsets = []
    for offset, token_id in enumerate(question_ids):
        token = normalize_concept_token(str(tokenizer.decode([int(token_id)])))
        if token and token not in READOUT_STOPWORDS:
            meaningful_offsets.append(offset)
    if not meaningful_offsets:
        meaningful_offsets = list(range(len(question_ids)))

    if len(meaningful_offsets) <= max_question_positions:
        chosen_offsets = meaningful_offsets
    else:
        last = len(meaningful_offsets) - 1
        chosen_offsets = sorted(
            {
                meaningful_offsets[round(last * fraction)]
                for fraction in (0.0, 0.34, 0.67, 1.0)
            }
        )
    return [start + offset for offset in chosen_offsets] + [-1]


def find_last_subsequence(values: list[int], target: list[int]) -> int:
    if not target or len(target) > len(values):
        return -1
    for start in range(len(values) - len(target), -1, -1):
        if values[start : start + len(target)] == target:
            return start
    return -1
