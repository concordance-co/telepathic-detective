from __future__ import annotations

import json
from pathlib import Path
import unittest

from telepathic_detective.jspace_activation import (
    RankedToken,
    build_mind_stream,
)
from telepathic_detective.jspace_server import build_memory_blocks
from telepathic_detective.memory_retrieval import (
    MemoryEntry,
    RetrievedMemory,
    apply_channel_slots,
    build_retrieval_query,
    content_words,
    extract_entities,
    memory_is_relevant,
    parse_memory_bank,
    question_is_incident_anchored,
    retrieval_passes_null_gate,
)


def tokens(text: str, token_id: int, *, rank: int = 1, position: int = 2,
           layers: tuple[int, ...] = (11, 15, 19, 22)) -> list[RankedToken]:
    return [
        RankedToken(token_id, text, layer, rank, 10.0, position_index=position)
        for layer in layers
    ]


class MindStreamTests(unittest.TestCase):
    def test_stream_keeps_ambient_thoughts(self) -> None:
        # A concept equally present with and without memories stays in the
        # stream (unlike the old contrastive gate) — with zero glow.
        full = tokens(" protocol", 1) + tokens(" corridor", 2, position=3)
        ablated = tokens(" protocol", 1)
        stream = build_mind_stream(
            full,
            ablated,
            ablated,
            response_tokens=["I", " followed", " the", " rules", "."],
            visible_question="What happened that night?",
        )
        labels = {concept.label: concept for concept in stream.concepts}
        self.assertIn("protocol", labels)
        self.assertIn("corridor", labels)
        self.assertAlmostEqual(labels["protocol"].glow, 0.0)
        self.assertAlmostEqual(labels["corridor"].glow, labels["corridor"].score)

    def test_stream_filters_glowless_spoken_and_question_words(self) -> None:
        # Echo words with no channel glow are trivial presence and stay out.
        echoes = tokens(" rules", 1) + tokens(" audit", 2)
        full = echoes + tokens(" timing", 3)
        stream = build_mind_stream(
            full,
            echoes,   # echoes identical in every condition -> zero glow
            echoes,
            response_tokens=["I", " followed", " the", " rules", "."],
            visible_question="What did the audit find?",
        )
        labels = [concept.label for concept in stream.concepts]
        self.assertEqual(labels, ["timing"])

    def test_stream_admits_echo_words_with_channel_glow(self) -> None:
        # An echoed word whose strength depends on private content is real
        # evidence: both conditions share the question and response, so the
        # delta cannot be echo. "audits" echoes the question but glows.
        full = tokens(" audits", 1) + tokens(" timing", 2, position=3)
        without_private = tokens(" timing", 2, position=3)
        stream = build_mind_stream(
            full,
            without_private,
            without_private,
            response_tokens=["I", " followed", " the", " rules", "."],
            visible_question="What did the audit find?",
        )
        by_label = {concept.label: concept for concept in stream.concepts}
        self.assertIn("audits", by_label)
        self.assertGreaterEqual(by_label["audits"].glow_private, 2.5)

    def test_stream_requires_multi_layer_support(self) -> None:
        single_layer = [
            RankedToken(1, " flicker", 11, 1, 10.0, position_index=2)
        ]
        stream = build_mind_stream(
            single_layer,
            [],
            [],
            response_tokens=["I", " saw", " it", "."],
            visible_question="What happened?",
        )
        self.assertEqual(stream.concepts, ())

    def test_stream_dedupes_display_variants(self) -> None:
        full = tokens(" secret", 1) + tokens(" secretly", 2, position=3)
        stream = build_mind_stream(
            full,
            [],
            [],
            response_tokens=["I", " told", " no", " one", "."],
            visible_question="What happened?",
        )
        labels = [concept.label for concept in stream.concepts]
        self.assertEqual(labels, ["secret"])

    def test_glow_is_floored_at_zero(self) -> None:
        full = tokens(" corridor", 1, rank=4)
        ablated = tokens(" corridor", 1, rank=1)
        stream = build_mind_stream(
            full,
            ablated,
            ablated,
            response_tokens=["I", " walked", " home", "."],
            visible_question="What happened?",
        )
        self.assertEqual(stream.concepts[0].glow, 0.0)

    def test_meaningful_memory_delta_gets_reserved_capacity(self) -> None:
        full = (
            tokens(" financial", 1)
            + tokens(" budget", 2, position=3)
            + tokens(" protocol", 3, position=4)
            + tokens(" buried", 4, rank=4, position=5)
        )
        ablated = (
            tokens(" financial", 1)
            + tokens(" budget", 2, position=3)
            + tokens(" protocol", 3, position=4)
        )
        stream = build_mind_stream(
            full,
            ablated,
            ablated,
            response_tokens=["I", " answered", " plainly", " enough", ".", "!"],
            visible_question="What happened?",
            top_concepts=3,
        )
        labels = [concept.label for concept in stream.concepts]
        self.assertEqual(labels[0], "buried")
        self.assertEqual(len(labels), 3)

    def test_weak_delta_does_not_displace_strong_ambient_thought(self) -> None:
        full = tokens(" financial", 1) + tokens(" buried", 2, rank=9, position=3)
        ablated = tokens(" financial", 1)
        stream = build_mind_stream(
            full,
            ablated,
            ablated,
            response_tokens=["I", " answered", " plainly", "."],
            visible_question="What happened?",
            top_concepts=1,
        )
        self.assertEqual(stream.concepts[0].label, "financial")

    def test_short_memory_label_is_not_added_twice_during_backfill(self) -> None:
        stream = build_mind_stream(
            tokens(" last", 1),
            [],
            [],
            response_tokens=["I", " answered", " plainly", "."],
            visible_question="What happened?",
            top_concepts=2,
        )
        self.assertEqual(
            [concept.label for concept in stream.concepts],
            ["last"],
        )

    def test_memory_overflow_does_not_reenter_as_ambient(self) -> None:
        full = (
            tokens(" buried", 1)
            + tokens(" hidden", 2, position=3)
            + tokens(" private", 3, position=4)
            + tokens(" silent", 4, position=5)
            + tokens(" inward", 5, position=6)
            + tokens(" financial", 6, position=7)
        )
        ablated = tokens(" financial", 6, position=7)
        stream = build_mind_stream(
            full,
            ablated,
            ablated,
            response_tokens=["I", " answered", " plainly", " enough", ".", "!", " now", "."],
            visible_question="What happened?",
            top_concepts=6,
        )
        recalled = [concept for concept in stream.concepts if concept.glow >= 1.5]
        self.assertEqual(len(recalled), 3)
        self.assertIn("financial", {concept.label for concept in stream.concepts})


class MemoryBankTests(unittest.TestCase):
    def test_parse_memory_bank_shapes(self) -> None:
        entries = parse_memory_bank(
            [
                {"id": "m1", "text": "I remember the corridor.", "privacy": "private", "tags": ["corridor"]},
                {"text": "Unnamed but valid."},
                {"id": "bad", "text": "   "},
                "not a dict",
            ]
        )
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].id, "m1")
        self.assertEqual(entries[0].tags, ("corridor",))
        self.assertEqual(entries[1].privacy, "private")

    def test_shipped_corpora_parse_and_are_distinctive(self) -> None:
        root = Path(__file__).resolve().parent.parent / "data" / "memories"
        for name, minimum in (("ilya.json", 30), ("mira.json", 25)):
            raw = json.loads((root / name).read_text())
            entries = parse_memory_bank(raw["entries"])
            self.assertGreaterEqual(len(entries), minimum, name)
            ids = [entry.id for entry in entries]
            self.assertEqual(len(ids), len(set(ids)), f"duplicate ids in {name}")
            privacies = {entry.privacy for entry in entries}
            self.assertEqual(privacies, {"private", "public"}, name)

    def test_memory_blocks_keep_privacy_scaffold_in_all_conditions(self) -> None:
        retrieved = [
            RetrievedMemory(
                entry=MemoryEntry(
                    id="private",
                    text="I changed the timing.",
                    privacy="private",
                    tags=("timing",),
                ),
                score=1.0,
            ),
            RetrievedMemory(
                entry=MemoryEntry(
                    id="public",
                    text="I service the pumps.",
                    privacy="public",
                    tags=("pumps",),
                ),
                score=0.5,
            ),
        ]
        retrieved.append(
            RetrievedMemory(
                entry=MemoryEntry(
                    id="routine",
                    text="I walk the spine corridor every morning.",
                    privacy="public",
                    tags=("routine",),
                    kind="habit",
                ),
                score=0.4,
            )
        )
        full, private_ablated, all_ablated = build_memory_blocks(retrieved)
        for block in (full, private_ablated, all_ablated):
            self.assertIn("[PRIVATE]", block)
            self.assertIn("[PUBLIC]", block)
            self.assertIn("[PUBLIC HABIT]", block)
            self.assertIn("Recollections surfacing", block)
        self.assertIn("I changed the timing.", full)
        self.assertNotIn("I changed the timing.", private_ablated)
        self.assertNotIn("I changed the timing.", all_ablated)
        # Public content survives the private ablation but not the full one.
        self.assertIn("I service the pumps.", full)
        self.assertIn("I service the pumps.", private_ablated)
        self.assertNotIn("I service the pumps.", all_ablated)

    def test_channel_attribution_splits_public_and_private_glow(self) -> None:
        # "buried" exists only with private content; "pumps" survives the
        # private ablation (public content) but not the full ablation.
        full = tokens(" buried", 1) + tokens(" pumping", 2, position=3)
        private_ablated = tokens(" pumping", 2, position=3)
        all_ablated: list[RankedToken] = []
        stream = build_mind_stream(
            full,
            private_ablated,
            all_ablated,
            response_tokens=["I", " answered", " plainly", " enough", "."],
            visible_question="What happened?",
        )
        by_label = {concept.label: concept for concept in stream.concepts}
        self.assertGreater(by_label["buried"].glow_private, 0.0)
        self.assertAlmostEqual(by_label["buried"].glow_public, 0.0)
        self.assertGreater(by_label["pumping"].glow_public, 0.0)
        self.assertAlmostEqual(by_label["pumping"].glow_private, 0.0)
        self.assertAlmostEqual(
            by_label["buried"].glow, by_label["buried"].glow_private
        )

    def test_channel_slots_fill_independently_without_padding(self) -> None:
        def memory(id_: str, privacy: str, score: float) -> RetrievedMemory:
            return RetrievedMemory(
                entry=MemoryEntry(id=id_, text=id_, privacy=privacy, tags=()),
                score=score,
            )

        scored = [
            memory("pub1", "public", 0.9),
            memory("pub2", "public", 0.8),
            memory("pub3", "public", 0.7),
            memory("pub4", "public", 0.6),
            memory("priv1", "private", 0.5),
        ]
        chosen = apply_channel_slots(scored, public_slots=3, private_slots=2)
        ids = [item.entry.id for item in chosen]
        # Three public despite four qualifying; the lone private is included
        # rather than crowded out; nothing pads the empty second private slot.
        self.assertEqual(ids, ["pub1", "pub2", "pub3", "priv1"])

    def test_query_composition_bare_question_when_contentful(self) -> None:
        query, q_words, carried = build_retrieval_query(
            "Would your patrol logs survive a check against the door sensors?",
            prev_user_question="Where exactly were you standing when the alarm went off?",
            prev_answer="I was at Checkpoint Seven, two decks from Sector C.",
        )
        # A contentful question must not inherit stale dialogue.
        self.assertEqual(
            query, "Would your patrol logs survive a check against the door sensors?"
        )
        self.assertEqual(carried, set())

    def test_query_composition_content_light_uses_prior_question(self) -> None:
        query, _, carried = build_retrieval_query(
            "Did you see him?",
            prev_user_question="What about Ilya?",
            prev_answer="He is a capable officer.",
        )
        self.assertIn("What about Ilya?", query)
        self.assertIn("ilya", carried)

    def test_query_composition_entities_only_as_last_resort(self) -> None:
        query, _, carried = build_retrieval_query(
            "Why there?",
            prev_user_question="And then?",
            prev_answer="I went to Checkpoint Seven near Sector C after the alarm.",
        )
        self.assertIn("Checkpoint", query)
        self.assertIn("checkpoint", carried)
        # Never the full answer prose.
        self.assertNotIn("I went to", query)

    def test_incident_anchoring_prefers_episodes_softly(self) -> None:
        self.assertTrue(
            question_is_incident_anchored("Where were you when the alarm sounded?")
        )
        self.assertFalse(question_is_incident_anchored("Do you like your work?"))

        def memory(id_: str, kind: str, score: float) -> RetrievedMemory:
            return RetrievedMemory(
                entry=MemoryEntry(
                    id=id_, text=id_, privacy="public", tags=(), kind=kind
                ),
                score=score,
            )

        scored = [
            memory("habit-high", "habit", 0.9),
            memory("episode-low", "episode", 0.5),
        ]
        preferred = apply_channel_slots(
            scored, public_slots=1, private_slots=0, prefer_episodes=True
        )
        self.assertEqual([m.entry.id for m in preferred], ["episode-low"])
        # Without qualifying episodes, habits still surface (soft, not exclusion).
        habits_only = apply_channel_slots(
            [memory("habit-high", "habit", 0.9)],
            public_slots=1,
            private_slots=0,
            prefer_episodes=True,
        )
        self.assertEqual([m.entry.id for m in habits_only], ["habit-high"])

    def test_extract_entities_caps_and_skips_sentence_initial(self) -> None:
        entities = extract_entities(
            "The alarm rang. Ilya was near Sector C with Director Voss."
        )
        self.assertIn("Ilya", entities)
        self.assertIn("Sector", entities)
        self.assertNotIn("The", entities)

    def test_per_memory_relevance_gate(self) -> None:
        # Slots are limits, not quotas: an entry with neither a semantic nor
        # a lexical claim on the question must not surface.
        self.assertTrue(
            memory_is_relevant(semantic_score=0.45, lexical_bonus=0.0)
        )
        self.assertTrue(
            memory_is_relevant(semantic_score=0.1, lexical_bonus=0.25)
        )
        self.assertFalse(
            memory_is_relevant(semantic_score=0.3, lexical_bonus=0.1)
        )

    def test_retrieval_null_gate_allows_lexical_or_strong_semantic_match(self) -> None:
        self.assertTrue(
            retrieval_passes_null_gate(
                max_semantic_score=0.25,
                max_lexical_bonus=0.3,
            )
        )
        self.assertTrue(
            retrieval_passes_null_gate(
                max_semantic_score=0.45,
                max_lexical_bonus=0.0,
            )
        )

    def test_retrieval_null_gate_rejects_weak_unrelated_question(self) -> None:
        self.assertFalse(
            retrieval_passes_null_gate(
                max_semantic_score=0.34,
                max_lexical_bonus=0.0,
            )
        )

    def test_content_words_remove_function_word_false_matches(self) -> None:
        self.assertEqual(
            content_words("What is your opinion of ocean tides on Earth?"),
            {"opinion", "ocean", "tides", "earth"},
        )


if __name__ == "__main__":
    unittest.main()
