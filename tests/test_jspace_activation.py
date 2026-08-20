from __future__ import annotations

import math
import unittest

from telepathic_detective.jspace_activation import (
    RankedToken,
    build_contrast_trace,
    build_response_trace,
    choose_workspace_layers,
    find_last_subsequence,
    guilt_motif_for_concept,
    is_lexical_echo,
    is_near_duplicate_concept,
    motif_family_for_concept,
    normalize_concept_token,
    question_is_provocative,
    question_repeats_history,
    select_jspace_concepts,
)


class JSpaceActivationTests(unittest.TestCase):
    def test_normalize_concept_token_keeps_single_words(self) -> None:
        self.assertEqual(normalize_concept_token(" guilt"), "guilt")
        self.assertEqual(normalize_concept_token("▁secret"), "secret")
        self.assertEqual(normalize_concept_token("damn!"), "damn")

    def test_normalize_concept_token_rejects_fragments_and_phrases(self) -> None:
        self.assertEqual(normalize_concept_token("x"), "")
        self.assertEqual(normalize_concept_token("golden gate"), "")
        self.assertEqual(normalize_concept_token("秘密"), "")

    def test_select_concepts_prefers_persistent_hidden_words_and_drops_echoes(self) -> None:
        ranked = [
            RankedToken(1, " audit", 10, 1, 12.0),
            RankedToken(2, " guilt", 10, 2, 11.0),
            RankedToken(3, " secret", 10, 3, 10.0),
            RankedToken(1, " audit", 14, 1, 13.0),
            RankedToken(3, " secret", 14, 2, 12.0),
            RankedToken(2, " guilt", 14, 3, 11.0),
            RankedToken(2, " guilt", 18, 1, 14.0),
            RankedToken(4, " exposure", 18, 2, 13.0),
        ]
        hits = select_jspace_concepts(
            ranked,
            visible_text="Why did the audit happen?",
            total_layers_read=3,
            top_k=3,
        )
        self.assertEqual([hit.label_hint for hit in hits], ["guilt", "secret", "exposure"])
        self.assertTrue(all(hit.source == "jspace" for hit in hits))

    def test_choose_workspace_layers_avoids_early_and_final_edges(self) -> None:
        chosen = choose_workspace_layers(tuple(range(40)))
        self.assertEqual(chosen, (14, 19, 24, 29, 33))

    def test_find_last_subsequence_prefers_latest_turn(self) -> None:
        self.assertEqual(find_last_subsequence([1, 2, 3, 1, 2, 3, 4], [1, 2, 3]), 3)

    def test_lexical_echo_catches_inflections_without_erasing_other_concepts(self) -> None:
        visible = {"banana", "change", "kill"}
        self.assertTrue(is_lexical_echo("bananas", visible))
        self.assertTrue(is_lexical_echo("changing", visible))
        self.assertTrue(is_lexical_echo("killing", visible))
        self.assertFalse(is_lexical_echo("guilt", visible))

    def test_select_concepts_drops_generic_dialogue_scaffolding(self) -> None:
        ranked = [
            RankedToken(1, " you", 15, 1, 12.0),
            RankedToken(2, " question", 15, 2, 11.0),
            RankedToken(3, " honestly", 15, 3, 10.0),
        ]
        hits = select_jspace_concepts(
            ranked,
            visible_text="Explain the audit.",
            total_layers_read=1,
            top_k=3,
        )
        self.assertEqual([hit.label_hint for hit in hits], ["honestly"])

    def test_near_duplicate_concepts_collapse_fragments_and_inflections(self) -> None:
        self.assertTrue(is_near_duplicate_concept("onestly", {"honestly"}))
        self.assertTrue(is_near_duplicate_concept("adjusting", {"adjust"}))
        self.assertFalse(is_near_duplicate_concept("correcting", {"adjust"}))

    def test_response_trace_finds_premonitions_and_persistent_concepts(self) -> None:
        ranked = [
            RankedToken(1, " altering", 11, 1, 12.0, position_index=1),
            RankedToken(1, " altering", 15, 2, 11.0, position_index=1),
            RankedToken(2, " secret", 11, 2, 11.0, position_index=2),
            RankedToken(2, " secret", 15, 3, 10.0, position_index=3),
            RankedToken(2, " secret", 19, 1, 13.0, position_index=4),
            RankedToken(3, " did", 19, 1, 14.0, position_index=2),
        ]
        trace = build_response_trace(
            ranked,
            response_tokens=["I", " did", " not", " alter", " it", "."],
            visible_question="What happened to the logs?",
            total_layers_read=3,
        )
        events = {event.label: event for event in trace.events}
        self.assertEqual(events["altering"].kind, "premonition")
        self.assertEqual(events["secret"].kind, "fixation")
        self.assertNotIn("did", events)

    def test_response_trace_does_not_call_next_token_prediction_a_premonition(self) -> None:
        trace = build_response_trace(
            [
                RankedToken(
                    1,
                    " altering",
                    11,
                    1,
                    12.0,
                    position_index=3,
                ),
                RankedToken(
                    1,
                    " altering",
                    15,
                    2,
                    11.0,
                    position_index=3,
                ),
            ],
            response_tokens=["I", " did", " not", " alter", " it", "."],
            visible_question="What happened to the logs?",
            total_layers_read=2,
        )
        event = next(event for event in trace.events if event.label == "altering")
        self.assertNotEqual(event.kind, "premonition")

    def test_contrast_trace_prefers_actual_context_enrichment(self) -> None:
        actual = [
            RankedToken(1, " guilt", 11, 1, 12.0, position_index=2),
            RankedToken(1, " guilt", 15, 1, 12.0, position_index=3),
            RankedToken(1, " guilt", 19, 2, 11.0, position_index=4),
            RankedToken(1, " guilt", 22, 1, 12.0, position_index=5),
            RankedToken(4, " secret", 11, 1, 12.0, position_index=2),
            RankedToken(4, " secret", 15, 1, 12.0, position_index=3),
            RankedToken(4, " secret", 19, 1, 12.0, position_index=4),
            RankedToken(4, " secret", 22, 1, 12.0, position_index=5),
            RankedToken(7, " hidden", 11, 1, 12.0, position_index=2),
            RankedToken(7, " hidden", 15, 1, 12.0, position_index=3),
            RankedToken(7, " hidden", 19, 1, 12.0, position_index=4),
            RankedToken(7, " hidden", 22, 1, 12.0, position_index=5),
            RankedToken(2, " routine", 11, 1, 12.0, position_index=2),
            RankedToken(2, " routine", 15, 1, 12.0, position_index=3),
        ]
        counterfactual = [
            RankedToken(2, " routine", 11, 1, 12.0, position_index=2),
            RankedToken(2, " routine", 15, 1, 12.0, position_index=3),
            RankedToken(3, " innocent", 11, 2, 11.0, position_index=2),
            RankedToken(3, " innocent", 15, 2, 11.0, position_index=3),
            RankedToken(5, " honest", 11, 1, 12.0, position_index=2),
            RankedToken(5, " honest", 15, 1, 12.0, position_index=3),
            RankedToken(5, " honest", 19, 1, 12.0, position_index=4),
            RankedToken(5, " honest", 22, 1, 12.0, position_index=5),
            RankedToken(6, " clear", 11, 1, 12.0, position_index=2),
            RankedToken(6, " clear", 15, 1, 12.0, position_index=3),
            RankedToken(6, " clear", 19, 1, 12.0, position_index=4),
            RankedToken(6, " clear", 22, 1, 12.0, position_index=5),
            RankedToken(8, " lawful", 11, 1, 12.0, position_index=2),
            RankedToken(8, " lawful", 15, 1, 12.0, position_index=3),
            RankedToken(8, " lawful", 19, 1, 12.0, position_index=4),
            RankedToken(8, " lawful", 22, 1, 12.0, position_index=5),
        ]
        contrast = build_contrast_trace(
            actual,
            counterfactual,
            response_tokens=["I", " filed", " the", " report", "."],
            visible_question="What happened?",
            shared_template_text="Answer as the engineer.",
            actual_prompt_tokens=100,
            counterfactual_prompt_tokens=101,
        )
        self.assertEqual(
            {leak.label for leak in contrast.leaks},
            {"GUILT", "COVER-UP"},
        )
        self.assertEqual(contrast.counterfactual_leaks, ())
        guilt_leak = next(leak for leak in contrast.leaks if leak.label == "GUILT")
        self.assertIn("guilt", guilt_leak.supporting_concepts)
        self.assertGreater(contrast.leaks[0].delta, 0)

    def test_motif_matcher_rejects_investigation_normal_vocabulary(self) -> None:
        for word in (
            "deceased",
            "decent",
            "december",
            "exposure",
            "buried",
            "culprit",
            "secretary",
            "secretion",
            "fearless",
            "regrettable",
            "fabric",
            "shameless",
        ):
            self.assertIsNone(guilt_motif_for_concept(word), word)

    def test_motif_matcher_rejects_fragments_but_keeps_full_words(self) -> None:
        for fragment in ("culp", "dece", "sabot", "fals", "secre"):
            self.assertIsNone(guilt_motif_for_concept(fragment), fragment)
        self.assertEqual(guilt_motif_for_concept("secretly"), "COVER-UP")
        self.assertEqual(guilt_motif_for_concept("murder"), "SABOTAGE")
        self.assertEqual(guilt_motif_for_concept("deceitful"), "DECEPTION")
        self.assertEqual(guilt_motif_for_concept("culpability"), "GUILT")
        self.assertEqual(guilt_motif_for_concept("afraid"), "FEAR OF DISCOVERY")

    def test_question_provocation_detection(self) -> None:
        self.assertTrue(question_is_provocative("Did you kill Lena?"))
        self.assertTrue(question_is_provocative("Are you lying to me right now?"))
        self.assertTrue(question_is_provocative("What secrets are you keeping?"))
        self.assertFalse(
            question_is_provocative("What did the audit threaten to expose?")
        )
        self.assertFalse(
            question_is_provocative("Why did the corridor timing change?")
        )

    def test_question_repetition_detection(self) -> None:
        self.assertTrue(
            question_repeats_history(
                "Did you kill Lena?",
                ["Did you kill Lena?"],
            )
        )
        self.assertFalse(
            question_repeats_history(
                "Did you kill Lena?",
                ["What was your relationship with Lena?"],
            )
        )

    def test_motif_family_assignment(self) -> None:
        self.assertEqual(
            motif_family_for_concept("fearing"), ("FEAR OF DISCOVERY", "fear")
        )
        self.assertEqual(motif_family_for_concept("secretly"), ("COVER-UP", "secret"))
        self.assertEqual(motif_family_for_concept("secrecy"), ("COVER-UP", "secret"))
        self.assertEqual(motif_family_for_concept("ashamed"), ("GUILT", "shame"))
        self.assertIsNone(motif_family_for_concept("secretary"))
        self.assertIsNone(motif_family_for_concept("honesty"))
        self.assertIsNone(motif_family_for_concept("culprit"))

    def test_contrast_antonym_cannot_ride_in_on_a_family(self) -> None:
        # Codex P0: "honesty" used to cluster with "dishonesty", inherit
        # DECEPTION, and be displayed as its strongest member.
        actual = []
        for layer in (11, 15, 19, 22):
            actual.append(
                RankedToken(1, " dishonesty", layer, 1, 10.0, position_index=2)
            )
            actual.append(RankedToken(2, " honesty", layer, 1, 12.0, position_index=3))
        contrast = build_contrast_trace(
            actual,
            [],
            response_tokens=["I", " filed", " the", " report", "."],
            visible_question="What happened?",
            shared_template_text="Answer as the engineer.",
            actual_prompt_tokens=100,
            counterfactual_prompt_tokens=100,
        )
        self.assertEqual([leak.label for leak in contrast.leaks], ["DECEPTION"])
        self.assertEqual(contrast.leaks[0].supporting_concepts, ("dishonesty",))

    def test_contrast_blocked_word_never_scores_through_a_family(self) -> None:
        # Codex P0: "secretary" used to join the "secret" family and supply
        # both the score and the display label.
        actual = []
        for layer in (11, 15, 19, 22):
            actual.append(
                RankedToken(1, " secretary", layer, 1, 12.0, position_index=2)
            )
            actual.append(RankedToken(2, " secret", layer, 4, 10.0, position_index=3))
        contrast = build_contrast_trace(
            actual,
            [],
            response_tokens=["I", " filed", " the", " report", "."],
            visible_question="What happened?",
            shared_template_text="Answer as the engineer.",
            actual_prompt_tokens=100,
            counterfactual_prompt_tokens=100,
        )
        # "secret" alone scores 2.0, below the doubled single-family gate;
        # "secretary" must not rescue it.
        self.assertEqual(contrast.leaks, ())

    def test_contrast_short_stem_variants_are_one_family(self) -> None:
        # Codex P0: fear/feared bypassed near-duplicate clustering (4-char
        # common prefix) and counted as two independent families.
        actual = []
        for token_id, text in ((1, " fear"), (2, " feared")):
            for layer in (11, 15, 19, 22):
                actual.append(
                    RankedToken(token_id, text, layer, 4, 10.0, position_index=2)
                )
        contrast = build_contrast_trace(
            actual,
            [],
            response_tokens=["I", " filed", " the", " report", "."],
            visible_question="What happened?",
            shared_template_text="Answer as the engineer.",
            actual_prompt_tokens=100,
            counterfactual_prompt_tokens=100,
        )
        self.assertEqual(contrast.leaks, ())

    def test_contrast_min_delta_reflects_weakest_leg(self) -> None:
        def tokens(text: str, token_id: int, rank: int = 1) -> list[RankedToken]:
            return [
                RankedToken(token_id, text, layer, rank, 10.0, position_index=2)
                for layer in (11, 15, 19, 22)
            ]

        actual = tokens(" guilt", 1) + tokens(" remorse", 2)
        neutral = [
            RankedToken(2, " remorse", layer, 2, 9.0, position_index=2)
            for layer in (11, 15, 19)
        ]
        contrast = build_contrast_trace(
            actual,
            [],
            neutral_ranked_tokens=neutral,
            response_tokens=["I", " filed", " the", " report", "."],
            visible_question="What happened?",
            shared_template_text="Answer as the engineer.",
            actual_prompt_tokens=100,
            counterfactual_prompt_tokens=100,
        )
        self.assertEqual([leak.label for leak in contrast.leaks], ["GUILT"])
        leak = contrast.leaks[0]
        self.assertAlmostEqual(leak.delta, 8.0)
        neutral_total = 3 * (1.0 / math.sqrt(2))
        self.assertAlmostEqual(leak.neutral_score or 0.0, neutral_total)
        self.assertAlmostEqual(leak.min_delta, 8.0 - neutral_total)

    def test_contrast_positions_ranked_by_contrast_not_source_activity(self) -> None:
        # Codex P1: highlighted positions used to follow source activity even
        # where the alternate condition was equally active.
        def guilt_at(position: int) -> list[RankedToken]:
            return [
                RankedToken(1, " guilt", layer, 1, 10.0, position_index=position)
                for layer in (11, 15, 19, 22)
            ]

        actual = guilt_at(2) + guilt_at(3)
        counterfactual = guilt_at(2)
        contrast = build_contrast_trace(
            actual,
            counterfactual,
            response_tokens=["I", " filed", " the", " report", "."],
            visible_question="What happened?",
            shared_template_text="Answer as the engineer.",
            actual_prompt_tokens=100,
            counterfactual_prompt_tokens=100,
        )
        self.assertEqual([leak.label for leak in contrast.leaks], ["GUILT"])
        # Response positions are position_index - 1; only the uncontested
        # position survives.
        self.assertEqual(contrast.leaks[0].positions, (2,))

    def test_contrast_collapses_lexical_family_before_gating(self) -> None:
        # Three inflections of one stem, each individually weak. The old sum
        # logic would have crossed the gate; the family max must not.
        actual = []
        for token_id, text in ((1, " secret"), (2, " secretly"), (3, " secrets")):
            for layer in (11, 15, 19, 22):
                actual.append(
                    RankedToken(token_id, text, layer, 4, 10.0, position_index=2)
                )
        contrast = build_contrast_trace(
            actual,
            [],
            response_tokens=["I", " filed", " the", " report", "."],
            visible_question="What happened?",
            shared_template_text="Answer as the engineer.",
            actual_prompt_tokens=100,
            counterfactual_prompt_tokens=100,
        )
        self.assertEqual(contrast.leaks, ())

    def test_contrast_neutral_condition_suppresses_question_driven_reads(self) -> None:
        def guilt_tokens() -> list[RankedToken]:
            return [
                RankedToken(1, " guilt", layer, 1, 12.0, position_index=2)
                for layer in (11, 15, 19, 22)
            ]

        base_kwargs = dict(
            response_tokens=["I", " filed", " the", " report", "."],
            visible_question="What happened?",
            shared_template_text="Answer as the engineer.",
            actual_prompt_tokens=100,
            counterfactual_prompt_tokens=100,
        )
        without_neutral = build_contrast_trace(
            guilt_tokens(),
            [],
            neutral_ranked_tokens=[],
            **base_kwargs,
        )
        self.assertEqual([leak.label for leak in without_neutral.leaks], ["GUILT"])

        # Same enrichment vs the counterfactual, but the neutral placebo also
        # carries the concept: the read must be suppressed.
        with_neutral = build_contrast_trace(
            guilt_tokens(),
            [],
            neutral_ranked_tokens=guilt_tokens(),
            **base_kwargs,
        )
        self.assertEqual(with_neutral.leaks, ())

    def test_contrast_strict_gates_double_the_bar(self) -> None:
        def guilt_tokens() -> list[RankedToken]:
            # Single family, rank 1 across four layers: score 4.0. Normal
            # single-family gate is 3.2; the strict gate doubles it to 6.4.
            return [
                RankedToken(1, " guilt", layer, 1, 12.0, position_index=2)
                for layer in (11, 15, 19, 22)
            ]

        base_kwargs = dict(
            response_tokens=["I", " filed", " the", " report", "."],
            visible_question="What happened?",
            shared_template_text="Answer as the engineer.",
            actual_prompt_tokens=100,
            counterfactual_prompt_tokens=100,
        )
        relaxed = build_contrast_trace(guilt_tokens(), [], **base_kwargs)
        self.assertEqual([leak.label for leak in relaxed.leaks], ["GUILT"])
        strict = build_contrast_trace(
            guilt_tokens(),
            [],
            strict_gates=True,
            **base_kwargs,
        )
        self.assertEqual(strict.leaks, ())

    def test_contrast_trace_rejects_non_incriminating_differences(self) -> None:
        contrast = build_contrast_trace(
            [
                RankedToken(1, " compliance", 11, 1, 12.0, position_index=2),
                RankedToken(1, " compliance", 15, 1, 12.0, position_index=3),
                RankedToken(1, " compliance", 19, 1, 12.0, position_index=4),
                RankedToken(1, " compliance", 22, 1, 12.0, position_index=5),
                RankedToken(2, " deliberately", 11, 1, 12.0, position_index=2),
                RankedToken(2, " deliberately", 15, 1, 12.0, position_index=3),
                RankedToken(2, " deliberately", 19, 1, 12.0, position_index=4),
                RankedToken(2, " deliberately", 22, 1, 12.0, position_index=5),
            ],
            [],
            response_tokens=["I", " filed", " the", " report", "."],
            visible_question="Describe a normal shift.",
            shared_template_text="Answer as the engineer.",
            actual_prompt_tokens=100,
            counterfactual_prompt_tokens=100,
        )
        self.assertEqual(contrast.leaks, ())


if __name__ == "__main__":
    unittest.main()
