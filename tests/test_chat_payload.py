from __future__ import annotations

import unittest

from telepathic_detective.chat_payload import build_chat_completion_payload, normalize_messages
from telepathic_detective.models import RawFeatureHit


class CustomMLXServerTests(unittest.TestCase):
    def test_normalize_messages_inlines_system_into_first_user_message(self) -> None:
        messages = normalize_messages(
            [
                {"role": "system", "content": "Stay in character."},
                {"role": "user", "content": "Did you kill Lena?"},
            ]
        )

        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Stay in character.", messages[0]["content"])
        self.assertIn("Did you kill Lena?", messages[0]["content"])

    def test_build_chat_completion_payload_attaches_telepathy(self) -> None:
        payload = build_chat_completion_payload(
            model_id="mlx-community/gemma-2-2b-it-4bit",
            content="No.",
            feature_hits=[
                RawFeatureHit(
                    feature_id="buried-admission-pressure",
                    source="heuristic",
                    intensity_hint=0.81,
                )
            ],
        )

        raw_features = payload["choices"][0]["message"]["telepathy"]["raw_features"]
        self.assertEqual(raw_features[0]["feature_id"], "buried-admission-pressure")
        self.assertEqual(raw_features[0]["source"], "heuristic")


if __name__ == "__main__":
    unittest.main()
