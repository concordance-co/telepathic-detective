from __future__ import annotations

import json
import time
from typing import Any



def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    system_messages = [str(message.get("content", "")) for message in messages if message.get("role") == "system"]
    non_system = [
        {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        for message in messages
        if message.get("role") != "system"
    ]
    if system_messages:
        system_blob = "\n\n".join(system_messages)
        if non_system and non_system[0]["role"] == "user":
            non_system[0]["content"] = f"{system_blob}\n\n{non_system[0]['content']}"
        else:
            non_system.insert(0, {"role": "user", "content": system_blob})
    return non_system


def build_chat_completion_payload(*, model_id: str, content: str, feature_hits) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "telepathy": {
                        "raw_features": [
                            {
                                "feature_id": hit.feature_id,
                                "source": hit.source,
                                "label_hint": hit.label_hint,
                                "explanation_hint": hit.explanation_hint,
                                "intensity_hint": hit.intensity_hint,
                            }
                            for hit in feature_hits
                        ]
                    },
                },
                "finish_reason": "stop",
            }
        ],
    }
