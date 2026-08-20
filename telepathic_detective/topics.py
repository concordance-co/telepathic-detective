from __future__ import annotations

import re

from telepathic_detective.models import TriggerTopic


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_topics(player_text: str, topics: list[TriggerTopic]) -> list[TriggerTopic]:
    normalized = normalize_text(player_text)
    matches: list[tuple[int, TriggerTopic]] = []
    for topic in topics:
      score = sum(1 for keyword in topic.keywords if keyword in normalized)
      if score > 0:
          matches.append((score, topic))
    matches.sort(key=lambda item: (-item[0], item[1].topic_id))
    return [topic for _, topic in matches]
