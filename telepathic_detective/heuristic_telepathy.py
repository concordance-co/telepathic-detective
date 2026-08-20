from __future__ import annotations

import re

from telepathic_detective.models import CaseFile, RawFeatureHit
from telepathic_detective.topics import detect_topics, normalize_text


def extract_heuristic_feature_hits(
    *,
    case_file: CaseFile,
    player_text: str,
    suspect_text: str,
) -> list[RawFeatureHit]:
    detected_topics = detect_topics(player_text, case_file.trigger_topics)
    normalized_response = normalize_text(suspect_text)
    hits: list[RawFeatureHit] = []

    for topic in detected_topics[:2]:
        for feature_id in topic.expected_clues[:2]:
            intensity = 0.72
            if _response_supports_feature(feature_id, normalized_response):
                intensity = 0.88
            hits.append(
                RawFeatureHit(
                    feature_id=feature_id,
                    source="heuristic",
                    intensity_hint=intensity,
                )
            )

    lexical_rules = [
        (("audit", "routine"), "inspection-pressure", 0.84),
        (("accident", "mechanism"), "forensic-reconstruction-fear", 0.83),
        (("triage",), "triage-rationalization", 0.87),
        (("protect", "alive"), "protector-identity", 0.83),
        (("not random",), "rationalized-inevitability", 0.9),
    ]
    combined_text = f"{normalize_text(player_text)} {normalized_response}"
    for cues, feature_id, intensity in lexical_rules:
        if all(cue in combined_text for cue in cues):
            hits.append(
                RawFeatureHit(
                    feature_id=feature_id,
                    source="heuristic",
                    intensity_hint=intensity,
                )
            )

    deduped: dict[str, RawFeatureHit] = {}
    for hit in hits:
        existing = deduped.get(hit.feature_id)
        if existing is None or hit.intensity_hint > existing.intensity_hint:
            deduped[hit.feature_id] = hit
    return list(deduped.values())


def extract_player_text_from_messages(messages: list[dict[str, str]]) -> str:
    user_messages = [message["content"] for message in messages if message.get("role") == "user"]
    if not user_messages:
        return ""
    latest = user_messages[-1]
    match = re.search(r"\[agitation=\d+\]\s*(.*)", latest, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return latest.strip()


def _response_supports_feature(feature_id: str, response: str) -> bool:
    cue_map = {
        "inspection-pressure": ("audit", "routine"),
        "imminent-exposure-panic": ("blame", "public"),
        "buried-admission-pressure": ("no.",),
        "rationalized-inevitability": ("need", "focus"),
        "triage-rationalization": ("triage",),
        "protector-identity": ("alive", "protect"),
        "forensic-reconstruction-fear": ("mechanism", "technical"),
    }
    cues = cue_map.get(feature_id)
    if not cues:
        return False
    return all(cue in response for cue in cues)
