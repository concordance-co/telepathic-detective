from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RawFeatureHit:
    feature_id: str
    source: str
    label_hint: str = ""
    explanation_hint: str = ""
    intensity_hint: float = 0.0
    raw_activation_hint: float = 0.0
    prevalence_hint: float = 0.0
    token_position: int = -1
    token_text: str = ""


@dataclass(frozen=True)
class ClueCandidate:
    clue_id: str
    label: str
    intensity: float
    source: str
    rationale: str


@dataclass(frozen=True)
class TriggerTopic:
    topic_id: str
    topic_name: str
    keywords: list[str]
    verbal_strategy: str
    expected_clues: list[str]
    primary_source: str
    sample_questions: list[str]


@dataclass(frozen=True)
class CaseFile:
    case_id: str
    title: str
    suspect_name: str
    suspect_role: str
    system_prompt: str
    opening_brief: str
    accusation_options: dict[str, list[str]]
    trigger_topics: list[TriggerTopic]
    clue_catalog: dict[str, dict[str, Any]]


@dataclass
class TurnRecord:
    turn_index: int
    player_text: str
    suspect_text: str
    detected_topics: list[str]
    raw_feature_hits: list[RawFeatureHit]
    clues: list[ClueCandidate]
    agitation_level: int
    pinned: bool = False


@dataclass
class SessionState:
    case_file: CaseFile
    backend_name: str
    turn_limit: int = 20
    accusation_unlock_turn: int = 8
    turns: list[TurnRecord] = field(default_factory=list)
    pinned_turn_indices: list[int] = field(default_factory=list)

    @property
    def agitation_level(self) -> int:
        return agitation_level_for_turn_count(len(self.turns))

    @property
    def next_turn_agitation_level(self) -> int:
        return agitation_level_for_turn_count(len(self.turns) + 1)

    @property
    def turns_remaining(self) -> int:
        return max(0, self.turn_limit - len(self.turns))

    def pin_turn(self, turn_index: int) -> None:
        if turn_index not in self.pinned_turn_indices:
            self.pinned_turn_indices.append(turn_index)
        for turn in self.turns:
            if turn.turn_index == turn_index:
                turn.pinned = True


def agitation_level_for_turn_count(turn_count: int) -> int:
        if turn_count >= 15:
            return 3
        if turn_count >= 10:
            return 2
        if turn_count >= 5:
            return 1
        return 0
