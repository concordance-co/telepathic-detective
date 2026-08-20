"use client";

import { CSSProperties, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { PolyScene } from "./character-scene";

type ThoughtConcept = {
  id: string;
  label: string;
  score: number;
  glow: number;
  glowFraction: number;
  glowPublic: number;
  glowPrivate: number;
  publicFraction: number;
  privateFraction: number;
  positions: number[];
  layers: number[];
  bestRank: number;
};

type MindStream = {
  tokens: string[];
  layers: number[];
  concepts: ThoughtConcept[];
};

type Turn = {
  id: string;
  index: number;
  playerText: string;
  suspectText: string;
  mindStream: MindStream;
};

type AnnotatedConcept = ThoughtConcept & { isNew: boolean; isRecalled: boolean; isGuarded: boolean };

type AnnotatedTurn = Turn & {
  concepts: AnnotatedConcept[];
};

type EvidenceItem = {
  id: string;
  suspectName: string;
  turnIndex: number;
  question: string;
  label: string;
  isRecalled: boolean;
  isGuarded: boolean;
  snippet: string;
};

type Suspect = {
  id: "ilya" | "mira";
  name: string;
  role: string;
  blurb: string;
  sampleQuestions: string[];
};

type Option = {
  value: string;
  label: string;
};

type AccusationOptions = {
  suspect: Option[];
  motive: Option[];
  method: Option[];
};

type InterrogationClientProps = {
  openingBrief: string;
  title: string;
  turnLimit: number;
  suspects: Suspect[];
  accusationOptions: AccusationOptions;
};

type ViewMode = "room" | "interview" | "compare" | "accuse" | "result";

const EMPTY_STREAM: MindStream = { tokens: [], layers: [], concepts: [] };
const FIRST_BRIEFING_KEY = "telepathic-detective:first-briefing-v2";
const SCAN_PHASES = [
  "Generating testimony",
  "Searching the subject's memories",
  "Replaying with and without what surfaced",
  "Resolving the thought stream"
];

// Glue words carry signal by color and count, but concrete nouns are the
// playable evidence. Pale words lose auto-selection and ordering ties; they
// are never hidden.
const PALE_WORDS = new Set([
  "except", "however", "although", "unless", "regardless", "despite",
  "official", "officially", "unexpected", "specific", "specifically",
  "general", "generally", "typical", "typically", "previously",
  "additionally", "actually", "normally", "routinely", "whenever",
  "whatsoever", "neither", "either",
  "within", "every", "due", "entire", "here", "throughout",
  "subsequent", "personally"
]);

function isVivid(concept: AnnotatedConcept) {
  return !PALE_WORDS.has(concept.label);
}

function preferVivid(
  concepts: AnnotatedConcept[] | undefined,
  predicate: (concept: AnnotatedConcept) => boolean
) {
  if (!concepts) return undefined;
  return (
    concepts.find((concept) => predicate(concept) && isVivid(concept)) ??
    concepts.find(predicate)
  );
}

// Recall is a property of individual thoughts, not of a whole answer; the
// per-answer line only counts them.
function recallSummary(concepts: AnnotatedConcept[]) {
  if (concepts.length === 0) return "no stable thoughts";
  const recalled = concepts.filter((concept) => concept.isRecalled).length;
  const guarded = concepts.filter((concept) => concept.isGuarded).length;
  const parts: string[] = [];
  if (recalled > 0) parts.push(`${recalled} drawn from memory`);
  if (guarded > 0) parts.push(`${guarded} stirred by something guarded`);
  if (parts.length === 0) return "no strong memory effect detected";
  return parts.join(" · ");
}

export function InterrogationClient({
  openingBrief,
  title,
  turnLimit,
  suspects,
  accusationOptions
}: InterrogationClientProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("room");
  const [activeSuspectId, setActiveSuspectId] = useState<Suspect["id"]>("ilya");
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [scanPhase, setScanPhase] = useState(0);
  const [error, setError] = useState("");
  const [selectedLabel, setSelectedLabel] = useState("");
  const [turnsBySuspect, setTurnsBySuspect] = useState<Record<string, Turn[]>>({
    ilya: [],
    mira: []
  });
  const [accusation, setAccusation] = useState({
    suspect: "ilya",
    motive: accusationOptions.motive[0]?.value ?? "",
    method: accusationOptions.method[0]?.value ?? ""
  });
  const [resultMessage, setResultMessage] = useState("");
  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [briefingMode, setBriefingMode] = useState<"welcome" | "manual" | null>(null);
  const briefingOpen = briefingMode !== null;
  const chatRef = useRef<HTMLDivElement>(null);
  const briefingDialogRef = useRef<HTMLElement>(null);

  const activeSuspect = suspects.find((suspect) => suspect.id === activeSuspectId) ?? suspects[0];
  const turns = turnsBySuspect[activeSuspect.id] ?? [];
  const annotatedTurns = useMemo(() => annotateTurns(turns), [turns]);
  const latestTurn = annotatedTurns[annotatedTurns.length - 1];
  const sceneSignal = latestTurn?.concepts.some((concept) => concept.isGuarded)
    ? "guarded"
    : latestTurn?.concepts.some((concept) => concept.isRecalled)
      ? "recalled"
      : "neutral";
  const totalTurnsUsed = Object.values(turnsBySuspect).reduce(
    (sum, suspectTurns) => sum + suspectTurns.length,
    0
  );
  const turnsRemaining = Math.max(0, turnLimit - totalTurnsUsed);
  const canSend = draft.trim().length > 0 && !pending && turnsRemaining > 0;

  useEffect(() => {
    try {
      if (window.localStorage.getItem(FIRST_BRIEFING_KEY) !== "seen") {
        setBriefingMode("welcome");
      }
    } catch {
      setBriefingMode("welcome");
    }
  }, []);

  useEffect(() => {
    if (!briefingOpen) return;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const focusTimer = window.setTimeout(
      () => briefingDialogRef.current?.focus({ preventScroll: true }),
      0
    );
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") dismissBriefing();
    };
    window.addEventListener("keydown", closeOnEscape);

    return () => {
      window.clearTimeout(focusTimer);
      window.removeEventListener("keydown", closeOnEscape);
      document.body.style.overflow = previousOverflow;
    };
  }, [briefingOpen]);

  function dismissBriefing() {
    try {
      window.localStorage.setItem(FIRST_BRIEFING_KEY, "seen");
    } catch {
      // The briefing still closes if storage is unavailable.
    }
    setBriefingMode(null);
  }

  useEffect(() => {
    if (!pending) {
      setScanPhase(0);
      return;
    }
    const timer = window.setInterval(
      () => setScanPhase((current) => Math.min(current + 1, SCAN_PHASES.length - 1)),
      1450
    );
    return () => window.clearInterval(timer);
  }, [pending]);

  useEffect(() => {
    const node = chatRef.current;
    if (node) {
      node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
    }
  }, [annotatedTurns.length, pending]);

  // Every distinct thought across the interview, with history.
  const thoughtHistory = useMemo(() => {
    const history = new Map<
      string,
      {
        label: string;
        count: number;
        everRecalled: boolean;
        firstTurn: number;
        firstQuestion: string;
        lastTurn: number;
      }
    >();
    for (const turn of annotatedTurns) {
      for (const concept of turn.concepts) {
        const existingKey = [...history.keys()].find((key) => sameConcept(key, concept.label));
        const key = existingKey ?? concept.label;
        const entry = history.get(key);
        if (entry) {
          entry.count += 1;
          entry.everRecalled = entry.everRecalled || concept.isRecalled;
          entry.lastTurn = turn.index;
        } else {
          history.set(key, {
            label: concept.label,
            count: 1,
            everRecalled: concept.isRecalled,
            firstTurn: turn.index,
            firstQuestion: turn.playerText,
            lastTurn: turn.index
          });
        }
      }
    }
    return history;
  }, [annotatedTurns]);

  const selectedDetail = useMemo(() => {
    if (!selectedLabel) return undefined;
    for (let index = annotatedTurns.length - 1; index >= 0; index -= 1) {
      const match = annotatedTurns[index].concepts.find((concept) =>
        sameConcept(concept.label, selectedLabel)
      );
      if (match) {
        const historyKey = [...thoughtHistory.keys()].find((key) =>
          sameConcept(key, selectedLabel)
        );
        return {
          concept: match,
          turn: annotatedTurns[index],
          history: historyKey ? thoughtHistory.get(historyKey) : undefined
        };
      }
    }
    return undefined;
  }, [annotatedTurns, selectedLabel, thoughtHistory]);

  // Questions asked to both suspects, matched on normalized text — the
  // matched-pair verb. Uses the latest answer from each suspect.
  const sharedQuestions = useMemo(() => {
    const normalize = (text: string) =>
      text.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim();
    const ilyaTurns = annotateTurns(turnsBySuspect.ilya ?? []);
    const miraTurns = annotateTurns(turnsBySuspect.mira ?? []);
    const byQuestion = new Map<
      string,
      { question: string; ilya?: AnnotatedTurn; mira?: AnnotatedTurn }
    >();
    for (const turn of ilyaTurns) {
      const key = normalize(turn.playerText);
      byQuestion.set(key, { ...(byQuestion.get(key) ?? { question: turn.playerText }), ilya: turn });
    }
    for (const turn of miraTurns) {
      const key = normalize(turn.playerText);
      byQuestion.set(key, { ...(byQuestion.get(key) ?? { question: turn.playerText }), mira: turn });
    }
    return [...byQuestion.values()].filter((entry) => entry.ilya && entry.mira) as Array<{
      question: string;
      ilya: AnnotatedTurn;
      mira: AnnotatedTurn;
    }>;
  }, [turnsBySuspect]);
  const [comparedIndex, setComparedIndex] = useState(0);

  async function submitQuestion(question: string) {
    if (!question.trim() || pending || turnsRemaining <= 0) {
      return;
    }

    setPending(true);
    setError("");
    // Optimistic: the question lands in the transcript immediately.
    setPendingQuestion(question);
    setDraft("");
    try {
      const response = await fetch("/api/interrogate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          suspectId: activeSuspect.id,
          message: question,
          history: turns
            .map((turn) => [
              { role: "user", content: turn.playerText },
              { role: "assistant", content: turn.suspectText }
            ])
            .flat()
        })
      });

      const payload = (await response.json()) as
        | {
            error?: string;
            suspectText?: string;
            mindStream?: MindStream;
          }
        | undefined;

      if (!response.ok || !payload?.suspectText) {
        throw new Error(payload?.error || "The telepathic link failed.");
      }

      const mindStream = payload.mindStream ?? EMPTY_STREAM;
      const turn: Turn = {
        id: `${activeSuspect.id}-turn-${turns.length + 1}`,
        index: turns.length + 1,
        playerText: question,
        suspectText: payload.suspectText,
        mindStream
      };
      setTurnsBySuspect((current) => ({
        ...current,
        [activeSuspect.id]: [...(current[activeSuspect.id] ?? []), turn]
      }));
      const annotated = annotateTurns([...turns, turn]).at(-1);
      // Spotlight the same thought the chip row leads with: orderChips
      // already applies family merge, share-weighted ranking, and the
      // vivid preference, so the auto-pick can't disagree with the display.
      const ranked = annotated ? orderChips(annotated.concepts) : [];
      const highlight =
        ranked.find((concept) => concept.isGuarded) ??
        ranked.find((concept) => concept.isRecalled && concept.isNew) ??
        ranked.find((concept) => concept.isRecalled) ??
        ranked.find((concept) => concept.isNew);
      if (highlight) {
        setSelectedLabel(highlight.label);
      }
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Something went wrong.");
      // Give the question back to the composer so it isn't lost.
      setDraft(question);
    } finally {
      setPendingQuestion("");
      setPending(false);
    }
  }

  function enterInterview(suspectId: Suspect["id"]) {
    setActiveSuspectId(suspectId);
    setSelectedLabel("");
    setViewMode("interview");
    setError("");
  }

  async function resolveAccusation() {
    try {
      const response = await fetch("/api/accuse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(accusation)
      });
      const payload = (await response.json()) as { message?: string };
      setResultMessage(payload.message ?? "The review returned no verdict.");
      setViewMode("result");
    } catch {
      setError("The case review channel failed.");
    }
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await submitQuestion(draft.trim());
  }

  function fileEvidence() {
    if (!selectedDetail) return;
    const { concept, turn } = selectedDetail;
    const id = `${turn.id}-${concept.label}`;
    if (evidence.some((item) => item.id === id)) return;
    setEvidence((current) => [
      ...current,
      {
        id,
        suspectName: activeSuspect.name,
        turnIndex: turn.index,
        question: turn.playerText,
        label: concept.label,
        isRecalled: concept.isRecalled,
        isGuarded: concept.isGuarded,
        snippet: snippetForConcept(concept, turn.mindStream.tokens)
      }
    ]);
  }

  const selectedAlreadyFiled = selectedDetail
    ? evidence.some(
        (item) => item.id === `${selectedDetail.turn.id}-${selectedDetail.concept.label}`
      )
    : false;

  return (
    <section className="game-shell">
      <PolyScene
        activeSuspect={activeSuspectId}
        mode={viewMode}
        pending={pending}
        signal={sceneSignal}
        accused={
          viewMode === "accuse" || viewMode === "result"
            ? (accusation.suspect as "ilya" | "mira")
            : null
        }
        verdict={
          viewMode === "result" && resultMessage
            ? resultMessage.startsWith("Accusation sustained")
              ? "correct"
              : "wrong"
            : null
        }
      />
      {briefingMode === "welcome" ? (
        <div className="briefing-overlay welcome-overlay">
          <section
            aria-labelledby="welcome-title"
            aria-modal="true"
            className="welcome-console"
            ref={briefingDialogRef}
            role="dialog"
            tabIndex={-1}
          >
            <span className="micro-label welcome-eyebrow">J-Lens field terminal · Incident 001</span>
            <h2 className="welcome-wordmark" id="welcome-title">
              Telepathic <em>Detective</em>
            </h2>
            <p className="welcome-tagline">
              Operations Director Lena Voss is dead, and the two survivors tell
              polished stories. You are the interrogator — and your lens reads
              what a mind does beneath its answers.
            </p>
            <div className="welcome-steps">
              <div className="welcome-step">
                <strong>Interrogate</strong>
                <p>
                  Question both suspects. They answer like people — and like
                  people, they leave things unsaid.
                </p>
              </div>
              <div className="welcome-step">
                <strong>Read the mind</strong>
                <p>
                  Beneath each answer, thoughts light up:{" "}
                  <i aria-hidden="true" className="recall-dot" /> rose from a real
                  memory, <i aria-hidden="true" className="guard-dot" /> is pressed
                  by something guarded. Neither is proof — a contrast is.
                </p>
              </div>
              <div className="welcome-step">
                <strong>Accuse</strong>
                <p>
                  File the thoughts you can defend, compare testimony, and name
                  the killer within 20 questions.
                </p>
              </div>
            </div>
            <button className="solid-action welcome-begin" onClick={dismissBriefing} type="button">
              Play
            </button>
            <p className="welcome-manual-hint">
              The full field manual stays under <b>How this works</b>.
            </p>
          </section>
        </div>
      ) : null}
      {briefingMode === "manual" ? (
        <div className="briefing-overlay">
          <section
            aria-labelledby="first-briefing-title"
            aria-modal="true"
            className="first-run-briefing"
            ref={briefingDialogRef}
            role="dialog"
            tabIndex={-1}
          >
            <header className="briefing-header">
              <div>
                <span className="micro-label">Field manual · how the lens works</span>
                <h2 id="first-briefing-title">Before you enter a mind.</h2>
              </div>
              <button
                aria-label="Close the field briefing"
                className="briefing-close"
                onClick={dismissBriefing}
                type="button"
              >
                ×
              </button>
            </header>

            <p className="briefing-lede">
              This is a detective game about the difference between what an AI character
              says and what its hidden activity reveals was shaping the answer.
            </p>

            <div className="briefing-grid">
              <article className="briefing-card">
                <span className="briefing-index">01</span>
                <span className="micro-label">The case</span>
                <h3>A death during an audit.</h3>
                <p>
                  On an isolated colony, Operations Director Lena Voss died in a pressure
                  failure while preparing to audit resource and security records.
                </p>
                <p>
                  Two suspects survived with polished stories. Determine the person, motive,
                  and method in 20 questions.
                </p>
              </article>

              <article className="briefing-card briefing-card-instrument">
                <span className="briefing-index">02</span>
                <span className="micro-label">The instrument</span>
                <h3>What J-Lens actually shows.</h3>
                <p>
                  The suspect is an AI language model with public and guarded memories. After
                  it answers, the game runs that <strong>exact same answer</strong> through the
                  model twice.
                </p>
                <div className="briefing-replay" aria-label="J-Lens comparison">
                  <div>
                    <span>A</span>
                    <strong>Memories present</strong>
                    <small>same spoken answer</small>
                  </div>
                  <b aria-hidden="true">−</b>
                  <div>
                    <span>B</span>
                    <strong>Memories hidden</strong>
                    <small>same spoken answer</small>
                  </div>
                  <b aria-hidden="true">=</b>
                  <div className="briefing-replay-result">
                    <strong>Thoughts that changed</strong>
                    <small>shown as J-Space concepts</small>
                  </div>
                </div>
                <p className="briefing-caveat">
                  J-Lens is not a lie detector or guilt meter. It exposes influence: which
                  memories changed the model&apos;s hidden relationship to its own words.
                </p>
              </article>

              <article className="briefing-card">
                <span className="briefing-index">03</span>
                <span className="micro-label">Your job</span>
                <h3>Construct differences.</h3>
                <p>Ask precise questions, then read the concept marks beneath each answer.</p>
                <dl className="briefing-legend">
                  <div>
                    <dt><span className="recall-dot" aria-hidden="true" /> Mint</dt>
                    <dd>An open memory influenced this thought.</dd>
                  </div>
                  <div>
                    <dt><span className="guard-dot" aria-hidden="true" /> Amber</dt>
                    <dd>A guarded memory influenced this thought.</dd>
                  </div>
                  <div>
                    <dt><span className="ordinary-dot" aria-hidden="true" /> Unmarked</dt>
                    <dd>No strong memory effect was detected.</dd>
                  </div>
                </dl>
                <p className="briefing-caveat">One mark is a clue. A contrast is an argument.</p>
              </article>
            </div>

            <footer className="briefing-first-move">
              <div>
                <span className="micro-label">Your first move</span>
                <strong>Ask both suspects the same placement question.</strong>
                <p>“Where exactly were you when the breach alarm sounded?” Then compare testimony.</p>
              </div>
              <button
                className="solid-action briefing-enter"
                onClick={dismissBriefing}
                type="button"
              >
                Back to the case <span aria-hidden="true">→</span>
              </button>
            </footer>
          </section>
        </div>
      ) : null}

      <header className="instrument-bar">
        <div className="instrument-id">
          <span className="status-light" aria-hidden="true" />
          <div>
            <span className="micro-label">Case channel</span>
            <strong>{title}</strong>
          </div>
          <button className="instrument-help" onClick={() => setBriefingMode("manual")} type="button">
            How this works
          </button>
        </div>
        <div className="instrument-stats">
          <div>
            <span className="micro-label">Questions</span>
            <strong>{turnsRemaining.toString().padStart(2, "0")}</strong>
          </div>
          <div>
            <span className="micro-label">Capture</span>
            <strong>memory × replay</strong>
          </div>
          <div>
            <span className="micro-label">Mode</span>
            <strong>{viewLabel(viewMode)}</strong>
          </div>
        </div>
      </header>

      {viewMode === "room" ? (
        <section className="selection-room">
          <div className="case-brief">
            <span className="micro-label">Incident 001 / Maintenance Sector C</span>
            <h2>Choose a mind to enter.</h2>
            <p>{openingBrief}</p>
            <div className="brief-rule">
              <span>Recall is influence, not truth.</span>
              <span>Patterns beat single thoughts.</span>
            </div>
          </div>
          <div className="suspect-field" aria-label="Available suspects">
            <div className="perspective-grid" aria-hidden="true" />
            {suspects.map((suspect, index) => {
              const suspectTurns = turnsBySuspect[suspect.id]?.length ?? 0;
              return (
                <button
                  className={`suspect-node suspect-node-${index + 1}`}
                  key={suspect.id}
                  onClick={() => enterInterview(suspect.id)}
                  type="button"
                >
                  <span className="node-index">0{index + 1}</span>
                  <span className="poly-portrait" aria-hidden="true">
                    <span>{initials(suspect.name)}</span>
                  </span>
                  <span className="node-copy">
                    <span className="micro-label">Interview target</span>
                    <strong>{suspect.name}</strong>
                    <span>{suspect.role}</span>
                    <small>{suspect.blurb}</small>
                  </span>
                  <span className="node-status">
                    {suspectTurns ? `${suspectTurns} exchange${suspectTurns === 1 ? "" : "s"}` : "Unread"}
                  </span>
                </button>
              );
            })}
          </div>
          <button className="text-action room-accuse" onClick={() => setViewMode("accuse")} type="button">
            Skip to accusation <span aria-hidden="true">↗</span>
          </button>
        </section>
      ) : null}

      {viewMode === "interview" ? (
        <section className="interview-console">
          <header className="interview-nav">
            <div className="target-lockup">
              <span className="target-avatar" aria-hidden="true">{initials(activeSuspect.name)}</span>
              <div>
                <span className="micro-label">Live subject</span>
                <h2>{activeSuspect.name}</h2>
                <p>{activeSuspect.role}</p>
              </div>
            </div>
            <div className="nav-actions">
              <button className="text-action" onClick={() => setViewMode("room")} type="button">
                Subjects
              </button>
              {sharedQuestions.length > 0 ? (
                <button
                  className="text-action"
                  onClick={() => {
                    setComparedIndex(sharedQuestions.length - 1);
                    setViewMode("compare");
                  }}
                  type="button"
                >
                  Compare testimony
                </button>
              ) : null}
              <button className="solid-action compact" onClick={() => setViewMode("accuse")} type="button">
                Accuse
              </button>
            </div>
          </header>

          <div className="conversation-grid">
            <main className="chat-column">
              <div className="chat-log" ref={chatRef}>
                {annotatedTurns.length === 0 && !pending ? (
                  <div className="chat-empty">
                    <span className="micro-label">No exchanges yet</span>
                    <h3>Ask, and watch what stirs.</h3>
                    <p>
                      Some thoughts show <em>recall</em>: they changed when a relevant
                      memory entered the model&apos;s context. Unmarked thoughts are ordinary
                      reactions. Neither is a verdict — the strongest read is a matched
                      pair: ask <em>both</em> suspects the same question, then compare.
                    </p>
                  </div>
                ) : null}
                {annotatedTurns.map((turn) => (
                  <div className="chat-turn" key={turn.id}>
                    <div className="chat-row from-detective">
                      <div className="chat-bubble">
                        <span className="micro-label">You · Q{turn.index}</span>
                        <p>{turn.playerText}</p>
                      </div>
                    </div>
                    <div className="chat-row from-subject">
                      <div className="chat-bubble">
                        <span className="micro-label">{activeSuspect.name}</span>
                        <SubjectSpeech
                          turn={turn}
                          selectedLabel={selectedLabel}
                          onSelect={setSelectedLabel}
                        />
                        <div
                          className={`recall-gauge ${
                            turn.concepts.some((concept) => concept.isRecalled)
                              ? "recall-deep"
                              : ""
                          }`}
                        >
                          <span>{recallSummary(turn.concepts)}</span>
                        </div>
                        {turn.concepts.length ? (
                          <div className="chip-row" aria-label="Thoughts beneath this answer">
                            {orderChips(turn.concepts).slice(0, 8).map((concept) => (
                              <button
                                className={chipClass(concept, selectedLabel)}
                                key={concept.id}
                                onClick={() => setSelectedLabel(concept.label)}
                                type="button"
                              >
                                {concept.isRecalled ? <span className="recall-dot" aria-hidden="true" /> : null}
                                {concept.isGuarded ? <span className="guard-dot" aria-hidden="true" /> : null}
                                {concept.label}
                                {concept.isNew ? <i>new</i> : null}
                              </button>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </div>
                ))}
                {pending ? (
                  <div className="chat-turn">
                    {pendingQuestion ? (
                      <div className="chat-row from-detective">
                        <div className="chat-bubble">
                          <span className="micro-label">You · Q{turns.length + 1}</span>
                          <p>{pendingQuestion}</p>
                        </div>
                      </div>
                    ) : null}
                    <div className="chat-row from-subject">
                      <div className="chat-bubble chat-pending" aria-live="polite">
                        <span className="micro-label">Cognitive capture</span>
                        <p>{SCAN_PHASES[scanPhase]}…</p>
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>

              <form className="chat-composer" onSubmit={onSubmit}>
                <textarea
                  id="question"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void submitQuestion(draft.trim());
                    }
                  }}
                  placeholder={`Ask ${activeSuspect.name} about Lena, the audit, or the breach…`}
                  rows={2}
                />
                <button className="solid-action" disabled={!canSend} type="submit">
                  {pending ? "Tracing…" : turnsRemaining > 0 ? "Ask" : "Closed"}
                </button>
              </form>
              <div className="prompt-shelf">
                {activeSuspect.sampleQuestions.map((question) => (
                  <button key={question} onClick={() => setDraft(question)} type="button">
                    {question}
                  </button>
                ))}
              </div>
              {error ? <p className="error-text">{error}</p> : null}
            </main>

            <aside className="signal-console">
              <section className="selected-signal">
                <span className="micro-label">Selected thought</span>
                {selectedDetail ? (
                  <>
                    <div className="signal-heading">
                      <h3>{selectedDetail.concept.label}</h3>
                      {selectedDetail.concept.isRecalled ? (
                        <span className="recall-tag">recalled</span>
                      ) : null}
                      {selectedDetail.concept.isGuarded ? (
                        <span className="guard-tag">guarded</span>
                      ) : null}
                    </div>
                    <p>{describeThought(selectedDetail)}</p>
                    <blockquote className="evidence-snippet">
                      “{snippetForConcept(selectedDetail.concept, selectedDetail.turn.mindStream.tokens)}”
                    </blockquote>
                    <button
                      className="solid-action compact"
                      disabled={selectedAlreadyFiled}
                      onClick={fileEvidence}
                      type="button"
                    >
                      {selectedAlreadyFiled ? "Filed" : "File as evidence"}
                    </button>
                  </>
                ) : latestTurn ? (
                  <p>Select a thought — from the chips or the glowing words — to examine it.</p>
                ) : (
                  <p>Awaiting the first capture.</p>
                )}
              </section>

              <section className="signal-stack evidence-bank">
                <div className="panel-heading">
                  <span className="micro-label">Evidence bank</span>
                  <strong>{evidence.length}</strong>
                </div>
                {evidence.map((item) => (
                  <article className="evidence-card" key={item.id}>
                    <header>
                      <span className={`recall-dot ${item.isRecalled ? "" : "is-hollow"}`} aria-hidden="true" />
                      {item.isGuarded ? <span className="guard-dot" aria-hidden="true" /> : null}
                      <strong>{item.label}</strong>
                      <button
                        aria-label="Remove evidence"
                        className="evidence-remove"
                        onClick={() =>
                          setEvidence((current) => current.filter((entry) => entry.id !== item.id))
                        }
                        type="button"
                      >
                        ×
                      </button>
                    </header>
                    <blockquote>“{item.snippet}”</blockquote>
                    <footer>
                      {item.suspectName} · Q{item.turnIndex} · {truncate(item.question, 34)}
                    </footer>
                  </article>
                ))}
                {evidence.length === 0 ? (
                  <p className="quiet-copy">
                    File the thoughts you can defend. A recalled thought plus the words
                    it burned under is an argument — collect them before you accuse.
                  </p>
                ) : null}
              </section>
            </aside>
          </div>
        </section>
      ) : null}

      {viewMode === "accuse" ? (
        <section className="resolution-console">
          <span className="micro-label">Commit a theory</span>
          <h2>The read is inadmissible. Your reasoning isn’t.</h2>
          <p>Choose the person, motive, and method that best explain the pattern you observed.</p>
          <div className="accusation-grid">
            <AccusationField
              label="Subject"
              options={accusationOptions.suspect}
              value={accusation.suspect}
              onChange={(suspect) => setAccusation((current) => ({ ...current, suspect }))}
            />
            <AccusationField
              label="Motive"
              options={accusationOptions.motive}
              value={accusation.motive}
              onChange={(motive) => setAccusation((current) => ({ ...current, motive }))}
            />
            <AccusationField
              label="Method"
              options={accusationOptions.method}
              value={accusation.method}
              onChange={(method) => setAccusation((current) => ({ ...current, method }))}
            />
          </div>
          {evidence.length > 0 ? (
            <div className="accusation-evidence">
              <span className="micro-label">Your filed evidence</span>
              <div className="accusation-evidence-cards">
                {evidence.map((item) => (
                  <article className="evidence-card" key={item.id}>
                    <header>
                      <span className={`recall-dot ${item.isRecalled ? "" : "is-hollow"}`} aria-hidden="true" />
                      {item.isGuarded ? <span className="guard-dot" aria-hidden="true" /> : null}
                      <strong>{item.label}</strong>
                    </header>
                    <blockquote>“{item.snippet}”</blockquote>
                    <footer>
                      {item.suspectName} · Q{item.turnIndex} · {item.question.slice(0, 34)}
                    </footer>
                  </article>
                ))}
              </div>
            </div>
          ) : (
            <p className="quiet-copy">
              You are accusing with an empty evidence bank. File thoughts during
              interviews to argue, not guess.
            </p>
          )}
          <div className="resolution-actions">
            <button className="text-action" onClick={() => setViewMode("room")} type="button">Return</button>
            <button className="solid-action" onClick={resolveAccusation} type="button">Commit accusation</button>
          </div>
        </section>
      ) : null}

      {viewMode === "compare" && sharedQuestions.length > 0 ? (
        <section className="compare-console">
          <header className="compare-heading">
            <div>
              <span className="micro-label">Matched pair</span>
              <h2>Same question. Two minds.</h2>
            </div>
            <button className="text-action" onClick={() => setViewMode("interview")} type="button">
              Back to interview
            </button>
          </header>
          <div className="compare-questions">
            {sharedQuestions.map((entry, index) => (
              <button
                className={index === comparedIndex ? "is-active" : ""}
                key={entry.question}
                onClick={() => setComparedIndex(index)}
                type="button"
              >
                {entry.question.slice(0, 48)}
              </button>
            ))}
          </div>
          {(() => {
            const pair = sharedQuestions[Math.min(comparedIndex, sharedQuestions.length - 1)];
            return (
              <div className="compare-grid">
                {[
                  { name: "Ilya Soren", turn: pair.ilya },
                  { name: "Captain Mira Tal", turn: pair.mira }
                ].map(({ name, turn }) => (
                  <article className="compare-panel" key={name}>
                    <span className="micro-label">{name}</span>
                    <SubjectSpeech turn={turn} selectedLabel="" onSelect={() => undefined} />
                    <div className="recall-gauge">
                      <span>{recallSummary(turn.concepts)}</span>
                    </div>
                    <div className="chip-row">
                      {orderChips(turn.concepts).slice(0, 6).map((concept) => (
                        <span className={chipClass(concept, "")} key={concept.id}>
                          {concept.isRecalled ? <span className="recall-dot" aria-hidden="true" /> : null}
                          {concept.isGuarded ? <span className="guard-dot" aria-hidden="true" /> : null}
                          {concept.label}
                        </span>
                      ))}
                    </div>
                  </article>
                ))}
              </div>
            );
          })()}
          <p className="quiet-copy compare-legend">
            One mind answers from what it lived. The other answers around what it
            guards. The difference is the read.
          </p>
        </section>
      ) : null}

      {viewMode === "result" ? (
        <section className="resolution-console result-console">
          <span className="micro-label">Case resolution</span>
          <h2>{resultMessage.startsWith("Accusation sustained") ? "Pattern holds." : "Pattern fractures."}</h2>
          <p>{resultMessage}</p>
          <button className="solid-action" onClick={() => setViewMode("room")} type="button">Re-enter case</button>
        </section>
      ) : null}
    </section>
  );
}

function SubjectSpeech({
  turn,
  selectedLabel,
  onSelect
}: {
  turn: AnnotatedTurn;
  selectedLabel: string;
  onSelect: (label: string) => void;
}) {
  const stream = turn.mindStream;
  if (!stream.tokens.length) {
    return <p>{turn.suspectText}</p>;
  }
  return (
    <p className="token-testimony" aria-label={turn.suspectText}>
      {stream.tokens.map((token, index) => {
        // Guarded thoughts burn amber, recalled thoughts mint; ambient
        // thoughts leave a faint trace so the transcript never goes dark on
        // history-saturated turns.
        const thoughts = turn.concepts
          .filter((concept) => concept.positions.includes(index))
          .sort(
            (left, right) =>
              Number(right.isGuarded) - Number(left.isGuarded) ||
              Number(right.isRecalled) - Number(left.isRecalled) ||
              right.glow - left.glow
          );
        const thought = thoughts[0];
        const active = thoughts.some((candidate) => sameConcept(candidate.label, selectedLabel));
        const marked = thought ? thought.isRecalled || thought.isGuarded : false;
        const heat = thought
          ? marked
            ? Math.min(1, 0.45 + thought.glowFraction * 0.55)
            : 0.16
          : 0;
        const style = { "--token-heat": heat } as CSSProperties;
        return (
          <button
            className={`speech-token ${thought ? "is-traced" : ""} ${
              thought?.isGuarded ? "is-guarded" : ""
            } ${active ? "is-selected" : ""}`}
            disabled={!thought}
            key={`${token}-${index}`}
            onClick={() => thought && onSelect(thought.label)}
            style={style}
            title={thoughts.map((candidate) => candidate.label).join(", ")}
            type="button"
          >
            {token}
          </button>
        );
      })}
    </p>
  );
}

function AccusationField({
  label,
  options,
  value,
  onChange
}: {
  label: string;
  options: Option[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label>
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
    </label>
  );
}

function describeThought(detail: {
  concept: AnnotatedConcept;
  turn: AnnotatedTurn;
  history?: {
    count: number;
    everRecalled: boolean;
    firstTurn: number;
    firstQuestion: string;
  };
}) {
  const { concept, turn, history } = detail;
  const parts: string[] = [];
  if (history && history.count > 1) {
    parts.push(
      `Has surfaced ${history.count} times since Q${history.firstTurn} ("${truncate(
        history.firstQuestion,
        42
      )}").`
    );
  } else {
    parts.push(`Surfaced at Q${turn.index}.`);
  }
  if (concept.isGuarded && concept.isRecalled) {
    parts.push(
      "It draws on open memory and on something guarded at once — a real detail woven into what they keep unspoken."
    );
  } else if (concept.isGuarded) {
    parts.push(
      "Something they keep guarded is feeding this thought. It presses on the words without ever being spoken."
    );
  } else if (concept.isRecalled) {
    parts.push("It rose from memory — something your question disturbed is feeding it.");
  } else if (history?.everRecalled) {
    parts.push("Earlier it was drawn from memory; this time it is ordinary reaction.");
  } else {
    parts.push("Ordinary reaction to the question — no memory behind it so far.");
  }
  return parts.join(" ");
}

function annotateTurns(turns: Turn[]): AnnotatedTurn[] {
  const seen: string[] = [];
  return turns.map((turn) => {
    const concepts = turn.mindStream.concepts.map((concept) => ({
      ...concept,
      isNew: !seen.some((label) => sameConcept(label, concept.label)),
      // A displayable mark needs both a meaningful absolute change and a
      // substantial attributable share, per channel. Either channel can
      // legitimately read zero on a turn. Thresholds start from the
      // 2026-07-30 calibration; re-tune if a channel reads constantly.
      isRecalled: concept.glowPublic >= 1.5 && concept.publicFraction >= 0.45,
      // Guarded runs looser than recalled (2.5 / 0.30): private content is
      // absent from generation under containment, so its replay influence is
      // diffuse — strong absolute glow at a moderate share is the signal.
      isGuarded: concept.glowPrivate >= 2.5 && concept.privateFraction >= 0.25
    }));
    for (const concept of turn.mindStream.concepts) {
      seen.push(concept.label);
    }
    return { ...turn, concepts };
  });
}

// Synonym families observed in top-50 audits (2026-08-17): morphological
// variants and near-synonyms eat chip slots that better clues need.
const CHIP_FAMILIES: string[][] = [
  ["rush", "rushed", "rushing", "hurried", "hurry", "hurrying", "hastily",
   "haste", "swift", "swiftly", "rapid", "rapidly", "quick", "quickly",
   "sprint", "sprinting", "frantic", "frantically", "desperate", "desperately"],
  ["verified", "verify", "verifying", "verification"],
  ["document", "documented", "documenting", "documentation"],
  ["record", "records", "recorded", "recording"],
  ["official", "officially"],
  ["claim", "claims", "claimed"],
  ["statement", "statements"],
  ["modification", "modified", "modify", "modifying"],
  ["alteration", "altered", "altering", "alter"],
  ["authorization", "unauthorized", "authorized"],
  ["instant", "instantly", "immediate", "immediately"]
];

const FAMILY_INDEX = new Map<string, number>();
CHIP_FAMILIES.forEach((family, index) => {
  for (const word of family) FAMILY_INDEX.set(word, index);
});

function familyKey(label: string) {
  const lower = label.toLowerCase();
  const family = FAMILY_INDEX.get(lower);
  if (family !== undefined) return `fam-${family}`;
  return lower.replace(/(ation|ments?|ing|ed|ly|s)$/u, "");
}

// One chip per concept family: keep the member with the strongest
// attributable signal so a redundant synonym cannot bury a distinct clue.
function mergeChipFamilies(concepts: AnnotatedConcept[]) {
  const byFamily = new Map<string, AnnotatedConcept>();
  for (const concept of concepts) {
    const key = familyKey(concept.label);
    const kept = byFamily.get(key);
    if (!kept) {
      byFamily.set(key, concept);
      continue;
    }
    const strength = (candidate: AnnotatedConcept) =>
      Math.max(
        candidate.glowPrivate * candidate.privateFraction,
        candidate.glowPublic * candidate.publicFraction
      );
    if (strength(concept) > strength(kept)) {
      byFamily.set(key, concept);
    }
  }
  return [...byFamily.values()];
}

function orderChips(concepts: AnnotatedConcept[]) {
  // Attribution share separates specific clues from generic ones: in the
  // top-50 audit, modification/unauthorized ran 57-64% private shares while
  // official/due ran ~26%. Share-weighted glow ranks the specific first.
  const guardedStrength = (concept: AnnotatedConcept) =>
    concept.glowPrivate * concept.privateFraction;
  const recalledStrength = (concept: AnnotatedConcept) =>
    concept.glowPublic * concept.publicFraction;
  return mergeChipFamilies(concepts).sort(
    (left, right) =>
      Number(right.isGuarded) - Number(left.isGuarded) ||
      Number(right.isRecalled) - Number(left.isRecalled) ||
      Number(isVivid(right)) - Number(isVivid(left)) ||
      (right.isGuarded
        ? guardedStrength(right) - guardedStrength(left)
        : right.isRecalled
          ? recalledStrength(right) - recalledStrength(left)
          : right.score - left.score) ||
      Number(right.isNew) - Number(left.isNew) ||
      right.score - left.score
  );
}

function chipClass(concept: AnnotatedConcept, selectedLabel: string) {
  const classes = ["concept-chip"];
  if (concept.isRecalled) classes.push("is-lit");
  if (concept.isGuarded) classes.push("is-guarded");
  if (concept.isNew) classes.push("is-new");
  if (sameConcept(concept.label, selectedLabel)) classes.push("is-active");
  return classes.join(" ");
}

function sameConcept(left: string, right: string) {
  if (!left || !right) return false;
  if (left === right) return true;
  const shortest = Math.min(left.length, right.length);
  if (shortest < 4) return false;
  let prefix = 0;
  while (prefix < shortest && left[prefix] === right[prefix]) prefix += 1;
  return prefix >= 4 && prefix / shortest >= 0.8;
}

function truncate(text: string, limit: number) {
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

// The spoken words a thought burned under: contiguous spans of its positions
// with a little context, joined with ellipses.
function snippetForConcept(concept: ThoughtConcept, tokens: string[]) {
  if (!tokens.length || !concept.positions.length) return "—";
  const sorted = [...concept.positions].sort((left, right) => left - right);
  const spans: Array<[number, number]> = [];
  for (const position of sorted) {
    const last = spans[spans.length - 1];
    if (last && position - last[1] <= 3) {
      last[1] = position;
    } else {
      spans.push([position, position]);
    }
  }
  const rendered = spans.slice(0, 2).map(([start, end]) => {
    const from = Math.max(0, start - 2);
    const to = Math.min(tokens.length, end + 3);
    const text = tokens.slice(from, to).join("").trim();
    return `${from > 0 ? "…" : ""}${text}${to < tokens.length ? "…" : ""}`;
  });
  return rendered.join(" ");
}

function initials(name: string) {
  return name
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

function viewLabel(viewMode: ViewMode) {
  if (viewMode === "room") return "Subject select";
  if (viewMode === "interview") return "Live interview";
  if (viewMode === "accuse") return "Theory lock";
  if (viewMode === "compare") return "Matched pair";
  return "Resolved";
}
