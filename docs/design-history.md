# RAG Mind-Stream — Prototype 04 — 2026-07-29

The contrastive guilt-detector architecture (Prototype 03) is retired as the
player-facing read. This build replaces it with memory retrieval and a
thought stream: the player watches a whole mind, and sees which thoughts rose
from what their question caused to surface.

## Why the pivot

Prototype 03's fixed-response contrast measured one thing: what a suspect's
3-sentence private-fact block contributed to the replay. After the noise was
gated out (audits of 2026-07-28), the honest signal turned out to be ~3 bits,
question-independent — every Ilya turn read COVER-UP. A trustworthy
instrument, but a one-note game. The bottleneck was the context diff, not the
lens: the lens reads context transport, so the fix is dynamic context.

## Architecture

1. **Memory corpus** per suspect (`data/memories/*.json`): first-person
   episodic entries with `privacy` and `tags`. Ilya: 36 entries, the crime
   broken into ~12 distinct memories (the 0230 override, the seal sound, the
   ledger rewrite, the deleted simulations, the rehearsed calm…), plus
   routine, sensory, and family texture. Mira: 30 entries — innocent of the
   murder, hiding a contraband-medicine log edit, carrying guilt about not
   reporting Ilya. Memory text NEVER reaches the browser.
2. **Retrieval** (`telepathic_detective/memory_retrieval.py`): embeddings are
   mean-pooled mid-layer hidden states from the same Qwen model (no new
   dependencies), corpus-mean-centered before cosine (raw causal-LM pooling
   is anisotropic — hub memories matched everything), plus a lexical bonus
   over contentful question words (so "your sister" decisively pulls the Mara
   memories). Deterministic; embeddings cached by content hash. Top-k = 4.
3. **Injection**: the route's system prompt carries a `{{MEMORIES}}`
   placeholder; the server substitutes "Memories surfacing in your mind right
   now: …" plus the standing rule never to state or closely paraphrase them.
4. **Generation** once (seeded; third-person-narration retry at seed 43).
5. **Two token-exact replays**: full context, and the same tokens with the
   memory block ablated. Question, history, and character shell are in both,
   so question-driven semantics cancel in the diff automatically — the gate
   stack (min-delta, provoked, repetition) is unnecessary and gone.
6. **The read** (`build_mind_stream`):
   - **Stream**: top ~10 unspoken concepts from the full replay alone —
     ordinary reactive thoughts included, deliberately. Filters: stopwords,
     question echo, spoken-word echo, ≥2-layer support, display dedup of
     inflections.
   - **Glow**: per concept, full-replay score minus ablated-replay score.
     Nothing is gated on glow; it only tints. Client rule: memory-lit when
     glow ≥ 3, or ≥ 1.5 with ≥ 30% share.
7. **UI**: thought-stream cards (ambient vs memory-lit), a "memories stirred"
   panel showing only privacy + tags ("guarded · guilt · dreams"), selected-
   thought detail, and case memory pinning memory-lit thoughts. Verdict
   cards, gate copy, and strength labels are gone.

## What was retired vs kept

Retired from the play path: counterfactual/neutral fact blocks and their
token alignment, min-delta gating, motif families and GUILT/COVER-UP verdict
labels, provoked/repeated strict gates, the saturated-read state. (The code
remains for the legacy contrast endpoint; `scripts/check_prompt_alignment.py`
is obsolete for this path.)

Kept unchanged: token-exact replay with decode-mismatch assertion, echo and
stopword filtering, the 2048-token budget with honest 422s, seeded
determinism, narration retry, roleplay-prefix stripping.

## Live results (seeded, through the real route)

The failure that triggered the pivot — every question reading COVER-UP — is
gone. First-turn streams are question-specific:

- "Tell me about the accident." → ambient *malfunction, disaster, safety,
  fatal, evacuation*; **memory-lit: emergency (16%), responsible (14%)** —
  the guilt-dream memory surfaced and *responsible* burned. Memories stirred:
  accident-history (open), guilt·dreams (guarded), Lena·last-words (guarded).
- "Why did the corridor timing change?" → retrieves the 0230 override, the
  ledger rewrite, and the shadow-log worry; **memory-lit: routine, normal,
  emergency** — his mind reaches for normalcy exactly when the crime memories
  surface.
- "What did Lena's audit threaten to expose?" → retrieves audit-fear and the
  motive arithmetic; lit: *adjustments*.
- "Tell me about your sister." → retrieves the Mara memory; he now answers
  about her (an earlier retrieval miss made him deny having a sister);
  lit: *health, personally, monitoring*.
- Mira, "What are you hiding about the patrol logs?" → retrieves her
  contraband log memories; **memory-lit: trustworthy (46%)** beneath a
  by-the-book answer — the red-herring subplot reads exactly as designed.
- Mira, "Did you kill Lena?" → surfaces her not-reporting-Ilya guilt memory;
  stream shows *murder, truth, trust, unequivocal* as the normal mind of an
  innocent person being accused — thoughts are characterization now, not
  verdicts.

Verification: 104 tests passing (7 new for stream/corpus/retrieval),
TypeScript clean, full browser loop confirmed (stream cards, memory chips,
pinning, Enter-to-submit). Latency ~4–9s per turn (generation + retrieval +
two replays).

## Known limitations / next work

- **Fiction must carry the reframe**: a player can still read *murder* in an
  accused innocent's stream as damning. The brief-room and empty-state copy
  gesture at "thoughts are not verdicts"; a proper tutorial beat should teach
  it.
- Retrieval is serviceable, not sharp: mid-layer causal-LM embeddings with
  centering + lexical bonus. A dedicated small embedder would improve
  surfacing; revisit if retrieval misses become player-visible.
- Occasional token fragments in the stream (`deton`) and testimony
  fact-drift (sector numbers, invented details) remain 4B-model artifacts.
- Per-memory leak attribution (leave-one-out replays), depth-gated retrieval
  (deep memories unlock under evidence pressure), and salience decay
  (surfaced memories wear out) are the designed next mechanics; all fit the
  current architecture without new instrumentation.

## Addendum — Prototype 04b (conversation view, same day)

Interface revision after first playtest of 04:

- **"Memories Stirred" removed entirely.** Retrieval still drives the mind
  server-side, but nothing about which memories surfaced — not even redacted
  tags or the guarded/open flag — is sent to the browser anymore. The chips
  were doing the detective work for free (`corridor · 0230` was nearly a
  literal answer) and leaked authored ground truth. All player information
  now comes through the stream itself.
- **Continuous two-sided chat.** The interview is one scrolling transcript
  (detective right, subject left) with glowing tokens inside the subject's
  bubbles and up to five concept chips beneath each answer. Composer pinned
  below; Enter sends.
- **Novelty tracking.** Each concept is compared (fuzzy-stem match) against
  everything previously seen in that suspect's conversation. First
  appearances get a NEW badge; the side dashboard's "latest read" lists the
  current turn's thoughts with new/recurring and ambient/memory-lit status,
  and a cumulative **concept board** shows every concept with occurrence
  count and last-seen turn (mint = was ever memory-lit). This is the
  cross-turn contrast: what each answer *added* to the mind, now that
  retrieval makes context vary per question.
- Client-side only change plus dropping `surfacedMemories` from the route
  payload; the lens pipeline is untouched. Verified in the browser: chips,
  badges, board, auto-scroll, and the recurring-vs-new distinction (Q2 showed
  `malfunction` recurring and newly memory-lit while four fresh concepts
  carried NEW badges).

## Addendum — Prototype 04c (recall language, same day)

First-principles UI pass now that the mechanic is stable. Principle: show the
player the two things the instrument actually measures well, in plain
language, and nothing else.

- **"Glow" is gone from the interface; the concept is now "recall."** A
  thought drawn from surfaced memory is marked **recalled** (mint dot);
  unmarked thoughts are ordinary reaction and get no label at all. All raw
  numbers (percentages, scores, layers, ranks) removed from the interface.
- **Answer-level recall gauge** under every reply: *deep recall / traces of
  recall / surface only* — the alibi tell as a displayed stat. Metric is
  total glow ÷ total score per turn, calibrated on 32 archived live runs
  (Mira's true alibi 0.216, Ilya's invented one 0.021; thresholds 0.10 /
  0.04). Note the sharp deep-vs-surface split shows best mid-conversation;
  cold-open alibis often both read "traces."
- **Side panel collapsed from three overlapping sections to a clean stack:**
  selected thought (with a plain-English sentence built from its history:
  when it first surfaced, how often it returns, whether memory fed it), this
  answer's stream, and **Preoccupations** (thoughts that recur or were ever
  recalled — replaces the concept board). A one-line legend anchors the
  reading: "Recalled thoughts are drawn from memory. A story with no recall
  behind it is running on invention alone."
- Brief-room copy now teaches both core reads: "Thoughts are not verdicts.
  Memory does not lie. Stories can."
- Two-color glow (memory vs fabrication ablation) was considered and dropped:
  a recited cover story is spoken text, and the echo filter removes spoken
  words — the fabrication channel would mostly cancel.

## Addendum — Prototype 04d (evidence bank, same day)

Two corrections from playtesting 04c:

- **Recall is per-activation, and the interface now says so.** The
  deep/traces/surface verdict wrongly stamped a single truth-status on a
  whole answer when the measurement is per-thought. Replaced with a literal
  count under each reply — "4 of 10 thoughts drawn from memory" / "nothing
  drawn from memory" — which preserves the alibi tell without inventing an
  aggregate judgment.
- **The side panel is now an evidence bank.** The player examines any thought
  (chips or glowing words), sees its recall status, its cross-interview
  history in plain English, and the exact spoken snippet it burned under —
  and can file that unit (activation + snippet + provenance: suspect, turn,
  question) as an evidence card. Cards are player-curated, removable, and
  persist across suspects. Auto-computed panels (this answer, preoccupations)
  were removed; the per-answer chips (now 8) carry the stream inline.

The natural next step is surfacing the filed evidence on the accusation
screen, so committing a theory means laying your cards down.

## Addendum — Prototype 05a (containment, 2026-08-01)

Response to Codex's review of 05, which caught private memories leaking
verbatim into testimony (Mira recounted her records-office sighting aloud —
the 4B model cannot reliably hold "keep unspoken").

- **Containment**: testimony is now generated from the private-ablated
  condition. The model sees that guarded recollections exist (label lines
  remain) but never their content, so private material cannot leak into
  speech by construction. Private influence exists only in the telepathy
  channels via the full-context replay. Epistemic note: the guarded channel
  is now a counterfactual read (what the secret does to the mind processing
  these words) rather than a causal one (what it fed into generation) — an
  accepted trade for hard containment on a small model.
- **Per-memory relevance gate**: channel slots are limits, not quotas — each
  entry must individually clear the semantic-or-lexical bar before slotting
  (Codex caught slots back-filling with irrelevant entries once the global
  gate passed).
- **Guarded thresholds recalibrated** for post-containment distributions:
  glow_private ≥ 2.5 and share ≥ 0.25, chosen from a threshold sweep that
  admits the true cases (knowingly/trusted on Mira's log question, events on
  the Ilya-sighting question) and nothing below them. Pronoun/filler
  stopwords added (she/her/him/his/including).

Confirmed live: Mira asked directly what she saw Ilya doing answers "I have
no record… nor did I witness any unusual activity" while **events** and
**related** burn guarded beneath the denial — the red-herring branch now
requires telepathy instead of being handed out in dialogue. Her log question
produces **knowingly [guarded]** and **trusted [recalled+guarded]**, the
first live woven mark. No private phrases appear in any probed testimony.
115 tests passing.

## Addendum — Echo admission and the recency experiment (2026-08-01)

**Echo admission (kept).** The question/spoken-word echo filter no longer
applies to the glow channels. Both replay conditions contain the question and
the response, so channel deltas structurally cannot be caused by echo — an
echoed word that glows is the most informative evidence available. Echo words
now enter the stream when they clear a mark threshold; glowless echoes stay
filtered from the ambient layer. Auxiliary/pronoun stopwords added
(has/had/doing/she/him/including).

**Wide-stream diagnostic (informative null).** Widening the stream to 30
concepts on the records-office question showed the informative content words
(*office*, *records*, *watched*, *log*) are **absent from the response-position
candidate pool entirely** — the closest is *outside* at glow 1.5 / 9% share.
No reranking or semantic-significance filter can surface words that are not
there. At response positions the lens measures the denial's own processing,
tilted by the memory; it detects a guarded presence with a vague topical
gradient, not the secret's content. This is the readout's ceiling, and it is
architectural.

**Recency experiment (tested, reverted).** Hypothesis: the recollection block
sits in the system prompt, 300–600 tokens upstream of generation, so recency
decay weakens its influence — moving it adjacent to the generation point
should strengthen the guarded channel. Implemented (carrier on a trailing
user turn, since Qwen's template rejects a trailing system message; also
fixed `player_text` extraction, which had been picking up the carrier).
Result across four matched questions: guarded marks **fell from 8 to 3**
(m-logs 2→0, i-timing 2→0, m-leak 2→1, i-ledgers 2→2). Plausible mechanism:
with recollections last, the answer leans more on the *public* memories,
which are present in both full and private-ablated conditions, shrinking the
private delta. Reverted; the `player_text` robustness fix and the
placeholder-drop were kept. A `stream_top_concepts` debug knob remains on the
backend payload for future diagnostics.

## Addendum — Injection-format experiments (2026-08-01, later)

Hypothesis (user): the attention mechanism may not be "noticing" recollections
in the system prompt; a more salient format — generation-adjacent placement,
or a `recall` tool-response trace (the format instruct models are trained to
integrate) — should strengthen response-position activations.

Both were built and A/B'd against the system-prompt baseline (8 guarded marks
across the matched-question set, labels knowingly/trusted/normally/routinely/
official/events):

1. **Trailing user-turn carrier**: 8 → 3 guarded marks. (Also surfaced and
   fixed a real bug: `player_text` extraction picked up the carrier, so
   retrieval was querying with the literal placeholder.)
2. **`recall` tool trace** (assistant tool_call + tool_response, private
   entries as sealed stubs at generation): 8 → 5 guarded marks, worse labels
   (except/actually/good), mild register leakage ("My memory only recalls…").
   Containment and the recalled channel held.

Both reverted. Conclusion: for this model and readout, **system-prompt
placement maximizes private-memory influence on response-position
activations** — the "unnoticed" hypothesis is falsified; distant context
integrated as global framing beats salient late-context formats, which
appear to make generation anchor on the immediately-present (public/stub)
material and shrink the private replay delta. Kept from the experiments:
`player_text` carrier-robustness, tool_calls passthrough in
`render_chat_input_ids`, and the `stream_top_concepts` debug knob.

## Addendum — System-prompt activation audit (2026-08-01, later still)

Three shell alterations A/B'd against the 5-question baseline (6 guarded
marks) for guarded-channel quality:

- **A. De-duplicated suppression language (adopted).** The privacy rule
  appeared in both the character shell and the per-turn scaffold, plus
  "KEEP UNSPOKEN" on every private label — saturating concealment vocabulary
  symmetrically across all replay conditions and compressing the guarded
  channel's rank-space headroom. Slimming the scaffold to bare
  [PUBLIC]/[PRIVATE] labels with the rule stated once: **6 → 13 marks**, and
  the first content-word labels appeared (*records* on the records-office
  question, *resources* on the ledgers question).
- **B. Vagueness-when-unbacked contract (adopted).** "Answer from your
  recollections when they cover the question; where they do not, keep your
  answer brief and unspecific rather than inventing detail." Mark count
  neutral (13), but labels gained content (*unauthorized*, *allocations*,
  *unofficial*), ungrounded testimony now hedges visibly ("I do not have a
  specific recollection…") — a second, textual tell aligned with the recall
  channel — and both alibi reads survived.
- **C. Memories at the top of the system prompt (rejected).** 13 → 6 marks,
  logs and timing reads lost. Combined with the cross-message experiments,
  the placement picture is now mapped: end-of-system-prompt is the optimum;
  both earlier (top of system) and later (trailing turns, tool traces) are
  measurably worse.

Locked configuration confirmed: logs → knowingly/unauthorized/trusted,
ledgers → allocations/adjustments/unofficial. 116 tests passing.

## Addendum — Codex review response (2026-08-01, end of day)

Positions and actions on Codex's four remaining issues:

1. **Habit→alibi fabrication (fixed).** Memories now carry a `kind`
   ("episode"/"habit"); 20 routine entries tagged across both corpora, labeled
   [PUBLIC HABIT] in the scaffold, with a shell rule: a habit describes what
   you usually do, never a specific memory of the incident window. Canon
   gained "cycle numbers" in the never-invent list and an "all crew were
   aboard" line (after a stray "I was away from the station" improvisation).
   Verified: Mira's sunrise answer now presents her habit as a habit with
   zero invented detail; the canteen-habit-as-specific-alibi conversion is
   gone.
2. **Accusations read flat (kept, as design).** Diagnosed before deciding:
   retrieval is not the bottleneck (the crime memory lexically matches
   accusation vocabulary); the denial's processing simply does not move — a
   prepared surface deflects. Codified rather than fought: the tutorial now
   teaches "blunt accusations bounce off a prepared surface — precision is
   what opens the seams." Supporting evidence, same session: the precision
   question "Would your logs survive a line-by-line check against the door
   sensors?" — which touches exactly what Mira's private memory fears —
   produced validated 7.6 / truthful 5.3 / submitted 4.0 guarded.
3. **Tutorial (fixed).** Step 02 now explains mint (drawn from open memory),
   amber (pressed by something guarded), unmarked (ordinary reaction), and
   that none is proof.
4. **Blank [PRIVATE] labels at generation (declined, with rationale).**
   Generating from a fully clean public-only prompt would break the property
   that the generation context equals the private-ablated replay exactly —
   which is what keeps the public channel a *causal* measurement. The blank
   labels are also diegetic (the character feels something guarded exists).
   Revisit only if the hedging register becomes grating.

## Addendum — Retrieval alias pass (2026-08-01, night)

Live session analysis showed the quality bottleneck has moved to retrieval
phrase-sensitivity: questions using the corpus's concrete nouns (logs,
sensors, corridor) retrieve; conceptual paraphrases (tampering, unusual,
convenient timing) missed, producing flat reads on exactly the questions a
good player asks — and in one case a retrieval miss caused an innocent
suspect to invent self-incriminating placement ("on patrol in Sector C")
because no memory covered her alibi phrasing.

Fix: alias tags on high-value memories (tags are pure retrieval hooks,
invisible to players — a retrieval thesaurus). Verified against the exact
missed phrasings: "notice anything unusual" now stirs her withheld-warning
memory (unaware 3.3 guarded beneath "I did not notice"); "records tampering"
now stirs the log secret (knowingly/aware); "where were you when the accident
happened" now retrieves checkpoint seven and she answers truthfully from
memory instead of inventing. The audit-insinuation phrasing stays flat —
consistent with the accusation-flatness pattern the tutorial now teaches.
Structural fix for paraphrase robustness (a proper small embedder) remains
the known next step if aliasing proves too whack-a-mole.

## Addendum — Dedicated retrieval embedder (2026-08-01, late)

Replaced mean-pooled suspect-model hidden states with `BAAI/bge-small-en-v1.5`
(plain transformers AutoModel, CLS pooling, bge query prefix — no new pip
dependencies; ~130MB weights loaded once at server start; falls back to the
old pooling if `--embedder` is empty). Anisotropy centering skipped for the
dedicated embedder (cosine-calibrated).

Gate recalibrated empirically before wiring: on the real hit/miss set,
positives scored 0.47–0.65 and negatives 0.25–0.31 — a 0.40 gate separates
with real margin, and every conceptual paraphrase the old embedder missed
("records tampering" 0.47, audit insinuation 0.55, "anything unusual" 0.54)
scores as a clear positive.

Live verification: all previously-missed phrasings now stir or answer from
memory; the locked regression reads held (logs integrity/trusted); the
corridor-timing question produced its strongest read ever (five guarded
marks: public 7.9, official 7.0, unauthorized 6.5, specific 7.4); Mira's
alibi answers from checkpoint seven in every phrasing tested; nonsense and
childhood nulls are fully quiet. The audit-insinuation remains guarded-flat,
consistent with the accusation-flatness design. Alias tags retained as
hybrid lexical support. Paraphrase robustness is now a property of the
system rather than an authoring chore.

## Addendum — Consensus punch list: composition, compare, evidence (2026-08-01)

Implemented the agreed ordering (evidence cards → retrieval composition →
compare verb → pale words; motive path still deferred behind verification).

**Three-tier retrieval composition** (`build_retrieval_query`): the current
question embeds alone whenever it is contentful (≥2 non-stopword tokens);
content-light follow-ups ("what about ilya?", "and then?") append the
previous player question; capped capitalized-entity fragments from the
previous answer are a last resort only when the combined referent is still
empty. Carried words take a 0.6 lexical discount so they steer, never
dominate. Incident-anchored questions (alarm/night/0300/exactly/…) softly
sort habit memories behind episodes in the public channel
(`prefer_episodes`), attacking the habit-becomes-alibi failure. Five unit
tests codify the tiers (121 total passing).

**Verification against Codex's failure sequences** (live, seed 42):
- Audit talk → "Where were you when the pressure failed?" no longer
  retrieves the intake-valve habit as an incident alibi (the exact reported
  failure). The answer is an unbacked console story with zero recalled
  marks — correctly reading as invention.
- Valve-heavy talk → private-slate question now lands its guarded read
  (actual 7.0, publicly 4.7, outside 3.4) where it previously missed.
- Referential follow-ups still resolve: "What about Ilya?" → "Did you see
  him that day?" retrieves her records-office sighting and marks
  **standing** guarded 4.3 — the memory of Ilya standing outside the
  records office pressing on her denial. Best multi-turn read to date.
- The door-sensor-after-checkpoint sequence read flat this sample; since
  composition now embeds that contentful question bare (identical to the
  fresh-interview case that reads guarded), the flatness is answer-sampling
  variance, not retrieval steering. Accepted.

**Matched-pair opening + compare verb**: both sample shelves lead with
"Where exactly were you when the breach alarm sounded?"; once both suspects
answer a normalized-identical question, a "Compare testimony" view unlocks
(question chips, side-by-side panels, per-panel channel footers, legend).
The opening pair verified in-browser as the intended pedagogy: Mira reads
RECALLED (distance/rushing/rushed — reliving the run), Ilya reads GUARDED
(instantly/ready/standing — the pump-room wait pressing on his timing
words while he speaks a valve-maintenance alibi). Same question, opposite
channels, on turn one.

**Evidence cards at accusation**: filed thoughts render as cards on the
theory-lock screen (concept, channel dot, burned-under snippet, suspect/Qn
attribution); accusing with an empty bank shows a warning instead. Bank
persists across suspect switches.

**Pale-word preference**: highlight auto-selection and chip ordering now
prefer vivid words over structural/pale ones (except/however/official/
normally/…) at equal glow, so the headline read is "INSTANTLY", not
"OFFICIALLY".

Also fixed: compare view's header mode label ("Matched pair", previously
fell through to "Resolved").

Next: motive-path authoring (public pressure + guarded connection), second
case, accusation evidence grading.

## Addendum — Character scene: lo-fi 3D interview room (2026-08-13)

Replaced the poly-3d branch's abstract wireframe mannequins with actual
character assets, procedurally built in Three.js on `experiment/poly-3d`
(`app/ui/character-scene.tsx`, exported as PolyScene so the client swap was
one import line).

**Art direction**: PS1-era busts — box-and-wedge facial geometry (cranium,
tapered jaw, nose wedge, ears, hair caps), flat shading, and the whole scene
rendered at 232px internal height then stretched with `image-rendering:
pixelated`. The low-res upscale is what sells it: assembled-primitive heads
read as intentional retro characters, not bad 3D. Ilya: gray buzz, stubble
shadow, olive engineer jacket, orange tool harness. Mira: dark hair, navy
security uniform, shoulder pauldron, badge.

**The room**: enclosed interview chamber — plated hull walls (one rusted
panel), viewport slit with drifting starfield, hanging cone lamp with
flicker over a brushed-metal table, the J-Lens recorder as a mint-glowing
puck, floor grating, wall conduits, amber hazard strips.

**Expressions are wired to the lens**: each rig has posable brows, sliding
eyelids, gaze-offset irises, and a mouth plane, interpolating between three
poses driven by the latest turn's channel signal — neutral / recalled (eyes
drift up-away, face opens) / guarded (brows down, lids narrowed, mouth
tight, pulls back). While testimony generates, the subject first breaks
gaze to think, then the mouth flaps. Blinks and idle head/breath motion run
continuously. Room lighting follows the same signal: amber strips pulse on
guarded, the recorder flares mint on recalled.

**Camera direction per mode**: subject-select is a two-shot across the
table; interview swings to a three-quarter close-up framed into the left
third (the chat console owns center screen); the inactive suspect fades
out. Compare pulls back to the two-shot; accusation goes high and wide. A
foreground detective shoulder anchors interview shots.

**Engineering notes hard-won this session**:
- Two `next dev` instances were sharing one `.next` (3000 from Aug 4 + 3002)
  — 3002 hung outright. Killed both, cleared `.next`, restarted 3002 only.
  A dev server serves the checked-out branch, so "main on 3000" stops being
  true the moment the working tree switches branches.
- Occluded tabs stop firing requestAnimationFrame entirely: the scene froze
  at its last composited frame, which poisoned screenshot-based iteration
  and made camera changes look ignored. Fixed properly with a 300ms
  watchdog ticker (also keeps the game alive in background tabs) and by
  converting every per-frame lerp constant to time-based damping
  (`1 - exp(-rate*dt)`), which also fixes 60Hz/120Hz behavior divergence.
- Fast Refresh does not re-run the mount-once scene effect — every scene
  edit needs a hard reload to actually appear. Several "fixes" were judged
  against stale closures before this was caught; a dev-only
  `window.__charScene` telemetry object now exposes camera/mode/opacity for
  direct verification.
- `.poly-scene` (fixed, inset 0) needed `pointer-events: none` — it was
  intercepting suspect-card clicks.

Verified live end-to-end: guarded read on Ilya's alibi question → brows
down/narrowed while amber marks land in the transcript; recalled read on
Mira's checkpoint-seven answer → open face with mint marks; thinking gaze
during capture; talk flap during generation. The two-channel mechanic now
has a physical performance attached to it.

## Addendum — Painted faces: CanvasTexture upgrade (2026-08-13, later)

Replaced the geometric facial features (brow/lid/iris/mouth meshes) with the
authentic PS1 technique: a 64×80 pixel-art face painted onto a per-character
`CanvasTexture`, mapped to a single plane over the head geometry
(NearestFilter, emissiveMap-matched to the body materials so lighting
stays consistent). The 3D nose mesh was retired — the painted nose (bridge
shadow, tip highlight, nostrils) reads better.

The face is drawn parametrically, not from fixed sprites: `draw(params)`
takes quantized expression parameters (brow slope/lift, lid rows, gaze
offset, mouth width/open/tilt, glabella crease) and redraws only when the
quantized values change, so blinks, gaze drift, and the talk flap stay
continuous while costing a handful of 64×80 canvas redraws per second.
Character detail lives in the texture now: Ilya gets forehead creases,
under-eye bags, and deterministic stubble stipple; Mira gets a lash line,
heavier brows, and a scar over the right brow.

Payoff verified live: the guarded read on Ilya's alibi now produces knitted
brows with a visible glabella crease, narrowed eyes, and a tightened mouth
— dramatically more legible than the geometric version at interview
distance. Expressiveness per unit effort is the entire argument for
painted-texture faces over assembled geometry, and it held.

## Addendum — PSX render pipeline research + dither pass (2026-08-13, later)

Researched how PSX-style indie games and three.js retro projects actually
achieve the look. Consensus pipeline: (1) low internal resolution with
nearest upscale — already had it; (2) flat/vertex lighting — already had
it; (3) Bayer-dithered color quantization emulating the PS1's 15-bit
framebuffer — missing, and identified as the element that unifies
mixed-fidelity scenes into one deliberate image; (4) vertex snapping /
affine texture warp — deliberately skipped: they add motion artifacts, and
the prior session's feedback was already "too much animation."

Implemented (3) with three's own postprocessing stack (no new deps):
EffectComposer → RenderPass → OutputPass → custom Bayer-4x4 ordered-dither
+ 31-level-per-channel quantization ShaderPass, composited at the internal
232px buffer so the dither pixels are chunky and authentic. Pitfall worth
recording: EffectComposer bypasses the renderer's tone mapping and sRGB
output — without an OutputPass before the dither stage the scene renders
linear (dark, oversaturated faces). Deepened fog (4.5→11) for atmosphere.
Side benefit: dither grain on the flat uniform colors reads as fabric
texture.

Asset research verdict: best CC0 option is elbolilloduro's "Characters
PSX" pack (79 rigged models, CC0, name-your-price) but it ships FBX/blend
only, A-posed, with textured faces and no facial rigs — adopting it means a
conversion pipeline, seated re-posing via bones, and losing per-frame
facial expression control. Rejected for now; the painted-face system keeps
expression control, which is the game's core payoff. Revisit only if the
character silhouettes themselves become the blocker.

## Addendum — Console reskin: classic-Xbox treatment (2026-08-13, later)

User called the UI "obvious AI-y" and pointed at classic Xbox. Accurate
diagnosis — the old skin was the canonical AI-dark-terminal cluster
(near-black, lone mint accent, Inter, mono microlabels everywhere).

Reskin, applied as an override token layer at the end of globals.css plus
next/font wiring in layout.tsx:
- Palette: acid green (#a3d629, hot #c8f453, deep #5c7f14) on top-lit
  metal slabs (gradient + inset top bevel + dark under-edge + faint green
  rim glow) over a green-black void with fog wisps instead of the fine
  terminal grid. The --mint token was redefined to acid so the recall
  channel recolored in one move; guarded amber untouched (semantic color,
  not accent). The 3D scene's mint constant matched to the same acid.
- Type: Michroma (Eurostile-Bold-Extended lookalike — the classic Xbox
  face) for the wordmark, headings, nav verbs, and button labels; Barlow
  replaces Inter for body; mono stays only on data readouts, which suits
  the diegetic instrument.
- Materials: convex acid buttons with dark ink text and gloss bevel
  (ACCUSE/ASK), suspect cards as glowing-rim dashboard tiles with Michroma
  index numerals, the status dot upgraded to a pulsing orb
  (reduced-motion-guarded), chips as dark metal keys.

Pitfall recorded: .selection-room and .interview-console are full-viewport
layout wrappers, not panels — giving them opaque slab backgrounds buried
the 3D scene until they were pulled out of the slab group.

## Addendum — Welcome experience + reskin spacing repair (2026-08-13, later)

Split Codex's first-play briefing into two surfaces: a first-run **welcome
screen** (title moment: Michroma wordmark with outlined DETECTIVE, one-line
premise, three-dot channel legend, single BEGIN THE INTERROGATION button,
game blurred behind) and the full three-card explainer, unchanged, now the
**field manual** behind "How this works" (relabeled; footer reads "Back to
the case"). First-run flag drives the welcome; the manual opens on demand.

Playthrough-driven spacing repair at a 1184×688 window:
- Interview vertical fit: chat log height re-budgeted, empty-state and nav
  compacted, chip shelf capped at two questions under a max-height 780px
  media query — composer and ASK no longer clip off-screen.
- Camera aspect compensation (wideComp, up to 1.5× pullback) so the
  two-shot clears the DOM brief panel at narrow aspects; suspects land
  above their own tiles again.
- Opaque titlebar (the 3D lamp was bleeding through as a floating blob).
- Font-war settlement: class-scoped Arial Narrow headings (case brief,
  briefing, resolution) now take Michroma at reduced clamps; "How this
  works" styled as a console key.

## Addendum — Ilya's missing public alibi (2026-08-17)

Playtest found Ilya placing himself in Sector C at 0300 on the opening
whereabouts question and doubling down when challenged. Root cause: both
his cover story (console diagnostic, m26) and the truth (pump room, m31)
are PRIVATE, so the public channel had nothing to offer a whereabouts
question except the Sector C valve habit — which the model laundered into
an incident-night alibi at the crime scene. Mira never failed this way
because she has a public whereabouts episode (checkpoint seven, m02); Ilya
simply lacked the equivalent. The habit-vs-episode retrieval preference
can only work when an episode exists.

Fix: authored ilya-m37 — his rehearsed official statement as a PUBLIC
episode ("statement to the investigation is on record: console in the
engineering bay, corridor diagnostic, session log confirms it"), tagged
with incident-anchor aliases. Verified on the exact failing sequence:
Q1 now yields the console alibi; the "Wait, you were at Sector C?"
challenge yields a consistent denial plus restatement.

Tradeoff noted honestly: the guarded channel on his whereabouts answer is
subtler now (official 2.7 vs the old instantly/standing ~7 press) because
the spoken text tracks the rehearsed alibi instead of hovering near the
pump-room wording. That matches the design's matched-contrast thesis —
Mira's truth reads mint-vivid, Ilya's lie reads rehearsed-flat with a
faint guard — but it does move the tell from "loud amber" to "contrast
between the pair."

## Addendum — Chip quality pass, ported from the thinking experiment (2026-08-18)

The thinking-suspect experiment (archived on experiment/thinking-suspect;
CoT is never shown and the mode will not ship) produced a top-50 concept
audit whose quality findings apply to the normal game. Ported here:

- TRACE_STOPWORDS extended with recurring discourse glue (according,
  saying, actually, relevant, nonetheless, ...).
- Chips rank by attribution share (glow × fraction) within each channel:
  the audit showed specific clues (modification 63%, unauthorized 64%
  private share) losing to generic high-scorers (official ~26%) under raw
  score ordering.
- Chip family merge (curated synonym groups + stem fallback): Mira's
  truthful whereabouts was spending 11 of 20 slots on one urgency family.
- PALE_WORDS additions: within, every, due, entire, here, throughout,
  subsequent, personally.

Verified on the corridor-timing question before porting: guarded chips
lead modification/publicly/verified/official, auto-select lands on
MODIFICATION [GUARDED] over his denial of tampering.

## Addendum — Codex playthrough fixes: spotlight alignment + verdict cinematic (2026-08-18)

Codex's poly-3d playthrough landed two priorities; both shipped:

1. Auto-selection now uses the exact ordering the chip row displays
   (orderChips: family merge + share-weighted ranking + vivid preference),
   so the spotlighted thought is by construction the leading chip — the
   "official instead of location / automated instead of approval" class of
   mismatch can no longer happen within a channel.
2. Verdict cinematic: on the result screen the camera slow-dollies onto
   the accused (non-accused fades out), the room drops dark except a hard
   key light, amber strips flood on a correct accusation, and the accused
   plays a "caught" pose (head bowed, eyes down, tight mouth). A wrong
   accusation gets a cold dim room and an unmoved suspect. The result
   console became a left-side gradient so the close-up owns the right of
   frame. Verified live: correct-accusation flow shows the dolly, the
   spotlight, and the bowed head beside "Pattern holds."

Ops note: consolidated back to ONE dev server (Codex's on 3004; the 3002
instance was killed) — two next dev processes sharing .next is the known
corruption trap.
