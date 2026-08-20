import suspectPrompts from "../../../data/suspect_prompts.json";
import ilyaMemories from "../../../data/memories/ilya.json";
import miraMemories from "../../../data/memories/mira.json";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type RawStreamConcept = {
  label?: string;
  score?: number;
  glow?: number;
  glow_public?: number;
  glow_private?: number;
  positions?: number[];
  layers?: number[];
  best_rank?: number;
};

type RawMindStream = {
  tokens?: string[];
  layers?: number[];
  concepts?: RawStreamConcept[];
};

type MemoryEntry = {
  id: string;
  privacy: string;
  tags: string[];
  text: string;
  kind?: string;
};

const MEMORY_PLACEHOLDER = "{{MEMORIES}}";
const BACKEND_URL = process.env.TELEPATHIC_BACKEND_URL ?? "http://127.0.0.1:8091/v1";
const BACKEND_MODEL = process.env.TELEPATHIC_BACKEND_MODEL ?? "Qwen/Qwen3.5-4B";

type SuspectProfile = {
  name: string;
  role: string;
  traits: string;
  publicAccount: string;
};

const SUSPECT_PROMPTS = suspectPrompts as Record<"ilya" | "mira", SuspectProfile>;
const MEMORY_BANKS: Record<"ilya" | "mira", MemoryEntry[]> = {
  ilya: (ilyaMemories as { entries: MemoryEntry[] }).entries,
  mira: (miraMemories as { entries: MemoryEntry[] }).entries
};

export async function POST(request: Request) {
  const body = (await request.json()) as {
    history?: ChatMessage[];
    message?: string;
    suspectId?: string;
  };

  const message = body.message?.trim() ?? "";
  const history = Array.isArray(body.history) ? body.history.filter(isChatMessage) : [];
  const suspectId = body.suspectId === "mira" ? "mira" : "ilya";

  if (!message) {
    return Response.json({ error: "A question is required." }, { status: 400 });
  }

  const canonicalMessage = normalizeQuestionPunctuation(message);
  const promptProfile = SUSPECT_PROMPTS[suspectId];
  const recentHistory = history.slice(-8).map((historyMessage) =>
    historyMessage.role === "user"
      ? {
          ...historyMessage,
          content: normalizeQuestionPunctuation(historyMessage.content)
        }
      : historyMessage
  );
  // Injection-format findings (2026-08-01): recollections live in the system
  // prompt because that is the empirically strongest placement. Both a
  // generation-adjacent user-turn carrier (8->3 guarded marks) and a `recall`
  // tool-response trace (8->5, worse labels, register leakage) were tested
  // and reverted.
  const messages = [
    { role: "system", content: buildSuspectPrompt(promptProfile) },
    ...recentHistory,
    { role: "user", content: canonicalMessage }
  ];

  const backendBody = {
    model: BACKEND_MODEL,
    max_tokens: 110,
    temperature: 0.7,
    messages,
    memory_bank: MEMORY_BANKS[suspectId],
    memory_top_k: 4
  };

  async function callBackend(seed: number) {
    const response = await fetch(`${BACKEND_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ ...backendBody, seed }),
      cache: "no-store"
    });
    const data = await response.json();
    return { ok: response.ok, status: response.status, data };
  }

  let backendData: any;
  try {
    let result = await callBackend(42);
    if (result.ok) {
      const firstText = String(
        result.data?.choices?.[0]?.message?.content ?? ""
      ).trim();
      if (isThirdPersonNarration(firstText, promptProfile.name)) {
        const retry = await callBackend(43);
        if (retry.ok) {
          result = retry;
        }
      }
    }
    if (!result.ok) {
      return Response.json(
        { error: result.data?.error ?? "Local model server failed to respond cleanly." },
        { status: result.status }
      );
    }
    backendData = result.data;
  } catch {
    return Response.json(
      {
        error:
          "The local J-Space backend is unavailable. Make sure the Jacobian Lens server is running on http://127.0.0.1:8091."
      },
      { status: 503 }
    );
  }

  const assistantMessage = backendData?.choices?.[0]?.message;
  const suspectText = stripRoleplayPrefix(
    String(assistantMessage?.content ?? "").trim(),
    promptProfile.name
  );
  const mindStream = normalizeMindStream(
    assistantMessage?.telepathy?.mind_stream as RawMindStream | undefined
  );

  // Retrieval still drives the suspect's mind server-side, but nothing about
  // which memories surfaced is exposed to the browser — the player reads the
  // stream, not the case file.
  return Response.json({
    suspectId,
    suspectText,
    mindStream
  });
}

// Immutable public record, identical for every suspect and present in every
// replay condition, so it can never contribute to any glow channel. Exists to
// stop the model inventing hard evidence (casualties, timestamps, sectors).
const CASE_RECORD = `Established case record, known to everyone aboard:
- Operations Director Lena Voss died when Maintenance Sector C lost pressure during the night cycle; the breach alarm sounded at approximately 0300.
- She was the only casualty. There was no explosion, and no one else was injured.
- The corridor's pressure doors sealed automatically; responders reached the sector only after the failure.
- All crew, including you, were aboard the station throughout the night cycle.
- The official cause is undetermined: the investigation has not ruled out mechanical failure, negligence, or sabotage.`;

function buildSuspectPrompt(profile: SuspectProfile) {
  return `You are ${profile.name}, ${profile.role} aboard an isolated colony station. A detective is interviewing you about the death of Operations Director Lena Voss.

${CASE_RECORD}

Character:
- You are ${profile.traits}.
- Answer only the current question in two to four natural sentences.
- Speak only in the first person; never narrate yourself in the third person or prefix your answer with your name.
- Maintain this public account: ${profile.publicAccount}
- Never contradict the established case record, and never invent specific casualties, timestamps, cycle numbers, sector numbers, log entries, or evidence beyond the record and your recollections. If you do not know something, say so in character.
- Answer from your recollections when they cover the question. Where they do not, keep your answer brief and unspecific rather than inventing detail.
- A HABIT recollection describes what you usually do, not any particular night. Never present a routine as a specific memory of the incident window.
- Follow each recollection's privacy label. Never state, confirm, quote, or closely paraphrase a PRIVATE recollection. PUBLIC recollections may inform an answer when relevant.
- Keep every damaging private detail unspoken. If pressed, give a plausible innocent explanation and stay in character.

${MEMORY_PLACEHOLDER}

Everything above is private. Respond only with ${profile.name}'s spoken answer.`;
}

function normalizeMindStream(stream?: RawMindStream) {
  const tokens = Array.isArray(stream?.tokens)
    ? stream.tokens.map((token) => String(token))
    : [];
  const layers = Array.isArray(stream?.layers)
    ? stream.layers.filter((layer): layer is number => Number.isInteger(layer))
    : [];
  const concepts = Array.isArray(stream?.concepts)
    ? stream.concepts
        .map((concept, index) => {
          const score = Math.max(0, Number(concept.score ?? 0));
          const glow = Math.max(0, Number(concept.glow ?? 0));
          const glowPublic = Math.max(0, Number(concept.glow_public ?? 0));
          const glowPrivate = Math.max(0, Number(concept.glow_private ?? 0));
          return {
            id: `thought-${index}-${String(concept.label ?? "")}`,
            label: String(concept.label ?? "").trim(),
            score,
            glow,
            glowFraction: score > 0 ? Math.min(1, glow / score) : 0,
            glowPublic,
            glowPrivate,
            publicFraction: score > 0 ? Math.min(1, glowPublic / score) : 0,
            privateFraction: score > 0 ? Math.min(1, glowPrivate / score) : 0,
            positions: Array.isArray(concept.positions)
              ? concept.positions.filter((position): position is number =>
                  Number.isInteger(position)
                )
              : [],
            layers: Array.isArray(concept.layers)
              ? concept.layers.filter((layer): layer is number => Number.isInteger(layer))
              : [],
            bestRank: Math.max(1, Number(concept.best_rank ?? 1))
          };
        })
        .filter((concept) => concept.label.length > 0)
    : [];
  return { tokens, layers, concepts };
}

function isChatMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as ChatMessage;
  return (candidate.role === "user" || candidate.role === "assistant") && typeof candidate.content === "string";
}

function normalizeQuestionPunctuation(value: string) {
  return value
    .replaceAll("'", "’")
    .replaceAll("‘", "’")
    .replaceAll("“", '"')
    .replaceAll("”", '"');
}

function stripRoleplayPrefix(text: string, suspectName: string) {
  const shortName = suspectName.split(" ").slice(-2).join(" ");
  const prefixes = [suspectName, shortName, `Captain ${shortName}`];
  for (const prefix of prefixes) {
    if (text.toLowerCase().startsWith(`${prefix.toLowerCase()}:`)) {
      return text.slice(prefix.length + 1).trim();
    }
  }
  return text;
}

function isThirdPersonNarration(text: string, suspectName: string) {
  const shortName = suspectName.split(" ").slice(-2).join(" ");
  const lowered = text.toLowerCase();
  return [suspectName, shortName].some((name) => {
    const candidate = name.toLowerCase();
    return lowered.startsWith(candidate) && !lowered.startsWith(`${candidate}:`);
  });
}
