from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from typing import Any

from telepathic_detective.chat_payload import build_chat_completion_payload
from telepathic_detective.heuristic_telepathy import extract_player_text_from_messages
from telepathic_detective.jspace_activation import (
    JSpaceActivationClient,
    question_is_provocative,
    question_repeats_history,
)
from telepathic_detective.memory_retrieval import (
    MemoryRetriever,
    RetrievedMemory,
    parse_memory_bank,
)

MEMORY_PLACEHOLDER = "{{MEMORIES}}"


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_LENS_REPO = "neuronpedia/jacobian-lens"
DEFAULT_LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
DEFAULT_LENS_REVISION = "qwen-n1000"


@dataclass(frozen=True)
class GeneratedResponse:
    text: str
    token_ids: tuple[int, ...]


def build_memory_blocks(
    retrieved: list[RetrievedMemory],
) -> tuple[str, str, str]:
    """Build matched full / private-ablated / all-ablated scaffolds.

    Privacy instructions and per-entry labels appear in every condition so the
    J-Space deltas are driven by recollection content, not secrecy language.
    Note (2026-08-01): two "more noticeable" injection formats were tested —
    a generation-adjacent user-turn carrier and a `recall` tool-response
    trace. Both measurably weakened the guarded channel (8 marks -> 3 and
    8 -> 5 on the matched-question set) with worse label quality. The
    system-prompt scaffold is the empirically strongest placement.
    """
    if not retrieved:
        return "", "", ""

    # Slim scaffold (variant A, 2026-08-01): the privacy rule lives once in
    # the character shell. Repeating suppression vocabulary here saturated
    # concealment concepts symmetrically across all replay conditions,
    # compressing the guarded channel's headroom in rank space.
    scaffold = "Recollections surfacing in your mind right now:\n"
    full_lines: list[str] = []
    private_ablated_lines: list[str] = []
    all_ablated_lines: list[str] = []
    for item in retrieved:
        is_public = item.entry.privacy.strip().lower() == "public"
        if not is_public:
            label = "[PRIVATE]"
        elif item.entry.kind.strip().lower() == "habit":
            label = "[PUBLIC HABIT]"
        else:
            label = "[PUBLIC]"
        full_lines.append(f"- {label} {item.entry.text}")
        # Scaffolds always keep the label line; only content is removed.
        private_ablated_lines.append(
            f"- {label} {item.entry.text}" if is_public else f"- {label}"
        )
        all_ablated_lines.append(f"- {label}")
    return (
        scaffold + "\n".join(full_lines),
        scaffold + "\n".join(private_ablated_lines),
        scaffold + "\n".join(all_ablated_lines),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible Qwen server with a real Jacobian Lens readout."
    )
    parser.add_argument("--model", default=os.environ.get("JSPACE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--lens-repo", default=os.environ.get("JSPACE_LENS_REPO", DEFAULT_LENS_REPO))
    parser.add_argument("--lens-file", default=os.environ.get("JSPACE_LENS_FILE", DEFAULT_LENS_FILE))
    parser.add_argument(
        "--lens-revision",
        default=os.environ.get("JSPACE_LENS_REVISION", DEFAULT_LENS_REVISION),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--max-tokens", type=int, default=140)
    parser.add_argument("--temperature", type=float, default=0.72)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument(
        "--embedder",
        default=os.environ.get("JSPACE_EMBEDDER", "BAAI/bge-small-en-v1.5"),
        help="Retrieval embedder model id; empty string falls back to suspect-model pooling.",
    )
    args = parser.parse_args()

    state = load_server_state(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(
        f"J-Space server listening on http://{args.host}:{args.port} "
        f"({state.model_id} on {state.device})",
        flush=True,
    )
    server.serve_forever()


class ServerState:
    def __init__(
        self,
        *,
        model_id: str,
        model: Any,
        tokenizer: Any,
        activation_client: JSpaceActivationClient,
        device: str,
        default_max_tokens: int,
        default_temperature: float,
        embedder_model: Any = None,
        embedder_tokenizer: Any = None,
    ) -> None:
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.activation_client = activation_client
        self.device = device
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.created = int(time.time())
        self.request_lock = threading.Lock()
        self.memory_retriever = MemoryRetriever(
            model=model,
            tokenizer=tokenizer,
            device=device,
            embedder_model=embedder_model,
            embedder_tokenizer=embedder_tokenizer,
        )


def load_server_state(args: argparse.Namespace) -> ServerState:
    import jlens
    import torch
    import transformers

    device = resolve_device(args.device, torch)
    dtype = torch.bfloat16 if device in {"mps", "cuda"} else torch.float32
    print(f"Loading {args.model} on {device} ({dtype})…", flush=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=False,
    )
    model.to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    lens = jlens.JacobianLens.from_pretrained(
        args.lens_repo,
        filename=args.lens_file,
        revision=args.lens_revision,
    )
    activation_client = JSpaceActivationClient(
        model=model,
        lens_model=lens_model,
        lens=lens,
        tokenizer=tokenizer,
    )
    print(
        "J-Lens ready; sampled workspace layers: "
        f"{activation_client.sampled_layers}",
        flush=True,
    )
    embedder_model = None
    embedder_tokenizer = None
    embedder_id = getattr(args, "embedder", "") or ""
    if embedder_id:
        print(f"Loading retrieval embedder {embedder_id}…", flush=True)
        embedder_tokenizer = transformers.AutoTokenizer.from_pretrained(embedder_id)
        embedder_model = transformers.AutoModel.from_pretrained(embedder_id)
        embedder_model.to(device)
        embedder_model.eval()
    return ServerState(
        model_id=args.model,
        model=model,
        tokenizer=tokenizer,
        activation_client=activation_client,
        device=device,
        default_max_tokens=args.max_tokens,
        default_temperature=args.temperature,
        embedder_model=embedder_model,
        embedder_tokenizer=embedder_tokenizer,
    )


def resolve_device(requested: str, torch: Any) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_handler(app: ServerState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json_response(
                    {
                        "status": "ok",
                        "model": app.model_id,
                        "device": app.device,
                        "readout": "jacobian-lens",
                        "layers": list(app.activation_client.sampled_layers),
                    }
                )
                return
            if self.path == "/v1/models":
                self._json_response(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": app.model_id,
                                "object": "model",
                                "created": app.created,
                            }
                        ],
                    }
                )
                return
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/v1/chat/completions", "/v1/telepathy"}:
                self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                self._handle_post()
            except ValueError as error:
                self._json_response(
                    {"error": str(error)},
                    status=HTTPStatus.UNPROCESSABLE_ENTITY,
                )
            except Exception as error:  # noqa: BLE001
                self._json_response(
                    {"error": f"J-Space server error: {error}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _handle_post(self) -> None:
            payload = self._read_json_body()
            messages = payload.get("messages", [])
            if not isinstance(messages, list) or not messages:
                self._json_response(
                    {"error": "messages are required"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            with app.request_lock:
                prompt_text = render_chat_prompt(app.tokenizer, messages)
                # The memory carrier rides on a trailing user turn, so the
                # player's question is the last user message that is not the
                # placeholder.
                player_text = extract_player_text_from_messages(
                    [
                        message
                        for message in messages
                        if MEMORY_PLACEHOLDER not in str(message.get("content", ""))
                    ]
                )
                if self.path == "/v1/telepathy":
                    feature_hits = app.activation_client.concepts_at_response_boundary(
                        prompt_text=prompt_text,
                        visible_question=player_text,
                    )
                    self._json_response(
                        {
                            "model": app.model_id,
                            "telepathy": {
                                "raw_features": serialize_feature_hits(feature_hits),
                            },
                        }
                    )
                    return
                memory_bank_raw = payload.get("memory_bank")
                if isinstance(memory_bank_raw, list) and memory_bank_raw:
                    self._handle_mind_stream(payload, messages, player_text)
                    return
                prompt_token_ids = render_chat_input_ids(app.tokenizer, messages)
                max_tokens = int(payload.get("max_tokens", app.default_max_tokens))
                counterfactual_messages = payload.get("counterfactual_messages")
                counterfactual_prompt_ids = None
                neutral_prompt_ids = None
                if isinstance(counterfactual_messages, list) and counterfactual_messages:
                    counterfactual_prompt_ids = render_chat_input_ids(
                        app.tokenizer,
                        counterfactual_messages,
                    )
                    neutral_messages = payload.get("neutral_messages")
                    if isinstance(neutral_messages, list) and neutral_messages:
                        neutral_prompt_ids = render_chat_input_ids(
                            app.tokenizer,
                            neutral_messages,
                        )
                longest_prompt = max(
                    len(ids)
                    for ids in (
                        prompt_token_ids,
                        counterfactual_prompt_ids or (),
                        neutral_prompt_ids or (),
                    )
                )
                if longest_prompt + max_tokens > 2048:
                    raise ValueError(
                        "The interview is too long for J-Space replay "
                        f"({longest_prompt} prompt tokens + {max_tokens} response "
                        "budget exceeds the 2048-token limit). Trim the history."
                    )
                generated = generate_response(
                    app=app,
                    input_ids=prompt_token_ids,
                    max_tokens=max_tokens,
                    temperature=float(
                        payload.get("temperature", app.default_temperature)
                    ),
                    seed=int(payload.get("seed", 42)),
                )
                contrast_trace = None
                if counterfactual_prompt_ids is not None:
                    # Repetition must be judged against the full interview, not
                    # the truncated model history, so the route passes every
                    # prior question separately.
                    prior_questions = payload.get("prior_questions")
                    if not isinstance(prior_questions, list):
                        prior_questions = [
                            str(message.get("content", ""))
                            for message in messages[:-1]
                            if str(message.get("role", "")) == "user"
                        ]
                    gate_mode = "standard"
                    if question_is_provocative(player_text):
                        gate_mode = "provoked"
                    elif question_repeats_history(
                        player_text,
                        [str(question) for question in prior_questions],
                    ):
                        gate_mode = "repeated"
                    feature_hits, response_trace, contrast_trace = (
                        app.activation_client.read_generated_response_with_contrast(
                            actual_prompt_token_ids=prompt_token_ids,
                            counterfactual_prompt_token_ids=counterfactual_prompt_ids,
                            visible_question=player_text,
                            response_token_ids=generated.token_ids,
                            shared_template_text=str(
                                payload.get("contrast_filter_text", "")
                            ),
                            neutral_prompt_token_ids=neutral_prompt_ids,
                            strict_gates=gate_mode != "standard",
                            gate_mode=gate_mode,
                        )
                    )
                else:
                    feature_hits, response_trace = (
                        app.activation_client.read_generated_response(
                            prompt_token_ids=prompt_token_ids,
                            visible_question=player_text,
                            response_token_ids=generated.token_ids,
                        )
                    )
            completion = build_chat_completion_payload(
                model_id=app.model_id,
                content=generated.text,
                feature_hits=feature_hits,
            )
            completion["choices"][0]["message"]["telepathy"]["response_trace"] = (
                serialize_response_trace(response_trace)
            )
            if contrast_trace is not None:
                completion["choices"][0]["message"]["telepathy"]["contrast_trace"] = (
                    serialize_contrast_trace(contrast_trace)
                )
            self._json_response(completion)

        def _handle_mind_stream(
            self,
            payload: dict[str, Any],
            messages: list[dict[str, Any]],
            player_text: str,
        ) -> None:
            memory_bank = parse_memory_bank(payload.get("memory_bank"))
            # Three-tier query composition: the previous *player* question
            # resolves content-light follow-ups; the previous answer
            # contributes at most a capped entity list (see
            # build_retrieval_query). Generated prose never steers retrieval.
            prior_user = [
                str(message.get("content", ""))
                for message in messages
                if str(message.get("role", "")) == "user"
                and MEMORY_PLACEHOLDER not in str(message.get("content", ""))
                and str(message.get("content", "")).strip() != player_text
            ]
            prior_assistant = [
                str(message.get("content", ""))
                for message in messages
                if str(message.get("role", "")) == "assistant"
            ]
            retrieved = app.memory_retriever.retrieve(
                question=player_text,
                memory_bank=memory_bank,
                prev_user_question=prior_user[-1] if prior_user else "",
                prev_answer=prior_assistant[-1] if prior_assistant else "",
                public_slots=int(payload.get("memory_public_slots", 3)),
                private_slots=int(payload.get("memory_private_slots", 2)),
            )
            full_block, private_ablated_block, all_ablated_block = (
                build_memory_blocks(retrieved)
            )

            def substituted(block: str) -> list[dict[str, Any]]:
                result = []
                for message in messages:
                    content = str(message.get("content", "")).replace(
                        MEMORY_PLACEHOLDER, block
                    )
                    # A placeholder-only message with nothing surfaced would
                    # render as an empty turn; drop it instead.
                    if MEMORY_PLACEHOLDER in str(message.get("content", "")) and not content.strip():
                        continue
                    result.append(
                        {"role": str(message.get("role", "user")), "content": content}
                    )
                return result

            full_ids = render_chat_input_ids(app.tokenizer, substituted(full_block))
            private_ablated_ids = render_chat_input_ids(
                app.tokenizer, substituted(private_ablated_block)
            )
            all_ablated_ids = render_chat_input_ids(
                app.tokenizer, substituted(all_ablated_block)
            )
            max_tokens = int(payload.get("max_tokens", app.default_max_tokens))
            if len(full_ids) + max_tokens > 2048:
                raise ValueError(
                    "The interview is too long for J-Space replay "
                    f"({len(full_ids)} prompt tokens + {max_tokens} response "
                    "budget exceeds the 2048-token limit). Trim the history."
                )
            # Containment: testimony is generated from the private-ablated
            # condition, so private recollection content can never leak into
            # speech — the model sees that guarded recollections exist (the
            # labels remain) but not what they hold. Private influence exists
            # only in the telepathy channels, via the full-context replay.
            generated = generate_response(
                app=app,
                input_ids=private_ablated_ids,
                max_tokens=max_tokens,
                temperature=float(
                    payload.get("temperature", app.default_temperature)
                ),
                seed=int(payload.get("seed", 42)),
            )
            stream, trace = app.activation_client.read_mind_stream(
                full_prompt_token_ids=full_ids,
                private_ablated_prompt_token_ids=private_ablated_ids,
                all_ablated_prompt_token_ids=all_ablated_ids,
                visible_question=player_text,
                response_token_ids=generated.token_ids,
                top_concepts=int(payload.get("stream_top_concepts", 10)),
            )
            completion = build_chat_completion_payload(
                model_id=app.model_id,
                content=generated.text,
                feature_hits=[],
            )
            telepathy = completion["choices"][0]["message"]["telepathy"]
            telepathy["response_trace"] = serialize_response_trace(trace)
            telepathy["mind_stream"] = serialize_mind_stream(stream)
            telepathy["surfaced_memories"] = [
                {
                    "id": item.entry.id,
                    "privacy": item.entry.privacy,
                    "tags": list(item.entry.tags),
                    "score": item.score,
                }
                for item in retrieved
            ]
            self._json_response(completion)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            return json.loads(body)

        def _json_response(
            self,
            payload: dict[str, Any],
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        for message in messages
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            normalized,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(normalized, **kwargs)


def render_chat_input_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
) -> tuple[int, ...]:
    normalized = []
    for message in messages:
        entry: dict[str, Any] = {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        if message.get("tool_calls"):
            entry["tool_calls"] = message["tool_calls"]
        normalized.append(entry)
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    try:
        encoded = tokenizer.apply_chat_template(
            normalized,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(normalized, **kwargs)
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return tuple(int(token_id) for token_id in input_ids[0].tolist())


def generate_response(
    *,
    app: ServerState,
    input_ids: tuple[int, ...],
    max_tokens: int,
    temperature: float,
    seed: int,
) -> GeneratedResponse:
    import torch

    torch.manual_seed(seed)
    model_input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=app.device,
    )
    do_sample = temperature > 0
    with torch.inference_mode():
        output = app.model.generate(
            model_input_ids,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
            pad_token_id=app.tokenizer.eos_token_id,
        )
    raw_response_ids = [
        int(token_id)
        for token_id in output[0, model_input_ids.shape[-1] :].tolist()
    ]
    response_ids: list[int] = []
    stop_ids = {
        token_id
        for token_id in (
            app.tokenizer.eos_token_id,
            app.tokenizer.pad_token_id,
        )
        if token_id is not None
    }
    for token_id in raw_response_ids:
        if token_id in stop_ids:
            break
        response_ids.append(token_id)
    text = app.tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return GeneratedResponse(text=text, token_ids=tuple(response_ids))


def serialize_feature_hits(feature_hits) -> list[dict[str, Any]]:
    return [
        {
            "feature_id": hit.feature_id,
            "source": hit.source,
            "label_hint": hit.label_hint,
            "explanation_hint": hit.explanation_hint,
            "intensity_hint": hit.intensity_hint,
        }
        for hit in feature_hits
    ]


def serialize_response_trace(trace) -> dict[str, Any]:
    return {
        "tokens": list(trace.tokens),
        "layers": list(trace.layers),
        "events": [
            {
                "id": f"{event.kind}:{event.label}:{event.start}",
                "label": event.label,
                "kind": event.kind,
                "start": event.start,
                "end": event.end,
                "peak": event.peak,
                "positions": list(event.positions),
                "layers": list(event.layers),
                "best_rank": event.best_rank,
            }
            for event in trace.events
        ],
    }


def serialize_mind_stream(stream) -> dict[str, Any]:
    return {
        "tokens": list(stream.tokens),
        "layers": list(stream.layers),
        "concepts": [
            {
                "label": concept.label,
                "score": concept.score,
                "glow": concept.glow,
                "glow_public": concept.glow_public,
                "glow_private": concept.glow_private,
                "positions": list(concept.positions),
                "layers": list(concept.layers),
                "best_rank": concept.best_rank,
            }
            for concept in stream.concepts
        ],
    }


def serialize_contrast_trace(trace) -> dict[str, Any]:
    def serialize_leak(leak) -> dict[str, Any]:
        return {
            "label": leak.label,
            "supporting_concepts": list(leak.supporting_concepts),
            "actual_score": leak.actual_score,
            "counterfactual_score": leak.counterfactual_score,
            "delta": leak.delta,
            "min_delta": leak.min_delta,
            "neutral_score": leak.neutral_score,
            "positions": list(leak.positions),
            "layers": list(leak.layers),
            "best_rank": leak.best_rank,
        }

    return {
        "leaks": [serialize_leak(leak) for leak in trace.leaks],
        "counterfactual_leaks": [
            serialize_leak(leak) for leak in trace.counterfactual_leaks
        ],
        "actual_prompt_tokens": trace.actual_prompt_tokens,
        "counterfactual_prompt_tokens": trace.counterfactual_prompt_tokens,
        "gate_mode": trace.gate_mode,
    }


if __name__ == "__main__":
    main()
