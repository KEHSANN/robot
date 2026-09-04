"""Model registry.

The cheap five-model panel runs on every event that survives Stage 0; the heavy
NVIDIA panel runs only on events the router marks critical. Every id here can be
overridden from the environment, because provider catalogues change model slugs
faster than code gets redeployed.

Run ``python run.py doctor`` to verify each configured id actually resolves
against its provider before relying on it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from services.config import _list, _str, settings


@dataclass(frozen=True)
class ModelSpec:
    """Everything the transport layer needs to call one model."""

    id: str
    provider: str
    label: str

    #: Provider honours a structured-output request (``response_format`` /
    #: ``responseMimeType``). When False we rely on prompt instructions plus the
    #: tolerant parser in :mod:`services.jsonparse`.
    supports_json_mode: bool = True
    #: Gemma served over the Gemini API rejects ``systemInstruction``, so the
    #: system prompt gets folded into the user turn instead.
    supports_system_prompt: bool = True
    max_output_tokens: int = 1024
    temperature: float = 0.1
    #: Set for reasoning models so we can budget extra tokens for the thinking
    #: they emit before the answer.
    reasoning: bool = False

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.id}"

    def with_limits(self, *, max_output_tokens: int | None = None, temperature: float | None = None) -> "ModelSpec":
        return replace(
            self,
            max_output_tokens=max_output_tokens or self.max_output_tokens,
            temperature=self.temperature if temperature is None else temperature,
        )


# --------------------------------------------------------------------------- #
# The cheap parallel panel (Stages 1-4)
# --------------------------------------------------------------------------- #
# These five ids are the ones this project already benchmarked successfully
# against Groq and the Gemini API.

DEFAULT_PANEL: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="openai/gpt-oss-120b",
        provider="groq",
        label="GPT-OSS 120B",
        max_output_tokens=1400,
    ),
    ModelSpec(
        id="qwen/qwen3.8-27b",
        provider="groq",
        label="Qwen 3.8 27B",
        max_output_tokens=2400,
        reasoning=True,
    ),
    ModelSpec(
        id="gemma-4-31b-it",
        provider="gemini",
        label="Gemma 4 31B",
        # Gemma over the Gemini API supports neither system instructions nor a
        # JSON response mime type.
        supports_json_mode=False,
        supports_system_prompt=False,
        max_output_tokens=1400,
    ),
    ModelSpec(
        id="gemini-3.5-flash-lite",
        provider="gemini",
        label="Gemini 3.5 Flash Lite",
        max_output_tokens=1400,
    ),
    ModelSpec(
        id="qwen/qwen3.6-27b",
        provider="groq",
        label="Qwen 3.6 27B",
        max_output_tokens=2400,
        reasoning=True,
    ),
)


# --------------------------------------------------------------------------- #
# The heavy NVIDIA panel (final layer only)
# --------------------------------------------------------------------------- #
# These three slugs were verified against NVIDIA's live ``/v1/models`` catalogue
# with `python run.py doctor`. Two details worth keeping in mind when they change:
#
# * DeepSeek is published under a *dated* build id. The undated alias
#   ``deepseek-ai/deepseek-v4-pro`` answers HTTP 410 "reached its end of life",
#   which is a fatal error the failover cannot route around — no other key will
#   make a retired model exist. Pin the dated build and bump it deliberately.
# * The Nemotron Ultra id does not carry a ``llama-`` prefix or a ``-v1`` suffix,
#   despite the previous generation doing both.
#
# Re-check with `python run.py doctor`; override with NVIDIA_FINAL_MODELS=... .

DEFAULT_FINAL_PANEL: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="deepseek-ai/deepseek-v4-pro-0813",
        provider="nvidia",
        label="DeepSeek V4 Pro",
        max_output_tokens=4096,
        temperature=0.2,
        reasoning=True,
    ),
    ModelSpec(
        id="moonshotai/kimi-k3",
        provider="nvidia",
        label="Kimi K3",
        max_output_tokens=4096,
        temperature=0.2,
    ),
    ModelSpec(
        id="nvidia/nemotron-3-ultra-550b-a55b",
        provider="nvidia",
        label="Nemotron Ultra 550B A55B",
        max_output_tokens=4096,
        temperature=0.2,
        reasoning=True,
    ),
)


#: Model used for Stage 0 embeddings. Dimensionality must match the DB column.
EMBED_MODEL = ModelSpec(
    id=settings.stage0.embed_model,
    provider=_str("EMBED_PROVIDER", "gemini"),
    label="Gemini Embedding",
)


# --------------------------------------------------------------------------- #
# Stage 0 fact extraction
# --------------------------------------------------------------------------- #
# Extraction is a transcription task, not a judgement call, so it runs on ONE
# model rather than a panel — a five-model vote on "what does this article say"
# would burn five times the tokens to agree with itself. The list is an ordered
# fallback chain, and it deliberately crosses providers: if Gemini is entirely
# down, Stage 0 still gets its facts from Groq instead of stalling the pipeline.

DEFAULT_EXTRACTORS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="gemini-3.5-flash-lite",
        provider="gemini",
        label="Gemini 3.5 Flash Lite",
        max_output_tokens=1600,
        temperature=0.0,
    ),
    ModelSpec(
        id="openai/gpt-oss-120b",
        provider="groq",
        label="GPT-OSS 120B",
        max_output_tokens=1600,
        temperature=0.0,
    ),
    ModelSpec(
        id="openai/gpt-oss-20b",
        provider="groq",
        label="GPT-OSS 20B",
        max_output_tokens=1600,
        temperature=0.0,
    ),
)


def _parse_override(env_name: str, defaults: tuple[ModelSpec, ...]) -> tuple[ModelSpec, ...]:
    """Build a panel from ``provider:model_id`` entries in an env variable.

    Unknown ids inherit sensible defaults; ids that match a known model reuse
    that model's quirk flags (JSON mode, system prompt support).
    """
    raw = _list(env_name)
    if not raw:
        return defaults

    known = {spec.id: spec for spec in DEFAULT_PANEL + DEFAULT_FINAL_PANEL + DEFAULT_EXTRACTORS}
    fallback_provider = defaults[0].provider if defaults else "groq"
    panel: list[ModelSpec] = []

    for entry in raw:
        provider, _, model_id = entry.rpartition(":")
        model_id = model_id.strip()
        provider = (provider or "").strip() or None
        if not model_id:
            continue
        if model_id in known:
            spec = known[model_id]
            panel.append(replace(spec, provider=provider or spec.provider))
        else:
            panel.append(
                ModelSpec(
                    id=model_id,
                    provider=provider or fallback_provider,
                    label=model_id,
                    max_output_tokens=defaults[0].max_output_tokens if defaults else 1400,
                    temperature=defaults[0].temperature if defaults else 0.1,
                )
            )
    return tuple(panel) or defaults


#: Stage 1-4 panel. Override with e.g.
#: STAGE_PANEL_MODELS=groq:openai/gpt-oss-120b,gemini:gemini-3.5-flash-lite
PANEL: tuple[ModelSpec, ...] = _parse_override("STAGE_PANEL_MODELS", DEFAULT_PANEL)

#: Heavy final panel. Override with NVIDIA_FINAL_MODELS=nvidia:<slug>,...
FINAL_PANEL: tuple[ModelSpec, ...] = _parse_override("NVIDIA_FINAL_MODELS", DEFAULT_FINAL_PANEL)

#: Stage 0 fact-extraction fallback chain, tried in order.
EXTRACTORS: tuple[ModelSpec, ...] = _parse_override("STAGE0_EXTRACT_MODELS", DEFAULT_EXTRACTORS)


def panel_for_stage(stage: int) -> tuple[ModelSpec, ...]:
    """Which models run a given stage.

    Stages 1-4 use the cheap panel. Stage 5 uses the same models but with a
    larger token budget, since it receives event history and cross-source
    evidence. The final layer is NVIDIA-only, per the architecture.
    """
    if stage in (1, 2, 3, 4):
        return PANEL
    if stage == 5:
        return tuple(
            spec.with_limits(max_output_tokens=max(spec.max_output_tokens, 3000))
            for spec in PANEL
        )
    return FINAL_PANEL


def find_model(model_id: str) -> ModelSpec | None:
    for spec in PANEL + FINAL_PANEL + EXTRACTORS + (EMBED_MODEL,):
        if spec.id == model_id:
            return spec
    return None


def all_models() -> tuple[ModelSpec, ...]:
    """Every model the pipeline may call, de-duplicated by provider:id."""
    seen: dict[str, ModelSpec] = {}
    for spec in PANEL + FINAL_PANEL + EXTRACTORS:
        seen.setdefault(spec.key, spec)
    return tuple(seen.values())


__all__ = [
    "DEFAULT_EXTRACTORS",
    "DEFAULT_FINAL_PANEL",
    "DEFAULT_PANEL",
    "EMBED_MODEL",
    "EXTRACTORS",
    "FINAL_PANEL",
    "PANEL",
    "ModelSpec",
    "all_models",
    "find_model",
    "panel_for_stage",
]
