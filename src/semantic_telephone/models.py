from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class StageType(StrEnum):
    TRANSLATION_CYCLE = "translation_cycle"
    CONSERVATIVE_REPAIR = "conservative_repair"
    RECONSTRUCTION = "reconstruction"
    CONTEXTUAL_RECONSTRUCTION = "contextual_reconstruction"
    MEMORY_EXTRACTION = "memory_extraction"
    FINAL_TRANSLATION = "final_translation"


@dataclass(slots=True)
class EngineRoutingConfig:
    mode: str = "single_engine"
    provider: str | None = None
    route: list[str] = field(default_factory=list)
    pairs: dict[str, str] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    fallback_order: list[str] = field(default_factory=list)
    avoid_same_engine_consecutively: bool = False


@dataclass(slots=True)
class ProviderConfig:
    provider: str = "mock"
    enabled: bool = True
    base_url: str | None = None
    model: str = "mock"
    revision: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 60.0
    options: dict[str, Any] = field(default_factory=dict)
    default_provider: str | None = None
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    tasks: dict[str, ProviderConfig] = field(default_factory=dict)
    engine_routing: EngineRoutingConfig = field(default_factory=EngineRoutingConfig)


@dataclass(slots=True)
class ChunkingConfig:
    strategy: str = "target_chars"
    target_chars: int = 1100
    max_chars: int = 2200
    target_tokens: int = 300
    sentence_window: int = 6
    paragraph_overlap: int = 1


@dataclass(slots=True)
class PipelineStageConfig:
    type: StageType
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    repeat: int = 1
    seed: int | None = None
    seed_offset: int = 0


@dataclass(slots=True)
class ContextConfig:
    enabled: bool = False
    previous_chunks: int = 2
    max_chars: int = 3500
    truncation: str = "tail"
    source: str = "final_generated_text"
    include_intermediate: bool = False


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = False
    half_life_chunks: float = 20.0
    minimum_count: int = 2
    maximum_items_in_prompt: int = 8


@dataclass(slots=True)
class RuntimeConfig:
    concurrency: int = 1
    requests_per_minute: int = 30
    retries: int = 4
    retry_backoff_seconds: float = 2.0
    resume: bool = True
    failure_policy: str = "stop"
    fallback_provider: str | None = None


@dataclass(slots=True)
class RouteConfig:
    mode: str = "fixed"
    languages: list[str] = field(default_factory=list)
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    min_hops: int = 3
    max_hops: int = 7
    hub_language: str = "en"
    hub_frequency: int = 1
    mutations: int = 1
    per: str = "chunk"
    return_to_target_after_cycle: bool = True


@dataclass(slots=True)
class RunConfig:
    name: str
    seed: int
    source_language: str
    target_language: str
    input_path: str
    input_encoding: str = "utf-8"
    output_root: str = "runs"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    translation: ProviderConfig = field(default_factory=ProviderConfig)
    generation: ProviderConfig = field(default_factory=ProviderConfig)
    route: RouteConfig = field(default_factory=RouteConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    pipeline: list[PipelineStageConfig] = field(default_factory=list)
    temperatures: dict[str, float] = field(
        default_factory=lambda: {
            "conservative_repair": 0.3,
            "reconstruction": 0.7,
            "contextual_reconstruction": 0.7,
            "memory_extraction": 0.1,
            "report_generation": 0.2,
        }
    )
    custom_prompts: dict[str, str] = field(default_factory=dict)
    config_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    index: int
    source_text: str
    char_start: int
    char_end: int
    paragraphs: list[int]
    context_prefix: str
    checksum: str


@dataclass(slots=True)
class TranslationResult:
    text: str
    provider: str
    model: str
    response_id: str | None = None
    usage: dict[str, int | float] | None = None
    warnings: list[str] = field(default_factory=list)
    deterministic: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    response_id: str | None = None
    usage: dict[str, int | float] | None = None
    warnings: list[str] = field(default_factory=list)
    deterministic: bool | None = None


@dataclass(slots=True)
class StageResult:
    input_text: str
    output_text: str
    stage_type: str
    provider: str
    model: str
    source_language: str | None
    target_language: str | None
    route: list[str]
    duration_seconds: float
    attempts: int
    warnings: list[str]
    error: str | None
    usage: dict[str, int | float] | None
    input_checksum: str
    output_checksum: str
    stage_checksum: str
    applied: bool = True
    response_id: str | None = None
    prompt_checksum: str | None = None
    execution_succeeded: bool = True
    checkpoint_reusable: bool = True
    provider_route: list[str] = field(default_factory=list)
    translation_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunManifest:
    resolved_config: dict[str, Any]
    application_version: str
    started_at: str
    seed: int
    environment: dict[str, str]
    processed_chunks: list[str]
    state: str
    paths: dict[str, str]
    metrics: dict[str, Any]
    custom_prompts: bool
    prompt_checksums: dict[str, str]
    deterministic: bool
    translation_models: list[dict[str, Any]] = field(default_factory=list)
    report_generation: dict[str, Any] | None = None
    completed_at: str | None = None

    @classmethod
    def create(
        cls,
        config: RunConfig,
        *,
        version: str,
        environment: dict[str, str],
        paths: dict[str, str],
        prompt_checksums: dict[str, str],
    ) -> RunManifest:
        return cls(
            resolved_config=config.to_dict(),
            application_version=version,
            started_at=datetime.now(UTC).isoformat(),
            seed=config.seed,
            environment=environment,
            processed_chunks=[],
            state="running",
            paths=paths,
            metrics={},
            custom_prompts=bool(config.custom_prompts),
            prompt_checksums=prompt_checksums,
            deterministic=(
                (
                    all(
                        provider.provider == "mock"
                        for provider in config.translation.providers.values()
                    )
                    if config.translation.providers
                    else config.translation.provider == "mock"
                )
                and config.generation.provider == "mock"
                and all(
                    provider.provider == "mock" for provider in config.generation.tasks.values()
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
