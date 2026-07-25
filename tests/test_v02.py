from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from semantic_telephone.checkpoints import CheckpointStore
from semantic_telephone.chunking import chunk_text
from semantic_telephone.cli import app
from semantic_telephone.config import ConfigError, load_config
from semantic_telephone.metrics import SemanticMetricsUnavailable, SentenceTransformerMetrics
from semantic_telephone.models import (
    BudgetConfig,
    ChunkingConfig,
    EngineRoutingConfig,
    PipelineStageConfig,
    ProviderConfig,
    RuntimeConfig,
    SemanticMetricConfig,
    StageType,
)
from semantic_telephone.pipeline import run_pipeline
from semantic_telephone.planning import _cache_checks, create_plan, doctor_config
from semantic_telephone.providers.router import TranslationProviderRouter
from semantic_telephone.resources import profile_names, prompt_text
from semantic_telephone.runtime import BudgetExceededError, RequestController
from semantic_telephone.utils.files import atomic_write_json, read_json


def test_packaged_resources_and_profile_init(tmp_path: Path) -> None:
    assert "translate_only" in profile_names()
    assert "<<<TEXT>>>" not in prompt_text("grammar_repair")
    destination = tmp_path / "starter"
    result = CliRunner().invoke(
        app,
        ["init", str(destination), "--profile", "nllb_only"],
    )
    assert result.exit_code == 0
    config = load_config(destination / "semantic-telephone.yaml")
    assert config.input_path == "input.txt"
    assert config.translation.default_provider == "nllb"
    conservative = tmp_path / "conservative"
    result = CliRunner().invoke(
        app,
        ["init", str(conservative), "--profile", "conservative_reconstruction"],
    )
    assert result.exit_code == 0
    conservative_config = load_config(conservative / "semantic-telephone.yaml")
    assert (
        conservative_config.custom_prompts["reconstruction"]
        == "builtin:restrained_reconstruction"
    )


def test_zero_overlap_is_empty_and_negative_overlap_is_rejected(config_file: Path) -> None:
    chunks = chunk_text(
        "first.\n\nsecond.\n\nthird.",
        ChunkingConfig(target_chars=5, max_chars=100, paragraph_overlap=0),
    )
    assert all(chunk.context_prefix == "" for chunk in chunks)
    trailing_newline = chunk_text(
        "one short paragraph.\n",
        ChunkingConfig(target_chars=100, max_chars=200),
    )
    assert [chunk.source_text for chunk in trailing_newline] == ["one short paragraph."]
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["chunking"]["paragraph_overlap"] = -1
    config_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="paragraph_overlap"):
        load_config(config_file)


def test_runtime_deprecation_and_malformed_section(config_file: Path) -> None:
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["runtime"]["resume"] = True
    config_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.warns(FutureWarning, match="runtime.resume"):
        load_config(config_file)
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["chunking"]["unknown"] = 1
    config_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid 'chunking'"):
        load_config(config_file)


def test_invalid_provider_numeric_option_is_a_config_error(config_file: Path) -> None:
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    raw["translation"]["timeout_seconds"] = "eventually"
    config_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="translation.timeout_seconds"):
        load_config(config_file)


def test_request_controller_rate_budget_usage_and_resume(tmp_path: Path) -> None:
    now = [0.0]
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)
        now[0] += delay

    runtime = RuntimeConfig(
        requests_per_minute=60,
        budgets=BudgetConfig(max_requests=2, max_total_tokens=10, max_cost_usd=1.0),
    )
    events = tmp_path / "events.jsonl"
    controller = RequestController(runtime, events, clock=lambda: now[0], sleeper=sleep)

    async def exercise() -> None:
        first = await controller.before_request(provider="remote", operation="generate")
        controller.request_succeeded(
            first,
            provider="remote",
            operation="generate",
            usage={"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.25},
        )
        second = await controller.before_request(
            provider="remote",
            provider_type="openai_compatible",
            operation="generate",
            task="reconstruction",
            retry_attempt=2,
        )
        controller.request_succeeded(
            second,
            provider="remote",
            operation="generate",
            usage={"total_tokens": 5, "cost_usd": 0.25},
        )
        with pytest.raises(BudgetExceededError, match="request budget"):
            await controller.before_request(provider="remote", operation="generate")

    asyncio.run(exercise())
    assert delays == [1.0]
    assert controller.summary()["total_tokens"] == 10
    assert controller.summary()["retries"] == 1
    request_events = [
        json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()
    ]
    assert request_events[2]["provider_alias"] == "remote"
    assert request_events[2]["provider_type"] == "openai_compatible"
    assert request_events[2]["retry_attempt"] == 2
    assert request_events[3]["outcome"] == "success"
    restored = RequestController(runtime, events)
    assert restored.summary()["http_attempts"] == 2
    assert restored.summary()["total_tokens"] == 10
    assert restored.summary()["retries"] == 1

    concurrent = RequestController(
        RuntimeConfig(requests_per_minute=60),
        tmp_path / "concurrent-events.jsonl",
        clock=lambda: now[0],
        sleeper=sleep,
    )

    async def start_concurrently() -> None:
        await asyncio.gather(
            *(
                concurrent.before_request(provider="remote", operation="generate")
                for _ in range(3)
            )
        )

    asyncio.run(start_concurrently())
    assert delays == [1.0, 1.0, 1.0]

    cost_controller = RequestController(
        RuntimeConfig(
            requests_per_minute=0,
            budgets=BudgetConfig(max_cost_usd=0.1),
        ),
        tmp_path / "cost-events.jsonl",
    )

    async def exceed_observed_cost() -> None:
        request_id = await cost_controller.before_request(
            provider="remote",
            operation="generate",
        )
        cost_controller.request_succeeded(
            request_id,
            provider="remote",
            operation="generate",
            usage={"cost_usd": 0.25},
        )
        with pytest.raises(BudgetExceededError, match="cost budget"):
            await cost_controller.before_request(
                provider="remote",
                operation="generate",
            )

    asyncio.run(exceed_observed_cost())
    assert cost_controller.summary()["cost_usd"] == 0.25


def test_budget_exhaustion_cannot_trigger_translation_fallback() -> None:
    fallback_called = False

    class Exhausted:
        def supports_pair(self, source: str, target: str) -> bool:
            return source != target

        async def translate(self, *args: object, **kwargs: object) -> object:
            raise BudgetExceededError("budget exhausted")

    class Fallback:
        def supports_pair(self, source: str, target: str) -> bool:
            return source != target

        async def translate(self, *args: object, **kwargs: object) -> object:
            nonlocal fallback_called
            fallback_called = True
            raise AssertionError("budget exhaustion must bypass provider fallback")

    router = TranslationProviderRouter(
        {"primary": Exhausted(), "fallback": Fallback()},  # type: ignore[arg-type]
        default_provider="primary",
        routing=EngineRoutingConfig(
            mode="quality_fallback",
            fallback_order=["primary", "fallback"],
        ),
    )
    with pytest.raises(BudgetExceededError):
        asyncio.run(
            router.translate_candidates(
                ["primary", "fallback"],
                "text",
                "en",
                "ru",
                seed=1,
            )
        )
    assert fallback_called is False


def test_memory_resume_replays_checkpointed_side_effects(config_file: Path) -> None:
    source_path = config_file.parent / "input.txt"
    source_path.write_text(
        "Лира вошла. Лира остановилась.\n\n"
        "Лира ответила. Лира снова остановилась.\n\n"
        "Лира ушла. Лира вернулась.",
        encoding="utf-8",
    )
    config = load_config(config_file)
    config.memory.enabled = True
    config.memory.minimum_count = 1
    config.context.enabled = False
    config.pipeline = [
        config.pipeline[0],
        config.pipeline[1],
        PipelineStageConfig(type=StageType.MEMORY_EXTRACTION),
        config.pipeline[2],
    ]
    directory = asyncio.run(run_pipeline(config))
    final_before = (directory / "final.txt").read_text(encoding="utf-8")
    state_before = read_json(directory / "memory" / "state.json")
    observations_path = directory / "memory" / "observations.jsonl"
    observations_before = (
        observations_path.read_text(encoding="utf-8") if observations_path.exists() else ""
    )

    # Simulate interruption after the checkpoint was committed but before its
    # memory side effect: replay must reconstruct exactly from checkpoints.
    (directory / "memory" / "state.json").unlink()
    if observations_path.exists():
        observations_path.unlink()
    asyncio.run(run_pipeline(config, run_directory=directory))

    assert (directory / "final.txt").read_text(encoding="utf-8") == final_before
    assert read_json(directory / "memory" / "state.json") == state_before
    assert (
        observations_path.read_text(encoding="utf-8") if observations_path.exists() else ""
    ) == observations_before
    memory_stage = next((directory / "chunks").glob("*/stage-03-memory-extraction.json"))
    assert read_json(memory_stage)["memory_event_id"]

    # Simulate interruption before checkpoint commit. The deterministic mock
    # extraction is regenerated, while later checkpoints remain reusable.
    memory_output = memory_stage.with_name(
        memory_stage.name.removesuffix(".json") + "-output.txt"
    )
    memory_stage.unlink()
    memory_output.unlink()
    (directory / "memory" / "state.json").unlink()
    observations_path.unlink()
    asyncio.run(run_pipeline(config, run_directory=directory))
    assert (directory / "final.txt").read_text(encoding="utf-8") == final_before
    assert read_json(directory / "memory" / "state.json") == state_before
    assert observations_path.read_text(encoding="utf-8") == observations_before


def test_schema_v1_memory_resume_is_rejected(config_file: Path) -> None:
    config = load_config(config_file)
    config.memory.enabled = True
    config.pipeline.insert(-1, PipelineStageConfig(type=StageType.MEMORY_EXTRACTION))
    directory = asyncio.run(run_pipeline(config))
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.pop("artifact_schema_version")
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="schema-v1"):
        asyncio.run(run_pipeline(config, run_directory=directory))
    assert "artifact_schema_version" not in read_json(manifest_path)


def test_schema_v1_non_memory_run_remains_readable_and_resumable(
    config_file: Path,
) -> None:
    config = load_config(config_file)
    directory = asyncio.run(run_pipeline(config))
    expected = (directory / "final.txt").read_text(encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.pop("artifact_schema_version")
    atomic_write_json(manifest_path, manifest)

    assert CliRunner().invoke(app, ["inspect", str(directory)]).exit_code == 0
    assert CliRunner().invoke(app, ["report", str(directory)]).exit_code == 0
    asyncio.run(run_pipeline(config, run_directory=directory))
    assert (directory / "final.txt").read_text(encoding="utf-8") == expected
    assert read_json(manifest_path)["state"] == "completed"


def test_hop_outputs_are_checkpoint_validated(config_file: Path) -> None:
    directory = asyncio.run(run_pipeline(load_config(config_file)))
    stage_path = next((directory / "chunks").glob("*/stage-01-translation-cycle.json"))
    stage = read_json(stage_path)
    detail = stage["translation_details"][0]
    hop_path = stage_path.parent / detail["output_path"]
    assert hop_path.exists()
    store = CheckpointStore(stage_path.parent)
    assert store.load(1, "translation_cycle", stage["stage_checksum"]) is not None
    hop_path.write_text("tampered", encoding="utf-8")
    assert store.load(1, "translation_cycle", stage["stage_checksum"]) is None


def test_plan_is_offline_and_credential_free(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(config_file)

    class ForbiddenClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("offline plan attempted network access")

    monkeypatch.setattr("httpx.AsyncClient", ForbiddenClient)
    monkeypatch.setattr(
        "semantic_telephone.planning.os.getenv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("offline plan attempted to read credentials")
        ),
    )
    plan = create_plan(config)
    assert plan["chunks"] > 0
    assert plan["planned_routes"]


def test_doctor_passes_for_bundled_mock_capabilities(config_file: Path) -> None:
    report = asyncio.run(doctor_config(load_config(config_file)))
    assert report["ok"] is True
    assert all(check["status"] != "fail" for check in report["checks"])


def test_doctor_uses_metadata_only_http_checks(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(config_file)
    config.translation = ProviderConfig(
        provider="libretranslate",
        base_url="https://libre.invalid",
    )
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        assert request.method == "GET"
        return httpx.Response(
            200,
            request=request,
            json=[{"code": "en"}, {"code": "ru"}],
        )

    transport = httpx.MockTransport(respond)
    real_client = httpx.AsyncClient

    def mock_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=transport, timeout=kwargs.get("timeout"))

    monkeypatch.setattr("semantic_telephone.planning.httpx.AsyncClient", mock_client)
    report = asyncio.run(doctor_config(config))
    assert report["ok"] is True
    assert seen == ["/languages"]


def test_doctor_downloads_only_exact_configured_models(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(config_file)
    config.translation = ProviderConfig(
        provider="opus_mt",
        options={
            "configured_pairs_only": True,
            "pairs": {
                "en-ru": "Helsinki-NLP/opus-mt-en-ru",
                "ru-en": "Helsinki-NLP/opus-mt-ru-en",
            },
            "revisions": {"en-ru": "revision-a"},
        },
    )
    downloads: list[tuple[str, str | None, bool]] = []
    module = ModuleType("huggingface_hub")

    def snapshot_download(
        *,
        repo_id: str,
        revision: str | None,
        local_files_only: bool,
    ) -> str:
        downloads.append((repo_id, revision, local_files_only))
        return "/fake/cache"

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    checks = _cache_checks(config, allow_downloads=True)
    assert all(check["status"] == "pass" for check in checks)
    assert downloads == [
        ("Helsinki-NLP/opus-mt-en-ru", "revision-a", False),
        ("Helsinki-NLP/opus-mt-ru-en", None, False),
    ]


def test_semantic_metrics_with_fake_encoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("sentence_transformers")
    encode_calls: list[list[str]] = []

    class FakeSentenceTransformer:
        def __init__(self, model: str, **kwargs: object) -> None:
            assert model

        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            assert kwargs["batch_size"] == 16
            encode_calls.append(texts)
            return [[1.0, 0.0] if "same" in text else [0.0, 1.0] for text in texts]

    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    metric = SentenceTransformerMetrics(
        SemanticMetricConfig(enabled=True, allow_downloads=False)
    )
    chunk = tmp_path / "chunks" / "0001"
    chunk.mkdir(parents=True)
    (chunk / "source.txt").write_text("same source", encoding="utf-8")
    (chunk / "final.txt").write_text("different result", encoding="utf-8")
    (chunk / "hop.txt").write_text("same hop", encoding="utf-8")
    atomic_write_json(
        chunk / "stage-01-translation-cycle.json",
        {
            "input_text": "different start",
            "stage_type": "translation_cycle",
            "translation_details": [
                {
                    "output_path": "hop.txt",
                    "source_language": "en",
                    "target_language": "ru",
                    "provider": "fake",
                    "model": "fake",
                }
            ],
        },
    )
    result = metric.calculate("same source", "different result", tmp_path)
    assert result["available"] is True
    assert result["overall_similarity"] == 0.0
    assert result["largest_drift_hop"]["provider"] == "fake"
    assert result["aggregate_route_drift"] == 1.0
    assert len(encode_calls) == 1


def test_semantic_metrics_report_cache_only_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("sentence_transformers")

    class MissingModel:
        def __init__(self, model: str, **kwargs: object) -> None:
            assert kwargs["local_files_only"] is True
            raise OSError(f"{model} is not cached")

    module.SentenceTransformer = MissingModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    with pytest.raises(SemanticMetricsUnavailable, match="local-cache-only"):
        SentenceTransformerMetrics(
            SemanticMetricConfig(enabled=True, allow_downloads=False)
        )
