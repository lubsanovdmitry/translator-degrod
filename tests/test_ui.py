from __future__ import annotations

import asyncio
import json
import time
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from semantic_telephone.cli import app
from semantic_telephone.ui import ConsoleService, UIError, _is_loopback_host
from semantic_telephone.utils.files import atomic_write_json

runner = CliRunner()


def _service(tmp_path: Path, *, worker: bool = False) -> ConsoleService:
    return ConsoleService(
        project_root=Path.cwd(),
        config_root=Path.cwd() / "configs",
        run_root=tmp_path / "runs",
        state_root=tmp_path / "ui-state",
        start_worker=worker,
    )


def _wait_for_terminal(service: ConsoleService, job_id: str) -> dict[str, Any]:
    assert service.queue is not None
    for _ in range(500):
        job = service.queue.get(job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


def test_profiles_are_classified_and_mock_is_explicit(tmp_path: Path) -> None:
    service = _service(tmp_path)
    profiles = {item["id"]: item for item in service.profiles()}

    assert profiles["mixed_local"]["kind"] == "real"
    assert profiles["translate_only"]["kind"] == "mock"
    assert "does not translate languages" in profiles["translate_only"]["note"]
    assert profiles["commercial_baseline"]["kind"] == "unavailable"
    assert profiles["commercial_baseline"]["runnable"] is False


def test_validation_patch_and_embedded_secret_rejection(tmp_path: Path) -> None:
    service = _service(tmp_path)
    profile = service.profile("translate_only")

    patched = service.patch_yaml(
        profile["yaml"],
        {
            "run.name": "guided-run",
            "run.seed": 12,
            "translation.languages": ["ru", "en", "ru"],
            "runtime.budgets.max_requests": None,
        },
    )

    assert patched["valid"] is True
    assert patched["view"]["run"]["name"] == "guided-run"
    assert patched["view"]["translation"]["languages"] == ["ru", "en", "ru"]

    with pytest.raises(UIError, match="embedded credential"):
        service.validate_yaml(profile["yaml"] + "\ngeneration_api_key: visible-secret\n")


def test_drafts_are_isolated_and_identifiers_cannot_traverse(tmp_path: Path) -> None:
    service = _service(tmp_path)
    original = (Path("configs") / "translate_only.yaml").read_text(encoding="utf-8")
    profile = service.profile("translate_only")

    draft = service.save_draft(name="My draft", yaml_text=profile["yaml"])

    assert service.draft(draft["id"])["yaml"]
    assert (Path("configs") / "translate_only.yaml").read_text(encoding="utf-8") == original
    with pytest.raises(UIError, match="invalid draft identifier"):
        service.draft("../outside")
    with pytest.raises(UIError, match="invalid run identifier"):
        service.run_detail("../outside")

    assert service.delete_draft(draft["id"]) == {"deleted": draft["id"]}


def test_launch_requires_the_exact_reviewed_plan(tmp_path: Path) -> None:
    service = _service(tmp_path, worker=True)
    try:
        profile = service.profile("translate_only")
        planned = service.plan(profile["yaml"], "A compact source.")

        with pytest.raises(UIError, match="changed after planning"):
            service.launch(
                yaml_text=profile["yaml"],
                source="A changed source.",
                confirmed_plan_hash=planned["plan_hash"],
            )
    finally:
        service.close()


def test_mock_job_completes_and_results_are_inspectable(tmp_path: Path) -> None:
    service = _service(tmp_path, worker=True)
    try:
        profile = service.profile("translate_only")
        source = "A house becomes a building in the deterministic smoke test."
        planned = service.plan(profile["yaml"], source)
        job = service.launch(
            yaml_text=profile["yaml"],
            source=source,
            confirmed_plan_hash=planned["plan_hash"],
        )

        completed = _wait_for_terminal(service, job["id"])

        assert completed["status"] == "completed"
        runs = service.runs()
        assert runs[0]["state"] == "completed"
        detail = service.run_detail(runs[0]["id"])
        assert detail["source"] == source
        assert detail["final"]
        assert detail["checkpoints"]
        artifact, content_type = service.artifact(runs[0]["id"], "manifest.json")
        assert json.loads(artifact)["state"] == "completed"
        assert content_type == "application/json"
        with pytest.raises(UIError, match="not exposed"):
            service.artifact(runs[0]["id"], "../../.env")
    finally:
        service.close()


def test_active_job_cancellation_is_cooperative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def slow_pipeline(config: Any, *, run_directory: Path) -> Path:
        del config
        run_directory.mkdir(parents=True)
        manifest_path = run_directory / "manifest.json"
        atomic_write_json(
            manifest_path,
            {
                "state": "running",
                "started_at": "now",
                "processed_chunks": [],
                "resolved_config": {"name": "slow"},
            },
        )
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["state"] = "interrupted"
            atomic_write_json(manifest_path, manifest)
            raise
        return run_directory

    monkeypatch.setattr("semantic_telephone.ui.run_pipeline", slow_pipeline)
    service = _service(tmp_path, worker=True)
    try:
        profile = service.profile("translate_only")
        planned = service.plan(profile["yaml"], "Cancellation source.")
        job = service.launch(
            yaml_text=profile["yaml"],
            source="Cancellation source.",
            confirmed_plan_hash=planned["plan_hash"],
        )
        assert service.queue is not None
        for _ in range(200):
            if service.queue.get(job["id"])["status"] == "running":
                break
            time.sleep(0.01)
        queued = service.launch(
            yaml_text=profile["yaml"],
            source="Cancellation source.",
            confirmed_plan_hash=planned["plan_hash"],
        )
        assert service.cancel_job(queued["id"])["status"] == "cancelled"
        service.cancel_job(job["id"])

        interrupted = _wait_for_terminal(service, job["id"])

        assert interrupted["status"] == "interrupted"
        assert service.runs()[0]["state"] == "interrupted"
    finally:
        service.close()


def test_queue_continues_after_a_failed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []

    async def controlled_pipeline(config: Any, *, run_directory: Path) -> Path:
        order.append(config.name)
        await asyncio.sleep(0.05)
        if config.name == "first":
            raise RuntimeError("deliberate failure")
        run_directory.mkdir(parents=True)
        atomic_write_json(
            run_directory / "manifest.json",
            {
                "state": "completed",
                "started_at": "now",
                "processed_chunks": ["chunk"],
                "resolved_config": {"name": config.name},
            },
        )
        return run_directory

    monkeypatch.setattr("semantic_telephone.ui.run_pipeline", controlled_pipeline)
    service = _service(tmp_path, worker=True)
    try:
        profile = service.profile("translate_only")
        first_config = service.patch_yaml(profile["yaml"], {"run.name": "first"})["yaml"]
        second_config = service.patch_yaml(profile["yaml"], {"run.name": "second"})["yaml"]
        source = "Queue recovery source."
        first_plan = service.plan(first_config, source)
        second_plan = service.plan(second_config, source)
        first = service.launch(
            yaml_text=first_config,
            source=source,
            confirmed_plan_hash=first_plan["plan_hash"],
        )
        second = service.launch(
            yaml_text=second_config,
            source=source,
            confirmed_plan_hash=second_plan["plan_hash"],
        )

        assert _wait_for_terminal(service, first["id"])["status"] == "failed"
        assert _wait_for_terminal(service, second["id"])["status"] == "completed"
        assert order == ["first", "second"]
    finally:
        service.close()


def test_partial_run_and_completed_comparison(tmp_path: Path) -> None:
    service = _service(tmp_path)
    partial = service.run_root / "partial"
    partial.mkdir()
    atomic_write_json(
        partial / "manifest.json",
        {
            "state": "running",
            "started_at": "now",
            "processed_chunks": [],
            "resolved_config": {
                "name": "partial",
                "api_key": "must-not-leave-the-server",
                "api_key_env": "OPENROUTER_API_KEY",
            },
        },
    )
    detail = service.run_detail("partial")
    assert detail["final"] == ""
    assert detail["manifest"]["resolved_config"]["api_key"] == "[REDACTED]"
    assert (
        detail["manifest"]["resolved_config"]["api_key_env"]
        == "OPENROUTER_API_KEY"
    )
    artifact, _ = service.artifact("partial", "manifest.json")
    assert "must-not-leave-the-server" not in artifact
    with pytest.raises(UIError, match="not completed"):
        service.compare("partial", "other")


def test_ui_cli_delegates_to_local_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_serve(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("semantic_telephone.ui.serve_console", fake_serve)
    result = runner.invoke(
        app,
        [
            "ui",
            "--no-open",
            "--port",
            "0",
            "--config-root",
            str(tmp_path / "configs"),
            "--run-root",
            str(tmp_path / "runs"),
        ],
    )

    assert result.exit_code == 0
    assert captured["open_browser"] is False
    assert captured["host"] == "127.0.0.1"


def test_ui_assets_are_packaged_and_non_loopback_hosts_are_rejected() -> None:
    root = files("semantic_telephone.resources").joinpath("ui")

    assert root.joinpath("index.html").is_file()
    assert root.joinpath("styles.css").is_file()
    assert root.joinpath("app.js").is_file()
    assert _is_loopback_host("localhost")
    assert _is_loopback_host("127.0.0.1")
    assert not _is_loopback_host("0.0.0.0")
