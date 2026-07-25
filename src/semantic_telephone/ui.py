from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import re
import threading
import webbrowser
from collections import deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from urllib.parse import unquote, urlparse
from uuid import uuid4

import yaml

from .config import ConfigError, load_config
from .pipeline import run_pipeline
from .planning import create_plan, doctor_config
from .resources import profile_names, profile_text
from .runtime import safe_diagnostic
from .utils.files import atomic_write_json, atomic_write_text, read_json

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_SOURCE_CHARACTERS = 1_000_000
MAX_CONFIG_CHARACTERS = 500_000
TOP_LEVEL_ARTIFACTS = {
    "events.jsonl",
    "final.txt",
    "manifest.json",
    "metrics.json",
    "report.md",
    "resolved_config.yaml",
    "source.txt",
}
CHUNK_ARTIFACT = re.compile(
    r"chunks/\d{4}/(?:"
    r"source\.txt|context\.txt|final\.txt|"
    r"stage-[a-zA-Z0-9._-]+\.(?:json|txt)|"
    r"stage-[a-zA-Z0-9._-]+-output\.txt|"
    r"stage-[a-zA-Z0-9._-]+-hop-\d+-output\.txt"
    r")"
)
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}")
SENSITIVE_KEY = re.compile(
    r"(?:.*api[_-]?key|access[_-]?token|auth[_-]?token|bearer[_-]?token|"
    r"token|.*secret|.*password|.*credential)",
    re.IGNORECASE,
)
SAFE_SENSITIVE_KEYS = {
    "api_key_env",
    "openrouter_api_key_env",
    "libretranslate_api_key_env",
    "max_total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
}
GUIDED_FIELDS: dict[str, type[Any]] = {
    "run.name": str,
    "run.seed": int,
    "run.source_language": str,
    "run.target_language": str,
    "chunking.strategy": str,
    "chunking.target_chars": int,
    "chunking.max_chars": int,
    "translation.route_mode": str,
    "translation.languages": list,
    "context.enabled": bool,
    "memory.enabled": bool,
    "runtime.concurrency": int,
    "runtime.requests_per_minute": int,
    "runtime.retries": int,
    "runtime.budgets.max_requests": int,
    "runtime.budgets.max_total_tokens": int,
    "runtime.budgets.max_cost_usd": float,
    "pipeline": list,
}


class UIError(ValueError):
    def __init__(self, message: str, *, status: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "-" for character in value
    ).strip("-")
    return normalized[:64] or "run"


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, OSError):
        return {}


def _object(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _yaml_object(yaml_text: str) -> dict[str, Any]:
    if len(yaml_text) > MAX_CONFIG_CHARACTERS:
        raise UIError("configuration exceeds the 500,000 character limit")
    try:
        value = yaml.safe_load(yaml_text)
    except yaml.YAMLError as error:
        raise UIError(f"invalid YAML: {safe_diagnostic(error)}") from error
    if not isinstance(value, dict):
        raise UIError("configuration root must be a mapping")
    _reject_embedded_secrets(value)
    return cast(dict[str, Any], value)


def _reject_embedded_secrets(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            if (
                SENSITIVE_KEY.fullmatch(key)
                and key.lower() not in SAFE_SENSITIVE_KEYS
                and item not in (None, "")
            ):
                location = ".".join((*path, key))
                raise UIError(
                    f"embedded credential field '{location}' is not accepted; "
                    "use an environment-variable reference"
                )
            _reject_embedded_secrets(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_embedded_secrets(item, (*path, str(index)))


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if SENSITIVE_KEY.fullmatch(str(key))
                and str(key).lower() not in SAFE_SENSITIVE_KEYS
                else _redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _portable_profile(yaml_text: str, *, base_directory: Path, run_root: Path) -> str:
    raw = _yaml_object(yaml_text)
    input_section = raw.setdefault("input", {})
    if not isinstance(input_section, dict):
        raise UIError("'input' must be a mapping")
    input_section["path"] = "input.txt"
    run = raw.setdefault("run", {})
    if not isinstance(run, dict):
        raise UIError("'run' must be a mapping")
    run["output_root"] = str(run_root)
    prompts = raw.get("prompts")
    if isinstance(prompts, dict):
        for key, value in list(prompts.items()):
            text = str(value)
            if text.startswith("builtin:") or Path(text).is_absolute():
                continue
            prompts[key] = str((base_directory / text).resolve())
    return yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)


def _config_view(raw: dict[str, Any]) -> dict[str, Any]:
    run = _object(raw.get("run"))
    chunking = _object(raw.get("chunking"))
    translation = _object(raw.get("translation"))
    context = _object(raw.get("context"))
    memory = _object(raw.get("memory"))
    runtime = _object(raw.get("runtime"))
    budgets = _object(runtime.get("budgets"))
    pipeline = raw.get("pipeline") if isinstance(raw.get("pipeline"), list) else []
    return {
        "run": {
            "name": run.get("name", ""),
            "seed": run.get("seed", 0),
            "source_language": run.get("source_language", "auto"),
            "target_language": run.get(
                "target_language", run.get("source_language", "en")
            ),
        },
        "chunking": {
            "strategy": chunking.get("strategy", "target_chars"),
            "target_chars": chunking.get("target_chars", 1100),
            "max_chars": chunking.get("max_chars", 2200),
        },
        "translation": {
            "route_mode": translation.get("route_mode", "fixed"),
            "languages": (
                translation.get("languages")
                if isinstance(translation.get("languages"), list)
                else []
            ),
        },
        "context": {"enabled": bool(context.get("enabled", False))},
        "memory": {"enabled": bool(memory.get("enabled", False))},
        "runtime": {
            "concurrency": runtime.get("concurrency", 1),
            "requests_per_minute": runtime.get("requests_per_minute", 30),
            "retries": runtime.get("retries", 4),
            "budgets": {
                "max_requests": budgets.get("max_requests"),
                "max_total_tokens": budgets.get("max_total_tokens"),
                "max_cost_usd": budgets.get("max_cost_usd"),
            },
        },
        "pipeline": pipeline,
    }


def _set_nested(raw: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = raw
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    if value is None:
        current.pop(parts[-1], None)
    else:
        current[parts[-1]] = value


def _validate_patch(fields: dict[str, Any]) -> None:
    unknown = sorted(set(fields) - set(GUIDED_FIELDS))
    if unknown:
        raise UIError("unsupported guided fields: " + ", ".join(unknown))
    for name, value in fields.items():
        if value is None and (
            name.startswith("runtime.budgets.")
            or name == "runtime.requests_per_minute"
        ):
            continue
        expected = GUIDED_FIELDS[name]
        if expected is int:
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif expected is float:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            valid = isinstance(value, expected)
        if not valid:
            raise UIError(f"guided field '{name}' has an invalid value")


def _plan_hash(yaml_text: str, source: str, plan: dict[str, Any]) -> str:
    payload = json.dumps(
        {"yaml": yaml_text, "source": source, "plan": plan},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _translation_provider_types(raw: dict[str, Any]) -> set[str]:
    translation = _object(raw.get("translation"))
    providers = _object(translation.get("providers"))
    result = {
        str(_object(item).get("type", _object(item).get("provider", "mock")))
        for item in providers.values()
        if isinstance(item, dict)
    }
    if not result:
        result.add(str(translation.get("type", translation.get("provider", "mock"))))
    return result


@dataclass(slots=True)
class ConsoleJob:
    id: str
    name: str
    status: str
    created_at: str
    config_path: str
    input_path: str
    run_directory: str
    plan: dict[str, Any]
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "run_directory": self.run_directory,
            "plan": self.plan,
        }

    def persisted(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "config_path": self.config_path,
            "input_path": self.input_path,
        }


async def _run_cancellable(
    config_path: str,
    input_path: str,
    run_root: Path,
    run_directory: Path,
    cancel_signal: threading.Event,
) -> Path:
    config = load_config(config_path)
    config = replace(config, input_path=input_path, output_root=str(run_root))
    pipeline_task = asyncio.create_task(
        run_pipeline(config, run_directory=run_directory)
    )
    # Let the pipeline establish its run directory and initial manifest before
    # observing cancellation so every job reported as running is resumable.
    await asyncio.sleep(0)
    while not pipeline_task.done():
        if cancel_signal.is_set():
            pipeline_task.cancel()
        await asyncio.wait({pipeline_task}, timeout=0.1)
    return await pipeline_task


class RunQueue:
    def __init__(self, state_root: Path, run_root: Path) -> None:
        self.state_root = state_root
        self.run_root = run_root
        self.jobs_root = state_root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, ConsoleJob] = {}
        self._pending: deque[str] = deque()
        self._condition = threading.Condition()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._active_task: asyncio.Task[Path] | None = None
        self._active_cancel: threading.Event | None = None
        self._active_id: str | None = None
        self._closing = False
        self._restore_jobs()
        self._thread = threading.Thread(
            target=self._worker, name="semantic-telephone-ui-worker", daemon=True
        )
        self._thread.start()

    def _restore_jobs(self) -> None:
        for path in sorted(self.jobs_root.glob("*/job.json")):
            raw = _read_json_object(path)
            try:
                status = str(raw["status"])
                if status in {"queued", "running"}:
                    status = "interrupted"
                    raw["finished_at"] = utc_now()
                    raw["error"] = "UI server stopped before the job completed"
                job = ConsoleJob(
                    id=str(raw["id"]),
                    name=str(raw["name"]),
                    status=status,
                    created_at=str(raw["created_at"]),
                    config_path=str(raw["config_path"]),
                    input_path=str(raw["input_path"]),
                    run_directory=str(raw["run_directory"]),
                    plan=cast(dict[str, Any], raw.get("plan", {})),
                    started_at=cast(str | None, raw.get("started_at")),
                    finished_at=cast(str | None, raw.get("finished_at")),
                    error=cast(str | None, raw.get("error")),
                    cancel_requested=bool(raw.get("cancel_requested", False)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self._jobs[job.id] = job
            self._persist(job)

    def enqueue(
        self,
        *,
        job_id: str,
        name: str,
        config_path: Path,
        input_path: Path,
        run_directory: Path,
        plan: dict[str, Any],
    ) -> ConsoleJob:
        job = ConsoleJob(
            id=job_id,
            name=name,
            status="queued",
            created_at=utc_now(),
            config_path=str(config_path),
            input_path=str(input_path),
            run_directory=str(run_directory),
            plan=plan,
        )
        with self._condition:
            if self._closing:
                raise UIError("run queue is shutting down", status=HTTPStatus.SERVICE_UNAVAILABLE)
            self._jobs[job.id] = job
            self._pending.append(job.id)
            self._persist(job)
            self._condition.notify()
        return job

    def list(self) -> list[dict[str, Any]]:
        with self._condition:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [self._public(job) for job in jobs]

    def get(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise UIError("job not found", status=HTTPStatus.NOT_FOUND)
            return self._public(job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise UIError("job not found", status=HTTPStatus.NOT_FOUND)
            if job.status == "queued":
                try:
                    self._pending.remove(job_id)
                except ValueError:
                    pass
                job.status = "cancelled"
                job.cancel_requested = True
                job.finished_at = utc_now()
                self._persist(job)
            elif job.status == "running":
                job.cancel_requested = True
                self._persist(job)
                if (
                    self._active_cancel is not None
                    and self._active_id == job_id
                ):
                    self._active_cancel.set()
            return self._public(job)

    def close(self) -> None:
        with self._condition:
            self._closing = True
            while self._pending:
                pending_id = self._pending.popleft()
                pending = self._jobs[pending_id]
                pending.status = "cancelled"
                pending.cancel_requested = True
                pending.finished_at = utc_now()
                self._persist(pending)
            if self._active_cancel is not None:
                self._active_cancel.set()
            self._condition.notify_all()
        self._thread.join(timeout=15)

    def _worker(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._condition:
            self._loop = loop
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._closing:
                        self._condition.wait()
                    if self._closing and not self._pending:
                        return
                    job_id = self._pending.popleft()
                    job = self._jobs[job_id]
                try:
                    with self._condition:
                        if job.status == "cancelled":
                            continue
                        cancel_signal = threading.Event()
                        task = loop.create_task(
                            _run_cancellable(
                                job.config_path,
                                job.input_path,
                                self.run_root,
                                Path(job.run_directory),
                                cancel_signal,
                            )
                        )
                        job.status = "running"
                        job.started_at = utc_now()
                        self._active_id = job_id
                        self._active_task = task
                        self._active_cancel = cancel_signal
                        self._persist(job)
                        if job.cancel_requested:
                            cancel_signal.set()
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    job.status = "interrupted"
                    job.error = "cancelled by user"
                except BaseException as error:  # noqa: BLE001 - worker boundary
                    job.status = "failed"
                    job.error = safe_diagnostic(error)
                else:
                    job.status = "completed"
                finally:
                    with self._condition:
                        job.finished_at = utc_now()
                        self._active_task = None
                        self._active_cancel = None
                        self._active_id = None
                        self._persist(job)
        finally:
            loop.close()
            with self._condition:
                self._loop = None

    def _public(self, job: ConsoleJob) -> dict[str, Any]:
        value = job.to_dict()
        manifest = _read_json_object(Path(job.run_directory) / "manifest.json")
        processed = manifest.get("processed_chunks")
        value["progress"] = {
            "processed": len(processed) if isinstance(processed, list) else 0,
            "total": job.plan.get("chunks", 0),
        }
        value["manifest_state"] = manifest.get("state")
        return value

    def _persist(self, job: ConsoleJob) -> None:
        destination = self.jobs_root / job.id / "job.json"
        atomic_write_json(destination, job.persisted())


class ConsoleService:
    def __init__(
        self,
        *,
        project_root: Path,
        config_root: Path,
        run_root: Path,
        state_root: Path | None = None,
        start_worker: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config_root = config_root.resolve()
        self.run_root = run_root.resolve()
        self.state_root = (
            state_root.resolve()
            if state_root is not None
            else self.project_root / ".semantic-telephone-ui"
        )
        self.drafts_root = self.state_root / "drafts"
        self.drafts_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.queue = (
            RunQueue(self.state_root, self.run_root) if start_worker else None
        )

    def close(self) -> None:
        if self.queue is not None:
            self.queue.close()

    def bootstrap(self) -> dict[str, Any]:
        return {
            "profiles": self.profiles(),
            "drafts": self.drafts(),
            "jobs": self.queue.list() if self.queue is not None else [],
            "runs": self.runs(),
            "source": self._read_text_optional(self.project_root / "input.txt"),
            "limits": {
                "request_bytes": MAX_REQUEST_BYTES,
                "source_characters": MAX_SOURCE_CHARACTERS,
                "config_characters": MAX_CONFIG_CHARACTERS,
            },
        }

    def profiles(self) -> list[dict[str, Any]]:
        names = set(profile_names())
        if self.config_root.exists():
            names.update(
                path.stem
                for path in self.config_root.glob("*.yaml")
                if "matrix" not in path.stem
            )
        return [self._profile_info(name) for name in sorted(names)]

    def profile(self, name: str) -> dict[str, Any]:
        self._identifier(name, "profile")
        info = self._profile_info(name)
        yaml_text = self._profile_yaml(name)
        validation = self.validate_yaml(yaml_text)
        return {**info, "yaml": yaml_text, **validation}

    def validate_yaml(self, yaml_text: str) -> dict[str, Any]:
        raw = _yaml_object(yaml_text)
        canonical = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
        with TemporaryDirectory(prefix="semantic-telephone-ui-") as temporary:
            root = Path(temporary)
            atomic_write_text(root / "input.txt", "validation input")
            materialized = self._materialized_yaml(raw, root / "input.txt")
            config_path = root / "config.yaml"
            atomic_write_text(config_path, materialized)
            try:
                config = load_config(config_path)
            except (ConfigError, OSError, ValueError) as error:
                raise UIError(safe_diagnostic(error)) from error
        return {
            "valid": True,
            "yaml": canonical,
            "view": _config_view(raw),
            "name": config.name,
        }

    def patch_yaml(self, yaml_text: str, fields: dict[str, Any]) -> dict[str, Any]:
        _validate_patch(fields)
        raw = _yaml_object(yaml_text)
        for name, value in fields.items():
            _set_nested(raw, name, value)
        updated = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
        return self.validate_yaml(updated)

    def plan(self, yaml_text: str, source: str) -> dict[str, Any]:
        self._validate_source(source)
        raw = _yaml_object(yaml_text)
        with TemporaryDirectory(prefix="semantic-telephone-ui-") as temporary:
            root = Path(temporary)
            input_path = root / "input.txt"
            atomic_write_text(input_path, source)
            config_path = root / "config.yaml"
            materialized = self._materialized_yaml(raw, input_path)
            atomic_write_text(config_path, materialized)
            try:
                result = create_plan(load_config(config_path))
            except (ConfigError, OSError, RuntimeError, ValueError) as error:
                raise UIError(safe_diagnostic(error)) from error
        return {"plan": result, "plan_hash": _plan_hash(yaml_text, source, result)}

    def doctor(self, yaml_text: str, source: str) -> dict[str, Any]:
        self._validate_source(source)
        raw = _yaml_object(yaml_text)
        with TemporaryDirectory(prefix="semantic-telephone-ui-") as temporary:
            root = Path(temporary)
            input_path = root / "input.txt"
            atomic_write_text(input_path, source)
            config_path = root / "config.yaml"
            atomic_write_text(config_path, self._materialized_yaml(raw, input_path))
            try:
                result = asyncio.run(doctor_config(load_config(config_path)))
            except (ConfigError, OSError, RuntimeError, ValueError) as error:
                raise UIError(safe_diagnostic(error)) from error
        return {"doctor": result}

    def save_draft(
        self, *, name: str, yaml_text: str, draft_id: str | None = None
    ) -> dict[str, Any]:
        validation = self.validate_yaml(yaml_text)
        identifier = draft_id or f"{_slug(name)}-{uuid4().hex[:8]}"
        self._identifier(identifier, "draft")
        path = self.drafts_root / f"{identifier}.json"
        existing = _read_json_object(path)
        now = utc_now()
        document = {
            "id": identifier,
            "name": name.strip() or str(validation["name"]),
            "yaml": validation["yaml"],
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
        atomic_write_json(path, document)
        return document

    def drafts(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in self.drafts_root.glob("*.json"):
            value = _read_json_object(path)
            if value:
                result.append({key: value.get(key) for key in ("id", "name", "updated_at")})
        return sorted(result, key=lambda item: str(item.get("updated_at", "")), reverse=True)

    def draft(self, draft_id: str) -> dict[str, Any]:
        path = self._draft_path(draft_id)
        value = _read_json_object(path)
        if not value:
            raise UIError("draft not found", status=HTTPStatus.NOT_FOUND)
        return value

    def delete_draft(self, draft_id: str) -> dict[str, Any]:
        path = self._draft_path(draft_id)
        if not path.exists():
            raise UIError("draft not found", status=HTTPStatus.NOT_FOUND)
        path.unlink()
        return {"deleted": draft_id}

    def launch(
        self, *, yaml_text: str, source: str, confirmed_plan_hash: str
    ) -> dict[str, Any]:
        if self.queue is None:
            raise UIError("run queue is unavailable", status=HTTPStatus.SERVICE_UNAVAILABLE)
        planned = self.plan(yaml_text, source)
        if not confirmed_plan_hash or confirmed_plan_hash != planned["plan_hash"]:
            raise UIError(
                "configuration or source changed after planning; review the new plan",
                status=HTTPStatus.CONFLICT,
            )
        raw = _yaml_object(yaml_text)
        if "commercial_nmt" in _translation_provider_types(raw):
            raise UIError(
                "commercial_baseline is an interface stub and cannot be launched",
                status=HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        job_id = uuid4().hex
        job_root = self.state_root / "jobs" / job_id
        input_path = job_root / "input.txt"
        config_path = job_root / "config.yaml"
        atomic_write_text(input_path, source)
        atomic_write_text(config_path, self._materialized_yaml(raw, input_path))
        config = load_config(config_path)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_directory = self.run_root / (
            f"{timestamp}-{_slug(config.name)}-{job_id[:8]}"
        )
        job = self.queue.enqueue(
            job_id=job_id,
            name=config.name,
            config_path=config_path,
            input_path=input_path,
            run_directory=run_directory,
            plan=cast(dict[str, Any], planned["plan"]),
        )
        return self.queue.get(job.id)

    def jobs(self) -> list[dict[str, Any]]:
        return self.queue.list() if self.queue is not None else []

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        self._identifier(job_id, "job")
        if self.queue is None:
            raise UIError("run queue is unavailable", status=HTTPStatus.SERVICE_UNAVAILABLE)
        return self.queue.cancel(job_id)

    def runs(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if not self.run_root.exists():
            return result
        paths = sorted(
            (
                path
                for path in self.run_root.iterdir()
                if path.is_dir() and (path / "manifest.json").exists()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths[:200]:
            manifest = _read_json_object(path / "manifest.json")
            resolved = _object(manifest.get("resolved_config"))
            result.append(
                {
                    "id": path.name,
                    "name": resolved.get("name", path.name),
                    "state": manifest.get("state", "unknown"),
                    "started_at": manifest.get("started_at"),
                    "completed_at": manifest.get("completed_at"),
                    "seed": manifest.get("seed"),
                    "processed_chunks": len(manifest.get("processed_chunks", []))
                    if isinstance(manifest.get("processed_chunks"), list)
                    else 0,
                    "metrics": manifest.get("metrics", {}),
                    "error": manifest.get("last_error"),
                }
            )
        return result

    def run_detail(self, run_id: str) -> dict[str, Any]:
        root = self._run_path(run_id)
        manifest = _read_json_object(root / "manifest.json")
        if not manifest:
            raise UIError("run manifest is unavailable", status=HTTPStatus.NOT_FOUND)
        checkpoints: list[dict[str, Any]] = []
        chunks_root = root / "chunks"
        if chunks_root.exists():
            for chunk in sorted(path for path in chunks_root.iterdir() if path.is_dir()):
                stages: list[dict[str, Any]] = []
                for path in sorted(chunk.glob("stage-*.json")):
                    metadata = _read_json_object(path)
                    output_path = path.with_name(path.stem + "-output.txt")
                    stages.append(
                        {
                            "file": str(path.relative_to(root)),
                            "stage_type": metadata.get("stage_type"),
                            "provider": metadata.get("provider"),
                            "model": metadata.get("model"),
                            "route": metadata.get("route", []),
                            "provider_route": metadata.get("provider_route", []),
                            "duration_seconds": metadata.get("duration_seconds"),
                            "warnings": metadata.get("warnings", []),
                            "error": metadata.get("error"),
                            "output": self._read_text_optional(output_path),
                        }
                    )
                checkpoints.append({"chunk": chunk.name, "stages": stages})
        return {
            "id": run_id,
            "manifest": _redact_secrets(manifest),
            "source": self._read_text_optional(root / "source.txt"),
            "final": self._read_text_optional(root / "final.txt"),
            "metrics": _redact_secrets(_read_json_object(root / "metrics.json")),
            "report": self._read_text_optional(root / "report.md"),
            "checkpoints": checkpoints,
        }

    def artifact(self, run_id: str, artifact: str) -> tuple[str, str]:
        root = self._run_path(run_id)
        normalized = unquote(artifact).strip("/")
        if normalized not in TOP_LEVEL_ARTIFACTS and not CHUNK_ARTIFACT.fullmatch(
            normalized
        ):
            raise UIError("artifact is not exposed", status=HTTPStatus.NOT_FOUND)
        path = (root / normalized).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise UIError("artifact not found", status=HTTPStatus.NOT_FOUND)
        content_type = (
            "application/json"
            if path.suffix in {".json", ".jsonl"}
            else "text/plain; charset=utf-8"
        )
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            try:
                text = json.dumps(
                    _redact_secrets(json.loads(text)),
                    ensure_ascii=False,
                    indent=2,
                )
            except json.JSONDecodeError:
                pass
        elif path.suffix == ".jsonl":
            redacted_lines: list[str] = []
            for line in text.splitlines():
                try:
                    redacted_lines.append(
                        json.dumps(
                            _redact_secrets(json.loads(line)), ensure_ascii=False
                        )
                    )
                except json.JSONDecodeError:
                    redacted_lines.append(line)
            text = "\n".join(redacted_lines) + ("\n" if text.endswith("\n") else "")
        elif path.suffix in {".yaml", ".yml"}:
            try:
                text = yaml.safe_dump(
                    _redact_secrets(yaml.safe_load(text)),
                    allow_unicode=True,
                    sort_keys=False,
                )
            except yaml.YAMLError:
                pass
        return text, content_type

    def compare(self, first: str, second: str) -> dict[str, Any]:
        if first == second:
            raise UIError("choose two different runs")
        left = self.run_detail(first)
        if left["manifest"].get("state") != "completed":
            raise UIError(f"run '{first}' is not completed")
        right = self.run_detail(second)
        if right["manifest"].get("state") != "completed":
            raise UIError(f"run '{second}' is not completed")
        return {"left": left, "right": right}

    def _profile_yaml(self, name: str) -> str:
        repository_path = self.config_root / f"{name}.yaml"
        if repository_path.is_file():
            return _portable_profile(
                repository_path.read_text(encoding="utf-8"),
                base_directory=repository_path.parent,
                run_root=self.run_root,
            )
        try:
            return _portable_profile(
                profile_text(name, standalone=True),
                base_directory=self.project_root,
                run_root=self.run_root,
            )
        except FileNotFoundError as error:
            raise UIError("profile not found", status=HTTPStatus.NOT_FOUND) from error

    def _profile_info(self, name: str) -> dict[str, Any]:
        yaml_text = self._profile_yaml(name)
        raw = _yaml_object(yaml_text)
        provider_types = _translation_provider_types(raw)
        commercial = name == "commercial_baseline" or "commercial_nmt" in provider_types
        mock = provider_types == {"mock"}
        return {
            "id": name,
            "name": name.replace("_", " ").title(),
            "kind": "unavailable" if commercial else ("mock" if mock else "real"),
            "runnable": not commercial,
            "note": (
                "Interface stub; a commercial client is not implemented."
                if commercial
                else (
                    "Smoke test only; it does not translate languages."
                    if mock
                    else "Real experiment; it may download models or call remote services."
                )
            ),
        }

    def _materialized_yaml(self, raw: dict[str, Any], input_path: Path) -> str:
        copied = cast(dict[str, Any], json.loads(json.dumps(raw)))
        input_section = copied.setdefault("input", {})
        if not isinstance(input_section, dict):
            raise UIError("'input' must be a mapping")
        input_section["path"] = str(input_path)
        run = copied.setdefault("run", {})
        if not isinstance(run, dict):
            raise UIError("'run' must be a mapping")
        run["output_root"] = str(self.run_root)
        return yaml.safe_dump(copied, allow_unicode=True, sort_keys=False)

    def _draft_path(self, draft_id: str) -> Path:
        self._identifier(draft_id, "draft")
        return self.drafts_root / f"{draft_id}.json"

    def _run_path(self, run_id: str) -> Path:
        self._identifier(run_id, "run")
        path = (self.run_root / run_id).resolve()
        if not path.is_relative_to(self.run_root) or not path.is_dir():
            raise UIError("run not found", status=HTTPStatus.NOT_FOUND)
        return path

    @staticmethod
    def _identifier(value: str, kind: str) -> None:
        if not IDENTIFIER.fullmatch(value):
            raise UIError(f"invalid {kind} identifier")

    @staticmethod
    def _validate_source(source: str) -> None:
        if len(source) > MAX_SOURCE_CHARACTERS:
            raise UIError("source exceeds the 1,000,000 character limit")
        if not source.strip():
            raise UIError("source text is empty")

    @staticmethod
    def _read_text_optional(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, UnicodeError):
            return ""


class ConsoleHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        service: ConsoleService,
    ) -> None:
        self.service = service
        super().__init__(server_address, ConsoleRequestHandler)


class ConsoleRequestHandler(BaseHTTPRequestHandler):
    server: ConsoleHTTPServer

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _dispatch(self, method: str) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if not path.startswith("/api/"):
                self._serve_static(path)
                return
            result = self._api(method, path)
            self._send_json(result)
        except _ResponseSent:
            return
        except UIError as error:
            self._send_json({"error": str(error)}, status=error.status)
        except Exception as error:  # noqa: BLE001 - HTTP boundary
            self._send_json(
                {"error": safe_diagnostic(error)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _api(self, method: str, path: str) -> dict[str, Any] | list[dict[str, Any]]:
        service = self.server.service
        if method == "GET" and path == "/api/bootstrap":
            return service.bootstrap()
        if method == "GET" and path == "/api/jobs":
            return service.jobs()
        if method == "GET" and path == "/api/runs":
            return service.runs()
        if method == "GET" and path.startswith("/api/profiles/"):
            return service.profile(unquote(path.removeprefix("/api/profiles/")))
        if method == "GET" and path == "/api/drafts":
            return service.drafts()
        if method == "GET" and path.startswith("/api/drafts/"):
            return service.draft(unquote(path.removeprefix("/api/drafts/")))
        if method == "DELETE" and path.startswith("/api/drafts/"):
            return service.delete_draft(unquote(path.removeprefix("/api/drafts/")))
        if method == "GET" and path.startswith("/api/runs/"):
            rest = path.removeprefix("/api/runs/")
            run_id, separator, artifact = rest.partition("/artifacts/")
            if separator:
                text, content_type = service.artifact(unquote(run_id), artifact)
                self._send_text(text, content_type=content_type)
                raise _ResponseSent()
            return service.run_detail(unquote(run_id))
        body = self._json_body()
        if method == "POST" and path == "/api/config/validate":
            return service.validate_yaml(self._string(body, "yaml"))
        if method == "POST" and path == "/api/config/patch":
            fields = body.get("fields")
            if not isinstance(fields, dict):
                raise UIError("'fields' must be an object")
            return service.patch_yaml(self._string(body, "yaml"), fields)
        if method == "POST" and path == "/api/plan":
            return service.plan(
                self._string(body, "yaml"), self._string(body, "source")
            )
        if method == "POST" and path == "/api/doctor":
            return service.doctor(
                self._string(body, "yaml"), self._string(body, "source")
            )
        if method == "POST" and path == "/api/drafts":
            draft_id = body.get("id")
            return service.save_draft(
                name=self._string(body, "name"),
                yaml_text=self._string(body, "yaml"),
                draft_id=str(draft_id) if draft_id is not None else None,
            )
        if method == "POST" and path == "/api/jobs":
            return service.launch(
                yaml_text=self._string(body, "yaml"),
                source=self._string(body, "source"),
                confirmed_plan_hash=self._string(body, "plan_hash"),
            )
        if method == "POST" and path.startswith("/api/jobs/") and path.endswith(
            "/cancel"
        ):
            identifier = path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
            return service.cancel_job(unquote(identifier))
        if method == "POST" and path == "/api/compare":
            return service.compare(
                self._string(body, "first"), self._string(body, "second")
            )
        raise UIError("endpoint not found", status=HTTPStatus.NOT_FOUND)

    def _json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as error:
            raise UIError("invalid Content-Length") from error
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise UIError(
                "request body is too large", status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            )
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UIError("request body must be valid UTF-8 JSON") from error
        if not isinstance(value, dict):
            raise UIError("request body must be a JSON object")
        return cast(dict[str, Any], value)

    def _serve_static(self, path: str) -> None:
        names = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        item = names.get(path)
        if item is None:
            raise UIError("page not found", status=HTTPStatus.NOT_FOUND)
        name, content_type = item
        resource = files("semantic_telephone.resources").joinpath("ui", name)
        self._send_bytes(resource.read_bytes(), content_type=content_type)

    def _send_json(
        self,
        value: dict[str, Any] | list[dict[str, Any]],
        *,
        status: int = HTTPStatus.OK,
    ) -> None:
        data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self._send_bytes(data, content_type="application/json; charset=utf-8", status=status)

    def _send_text(self, value: str, *, content_type: str) -> None:
        self._send_bytes(value.encode("utf-8"), content_type=content_type)

    def _send_bytes(
        self,
        value: bytes,
        *,
        content_type: str,
        status: int = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(value)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(value)

    @staticmethod
    def _string(body: dict[str, Any], key: str) -> str:
        value = body.get(key)
        if not isinstance(value, str):
            raise UIError(f"'{key}' must be a string")
        return value


class _ResponseSent(Exception):
    pass


def serve_console(
    *,
    project_root: Path,
    config_root: Path,
    run_root: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    if not _is_loopback_host(host):
        raise UIError("V1 may only bind to localhost or a loopback IP address")
    service = ConsoleService(
        project_root=project_root,
        config_root=config_root,
        run_root=run_root,
    )
    server = ConsoleHTTPServer((host, port), service)
    address, actual_port = cast(tuple[str, int], server.server_address[:2])
    display_host = "127.0.0.1" if address in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{actual_port}/"
    print(f"Semantic Telephone Research Console: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping Semantic Telephone Research Console")
    finally:
        server.server_close()
        service.close()
