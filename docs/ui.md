# Research Console

The Research Console is a local web interface for configuring, planning,
running, and inspecting Semantic Telephone experiments.

```bash
semantic-telephone ui
```

The command starts a server on `http://127.0.0.1:8765/` and opens the browser.
Use `--no-open` when a browser should not be opened automatically:

```bash
semantic-telephone ui --no-open
semantic-telephone ui --port 9000 --config-root configs --run-root runs
```

V1 accepts only `localhost` and loopback addresses. It is a single-user local
tool and has no remote hosting or authentication mode.

## Workflow

1. Select a read-only repository or built-in profile.
2. Paste or edit the source text.
3. Use the guided controls, or edit the complete YAML and select
   **Validate & format**.
4. Select **Build plan**. Planning is offline and does not download models.
5. Review language routes, engine candidates, request bounds, downloads, remote
   services, and warnings.
6. Select **Review & launch**, then explicitly confirm the exact reviewed plan.
7. Watch the FIFO queue or inspect the run artifacts and compare completed runs.

Changing either the source or YAML invalidates the previous plan. The server
recomputes it before launch and rejects a stale confirmation.

## Drafts and snapshots

Repository files under `configs/` are never overwritten. Named drafts, input
snapshots, and job metadata are stored in:

```text
.semantic-telephone-ui/
├── drafts/
└── jobs/
```

Each launched job receives its own validated configuration and input snapshot.
Normal run artifacts still use the configured `runs/` directory and remain
compatible with CLI commands such as `inspect`, `resume`, and `report`.

## Queue and cancellation

The console executes one job at a time to avoid competing for RAM, VRAM, API
rate limits, and model caches. Additional jobs wait in FIFO order.

- Cancelling a queued job removes it immediately.
- Cancelling a running job cooperatively cancels the pipeline.
- A run that has started records `interrupted` in its manifest and preserves
  existing checkpoints.
- Closing the browser does not stop the server or active run.
- Stopping the CLI server cancels the active run and cancels queued jobs.

## Safety boundaries

- Mock profiles are labelled smoke tests and are not presented as translation
  quality results.
- `commercial_baseline` is marked unavailable because its provider is still an
  interface stub.
- The launch review reports possible downloads and remote provider calls.
- Doctor warnings are advisory; they do not replace explicit launch
  confirmation.
- API keys remain in `.env` or the process environment. The UI never accepts or
  returns credential values, and embedded credential-like YAML fields are
  rejected.
- Only known artifact paths inside the configured run root can be read through
  the UI.
