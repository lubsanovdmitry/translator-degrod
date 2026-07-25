# Semantic Telephone agent guide

This file applies to the entire repository.

## Product intent

`semantic-telephone` runs reproducible text-degradation experiments across
language routes and, when configured, performs restrained LLM repair,
reconstruction, memory extraction, and report summarization.

Keep these two routes distinct:

- A language route is a sequence such as `ru -> ka -> en -> de -> ru`.
- An engine route selects an MT provider independently for every transition.

Do not optimize every experiment toward the most fluent translator. Differences
between architectures, scripts, entity handling, gender, segmentation, and
literal meanings are part of the experiment.

`mixed_local` is the recommended heterogeneous experiment.
`nllb_only` is the deterministic local MT baseline.

## Mock is only a smoke test

The older configs such as `configs/translate_only.yaml` use mock providers.
They verify orchestration, checkpoints, resume behavior, and artifacts. They do
not translate languages and must never be presented as a meaningful product
run or model-quality result.

For a real run, follow `README.md` and use:

```bash
pip install -e '.[local-mt]'
cp -n .env.example .env
# Fill OPENROUTER_API_KEY and OPENROUTER_MODEL.
semantic-telephone run configs/nllb_only.yaml --verbose
```

The `local-mt` extra installs runtime libraries only. It does not install or
download model weights. `validate` is provider-independent and downloads
nothing; `run` lazily downloads a configured Hugging Face model on its first
actual use.

The first real run needs network access to download Hugging Face models and to
call OpenRouter. CPU fallback is supported but can be slow.

## Architecture map

- `src/semantic_telephone/config.py`
  - Parses legacy single-provider YAML and nested multi-provider YAML.
  - Separates language-route settings from engine-route settings.
  - Rehydrates resolved configuration stored in manifests.
- `src/semantic_telephone/models.py`
  - Dataclasses for configuration, provider results, stages, and manifests.
- `src/semantic_telephone/routes.py`
  - Generates seeded language routes only.
- `src/semantic_telephone/providers/factory.py`
  - Creates generation and translation providers from resolved config.
- `src/semantic_telephone/providers/local_mt.py`
  - Direct Transformers implementations of NLLB and M2M100.
- `src/semantic_telephone/providers/opus_mt.py`
  - Pair-specific Marian models, English hub routing, preflight, and LRU cache.
- `src/semantic_telephone/providers/translation.py`
  - LibreTranslate HTTP client.
- `src/semantic_telephone/providers/router.py`
  - Seeded per-hop engine selection and quality fallback.
- `src/semantic_telephone/providers/openai_compatible.py`
  - OpenAI-compatible client and the explicit OpenRouter provider.
- `src/semantic_telephone/stages/`
  - Stage-level prompt and translation operations.
- `src/semantic_telephone/pipeline.py`
  - Checkpointed orchestration and artifact/manifest recording.
- `configs/`
  - Both mock smoke-test profiles and real MT profiles.
- `prompts/`
  - Generic repair, reconstruction, memory, and report instructions.
- `tests/`
  - Offline tests; they must not download models or call external APIs.

## Provider contracts

Translation providers implement:

```python
async def translate(
    text: str,
    source_language: str,
    target_language: str,
    *,
    seed: int | None = None,
) -> TranslationResult:
    ...
```

Providers used by engine routing should also expose a synchronous
`supports_pair(source, target)` method. Pair-specific or HTTP providers may
expose async preflight methods used by `TranslationProviderRouter`.

Every successful translation result should identify:

- provider type and configured engine alias;
- exact model and revision when available;
- deterministic/sampling status;
- usage;
- segment or internal-leg details;
- server version for HTTP engines when available.

Do not use the Transformers pipeline API for NLLB or M2M100. Use tokenizer and
seq2seq model objects directly. Never pass an input beyond
`max_input_tokens`; preserve segment logs. Keep explicit output bounds and
anti-repetition decoding in real profiles so a single greedy-decoding loop
cannot expand into an unbounded downstream reconstruction prompt.

Keep `torch`, `transformers`, and model imports lazy. The base package and all
tests must work without the `local-mt` extra.

## OpenRouter rules

OpenRouter is for generative tasks, not the main translation layer. Supported
task keys are:

- `conservative_repair`
- `reconstruction`
- `contextual_reconstruction`
- `memory_extraction`
- `report_generation`

Model, temperature, token limit, and additional parameters remain configurable
per task. Do not hard-code an OpenRouter model. Use
`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `OPENROUTER_MODEL` as
credential/default settings.

Treat HTTP 200 as transport success, not proof of usable generation. Inspect
embedded errors, `finish_reason`, and `message.content`. Accept text content
blocks as well as plain strings, but never substitute private reasoning for the
final text. A reasoning-only or length-truncated response is non-retryable with
the same parameters; report safe diagnostics and ask for a larger output budget
or disabled/lower reasoning. Do not assume that `reasoning.effort` is supported
by every configured model, and do not mistake `reasoning.exclude` for disabling
reasoning.

## OPUS-MT download rules

- `allow_downloads: false` means model loading is local-cache-only.
- `allow_downloads: true` permits model download during preflight.
- `configured_pairs_only: true` restricts selection and download to explicit
  `pairs` entries.
- Record exact model names and resolved revisions.
- Keep the number of loaded pair models bounded and move evicted models off
  VRAM.

Do not silently broaden a configured pair table or download an unlisted model
when `configured_pairs_only` is enabled.

## Engine routing invariants

Supported modes:

- `single_engine`
- `fixed_engine_route`
- `weighted_random`
- `alternating`
- `quality_fallback`
- `heterogeneous`

All randomized decisions must derive from the run/stage seed. Weights belong in
YAML and must not be embedded as policy constants. `alternating` avoids a
repeat only when another provider supports the pair. `quality_fallback` records
all failed attempts. `heterogeneous` balances architectures, not merely engine
aliases.

## Artifacts and reproducibility

A completed run writes source/final text, per-stage checkpoints, `events.jsonl`,
metrics, report, resolved config, and manifest. Preserve:

- prompt checksums;
- input/output/stage checksums;
- stage seed and language route;
- actual provider route;
- model/revision metadata;
- all segment or OPUS hub-leg details;
- provider warnings and errors.

Any configuration field that changes output or provider selection must be part
of the resolved config and checkpoint checksum.

Never pass original text into a reconstruction stage intended to see only
damaged text. Memory extraction must operate only after a successful
translation cycle and must validate its JSON schema.

## Commercial providers

`commercial_baseline.yaml` currently points to an explicit interface stub in
the `commercial_nmt` category. Do not claim that it is runnable until a real
client is implemented and tested. Keep commercial results distinguishable from
local and self-hosted MT.

## Development workflow

Use Python 3.12 or newer.

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests
mypy
python -m compileall -q src tests
```

Before handing off a provider or routing change:

1. Validate every real profile.
2. Run the offline test suite.
3. Run Ruff and strict mypy.
4. Confirm tests did not access the network or download a model.
5. Update README when setup, credentials, downloads, or run behavior changes.
6. Label mock instructions as smoke tests, never as the real quick start.

Use `httpx.MockTransport` and fake tokenizers/models in tests. Do not require a
GPU, Hugging Face cache, LibreTranslate server, or API secret in CI.

Preserve unrelated user changes in a dirty worktree. Avoid destructive Git or
filesystem commands.
