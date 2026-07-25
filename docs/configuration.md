# Конфигурация

Конфигурация запуска — YAML. Относительные пути считаются от каталога, где
лежит YAML.

Проверка без сети:

```bash
semantic-telephone validate CONFIG
semantic-telephone plan CONFIG
```

## Минимальный пример

```yaml
run:
  name: example
  seed: 1080
  source_language: ru
  target_language: ru
  output_root: runs

input:
  path: input.txt
  encoding: utf-8

chunking:
  strategy: target_chars
  target_chars: 900
  max_chars: 1800

translation:
  provider: mock
  route_mode: fixed
  languages: [ru, ka, en, ru]

generation:
  provider: mock

context: {enabled: false}
memory: {enabled: false}

pipeline:
  - type: translation_cycle
  - type: final_translation

runtime:
  retries: 4
  retry_backoff_seconds: 2
  failure_policy: stop
```

Этот пример использует mock и не переводит языки.

## `run` и `input`

```yaml
run:
  name: experiment-name
  seed: 1080
  source_language: ru
  target_language: ru
  output_root: runs

input:
  path: input.txt
  encoding: utf-8
```

- `name` входит в имя каталога результата.
- `seed` управляет языковым маршрутом, engine routing и вероятностными stages.
- `source_language` и `target_language` — начальный и конечный языки цикла.
- `output_root` — каталог запусков.

## Chunking

Доступные стратегии:

- `paragraph`;
- `target_chars`;
- `target_tokens`;
- `sentence_window`.

```yaml
chunking:
  strategy: target_chars
  target_chars: 900
  max_chars: 1800
  paragraph_overlap: 0
```

`paragraph_overlap` сохраняется как отдельный `context.txt` и не дублируется в
`final.txt`.

## Языковой маршрут

```yaml
translation:
  route_mode: fixed
  languages: [ru, ka, en, de, ru]
  min_hops: 4
  max_hops: 7
```

Режимы:

- `fixed`;
- `random`;
- `stratified`;
- `hubbed`;
- `mutating_fixed`.

`allow` и `deny` ограничивают промежуточные языки. Соседние одинаковые языки
удаляются. Конечный язык добавляется автоматически.

## Один MT-провайдер

```yaml
translation:
  default_provider: nllb
  route_mode: fixed
  languages: [ru, ka, en, ru]
  providers:
    nllb:
      type: nllb
      model: facebook/nllb-200-distilled-600M
      revision: null
      device: auto
      dtype: auto
      max_input_tokens: 450
      decoding:
        mode: greedy
        num_beams: 1
        max_new_tokens: 512
        no_repeat_ngram_size: 3
  engine_routing:
    mode: single_engine
    provider: nllb
```

`device: auto` выбирает CUDA при наличии, иначе CPU. Локальные провайдеры
используют tokenizer и seq2seq model напрямую.

## Несколько MT-провайдеров

```yaml
translation:
  default_provider: nllb
  providers:
    nllb:
      type: nllb
      model: facebook/nllb-200-distilled-600M
    m2m100:
      type: m2m100
      model: facebook/m2m100_418M
    opus:
      type: opus_mt
      allow_downloads: true
      configured_pairs_only: true
      pairs:
        ru-en: Helsinki-NLP/opus-mt-ru-en
        en-ru: Helsinki-NLP/opus-mt-en-ru
  engine_routing:
    mode: heterogeneous
    avoid_same_engine_consecutively: true
    weights: {nllb: 0.5, m2m100: 0.3, opus: 0.2}
    fallback_order: [nllb, m2m100, opus]
```

Engine routing выбирает провайдера отдельно для каждого перехода языкового
маршрута.

Режимы engine routing:

- `single_engine`;
- `fixed_engine_route`;
- `weighted_random`;
- `alternating`;
- `quality_fallback`;
- `heterogeneous`.

Случайный выбор всегда выводится из stage seed. Провайдер выбирается только
если поддерживает языковую пару.

## Generation и OpenRouter

OpenRouter используется для repair, reconstruction, memory и отчёта, но не
как основной MT-слой.

```yaml
generation:
  provider: openrouter
  api_key_env: OPENROUTER_API_KEY
  parameters:
    reasoning: {enabled: false}
  tasks:
    conservative_repair:
      temperature: 0.1
      max_tokens: 2048
    reconstruction:
      temperature: 0.35
      max_tokens: 4096
    memory_extraction:
      temperature: 0.1
      max_tokens: 2048
    report_generation:
      temperature: 0.2
      max_tokens: 2048
```

Модель берётся из YAML или `OPENROUTER_MODEL`. Не все модели поддерживают
одинаковые reasoning-параметры.

## Pipeline

Доступные stages:

- `translation_cycle`;
- `conservative_repair`;
- `reconstruction`;
- `contextual_reconstruction`;
- `memory_extraction`;
- `final_translation`.

Общие параметры stage:

```yaml
pipeline:
  - type: translation_cycle
    enabled: true
    probability: 1.0
    repeat: 1
    seed_offset: 0
  - type: reconstruction
    max_length_ratio: 1.15
    max_new_sentences_per_chunk: 2
    repetition_policy: preserve
    allow_new_events: false
    allow_scene_expansion: false
```

Параметры, влияющие на результат, входят в stage checksum.

## Context и memory

```yaml
context:
  enabled: true
  previous_chunks: 2
  max_chars: 3500
  truncation: tail
  source: final_generated_text
  include_intermediate: false

memory:
  enabled: true
  half_life_chunks: 20
  minimum_count: 2
  maximum_items_in_prompt: 8
```

При включённом context или memory chunks выполняются последовательно.

Memory extraction разрешён только после успешно выполненного
`translation_cycle`. Оригинальный текст в memory extraction не передаётся.
Ответ должен содержать валидный JSON с массивом `observations`.

## Prompts

Встроенный prompt:

```yaml
prompts:
  reconstruction: builtin:restrained_reconstruction
```

Пользовательский файл:

```yaml
prompts:
  reconstruction: prompts/my_reconstruction.txt
```

Текст prompt и его checksum записываются в manifest. Изменение prompt делает
старые checkpoints несовместимыми; resume такого запуска отклоняется.

## Runtime

```yaml
runtime:
  concurrency: 4
  requests_per_minute: 30
  retries: 4
  retry_backoff_seconds: 2
  failure_policy: stop
  fallback_provider: null
  budgets:
    max_requests: 200
    max_total_tokens: 250000
    max_cost_usd: 10.0
```

`failure_policy`:

- `stop` — сохранить ошибку и остановить запуск;
- `skip` — сохранить предупреждение и передать вход stage дальше;
- `fallback` — использовать встроенный mock fallback; для него нужно задать
  `fallback_provider: mock`.

`runtime.resume` устарел. Для продолжения используется команда
`semantic-telephone resume RUN_DIRECTORY`.

## Semantic metrics

Установка:

```bash
pip install -e '.[semantic-metrics]'
```

Конфигурация:

```yaml
metrics:
  semantic:
    enabled: true
    provider: sentence_transformers
    model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    revision: null
    device: auto
    batch_size: 16
    allow_downloads: false
```

По умолчанию модель должна уже находиться в кэше. Для разрешения загрузки
установите `allow_downloads: true`.
