# Semantic Telephone

`semantic-telephone` — локальный CLI-проект на Python 3.12 для
воспроизводимых экспериментов с рекурсивным искажением текста. Он разбивает
исходник на содержательные блоки, проводит каждый блок через заданные языковые
маршруты и, если это включено, умеренно восстанавливает грамматику или локальную
связность. Все промежуточные поколения сохраняются.

Система не гарантирует смешной, сюрреалистичный или художественно удачный
результат. Качество и характер изменений зависят от провайдеров, маршрутов,
исходного текста и настроек. Метрики являются диагностическими эвристиками, а
не объективной оценкой текста или доказательством научного эффекта.

## Реальный запуск

Требуется Python 3.12 или новее.

Установите проект и библиотеки, необходимые для запуска локальных MT-моделей:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[local-mt]'
cp -n .env.example .env
```

Эта команда **не скачивает веса переводчиков**. Она устанавливает `torch`,
`transformers`, `sentencepiece` и `sacremoses`. Веса скачиваются лениво только
командой `run`, когда конкретный движок впервые выбран для перехода.
`validate` ничего не скачивает.

Откройте `.env` и заполните:

```env
OPENROUTER_API_KEY=ваш-ключ
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=provider/model-name
```

Конкретная OpenRouter-модель не зашита в проект. `OPENROUTER_MODEL` используется
для LLM-сводки отчёта и как default для задач без собственного `model` в YAML.

Поместите свой текст в корневой `input.txt`, затем запустите реальный NLLB
baseline:

```bash
semantic-telephone validate configs/nllb_only.yaml
semantic-telephone run configs/nllb_only.yaml --verbose
```

Это настоящий запуск:

- `facebook/nllb-200-distilled-600M` переводит все переходы маршрута;
- при доступной CUDA модель использует GPU и FP16, иначе CPU;
- первый запуск скачивает модель с Hugging Face и поэтому требует сеть;
- OpenRouter создаёт диагностическую сводку отчёта;
- финальный текст, каждый переход, сегменты, модели и метаданные сохраняются в
  напечатанном CLI каталоге `runs/...-nllb-only/`.

Откройте результат:

```bash
cat runs/...-nllb-only/final.txt
semantic-telephone inspect runs/...-nllb-only --chunk 1
```

Вместо `...` используйте точный путь, напечатанный командой `run`.

Для основного смешанного эксперимента:

```bash
semantic-telephone validate configs/mixed_local.yaml
semantic-telephone run configs/mixed_local.yaml --verbose
```

`mixed_local` действительно использует NLLB, M2M100 и явно перечисленные
OPUS-MT пары, а OpenRouter выполняет reconstruction и формирует сводку отчёта.
На первом запуске нужные модели скачиваются до соответствующего перевода.
Загрузка OPUS ограничена таблицей `pairs`; произвольные угаданные pair-модели
этот профиль не скачивает. Понадобится заметно больше времени, RAM/VRAM и места
в model cache, чем для `nllb_only`.

### Сравнение режимов реконструкции

Исходный `mixed_local.yaml` сохранён без изменения поведения и считается
legacy-вариантом агрессивной реконструкции. Для контролируемого сравнения
добавлены четыре явных профиля. У них совпадают вход, chunking, seed, языковой
маршрут, набор MT-провайдеров и engine routing, поэтому этап перевода перед
LLM одинаков:

| Профиль | Обработка после MT | Основные ограничения |
|---|---|---|
| `raw_translation` | Нет LLM, включая LLM-сводку отчёта | Контрольный повреждённый перевод |
| `grammar_repair` | Только грамматика, `temperature: 0.1` | `max_length_ratio: 1.05`, без новых событий |
| `conservative_reconstruction` | Короткие локальные связки, `temperature: 0.35` | ratio `1.15`, не более двух новых предложений, повторы сохраняются |
| `aggressive_reconstruction` | Расширяющая реконструкция, `temperature: 0.7` | ratio `2.5`, повторы рационализируются, расширение сцен разрешено |

Консервативный prompt запрещает превращать повторы в отдельную атмосферную
сцену, изобретать конфликт или сверхъестественное объяснение, заменять
конкретную бессмыслицу философскими рассуждениями и стилистически выравнивать
разнородные части текста. Параметры режима входят в resolved config и checksum
этапа. Если провайдер нарушит ratio или лимит новых предложений, результат
сохранится для анализа, а нарушение попадёт в warnings.

Запустить все варианты с одним seed и получить общую таблицу:

```bash
semantic-telephone matrix configs/mixed_local_reconstruction_matrix.yaml
```

Этот matrix — реальный эксперимент: ему нужны локальные MT-зависимости, сеть
для первой загрузки весов и `OPENROUTER_API_KEY` для трёх LLM-вариантов.

### Что именно скачивается

NLLB и M2M100 — по одной многоязычной модели, а не по модели на каждый язык.
OPUS-MT, наоборот, использует отдельные модели для языковых пар.

| Профиль | Локальные веса при первом использовании |
|---|---|
| `nllb_only` | Только `facebook/nllb-200-distilled-600M` |
| `m2m100_only` | Только `facebook/m2m100_418M` |
| `pairwise_opus` | Четыре явно указанные OPUS-модели для `ru-en`, `en-ru`, `en-de`, `de-en` |
| `mixed_local` | NLLB, M2M100 и только те явно указанные OPUS-пары, которые потребовались seeded engine route |
| `raw_translation`, `grammar_repair`, `conservative_reconstruction`, `aggressive_reconstruction` | Тот же локальный MT-набор, что у `mixed_local` |
| `mixed_with_libretranslate` | То же, но LibreTranslate вызывается по HTTP и локальных весов в этот проект не добавляет |

OpenRouter-модель локально не скачивается: она вызывается через API.

Обычно Hugging Face хранит скачанные веса в
`~/.cache/huggingface/hub`. Путь можно перенести, задав `HF_HOME` до запуска.
На следующих запусках `transformers` использует cache. Во время выполнения
выбранные NLLB/M2M100 загружаются из cache в RAM и, при доступной CUDA, в VRAM.
OPUS держит в памяти ограниченное число pair-моделей и вытесняет старые из
VRAM; это не удаляет их с диска.

Для собственного файла и отдельного каталога удобнее скопировать профиль,
изменить `input.path`, `run.name`, языковой маршрут и параметры движков, затем
запустить копию той же командой.

## Проверка установки — только mock, не настоящий перевод

Следующая команда нужна исключительно как быстрый smoke test CLI, checkpoint,
маршрутов и файлов результатов:

```bash
pip install -e .
semantic-telephone validate configs/translate_only.yaml
semantic-telephone run configs/translate_only.yaml
```

`translate_only.yaml` использует `MockTranslationProvider`. Он не знает языков
и не выполняет машинный перевод: это детерминированная тестовая порча текста.
Его результат не следует оценивать как результат работы NLLB, M2M100, OPUS-MT
или LibreTranslate.

Для разработки:

```bash
pip install -e '.[dev]'
pytest
ruff check .
mypy
```

Команда `semantic-telephone init my-experiment` создаст неразрушающим образом
пустой `input.txt`, `.env` и mock-конфигурацию для smoke test. Для реального
перевода скопируйте один из real-профилей, описанных выше.

## Пять режимов

- `translate_only` — только последовательные переводы; базовая линия чистого
  переводческого распада.
- `sparse_repair` — два переводческих цикла и вероятностное консервативное
  исправление грамматики между ними.
- `iterative_reconstruction` — чередование перевода и умеренного локального
  восстановления связности.
- `rolling_context` — реконструктор дополнительно получает ограниченный хвост
  **предыдущих повреждённых результатов**, но никогда не предыдущий оригинал.
- `inferred_memory` — экспериментальная автоматически извлекаемая память с
  decay. Этот режим выключен по умолчанию.

Исходные пять pipeline-конфигураций используют mock-провайдеры и предназначены
для проверки механики эксперимента. Реальные MT-профили имеют суффиксы
`nllb_only`, `m2m100_only`, `pairwise_opus`, `mixed_local` и
`mixed_with_libretranslate`.

## Конфигурация

Пользовательская конфигурация — YAML. Неизвестный тип этапа, неверная
вероятность, несовместимые настройки памяти и некорректные лимиты приводят к
понятной ошибке `validate`.

```yaml
run:
  name: rolling-context-demo
  seed: 1080
  source_language: ru
  target_language: ru
  output_root: runs

input:
  path: input.txt
  encoding: utf-8

chunking:
  strategy: target_chars
  target_chars: 1100
  max_chars: 1800
  paragraph_overlap: 1

translation:
  provider: mock
  route_mode: stratified
  min_hops: 5
  max_hops: 8
  languages:
    allow: []
    deny: []

generation:
  provider: mock
  model: mock
  temperature:
    repair: 0.3
    reconstruction: 0.8
    memory_extraction: 0.1

context:
  enabled: true
  previous_chunks: 2
  max_chars: 3500
  truncation: tail
  source: final_generated_text

memory:
  enabled: false
  half_life_chunks: 20
  minimum_count: 2
  maximum_items_in_prompt: 8

pipeline:
  - type: translation_cycle
    hops: {min: 5, max: 8}
  - type: conservative_repair
    probability: 0.25
  - type: contextual_reconstruction
    probability: 0.7
  - type: final_translation

runtime:
  concurrency: 1
  retries: 4
  retry_backoff_seconds: 2
  resume: true
  failure_policy: stop
```

Стратегии разбиения: `paragraph`, `target_chars`, `target_tokens` и
`sentence_window`. Перекрытие хранится в `context.txt` отдельно от тела блока и
не попадает повторно в `final.txt`.

Режимы маршрутов: `fixed`, `random`, `stratified`, `hubbed` и
`mutating_fixed`. Генератор удаляет соседние одинаковые языки, соблюдает
allow/deny-списки, всегда возвращается в целевой язык и воспроизводит выбор по
seed. Категории в `stratified` — удобная экспериментальная стратификация, а не
утверждение о том, что какой-либо язык даёт определённый или «более смешной»
результат.

У необязательного этапа есть `enabled`, `probability`, `repeat`, `seed` и
`seed_offset`. Решение о применении входит в checkpoint и `events.jsonl`.

## Реальные провайдеры

Библиотеки для локальных NLLB, M2M100 и OPUS-MT устанавливаются отдельным
набором зависимостей:

```bash
pip install -e '.[local-mt]'
```

Этот extra устанавливает runtime-зависимости, но не веса моделей. Веса
загружаются `transformers.from_pretrained()` лениво при фактическом выборе
провайдера и затем переиспользуются из Hugging Face cache.

NLLB и M2M100 используют `transformers` напрямую, CUDA/FP16 при наличии и CPU
fallback. Длинные предложения режутся по токенам ниже `max_input_tokens`;
параметры и результат каждого сегмента сохраняются в checkpoint и
`events.jsonl`. Real-профили ограничивают каждый результат сегмента
`max_new_tokens: 512` и запрещают повтор одной и той же 3-граммы, чтобы greedy
decoding не раздувал единичную ошибку в сотни одинаковых слов. OPUS-MT выбирает
отдельную Marian-модель по паре, умеет идти
через английский hub и держит ограниченный LRU-кэш моделей. При
`allow_downloads: false` принимаются только явно перечисленные и уже локально
доступные модели. Сочетание `allow_downloads: true` и
`configured_pairs_only: true` разрешает первый download, но только для
перечисленных в `pairs` моделей. Preflight выполняется до первого перевода
соответствующего маршрута.

Языковой маршрут и маршрут движков настраиваются независимо. Доступны
`single_engine`, `fixed_engine_route`, `weighted_random`, `alternating`,
`quality_fallback` и `heterogeneous`; все случайные решения зависят от seed.
`quality_fallback` сохраняет ошибки неудачных движков и фактически выбранный
provider. `LibreTranslateProvider` проверяет `/languages`, кэширует ответ,
вызывает `/translate`, повторяет временные ошибки и записывает версию сервера,
если она опубликована в заголовках.

Рекомендуемый экспериментальный профиль — `configs/mixed_local.yaml`.
`configs/nllb_only.yaml` остаётся чистой baseline-конфигурацией. Также доступны
`m2m100_only.yaml`, `pairwise_opus.yaml`, `mixed_with_libretranslate.yaml` и
интерфейсный `commercial_baseline.yaml`. Клиенты коммерческих MT в первой
версии оставлены явными заглушками категории `commercial_nmt`.

Грамматический ремонт, реконструкция, извлечение памяти и LLM-сводка отчёта
используют OpenRouter. Модель и generation-параметры можно менять отдельно для
каждой задачи:

```yaml
generation:
  provider: openrouter
  api_key_env: OPENROUTER_API_KEY
  tasks:
    conservative_repair:
      model: vendor/repair-model
      max_tokens: 1200
      parameters: {top_p: 0.9}
    reconstruction:
      model: vendor/reconstruction-model
      max_tokens: 1800
    contextual_reconstruction:
      model: vendor/context-model
      max_tokens: 2000
    memory_extraction:
      model: vendor/json-model
      max_tokens: 1000
    report_generation:
      model: vendor/report-model
      max_tokens: 1200
  temperature:
    conservative_repair: 0.2
    reconstruction: 0.6
    contextual_reconstruction: 0.6
    memory_extraction: 0.1
    report_generation: 0.2
```

`OPENROUTER_BASE_URL` по умолчанию равен
`https://openrouter.ai/api/v1`, а `OPENROUTER_MODEL` служит моделью по умолчанию
для задач без собственного `model`. Для LibreTranslate используются
`LIBRETRANSLATE_BASE_URL` и необязательный `LIBRETRANSLATE_API_KEY`. Секреты
читаются только из окружения или `.env`; endpoint, точные модели/revision,
response ID, usage, хеши и фактический маршрут движков записываются в
артефакты.

### OpenRouter вернул `200 OK`, но текста нет

Reasoning-модель может потратить весь `max_tokens` на reasoning и завершиться с
`finish_reason: length`, `message.content: null`. HTTP-запрос при этом формально
успешен. `exclude: true` только исключает reasoning из ответа, но не отключает
его и не экономит output tokens. Кроме того, не все модели поддерживают
`reasoning.effort`. Поэтому стандартные real-профили явно отключают reasoning
для служебных задач и дают reconstruction 4096 output tokens:

```yaml
generation:
  provider: openrouter
  parameters:
    reasoning: {enabled: false}
  tasks:
    reconstruction: {max_tokens: 4096}
```

Если конкретной задаче действительно нужен reasoning, переопределите
`generation.tasks.<task>.parameters.reasoning` согласно возможностям выбранной
модели и увеличьте `max_tokens`: reasoning входит в output budget.

Клиент выводит `finish_reason`, тип content, возвращён ли reasoning, и token usage,
не сохраняя полный приватный ответ. Ответ без финального текста считается
non-retryable: тот же платный запрос не отправляется ещё четыре раза.

После изменения YAML запускайте новую команду `run`. Обычный `resume` намеренно
использует resolved config старого запуска, поэтому сохранит прежний token
budget.

## CLI

```text
semantic-telephone init [DIRECTORY]
semantic-telephone validate CONFIG
semantic-telephone run CONFIG
semantic-telephone resume RUN_DIRECTORY
semantic-telephone matrix MATRIX_CONFIG
semantic-telephone inspect RUN_DIRECTORY --chunk 1
semantic-telephone compare RUN_A RUN_B --output comparison.md
semantic-telephone report RUN_DIRECTORY
semantic-telephone memory show RUN_DIRECTORY
semantic-telephone memory clear RUN_DIRECTORY
```

`inspect` показывает цепочку Original → stages → Final. `compare` создаёт
side-by-side Markdown. `report` пересобирает отчёт из сохранённых артефактов.

## Результаты и продолжение запуска

Каждый запуск получает отдельный каталог:

```text
runs/<timestamp>-<name>/
├── manifest.json
├── resolved_config.yaml
├── source.txt
├── final.txt
├── metrics.json
├── events.jsonl
├── report.md
├── chunks/
│   └── 0001/
│       ├── source.txt
│       ├── context.txt
│       ├── stage-01-translation-cycle.json
│       ├── stage-01-translation-cycle-output.txt
│       └── final.txt
└── memory/
    ├── observations.jsonl
    └── state.json
```

Запись основных файлов атомарна. Checkpoint считается пригодным только при
совпадении хеша входа, конфигурации этапа, seed, маршрута, модели, контекста,
памяти и промпта, а также checksum выходного файла. Поэтому
`semantic-telephone resume` повторно использует только полностью завершённые и
совместимые этапы. Исходные и уже готовые результаты при ошибке не удаляются.

## Rolling context

Контекст формируется из последних успешно обработанных повреждённых блоков.
Доступны число блоков, лимит символов и обрезка начала/конца. Оригинальные
предыдущие абзацы и исходный текущий блок не добавляются в реконструкционный
промпт. Контекст называется ненадёжным: реконструктор вправе его игнорировать и
не обязан поддерживать возникшие мотивы.

Независимые блоки без rolling context и памяти обрабатываются ограниченным
числом задач из `runtime.concurrency`. Этапы внутри блока и переходы языкового
маршрута остаются последовательными, а финальный текст, маршруты, warnings и
`processed_chunks` собираются в исходном порядке независимо от порядка
завершения задач.

Чтобы исключить причинную неоднозначность, при включённом rolling context или
памяти эффективная параллельность блоков автоматически снижается до `1`. Она
записывается в manifest как `effective_chunk_concurrency`, а снижение
появляется в warnings отчёта. Локальные Transformers-модели также допускают
только один inference на экземпляр модели: это защищает загрузку, RNG,
tokenizer и VRAM, хотя независимые сетевые этапы других блоков могут
продолжаться параллельно.

Если engine router содержит OPUS-MT, preflight и все переходы одного языкового
маршрута защищаются общей блокировкой. Это не даёт preflight другого блока
вытеснить модель пары из ограниченного LRU-кэша до её использования. Остальные
этапы блоков, включая OpenRouter reconstruction, по-прежнему могут идти
параллельно. Manifest отмечает этот режим полем
`translation_routes_serialized`.

## Автоматическая память

Память строится отдельным JSON-этапом только из повреждённого результата. API
этого этапа намеренно не принимает оригинал. Ответ проверяется по структуре;
невалидный JSON повторяется согласно retry-настройкам.

Запись содержит ключ сущности, текст наблюдения, count, confidence и границы
видимости по блокам. Вес — документированная эвристика:

```text
weight = confidence × log(1 + count) × exp(-age / half_life)
```

Это не научно обоснованная формула и не канон мира. Одноразовые наблюдения по
умолчанию не попадают в промпт, повторения повышают вес, старые записи затухают,
а реконструктор может проигнорировать всё. Память выключена по умолчанию,
поскольку она меняет эксперимент: устойчивость мотивов без неё нельзя
приписывать отдельному хранилищу модели.

## Экспериментальная матрица

```bash
semantic-telephone matrix configs/experiment_matrix.yaml
```

Матрица запускает пять вариантов с заданными seed. Она создаёт независимые
каталоги запусков, `summary.json`, `summary.csv` и общий `report.md` со ссылками
на финальные тексты. Usage агрегируется в файлах этапов; денежная стоимость
показывается только если провайдер её вернул.

`experiment.concurrency` задаёт максимальное число одновременно выполняемых
запусков матрицы (по умолчанию `1`). Строки summary сохраняют порядок вариантов
и seed из YAML. Этот лимит умножается на `runtime.concurrency` каждого запуска,
поэтому для локальных MT-моделей его следует повышать только при достаточном
объёме RAM/VRAM.

## Метрики и воспроизводимость

Отчёт включает структурные показатели, token overlap, символьные 4-граммы,
новые слова, повторы, пунктуацию, простые кандидаты сущностей и «возможное
генеративное расширение». Последнее — только эвристика, не доказательство
галлюцинации. Embedding-метрики помечаются недоступными, пока embedding-провайдер
не настроен.

Один seed управляет маршрутами, количеством переходов, вероятностными этапами и
порядком локальных решений. С mock-провайдерами одинаковые конфигурация, текст и
seed дают одинаковый финальный текст. Время выполнения и timestamp закономерно
различаются.

Стандартные промпты находятся в [`prompts/`](prompts), хешируются и не получают
название произведения или исходник одновременно с повреждённым текстом.
Пользовательские пути в секции `prompts` отмечаются в manifest и меняют
checkpoint checksum.

## Why this is not a random absurdity generator

В коде и стандартных промптах нет заранее заданных персонажей, привычек,
стран, сюжетных мотивов или инструкции создавать пародию. Mock-провайдер
выполняет общие детерминированные операции над формой текста: перестановку,
удаление местоимения, близкую замену, изменение пунктуации и умеренное
соединение фраз. Повторяющиеся мотивы могут появиться только из текущего текста,
маршрутов, повторной обработки повреждённого результата, ограниченного
контекста или явно включённой автоматической памяти.

Сильное повреждение часто даёт нечитаемую кашу. Слишком сильная реконструкция
может превратить результат в обычный фанфик. Практически интересен баланс между
повреждением и умеренным ремонтом, но устойчивый мотив всё равно может быть
случайностью. Сам факт повторения не доказывает наличие скрытой долговременной
памяти у модели.

## Ограничения первой версии

- Эвристика сущностей не заменяет полноценный NER.
- Автоматическое определение фактического языка ответа не включено; поддержка
  языковой пары остаётся ответственностью переводческого endpoint.
- Rate-limit budget хранится в конфигурации и manifest, но отдельный глобальный
  планировщик запросов пока не реализован.
- Коммерческие MT-клиенты пока представлены интерфейсными заглушками; локальные
  провайдеры и LibreTranslate реализованы полностью.
- Семантические embedding-метрики и расчёт денежной стоимости опциональны и в
  mock-режиме отсутствуют.

Архитектура намеренно файловая и локальная: без очередей, базы данных,
микросервисов и веб-интерфейса.
