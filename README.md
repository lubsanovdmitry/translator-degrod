# Semantic Telephone

`semantic-telephone` запускает воспроизводимые эксперименты с последовательным
переводом текста через несколько языков. Каждый переход, итоговый текст,
конфигурация, seed и данные провайдеров сохраняются в каталоге запуска.

Python: 3.12 или новее.

## Быстрый запуск из репозитория

### 1. Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[local-mt]'
cp -n .env.example .env
```

`local-mt` устанавливает `torch`, `transformers`, `sentencepiece` и
`sacremoses`. Веса моделей в пакет не входят.

### 2. Настройка

Положите исходный текст в `input.txt`.

Профиль `nllb_only` использует OpenRouter для итоговой диагностической сводки,
поэтому заполните `.env`:

```env
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=provider/model-name
```

### 3. Проверка и запуск

```bash
semantic-telephone plan configs/nllb_only.yaml
semantic-telephone doctor configs/nllb_only.yaml
semantic-telephone run configs/nllb_only.yaml --verbose
```

- `plan` работает без сети, ключей API и загрузки моделей.
- `doctor` проверяет зависимости, ключи API, HTTP-сервисы и локальный кэш.
- `run` запускает эксперимент. При первом использовании Transformers может
  скачать настроенную модель.

Путь к результату печатается после завершения:

```bash
cat runs/<точный-каталог>/final.txt
semantic-telephone inspect runs/<точный-каталог> --chunk 1
```

## `init` и `configs/`

Это два способа получить конфигурацию. Одновременно они не нужны.

| Ситуация | Что использовать |
|---|---|
| У вас есть копия репозитория | Готовый файл `configs/<profile>.yaml` |
| Пакет установлен, но репозитория рядом нет | `semantic-telephone init ... --profile <profile>` |

Из репозитория:

```bash
semantic-telephone run configs/nllb_only.yaml
```

После установки wheel-пакета:

```bash
semantic-telephone init my-run --profile nllb_only
cd my-run
semantic-telephone plan semantic-telephone.yaml
semantic-telephone doctor semantic-telephone.yaml
semantic-telephone run semantic-telephone.yaml
```

`init` ничего не запускает. Команда создаёт отдельную рабочую папку:

```text
my-run/
├── .env
├── input.txt
├── semantic-telephone.yaml
└── SEMANTIC_TELEPHONE.md
```

`semantic-telephone.yaml` — копия выбранного встроенного профиля с путями,
подходящими для этой папки. Её можно изменять.

Wheel — установочный файл Python с расширением `.whl`. Он содержит код,
профили и prompts, но не веса моделей.

## Реальные профили

| Профиль | Перевод | LLM после перевода |
|---|---|---|
| `nllb_only` | NLLB | Только сводка отчёта |
| `m2m100_only` | M2M100 | Только сводка отчёта |
| `pairwise_opus` | OPUS-MT | Только сводка отчёта |
| `mixed_local` | NLLB + M2M100 + OPUS-MT | Reconstruction и сводка |
| `raw_translation` | NLLB + M2M100 + OPUS-MT | Нет |
| `grammar_repair` | Тот же mixed MT | Repair и сводка |
| `conservative_reconstruction` | Тот же mixed MT | Ограниченная реконструкция и сводка |
| `aggressive_reconstruction` | Тот же mixed MT | Расширяющая реконструкция и сводка |
| `mixed_with_libretranslate` | Mixed MT + LibreTranslate | Reconstruction и сводка |

Рекомендуемый эксперимент с разными MT-архитектурами — `mixed_local`.
Детерминированный MT-baseline — `nllb_only`; OpenRouter в нём нужен только для
сводки отчёта.

`commercial_baseline` пока не запускается: коммерческий клиент представлен
только интерфейсной заглушкой.

Полная таблица требований и загрузок: [docs/profiles.md](docs/profiles.md).

## Mock-профили

`translate_only`, `sparse_repair`, `iterative_reconstruction`,
`rolling_context` и `inferred_memory` используют mock-провайдеры. Они проверяют
CLI, маршруты, checkpoints, продолжение запуска и артефакты, но не переводят
языки.

Smoke test без моделей и API:

```bash
pip install -e .
semantic-telephone validate configs/translate_only.yaml
semantic-telephone run configs/translate_only.yaml
```

Не используйте результат mock-профиля для оценки качества перевода.

## Языковой и engine-маршрут

Это разные настройки:

- Языковой маршрут: `ru -> ka -> en -> de -> ru`.
- Engine-маршрут: какой MT-провайдер выполняет каждый отдельный переход.

В mixed-профилях выбор MT-движка для перехода зависит от режима маршрутизации,
поддержки языковой пары и seed. Фактически использованный маршрут сохраняется в
checkpoint этапа и manifest.

Описание YAML: [docs/configuration.md](docs/configuration.md).

## Загрузка моделей

- NLLB и M2M100 используют по одной многоязычной модели.
- OPUS-MT использует отдельные модели для языковых пар.
- `pip install -e '.[local-mt]'` устанавливает библиотеки, но не веса.
- `validate` и `plan` ничего не скачивают.
- `doctor` без флага проверяет кэш.
- `doctor --allow-downloads` разрешает загрузить только модели, указанные в
  конфигурации.
- `run` лениво загружает модель при первом фактическом использовании.

Обычно Hugging Face хранит веса в `~/.cache/huggingface/hub`. Для другого
расположения задайте `HF_HOME`.

## Основные команды

```bash
# Проверить YAML без сети
semantic-telephone validate CONFIG

# Показать маршруты, провайдеры и верхнюю оценку запросов без сети
semantic-telephone plan CONFIG

# Проверить окружение, кэш и HTTP-сервисы
semantic-telephone doctor CONFIG

# Разрешить doctor скачать явно настроенные модели
semantic-telephone doctor CONFIG --allow-downloads

# Новый запуск
semantic-telephone run CONFIG --verbose

# Продолжить существующий запуск
semantic-telephone resume RUN_DIRECTORY --verbose

# Посмотреть этапы одного chunk
semantic-telephone inspect RUN_DIRECTORY --chunk 1

# Пересобрать report.md из артефактов
semantic-telephone report RUN_DIRECTORY

# Сравнить два результата
semantic-telephone compare RUN_A RUN_B --output comparison.md

# Запустить matrix
semantic-telephone matrix MATRIX_CONFIG
```

## Resume

Продолжение всегда задаётся отдельной командой:

```bash
semantic-telephone resume runs/<точный-каталог>
```

Checkpoint переиспользуется только при совпадении входа, stage config, seed,
маршрута, настроек провайдеров, контекста, памяти, prompt checksum и checksum
выходных файлов.

Если prompt изменился, resume останавливается. Старый schema-v1 запуск без
memory можно продолжить; schema-v1 запуск с memory нужно запустить заново.

Подробности: [docs/artifacts.md](docs/artifacts.md).

## Ограничения ресурсов

Для удалённых HTTP-провайдеров:

```yaml
runtime:
  requests_per_minute: 30
  retries: 4
  budgets:
    max_requests: 200
    max_total_tokens: 250000
    max_cost_usd: 10.0
```

Request budget проверяется до отправки. Token и cost budgets работают только
если провайдер возвращает соответствующий usage. Неполная enforceability
записывается в warnings и manifest.

## Matrix

```bash
semantic-telephone matrix configs/mixed_local_reconstruction_matrix.yaml
```

Matrix запускает несколько профилей с заданными seeds и создаёт:

- `summary.json` и `summary.csv`;
- `aggregates.csv`;
- paired deltas относительно baseline;
- общий `report.md`.

`experiment.failure_policy` принимает `stop` или `continue`.

## Разработка

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests
mypy
python -m compileall -q src tests
python -m semantic_telephone.resource_sync --check
```

Offline-тесты не должны обращаться к сети, загружать модели или требовать GPU,
API-ключи и LibreTranslate server.

## Документация

- [Профили и требования](docs/profiles.md)
- [Конфигурация YAML](docs/configuration.md)
- [Артефакты и resume](docs/artifacts.md)
- [История изменений](CHANGELOG.md)
