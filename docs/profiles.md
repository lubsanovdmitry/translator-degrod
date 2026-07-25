# Профили

Профиль — готовый YAML-файл. В копии репозитория он находится в `configs/`. После
`semantic-telephone init --profile NAME` его копия называется
`semantic-telephone.yaml`.

## Mock

Эти профили не выполняют машинный перевод:

| Профиль | Что проверяет |
|---|---|
| `translate_only` | Языковой маршрут, stages, checkpoints и артефакты |
| `sparse_repair` | Вероятностный repair между циклами |
| `iterative_reconstruction` | Чередование mock-перевода и reconstruction |
| `rolling_context` | Контекст из предыдущих повреждённых chunks |
| `inferred_memory` | Извлечение и повторное использование mock-memory |

Установка:

```bash
pip install -e .
semantic-telephone run configs/translate_only.yaml
```

## Реальный MT

Для локальных моделей:

```bash
pip install -e '.[local-mt]'
```

| Профиль | MT | OpenRouter | Дополнительные требования |
|---|---|---:|---|
| `nllb_only` | `facebook/nllb-200-distilled-600M` | Сводка отчёта | Нет |
| `m2m100_only` | `facebook/m2m100_418M` | Сводка отчёта | Нет |
| `pairwise_opus` | Четыре настроенные OPUS-модели | Сводка отчёта | Нет |
| `mixed_local` | NLLB, M2M100, OPUS-MT | Reconstruction и сводка | Больше RAM/VRAM и места в кэше |
| `raw_translation` | Тот же mixed MT | Нет | Нет |
| `grammar_repair` | Тот же mixed MT | Repair и сводка | Нет |
| `conservative_reconstruction` | Тот же mixed MT | Reconstruction и сводка | Нет |
| `aggressive_reconstruction` | Тот же mixed MT | Reconstruction и сводка | Нет |
| `mixed_with_libretranslate` | Mixed MT + LibreTranslate | Reconstruction и сводка | Запущенный LibreTranslate |

Для профилей с OpenRouter:

```env
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=provider/model-name
```

`raw_translation` — готовый real-профиль без LLM-вызовов.

## Что скачивается

| Провайдер | Модель |
|---|---|
| NLLB | Одна многоязычная модель на весь маршрут |
| M2M100 | Одна многоязычная модель на весь маршрут |
| OPUS-MT | Отдельная модель для каждой настроенной пары |
| OpenRouter | Локально ничего; используется HTTP API |
| LibreTranslate | Ничего в этот проект; используется настроенный сервер |

В `mixed_local` OPUS работает с `configured_pairs_only: true`. Провайдер не
угадывает и не загружает произвольные пары за пределами таблицы `pairs`.

Перед запуском:

```bash
semantic-telephone plan configs/mixed_local.yaml
semantic-telephone doctor configs/mixed_local.yaml
semantic-telephone doctor configs/mixed_local.yaml --allow-downloads
```

Последняя команда разрешает `doctor` скачать только явно настроенные
репозитории моделей.

## Reconstruction-профили

Эти четыре профиля используют одинаковый seed, языковой маршрут и mixed MT.
Различается обработка после перевода:

| Профиль | После MT | Ограничения |
|---|---|---|
| `raw_translation` | Ничего | Контрольный повреждённый текст |
| `grammar_repair` | Исправление грамматики | Ratio 1.05, без новых событий |
| `conservative_reconstruction` | Короткие локальные связки | Ratio 1.15, до двух новых предложений |
| `aggressive_reconstruction` | Расширяющая реконструкция | Ratio 2.5, разрешено расширение сцен |

Запуск сравнения:

```bash
semantic-telephone matrix configs/mixed_local_reconstruction_matrix.yaml
```

## LibreTranslate

Для локального сервера:

```env
LIBRETRANSLATE_BASE_URL=http://localhost:5000
LIBRETRANSLATE_API_KEY=
```

`doctor` вызывает `/languages`, а `run` — `/translate`.

## Commercial baseline

`commercial_baseline.yaml` содержит интерфейсную заглушку
`commercial_nmt`. Реального клиента нет, поэтому профиль не является
запускаемым baseline.
