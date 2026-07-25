# Артефакты и resume

Каждый новый запуск создаёт отдельный каталог:

```text
runs/<timestamp>-<name>/
├── manifest.json
├── resolved_config.yaml
├── source.txt
├── final.txt
├── events.jsonl
├── metrics.json
├── report.md
├── chunks/
│   └── 0001/
│       ├── source.txt
│       ├── context.txt
│       ├── stage-01-translation-cycle.json
│       ├── stage-01-translation-cycle-output.txt
│       ├── stage-01-translation-cycle-hop-01-output.txt
│       └── final.txt
└── memory/
    ├── observations.jsonl
    └── state.json
```

## Основные файлы

| Файл | Содержимое |
|---|---|
| `source.txt` | Зафиксированный исходный текст запуска |
| `final.txt` | Итоговый собранный текст |
| `resolved_config.yaml` | Полностью разобранная конфигурация |
| `manifest.json` | Версия schema, seed, checksums, модели, состояние и usage |
| `events.jsonl` | Stage events и HTTP request ledger |
| `metrics.json` | Структурные, lexical и опциональные semantic metrics |
| `report.md` | Читаемая сводка запуска |

## Stage checkpoint

Для каждого stage сохраняются metadata JSON и полный output. Для
`translation_cycle` дополнительно сохраняется output каждого языкового
перехода.

Checkpoint содержит:

- input и output checksums;
- stage checksum;
- seed и языковой маршрут;
- фактический provider route;
- model и revision;
- decoding и deterministic/sampling status;
- usage, warnings и errors;
- segment logs или OPUS hub legs;
- response ID;
- prompt checksum;
- memory payload и event ID, если применимо.

## Resume

```bash
semantic-telephone resume runs/<timestamp>-<name> --verbose
```

При resume программа читает конфигурацию из `manifest.json`, а исходный текст —
из сохранённого `source.txt`.

Checkpoint используется повторно только если:

- совпадает stage checksum;
- совпадает checksum output-файла;
- существуют и совпадают checksums всех hop outputs;
- checkpoint помечен reusable;
- prompt checksums запуска не изменились.

Если проверка не проходит, stage выполняется заново. Исходные артефакты не
удаляются.

## Memory

Schema v2 хранит memory extraction как checkpointed side effect. При resume
memory восстанавливается последовательно от ранних chunks к поздним. Повторный
event ID не применяется дважды.

- Schema-v1 запуск без memory можно продолжить.
- Schema-v1 запуск с включённой memory отклоняется; его нужно начать заново.

## Состояния manifest

`manifest.json` использует состояния:

- `running`;
- `completed`;
- `failed`;
- `interrupted`;
- `budget_exceeded`.

При ошибке сохраняются безопасная диагностика, request summary и текущий
budget status.

## Usage и budgets

Удалённые запросы записываются в `events.jsonl`. При resume контроллер
восстанавливает число запросов и записанный usage из этого журнала.

Доступные агрегаты:

- HTTP attempts;
- retries;
- prompt, completion и total tokens;
- cost, если провайдер её сообщил;
- input/output characters;
- segments.

Token и cost budgets нельзя строго контролировать, если провайдер не возвращает
usage. Это отмечается в manifest warnings.

## Просмотр

```bash
semantic-telephone inspect RUN_DIRECTORY --chunk 1
semantic-telephone report RUN_DIRECTORY
semantic-telephone memory show RUN_DIRECTORY
semantic-telephone compare RUN_A RUN_B --output comparison.md
```

`report` не повторяет перевод и generation. Он пересобирает `report.md` из
сохранённых артефактов.
