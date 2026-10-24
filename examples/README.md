# Fictional example data

All documents and records in this directory were written for this standalone
example. They describe an invented software service.

- `docs/` is the Markdown corpus used by indexing.
- `catalog.json` supplies supported alarm codes and descriptive metadata.
- `alarm-report.json` is a sample CLI input and HTTP resolve request.
- `reference-answers.json` supplies a reference for evaluation.

Replace these examples with documents and records you have permission to use.
Catalogue entries require `code`, `title`, `domain`, `severity`,
`severity_category`, and `alarm_text` for the HTTP API. Resolve requests require
`code` and an `env` object containing `versions` and `locale`.
