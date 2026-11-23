# Document retrieval benchmarks

The runner takes plain search text from each question and uses the engine's
vector search, reranking, and evidence stitching with a shared-ingestion index.
Expected sources, pages, and answers are used only for evaluation.

## Acquire raw documents and labels

```bash
export PYTHONPATH=src
python -m pip install -r requirements/requirements-ingestion.txt
python -m rag_engine.benchmarks.prepare linuxcnc
python -m rag_engine.benchmarks.prepare techqa
python -m rag_engine.benchmarks.prepare techmanualqa \
  --pdf-dir data/raw/verified-manuals
```

TechQA downloads the complete article corpus and supplied labels. TechManualQA
requires your separately obtained manufacturer PDFs; every PDF you supply is
included, even if it has no evaluation questions. Only questions whose source
manual is available are eligible. LinuxCNC downloads two complete version 2.9
HTML documents, rather than extracting the sections mentioned by the labels.
Its ten project-authored questions remain a small development evaluation set.

Each acquisition goes under `data/raw/benchmarks/<dataset>/` and contains
the following files.

- `documents/`: original files, with no answer-driven conversion or selection.
- `cases.jsonl`: questions, answerability, reference answers, and source locations.
- `sources.json`: original source locations, independent of the question labels.

The original sources are [TechQA-RAG-Eval](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval),
[TechManualQA labels](https://zenodo.org/records/17410809) and
[manual source list](https://github.com/tduricic/techmanualqa#covered-manuals), and
LinuxCNC's [stepper diagnostics](https://linuxcnc.org/docs/2.9/html/config/stepper-diagnostics.html)
and [latency testing](https://linuxcnc.org/docs/2.9/html/install/latency-test.html).

## Use the regular application pipeline

```bash
raw=data/raw/benchmarks/linuxcnc
dataset=data/processed/general-ingestion/linuxcnc
embedding=sentence-transformers/all-MiniLM-L6-v2

python -m rag_engine.cli.ingest "$raw/documents" \
  --output "$dataset" --sources "$raw/sources.json"
python -m rag_engine.cli.index --docs-root "$dataset/documents" \
  --database-path "$dataset/index" --collection benchmark --embedding-model "$embedding"

# Measure retrieval over the labelled questions:
python -m rag_engine.benchmarks.run "$dataset" --cases "$raw/cases.jsonl" \
  --embedding-model "$embedding" \
  --output output/linuxcnc-application.json
```

The evaluator accepts `--reranker`, `--chunk-limit`, `--file-limit`,
`--char-budget`, and `--evidence-rank-limit`. The default uses vector ranking.
`--limit 40 --seed 42` selects a reproducible sample with both answerability
classes when possible. Omit `--limit` to run all cases.

Use fresh ingestion directories, indexes, and output files. Earlier prepared
corpora and indexes used different source identifiers; rebuild through the new
pipeline rather than mixing them with new evaluation labels.

## Read the results

Each case saves the complete application result: question, retrieval metadata,
exact evidence text, original source, and PDF pages. The evaluator scores the
result against the expected sources and locations.

- **Gold source recall:** fraction of labelled original documents represented
  in selected evidence, averaged per labelled case.
- **Gold page recall:** fraction of labelled PDF pages represented in selected
  evidence. Finding the right manual alone does not pass this measure.
- **Gold section recall:** fraction of labelled headings represented in selected
  evidence from the correct document. LinuxCNC uses this alongside document
  coverage. Heading matches measure section coverage, and the saved evidence
  lets readers check whether it contains the relevant passage.

PDF conversion is an extraction stage with its own possible failures. Inspect
normalized Markdown and structured output against the source pages; source/page
coverage cannot establish correct table interpretation or complete procedures.
The benchmark documents are development data. Claims about generalization need
additional documents and questions held out from implementation choices.

## Retrieval through the shared ingestion pipeline

The TechQA run covered all 910 rows in NVIDIA's release, including 300 marked
unanswerable. The LinuxCNC run covered all 10 locally authored questions across
the two selected documents. TechManualQA's release has 20 manuals and 700
questions, including 100 procedural and 100 unanswerable cases. We prepared 12
source PDFs and 420 matching labels, and completed retrieval on three manuals
with 105 questions. The separate Makita run uses a subset of those same questions.

These runs use MiniLM embeddings, 50 candidates, vector ranking, a three-file
limit, and a 6,000-character evidence budget. The runner used retrieval only.

| Corpus | Questions | Labelled sources found | Labelled locations found |
| --- | ---: | ---: | ---: |
| LinuxCNC, both full HTML documents | 10 | 7 of 7 answerable cases | 8 of 8 section references |
| TechQA, all 28,481 articles | 910 | 291 of 610 answerable cases (47.7%) | |
| TechManualQA, three shortest available manuals | 105 | 90 of 90 answerable cases | 81 of 90 labelled pages (90.0%) |
| TechManualQA, Makita manual only | 35 | 30 of 30 answerable cases | 29 of 30 labelled pages (96.7%) |

For TechQA, 182 of the 319 missed sources appeared in the initial 50 vector
candidates, and 137 were absent. The full index contains 97,076 chunks. The
rebuild reused 97,075 MiniLM vectors for identical chunk text and encoded one
new chunk. A comparison of 25 reused vectors with fresh encodings found a
minimum dot product of 0.99999988.

The three-manual corpus used LG, Makita, and Omron, selected by PDF page count.
All nine missed pages were present in the conversion manifest. On visual
inspection of Makita pages 5–7, the Markdown kept the main section order and the
speed-change table. It lost the indicator-lamp symbols in the page 5
battery-capacity table.

The reports are `output/linuxcnc-application-retrieval.json`,
`output/techqa-application-retrieval.json`,
`output/techmanualqa-three-manual-application-retrieval.json`, and
`output/techmanualqa-makita-application-retrieval.json`.

## Historical results before the ingestion migration

The results below used the previous benchmark-specific conversion and runner.
They are retained as history and are not directly comparable with the general
application path: corpus boundaries, source identifiers, evidence rules, and
query construction have changed.

## Initial LinuxCNC results

The ten-case retrieval run used the MiniLM embeddings, vector ranking, up to
three files, and a 6,000-character budget. Both retrieval conditions returned
every labelled source across the seven answerable cases. Stitched evidence
averaged 4,195 characters per case. Individual chunks averaged 2,219.

A separate three-case generation run used Qwen3 4B and seed 42. Both retrieval
conditions supplied the requested answer for the two answerable cases. Both
failed to abstain when asked to identify the exact peripheral that caused a
realtime delay. The cited docs described possible causes and diagnostic tests.
They supplied no evidence identifying that peripheral on the user's machine.

## Initial retrieval comparison

Both 40-question samples used seed 42, MiniLM embeddings, vector ranking, 50
candidate chunks, up to three files, and a 6,000-character text budget. Gold
source recall averages over the answerable cases.

| Corpus | Answerable cases | Individual chunks | Stitched sections |
| --- | ---: | ---: | ---: |
| Full TechQA corpus, 97,076 chunks | 27 | 55.6% | 44.4% |
| TechManualQA subset, 2,560 chunks | 34 | 67.6% | 67.6% |

The TechManualQA subset used 12 PDFs for a total of 420 eligible cases.
The filenames were `apple_watch.pdf`, `bosch_oven.pdf`, `dji_mavic_pro.pdf`,
`electrolux_oven.pdf`, `ford_mach_e.pdf`, `laptop_lenovo_tc_x1.pdf`,
`lg_home_theater.pdf`, `makita_drill.pdf`, `nintendo_2ds_xl.pdf`,
`omron_monitor.pdf`, `prusa_3d-printer.pdf`, and `spot_boston_dynamics.pdf`.

## Evaluate the bundled alarm example

The evaluator uses the fictional example catalogue and references. Its default
judge runs locally through Ollama. Follow the [service setup](../SETUP.md) to
prepare the index and model, then run:

```bash
PYTHONPATH=src python -m rag_engine.evaluation.run_all --codes demo.svc.0001
```
