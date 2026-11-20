# Public data benchmarks

The importer writes each corpus to its own directory under
`data/benchmarks/`. Index only `docs/`. The adjacent `cases.jsonl` contains
the expected answers, answerability labels, and references. `sources.json`
maps indexed paths to the original source locations. Downloads, indexes, and
run results are kept outside Git.

## Prepare the datasets

```bash
export PYTHONPATH=src
python -m rag_engine.benchmarks.prepare techqa
python -m rag_engine.benchmarks.prepare linuxcnc
```

The [TechQA release](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/tree/0b5bbc84b7f07d6d09d063130e90b716d8d4a32a)
provides an Apache-2.0 dataset card and the article archive. The importer indexes
the entire corpus, including articles that are irrelevant to the sampled
questions. The published package labels its question split `train`.

For TechManualQA, obtain manufacturer PDFs from the [authors' source list](https://github.com/tduricic/techmanualqa#covered-manuals).
Name each PDF after the `doc_id` in the [CC BY 4.0 labels archive](https://zenodo.org/records/17410809).
The importer reports which filenames are missing. It selects questions for the
PDFs you supplied and checks that each labelled answer page has extractable text.
Page numbers are one-based PDF page positions.

For example, the [Makita manual](https://cdn.makitatools.com/apps/cms/doc/prod/XFD/e2dc6a61-63d3-4455-b4dc-940ffa84e99d_XFD12_IM_885510-943.pdf)
indexes 20 pages and enables 35 cases.

```bash
python -m pip install -r requirements/requirements-benchmarks.txt
mkdir -p data/manuals
curl -fL -o data/manuals/makita_drill.pdf \
  https://cdn.makitatools.com/apps/cms/doc/prod/XFD/e2dc6a61-63d3-4455-b4dc-940ffa84e99d_XFD12_IM_885510-943.pdf
python -m rag_engine.benchmarks.prepare techmanualqa --pdf-dir data/manuals
```

The LinuxCNC cases are authored in
[linuxcnc.py](../src/rag_engine/benchmarks/linuxcnc.py). The corpus is retrieved
from the version 2.9 [stepper diagnostics](https://linuxcnc.org/docs/2.9/html/config/stepper-diagnostics.html)
and [latency testing](https://linuxcnc.org/docs/2.9/html/install/latency-test.html)
documentation. Citations include the original section anchors.

## Compare retrieval with a fixed generator

Example commands for a seeded 40-question TechQA sample. The sample includes
both answerable and unanswerable cases. Omit `--limit` to run every question.
The MiniLM model is used here for a small local embedding setup. The build and
the benchmark must use the same embedding model.

```bash
dataset=data/benchmarks/techqa
embedding=sentence-transformers/all-MiniLM-L6-v2

python -m rag_engine.cli.index --docs-root "$dataset/docs" \
  --database-path "$dataset/index" --collection benchmark --embedding-model "$embedding"
python -m rag_engine.benchmarks.run "$dataset" --embedding-model "$embedding" \
  --limit 40 --seed 42 --model qwen3:4b --output output/techqa-qwen3.json
```

The two retrieval conditions share the query, stored vectors, candidate count,
optional reranker, file limit, and text budget. Set `--reranker cross_encoder`
to apply reranking to both conditions. The reference condition supplies labelled
documents or pages within the same text budget. It can still omit relevant
material from long references that are clipped to fit.

Omit `--model` to run retrieval without Ollama. Use a new output filename for
each run. Build a fresh collection after changing the corpus or chunking. Older
collections created before text was stored with vectors need a fresh build.

## Read the results

- **Gold source recall** averages the fraction of labelled articles, pages, or
  sections present in each answerable case's selected evidence. It measures
  source coverage. Review the saved passage text for coverage of required facts.
- **Answerability** reports how often the model answered answerable questions
  and abstained on unanswerable ones. The reference condition has evidence labels
  only for answerable questions.
- **Answer quality** requires reviewing the claims against the cited passages and
  the reference answer. Check the procedure, applicability, prerequisites, and
  step order. Mark the answer as a pass only when it covers the required points
  and avoids consequential errors. For unanswerable cases, require an
  abstention or a request for the missing detail.

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
