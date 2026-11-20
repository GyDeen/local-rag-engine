# Local RAG Engine

A local retrieval-augmented generation engine for technical support questions. It
indexes manuals and support articles, retrieves evidence, and uses Ollama to
generate answers with citations for each claim or procedure step.

The benchmark compares individual chunks, stitched sections, and reference
evidence on the same questions. It captures the exact text passed to the model
so you can review whether an answer followed the applicable procedure.

## Run LinuxCNC locally

Use Python 3.14 and run these commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/requirements.txt
export PYTHONPATH=src

python -m rag_engine.benchmarks.prepare linuxcnc

dataset=data/benchmarks/linuxcnc
embedding=sentence-transformers/all-MiniLM-L6-v2
python -m rag_engine.cli.index --docs-root "$dataset/docs" \
  --database-path "$dataset/index" --collection benchmark --embedding-model "$embedding"
python -m rag_engine.benchmarks.run "$dataset" --embedding-model "$embedding" \
  --output output/linuxcnc-retrieval.json
```

Start Ollama with `ollama serve` in a separate terminal and pull `qwen3:4b` with
`ollama pull qwen3:4b`. Ask a free-form question using the same index.

```bash
python -m rag_engine.benchmarks.run "$dataset" --embedding-model "$embedding" \
  --model qwen3:4b \
  --question 'Why can LinuxCNC 2.9 report a following error on an open-loop stepper?'
```

## Benchmark sources

| Source | Indexed evidence | Evaluation labels |
| --- | --- | --- |
| [NVIDIA TechQA-RAG-Eval](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval) | 28,481 IBM technical support articles | 910 questions, including 300 marked unanswerable |
| [TechManualQA-700](https://zenodo.org/records/17410809) | Pages from 20 manufacturer manuals | 700 questions with page references, including 100 procedural and 100 unanswerable cases |
| [LinuxCNC 2.9](https://linuxcnc.org/docs/2.9/html/config/stepper-diagnostics.html) | Nine sections from the stepper diagnostics and latency testing docs | Ten project-authored cases covering diagnostics, missing machine details, and version applicability |

[Benchmark setup and evaluation](docs/benchmarks.md) covers the full import
commands, paired experiments, and how to audit answer quality.

## How it works

1. Split documents at headings while keeping procedures and related text together.
2. Embed each chunk and store its text and offsets with its vector in Chroma.
3. Retrieve candidates and optionally rerank them with a cross-encoder.
4. Compare the highest-ranked individual chunk from each selected file with
   evidence stitched from chunks in the same section.
5. Generate from the selected evidence and map each citation to a source path,
   section, or PDF page.

[Service setup](SETUP.md) covers the catalogue-based HTTP API, Docker and the
fictional alarm examples used to exercise that API. See [LICENSE](LICENSE) for
the software terms. The datasets and source manuals have their own terms.
