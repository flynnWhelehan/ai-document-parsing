# AI Document Parsing - A Playground

A local-first, modular playground for exploring emerging AI models focused on document understanding and parsing.

This repo is to test, compare, and integrate models from [**HuggingFace**](https://huggingface.co/), [**MistralAI**](https://mistral.ai/) and others. It will start local and later go into APIs, containerization, and a GUI interface for document uploads and switching models.

---

## Some Project Goals (in loose priority order)

- Run and evaluate lightweight document-focused AI models locally.
- Have clean model wrappers with testable interfaces.
- Expose models as APIs.
- Containerize the API and model environment (Docker).
- .NET system that consumes the API and performs further processing.
- GUI to upload PDFs and select models.

---

## Directory Structure

| Path | Description |
| - | - |
| `models/` | Wrappers for document-focused AI models (e.g. SmolDocling). |
| `examples/` | Experiments with models. |
| `api/` | API app to expose models via HTTP API. |
| `tests/` | Unit and integration tests. |

---

## Quickstart

This section will provide a user manual when the project matures.

For now:
```bash
pip install -r requirements.txt
Run models/download_smol_docling.py
Run examples/smol_docling/run_smol_docling
Assign directory variables to match paths
