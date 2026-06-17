# doc_pipeline

A reusable document processing pipeline that combines **Docling** (structured document parsing + GPU-accelerated figure extraction), a **Vision Language Model** (VLM, local via Ollama), and a **text LLM** (structured JSON extraction) into a single pluggable five-stage flow.

Build any document-intelligence app by keeping the pipeline skeleton and only swapping the extraction schema and the output renderer.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Project structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick start — three real examples](#quick-start--three-real-examples)
6. [Configuration — every field explained](#configuration--every-field-explained)
7. [Running the pipeline](#running-the-pipeline)
8. [Editing the input](#editing-the-input)
9. [Editing the output](#editing-the-output)
10. [State object reference](#state-object-reference)
11. [Building a new app from scratch](#building-a-new-app-from-scratch)

---

## How it works

```
PDF / DOCX / PPTX
        │
        ▼  Stage 1 — Docling Ingest
        │  Parses the document into structured text.
        │  GPU-accelerated (CUDA auto-detected, CPU fallback).
        │  Every figure is extracted as a high-res PNG image.
        │  A [[FIGURE_N]] placeholder is inserted at each figure's exact position.
        │
        ▼  Stage 2 — VLM (Vision Language Model)
        │  Each [[FIGURE_N]] image is sent to an Ollama vision model (llava, …).
        │  The model returns a detailed plain-text description of the figure.
        │
        ▼  Stage 3 — Recombine
        │  [[FIGURE_N]] placeholders are replaced by the VLM descriptions.
        │  Result: combined_text — the full document, figures included, as plain text.
        │
        ▼  Stage 4 — LLM Extraction
        │  combined_text is sent to a text LLM with your JSON schema.
        │  The LLM returns a structured JSON object — one per document.
        │
        ▼  Stage 5 — Output
           Your renderer function formats the final result in any shape:
           markdown, JSON, HTML, database write, file save — anything.
```

Figures are never discarded. They travel through the pipeline and land back at their
exact original position in the document as structured text the extraction LLM can
reason over, including data from charts, captions from diagrams, and text from tables
rendered as images.

---

## Project structure

```
doc_pipeline/
├── pyproject.toml
├── requirements.txt
├── README.md
│
├── core/
│   ├── state.py          PipelineState, ParsedDocument, Figure, ExtractedData
│   ├── config.py         PipelineConfig, LLMConfig, VLMConfig, DoclingConfig, ExtractionConfig
│   └── pipeline.py       DocumentPipeline orchestrator
│
├── stages/
│   ├── base.py           BaseStage abstract class — implement this to add a custom stage
│   ├── ingest.py         DoclingIngestStage — parse + extract figures (GPU/CPU)
│   ├── vlm.py            VLMStage — describe figures with Ollama vision model
│   ├── recombine.py      RecombineStage — splice VLM text back into document
│   ├── extract.py        ExtractStage — LLM JSON extraction from combined_text
│   └── output.py         OutputStage — pluggable final renderer
│
├── providers/
│   └── llm.py            Ollama / OpenAI / LM Studio client wrappers
│
└── examples/
    ├── talent_match.py       CV ranking report
    ├── invoice_extractor.py  Payment summary JSON
    ├── contract_reviewer.py  Legal risk brief
    └── fastapi_app.py        FastAPI HTTP wrapper
```

---

## Requirements

| Requirement | Version | Purpose |
|---|---|---|
| Python | ≥ 3.11 | `match` statements, union type hints |
| [Ollama app](https://ollama.com/download) | latest | Runs VLM + text LLM locally, free |
| `docling` | ≥ 2.0.0 | Document parsing + GPU figure extraction |
| `Pillow` | ≥ 10.0.0 | Encode extracted figures as PNG bytes |
| `ollama` (Python) | ≥ 0.2.0 | Ollama API client |
| `pydantic` | ≥ 2.7.0 | Config + schema validation |
| PyTorch + CUDA | see below | GPU acceleration for Docling (optional) |

Optional — needed only for specific features:

| Package | When needed |
|---|---|
| `fastapi`, `uvicorn`, `python-multipart` | FastAPI HTTP wrapper |
| `openai` | OpenAI GPT-4o or LM Studio as text LLM |

---

## Installation

### Step 1 — Check your GPU (optional but recommended)

```powershell
nvidia-smi   # shows CUDA version in top-right corner, e.g. "CUDA Version: 12.4"
```

If you have a CUDA-capable GPU, install PyTorch with the matching CUDA version.
This makes Docling's layout detection and OCR run on GPU — significantly faster for large PDFs.

```powershell
# CUDA 12.1 (most common for RTX 30xx / 40xx):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (older cards, GTX 10xx / 16xx / 20xx):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# No GPU / CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify CUDA is visible to PyTorch:

```python
import torch
print(torch.cuda.is_available())   # True = GPU ready
print(torch.cuda.get_device_name(0))  # e.g. "NVIDIA GeForce RTX 3080"
```

### Step 2 — Install the Ollama application

Download and install from **[https://ollama.com/download](https://ollama.com/download)** (Windows installer — no config needed).

After installation open a **new** terminal and pull the models:

```powershell
# Vision model — Stage 2 (choose one based on your GPU VRAM)
ollama pull llava             # 7 B  — 8 GB VRAM, recommended default
ollama pull moondream         # 1.7 B — 4 GB VRAM, fastest
ollama pull minicpm-v         # 8 B  — 8 GB VRAM, best quality
ollama pull llama3.2-vision   # 11 B — 12 GB VRAM, most accurate

# Text model — Stage 4 (choose one)
ollama pull mistral           # 7 B  — 8 GB VRAM, fast, good JSON
ollama pull llama3            # 8 B  — 8 GB VRAM, more accurate
ollama pull phi3              # 3.8 B — 4 GB VRAM, lightweight
```

Verify Ollama is running and models are ready:

```powershell
ollama list    # lists installed models with sizes
ollama ps      # lists currently loaded models
ollama run mistral "say hi"   # quick sanity check
```

### Step 3 — Install Python dependencies

```powershell
# From d:\TalentMatchMicroServices\  (the parent of doc_pipeline/)
pip install -e doc_pipeline

# Or from inside doc_pipeline/:
pip install -r requirements.txt
```

> If you are working inside the TalentMatch virtualenv, `docling`, `ollama`, `Pillow`,
> and `pydantic` are already installed — only PyTorch and the Ollama app are new.

### Step 4 — Verify the installation

```python
# verify_install.py — run this to confirm everything is wired up
import asyncio, base64
from doc_pipeline import DocumentPipeline, PipelineConfig, LLMConfig, VLMConfig

async def check():
    pipeline = DocumentPipeline(PipelineConfig(
        vlm=VLMConfig(model="moondream"),   # smallest model for a quick test
        llm=LLMConfig(model="mistral"),
    ))
    files = [{
        "name": "test.txt",
        "content_base64": base64.b64encode(b"Hello, this is a test document.").decode(),
        "mime_type": "text/plain",
    }]
    state = await pipeline.run(files, query="test")
    print("OK — accelerator used:", state.documents[0].metadata.get("accelerator_used"))
    print("Output:\n", state.output)

asyncio.run(check())
```

---

## Quick start — three real examples

Each example is fully self-contained with an embedded sample document.
No files on disk required.

---

### Example 1 — CV / Resume analyser

```python
import asyncio
import base64
import json
from doc_pipeline import (
    DocumentPipeline, PipelineConfig,
    LLMConfig, VLMConfig, ExtractionConfig,
)
from doc_pipeline.core.state import PipelineState

SAMPLE_CV = b"""
Jane Smith  —  jane.smith@email.com  —  +216 22 333 444  —  Tunis, Tunisia

PROFESSIONAL SUMMARY
Senior Python developer with 8 years of experience building high-throughput
microservices and data pipelines. Led teams of up to 6 engineers.

SKILLS
Python (expert) · FastAPI (advanced) · PostgreSQL (advanced) · Docker (advanced)
Redis · Kafka · AWS (EC2, S3, Lambda) · Machine Learning (intermediate)

EXPERIENCE
Lead Backend Engineer — Acme Corp, Tunis  (2021–present, 3 years)
  Built microservices platform handling 50,000 requests/second.
  Reduced p99 latency from 400 ms to 38 ms via query optimisation.

Python Developer — StartupXYZ, Remote  (2019–2021, 2 years)
  Designed REST APIs with FastAPI and SQLAlchemy serving 200k daily users.

Junior Developer — WebCo, Tunis  (2016–2019, 3 years)
  Full-stack Django applications for e-commerce clients.

EDUCATION
BSc Computer Science — University of Tunis el Manar  (2016)

CERTIFICATIONS
AWS Solutions Architect Associate (2022)
"""

CV_SCHEMA = {
    "name": "string",
    "email": "string",
    "phone": "string",
    "location": "string",
    "summary": "string — 1-2 sentence professional summary",
    "experience_years": "number — total years of professional experience",
    "seniority_level": "one of: junior | mid | senior | lead | executive",
    "skills": [
        {"name": "string", "level": "one of: beginner | intermediate | advanced | expert"}
    ],
    "experiences": [
        {
            "title": "string",
            "company": "string",
            "duration_years": "number",
            "highlight": "string — most impressive achievement in one sentence",
        }
    ],
    "educations": [{"degree": "string", "institution": "string", "year": "number"}],
    "certifications": [{"name": "string", "year": "number or null"}],
    "figure_insights": "string — any charts or visual elements found in the CV",
}

async def run():
    pipeline = DocumentPipeline(
        PipelineConfig(
            vlm=VLMConfig(model="llava"),
            llm=LLMConfig(model="mistral"),
            extraction=ExtractionConfig(schema_definition=CV_SCHEMA),
        )
    )
    files = [{
        "name": "jane_smith_cv.txt",
        "content_base64": base64.b64encode(SAMPLE_CV).decode(),
        "mime_type": "text/plain",
    }]
    state = await pipeline.run(
        files,
        query="Senior Python backend developer with microservices experience",
    )
    doc = state.documents[0]
    ext = state.extractions[0]
    print(f"Pages: {doc.page_count}  |  Figures: {len(doc.figures)}")
    print(json.dumps(ext.data, indent=2, ensure_ascii=False))

asyncio.run(run())
```

**Expected output:**

```json
{
  "name": "Jane Smith",
  "email": "jane.smith@email.com",
  "phone": "+216 22 333 444",
  "location": "Tunis, Tunisia",
  "summary": "Senior Python developer with 8 years of experience in high-throughput microservices.",
  "experience_years": 8,
  "seniority_level": "senior",
  "skills": [
    {"name": "Python",     "level": "expert"},
    {"name": "FastAPI",    "level": "advanced"},
    {"name": "PostgreSQL", "level": "advanced"},
    {"name": "Docker",     "level": "advanced"},
    {"name": "AWS",        "level": "intermediate"}
  ],
  "experiences": [
    {"title": "Lead Backend Engineer", "company": "Acme Corp",   "duration_years": 3,
     "highlight": "Reduced p99 latency from 400 ms to 38 ms via query optimisation."},
    {"title": "Python Developer",      "company": "StartupXYZ",  "duration_years": 2,
     "highlight": "Designed REST APIs serving 200k daily users."}
  ],
  "educations": [{"degree": "BSc Computer Science", "institution": "University of Tunis el Manar", "year": 2016}],
  "certifications": [{"name": "AWS Solutions Architect Associate", "year": 2022}],
  "figure_insights": "No visual figures detected in this document."
}
```

---

### Example 2 — Invoice processor

```python
import asyncio
import base64
import json
from datetime import datetime
from doc_pipeline import DocumentPipeline, PipelineConfig, LLMConfig, VLMConfig, ExtractionConfig
from doc_pipeline.core.state import PipelineState

SAMPLE_INVOICE = b"""
=====================================
         INVOICE
=====================================
Invoice Number : INV-2026-0089
Vendor         : TechSupply SARL
Vendor email   : billing@techsupply.tn
Invoice Date   : 2026-06-01
Due Date       : 2026-06-30
Currency       : TND

ITEMS
1. Annual Software License     x1     4,800.00 TND
2. Priority Support Package    x1     1,200.00 TND
3. Onboarding Training (2 days)x2       900.00 TND

Subtotal  6,900.00 TND   |   VAT 19%: 1,311.00 TND   |   TOTAL: 8,211.00 TND

Payment Terms : Net 30
Bank IBAN     : TN59 1234 5678 9012 3456 7890
PO Reference  : PO-2026-112
Status        : pending
=====================================
"""

INVOICE_SCHEMA = {
    "invoice_number": "string",
    "vendor_name": "string",
    "vendor_email": "string or null",
    "invoice_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD",
    "currency": "ISO code e.g. TND USD EUR",
    "subtotal": "number",
    "vat_amount": "number",
    "total_amount": "number",
    "line_items": [
        {"description": "string", "quantity": "number", "unit_price": "number", "total": "number"}
    ],
    "payment_terms": "string",
    "bank_iban": "string or null",
    "purchase_order_ref": "string or null",
    "status": "one of: paid | pending | overdue",
    "visual_elements": "string — stamps, QR codes, logos found in figures",
}

async def run():
    async def renderer(state: PipelineState, config) -> dict:
        today = datetime.utcnow().date()
        invoices = []
        for ext in state.extractions:
            inv = dict(ext.data)
            inv["_source"] = ext.document_name
            if inv.get("due_date") and inv.get("status") != "paid":
                try:
                    if datetime.strptime(inv["due_date"], "%Y-%m-%d").date() < today:
                        inv["status"] = "overdue"
                except ValueError:
                    pass
            invoices.append(inv)
        total_due = sum(i.get("total_amount", 0) for i in invoices if i.get("status") != "paid")
        return {"invoices": invoices, "total_due": round(total_due, 2)}

    pipeline = DocumentPipeline(
        PipelineConfig(
            vlm=VLMConfig(model="llava"),
            llm=LLMConfig(model="mistral"),
            extraction=ExtractionConfig(schema_definition=INVOICE_SCHEMA),
        ),
        renderer=renderer,
    )
    files = [{
        "name": "INV-2026-0089.txt",
        "content_base64": base64.b64encode(SAMPLE_INVOICE).decode(),
        "mime_type": "text/plain",
    }]
    state = await pipeline.run(files)
    print(json.dumps(state.output, indent=2, ensure_ascii=False))

asyncio.run(run())
```

**Expected output:**

```json
{
  "invoices": [{
    "invoice_number": "INV-2026-0089",
    "vendor_name": "TechSupply SARL",
    "invoice_date": "2026-06-01",
    "due_date": "2026-06-30",
    "currency": "TND",
    "subtotal": 6900.0,
    "vat_amount": 1311.0,
    "total_amount": 8211.0,
    "line_items": [
      {"description": "Annual Software License",      "quantity": 1, "unit_price": 4800.0, "total": 4800.0},
      {"description": "Priority Support Package",     "quantity": 1, "unit_price": 1200.0, "total": 1200.0},
      {"description": "Onboarding Training (2 days)", "quantity": 2, "unit_price":  450.0, "total":  900.0}
    ],
    "payment_terms": "Net 30",
    "bank_iban": "TN59 1234 5678 9012 3456 7890",
    "status": "pending",
    "_source": "INV-2026-0089.txt"
  }],
  "total_due": 8211.0
}
```

---

### Example 3 — Research paper with figure extraction

When a real PDF is provided, the VLM describes every chart and diagram;
those descriptions are merged back into the text before extraction.

```python
import asyncio
import base64
from doc_pipeline import (
    DocumentPipeline, PipelineConfig,
    LLMConfig, VLMConfig, DoclingConfig, ExtractionConfig,
)
from doc_pipeline.core.state import PipelineState

SAMPLE_PAPER = b"""
Title: Efficient Transformer Architectures for Low-Resource NLP
Authors: A. Ben Ali, S. Trabelsi, M. Chakroun  —  ENIT, Tunis
Published: NeurIPS 2026 Workshop on Efficient ML

ABSTRACT
We propose LiteFormer achieving 94.2% of BERT-base accuracy on GLUE
with 60% fewer parameters and 3.8x faster CPU inference.

RESULTS
Table 1: GLUE Benchmark
Model       | MNLI  | QQP   | SST-2 | Avg
BERT-base   | 84.6  | 91.3  | 93.5  | 89.8
DistilBERT  | 82.1  | 88.5  | 91.3  | 87.3
LiteFormer  | 84.0  | 90.7  | 93.1  | 89.3

Figure 1: Inference speed vs accuracy scatter plot — LiteFormer top-left quadrant.

Keywords: transformer, efficiency, low-resource NLP
"""

PAPER_SCHEMA = {
    "title": "string",
    "authors": ["string"],
    "venue": "string — conference or journal",
    "abstract": "string",
    "key_findings": ["string — each main result including numbers from tables or charts"],
    "figure_descriptions": ["string — what each figure or chart shows"],
    "keywords": ["string"],
}

async def run():
    async def renderer(state: PipelineState, config) -> str:
        data = state.extractions[0].data
        doc  = state.documents[0]
        lines = [
            f"# {data.get('title')}",
            f"**Authors:** {', '.join(data.get('authors', []))}",
            f"**Venue:** {data.get('venue')}",
            "", "## Abstract", data.get("abstract", ""),
            "", "## Key Findings",
            *[f"- {f}" for f in data.get("key_findings", [])],
            "", "## Figures (VLM descriptions)",
        ]
        if doc.figures:
            for fig in doc.figures:
                lines.append(f"- **{fig.placeholder}** p.{fig.page}: {fig.vlm_description}")
        else:
            for fd in data.get("figure_descriptions", []):
                lines.append(f"- {fd}")
        return "\n".join(lines)

    pipeline = DocumentPipeline(
        PipelineConfig(
            docling=DoclingConfig(images_scale=2.0, accelerator="auto"),
            vlm=VLMConfig(
                model="llava",
                prompt=(
                    "This is a figure from a scientific paper. State the chart type, "
                    "axis labels, units, data series, key values, and the conclusion it supports."
                ),
            ),
            llm=LLMConfig(model="mistral"),
            extraction=ExtractionConfig(schema_definition=PAPER_SCHEMA),
        ),
        renderer=renderer,
    )
    files = [{
        "name": "liteformer.txt",
        "content_base64": base64.b64encode(SAMPLE_PAPER).decode(),
        "mime_type": "text/plain",
    }]
    state = await pipeline.run(files)
    print(state.output)

asyncio.run(run())
```

---

## Configuration — every field explained

### `DoclingConfig` — document parsing

```python
from doc_pipeline import DoclingConfig

DoclingConfig(
    # Resolution of extracted figure images.
    # Higher = better VLM descriptions but more memory and slower.
    # 1.0 = screen resolution, 2.0 = default (recommended), 4.0 = print quality
    images_scale = 2.0,

    # Enable OCR for scanned PDFs where text is embedded in images (e.g. scanned contracts).
    # Disabled by default because it is slow even on GPU.
    ocr_enabled = False,

    # GPU accelerator for Docling's layout detection and OCR models.
    # "auto"  — auto-detect CUDA; falls back to CPU if not found  [DEFAULT]
    # "cuda"  — force CUDA GPU; falls back to CPU if CUDA init fails
    # "cpu"   — always use CPU (safe, slower for large PDFs)
    # "mps"   — Apple Silicon GPU (Mac only)
    accelerator = "auto",

    # Number of CPU threads used when running on CPU or for CPU-bound parts.
    num_threads = 4,
)
```

**When to change `accelerator`:**

| Situation | Setting |
|---|---|
| You have a GPU and want maximum speed | `"auto"` (default) or `"cuda"` |
| You want to force CPU to save GPU memory for the VLM | `"cpu"` |
| You get CUDA out-of-memory errors | `"cpu"` |
| You are on a Mac with Apple Silicon | `"mps"` |

---

### `VLMConfig` — vision language model

```python
from doc_pipeline import VLMConfig

VLMConfig(
    # Set False to skip Stage 2 entirely (no VLM calls, no figure descriptions).
    # Use when your documents have no figures, or when speed is critical.
    enabled = True,

    # Ollama vision model name.
    # Must be pulled first: ollama pull <model>
    # Options by VRAM:
    #   4 GB VRAM  → "moondream"
    #   6 GB VRAM  → "llava-phi3"
    #   8 GB VRAM  → "llava" (default) or "minicpm-v"
    #   12 GB VRAM → "llama3.2-vision"
    model = "llava",

    # Ollama server address. Change only if Ollama runs on another machine.
    base_url = "http://localhost:11434",

    # Prompt sent to the VLM for every figure.
    # Customise this to focus on what matters for your documents.
    # Available context injected automatically before this prompt:
    #   "Figure caption: <caption>\n(Page N of the document)\n\n"
    prompt = (
        "Describe this figure in full detail. "
        "If it is a chart or graph, state its type, axes, values, and trends. "
        "If it is a diagram, describe all components and relationships. "
        "If it is a table image, transcribe it. "
        "Be precise — your description replaces the image in a document analysis pipeline."
    ),

    # Number of figures processed in parallel.
    # Keep at 1 per GPU. Increase only if you have multiple GPUs or use CPU.
    concurrency = 1,
)
```

**VLM prompt examples for specific document types:**

```python
# For financial reports (charts with numbers)
prompt = (
    "This figure is from a financial report. "
    "State the chart type. List all axis labels and units. "
    "Extract all data values visible in the chart. "
    "State the time period covered and the main trend."
)

# For technical diagrams / architecture diagrams
prompt = (
    "This is a technical or architectural diagram. "
    "List all components, services, or boxes visible. "
    "Describe all connections, arrows, and relationships between them. "
    "Note any labels, protocols, or flow directions."
)

# For medical / scientific images
prompt = (
    "This is a scientific figure. "
    "Identify the type of image (microscopy, MRI, gel, plot, etc.). "
    "Describe all visible structures, labels, scale bars, and measurements. "
    "State what the figure is intended to demonstrate."
)

# Fast / minimal (for speed-sensitive pipelines)
prompt = "Briefly describe what this image shows in 1-2 sentences."
```

---

### `LLMConfig` — text extraction LLM

```python
from doc_pipeline import LLMConfig

LLMConfig(
    # LLM provider.
    # "ollama"   — local Ollama (default, free, private)
    # "openai"   — OpenAI API (GPT-4o, GPT-3.5)
    # "lmstudio" — LM Studio local OpenAI-compatible server
    provider = "ollama",

    # Model name (must match provider).
    # Ollama:   "mistral", "llama3", "phi3", "gemma2", ...
    # OpenAI:   "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
    # LMStudio: whatever model you have loaded
    model = "mistral",

    # Ollama or LM Studio server URL.
    # For OpenAI: leave as default (the OpenAI client ignores it).
    base_url = "http://localhost:11434",

    # API key — required for OpenAI provider, ignored for Ollama.
    api_key = None,

    # Lower = more deterministic JSON. 0.0 for maximum consistency.
    temperature = 0.1,

    # Maximum tokens in the LLM response. Increase for very long schemas.
    max_tokens = 4096,

    # Request timeout in seconds. Increase for very long documents.
    timeout = 120,
)
```

**Provider-specific examples:**

```python
# OpenAI GPT-4o (best accuracy, costs money)
LLMConfig(provider="openai", model="gpt-4o", api_key="sk-...")

# OpenAI GPT-4o-mini (cheaper, still very good)
LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-...")

# LM Studio (free, local, OpenAI-compatible)
LLMConfig(
    provider="lmstudio",
    model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

# Ollama with a larger model for better accuracy
LLMConfig(provider="ollama", model="llama3", base_url="http://localhost:11434")
```

---

### `ExtractionConfig` — what to extract

```python
from doc_pipeline import ExtractionConfig

ExtractionConfig(
    # JSON schema dict — describes every field the LLM should extract.
    # Format: { "field_name": "type — description of what to put here" }
    # For lists: [ {"field": "type"} ] (a list with one example dict)
    schema_definition = {
        "field_name": "type — description",
    },

    # System prompt sent to the LLM before the document.
    # Override when you need domain-specific instructions.
    system_prompt = (
        "You are a precise data-extraction assistant. "
        "Return ONLY valid JSON that matches the schema. No prose, no markdown fences."
    ),

    # Template for the user message. Two placeholders are required:
    #   {document_text} — the combined_text (text + VLM figure descriptions)
    #   {schema}        — your schema_definition serialised as JSON
    user_prompt_template = (
        "Extract information from the following document.\n\n"
        "Document:\n{document_text}\n\n"
        "Return a JSON object matching this schema:\n{schema}"
    ),

    # If True (default): return {} for a document if the LLM returns invalid JSON,
    #   and record the error in ext.error — the pipeline never crashes.
    # If False: raise an exception on parse failure.
    fallback_on_error = True,
)
```

**Schema writing guide:**

```python
# Simple string fields
"title": "string"
"status": "one of: active | inactive | pending"
"date":   "YYYY-MM-DD or null"

# Numbers
"revenue": "number in millions USD"
"page_count": "integer"

# Booleans
"is_signed": "boolean — true if a signature is present"

# Lists of strings
"keywords": ["string — one keyword per item"]
"risks":    ["string — each risk in one sentence"]

# Lists of objects
"line_items": [
    {
        "description": "string",
        "quantity":    "number",
        "unit_price":  "number",
        "total":       "number",
    }
]

# Nested objects
"author": {
    "name":  "string",
    "email": "string or null",
    "affiliation": "string or null",
}

# Tip: always add a "figures" field to capture VLM insights
"visual_content": "string — describe any charts, tables, or diagrams found in the document"
```

---

### `PipelineConfig` — top-level

```python
from doc_pipeline import PipelineConfig

PipelineConfig(
    docling   = DoclingConfig(...),     # Stage 1
    vlm       = VLMConfig(...),         # Stage 2
    llm       = LLMConfig(...),         # Stage 4
    extraction = ExtractionConfig(...), # Stage 4 schema

    # Number of documents processed in parallel during ingest and extract.
    # Reduce to 1 if you run out of GPU memory when processing many files at once.
    max_concurrency = 2,
)
```

---

## Running the pipeline

### From a Python script

```python
# run_pipeline.py
import asyncio
import base64
from doc_pipeline import DocumentPipeline, PipelineConfig, LLMConfig, VLMConfig, ExtractionConfig

MY_SCHEMA = {"title": "string", "summary": "string"}

pipeline = DocumentPipeline(
    PipelineConfig(
        vlm=VLMConfig(model="llava"),
        llm=LLMConfig(model="mistral"),
        extraction=ExtractionConfig(schema_definition=MY_SCHEMA),
    )
)

async def main():
    with open("my_document.pdf", "rb") as f:
        files = [{
            "name": "my_document.pdf",
            "content_base64": base64.b64encode(f.read()).decode(),
            "mime_type": "application/pdf",
        }]
    state = await pipeline.run(files, query="")
    print(state.output)
    print(state.extractions[0].data)

asyncio.run(main())
```

Run it:

```powershell
python run_pipeline.py
```

### From the command line using the built-in examples

```powershell
# From d:\TalentMatchMicroServices\

python -m doc_pipeline.examples.talent_match
python -m doc_pipeline.examples.invoice_extractor
python -m doc_pipeline.examples.contract_reviewer
```

### As a FastAPI HTTP server

```powershell
# Start the server (auto-reloads on code changes with --reload)
uvicorn doc_pipeline.examples.fastapi_app:app --reload --port 8080

# Check it is running
curl http://localhost:8080/health
```

Send a file:

```powershell
# PowerShell
Invoke-RestMethod `
  -Uri "http://localhost:8080/process" `
  -Method POST `
  -Form @{ query = "Senior Python developer"; files = Get-Item ".\cv.pdf" }

# curl (bash or Git Bash)
curl -X POST http://localhost:8080/process \
  -F "query=Senior Python developer" \
  -F "files=@cv.pdf"
```

Server response:

```json
{
  "session_id": "3f8a2c1d-...",
  "output": "...your renderer output...",
  "figures_total": 3,
  "errors": []
}
```

### Checking GPU is being used

After a run, inspect the document metadata:

```python
state = await pipeline.run(files)
for doc in state.documents:
    print(doc.name, "→ accelerator:", doc.metadata.get("accelerator_used"))
    if "accelerator_fallback_reason" in doc.metadata:
        print("  Fell back to CPU because:", doc.metadata["accelerator_fallback_reason"])
```

Expected output when GPU is working:

```
my_document.pdf → accelerator: auto
```

Expected output when GPU failed and CPU fallback was used:

```
my_document.pdf → accelerator: cpu
  Fell back to CPU because: RuntimeError: CUDA out of memory
```

---

## Editing the input

### Input file format

Each file in the `files` list is a plain Python dict with three required keys:

```python
{
    "name":           "filename.pdf",          # string — used as document identifier
    "content_base64": "JVBERi0xLjQ...",        # string — base64-encoded file bytes
    "mime_type":      "application/pdf",       # string — tells Docling which parser to use
}
```

### Loading a single file from disk

```python
import base64

def load_file(path: str, mime_type: str | None = None) -> dict:
    mime_map = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".html": "text/html",
        ".txt":  "text/plain",
    }
    import os
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        content = f.read()
    return {
        "name":           os.path.basename(path),
        "content_base64": base64.b64encode(content).decode(),
        "mime_type":      mime_type or mime_map.get(ext, "application/octet-stream"),
    }

# Usage
files = [load_file(r"C:\Documents\report.pdf")]
state = await pipeline.run(files, query="quarterly revenue")
```

### Loading multiple files from a folder

```python
import glob

# All PDFs in a folder
files = [load_file(p) for p in glob.glob(r"C:\invoices\*.pdf")]

# Mix of PDF and DOCX
files = [
    load_file(p)
    for pattern in [r"C:\contracts\*.pdf", r"C:\contracts\*.docx"]
    for p in glob.glob(pattern)
]

state = await pipeline.run(files, query="flag any unlimited liability clauses")
print(f"Processed {len(state.documents)} documents")
```

### Processing files one at a time (streaming pattern)

```python
import asyncio

async def process_folder(folder: str):
    import glob, os
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        files = [load_file(path)]
        state = await pipeline.run(files)
        yield state.extractions[0].data, path

async def main():
    async for data, path in process_folder(r"C:\reports"):
        print(path, "→", data.get("title"))

asyncio.run(main())
```

### The `query` parameter

`query` is a free-text string passed through the pipeline. Use it to:

- Provide context for the extraction LLM (e.g. a job description when matching CVs)
- Focus the output renderer on a specific aspect
- It appears in `state.query` and is available inside your renderer

```python
# For CV matching — the query is the job description
state = await pipeline.run(cv_files, query="Senior FastAPI developer, 5+ years, Tunis")

# For document Q&A
state = await pipeline.run(report_files, query="What was the Q3 revenue growth?")

# For batch processing where no context is needed
state = await pipeline.run(invoice_files, query="")
```

### MIME types reference

| File type | `mime_type` value |
|---|---|
| PDF | `application/pdf` |
| Word (.docx) | `application/vnd.openxmlformats-officedocument.wordprocessingml.document` |
| PowerPoint (.pptx) | `application/vnd.openxmlformats-officedocument.presentationml.presentation` |
| Excel (.xlsx) | `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` |
| HTML | `text/html` |
| Plain text | `text/plain` |

---

## Editing the output

The output is controlled entirely by the `renderer` coroutine you pass to `DocumentPipeline`.
It receives the full `PipelineState` after all four preceding stages have run,
and returns whatever you want stored in `state.output`.

### Default output (no renderer)

When no renderer is provided, the pipeline produces a markdown string with one section
per document showing the figure list and extracted JSON. Useful for debugging.

```python
pipeline = DocumentPipeline(config)   # no renderer
state = await pipeline.run(files)
print(state.output)   # markdown string
```

### Return a dict / JSON

```python
async def json_renderer(state, config) -> dict:
    return {
        "session": state.session_id,
        "documents": [
            {
                "name":    ext.document_name,
                "data":    ext.data,
                "figures": len(doc.figures),
                "pages":   doc.page_count,
            }
            for doc, ext in zip(state.documents, state.extractions)
        ],
        "errors": state.errors,
    }

pipeline = DocumentPipeline(config, renderer=json_renderer)
state = await pipeline.run(files)
# state.output is a dict — serialize with json.dumps(state.output)
```

### Return a markdown report

```python
async def markdown_renderer(state, config) -> str:
    lines = [f"# Report — {len(state.documents)} document(s)\n"]
    for doc, ext in zip(state.documents, state.extractions):
        lines += [
            f"## {doc.name}",
            f"Pages: {doc.page_count}  |  Figures: {len(doc.figures)}",
            "",
        ]
        for fig in doc.figures:
            lines.append(f"- **{fig.placeholder}** (p.{fig.page}): {fig.vlm_description[:150]}…")
        lines += ["", "**Extracted:**", f"```json\n{__import__('json').dumps(ext.data, indent=2)}\n```", ""]
    return "\n".join(lines)
```

### Save figures to disk inside the renderer

```python
import os

async def save_figures_renderer(state, config) -> dict:
    os.makedirs("extracted_figures", exist_ok=True)
    saved = []
    for doc in state.documents:
        for fig in doc.figures:
            if fig.image_bytes:
                fname = f"extracted_figures/{doc.name}_fig{fig.index}_p{fig.page}.png"
                with open(fname, "wb") as f:
                    f.write(fig.image_bytes)
                saved.append({"file": fname, "description": fig.vlm_description})
    return {
        "extracted_data": [e.data for e in state.extractions],
        "saved_figures": saved,
    }
```

### Write output to a file inside the renderer

```python
import json

async def file_renderer(state, config) -> str:
    result = {"extractions": [e.data for e in state.extractions]}
    output_path = f"output_{state.session_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return output_path   # state.output will be the file path

pipeline = DocumentPipeline(config, renderer=file_renderer)
state = await pipeline.run(files)
print("Saved to:", state.output)
```

### Access intermediate results directly

You do not need a custom renderer to read intermediate data — just inspect `state` after the run:

```python
state = await pipeline.run(files)

# Raw text Docling extracted (before VLM)
print(state.documents[0].text_with_placeholders)

# Text after VLM descriptions were spliced in
print(state.documents[0].combined_text)

# Per-figure VLM description
for fig in state.documents[0].figures:
    print(f"{fig.placeholder} (page {fig.page}): {fig.vlm_description}")

# Raw PNG bytes of a specific figure
fig = state.documents[0].figures[0]
with open("figure_0.png", "wb") as f:
    f.write(fig.image_bytes)

# JSON extracted by the LLM
print(state.extractions[0].data)

# Error that occurred during extraction (None if OK)
print(state.extractions[0].error)

# Which GPU/CPU was used by Docling
print(state.documents[0].metadata["accelerator_used"])
```

---

## State object reference

Complete list of every field available after `pipeline.run()`:

```python
state.session_id      # str  — unique UUID for this run
state.query           # str  — the query passed to pipeline.run()
state.errors          # list[str] — non-fatal errors (pipeline always completes)
state.output          # Any — return value of your renderer

# ── Per document ───────────────────────────────────────────────────────────────
doc = state.documents[0]           # ParsedDocument
doc.name                           # str — original filename
doc.page_count                     # int — total pages
doc.mime_type                      # str — MIME type as provided
doc.text_with_placeholders         # str — text with [[FIGURE_N]] markers (after Stage 1)
doc.combined_text                  # str — text + VLM descriptions (after Stage 3)
doc.metadata["accelerator_used"]   # str — "auto" / "cuda" / "cpu" / "mps"
doc.metadata.get("accelerator_fallback_reason")  # str or None

# ── Per figure ─────────────────────────────────────────────────────────────────
fig = doc.figures[0]               # Figure
fig.placeholder                    # str  — "[[FIGURE_0]]"
fig.index                          # int  — 0-based order in document
fig.page                           # int  — 1-based page number
fig.caption                        # str  — caption from Docling (may be empty)
fig.image_bytes                    # bytes — raw PNG
fig.image_base64                   # str  — base64 string (used internally by VLMStage)
fig.vlm_description                # str  — filled by VLMStage

# ── Per extraction ─────────────────────────────────────────────────────────────
ext = state.extractions[0]         # ExtractedData
ext.document_name                  # str  — matches doc.name
ext.data                           # dict — structured JSON from your schema
ext.raw_text                       # str  — the combined_text that was sent to the LLM
ext.error                          # str or None — error message if extraction failed
```

---

## Building a new app from scratch

### Minimum boilerplate

```python
import asyncio, base64, json
from doc_pipeline import DocumentPipeline, PipelineConfig, LLMConfig, VLMConfig, ExtractionConfig
from doc_pipeline.core.state import PipelineState

# 1. Define what to extract
MY_SCHEMA = {
    "field_one": "string — description",
    "field_two": "number",
    "field_three": ["string — list item"],
}

# 2. Define how to format the output
async def my_renderer(state: PipelineState, config) -> dict:
    return {
        "results": [e.data for e in state.extractions],
        "figures": [
            {"page": fig.page, "description": fig.vlm_description}
            for doc in state.documents
            for fig in doc.figures
        ],
    }

# 3. Build and run
pipeline = DocumentPipeline(
    PipelineConfig(
        vlm=VLMConfig(model="llava"),
        llm=LLMConfig(model="mistral"),
        extraction=ExtractionConfig(schema_definition=MY_SCHEMA),
    ),
    renderer=my_renderer,
)

async def main():
    with open("your_file.pdf", "rb") as f:
        files = [{"name": "your_file.pdf",
                  "content_base64": base64.b64encode(f.read()).decode(),
                  "mime_type": "application/pdf"}]
    state = await pipeline.run(files, query="your query here")
    print(json.dumps(state.output, indent=2))

asyncio.run(main())
```

### Replacing a full stage

Subclass `BaseStage` and swap it in for any of the five stages:

```python
from doc_pipeline.stages.base import BaseStage
from doc_pipeline.stages.vlm import VLMStage
from doc_pipeline.core.state import PipelineState
from doc_pipeline.core.config import PipelineConfig

class MyVLMStage(BaseStage):
    """Replace Ollama VLM with a cloud vision API."""
    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        for doc in state.documents:
            for fig in doc.figures:
                if fig.image_bytes:
                    fig.vlm_description = await my_cloud_vision_api(fig.image_bytes)
        return state

pipeline = DocumentPipeline(config)
pipeline.replace_stage(VLMStage, MyVLMStage())
```

### Supported document formats

| Format | Extension | Notes |
| --- | --- | --- |
| PDF | `.pdf` | Full text + figure extraction. Set `ocr_enabled=True` for scanned PDFs. |
| Word | `.docx` | Text + embedded images |
| PowerPoint | `.pptx` | Slides as text + slide images |
| Excel | `.xlsx` | Spreadsheet content |
| HTML | `.html` | Web pages |
| Plain text | `.txt` | No figure extraction |
