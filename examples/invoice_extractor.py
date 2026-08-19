"""
Example 2: Invoice Extractor
============================
Same pipeline — different schema and output.
Figures in invoices (e.g. logos, stamps, QR codes) are described by the VLM
and included in the extraction context.
"""
from __future__ import annotations
import asyncio
import base64
import json
from datetime import datetime

from doc_pipeline.core.pipeline import DocumentPipeline
from doc_pipeline.core.config import (
    PipelineConfig, LLMConfig, VLMConfig, DoclingConfig, ExtractionConfig,
)
from doc_pipeline.core.state import PipelineState


INVOICE_SCHEMA: dict = {
    "invoice_number": "string",
    "vendor_name": "string",
    "vendor_email": "string or null",
    "invoice_date": "YYYY-MM-DD or null",
    "due_date": "YYYY-MM-DD or null",
    "currency": "ISO 4217 code e.g. USD EUR TND",
    "subtotal": "number",
    "tax_amount": "number",
    "total_amount": "number",
    "line_items": [{"description": "string", "quantity": "number", "unit_price": "number", "total": "number"}],
    "payment_terms": "string or null",
    "status": "one of: paid | pending | overdue",
    "figures_found": "string — describe any stamps, signatures, QR codes or logos the VLM detected",
}


async def _invoice_renderer(state: PipelineState, config: PipelineConfig) -> dict:
    today = datetime.utcnow().date()
    invoices = []

    for ext in state.extractions:
        inv = dict(ext.data)
        inv["_source"] = ext.document_name
        inv["_figures"] = len([d for d in state.documents if d.name == ext.document_name][0].figures)

        if inv.get("due_date") and inv.get("status") != "paid":
            try:
                due = datetime.strptime(inv["due_date"], "%Y-%m-%d").date()
                if due < today:
                    inv["status"] = "overdue"
            except ValueError:
                pass

        invoices.append(inv)

    total_due = sum(
        i.get("total_amount", 0) or 0
        for i in invoices if i.get("status") in ("pending", "overdue")
    )

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "invoices": invoices,
        "summary": {
            "total_invoices": len(invoices),
            "total_due": round(total_due, 2),
            "overdue_count": sum(1 for i in invoices if i.get("status") == "overdue"),
            "paid_count": sum(1 for i in invoices if i.get("status") == "paid"),
        },
    }


def build_invoice_pipeline(
    text_model: str = "mistral",
    vlm_model: str = "llava",
) -> DocumentPipeline:
    config = PipelineConfig(
        docling=DoclingConfig(images_scale=1.0,accelerator="cpu",num_threads=4,ocr_enabled=False, ),
        vlm=VLMConfig(model=vlm_model),
        llm=LLMConfig(provider="ollama", model=text_model),
        extraction=ExtractionConfig(
            schema_definition=INVOICE_SCHEMA,
            system_prompt="Extract invoice data. For missing fields use null. Return ONLY valid JSON.",
        ),
    )
    return DocumentPipeline(config, renderer=_invoice_renderer)


if __name__ == "__main__":
    SAMPLE = b"""
    INVOICE #INV-2026-0042
    Vendor: Acme Software Ltd  |  vendor@acme.io
    Invoice Date: 2026-06-01   |  Due: 2026-06-30  |  Currency: USD
    Items:
      Software License (annual)  x1  $2,400.00
      Support Package            x1    $600.00
    Subtotal: $3,000.00  Tax (19%): $570.00  TOTAL: $3,570.00
    Payment Terms: Net 30  |  Status: pending
    """
    files = [{
        "name": "INV-2026-0042.txt",
        "content_base64": base64.b64encode(SAMPLE).decode(),
        "mime_type": "text/plain",
    }]
    pipeline = build_invoice_pipeline()
    state = asyncio.run(pipeline.run(files))
    print(json.dumps(state.output, indent=2))
