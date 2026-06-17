"""
Example 3: Legal Contract Reviewer
====================================
Figures in contracts (signatures, seals, org charts) are described by the VLM
and included in the extraction context before LLM analysis.
"""
from __future__ import annotations
import asyncio
import base64

from doc_pipeline.core.pipeline import DocumentPipeline
from doc_pipeline.core.config import (
    PipelineConfig, LLMConfig, VLMConfig, DoclingConfig, ExtractionConfig,
)
from doc_pipeline.core.state import PipelineState


CONTRACT_SCHEMA: dict = {
    "contract_type": "string e.g. NDA, SaaS Agreement, Employment, Lease",
    "parties": [{"role": "string e.g. Vendor/Client", "name": "string"}],
    "effective_date": "YYYY-MM-DD or null",
    "expiry_date": "YYYY-MM-DD or null",
    "governing_law": "string or null",
    "key_obligations": [{"party": "string", "obligation": "string — one sentence"}],
    "liability_cap": "string or null",
    "confidentiality": "boolean",
    "ip_assignment": "boolean",
    "auto_renewal": "boolean",
    "termination_clauses": ["string — each trigger in one sentence"],
    "dispute_resolution": "string or null",
    "penalties": ["string — penalty clause in one sentence"],
    "red_flags": ["string — unusual or risky clause"],
    "visual_elements": "string — describe any signatures, stamps, or seals found in figures",
}

HIGH_RISK_TERMS = [
    "unlimited liability", "irrevocable", "perpetual", "waive all rights",
    "exclusive jurisdiction", "non-compete", "liquidated damages", "sole discretion",
]

RISK_EMOJI = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}


async def _contract_renderer(state: PipelineState, config: PipelineConfig) -> str:
    contracts = []
    for doc, ext in zip(state.documents, state.extractions):
        data = ext.data
        flags = list(data.get("red_flags", []))

        text_lower = (doc.combined_text or "").lower()
        for term in HIGH_RISK_TERMS:
            if term in text_lower and term not in " ".join(flags).lower():
                flags.append(f"Contains term: '{term}'")

        risk_score = min(1.0, len(flags) * 0.15)
        risk_level = "High" if risk_score >= 0.6 else "Medium" if risk_score >= 0.3 else "Low"

        contracts.append({
            "source": doc.name,
            "data": data,
            "flags": flags,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "figures": len(doc.figures),
        })

    contracts.sort(key=lambda c: c["risk_score"], reverse=True)

    lines = ["# Contract Review Brief\n"]
    for c in contracts:
        emoji = RISK_EMOJI.get(c["risk_level"], "⚪")
        d = c["data"]
        lines += [
            f"## {emoji} {c['source']} — {c['risk_level']} Risk ({int(c['risk_score']*100)}%)",
            f"**Type:** {d.get('contract_type', '?')}  |  "
            f"**Law:** {d.get('governing_law') or '—'}  |  "
            f"**Auto-renewal:** {'Yes' if d.get('auto_renewal') else 'No'}  |  "
            f"**Figures:** {c['figures']}",
            "",
        ]
        if c["flags"]:
            lines += ["**Red Flags**", *[f"- ⚠️ {f}" for f in c["flags"]], ""]
        if d.get("key_obligations"):
            lines += ["**Obligations**", *[f"- {o['party']}: {o['obligation']}" for o in d["key_obligations"]], ""]
        if d.get("visual_elements"):
            lines += [f"**Visual Elements (VLM):** {d['visual_elements']}", ""]
        lines.append("---\n")

    return "\n".join(lines)


def build_contract_pipeline(text_model: str = "mistral", vlm_model: str = "llava") -> DocumentPipeline:
    config = PipelineConfig(
        docling=DoclingConfig(images_scale=2.0),
        vlm=VLMConfig(model=vlm_model),
        llm=LLMConfig(provider="ollama", model=text_model),
        extraction=ExtractionConfig(
            schema_definition=CONTRACT_SCHEMA,
            system_prompt=(
                "Extract contract data. For boolean fields return true/false. "
                "Return ONLY valid JSON."
            ),
        ),
        max_concurrency=1,
    )
    return DocumentPipeline(config, renderer=_contract_renderer)


if __name__ == "__main__":
    SAMPLE = b"""
    SOFTWARE LICENSE AGREEMENT  (2026-06-01)
    Parties: Acme Software Ltd (Vendor) & Startup Inc (Client)

    1. LICENSE: irrevocable, worldwide, non-exclusive.
    2. PAYMENT: $5,000/month. Late: 2% monthly penalty.
    3. LIABILITY: capped at fees paid in prior 3 months.
    4. CONFIDENTIALITY: yes — mutual.
    5. IP ASSIGNMENT: custom developments → Vendor sole property.
    6. TERMINATION: 90 days notice; immediate on non-payment. Auto-renewal annual.
    7. GOVERNING LAW: England & Wales. Arbitration in London.
    """
    files = [{
        "name": "software_license.txt",
        "content_base64": base64.b64encode(SAMPLE).decode(),
        "mime_type": "text/plain",
    }]
    pipeline = build_contract_pipeline()
    state = asyncio.run(pipeline.run(files))
    print(state.output)
