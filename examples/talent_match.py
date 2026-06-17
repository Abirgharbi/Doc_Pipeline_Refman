"""
Example 1: Talent Matching
==========================
Input : CV files (PDF/DOCX) + job description query
Output: ranked markdown report

The pipeline extracts figures from CVs (e.g. portfolio charts, skill graphs) via VLM
and includes those descriptions in the extraction context — giving the LLM a complete
picture of the candidate even from visual-heavy CVs.
"""
from __future__ import annotations
import asyncio
import base64
import json

from doc_pipeline.core.pipeline import DocumentPipeline
from doc_pipeline.core.config import (
    PipelineConfig, LLMConfig, VLMConfig, DoclingConfig, ExtractionConfig,
)
from doc_pipeline.core.state import PipelineState


CV_SCHEMA: dict = {
    "name": "string — candidate full name",
    "email": "string",
    "summary": "string — 1-2 sentence professional summary",
    "experience_years": "number",
    "seniority_level": "one of: junior | mid | senior | lead | executive",
    "skills": [{"name": "string", "category": "string", "level": "string"}],
    "experiences": [{"title": "string", "company": "string", "years": "number"}],
    "educations": [{"degree": "string", "institution": "string"}],
    "certifications": [{"name": "string", "issuer": "string"}],
    "figure_insights": "string — any notable information found in figures/charts within the CV",
}


async def score_candidates(state: PipelineState, config: PipelineConfig) -> list[dict]:
    from doc_pipeline.providers.llm import get_llm_provider
    llm = get_llm_provider(config.llm)
    results = []

    for ext in state.extractions:
        if not ext.data:
            results.append({
                "name": ext.document_name, "email": "", "source": ext.document_name,
                "score": 0.0, "strengths": [], "gaps": [ext.error or "extraction failed"],
            })
            continue

        prompt = f"""
Job Description:
{state.query}

Candidate Profile (JSON):
{json.dumps(ext.data, indent=2)}

Score this candidate 0.00–1.00 for the job. List 3 strengths and 2 gaps.
Return ONLY this JSON:
{{"score": 0.00, "strengths": ["...", "...", "..."], "gaps": ["...", "..."]}}
"""
        try:
            raw = await llm.complete(
                system="You are a talent evaluator. Return only valid JSON.",
                user=prompt,
            )
            result = json.loads(raw.strip())
        except Exception:
            result = {"score": 0.0, "strengths": [], "gaps": ["scoring failed"]}

        results.append({
            "name": ext.data.get("name") or ext.document_name,
            "email": ext.data.get("email", ""),
            "source": ext.document_name,
            "score": float(result.get("score", 0.0)),
            "strengths": result.get("strengths", []),
            "gaps": result.get("gaps", []),
        })

    return sorted(results, key=lambda c: c["score"], reverse=True)


async def render_ranking_report(state: PipelineState, config: PipelineConfig) -> str:
    # state.metadata["scored"] is set by the custom renderer — but here we work
    # directly from extractions since we pass renderer= to OutputStage
    candidates: list[dict] = state.metadata.get("scored", [])
    medals = ["🥇", "🥈", "🥉"]
    lines = [f"# Talent Match Report\n\n**Query:** {state.query}\n\n---\n"]

    for i, c in enumerate(candidates):
        rank = medals[i] if i < 3 else f"#{i + 1}"
        lines += [
            f"## {rank} {c['name']} — {int(c['score'] * 100)}% match",
            f"**Source:** `{c['source']}`",
            "",
            "**Strengths**",
            *[f"- {s}" for s in c["strengths"]],
            "",
            "**Gaps**",
            *[f"- {g}" for g in c["gaps"]],
            "\n---\n",
        ]
    return "\n".join(lines)


async def _talent_renderer(state: PipelineState, config: PipelineConfig) -> str:
    """Combined scorer + renderer called by OutputStage."""
    scored = await score_candidates(state, config)
    state.metadata["scored"] = scored

    medals = ["🥇", "🥈", "🥉"]
    lines = [f"# Talent Match Report\n\n**Query:** {state.query}\n\n---\n"]
    for i, c in enumerate(scored):
        rank = medals[i] if i < 3 else f"#{i + 1}"
        lines += [
            f"## {rank} {c['name']} — {int(c['score'] * 100)}% match",
            f"**Source:** `{c['source']}`",
            "",
            "**Strengths**", *[f"- {s}" for s in c["strengths"]],
            "", "**Gaps**",   *[f"- {g}" for g in c["gaps"]],
            "\n---\n",
        ]
    return "\n".join(lines)


def build_talent_pipeline(
    text_model: str = "mistral",
    vlm_model: str = "llava",
    ollama_url: str = "http://localhost:11434",
) -> DocumentPipeline:
    config = PipelineConfig(
        docling=DoclingConfig(images_scale=2.0),
        vlm=VLMConfig(model=vlm_model, base_url=ollama_url),
        llm=LLMConfig(provider="ollama", model=text_model, base_url=ollama_url),
        extraction=ExtractionConfig(schema_definition=CV_SCHEMA),
    )
    return DocumentPipeline(config, renderer=_talent_renderer)


if __name__ == "__main__":
    SAMPLE_CV = b"""
    Jane Smith  —  jane@example.com
    Senior Python Developer | 8 years experience

    Skills: Python (expert), FastAPI (advanced), PostgreSQL (advanced), Docker (intermediate)

    Experience:
    - Lead Backend Engineer @ Acme Corp (3 years) — built microservices at 50k req/s
    - Python Developer @ StartupXYZ (2 years) — REST APIs with FastAPI + SQLAlchemy
    - Junior Developer @ WebCo (3 years) — full-stack Django

    Education: BSc Computer Science — University of Tunis (2015)
    Certifications: AWS Solutions Architect Associate
    """
    files = [{
        "name": "jane_smith.txt",
        "content_base64": base64.b64encode(SAMPLE_CV).decode(),
        "mime_type": "text/plain",
    }]
    pipeline = build_talent_pipeline()
    state = asyncio.run(pipeline.run(files, query="Senior Python backend developer with FastAPI"))
    print(state.output)
