"""
Example 4: FastAPI HTTP wrapper
================================
Run:  uvicorn examples.fastapi_app:app --reload --port 8080
"""
from __future__ import annotations
import base64
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from doc_pipeline.examples.talent_match import build_talent_pipeline

app = FastAPI(title="Document Pipeline API", version="2.0.0")

_pipeline = build_talent_pipeline()


@app.post("/process")
async def process_documents(
    query: Annotated[str, Form()] = "",
    files: Annotated[list[UploadFile], File()] = [],
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    file_entries = [
        {
            "name": f.filename or "file",
            "content_base64": base64.b64encode(await f.read()).decode(),
            "mime_type": f.content_type or "application/octet-stream",
        }
        for f in files
    ]

    state = await _pipeline.run(file_entries, query=query)
    return JSONResponse({
        "session_id": state.session_id,
        "output": state.output,
        "figures_total": sum(len(d.figures) for d in state.documents),
        "errors": state.errors,
    })


@app.get("/health")
async def health():
    return {"status": "ok"}
