"""检索问答 API：普通问答、SSE 流式、仅检索"""

import json
from typing import List, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from retrieval.hybrid import hybrid_search
from retrieval.reranker import NeKoReranker
from generation.generator import NeKoGenerator
from .schemas import (
    QueryRequest,
    QueryResponse,
    RetrievalResponse,
    SourceInfo,
)

router = APIRouter(tags=["query"])


def _build_sources(results) -> List[SourceInfo]:
    """从 RerankResult 列表提取前端可用的来源信息"""
    sources: List[SourceInfo] = []
    for r in results:
        meta = r.metadata
        title_parts = [
            meta.get(k)
            for k in ("Header 1", "Header 2", "Header 3")
            if meta.get(k)
        ]
        sources.append(SourceInfo(
            chunk_id=r.chunk_id,
            text=r.text,
            score=r.score,
            source_title=" > ".join(title_parts) if title_parts else r.chunk_id,
        ))
    return sources


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """普通问答：检索 + 精排 + LLM 生成"""
    try:
        candidates = hybrid_search(
            query=request.query, final_top_k=20, candidate_k=30
        )

        ranker = NeKoReranker()
        results = ranker.rerank(
            query=request.query, chunk_ids=candidates, top_k=5
        )

        if not results:
            return QueryResponse(
                answer="未检索到相关内容，无法回答。",
                sources=[],
            )

        generator = NeKoGenerator()
        answer = generator.generate(query=request.query, results=results)

        return QueryResponse(
            answer=answer,
            sources=_build_sources(results),
        )

    except Exception as e:
        raise HTTPException(500, f"查询失败: {e}")


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """SSE 流式问答：逐步返回 LLM token"""

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            candidates = hybrid_search(
                query=request.query, final_top_k=20, candidate_k=30
            )

            ranker = NeKoReranker()
            results = ranker.rerank(
                query=request.query, chunk_ids=candidates, top_k=5
            )

            if not results:
                yield f"data: {json.dumps({'error': '未检索到相关内容'})}\n\n"
                return

            sources = _build_sources(results)
            source_data = [s.model_dump() for s in sources]
            yield f"data: {json.dumps({'type': 'sources', 'data': source_data})}\n\n"

            generator = NeKoGenerator()
            for token in generator.generate_stream(
                query=request.query, results=results
            ):
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/query/retrieval-only", response_model=RetrievalResponse)
async def retrieval_only(request: QueryRequest):
    """仅检索（调试用）：跳过 LLM 生成，返回精排后的切片"""
    try:
        candidates = hybrid_search(
            query=request.query, final_top_k=20, candidate_k=30
        )

        ranker = NeKoReranker()
        results = ranker.rerank(
            query=request.query, chunk_ids=candidates, top_k=5
        )

        return RetrievalResponse(
            query=request.query,
            results=_build_sources(results),
        )

    except Exception as e:
        raise HTTPException(500, f"检索失败: {e}")
