"""Cross-encoder 精排模块：通过硅基流动云端 API 对候选集重新打分"""

import os
from typing import List

import chromadb
import requests

from schema.chunk_schema import RerankResult

# 硅基流动 Reranker API
_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"


class NeKoReranker:
    """
    云端 Cross-encoder 精排器
    使用硅基流动的 reranker API，国内直连无需代理
    """

    def __init__(self):
        self._api_key = os.getenv("SILICONFLOW_RERANKER_MODEL_API_KEY", "")
        self._model = os.getenv(
            "SILICONFLOW_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        if not self._api_key:
            print("[NeKoReranker] ⚠ 未设置 SILICONFLOW_RERANKER_MODEL_API_KEY，reranker 不可用")

    def rerank(
        self,
        query: str,
        chunk_ids: List[str],
        top_k: int = 5,
    ) -> List[RerankResult]:
        """
        对候选集做 cross-encoder 精排，直接返回下游可用的完整结果

        Args:
            query: 用户原始查询
            chunk_ids: 前序检索（hybrid_search）返回的候选 chunk_id 列表
            top_k: 最终返回数量
        """
        if not chunk_ids or not self._api_key:
            return []

        # 一次性从 ChromaDB 拉取文本和元数据，下游无需再查库
        cache = self._fetch_from_chroma(chunk_ids)
        texts = [cache[cid]["text"] for cid in chunk_ids if cid in cache]
        if not texts:
            return []

        resp = requests.post(
            _RERANK_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                
                "query": query,
                "documents": texts,
                "top_n": top_k,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        ranked: List[RerankResult] = []
        for item in data.get("results", []):
            idx = item["index"]
            cid = chunk_ids[idx]
            ranked.append(RerankResult(
                chunk_id=cid,
                text=cache[cid]["text"],
                metadata=cache[cid]["metadata"],
                score=item["relevance_score"],
            ))

        return ranked

    def _fetch_from_chroma(self, chunk_ids: List[str]) -> dict:
        """
        从 ChromaDB 批量拉取文本和元数据
        返回 {chunk_id: {"text": str, "metadata": dict}, ...}
        """
        db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "neko_collection")
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name=collection_name)

        result = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []

        cache: dict = {}
        for cid, text, meta in zip(ids, docs, metas):
            cache[cid] = {"text": text, "metadata": meta}
        return cache
