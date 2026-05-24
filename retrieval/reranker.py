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
        if not resp.ok:
            print(
                f"[NeKoReranker] API 返回 {resp.status_code}: {resp.text[:500]}\n"
                f"模型: {self._model}, key 前缀: {self._api_key[:8]}..."
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


def expand_context(
    results: List[RerankResult], depth: int = 1
) -> List[RerankResult]:
    """
    沿 prev_chunk_id / next_chunk_id 指针逐跳扩展上下文

    检索只能抓到最相关的几个 chunk，但文档内容往往分布在相邻 chunk 中。
    利用入库时就织好的链表指针，把左右邻居也拉进来。
    depth=1 拉左右各一个，depth=2 拉两层邻居，以此类推。

    Args:
        results: 精排后的结果列表
        depth: 沿链表向左右各走几步

    Returns:
        去重后的完整上下文列表，核心结果在前
    """
    if not results:
        return []

    db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "neko_collection")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    # id → metadata 映射，逐跳扩展时往里面追加
    id_to_meta: dict[str, dict] = {
        r.chunk_id: r.metadata for r in results
    }
    collected_ids: set[str] = set(id_to_meta.keys())
    # 当前要扩展的边界
    frontier: set[str] = set(collected_ids)

    for level in range(depth):
        neighbour_ids: set[str] = set()
        for cid in frontier:
            meta = id_to_meta.get(cid, {})
            prev_id = meta.get("prev_chunk_id")
            next_id = meta.get("next_chunk_id")
            if prev_id and prev_id not in collected_ids:
                neighbour_ids.add(prev_id)
            if next_id and next_id not in collected_ids:
                neighbour_ids.add(next_id)

        if not neighbour_ids:
            break

        # 从 ChromaDB 拉取这一层邻居的元数据
        fetched = collection.get(
            ids=list(neighbour_ids),
            include=["documents", "metadatas"],
        )
        for cid, meta in zip(
            fetched.get("ids") or [],
            fetched.get("metadatas") or [],
        ):
            id_to_meta[cid] = meta
            collected_ids.add(cid)

        # 下一跳从这一层新增的邻居出发
        frontier = neighbour_ids
        print(
            f"[expand_context] 第 {level + 1} 跳: 扩展 {len(neighbour_ids)} 个相邻 chunk"
        )

    # 从 ChromaDB 拉取所有扩展出来的邻居文本（去重后）
    all_neighbour_ids = collected_ids - {r.chunk_id for r in results}
    neighbours: List[RerankResult] = []
    if all_neighbour_ids:
        fetched = collection.get(
            ids=list(all_neighbour_ids),
            include=["documents", "metadatas"],
        )
        for cid, text, meta in zip(
            fetched.get("ids") or [],
            fetched.get("documents") or [],
            fetched.get("metadatas") or [],
        ):
            neighbours.append(RerankResult(
                chunk_id=cid,
                text=text,
                metadata=meta,
                score=0.0,
            ))

    print(
        f"[expand_context] 核心 {len(results)} 个 + "
        f"扩展 {len(neighbours)} 个相邻 chunk"
    )
    return results + neighbours
