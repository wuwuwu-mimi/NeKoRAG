"""混合检索：RRF 融合稠密向量检索（Chroma）与稀疏关键词检索（BM25）"""

from typing import List, Tuple

import chromadb

from schema.chunk_schema import DocumentChunk
from .embeddings import get_embedding_model
from .bm25 import get_bm25_retriever

# RRF 平滑常数，防止单个检索器的高排名项主导最终结果
# k=60 是学术界和工业界的经验值
_RRF_K = 60


def _dense_search(
    query: str, collection: chromadb.Collection, candidate_k: int
) -> List[Tuple[str, int]]:
    """
    稠密向量检索（Chroma 语义相似度）
    返回 [(chunk_id, 排名), ...]，排名从 1 开始
    """
    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=candidate_k,
        include=["metadatas", "documents", "distances"],
    )

    ranked: List[Tuple[str, int]] = []
    if results["ids"] and results["ids"][0]:
        for rank, chunk_id in enumerate(results["ids"][0], start=1):
            ranked.append((chunk_id, rank))
    return ranked


def _sparse_search(
    query: str, candidate_k: int
) -> List[Tuple[str, int]]:
    """
    稀疏关键词检索（BM25）
    返回 [(chunk_id, 排名), ...]，排名从 1 开始
    """
    bm25 = get_bm25_retriever()
    if bm25.bm25 is None:
        print("[hybrid] BM25 索引未就绪，跳过稀疏检索")
        return []

    results = bm25.search(query, top_k=candidate_k)
    ranked: List[Tuple[str, int]] = []
    for rank, (chunk, _score) in enumerate(results, start=1):
        ranked.append((chunk.metadata.chunk_id, rank))
    return ranked


def rrf_fusion(
    dense_ranked: List[Tuple[str, int]],
    sparse_ranked: List[Tuple[str, int]],
    final_k: int,
    k: int = _RRF_K,
) -> List[str]:
    """
    Reciprocal Rank Fusion: 对多路检索结果按排名融合
    每个 chunk 的 RRF 分数 = sum( 1 / (k + rank_i) )
    """
    rrf_scores: dict[str, float] = {}

    for chunk_id, rank in dense_ranked:
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    for chunk_id, rank in sparse_ranked:
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    # 按 RRF 分数降序排序，取前 final_k 个
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    return sorted_ids[:final_k]


def hybrid_search(
    query: str,
    final_top_k: int = 10,
    candidate_k: int = 20,
) -> List[str]:
    """
    执行混合检索：
    1. 分别从 Chroma 和 BM25 各取 candidate_k 个候选
    2. RRF 融合两路排名
    3. 返回前 final_top_k 个 chunk_id

    candidate_k 设得比 final_top_k 大，保证融合时有足够候选池
    """
    # 获取 Chroma 集合（与入库时使用的相同）
    import os
    db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "neko_collection")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)

    # 双路召回
    dense_ranked = _dense_search(query, collection, candidate_k)
    sparse_ranked = _sparse_search(query, candidate_k)

    print(
        f"[hybrid] 稠密召回: {len(dense_ranked)} 条, "
        f"稀疏召回: {len(sparse_ranked)} 条"
    )

    # RRF 融合
    fused_ids = rrf_fusion(dense_ranked, sparse_ranked, final_top_k)
    print(f"[hybrid] RRF 融合后返回 {len(fused_ids)} 条结果")

    return fused_ids
