"""串联 loader → chunker → (dense embedding + sparse BM25) → vector_store 的完整入库管线"""

from ingest.loader import document_load
from ingest.chunker import document_chunk
from retrieval.vector_store import chunk_to_vector
from retrieval.bm25 import get_bm25_retriever


def run_ingest_pipeline() -> int:
    """执行一次完整的文档入库流程，返回写入的切片总数"""
    print("=" * 60)
    print("[Pipeline] 阶段 1/4: 加载文档")
    docs = document_load()
    if not docs:
        print("[Pipeline] 未发现任何文档，终止")
        return 0
    print(f"[Pipeline] 加载完成，共 {len(docs)} 篇文档")

    print("\n" + "=" * 60)
    print("[Pipeline] 阶段 2/4: 切片")
    chunks = document_chunk(docs)
    if not chunks:
        print("[Pipeline] 切片结果为空，终止")
        return 0
    print(f"[Pipeline] 切片完成，共产生 {len(chunks)} 个切片")

    print("\n" + "=" * 60)
    print("[Pipeline] 阶段 3/4: 稠密向量化并入库 (ChromaDB)")
    count = chunk_to_vector(chunks)
    print(f"[Pipeline] 稠密向量入库完成，共 {count} 条")

    print("\n" + "=" * 60)
    print("[Pipeline] 阶段 4/4: 构建 BM25 稀疏索引")
    bm25 = get_bm25_retriever()
    bm25.index(chunks)

    print(f"\n[Pipeline] ✅ 全部完成，成功入库 {count} 条向量 + BM25 索引")
    return count
