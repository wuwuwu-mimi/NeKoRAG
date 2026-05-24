"""串联 loader → chunker → (dense embedding + sparse BM25) → vector_store 的完整入库管线"""

from typing import Optional

from ingest.loader import document_load
from ingest.chunker import document_chunk
from retrieval.vector_store import chunk_to_vector
from retrieval.bm25 import get_bm25_retriever


def run_ingest_pipeline(file_path: Optional[str] = None) -> int:
    """
    执行文档入库流程

    Args:
        file_path: 指定单个文件路径时只处理该文件；None 时全量处理 uploads 目录

    Returns:
        本次写入的切片总数
    """
    is_incremental = file_path is not None

    print("=" * 60)
    print("[Pipeline] 阶段 1/4: 加载文档")
    docs = document_load(file_path=file_path)
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

    if is_incremental:
        # 增量上传：跳过 BM25（由调用方从 ChromaDB 全量重建，保证与已有数据一致）
        print("[Pipeline] 增量模式: 跳过 BM25 索引，由调用方全量重建")
    else:
        print("\n" + "=" * 60)
        print("[Pipeline] 阶段 4/4: 构建 BM25 稀疏索引")
        bm25 = get_bm25_retriever()
        bm25.index(chunks)

    print(f"\n[Pipeline] ✅ 完成，本次入库 {count} 条向量")
    return count
