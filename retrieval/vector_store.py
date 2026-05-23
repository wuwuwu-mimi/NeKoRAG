"""向量入库：将切片批量化嵌入后写入 ChromaDB"""

import os
import time
from typing import List
from schema.chunk_schema import DocumentChunk

from .embeddings import get_embedding_model, _EMBED_BATCH_SIZE, _MAX_RETRIES, _RETRY_BASE_DELAY

# 向量库持久化路径，通过环境变量可配置
_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "neko_collection")


def chunk_to_vector(chunks: List[DocumentChunk]) -> int:
    """将切分后的文本分批向量化并写入 ChromaDB，已存在的 ID 会覆盖更新"""
    if not chunks:
        print("[chunk_to_vector] 没有需要入库的切片")
        return 0

    embedding_model = get_embedding_model()
    os.makedirs(_DB_PATH, exist_ok=True)

    import chromadb
    client = chromadb.PersistentClient(path=_DB_PATH)
    collection = client.get_or_create_collection(name=_COLLECTION_NAME)

    total_inserted = 0
    # 分批处理，防止大量文本一次性发给 Ollama 导致超时或 OOM
    for batch_start in range(0, len(chunks), _EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + _EMBED_BATCH_SIZE]

        texts = [chunk.page_content for chunk in batch]
        ids = [chunk.metadata.chunk_id for chunk in batch]
        metadatas = [
            {k: (v if v is not None else "") for k, v in chunk.metadata.model_dump(by_alias=True).items()}
            for chunk in batch
        ]

        # 带重试的 embedding 调用
        embeddings = None
        for attempt in range(_MAX_RETRIES):
            try:
                embeddings = embedding_model.embed_documents(texts)
                break
            except Exception:
                if attempt == _MAX_RETRIES - 1:
                    raise
                delay = _RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"[chunk_to_vector] 批次 {batch_start // _EMBED_BATCH_SIZE + 1} "
                    f"向量化失败，{delay:.1f}s 后重试 (attempt {attempt + 1}/{_MAX_RETRIES})"
                )
                time.sleep(delay)

        if embeddings is None:
            continue  # 理论上不会到这里

        # 使用 upsert 替代 add，避免重复入库时的 ID 唯一键冲突
        collection.upsert(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        total_inserted += len(ids)
        print(
            f"[chunk_to_vector] 批次 {batch_start // _EMBED_BATCH_SIZE + 1}: "
            f"已写入 {len(ids)} 条 (累计 {total_inserted}/{len(chunks)})"
        )

    return total_inserted
