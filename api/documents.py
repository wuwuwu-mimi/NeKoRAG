"""文档管理 API：上传、列表、删除、状态"""

import json
import os
import uuid
from typing import List

import chromadb
from fastapi import APIRouter, File, HTTPException, UploadFile

from ingest.pipeline import run_ingest_pipeline
from retrieval.bm25 import get_bm25_retriever
from schema.chunk_schema import DocumentChunk
from .schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    StatusResponse,
    UploadResponse,
)

router = APIRouter(tags=["documents"])

_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "uploads"
)


# ==================== ChromaDB / BM25 工具 ====================


def _get_collection():
    """获取 ChromaDB 集合"""
    db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "neko_collection")
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)


def _rebuild_bm25():
    """从 ChromaDB 全部切片重建 BM25 索引"""
    collection = _get_collection()
    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    chunks: List[DocumentChunk] = []
    for cid, text, meta in zip(ids, docs, metas):
        chunks.append(DocumentChunk(page_content=text, metadata=meta))

    bm25 = get_bm25_retriever()
    bm25.index(chunks)
    return len(chunks)


# ==================== UUID ↔ 原始文件名映射（sidecar 文件） ====================


def _write_meta(uuid_str: str, original_name: str) -> None:
    """写入 {uuid}.meta.json，存储原始文件名"""
    meta_path = os.path.join(_UPLOAD_DIR, f"{uuid_str}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"original_name": original_name}, f, ensure_ascii=False)


def _read_meta(uuid_str: str) -> dict:
    """读取 {uuid}.meta.json"""
    meta_path = os.path.join(_UPLOAD_DIR, f"{uuid_str}.meta.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _delete_meta(uuid_str: str) -> None:
    """删除 sidecar 元数据文件"""
    meta_path = os.path.join(_UPLOAD_DIR, f"{uuid_str}.meta.json")
    if os.path.isfile(meta_path):
        os.remove(meta_path)


def _uuid_from_doc_id(doc_id: str) -> str:
    """从 doc_id（全路径）中提取 UUID，例如 /path/to/abc123.md → abc123"""
    basename = os.path.basename(doc_id)
    # 去掉扩展名，UUID 是纯 hex，不带下划线或中文
    return os.path.splitext(basename)[0]


# ==================== API 端点 ====================


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档并触发入库管线"""
    if not file.filename:
        raise HTTPException(400, "文件名不能为空")

    if not file.filename.endswith((".md", ".txt")):
        raise HTTPException(400, "仅支持 .md 或 .txt 文件")

    # 用 UUID 重命名，避免同名冲突；原始文件名存入 sidecar 供前端展示
    doc_uuid = uuid.uuid4().hex[:12]
    ext = os.path.splitext(file.filename)[1]
    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(_UPLOAD_DIR, f"{doc_uuid}{ext}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 持久化原始文件名
    _write_meta(doc_uuid, file.filename)

    try:
        chunk_count = run_ingest_pipeline(file_path=file_path)
        _rebuild_bm25()
    except Exception as e:
        # 入库失败时清理已保存的文件
        if os.path.isfile(file_path):
            os.remove(file_path)
        _delete_meta(doc_uuid)
        raise HTTPException(500, f"入库失败: {e}")

    return UploadResponse(
        doc_id=doc_uuid,
        original_name=file.filename,
        chunk_count=chunk_count,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """列出所有已入库文档及切片数量"""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    metas = result.get("metadatas") or []

    doc_stats: dict[str, dict] = {}
    for meta in metas:
        doc_id = meta.get("doc_id", "unknown")
        uuid_str = _uuid_from_doc_id(doc_id)
        if uuid_str not in doc_stats:
            sidecar = _read_meta(uuid_str)
            doc_stats[uuid_str] = {
                "doc_id": uuid_str,
                "original_name": sidecar.get("original_name", uuid_str),
                "chunk_count": 0,
            }
        doc_stats[uuid_str]["chunk_count"] += 1

    docs = [DocumentInfo(**d) for d in doc_stats.values()]
    return DocumentListResponse(documents=docs)


@router.delete("/documents/by-id", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """通过 UUID 删除文档及关联切片"""
    collection = _get_collection()

    # doc_id 存储在 ChromaDB 是全路径，需要模糊匹配
    result = collection.get(include=["metadatas"])
    all_ids = result.get("ids") or []
    all_metas = result.get("metadatas") or []

    # 找到所有属于此 UUID 的 chunk
    ids_to_delete = []
    doc_path = ""
    for cid, meta in zip(all_ids, all_metas):
        meta_doc_id = meta.get("doc_id", "")
        if _uuid_from_doc_id(meta_doc_id) == doc_id:
            ids_to_delete.append(cid)
            if not doc_path:
                doc_path = meta_doc_id

    if not ids_to_delete:
        raise HTTPException(404, f"未找到文档: {doc_id}")

    deleted_count = len(ids_to_delete)

    # 逐个删除（ChromaDB 按 ID 删除）
    for cid in ids_to_delete:
        collection.delete(ids=[cid])

    # 清理源文件和 sidecar
    if doc_path and os.path.isfile(doc_path):
        os.remove(doc_path)
    _delete_meta(doc_id)

    remaining = _rebuild_bm25()
    print(f"[documents] 已删除 {deleted_count} 个切片，BM25 重建完成 ({remaining} 个切片)")

    return DeleteResponse(doc_id=doc_id, deleted_chunks=deleted_count)


@router.post("/documents/reset")
async def reset_all():
    """清空向量库 + BM25 索引，从 data/uploads/ 全量重建"""
    import shutil

    db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    bm25_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "bm25_index"
    )

    # 清空
    shutil.rmtree(db_path, ignore_errors=True)
    shutil.rmtree(bm25_dir, ignore_errors=True)
    print("[reset] 已清空 ChromaDB 和 BM25 索引")

    # 全量重建（不传 file_path，扫描整个 uploads 目录）
    count = run_ingest_pipeline()
    # 全量模式 pipeline 内部已调用 bm25.index()，无需额外重建

    return {
        "message": f"重建完成，共 {count} 个切片",
        "chunk_count": count,
    }


@router.get("/documents/by-id/status", response_model=StatusResponse)
async def get_document_status(doc_id: str):
    """通过 UUID 查询文档处理状态"""
    sidecar = _read_meta(doc_id)
    original_name = sidecar.get("original_name", doc_id)

    # 检查源文件
    file_exists = False
    for ext in (".md", ".txt"):
        if os.path.isfile(os.path.join(_UPLOAD_DIR, f"{doc_id}{ext}")):
            file_exists = True
            break

    # 检查 ChromaDB
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    metas = result.get("metadatas") or []
    chunk_count = sum(
        1 for m in metas if _uuid_from_doc_id(m.get("doc_id", "")) == doc_id
    )

    if chunk_count > 0:
        status = "ready"
    elif file_exists:
        status = "pending"
    else:
        status = "not_found"

    return StatusResponse(
        doc_id=doc_id,
        original_name=original_name,
        status=status,
        chunk_count=chunk_count,
    )
