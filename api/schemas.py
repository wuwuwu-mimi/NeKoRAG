"""API 请求 / 响应模型"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ==================== 文档管理 ====================


class UploadResponse(BaseModel):
    doc_id: str = Field(..., description="文档唯一标识（UUID）")
    original_name: str = Field(..., description="用户上传时的原始文件名")
    chunk_count: int = Field(..., description="生成的切片数量")


class DocumentInfo(BaseModel):
    doc_id: str = Field(..., description="文档唯一标识（UUID）")
    original_name: str = Field(..., description="原始文件名")
    chunk_count: int = Field(..., description="切片数量")


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]


class DeleteResponse(BaseModel):
    doc_id: str = Field(..., description="已删除的文档 ID")
    deleted_chunks: int = Field(..., description="删除的切片数量")


class StatusResponse(BaseModel):
    doc_id: str
    original_name: str
    status: str = Field(..., description="ready / pending / not_found")
    chunk_count: int = 0


# ==================== 检索问答 ====================


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")


class SourceInfo(BaseModel):
    chunk_id: str = Field(..., description="切片 ID")
    text: str = Field(..., description="切片原始文本")
    score: float = Field(..., description="精排相关度分数")
    source_title: str = Field("", description="来源标题（Header 1 > Header 2 > Header 3）")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="LLM 生成的回答")
    sources: List[SourceInfo] = Field(default_factory=list, description="引用的来源切片")


class RetrievalResponse(BaseModel):
    query: str = Field(..., description="原始查询")
    results: List[SourceInfo] = Field(default_factory=list, description="检索 + 精排结果")
