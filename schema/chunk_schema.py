from typing import Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    RAG 切片专属结构化元数据模型
    """

    header_1: Optional[str] = Field(
        None, alias="Header 1", description="Markdown 一级标题"
    )
    header_2: Optional[str] = Field(
        None, alias="Header 2", description="Markdown 二级标题"
    )
    header_3: Optional[str] = Field(
        None, alias="Header 3", description="Markdown 三级标题"
    )

    doc_id: str = Field(..., description="原始所属文档的唯一ID")
    chunk_id: str = Field(..., description="全局绝对唯一的切片ID")
    parent_chunk_idx: int = Field(..., description="父级 Markdown 大块的索引编号")
    sub_chunk_idx: int = Field(..., description="当前字数微切段落的子索引编号")
    chunk_length: int = Field(..., description="当前切片的字符/Token长度")

    # 级联上下文指针（由于第一块和最后一块可能没有前后关联，所以设为 Optional）
    prev_chunk_id: Optional[str] = Field(None, description="前一个相邻切片的 ID")
    next_chunk_id: Optional[str] = Field(None, description="后一个相邻切片的 ID")

    class Config:
        populate_by_name = True


class DocumentChunk(BaseModel):
    """
    代表一个完整的、全装备的入库切片对象
    """

    page_content: str = Field(..., description="当前切片的纯文本内容")
    metadata: ChunkMetadata = Field(..., description="绑定的结构化元数据对象")


class RerankResult(BaseModel):
    """
    精排后的单条检索结果，包含下游 LLM / 前端所需的所有字段
    """

    chunk_id: str = Field(..., description="切片 ID")
    text: str = Field(..., description="切片原始文本")
    metadata: dict = Field(..., description="结构化元数据（标题、上下文指针等）")
    score: float = Field(..., description="cross-encoder 相关性分数，越高越相关")
