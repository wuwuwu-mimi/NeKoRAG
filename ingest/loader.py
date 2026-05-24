from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
import os

# 项目根目录下的上传文件夹，所有待入库的原始文档存放于此
_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "uploads"
)


def document_load(file_path: Optional[str] = None) -> List[Document]:
    """
    加载待入库文档

    Args:
        file_path: 指定单个文件路径时只加载该文件；为 None 时加载整个 uploads 目录

    Returns:
        LangChain Document 列表
    """
    upload_path = os.path.normpath(_UPLOAD_DIR)

    if file_path:
        # 单文件模式：API 上传时只处理新文件，避免重跑全量
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if not file_path.endswith((".md", ".txt")):
            raise ValueError(f"仅支持 .md 或 .txt 文件: {file_path}")
        loader = TextLoader(file_path, autodetect_encoding=True)
        return loader.load()

    # 全量模式：扫描整个 uploads 目录
    if not os.path.isdir(upload_path):
        raise FileNotFoundError(f"上传目录不存在: {upload_path}")

    loader = DirectoryLoader(
        path=upload_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        silent_errors=True,
        show_progress=True,
    )

    return loader.load()
