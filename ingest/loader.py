from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
import os

# 项目根目录下的上传文件夹，所有待入库的原始文档存放于此
_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "uploads"
)


def document_load() -> List[Document]:
    """从 data/uploads 目录加载所有 Markdown 文件"""
    # 规范化路径并校验目录是否存在，避免路径错误时的隐蔽异常
    upload_path = os.path.normpath(_UPLOAD_DIR)
    if not os.path.isdir(upload_path):
        raise FileNotFoundError(f"上传目录不存在: {upload_path}")

    loader = DirectoryLoader(
        path=upload_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        silent_errors=True,
        show_progress=True,
    )

    docs = loader.load()
    return docs
