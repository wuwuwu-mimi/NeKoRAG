"""封装 embedding 模型，提供重试、超时、批处理能力"""

import os
import time
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from .get_ip import get_windows_ip

# 每次 embed_documents 调用的最大文本条数，防止 Ollama OOM 或超时
_EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
# 遇到网络 / 服务端错误时的最大重试次数
_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "3"))
# 重试之间的基础等待秒数（指数退避：sleep(base * 2^attempt)）
_RETRY_BASE_DELAY = float(os.getenv("EMBED_RETRY_BASE_DELAY", "1.0"))


class NeKoEmbeddings(Embeddings):
    """带重试和超时保护的 Ollama Embedding 封装"""

    def __init__(self, model_name: str = "bge-m3:latest"):
        win_ip = get_windows_ip()
        default_url = f"http://{win_ip}:11434"
        base_url = os.getenv("EMBEDDING_BASE_URL", default=default_url)

        print(f"[NeKoEmbeddings] 初始化向量模型 [{model_name}]，目标地址: {base_url}")

        self._underlying_embeddings = OllamaEmbeddings(
            base_url=base_url, model=model_name
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对文档列表进行向量化，内置重试"""
        if not texts:
            return []

        for attempt in range(_MAX_RETRIES):
            try:
                return self._underlying_embeddings.embed_documents(texts=texts)
            except Exception as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                delay = _RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"[NeKoEmbeddings] embed_documents 失败 (attempt {attempt + 1}/{_MAX_RETRIES})，"
                    f"{delay:.1f}s 后重试: {e}"
                )
                time.sleep(delay)

        return []  # unreachable，但让类型检查满意

    def embed_query(self, text: str) -> List[float]:
        """对单条查询文本进行向量化，内置重试"""
        if not text:
            return []

        for attempt in range(_MAX_RETRIES):
            try:
                return self._underlying_embeddings.embed_query(text=text)
            except Exception as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                delay = _RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"[NeKoEmbeddings] embed_query 失败 (attempt {attempt + 1}/{_MAX_RETRIES})，"
                    f"{delay:.1f}s 后重试: {e}"
                )
                time.sleep(delay)

        return []  # unreachable


def get_embedding_model() -> Embeddings:
    """获取系统统一配置的 Embedding 模型实例"""
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "bge-m3:latest")
    return NeKoEmbeddings(model_name=model_name)
