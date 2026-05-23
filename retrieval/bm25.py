"""BM25 稀疏检索模块：基于分词的关键词匹配召回"""

import os
import pickle
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi

from schema.chunk_schema import DocumentChunk

# BM25 索引持久化路径
_BM25_INDEX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "bm25_index"
)
_CORPUS_FILE = "corpus.pkl"
_CHUNKS_FILE = "chunks.pkl"


class NeKoBM25Retriever:
    """封装 rank_bm25，提供中文分词 → 索引构建 → 检索 → 持久化全流程"""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        # 分词后的语料（List[List[str]]），用于重建 BM25Okapi 对象
        self.tokenized_corpus: List[List[str]] = []
        # 原始切片对象，检索时按索引返回
        self.chunks: List[DocumentChunk] = []

    # ==================== 索引构建 ====================

    def index(self, chunks: List[DocumentChunk]) -> None:
        """对切片列表进行分词并构建 BM25 索引，同时持久化到磁盘"""
        self.chunks = chunks
        # jieba 精确模式分词：中文按语义切分，英文按空格 + 标点切分
        # 过滤掉单字符 token，减少噪音维度
        self.tokenized_corpus = [
            [
                token
                for token in jieba.lcut(chunk.page_content)
                if len(token.strip()) > 1
            ]
            for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._persist()
        print(f"[NeKoBM25Retriever] BM25 索引构建完成，共 {len(chunks)} 个切片")

    # ==================== 检索 ====================

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """对查询分词后执行 BM25 检索，返回 [(chunk, score), ...] 按分数降序"""
        if self.bm25 is None:
            raise RuntimeError("BM25 索引未构建或未加载，请先调用 index() 或 load()")

        tokenized_query = [
            token for token in jieba.lcut(query) if len(token.strip()) > 1
        ]
        scores = self.bm25.get_scores(tokenized_query)
        # 取 top_k 个最高分的索引
        if len(scores) == 0:
            return []
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [(self.chunks[i], scores[i]) for i in top_indices if scores[i] > 0]

    # ==================== 持久化 ====================

    def _persist(self) -> None:
        """将分词语料和原始切片序列化到磁盘，供下次启动时加载"""
        os.makedirs(_BM25_INDEX_DIR, exist_ok=True)
        corpus_path = os.path.join(_BM25_INDEX_DIR, _CORPUS_FILE)
        chunks_path = os.path.join(_BM25_INDEX_DIR, _CHUNKS_FILE)
        with open(corpus_path, "wb") as f:
            pickle.dump(self.tokenized_corpus, f)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[NeKoBM25Retriever] 索引已持久化至 {_BM25_INDEX_DIR}")

    def load(self) -> bool:
        """从磁盘加载持久化的索引，返回是否加载成功"""
        corpus_path = os.path.join(_BM25_INDEX_DIR, _CORPUS_FILE)
        chunks_path = os.path.join(_BM25_INDEX_DIR, _CHUNKS_FILE)
        if not (os.path.isfile(corpus_path) and os.path.isfile(chunks_path)):
            print("[NeKoBM25Retriever] 未找到持久化索引，需要重新构建")
            return False
        with open(corpus_path, "rb") as f:
            self.tokenized_corpus = pickle.load(f)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(
            f"[NeKoBM25Retriever] 已从磁盘加载 BM25 索引，共 {len(self.chunks)} 个切片"
        )
        return True


# 模块级单例，保证索引只构建/加载一次
_bm25_retriever: NeKoBM25Retriever | None = None


def get_bm25_retriever() -> NeKoBM25Retriever:
    """获取全局 BM25 检索器单例，优先从磁盘加载已有索引"""
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = NeKoBM25Retriever()
        _bm25_retriever.load()  # 尝试加载已有索引，未找到则返回 False
    return _bm25_retriever
