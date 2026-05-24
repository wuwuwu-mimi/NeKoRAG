"""NeKoRAG — 混合检索 + 精排 + 上下文扩展的 RAG 命令行测试入口

启动 API 服务: uv run uvicorn api:app --port 8000 --reload
"""

from dotenv import load_dotenv

load_dotenv()
from ingest.pipeline import run_ingest_pipeline
from retrieval.hybrid import hybrid_search
from retrieval.reranker import NeKoReranker, expand_context
from generation.generator import NeKoGenerator


if __name__ == "__main__":
    # 1. 文档入库
    run_ingest_pipeline()

    query = "你的问题"

    # 2. 混合检索（Dense + BM25 → RRF）
    candidates = hybrid_search(query=query, final_top_k=20, candidate_k=30)

    # 3. Cross-encoder 精排
    ranker = NeKoReranker()
    results = ranker.rerank(query=query, chunk_ids=candidates, top_k=8)

    # 4. 上下文扩展（沿链表指针拉取相邻 chunk）
    expanded = expand_context(results, depth=2)

    # 5. LLM 生成回答
    generator = NeKoGenerator()
    answer = generator.generate(query=query, results=expanded)

    print(answer)
