"""NeKoRAG 入口"""

from dotenv import load_dotenv

load_dotenv()
from ingest.pipeline import run_ingest_pipeline
from retrieval.hybrid import hybrid_search
from retrieval.reranker import NeKoReranker
from generation.generator import NeKoGenerator


if __name__ == "__main__":
    nums = run_ingest_pipeline()
    query = "SiliconFlow的产品特性都有什么"

    # 1. 混合检索（Dense + BM25 → RRF）
    candidates = hybrid_search(query=query, final_top_k=20, candidate_k=30)

    # 2. Cross-encoder 精排
    ranker = NeKoReranker()
    results = ranker.rerank(query=query, chunk_ids=candidates, top_k=5)

    # 3. LLM 生成回答
    generator = NeKoGenerator()
    answer = generator.generate(query=query, results=results)

    print(answer)
