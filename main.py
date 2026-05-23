"""NeKoRAG 入口"""

from dotenv import load_dotenv
load_dotenv()
# 必须在所有其他 import 之前加载 .env，保证 os.getenv 能读到配置
from ingest.pipeline import run_ingest_pipeline
from retrieval.hybrid import hybrid_search
from retrieval.reranker import NeKoReranker


if __name__ == "__main__":
    query = "什么是CPO技术"
    ids = hybrid_search(query=query)

    ranker = NeKoReranker()
    results = ranker.rerank(query=query, chunk_ids=ids, top_k=5)

    for i, res in enumerate(results):
        print(f"top{i + 1}条是  {res.model_dump_json()}")
        print("-" * 50)
