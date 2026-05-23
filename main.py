"""NeKoRAG 入口：一键执行文档入库管线"""

from ingest.pipeline import run_ingest_pipeline

if __name__ == "__main__":
    run_ingest_pipeline()
