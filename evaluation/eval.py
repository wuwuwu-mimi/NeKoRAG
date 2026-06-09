"""RAG 检索质量评估：对比 纯Dense / 纯BM25 / 混合检索 / 混合+Reranker 四种策略

指标：MRR、Hit Rate@K、NDCG@K
"""

import os
import sys
import json
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

import chromadb
from retrieval.embeddings import get_embedding_model
from retrieval.bm25 import get_bm25_retriever
from retrieval.reranker import NeKoReranker
from retrieval.hybrid import rrf_fusion


# ==================== 1. 测试查询集（标注了预期相关的 chunk 关键词） ====================

@dataclass
class TestQuery:
    """单个测试查询"""
    query: str
    category: str  # "keyword" | "semantic" | "mixed"
    # 预期相关 chunk 的 ID 子串匹配列表
    expected_matches: List[str]


# 12 个测试查询，按检索类型分三类，充分体现 Dense/BM25 互补性
TEST_QUERIES: List[TestQuery] = [
    # ═══════ 类别 A：精确关键词匹配（BM25 优势场景）═══════
    # 包含特定缩写、编号、技术术语，向量模型难以编码，BM25 能精确命中
    TestQuery(
        query="NPO 近端封装光学技术介绍",
        category="keyword",
        expected_matches=[
            "a0adb6c1719e.md::ch2",  # "NPO" 仅在此处出现
        ],
    ),
    TestQuery(
        query="SerDes 功耗降低了百分之多少",
        category="keyword",
        expected_matches=[
            "a0adb6c1719e.md::ch4",  # 包含 SerDes 功耗具体数据
        ],
    ),
    TestQuery(
        query="1.6 Tbps 带宽以上的光模块需求",
        category="keyword",
        expected_matches=[
            "a0adb6c1719e.md::ch1",  # 提及 1.6 Tbps 带宽需求
        ],
    ),
    TestQuery(
        query="Workflow和Agent的边界区别",
        category="keyword",
        expected_matches=[
            "5109f346ed6f.md::ch14",  # Day 8: Workflow 与 Agent 边界
            "5109f346ed6f.md::ch2",   # 第2周总览
        ],
    ),

    # ═══════ 类别 B：语义匹配（Dense 优势场景）═══════
    # 用自然语言描述、同义词、改写表达，BM25 关键词无法匹配
    TestQuery(
        query="大规模AI模型训练中，芯片和芯片之间的数据传输遇到了什么困难",
        category="semantic",
        expected_matches=[
            "a0adb6c1719e.md::ch0",  # 引言：通信瓶颈
            "a0adb6c1719e.md::ch1",  # 核心痛点：电互连难以为继
        ],
    ),
    TestQuery(
        query="用什么样的思路能不断提高大语言模型的回答质量",
        category="semantic",
        expected_matches=[
            "7ce154854e08.md::ch0",  # 迭代式 Prompt 开发
            "7ce154854e08.md::ch1",  # 案例
            "7ce154854e08.md::ch2",  # 第一次迭代
            "7ce154854e08.md::ch3",  # 第二次迭代
            "7ce154854e08.md::ch4",  # 第三次迭代
        ],
    ),
    TestQuery(
        query="新手想做AI智能体开发，从哪里开始入手比较合适",
        category="semantic",
        expected_matches=[
            "5109f346ed6f.md::ch0",   # 目标
            "5109f346ed6f.md::ch1",   # 第1周
            "5109f346ed6f.md::ch7",   # Day 1
        ],
    ),
    TestQuery(
        query="云端模型服务平台有哪些代表产品",
        category="semantic",
        expected_matches=[
            "8ea3ea2fa6ac.md::ch0",  # SiliconFlow 产品介绍
            "8ea3ea2fa6ac.md::ch1",  # 核心优势
        ],
    ),

    # ═══════ 类别 C：混合场景（需要两路互补）═══════
    TestQuery(
        query="CPO共封装光学的功耗跟传统800G光模块比有什么优势",
        category="mixed",
        expected_matches=[
            "a0adb6c1719e.md::ch4",  # CPO vs 800G 数据
            "a0adb6c1719e.md::ch2",  # CPO 技术解释
            "a0adb6c1719e.md::ch3",  # CPO 封装细节
        ],
    ),
    TestQuery(
        query="结构化输出（JSON格式）在Agent系统里具体有什么用",
        category="mixed",
        expected_matches=[
            "5109f346ed6f.md::ch9",   # Day 3: 结构化输出
            "5109f346ed6f.md::ch1",   # 第1周概览
        ],
    ),
    TestQuery(
        query="Prompt工程的迭代优化方法在实际产品文案写作中怎么用",
        category="mixed",
        expected_matches=[
            "7ce154854e08.md::ch0",  # 迭代概念
            "7ce154854e08.md::ch1",  # 案例引入
            "7ce154854e08.md::ch2",  # 迭代1
            "7ce154854e08.md::ch3",  # 迭代2
            "7ce154854e08.md::ch4",  # 迭代3
        ],
    ),
    TestQuery(
        query="第一代和第二代CPO架构的核心差异是什么",
        category="mixed",
        expected_matches=[
            "a0adb6c1719e.md::ch5",  # 第一代 vs 第二代 CPO
            "a0adb6c1719e.md::ch2",  # CPO 技术突破
        ],
    ),
]


# ==================== 2. 评估指标计算 ====================

def calc_mrr(ranked_lists: List[List[str]], ground_truths: List[List[str]]) -> float:
    """Mean Reciprocal Rank：第一个相关结果排名的倒数平均值"""
    reciprocal_ranks = []
    for ranked, gt in zip(ranked_lists, ground_truths):
        for rank, chunk_id in enumerate(ranked, start=1):
            if any(pattern in chunk_id for pattern in gt):
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def calc_hit_rate(
    ranked_lists: List[List[str]], ground_truths: List[List[str]], k: int
) -> float:
    """Hit Rate@K：Top-K 中至少命中一个相关结果的比例"""
    hits = 0
    for ranked, gt in zip(ranked_lists, ground_truths):
        top_k = ranked[:k]
        if any(any(pattern in cid for pattern in gt) for cid in top_k):
            hits += 1
    return hits / len(ranked_lists) if ranked_lists else 0.0


def calc_ndcg(
    ranked_lists: List[List[str]], ground_truths: List[List[str]], k: int
) -> float:
    """Normalized Discounted Cumulative Gain@K

    使用 binary relevance（命中=1，否则=0），DCG = Σ rel_i / log2(i+1)
    """
    ndcg_scores = []
    for ranked, gt in zip(ranked_lists, ground_truths):
        top_k = ranked[:k]
        dcg = 0.0
        for i, cid in enumerate(top_k, start=1):
            rel = 1.0 if any(pattern in cid for pattern in gt) else 0.0
            dcg += rel / math.log2(i + 1)

        # IDCG：理想情况下所有相关结果排在最前面
        num_relevant = min(len(gt), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


# ==================== 3. 检索策略实现 ====================

def _get_collection():
    db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "neko_collection")
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)


def dense_only_search(query: str, top_k: int = 10) -> List[str]:
    """纯稠密向量检索"""
    collection = _get_collection()
    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )
    return results["ids"][0] if results["ids"] else []


def bm25_only_search(query: str, top_k: int = 10) -> List[str]:
    """纯 BM25 关键词检索"""
    bm25 = get_bm25_retriever()
    if bm25.bm25 is None:
        return []
    results = bm25.search(query, top_k=top_k)
    return [chunk.metadata.chunk_id for chunk, _ in results]


def hybrid_retrieval(query: str, top_k: int = 10) -> List[str]:
    """Dense + BM25 → RRF 融合"""
    from retrieval.hybrid import hybrid_search
    return hybrid_search(query, final_top_k=top_k, candidate_k=20)


def hybrid_with_rerank(query: str, top_k: int = 10) -> List[str]:
    """混合检索 + Cross-encoder 精排"""
    from retrieval.hybrid import hybrid_search
    candidates = hybrid_search(query, final_top_k=20, candidate_k=30)
    ranker = NeKoReranker()
    results = ranker.rerank(query=query, chunk_ids=candidates, top_k=top_k)
    return [r.chunk_id for r in results]


# ==================== 4. 主评估流程 ====================

def run_evaluation():
    """跑全部 10 个查询 × 4 种策略，输出对比报告"""
    queries = [tq.query for tq in TEST_QUERIES]
    ground_truths = [tq.expected_matches for tq in TEST_QUERIES]

    strategies = {
        "纯 Dense (Chroma)": dense_only_search,
        "纯 BM25 (jieba)": bm25_only_search,
        "混合 Dense+BM25+RRF": hybrid_retrieval,
        "混合 + Reranker 精排": hybrid_with_rerank,
    }

    # 收集每种策略的排名列表
    all_rankings: Dict[str, List[List[str]]] = {
        name: [] for name in strategies
    }

    print("=" * 80)
    print("  NeKoRAG 检索质量评估")
    print("=" * 80)
    print(f"  测试查询数: {len(queries)}")
    print(f"  Top-K 设置: K=1,3,5")
    print()

    # 逐查询运行
    for i, query in enumerate(queries):
        print(f"  [{i + 1}/{len(queries)}] {query}")
        for name, strategy_fn in strategies.items():
            try:
                ranked = strategy_fn(query, top_k=10)
                all_rankings[name].append(ranked)
                # 显示命中情况
                gt = ground_truths[i]
                hits = [
                    f"#{r}" for r, cid in enumerate(ranked[:5], 1)
                    if any(p in cid for p in gt)
                ]
                hit_str = f"命中位次: {', '.join(hits)}" if hits else "❌ 未命中"
                print(f"      {name:30s} → Top-5: {hit_str}")
            except Exception as e:
                print(f"      {name:30s} → ⚠ 错误: {e}")
                all_rankings[name].append([])
        print()

    # ─── 计算指标 ───
    print("=" * 80)
    print("  评估指标对比")
    print("=" * 80)

    metrics = {}
    for name, rankings in all_rankings.items():
        mrr = calc_mrr(rankings, ground_truths)
        hr1 = calc_hit_rate(rankings, ground_truths, k=1)
        hr3 = calc_hit_rate(rankings, ground_truths, k=3)
        hr5 = calc_hit_rate(rankings, ground_truths, k=5)
        ndcg3 = calc_ndcg(rankings, ground_truths, k=3)
        ndcg5 = calc_ndcg(rankings, ground_truths, k=5)
        metrics[name] = {
            "MRR": mrr,
            "Hit@1": hr1,
            "Hit@3": hr3,
            "Hit@5": hr5,
            "NDCG@3": ndcg3,
            "NDCG@5": ndcg5,
        }

    # ─── 按类别计算指标 ───
    categories = {"keyword": "A. 精确关键词", "semantic": "B. 语义匹配", "mixed": "C. 混合场景"}
    cat_metrics: Dict[str, Dict[str, dict]] = {}

    for cat_key, cat_label in categories.items():
        cat_indices = [i for i, tq in enumerate(TEST_QUERIES) if tq.category == cat_key]
        cat_rankings = {
            name: [rankings[i] for i in cat_indices]
            for name, rankings in all_rankings.items()
        }
        cat_gt = [ground_truths[i] for i in cat_indices]

        cat_metrics[cat_label] = {}
        for name in strategies:
            cat_metrics[cat_label][name] = {
                "MRR": calc_mrr(cat_rankings[name], cat_gt),
                "Hit@5": calc_hit_rate(cat_rankings[name], cat_gt, k=5),
                "count": len(cat_indices),
            }

    # ─── 打印总览表 ───
    header = f"{'策略':<35s} {'MRR':>6s} {'Hit@1':>6s} {'Hit@3':>6s} {'Hit@5':>6s} {'NDCG@3':>7s} {'NDCG@5':>7s}"
    print(header)
    print("-" * len(header))

    baseline = metrics.get("纯 Dense (Chroma)", None)

    for name in strategies:
        m = metrics.get(name)
        if m is None:
            continue
        line = (
            f"{name:<35s} "
            f"{m['MRR']:>6.3f} "
            f"{m['Hit@1']:>6.1%} "
            f"{m['Hit@3']:>6.1%} "
            f"{m['Hit@5']:>6.1%} "
            f"{m['NDCG@3']:>7.3f} "
            f"{m['NDCG@5']:>7.3f}"
        )
        print(line)

    # ─── 按类别分表 ───
    print()
    print("=" * 80)
    print("  按检索类型分项对比 — 揭示 Dense/BM25 互补性")
    print("=" * 80)
    for cat_label in categories.values():
        cm = cat_metrics[cat_label]
        count = list(cm.values())[0]["count"]
        print(f"\n  【{cat_label}】({count} 个查询)")
        sub_header = f"  {'策略':<35s} {'MRR':>6s} {'Hit@5':>6s}"
        print(sub_header)
        print("  " + "-" * (len(sub_header) - 2))
        for name in strategies:
            m = cm.get(name, {})
            if not m:
                continue
            print(f"  {name:<35s} {m['MRR']:>6.3f} {m['Hit@5']:>6.1%}")

        # 标出每类最优
        best_mrr = max((cm[n]["MRR"] for n in strategies if n in cm), key=lambda x: x)
        best_hit5 = max((cm[n]["Hit@5"] for n in strategies if n in cm), key=lambda x: x)
        best_mrr_names = [n for n in strategies if n in cm and cm[n]["MRR"] == best_mrr]
        best_hit5_names = [n for n in strategies if n in cm and cm[n]["Hit@5"] == best_hit5]
        print(f"  → MRR 最优: {', '.join(best_mrr_names)} ({best_mrr:.3f})")
        print(f"  → Hit@5 最优: {', '.join(best_hit5_names)} ({best_hit5:.1%})")

    # ─── 整体提升对比 ───
    if baseline:
        print()
        print("=" * 80)
        print("  相对于纯 Dense 检索的指标变化")
        print("=" * 80)
        header2 = f"{'策略':<35s} {'MRR Δ':>7s} {'Hit@5 Δ':>8s} {'NDCG@5 Δ':>9s}"
        print(header2)
        print("-" * len(header2))
        for name in strategies:
            if name == "纯 Dense (Chroma)":
                continue
            m = metrics.get(name)
            if m is None:
                continue
            mrr_delta = m["MRR"] - baseline["MRR"]
            hit5_delta = m["Hit@5"] - baseline["Hit@5"]
            ndcg5_delta = m["NDCG@5"] - baseline["NDCG@5"]
            line = (
                f"{name:<35s} "
                f"{mrr_delta:>+7.3f} "
                f"{hit5_delta:>+8.1%} "
                f"{ndcg5_delta:>+9.3f}"
            )
            print(line)

    print()
    print("=" * 80)
    print("  评估完成 ✅")
    print("=" * 80)

    return metrics


# ==================== 5. 导出 JSON 报告 ====================

def export_report(metrics: dict, path: str = "data/eval_report.json"):
    """将评估结果导出为 JSON 文件，供前端展示"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "test_queries": [
            {"query": tq.query, "expected_match_count": len(tq.expected_matches)}
            for tq in TEST_QUERIES
        ],
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 评估报告已导出至: {path}")


if __name__ == "__main__":
    metrics = run_evaluation()
    export_report(metrics)
