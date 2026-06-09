"""生成不同类型 PDF 测试集，验证多策略解析器"""
import os
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "test_pdfs"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def make_text_pdf(path: str):
    """类型1：纯文本 PDF — 最简场景，所有策略应能正常解析"""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "NeKoRAG Project Overview", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", size=11)

    body = (
        "NeKoRAG is a hybrid retrieval-augmented generation system. "
        "It combines dense vector search using ChromaDB with bge-m3 embeddings, "
        "and sparse keyword search using BM25 with jieba tokenization. "
        "The two retrieval paths are merged via Reciprocal Rank Fusion (RRF), "
        "which requires no weight tuning and is robust to different score distributions. "
        "After fusion, a cross-encoder reranker (BGE-reranker-v2-m3) re-scores "
        "the candidates for final ranking. "
        "Context expansion follows linked-list pointers between adjacent chunks "
        "to retrieve surrounding context, preventing information loss across chunk boundaries. "
        "The system supports streaming SSE generation via DeepSeek API, "
        "with automatic source citation in responses."
    )
    pdf.multi_cell(0, 6, body)

    pdf.ln(5)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, "Key Features:", ln=True)
    pdf.set_font("Helvetica", size=11)
    features = [
        "Hybrid retrieval (Dense + BM25 + RRF)",
        "Cross-encoder reranking for precision",
        "Context expansion via linked-list traversal",
        "Streaming SSE generation with citations",
        "PDF / Markdown / TXT document support",
        "Docker containerization with health checks",
    ]
    for feat in features:
        pdf.cell(10)
        pdf.cell(0, 6, f"- {feat}", ln=True)

    pdf.output(path)
    print(f"  ✅ 纯文本 PDF: {path}")


def make_table_pdf(path: str):
    """类型2：表格密集型 PDF — 测试表格提取能力"""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "Quarterly Model API Performance Report", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, "Benchmark period: 2026 Q1-Q2 | Platform: SiliconFlow", ln=True)
    pdf.ln(5)

    # ---- Table 1: 模型性能 ----
    pdf.set_font("Helvetica", size=13)
    pdf.cell(0, 8, "Table 1: Model Performance Comparison", ln=True)
    pdf.ln(2)

    col_w = [55, 30, 35, 35, 35]
    headers = ["Model", "Params", "Throughput", "Latency", "Cost/1M"]
    pdf.set_font("Helvetica", size=10)

    # 画表头
    for i, (h, w) in enumerate(zip(headers, col_w)):
        pdf.cell(w, 7, h, border=1)
    pdf.ln()

    rows = [
        ["DeepSeek-V4", "236B", "156 tok/s", "0.8s", "$0.14"],
        ["Qwen3.5-72B", "72B", "210 tok/s", "0.5s", "$0.09"],
        ["Kimi-K2.6", "128B", "180 tok/s", "0.6s", "$0.12"],
        ["GPT-4o-mini", "8B", "240 tok/s", "0.4s", "$0.15"],
        ["Claude-Haiku", "6B", "320 tok/s", "0.3s", "$0.25"],
    ]
    for row in rows:
        for cell, w in zip(row, col_w):
            pdf.cell(w, 7, cell, border=1)
        pdf.ln()

    pdf.ln(8)

    # ---- Table 2: 检索策略 ----
    pdf.set_font("Helvetica", size=13)
    pdf.cell(0, 8, "Table 2: Retrieval Strategy Comparison (MRR)", ln=True)
    pdf.ln(2)

    col_w2 = [65, 45, 45, 45]
    headers2 = ["Strategy", "MRR", "Hit@1", "NDCG@5"]
    for h, w in zip(headers2, col_w2):
        pdf.cell(w, 7, h, border=1)
    pdf.ln()

    rows2 = [
        ["Pure Dense", "0.903", "83.3%", "0.842"],
        ["Pure BM25", "0.840", "75.0%", "0.702"],
        ["Hybrid + RRF", "0.917", "83.3%", "0.793"],
        ["Hybrid + Reranker", "0.933", "91.7%", "0.851"],
    ]
    for row in rows2:
        for cell, w in zip(row, col_w2):
            pdf.cell(w, 7, cell, border=1)
        pdf.ln()

    pdf.ln(8)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "Note: All benchmarks run on Standard GPU instances. "
        "Latency measured at P50. Throughput measured at concurrency=32. "
        "Cost estimates based on public API pricing as of 2026-06."
    )

    pdf.output(path)
    print(f"  ✅ 表格 PDF:   {path}")


def make_mixed_pdf(path: str):
    """类型3：混合内容 PDF — 文本 + 表格 + 多页"""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1: 文本
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "NeKoRAG Technical Design Document", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, "Version 0.2.0 | 2026-06-09 | Author: Engineering Team", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, "1. Architecture Overview", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "NeKoRAG adopts a three-stage retrieval pipeline. "
        "Stage 1 performs dual-path recall: dense semantic search via ChromaDB "
        "and sparse keyword search via BM25. Stage 2 merges results using "
        "Reciprocal Rank Fusion (k=60). Stage 3 applies cross-encoder reranking "
        "followed by context expansion along linked-list chunk pointers."
    )

    pdf.ln(3)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, "2. Document Ingestion Flow", ln=True)
    pdf.set_font("Helvetica", size=10)
    steps = [
        "Upload document (.md/.txt/.pdf) via REST API",
        "Parse content with format-specific loader",
        "Chunk text using Markdown-header-aware splitter",
        "Generate dense embeddings via Ollama + bge-m3",
        "Store vectors in ChromaDB (persistent mode)",
        "Build BM25 index with jieba tokenization",
    ]
    for i, step in enumerate(steps, 1):
        pdf.cell(10)
        pdf.cell(0, 5, f"{i}. {step}", ln=True)

    # Page 2: 混合表格
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, "3. Evaluation Results", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, "3.1 Per-Category MRR Comparison", ln=True)
    pdf.ln(2)

    col_w = [50, 35, 35, 35, 40]
    headers = ["Category", "Dense", "BM25", "Hybrid", "Reranker"]
    pdf.set_font("Helvetica", size=10)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 7, h, border=1)
    pdf.ln()

    rows = [
        ["Keyword (n=4)", "0.833", "0.875", "0.875", "1.000"],
        ["Semantic (n=4)", "0.875", "0.646", "0.875", "0.800"],
        ["Mixed (n=4)", "1.000", "1.000", "1.000", "1.000"],
        ["Overall (n=12)", "0.903", "0.840", "0.917", "0.933"],
    ]
    for row in rows:
        for cell, w in zip(row, col_w):
            if row == rows[-1]:
                pdf.set_font("Helvetica", style="B", size=10)
            pdf.cell(w, 7, cell, border=1)
        pdf.set_font("Helvetica", size=10)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, "3.2 Key Findings", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "The evaluation demonstrates clear complementary behavior between "
        "dense and sparse retrieval. On keyword-precise queries, BM25 achieves "
        "a higher MRR (0.875) than pure Dense (0.833). On semantic queries, "
        "Dense significantly outperforms BM25 (0.875 vs 0.646). "
        "Hybrid retrieval with RRF maintains the best of both, and cross-encoder "
        "reranking further improves overall MRR to 0.933 -- a 3.3% improvement "
        "over pure Dense and 11.1% over pure BM25."
    )

    pdf.output(path)
    print(f"  ✅ 混合 PDF:   {path}")


def make_minimal_text_pdf(path: str):
    """类型4：极低文本 PDF — 每页只有几行，测试回退逻辑"""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, "Slide Deck: Q2 Review", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, "Key Metric: MRR improved 3.3%", ln=True)
    pdf.ln(3)
    pdf.cell(0, 8, "Action: Deploy to production", ln=True)

    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, "Next Steps", ln=True)
    pdf.ln(3)
    items = ["Add Word support", "CI/CD pipeline", "Cloud deployment"]
    for item in items:
        pdf.cell(0, 6, f"  - {item}", ln=True)

    pdf.output(path)
    print(f"  ✅ 低文本 PDF:  {path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("生成测试 PDF 数据集...\n")

    make_text_pdf(str(TEST_DIR / "01_text_only.pdf"))
    make_table_pdf(str(TEST_DIR / "02_table_heavy.pdf"))
    make_mixed_pdf(str(TEST_DIR / "03_mixed_content.pdf"))
    make_minimal_text_pdf(str(TEST_DIR / "04_minimal_text.pdf"))

    print(f"\n📁 测试集位置: {TEST_DIR}")
    print(f"   共 4 个测试 PDF，覆盖文本/表格/混合/低文本四类场景")

    # 运行验证
    print("\n" + "=" * 60)
    print("验证多策略解析器")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ingest.loader import PDFTextLoader, _pages_to_text

    for pdf_name in sorted(os.listdir(TEST_DIR)):
        if not pdf_name.endswith(".pdf"):
            continue
        filepath = str(TEST_DIR / pdf_name)
        print(f"\n📄 {pdf_name}")
        loader = PDFTextLoader(filepath)
        docs = loader.load()
        if docs:
            total_chars = len(_pages_to_text(docs))
            parser = docs[0].metadata.get("parser", "unknown")
            print(f"   ✅ {len(docs)} 页, {total_chars} 字符, 策略: {parser}")
            # 检查是否有表格
            table_count = sum(
                1 for d in docs if "表格" in d.page_content or "|" in d.page_content
            )
            if table_count:
                print(f"   📊 检测到 {table_count} 页包含表格")
        else:
            print(f"   ❌ 无法解析! (可能需要 OCR 环境)")
