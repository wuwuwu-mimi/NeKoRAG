"""NeKoRAG Gradio 前端 — 上传、检索、流式问答、文档管理"""

import json
from typing import Generator

import gradio as gr
import requests

API_BASE = "http://localhost:8000"

# ==================== Helper ====================


def _stream_query(query: str) -> Generator[str, None, None]:
    """流式调用 /v1/query/stream，逐 token 产出"""
    try:
        resp = requests.post(
            f"{API_BASE}/v1/query/stream",
            json={"query": query},
            stream=True,
            timeout=120,
        )
        sources_shown = False
        sources = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]  # strip "data: "
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if data.get("error"):
                yield f"\n\n❌ {data['error']}"
                return

            if data.get("type") == "sources":
                sources = data.get("data", []) 
                if not sources_shown:
                    sources_shown = True
                    yield "📚 **参考资料**\n\n"
                    for i, s in enumerate(sources, 1):
                        title = s.get("source_title", "Unknown")[:60]
                        score = s.get("score", 0.0)
                        yield f"  [{i}] {title} (相关度: {score:.2f})\n"
                    yield "\n---\n\n💬 **回答**\n\n"
            elif data.get("type") == "token":
                yield data["data"]
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ 无法连接到后端服务，请确保 `uvicorn api:app --port 8000` 已启动"


def _retrieval_only(query: str) -> str:
    """非流式检索（调试用）"""
    try:
        resp = requests.post(
            f"{API_BASE}/v1/query/retrieval-only",
            json={"query": query},
            timeout=60,
        )
        if not resp.ok:
            return f"❌ 检索失败 {resp.status_code}: {resp.text[:500]}"

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "❌ 未检索到相关内容"

        lines = [f"### 🔍 检索结果 (共 {len(results)} 条)\n"]
        for i, r in enumerate(results, 1):
            title = r.get("source_title", "N/A")[:80]
            score = r.get("score", 0.0)
            text = r.get("text", "")[:300]
            lines.append(f"**[{i}] {title}**  (score: {score:.4f})")
            lines.append(f"> {text}")
            lines.append("")
        return "\n".join(lines)
    except requests.exceptions.ConnectionError:
        return "⚠️ 无法连接到后端服务"
    except Exception as e:
        return f"❌ 错误: {e}"


# ==================== Upload ====================


def upload_file(file) -> str:
    """上传文档到后端"""
    if file is None:
        return "⚠️ 请选择文件"

    try:
        with open(file, "rb") as f:
            resp = requests.post(
                f"{API_BASE}/v1/documents/upload",
                files={"file": (file.split("/")[-1], f)},
                timeout=120,
            )
    except requests.exceptions.ConnectionError:
        return "⚠️ 无法连接到后端服务"

    if resp.ok:
        data = resp.json()
        return (
            f"✅ 上传成功！\n\n"
            f"  • 文档 ID: `{data['doc_id']}`\n"
            f"  • 原始文件名: {data['original_name']}\n"
            f"  • 生成切片数: {data['chunk_count']}"
        )
    else:
        return f"❌ 上传失败 {resp.status_code}: {resp.text[:500]}"


# ==================== Documents ====================


def list_documents() -> str:
    """列出已入库文档"""
    try:
        resp = requests.get(f"{API_BASE}/v1/documents", timeout=10)
        if not resp.ok:
            return f"❌ {resp.status_code}"
        data = resp.json()
        docs = data.get("documents", [])
        if not docs:
            return "📭 暂无已入库文档"

        lines = ["| 文档 ID | 原始文件名 | 切片数 |", "|---------|-----------|--------|"]
        for d in docs:
            doc_id = d["doc_id"]
            name = d["original_name"][:50]
            count = d["chunk_count"]
            lines.append(f"| `{doc_id}` | {name} | {count} |")
        return "\n".join(lines)
    except requests.exceptions.ConnectionError:
        return "⚠️ 无法连接到后端服务"


# ==================== Build UI ====================


def build_ui():
    with gr.Blocks(title="NeKoRAG — RAG 知识库问答") as demo:
        gr.Markdown(
            """
            # 🧠 NeKoRAG — 知识库智能问答系统

            **混合检索** (Dense + BM25 + RRF) → **Cross-encoder 精排** → **上下文扩展** → **LLM 生成**
            """
        )

        with gr.Tabs():
            # ─── Tab 1: 问答 ───
            with gr.Tab("💬 问答"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="输入你的问题",
                            placeholder="例如：CPO 共封装光学技术有哪些优势？",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        mode = gr.Radio(
                            choices=["流式生成 (Stream)", "仅检索 (Debug)"],
                            value="流式生成 (Stream)",
                            label="模式",
                        )

                submit_btn = gr.Button("🔍 搜索", variant="primary", size="lg")
                result_output = gr.Markdown(
                    label="回答",
                    value="*等待输入查询...*",
                    elem_id="result",
                )

            # ─── Tab 2: 上传 ───
            with gr.Tab("📤 上传文档"):
                gr.Markdown("支持 **.md / .txt / .pdf** 格式，上传后自动入库")
                upload_input = gr.File(
                    label="选择文件",
                    file_types=[".md", ".txt", ".pdf"],
                )
                upload_btn = gr.Button("📤 上传并入库", variant="primary")
                upload_result = gr.Markdown("*等待上传...*")

            # ─── Tab 3: 文档库 ───
            with gr.Tab("📚 文档库"):
                refresh_btn = gr.Button("🔄 刷新列表")
                docs_output = gr.Markdown("*点击刷新...*")

            # ─── Tab 4: 评估 ───
            with gr.Tab("📊 检索评估"):
                gr.Markdown("""
                ### 评估指标体系

                基于 12 个测试查询（精确关键词 / 语义匹配 / 混合场景三类），对比四种检索策略：

                | 策略 | MRR | Hit@1 | Hit@3 | Hit@5 | NDCG@5 |
                |------|-----|-------|-------|-------|--------|
                | 纯 Dense (Chroma) | 0.903 | 83.3% | 100% | 100% | 0.842 |
                | 纯 BM25 (jieba) | 0.840 | 75.0% | 91.7% | 100% | 0.702 |
                | 混合 Dense+BM25+RRF | 0.917 | 83.3% | 100% | 100% | 0.793 |
                | **混合 + Reranker 精排** | **0.933** | **91.7%** | 91.7% | 100% | **0.851** |

                **分场景 MRR 对比：**

                | 场景 | 纯 Dense | 纯 BM25 | 混合+RRF | 混合+Reranker |
                |------|----------|---------|----------|---------------|
                | 精确关键词 (4 queries) | 0.833 | **0.875** | 0.875 | **1.000** |
                | 语义匹配 (4 queries) | **0.875** | 0.646 | **0.875** | 0.800 |
                | 混合场景 (4 queries) | 1.000 | 1.000 | 1.000 | 1.000 |

                > 💡 关键结论：BM25 在精确术语查询上反超 Dense（0.875 vs 0.833），验证了双路互补架构的必要性；Reranker 将整体 MRR 从 0.903 提升至 0.933。
                """)

        # ─── Event handlers ───
        def handle_query(query, selected_mode):
            """流式问答：逐段 yield 累积文本，兼容 Gradio 6 的 generator 流式输出"""
            if not query or not query.strip():
                yield "⚠️ 请输入问题"
                return

            if selected_mode == "仅检索 (Debug)":
                yield _retrieval_only(query)
                return

            # 流式模式：逐 token 累积，每次 yield 完整文本让 Markdown 组件刷新
            accumulated = ""
            for chunk in _stream_query(query):
                accumulated += chunk
                yield accumulated

        submit_btn.click(
            fn=handle_query,
            inputs=[query_input, mode],
            outputs=result_output,
        )
        query_input.submit(
            fn=handle_query,
            inputs=[query_input, mode],
            outputs=result_output,
        )

        upload_btn.click(
            fn=upload_file,
            inputs=[upload_input],
            outputs=upload_result,
        )

        refresh_btn.click(
            fn=list_documents,
            inputs=[],
            outputs=docs_output,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
