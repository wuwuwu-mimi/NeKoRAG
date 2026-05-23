# RAG 企业知识库问答系统 — 实现计划

## Context

用户已有 Cocon（多 Agent 协作引擎）项目，需要补充一个 RAG 项目来完成求职作品组合。Cocon 证明 Agent 能力，这个项目证明 RAG 能力，两者互补。目标是做出一个面试能演示、代码能展示深度、架构能讲清楚的项目。

---

## 项目定位

一个**企业知识库智能问答系统**，核心卖点：
- 上传文档 → 自动处理 → 语义检索 → 带来源引用的精准回答
- 从"能跑"到"生产可部署"——覆盖完整 RAG pipeline
- 混合检索 + Reranker 重排序 → 检索质量有数据支撑，不是调个API就完事
- 可以和 Cocon 联动：Cocon Agent 通过工具调用这个 RAG 系统

---

## 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| Web 框架 | FastAPI | 和 Cocon 一致，异步支持好 |
| LLM 调用 | langchain-openai | 兼容 DeepSeek API |
| RAG 编排 | LlamaIndex | 专注 RAG，比 LangChain 的 RAG 模块更成熟 |
| 向量数据库 | Chroma（本地）+ Milvus（可选，展示升级能力） | Chroma 零配置启动，Milvus 证明你懂生产级 |
| 文档解析 | unstructured + pdfplumber + python-docx | 覆盖 PDF/Word/MD/TXT |
| Embedding | BAAI/bge-m3 或 text-embedding-3-small | 支持中英文，效果好 |
| 检索策略 | BM25（关键词）+ 向量（语义）混合 | 这是 RAG 面试的核心考点 |
| Reranker | BAAI/bge-reranker-v2-m3 | 开源免费，效果接近 Cohere |
| 前端 | Streamlit | 和 Cocon 一致，快速搭建 |
| 部署 | Docker + docker-compose | 一键启动 |
| 可观测性 | LangFuse（可选） | 追踪检索质量 + Token 成本 |
| 包管理 | uv + pyproject.toml | 和 Cocon 一致 |

---

## 项目结构

```
rag_vault/
├── main.py                     # FastAPI 入口
├── pyproject.toml
├── .env / .env.example
├── docker-compose.yml
├── Dockerfile
├── data/                       # 用户上传的文档 + Chroma 持久化
│   ├── uploads/
│   └── chroma_db/
├── ingest/
│   ├── __init__.py
│   ├── loader.py               # 文档加载（PDF/Word/MD/TXT）
│   ├── chunker.py              # 智能分块（语义感知 + 滑动窗口）
│   └── pipeline.py             # 完整摄入 pipeline
├── retrieval/
│   ├── __init__.py
│   ├── embeddings.py            # Embedding 模型封装
│   ├── vector_store.py          # Chroma 向量数据库封装
│   ├── bm25.py                  # BM25 关键词检索
│   ├── hybrid.py               # 混合检索（融合向量 + BM25）
│   └── reranker.py             # Reranker 重排序
├── generation/
│   ├── __init__.py
│   └── generator.py            # 基于检索结果生成回答（带引用）
├── evaluation/
│   ├── __init__.py
│   └── eval.py                 # 检索质量评估（MRR, Hit Rate, NDCG）
├── api/
│   ├── __init__.py
│   ├── documents.py            # 文档上传/管理 API
│   └── query.py                # 问答 API（普通 + 流式 SSE）
├── frontend/
│   └── app.py                  # Streamlit 交互界面
└── tests/
```

---

## 分模块实现

### 阶段1：文档摄入 pipeline（2-3天）

**文件：`ingest/loader.py`**
- `DocumentLoader` 类：根据文件后缀自动选择解析器
- 支持 PDF（pdfplumber）、Word（python-docx）、Markdown、TXT
- 输出统一格式：`{"text", "metadata": {"source", "page", "filename"}}`

**文件：`ingest/chunker.py`**
- `SemanticChunker`：优先按语义边界（段落/章节）分块，而非硬切
- `SlidingWindowChunker`：带重叠的滑动窗口，保证上下文连贯
- chunk_size: 512 tokens, overlap: 50 tokens（可配置）
- 返回 `List[Document]`，LlamaIndex 兼容格式

**文件：`ingest/pipeline.py`**
- `IngestPipeline`：串联 loader → chunker → embedding → vector_store
- 支持去重：文档hash对比，避免重复摄入
- 异步批量处理：`asyncio.gather` 并发处理多文档
- 处理进度回调（给前端用）

### 阶段2：检索系统（3-4天）★核心

**文件：`retrieval/embeddings.py`**
- `EmbeddingManager`：封装 Embedding 模型
- 默认使用 BGE-m3（HuggingFace 本地），也支持 OpenAI text-embedding-3
- 支持批量向量化（batch_size 可配置）

**文件：`retrieval/vector_store.py`**
- `VectorStoreManager`：封装 Chroma
- 集合管理（collection CRUD）
- 相似度搜索 + 过滤（按文件名、日期等元数据过滤）
- 持久化到本地磁盘 `data/chroma_db/`

**文件：`retrieval/bm25.py`**
- `BM25Retriever`：基于 jieba 分词的中文 BM25
- 收录中英文混合文档时自动切换分词策略
- 构建稀疏检索索引（内存级，可用 SQLite FTS5 做更大规模）

**文件：`retrieval/hybrid.py`** ★核心亮点
- `HybridRetriever`：融合向量检索和关键词检索
- 融合策略：RRF（Reciprocal Rank Fusion），面试高频考点
- 参数可调：向量权重 vs 关键词权重
- 返回 Top-K + 融合分数

**文件：`retrieval/reranker.py`**
- `RerankManager`：对混合检索结果做重排序
- 默认用 BGE-reranker-v2-m3（Cross-Encoder）
- 精排 Top-N（如混合检索拿20条 → Reranker精排Top-5）
- Reranker 前后的分数对比，用于评估

### 阶段3：生成与引用（1-2天）

**文件：`generation/generator.py`**
- `RAGGenerator`：检索结果 + System Prompt → LLM → 带引用的回答
- System Prompt 设计：要求引用来源、不编造、标注不确定
- 输出格式：Markdown，每个关键断言后标注 `[来源: xxx.pdf, 第X页]`
- 支持流式输出（SSE），兼容 Cocon 的流式接口

### 阶段4：API 服务（1-2天）

**文件：`api/documents.py`**
| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/documents/upload` | POST | 上传文档，触发 ingest pipeline |
| `/v1/documents` | GET | 列出已入库的文档 |
| `/v1/documents/{id}` | DELETE | 删除文档及对应向量 |
| `/v1/documents/{id}/status` | GET | 查询文档处理状态 |

**文件：`api/query.py`**
| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/query` | POST | 问答（普通） |
| `/v1/query/stream` | POST | 问答（SSE 流式） |
| `/v1/query/retrieval-only` | POST | 仅检索不生成（调试用） |

### 阶段5：前端（1天）

**文件：`frontend/app.py`** — Streamlit 界面
- 左侧：文档上传面板（拖拽上传 + 处理进度 + 文档列表）
- 右侧：问答面板（对话历史 + 回答 + 来源高亮）
- 可折叠的"检索详情"面板：展示中间检索结果和 Reranker 分数
- 和 Cocon 前端的视觉风格保持一致

### 阶段6：评估（1天）

**文件：`evaluation/eval.py`**
- `RAGEvaluator`：用合成问题 + 已知答案评估检索质量
- 指标：MRR（Mean Reciprocal Rank）、Hit Rate@K、NDCG
- 对比实验：向量检索 only vs 混合检索 vs 混合检索+Reranker
- 输出对比报告（终端打印）

### 阶段7：部署（1天）

**文件：`docker-compose.yml`** + **`Dockerfile`**
- Dockerfile：Python 3.14 + uv sync + uvicorn
- docker-compose：app + Chroma（可选独立容器）
- `.env.example` 清晰标注所有配置项

---

## 和 Cocon 的联动

Cocon Agent 可以通过工具调用这个 RAG 系统：

```python
# Cocon tools/builtin/rag_search.py
async def rag_search(query: str, collection: str = "default") -> dict:
    """检索企业知识库"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8001/v1/query/retrieval-only",
            json={"query": query, "collection": collection}
        )
    return resp.json()
```

面试时的叙事：
> "Cocon 是编排引擎，负责拆解任务和调用工具；RAG Vault 是知识检索系统，负责文档理解和精准检索。两者通过 HTTP API 连接。Agent 收到用户问题后，先判断是否需要查知识库，需要就调用 RAG Vault，拿回检索结果再推理回答。"

---

## 时间规划

| 阶段 | 内容 | 时间 |
|------|------|------|
| 1 | 文档摄入 pipeline | 2-3 天 |
| 2 | 检索系统（核心） | 3-4 天 |
| 3 | 生成与引用 | 1-2 天 |
| 4 | API 服务 | 1-2 天 |
| 5 | 前端 | 1 天 |
| 6 | 评估 | 1 天 |
| 7 | 部署 + 文档 | 1 天 |

**总计：10-14 天**

---

## 验证计划

1. **单元测试**：每个模块的关键函数（chunker 分块正确性、hybrid 融合排序、reranker 提升效果）
2. **集成测试**：上传一个 PDF → 提问 → 验证回答引用了正确来源
3. **评估脚本**：跑 `evaluation/eval.py`，对比三种检索策略的 MRR/NDCG
4. **端到端验证**：`docker-compose up`，浏览器打开 Streamlit，完成上传+提问全流程
5. **与 Cocon 联动验证**：Cocon 的任务里配置 rag_search 工具，验证 Agent 能正确调用
