# NeKoRAG

基于 **混合检索 + 精排 + 上下文扩展** 的 RAG 后台服务，为下游 Agent / Chatbot 提供知识库检索与生成 API。

## 检索流程

```
用户 Query
  │
  ├─ 稠密检索 (ChromaDB + bge-m3)  ─┐
  ├─ 稀疏检索 (BM25 + jieba)         ─┤  RRF 融合  ─→  Top-20 候选
  └───────────────────────────────────┘
                    │
                    ▼
           Cross-encoder 精排 (Jina Reranker)
                    │
                    ▼
           expand_context (沿链表指针扩展相邻 chunk)
                    │
                    ▼
           LLM 生成回答 (DeepSeek, SSE 流式)
```

### 关键模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 文档加载 | `ingest/loader.py` | 支持 .md/.txt，UUID 重命名防冲突 |
| 文本切片 | `ingest/chunker.py` | Markdown 层级感知切片，织入链表指针 |
| 入库管线 | `ingest/pipeline.py` | 全量 / 增量模式，自动重建索引 |
| 稠密检索 | `retrieval/embeddings.py` | Ollama bge-m3:latest 向量化 |
| 稀疏检索 | `retrieval/bm25.py` | jieba 分词 + BM25，索引持久化磁盘 |
| 混合检索 | `retrieval/hybrid.py` | RRF 融合双路召回 |
| 精排 | `retrieval/reranker.py` | Jina AI Cross-encoder，支持上下文扩展 |
| 生成 | `generation/generator.py` | DeepSeek API，支持流式 SSE |
| 向量存储 | `retrieval/vector_store.py` | ChromaDB 持久化 |
| REST API | `api/` | FastAPI，CORS 开放 |

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置 .env
cp .env.example .env
# 编辑 .env: JINA_API_KEY, DEEPSEEK_API_KEY 等

# 3. 确保 Ollama 运行并拉取 embedding 模型
ollama pull bge-m3:latest

# 4. 上传文档
curl -X POST http://localhost:8000/v1/documents/upload \
  -F "file=@docs/example.md"

# 5. 检索问答
curl -X POST http://localhost:8000/v1/query/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "你的问题"}'
```

## API 端点

### 文档管理

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/v1/documents/upload` | 上传文档，触发入库 |
| GET | `/v1/documents` | 列出已入库文档 |
| DELETE | `/v1/documents/by-id?doc_id=xxx` | 删除文档及切片 |
| GET | `/v1/documents/by-id/status?doc_id=xxx` | 查询文档状态 |
| POST | `/v1/documents/reset` | 清空并重建全部索引 |

### 检索问答

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/v1/query/generate` | 检索 + LLM 生成回答 |
| POST | `/v1/query/stream` | SSE 流式生成 |
| POST | `/v1/query/retrieval-only` | 仅检索，跳过 LLM（调试用） |

### 健康检查

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/health` | 服务状态 |

## 混合检索原理

使用 **RRF (Reciprocal Rank Fusion)** 融合两路独立检索：

1. **稠密路径 (Dense):** 查询向量 → ChromaDB 语义相似度 Top-K
2. **稀疏路径 (Sparse):** 查询分词 → BM25 关键词匹配 Top-K
3. **融合:** 对每个 chunk 计算 `RRF_score = Σ 1/(k + rank_i)`，k=60
4. **精排:** Cross-encoder 对融合结果重新打分
5. **扩展:** 沿链表指针拉取相邻 chunk，覆盖跨切片内容

两路检索互补：向量检索擅长语义相近但用词不同的内容，BM25 检索擅长精确关键词匹配。

## 项目结构

```
NeKoRAG/
├── api/                  # FastAPI 应用
│   ├── __init__.py       # app 定义 + CORS
│   ├── documents.py      # 文档管理端点
│   ├── query.py          # 检索问答端点
│   └── schemas.py        # Pydantic 模型
├── generation/           # LLM 生成
│   └── generator.py      # DeepSeek 同步 + 流式
├── ingest/               # 入库
│   ├── loader.py         # 文件加载
│   ├── chunker.py        # 文本切片
│   └── pipeline.py       # 入库管线
├── retrieval/            # 检索
│   ├── embeddings.py     # 向量化
│   ├── bm25.py           # BM25 索引
│   ├── hybrid.py         # RRF 混合检索
│   ├── reranker.py       # 精排 + 上下文扩展
│   └── vector_store.py   # ChromaDB 写入
├── schema/               # 数据模型
│   └── chunk_schema.py   # DocumentChunk
├── data/                 # 持久化数据
│   ├── uploads/          # 上传文件 + sidecar
│   ├── chroma_db/        # ChromaDB
│   └── bm25_index/       # BM25 索引
├── main.py               # CLI 测试入口
└── pyproject.toml
```

## 启动

```bash
uv run uvicorn api:app --port 8000 --reload
```
