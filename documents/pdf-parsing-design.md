# PDF 多策略解析：当 pdfplumber 不够用，让三引擎协同作战

> NeKoRAG 文档解析模块的设计复盘：为什么单一解析器覆盖不了真实世界的 PDF，以及三层回退架构如何兜底

---

## 问题：一个解析器走不远的现实

NeKoRAG 最初接入 PDF 时只用了 **pdfplumber**，API 上线后很快暴露了几个失败场景：

### 场景一：扫描件 / 图片型 PDF

某份供应商发来的产品规格书是扫描版——每一页都是图片，pdfplumber 提取到的文本为空字符串。用户上传后系统静默入库了 0 个切片，查询时永远命中不了这份文档。

### 场景二：表格结构丢失

一份财务季报含大量数据表格。pdfplumber 虽能提取出文字，但默认的 `extract_text()` 把表格的单元格按坐标排序后拼成纯文本：

```
Model Params Throughput Latency Cost/1M
DeepSeek-V4 236B 156 tok/s 0.8s $0.14
```

这对人眼可读，但对 LLM 来说，列与列之间的语义关系被抹平了。Reranker 给这段文字的分数偏低，因为 query 中 "DeepSeek-V4 的吞吐量" 跟 "156 tok/s" 在文本中的距离被"236B"隔开了。

### 场景三：性能取舍

PyMuPDF 提取一页只需 ~2ms，pdfplumber 需要 ~15ms。对于 200 页的技术手册，pdfplumber 全量解析耗时超过 3 秒，API 响应超时。

**核心矛盾：没有一种解析器能同时满足速度、表格质量、扫描件兜底三个需求。**

---

## 解法：三层回退架构

参考数据库领域"写时优化 vs 读时优化"的思路，将解析拆为三个独立策略，按质量优先 → 速度优先 → 兜底的顺序自动回退：

```
PDF 文件
  │
  ├─ 策略1: pdfplumber     ← 默认优先
  │   ├─ 文本提取 + Markdown 表格转换
  │   ├─ 适用: 企业文档（合同/报表/说明书）
  │   └─ 失败判定: 平均每页 < 50 字符 → 触发回退
  │
  ├─ 策略2: PyMuPDF (fitz)  ← 速度备选
  │   ├─ 原生 C 引擎，纯文本模式
  │   ├─ 适用: 文字型 PDF、大文件快速处理
  │   └─ 失败判定: 同样按 50 字符/页 阈值
  │
  └─ 策略3: Tesseract OCR   ← 最后防线
      ├─ pdf2image 渲染 → pytesseract 识别
      ├─ 适用: 扫描件、图片型 PDF
      └─ 前提: 系统已安装 tesseract
```

### 为什么 pdfplumber 排第一而不是 PyMuPDF？

实测同一份表格 PDF（`02_table_heavy.pdf`）：

| 指标 | PyMuPDF | pdfplumber |
|------|---------|------------|
| 提取字符数 | 722 | **1351** |
| 表格结构 | 散列为分行文本 | **完整 Markdown 表格** |
| 下游检索 Top-1 分数 | 0.82 | **0.99** |

PyMuPDF 的 `get_text("text")` 按物理坐标排序文本，表格的单元格被打散成独立的行——"DeepSeek-V4" 和 "156 tok/s" 之间插入了 "236B"，导致向量检索时语义关联减弱。

pdfplumber 的 `extract_tables()` 能识别表格区域，通过 `_table_to_markdown()` 转为 GitHub-flavored Markdown：

```markdown
| Model | Params | Throughput | Latency | Cost/1M |
|-------|--------|------------|---------|---------|
| DeepSeek-V4 | 236B | 156 tok/s | 0.8s | $0.14 |
| Qwen3.5-72B | 72B | 210 tok/s | 0.5s | $0.09 |
```

Markdown 表格保留了列间语义关系。LLM 读到这段时，"156 tok/s" 被明确标注在 "Throughput" 列下，对 query 中"吞吐量"的理解误差大幅降低。

---

## 核心实现

### 策略接口

三个策略函数签-名统一为 `(file_path: str) -> List[Document]`，元数据中标记 `parser` 字段方便追溯：

```python
def _parse_pdfplumber(file_path: str) -> List[Document]:
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    text += "\n" + _table_to_markdown(table)
            docs.append(Document(page_content=text, metadata={"parser": "pdfplumber"}))
    return docs
```

### 表格 → Markdown 转换

pdfplumber 的 `extract_tables()` 返回 `List[List[str]]`，需要转为人类可读且 LLM 友好的格式：

```python
def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    # 1. 过滤全空行
    rows = [[cell or "" for cell in row] for row in table if any(cell for cell in row)]
    # 2. 计算每列宽度做对齐
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(col_count)]
    # 3. 生成 GFM 格式
    header = "| " + " | ".join(...) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    ...
```

### 回退逻辑

`auto` 模式下，按策略列表顺序尝试。每个策略执行后检查质量：

```python
MIN_TEXT_PER_PAGE = 50  # 单页最低字符阈值

for strategy_name, strategy_fn in strategies:
    docs = strategy_fn(file_path)
    avg_text = total_chars / max(len(docs), 1)

    if docs and avg_text >= MIN_TEXT_PER_PAGE:
        return docs       # ← 质量达标，停止回退
    elif docs:
        all_docs = docs   # ← 保留低质量结果，继续尝试下个策略
```

- 阈值 50 字符/页：基于经验——正常 A4 页至少数百字符，低于 50 几乎肯定是扫描件
- 保留低质量结果：防止所有策略都失败时返回空（比如 tesseract 未安装时 OCR 策略直接 return []）

### 环境变量控制

```bash
# 默认 auto，按 pdfplumber → pymupdf → ocr 回退
PDF_PARSE_STRATEGY=auto

# 强制指定（调试 / 性能优化）
PDF_PARSE_STRATEGY=pymupdf    # 纯文本 PDF，追求速度
PDF_PARSE_STRATEGY=pdfplumber # 表格多，追求质量
PDF_PARSE_STRATEGY=ocr        # 扫描件批量处理
```

---

## 测试集设计

在 `data/test_pdfs/` 下准备了 4 类 PDF（通过 `evaluation/test_pdfs.py` 生成）：

| 测试文件 | 类型 | 页数 | 特征 |
|---------|------|------|------|
| `01_text_only.pdf` | 纯文本 | 1 | 段落 + 列表，最简场景 |
| `02_table_heavy.pdf` | 表格密集 | 1 | 2 张数据表，含性能指标 |
| `03_mixed_content.pdf` | 混合内容 | 2 | 文本 + 表格 + 多页 |
| `04_minimal_text.pdf` | 低文本 | 2 | 模拟 PPT 导出，每页几行 |

验证结果（auto 模式）：

```
01_text_only.pdf      pdfplumber   1页  1001字符
02_table_heavy.pdf    pdfplumber   1页  1351字符  ← 含 Markdown 表格
03_mixed_content.pdf  pdfplumber   2页  1760字符  ← 含 1 页表格
04_minimal_text.pdf   pdfplumber   2页   145字符
```

### 如何测试更复杂的真实 PDF

上述测试集用 `fpdf2` 生成，无法覆盖扫描件和中文混排。真实场景验证建议：

1. **中文合同/标书 PDF**：找一份含表格和印章的真实合同，用 `PDF_PARSE_STRATEGY=pdfplumber` 测试表格提取，观察中文列名是否完整
2. **扫描版论文**：找一份扫描版 PDF（arxiv 老论文的扫描版很常见），用 `PDF_PARSE_STRATEGY=ocr` 测试，前提是安装 `tesseract-ocr-chi-sim`
3. **财务报告 PDF**：上市公司年报（数字表格密集），对比三种策略的检索命中率

---

## 局限与后续方向

### 1. 复杂版式仍会翻车

三栏排版、嵌套表格、图文混排——pdfplumber 的坐标排序会错位。解法：

- 接入 **Docling**（IBM 开源）：深度学习布局分析，能识别标题/正文/表格/图片区域
- 或 **LlamaParse**：VLM 驱动的 Agentic OCR，对复杂版式鲁棒

### 2. 扫描件 OCR 需手动装依赖

```bash
sudo apt install tesseract-ocr tesseract-ocr-chi-sim
```

Docker 镜像里预装即可，不算大问题。但 Tesseract 对中文手写体、竖排文字、低分辨率扫描件的识别率有限——生产环境建议用 PaddleOCR 或云端 OCR API 替代。

### 3. 大 PDF 全内存加载

`pdfplumber.open()` 和 `fitz.open()` 都会把整个 PDF 加载到内存。200MB+ 的 PDF 会导致 OOM。解法：

- 流式解析：逐页 `pdfplumber.open(path, pages=[i])` 
- 分页入库：解析一页 → 向量化一页 → 释放内存

### 4. Word / PPT 支持

当前仅支持 .pdf / .md / .txt。企业场景下 Word 和 PPT 占比很高：

```python
# 后续添加
# ingest/loader.py
elif ext == ".docx":
    import docx
    text = "\n".join(p.text for p in docx.Document(path).paragraphs)
```

### 5. 缺少解析质量评估

当前只有人工抽查。理想方案：

- 准备一批"标准答案"PDF（已知文本内容）
- 对比不同策略提取的文本与标准答案的编辑距离
- 作为 CI 流程的一部分，每次改 loader 时自动跑

---

## 架构图

```
┌──────────────────────────────────────────────────────┐
│                    PDFTextLoader                      │
│                                                      │
│  strategy = os.getenv("PDF_PARSE_STRATEGY", "auto")  │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  pdfplumber  │  │   PyMuPDF    │  │ Tesseract  │ │
│  │              │  │              │  │    OCR     │ │
│  │ • 文本提取   │  │ • 文本提取   │  │ • 图片渲染 │ │
│  │ • 表格检测   │  │ • 原生C引擎  │  │ • 中英OCR  │ │
│  │ • MD表格生成 │  │ • 速度最快   │  │ • 兜底策略 │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                 │                │         │
│         └────────┬────────┴───────┬────────┘         │
│                  │                │                  │
│              quality_check   quality_check           │
│           (avg_chars/page >= 50)                     │
│                  │                                   │
│                  ▼                                   │
│          ┌──────────────┐                            │
│          │ List[Document]│                            │
│          │  metadata:    │                            │
│          │   - source    │                            │
│          │   - page      │                            │
│          │   - parser    │  ← 追溯用哪条策略          │
│          └──────────────┘                            │
└──────────────────────────────────────────────────────┘
```

---

> 关键词：PDF Parsing · Multi-Strategy · pdfplumber · PyMuPDF · OCR · Table Extraction · RAG Ingestion
