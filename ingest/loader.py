"""文档加载器：支持 .md / .txt / .pdf，PDF 采用多策略回退解析"""

from typing import List, Optional, Callable
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
import os
import logging

logger = logging.getLogger(__name__)

# 项目根目录下的上传文件夹
_UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "uploads"
)

# 支持的文件扩展名
_SUPPORTED_EXTS = (".md", ".txt", ".pdf")

# 单页最低文本长度阈值：低于此值认为该页为扫描件/图片，触发 OCR 回退
_MIN_TEXT_PER_PAGE = 50


# ═══════════════════════════════════════════════════════════════════
# PDF 解析策略
# ═══════════════════════════════════════════════════════════════════

def _parse_pymupdf(file_path: str) -> List[Document]:
    """策略1: PyMuPDF (fitz) — 最快，适合大多数数字 PDF

    提取纯文本，对表格区域用 markdown 格式标注。
    """
    import fitz  # pymupdf

    docs: List[Document] = []
    with fitz.open(file_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text")  # 纯文本模式
            if not text or not text.strip():
                continue
            docs.append(Document(
                page_content=text.strip(),
                metadata={
                    "source": file_path,
                    "page": page_num + 1,
                    "parser": "pymupdf",
                },
            ))
    return docs


def _parse_pdfplumber(file_path: str) -> List[Document]:
    """策略2: pdfplumber — 表格提取更强，适合有表格的 PDF"""
    import pdfplumber

    docs: List[Document] = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            # 检测表格并追加到文本末尾（markdown 格式）
            tables = page.extract_tables()
            if tables:
                text += "\n\n---\n"
                for t_idx, table in enumerate(tables, 1):
                    text += f"\n**表格 {t_idx}:**\n"
                    text += _table_to_markdown(table)
                    text += "\n"

            docs.append(Document(
                page_content=text.strip(),
                metadata={
                    "source": file_path,
                    "page": page_num,
                    "parser": "pdfplumber",
                },
            ))
    return docs


def _parse_ocr(file_path: str) -> List[Document]:
    """策略3: Tesseract OCR — 扫描件/图片型 PDF 的最后防线

    需要系统安装 tesseract-ocr:
      sudo apt install tesseract-ocr tesseract-ocr-chi-sim
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        logger.warning("[OCR] pytesseract 或 pdf2image 未安装，跳过 OCR")
        return []

    # 检查 tesseract 是否可用
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        logger.warning("[OCR] Tesseract 未安装或不在 PATH 中，跳过 OCR")
        return []

    docs: List[Document] = []
    try:
        images = convert_from_path(file_path, dpi=200)
    except Exception as e:
        logger.warning(f"[OCR] pdf2image 转换失败: {e}")
        return []

    for page_num, image in enumerate(images, start=1):
        try:
            # 中英文混合 OCR
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")
        except Exception:
            # 回退到仅英文
            try:
                text = pytesseract.image_to_string(image, lang="eng")
            except Exception as e:
                logger.warning(f"[OCR] 第 {page_num} 页识别失败: {e}")
                continue

        if not text or not text.strip():
            continue

        docs.append(Document(
            page_content=text.strip(),
            metadata={
                "source": file_path,
                "page": page_num,
                "parser": "ocr+tesseract",
            },
        ))

    return docs


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
    """将 pdfplumber 提取的二维表格转为 GitHub-flavored markdown"""
    if not table:
        return ""

    # 过滤全空行
    rows = [[cell or "" for cell in row] for row in table if any(cell for cell in row)]
    if not rows:
        return ""

    # 计算每列最大宽度
    col_count = max(len(row) for row in rows)
    col_widths = [0] * col_count
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    lines = []
    # 表头
    header = "| " + " | ".join(str(rows[0][i]) if i < len(rows[0]) else "" for i in range(col_count)) + " |"
    lines.append(header)
    # 分隔线
    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    lines.append(sep)
    # 数据行
    for row in rows[1:]:
        line = "| " + " | ".join(str(row[i]) if i < len(row) else "" for i in range(col_count)) + " |"
        lines.append(line)

    return "\n".join(lines)


def _pages_to_text(docs: List[Document]) -> str:
    """将所有页合并为单个文本，用于质量检查"""
    return "\n".join(d.page_content for d in docs)


# ═══════════════════════════════════════════════════════════════════
# 主加载器
# ═══════════════════════════════════════════════════════════════════

class PDFTextLoader(TextLoader):
    """多策略 PDF 加载器，兼容 LangChain TextLoader 接口

    解析策略（按优先级自动回退）：
      1. PyMuPDF (fitz)    — 数字 PDF，最快
      2. pdfplumber         — 表格多时使用
      3. Tesseract OCR     — 扫描件 / 图片型 PDF

    可通过环境变量 PDF_PARSE_STRATEGY 强制指定策略：
      - "pymupdf"   — 仅用 PyMuPDF
      - "pdfplumber" — 仅用 pdfplumber（含表格提取）
      - "ocr"       — 仅用 OCR
      - "auto"      — 自动回退（默认）
    """

    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, autodetect_encoding=True, **kwargs)
        self._strategy = os.getenv("PDF_PARSE_STRATEGY", "auto")

    def load(self) -> List[Document]:
        strategies: List[tuple] = []

        if self._strategy == "auto":
            # pdfplumber 优先：企业文档常含表格，其 Markdown 表格输出质量最高
            # pymupdf 次之：纯文本 PDF 更快更轻量
            # ocr 兜底：扫描件 / 图片型 PDF
            strategies = [
                ("pdfplumber", _parse_pdfplumber),
                ("pymupdf", _parse_pymupdf),
                ("ocr", _parse_ocr),
            ]
        elif self._strategy == "pymupdf":
            strategies = [("pymupdf", _parse_pymupdf)]
        elif self._strategy == "pdfplumber":
            strategies = [("pdfplumber", _parse_pdfplumber)]
        elif self._strategy == "ocr":
            strategies = [("ocr", _parse_ocr)]
        else:
            logger.warning(f"未知策略 '{self._strategy}'，使用 auto 模式")
            strategies = [
                ("pymupdf", _parse_pymupdf),
                ("pdfplumber", _parse_pdfplumber),
                ("ocr", _parse_ocr),
            ]

        all_docs: List[Document] = []
        used_strategy = "none"

        for strategy_name, strategy_fn in strategies:
            try:
                docs = strategy_fn(self.file_path)
            except Exception as e:
                logger.warning(f"[PDF] {strategy_name} 解析异常: {e}")
                continue

            total_chars = len(_pages_to_text(docs))
            avg_text = total_chars / max(len(docs), 1)

            if docs and avg_text >= _MIN_TEXT_PER_PAGE:
                all_docs = docs
                used_strategy = strategy_name
                break
            elif docs:
                # 提取到文本但质量不足，保留结果继续尝试下一策略
                all_docs = docs
                used_strategy = f"{strategy_name} (low quality)"

        if not all_docs:
            logger.warning(
                f"[PDF] ⚠ {os.path.basename(self.file_path)}  "
                f"所有策略均未能提取有效文本（可能是纯图片扫描件且无 OCR 环境）"
            )
        else:
            page_count = len(all_docs)
            total_chars = len(_pages_to_text(all_docs))
            logger.info(
                f"[PDF] {os.path.basename(self.file_path)} → "
                f"{used_strategy} | {page_count} 页 | {total_chars} 字符"
            )

        return all_docs


def _get_loader(file_path: str) -> TextLoader:
    """根据文件扩展名返回合适的加载器"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PDFTextLoader(file_path)
    return TextLoader(file_path, autodetect_encoding=True)


def document_load(file_path: Optional[str] = None) -> List[Document]:
    """
    加载待入库文档

    Args:
        file_path: 指定单个文件路径时只加载该文件；为 None 时加载整个 uploads 目录

    Returns:
        LangChain Document 列表
    """
    upload_path = os.path.normpath(_UPLOAD_DIR)

    if file_path:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in _SUPPORTED_EXTS:
            raise ValueError(
                f"仅支持 {', '.join(_SUPPORTED_EXTS)} 文件: {file_path}"
            )
        loader = _get_loader(file_path)
        return loader.load()

    # 全量模式：扫描整个 uploads 目录
    if not os.path.isdir(upload_path):
        raise FileNotFoundError(f"上传目录不存在: {upload_path}")

    all_docs: List[Document] = []
    for ext, loader_cls in (
        (".md", TextLoader),
        (".txt", TextLoader),
        (".pdf", PDFTextLoader),
    ):
        loader = DirectoryLoader(
            path=upload_path,
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            silent_errors=True,
            show_progress=True,
        )
        all_docs.extend(loader.load())

    return all_docs
