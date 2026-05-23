import uuid
from typing import List
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document

# Rich 库控制台美化
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from schema import ChunkMetadata, DocumentChunk

console = Console()

# ==================== 2. 可视化与打印逻辑 ====================


def print_chunks(chunks: list, stage_name: str, parent_idx: int = None):
    """
    统一的切分结果美化打印函数（支持 LangChain Document 和 Pydantic DocumentChunk）
    """
    console.print(
        f"[bold green]▶ [{stage_name}] 产生了 [bold yellow]{len(chunks)}[/bold yellow] 个分块：[/bold green]"
    )

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, DocumentChunk):
            page_content = chunk.page_content
            raw_metadata = chunk.metadata.model_dump(by_alias=True)
            # 动态获取当前切片结构内部的真实索引
            current_parent_idx = chunk.metadata.parent_chunk_idx
            current_sub_idx = chunk.metadata.sub_chunk_idx
        else:
            page_content = chunk.page_content
            raw_metadata = chunk.metadata
            current_parent_idx = parent_idx
            current_sub_idx = i

        # 提取并高亮元数据
        meta_items = []
        for k, v in raw_metadata.items():
            if v is not None and k not in ["prev_chunk_id", "next_chunk_id"]:
                meta_items.append(f"[cyan]{k}:[/cyan] {v}")

        meta_info = (
            " | ".join(meta_items)
            if meta_items
            else "[dim white]暂无基础元数据[/dim white]"
        )

        # 组装面板正文
        content_text = Text()
        content_text.append("📌 元数据: ", style="bold magenta")
        content_text.append(Text.from_markup(meta_info))

        # 此时打印，由于指针已全部织好，NEXT 就会完美显现
        if isinstance(chunk, DocumentChunk):
            prev_id = (
                chunk.metadata.prev_chunk_id[-8:]
                if chunk.metadata.prev_chunk_id
                else "None"
            )
            next_id = (
                chunk.metadata.next_chunk_id[-8:]
                if chunk.metadata.next_chunk_id
                else "None"
            )
            content_text.append(
                f"\n🔗 链式上下文: [dim]PREV ->[/dim] {prev_id} | [dim]NEXT ->[/dim] {next_id}",
                style="bold yellow",
            )

        content_text.append("\n" + "─" * 50 + "\n", style="dim")
        content_text.append(page_content, style="white")

        # 动态计算编号：如果是二阶段，展示其真实的 parent-sub 关系
        if isinstance(chunk, DocumentChunk):
            idx_str = f"{current_parent_idx}-{current_sub_idx}"
        else:
            idx_str = (
                f"{current_parent_idx}-{current_sub_idx}"
                if current_parent_idx is not None
                else f"{current_sub_idx}"
            )

        panel_title = f"[bold sky_blue3]第 {idx_str} 块[/bold sky_blue3]"
        panel_subtitle = f"[bold dark_sea_green4] 📏 长度: {len(page_content)} 字 [/bold dark_sea_green4]"

        console.print(
            Panel(
                content_text,
                title=panel_title,
                subtitle=panel_subtitle,
                subtitle_align="right",
                border_style="deep_sky_blue1"
                if not isinstance(chunk, DocumentChunk)
                else "purple",
                expand=False,
            )
        )
    console.print("")


# ==================== 3. 核心切分逻辑 ====================

# 将无状态的 splitter 提升到模块级别，避免每次调用 document_chunk 时重复实例化
_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ],
)

_text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " ", ""],
    # 中文场景下 800 字能保留较完整语义；150 字重叠防止跨 chunk 边界截断关键上下文
    chunk_size=800,
    chunk_overlap=150,
)


def document_chunk(docs: List[Document]) -> List[DocumentChunk]:
    """对文档进行双层切分，并将结果封装为 Pydantic 数据模型列表返回"""
    final_chunks: List[DocumentChunk] = []

    for doc_idx, doc in enumerate(docs):
        console.print(
            f"\n[bold reverse ocean_blue] 📄 开始处理第 {doc_idx + 1} 个原始文档 [/bold reverse ocean_blue]\n"
        )

        source_file = doc.metadata.get("source", f"doc_{uuid.uuid4().hex[:8]}")

        # === 🔧 第一阶段：Markdown 标题切分 ===
        md_chunks = _md_splitter.split_text(doc.page_content)
        print_chunks(md_chunks, stage_name="一阶段：Markdown 标题切分")

        doc_start_idx = len(final_chunks)

        # === 🔧 第二阶段：针对大块进行字数微切分并组装 Pydantic 模型 ===
        for i, md_chunk in enumerate(md_chunks):
            sub_chunks = _text_splitter.split_documents([md_chunk])

            for j, sub_chunk in enumerate(sub_chunks):
                # 过滤掉纯空白 chunk，避免入库无意义的零向量
                if not sub_chunk.page_content.strip():
                    continue

                # 使用 :: 替代 # 作为分隔符，避免某些向量数据库的 URL 编码兼容问题
                current_chunk_id = f"{source_file}::ch{i}::sub{j}"

                metadata_obj = ChunkMetadata(
                    header_1=sub_chunk.metadata.get("Header 1"),
                    header_2=sub_chunk.metadata.get("Header 2"),
                    header_3=sub_chunk.metadata.get("Header 3"),
                    doc_id=source_file,
                    chunk_id=current_chunk_id,
                    parent_chunk_idx=i,
                    sub_chunk_idx=j,
                    chunk_length=len(sub_chunk.page_content),
                )

                chunk_model = DocumentChunk(
                    page_content=sub_chunk.page_content, metadata=metadata_obj
                )

                # 建立相邻块的前后指针，仅限于同一文档内的连续 chunk
                if len(final_chunks) > 0:
                    prev_chunk_model = final_chunks[-1]
                    if prev_chunk_model.metadata.doc_id == source_file:
                        prev_chunk_model.metadata.next_chunk_id = current_chunk_id
                        chunk_model.metadata.prev_chunk_id = (
                            prev_chunk_model.metadata.chunk_id
                        )

                final_chunks.append(chunk_model)

        # 文档处理完毕后统一打印，此时指针已全部织好
        current_doc_chunks = final_chunks[doc_start_idx:]
        print_chunks(current_doc_chunks, stage_name="二阶段：字数微切编织")

    return final_chunks
