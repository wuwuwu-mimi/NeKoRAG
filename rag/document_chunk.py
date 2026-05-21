from typing import List
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def print_chunks(chunks: List[Document], stage_name: str, parent_idx: int = None):
    """
    统一的切分结果美化打印函数
    :param chunks: 切分后的 Document 列表
    :param stage_name: 当前处于哪一个切分阶段
    :param parent_idx: 父级分块的索引（用于二次切分时标识所属关系）
    """
    console.print(f"[bold green]▶ [{stage_name}] 产生了 [bold yellow]{len(chunks)}[/bold yellow] 个分块：[/bold green]")
    
    for i, chunk in enumerate(chunks):
        # 提取并高亮元数据
        meta_info = " | ".join([f"[cyan]{k}:[/cyan] {v}" for k, v in chunk.metadata.items()])
        if not meta_info:
            meta_info = "[dim white]暂无元数据[/dim white]"
            
        # 组装面板正文
        content_text = Text()
        content_text.append("📌 元数据: ", style="bold magenta")
        content_text.append(Text.from_markup(meta_info))
        content_text.append("\n" + "─" * 50 + "\n", style="dim")
        content_text.append(chunk.page_content, style="white")
        
        # 动态计算编号：如果存在父索引，则显示为 "X-Y"，否则显示 "X"
        idx_str = f"{parent_idx}-{i}" if parent_idx is not None else f"{i}"
        
        panel_title = f"[bold sky_blue3]第 {idx_str} 块[/bold sky_blue3]"
        panel_subtitle = f"[bold dark_sea_green4] 📏 长度: {len(chunk.page_content)} 字 [/bold dark_sea_green4]"
        
        console.print(
            Panel(
                content_text,
                title=panel_title,
                subtitle=panel_subtitle,
                subtitle_align="right",
                border_style="deep_sky_blue1" if parent_idx is None else "purple", # 二次切分用紫色框区分
                expand=False
            )
        )
    console.print("") # 阶段结束后留空行


def document_chunk(docs: List[Document]) -> List[Document]:
    """
    对文档进行双层切分，确保元数据不丢失
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "，", " ", ""],
        chunk_size=400,
        chunk_overlap=20,
    )

    final_chunks = [] # 用于存放最终切分好的最细颗粒度 Document

    for doc_idx, doc in enumerate(docs):
        console.print(f"\n[bold reverse ocean_blue] 📄 开始处理第 {doc_idx + 1} 个原始文档 [/bold reverse ocean_blue]\n")
        
        # === 🔧 第一阶段：Markdown 标题切分 ===
        md_chunks = md_splitter.split_text(doc.page_content)
        print_chunks(md_chunks, stage_name="一阶段：Markdown 标题切分")
        
        # === 🔧 第二阶段：针对大块进行字数微切分 ===
        for i, md_chunk in enumerate(md_chunks):
            # 🔥 关键点：传入 [md_chunk] 对象列表，而不是 md_chunk.page_content 字符串
            # 这样 text_splitter 会自动把 md_chunk.metadata 复制给切分出来出来的每一个子 Document
            sub_chunks = text_splitter.split_documents([md_chunk])
            
            # 打印展示：带上父级编号 i，并且能看到元数据被完美继承了
            print_chunks(sub_chunks, stage_name=f"二阶段：对第 {i} 块进行字数微切", parent_idx=i)
            
            # 收集最终结果
            final_chunks.extend(sub_chunks)
            
    return final_chunks