from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
from document_chunk import document_chunk

current_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(current_dir, "../documents")
loader = DirectoryLoader(
    path=docs_dir,
    glob="**/*.md",
    loader_cls=TextLoader,
    silent_errors=True,
    show_progress=True,
)

docs = loader.load()

print(f"一共加载了{len(docs)}个文档")

document_chunk(docs=docs)
