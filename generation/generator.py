"""基于检索结果调用 LLM 生成带引用的回答"""

import os
from typing import Generator, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from schema.chunk_schema import RerankResult

_SYSTEM_PROMPT = """你是一个严谨的知识问答助手。请严格遵循以下规则：

1. 只根据【参考资料】中的内容回答问题，不要使用你自身的知识
2. 引用某条资料时，在句末标注来源编号，例如 [1]、[2][3]
3. 如果资料不足以回答问题，直接说"根据现有资料无法回答"，不要编造
4. 回答简洁、结构化，优先使用列表或分点说明"""


class NeKoGenerator:
    """RAG 生成器：将检索结果拼入 prompt，调用 LLM 生成回答"""

    def __init__(self):
        # DeepSeek 兼容 OpenAI 接口格式，直接用 ChatOpenAI
        self._model = os.getenv("CHAT_MODEL") or "deepseek-v4-flash"
        self._base_url = os.getenv("CHAT_BASE_URL") or "https://api.deepseek.com"
        self._api_key = os.getenv("CHAT_MODEL_APIKEY", "")
        if not self._api_key:
            print("[NeKoGenerator] ⚠ 未设置 CHAT_MODEL_APIKEY，生成器不可用")

        self._llm = ChatOpenAI(
            model=self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            temperature=0.3,
        )

    def _build_messages(self, query: str, results: List[RerankResult]):
        """组装带编号参考资料的 System + Human messages"""
        context_parts = []
        for i, r in enumerate(results, start=1):
            meta = r.metadata
            title_parts = [
                meta.get(k) for k in ("Header 1", "Header 2", "Header 3")
                if meta.get(k)
            ]
            source = " > ".join(title_parts) if title_parts else r.chunk_id
            context_parts.append(f"[{i}] 来源: {source}\n{r.text}")

        context = "\n\n".join(context_parts)

        user_prompt = (
            f"【问题】\n{query}\n\n【参考资料】\n{context}\n\n"
            "请根据以上资料回答问题，并标注引用来源。"
        )

        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

    def generate(self, query: str, results: List[RerankResult]) -> str:
        """
        将检索结果作为上下文喂给 LLM，生成最终回答

        Args:
            query: 用户原始问题
            results: reranker 精排后的检索结果

        Returns:
            LLM 生成的回答文本
        """
        if not results:
            return "未检索到相关内容，无法回答。"

        messages = self._build_messages(query, results)
        response = self._llm.invoke(messages)
        return response.content

    def generate_stream(
        self, query: str, results: List[RerankResult]
    ) -> Generator[str, None, None]:
        """
        流式生成：逐个 token 返回，供 SSE 端点使用

        Args:
            query: 用户原始问题
            results: reranker 精排后的检索结果

        Yields:
            逐 token 字符串
        """
        if not results:
            yield "未检索到相关内容，无法回答。"
            return

        messages = self._build_messages(query, results)
        for chunk in self._llm.stream(messages):
            if chunk.content:
                yield chunk.content
