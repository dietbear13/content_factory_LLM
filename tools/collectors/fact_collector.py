import requests
from tools.parsers.google_parser import parse_google_headlines
from trafilatura import extract, fetch_url
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain


def fetch_articles_from_xmlriver(theme: str, limit: int = 6) -> list[str]:
    headlines = parse_google_headlines(query=theme, num_results=limit)
    texts = []
    for h in headlines:
        url = fetch_url(h)
        if url:
            content = extract(url)
            if content:
                texts.append(content.strip())
    return texts


class FactCollector:
    """
    Используется агентами для получения фактов по подзаголовку на основе поисковых данных.
    """

    def __init__(self, model_name="gpt-4"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.2)

        self.system_prompt = """
Ты — ассистент-аналитик. Твоя задача — извлекать важные факты из текстов.
Дай 3–5 конкретных, проверяемых фактов, относящихся к теме: "{subheading}".
Факты должны быть краткими, точными и полезными для статьи.

Если фактов недостаточно — напиши только те, что есть.
"""

        self.human_prompt = """
Вот статьи (объединённый текст):
{context}

Тема подзаголовка: {subheading}
"""

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template(self.human_prompt)
        ])

    def extract_facts(self, full_texts: list[str], subheading: str) -> list[str]:
        combined_text = "\n\n".join(full_texts)
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        result = chain.run({"context": combined_text, "subheading": subheading})
        return [line.strip("-• ").strip() for line in result.strip().split("\n") if line.strip()]
