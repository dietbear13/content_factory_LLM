# fact_collector.py
# tools/collectors/fact_collector.py

import logging
from tools.parsers.google_parser import parse_google_results
from tools.parsers.article_parser import get_article_html, parse_article_content

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain


def fetch_articles_from_xmlriver(theme: str, limit: int = 6) -> list[str]:
    """
    Получает заголовки из Google XMLriver и парсит содержимое статей.
    Возвращает список очищенных текстов для дальнейшего анализа.
    """
    headlines = parse_google_results(query=theme, num_results=limit)
    texts = []

    for url in headlines:
        try:
            html = get_article_html(url)
            if html:
                parsed = parse_article_content(html)
                if parsed:
                    texts.append(parsed.strip())
        except Exception as e:
            logging.warning(f"[FactCollector] Ошибка при обработке URL {url}: {e}")

    return texts


class FactCollector:
    """
    Инструмент сбора релевантных фактов по подзаголовку из набора текстов.
    Используется агентами генерации и редактуры.
    """

    def __init__(self, model_name="gpt-4"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.2)

        self.system_prompt = """
Ты — аналитик. Изучи представленные тексты и выдели 3–5 кратких, важных и проверяемых фактов,
относящихся к теме подзаголовка: "{subheading}".

Если фактов мало — покажи только найденные.
Формат ответа — маркированный список.
"""

        self.human_prompt = """
Вот собранные фрагменты из разных источников:

{context}

Подзаголовок: {subheading}
"""

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template(self.human_prompt)
        ])

    def extract_facts(self, full_texts: list[str], subheading: str) -> list[str]:
        """
        Прогоняет собранные тексты через LLM и возвращает отфильтрованные факты.
        (по умолчанию 3-5 шт.)
        """
        combined_text = "\n\n".join(full_texts)
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        result = chain.run({"context": combined_text, "subheading": subheading})
        return [line.strip("-• ").strip() for line in result.strip().split("\n") if line.strip()]

    # --- NEW CODE ---
    def collect_raw_facts(self, full_texts: list[str]) -> list[str]:
        """
        Собирает "сырые" факты/фрагменты напрямую без сильной фильтрации.
        - Просто берём каждый блок текста и разбиваем его на предложения ~<= 2-3 строки,
          чтобы потом FactFilter мог с ними работать.
        """
        logging.info("[FactCollector] Сбор 'сырых' фактов (без жёсткого лимита).")
        raw_facts = []
        for text in full_texts:
            # простое разбиение по абзацам
            blocks = text.split("\n\n")
            for block in blocks:
                block = block.strip()
                if len(block) > 20:  # примитивная проверка, чтобы отбросить пустое
                    raw_facts.append(block)
        return raw_facts
