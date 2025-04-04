import json
import os
import logging
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import LLMChain
from tools.collectors.fact_collector import FactCollector, fetch_articles_from_xmlriver


class ContentGenerator:
    """
    Генерирует информативный текст по заголовку в заданной теме.
    Встраивает релевантные факты, избегает шаблонов и лишнего.
    """

    def __init__(self, config_path=None):
        config_path = config_path or os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs", "content_config.json"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Модель и параметры
        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.3)
        self.top_p = self.config.get("top_p", 1.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.0)
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)

        # Стиль и настройки генерации
        self.default_length = self.config.get("default_length", 400)
        self.style = self.config.get("style", "информативный")
        self.tone = self.config.get("tone", "нейтральный")
        self.criteria = self.config.get("criteria", {})
        self.use_citations = self.config.get("use_citations", False)
        self.citations_style = self.config.get("citations_style", "APA")
        self.use_fact_tool = self.config.get("use_fact_tool", True)

        self.fact_collector = FactCollector(model_name=self.model_name)

        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty
        )

        self.criteria_block = self._build_criteria_block()
        self.chat_prompt = self._build_prompt()

    def _build_criteria_block(self) -> str:
        lines = []
        if self.criteria.get("use_examples"):
            lines.append("- Приводи примеры.")
        if self.criteria.get("use_numerical_data"):
            lines.append("- Используй числовые данные.")
        if self.criteria.get("min_paragraphs") or self.criteria.get("max_paragraphs"):
            lines.append(f"- От {self.criteria.get('min_paragraphs', 2)} до {self.criteria.get('max_paragraphs', 4)} абзацев.")
        return "\n".join(lines) if lines else "- Нет дополнительных критериев."

    def _build_prompt(self):
        system_template = """
Ты — опытный автор фанат методов написания Максима Ильяхова «Пиши, сокращай», создающий информативные тексты на русском языке.
Стиль: {style}. Тон: {tone}.

📌 Пример стиля:
Вот пример текста, в стиле которого стоит писать. Не копируй содержание — следуй структуре, стилистике и формулировкам:

{example_text}

📌 Пиши только по существу:
- Без вступлений, общих фраз и переходов между разделами.
- Не используй клише и шаблонные фразы.
- Не повторяйся.

📌 Структура:
- 3–4 абзаца по 100–150 слов.
- Каждый абзац должен раскрывать отдельный аспект подзаголовка.

📌 Критерии:
{criteria_block}

📌 Работа с фактами:
- Используй эти факты, если они даны:
{relevant_facts}
- Если {use_citations} = true — вставляй ссылки в стиле {citations_style}.
- Если фактов нет — пиши по общим принципам.
"""
        human_template = """
Напиши развёрнутый контент (3–4 абзаца) по заголовку:
"{headline}"

В контексте общей темы:
"{global_theme}"

Объём: ~{default_length} слов.
"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def _prepare_facts(self, headline: str) -> list[str]:
        if not self.use_fact_tool:
            return []

        logging.info("[ContentGenerator] Получение фактов через XMLriver и LLM...")
        theme_for_search = headline.split(":")[0] if ":" in headline else headline
        articles = fetch_articles_from_xmlriver(theme_for_search, limit=6)
        return self.fact_collector.extract_facts(articles, subheading=headline)

    def _build_chain_input(self, headline: str, global_theme: str, example_text: str, facts: list[str]) -> dict:
        facts_text = "\n".join(f"- {fact}" for fact in facts) if facts else "Нет доступных фактов."

        return {
            "style": self.style,
            "tone": self.tone,
            "use_citations": str(self.use_citations).lower(),
            "citations_style": self.citations_style,
            "headline": headline,
            "global_theme": global_theme,
            "default_length": self.default_length,
            "criteria_block": self.criteria_block,
            "relevant_facts": facts_text,
            "example_text": example_text.strip()
        }

    def run(self, headline: str, global_theme: str, example_text: str = "") -> str:
        logging.info(f"[ContentGenerator] Генерация текста: «{headline}» (в теме: {global_theme})")
        facts = self._prepare_facts(headline)
        return self._run_chain(headline, global_theme, example_text, facts)

    def run_with_facts(self, headline: str, global_theme: str, example_text: str, filtered_facts: list[str]) -> str:
        logging.info(f"[ContentGenerator] Генерация текста с заранее отфильтрованными фактами: '{headline}'")
        return self._run_chain(headline, global_theme, example_text, filtered_facts)

    def _run_chain(self, headline: str, global_theme: str, example_text: str, facts: list[str]) -> str:
        chain_input = self._build_chain_input(headline, global_theme, example_text, facts)
        chain = LLMChain(llm=self.llm, prompt=self.chat_prompt)
        return chain.run(chain_input)
