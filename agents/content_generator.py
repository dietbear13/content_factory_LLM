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


class ContentGenerator:
    """
    Генерирует информативный текст по заголовку в заданной теме.
    Упор на факты, избегание шаблонов, структурированный результат.
    """

    def __init__(self, config_path=None):
        if config_path is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(script_dir, "configs", "content_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.3)
        self.top_p = self.config.get("top_p", 1.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.0)
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)

        self.default_length = self.config.get("default_length", 400)
        self.style = self.config.get("style", "информативный")
        self.tone = self.config.get("tone", "нейтральный")

        self.criteria = self.config.get("criteria", {})
        self.use_citations = self.config.get("use_citations", False)
        self.citations_style = self.config.get("citations_style", "APA")

        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty
        )

        self.system_message_template = """
Ты — опытный автор, создающий информативные тексты на русском языке.
Стиль: {style}. Тон: {tone}.
Следуй принципам «Пиши, сокращай».

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
- Если {use_citations} = true — вставляй ссылки в стиле {citations_style}.
- Если информация не подтверждена — формулируй нейтрально.
- Если нет данных — дай обзор на основе общих принципов (fallback).
"""

        self.human_message_template = """
Напиши развёрнутый контент (3–4 абзаца) по заголовку:
"{headline}"

В контексте общей темы:
"{global_theme}"

Объём: ~{default_length} слов.
"""

        # Формируем текст критериев
        criteria_lines = []
        if self.criteria.get("use_examples"):
            criteria_lines.append("- Приводи примеры.")
        if self.criteria.get("use_numerical_data"):
            criteria_lines.append("- Используй числовые данные.")
        if self.criteria.get("min_paragraphs") or self.criteria.get("max_paragraphs"):
            criteria_lines.append(f"- От {self.criteria.get('min_paragraphs', 2)} до {self.criteria.get('max_paragraphs', 4)} абзацев.")

        self.criteria_block = "\n".join(criteria_lines) if criteria_lines else "- Нет дополнительных критериев."

        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_message_template),
            HumanMessagePromptTemplate.from_template(self.human_message_template)
        ])

    def run(self, headline: str, global_theme: str) -> str:
        logging.info(f"[ContentGenerator] Генерация текста: «{headline}» (в теме: {global_theme})")

        chain_input = {
            "style": self.style,
            "tone": self.tone,
            "use_citations": str(self.use_citations).lower(),
            "citations_style": self.citations_style,
            "headline": headline,
            "global_theme": global_theme,
            "default_length": self.default_length,
            "criteria_block": self.criteria_block
        }

        chain = LLMChain(llm=self.llm, prompt=self.chat_prompt)
        return chain.run(chain_input)
