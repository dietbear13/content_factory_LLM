import json
import os
import logging

# LangChain и OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import LLMChain


class ContentGenerator:
    """
    Генерирует основной контент (развернутый текст) для каждого заголовка.
    Использует конфигурацию из content_config.json.
    """

    def __init__(self, config_path=None):
        if config_path is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(script_dir, "configs", "content_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Извлекаем параметры модели:
        # (model_name, temperature, top_p, presence_penalty, frequency_penalty и т.п.)
        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.1)
        self.top_p = self.config.get("top_p", 1.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.0)
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)

        # Извлекаем поля для промпта
        self.content_prompt = self.config.get("content_prompt", "")
        self.default_length = self.config.get("default_length", 400)
        self.style = self.config.get("style", "informative")
        self.tone = self.config.get("tone", "semi-formal")
        self.use_citations = self.config.get("use_citations", False)
        self.citations_style = self.config.get("citations_style", "APA")
        # self.knowledge_base = self.config.get("knowledge_base", [])
        self.criteria = self.config.get("criteria", {})
        self.intro_phrases = self.config.get("intro_phrases", [])
        self.closing_phrases = self.config.get("closing_phrases", [])
        self.fallback_strategies = self.config.get("fallback_strategies", [])

        # Создаём реальный LLM-объект
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty
        )

        # Подготовим шаблон (System + Human) для цепочки
        self.system_message_template = """
Ты — профессиональный контент-creator.
Твой стиль написания: {style}.
Тон: {tone}.

При работе учитывай следующие критерии:
- Минимум абзацев: {min_paragraphs}
- Максимум абзацев: {max_paragraphs}
- Использовать примеры: {use_examples}
- Использовать числовые данные: {use_numerical_data}

Если нужно, включай ссылки на источники в формате {citations_style}, только если {use_citations} = true.

У тебя есть knowledge_base:
{kb_info}

Дополнительные фразы для введения:
{intro_phrases_list}

Фразы для завершения абзацев или секции:
{closing_phrases_list}

Если информация недоступна или возникает конфликт, следуй fallback-стратегиям:
{fallback_strategies_list}

Работай аккуратно, старайся писать связно и по сути.
        """

        self.human_message_template = """
Напиши развернутый контент для заголовка "{headline}" в контексте глобальной темы "{global_theme}".
Ориентировочная длина текста: ~{default_length} слов.

Ответ на русском языке. Не добавляй лишних заголовков, просто сплошной текст или несколько абзацев.
        """

        # Объединим всё в ChatPromptTemplate
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_message_template),
            HumanMessagePromptTemplate.from_template(self.human_message_template)
        ])

    def run(self, headline: str, global_theme: str):
        """
        Генерирует «сырой» текст по заголовку и общей теме.
        Возвращает строку с текстом.
        """
        logging.info(f"[ContentGenerator] Генерация контента для '{headline}' (Тема: {global_theme})")

        # Извлекаем нужные поля из criteria
        min_paragraphs = self.criteria.get("min_paragraphs", 2)
        max_paragraphs = self.criteria.get("max_paragraphs", 6)
        use_examples = self.criteria.get("use_examples", True)
        use_numerical_data = self.criteria.get("use_numerical_data", True)

        # Готовим строку для knowledge_base
        # kb_info = "\n".join([
        #     f"- {doc['title']} (keywords: {', '.join(doc['keywords'])})"
        #     for doc in self.knowledge_base
        # ])

        # Подготовка списков
        intro_phrases_list = "\n".join(self.intro_phrases) if self.intro_phrases else "Нет введений."
        closing_phrases_list = "\n".join(self.closing_phrases) if self.closing_phrases else "Нет завершающих фраз."
        fallback_strategies_list = "\n".join(self.fallback_strategies) if self.fallback_strategies else "Нет fallback-стратегий."

        # Подготовка входных данных для chain
        chain_input = {
            "style": self.style,
            "tone": self.tone,
            "min_paragraphs": min_paragraphs,
            "max_paragraphs": max_paragraphs,
            "use_examples": str(use_examples).lower(),
            "use_numerical_data": str(use_numerical_data).lower(),
            "use_citations": str(self.use_citations).lower(),
            "citations_style": self.citations_style,
            # "kb_info": kb_info,
            "intro_phrases_list": intro_phrases_list,
            "closing_phrases_list": closing_phrases_list,
            "fallback_strategies_list": fallback_strategies_list,

            "headline": headline,
            "global_theme": global_theme,
            "default_length": self.default_length
        }

        chain = LLMChain(llm=self.llm, prompt=self.chat_prompt)
        result_text = chain.run(chain_input)

        # Возвращаем полученный текст
        return result_text
