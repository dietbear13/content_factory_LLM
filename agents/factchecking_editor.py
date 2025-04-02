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


class FactCheckingEditor:
    """
    Проверяет фактическую точность сгенерированного текста и исправляет ошибки.
    Использует конфигурацию factcheck_config.json.
    """
    def __init__(self, config_path=None):
        if config_path is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(script_dir, "configs", "factcheck_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Извлекаем настройки модели
        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.0)
        self.top_p = self.config.get("top_p", 1.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.0)
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)

        # Другие параметры
        self.factcheck_prompt = self.config.get("factcheck_prompt", "")
        self.strictness_level = self.config.get("strictness_level", 5)
        self.use_external_apis = self.config.get("use_external_apis", False)
        # self.knowledge_base_sources = self.config.get("knowledge_base_sources", [])
        self.fallback_strategies = self.config.get("fallback_strategies", [])
        self.error_handling = self.config.get("error_handling", {})
        self.checklist = self.config.get("checklist", [])

        # Создаём LLM
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty
        )

        # Формируем System+Human шаблон
        self.system_message_template = """
Ты — фактчекинг-редактор. Твоя задача: проверить фактическую корректность текста.
Уровень строгости проверки: {strictness_level}/10.

Чеклист:
{checklist_items}

Fallback-стратегии:
{fallback_strategies_list}

Если встречаешь критические ошибки, смотри на {on_critical_error}.
Если замечаешь мелкие ошибки, смотри на {on_minor_error}.

Обязательно исправляй фактические неточности, если найдёшь.
Если факт не подтверждён, переформулируй в нейтральной форме ("Существует мнение..." и т.д.).
        """

        # Имеется knowledge_base_sources: {knowledge_base_sources}

        self.human_message_template = """
Вот фрагмент текста, который надо проверить и исправить (при необходимости):

"{text_block}"

Внеси необходимые изменения для устранения недостоверной информации. Сохрани общий стиль текста.
Если всё корректно — просто верни текст без изменений.
        """

        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_message_template),
            HumanMessagePromptTemplate.from_template(self.human_message_template)
        ])

    def run(self, text: str):
        """
        Принимает "сырой" текст, проверяет факты, возвращает откорректированный текст.
        """
        logging.info("[FactCheckingEditor] Проверка фактов и исправление неточностей.")

        # Сформируем дополнительные поля
        # knowledge_base_str = ", ".join(self.knowledge_base_sources) if self.knowledge_base_sources else "нет"
        checklist_items = "\n".join([f"- {item}" for item in self.checklist]) if self.checklist else "нет"
        fallback_strategies_list = "\n".join(self.fallback_strategies) if self.fallback_strategies else "нет"

        on_critical_error = self.error_handling.get("on_critical_error", "raise_exception")
        on_minor_error = self.error_handling.get("on_minor_error", "insert_comment")

        chain_input = {
            "strictness_level": self.strictness_level,
            # "knowledge_base_sources": knowledge_base_str,
            "checklist_items": checklist_items,
            "fallback_strategies_list": fallback_strategies_list,
            "on_critical_error": on_critical_error,
            "on_minor_error": on_minor_error,

            "text_block": text
        }

        chain = LLMChain(llm=self.llm, prompt=self.chat_prompt)
        result_text = chain.run(chain_input)

        return result_text
