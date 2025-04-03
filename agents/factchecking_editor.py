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


class FactCheckingEditor:
    """
    Проверяет текст на фактологические ошибки и переформулирует спорные участки.
    Сохраняет нейтральность, стиль и структуру без комментариев или оценочных суждений.
    """

    def __init__(self, config_path=None):
        if config_path is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(script_dir, "configs", "factcheck_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.2)
        self.top_p = self.config.get("top_p", 1.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.0)
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)

        self.strictness_level = self.config.get("strictness_level", 5)
        self.checklist = self.config.get("checklist", [])
        self.rewrite_uncertain = self.config.get("rewrite_uncertain", True)

        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty
        )

        checklist_items = self.checklist or [
            "Даты и числовые данные",
            "Термины и определения",
            "Статистика и факты",
            "Причинно-следственные связи",
            "Ссылки на источники"
        ]
        self.checklist_block = "\n".join(f"- {item}" for item in checklist_items)

        self.system_message_template = """
Ты — профессиональный фактчекинг-редактор.

🧠 Задача:
Проверь предоставленный текст и перепиши спорные, неподтверждённые или преувеличенные фразы.

📌 Как работать:
- Не добавляй пояснений или комментариев.
- Не пиши «всё корректно».
- Верни только отредактированный текст — без заголовков, форматирования, лишних вводных.
- Соблюдай стиль и структуру оригинала.
- Переформулируй нейтрально, особенно если данные сомнительны.

🔍 Строгость проверки: {strictness_level}/10

📋 Проверь по чеклисту:
{checklist_block}
"""

        self.human_message_template = """
Вот текст для редактирования:

"{text_block}"

Проверь факты и переформулируй спорные места.
"""

        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_message_template),
            HumanMessagePromptTemplate.from_template(self.human_message_template)
        ])

    def run(self, text: str) -> str:
        logging.info("[FactCheckingEditor] Запуск фактчекинга и редактуры текста.")

        chain_input = {
            "strictness_level": self.strictness_level,
            "checklist_block": self.checklist_block,
            "text_block": text
        }

        chain = LLMChain(llm=self.llm, prompt=self.chat_prompt)
        return chain.run(chain_input)
