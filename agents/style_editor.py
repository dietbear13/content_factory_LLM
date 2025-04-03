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


class StyleEditor:
    """
    Делает текст плавным, литературным, избавляет от сухих оборотов.
    Не переписывает смысл, а улучшает читаемость, стиль и логику.
    """

    def __init__(self, config_path=None):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if config_path is None:
            config_path = os.path.join(script_dir, "configs", "style_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.3)
        self.top_p = self.config.get("top_p", 1.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.0)
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)

        self.tone = self.config.get("tone", "профессиональный")
        self.preferred_person = self.config.get("preferred_person", "третье лицо")
        self.avoid_jargon = self.config.get("avoid_jargon", True)
        self.use_simplification = self.config.get("use_simplification", True)
        self.additional_rules = self.config.get("additional_rules", [])

        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty
        )

        self.system_message_template = """
Ты — профессиональный литературный редактор. Получаешь текст, который нужно улучшить.

Цель: сделать стиль плавным, выразительным, лёгким для восприятия.

- Тон: {tone}
- Лицо: {preferred_person}
- Упростить сложные конструкции: {use_simplification}
- Избегать жаргона: {avoid_jargon}
- Не менять смысл, но улучшить подачу.
- Не добавляй комментарии. Верни только улучшенный текст.

Дополнительные правила:
{additional_rules}
"""

        self.human_message_template = """
Вот текст, который нужно стилистически улучшить:

"{original_text}"
"""

        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_message_template),
            HumanMessagePromptTemplate.from_template(self.human_message_template)
        ])

    def run(self, text: str) -> str:
        logging.info("[StyleEditor] Стилистическая обработка текста.")

        additional_rules = "\n".join(self.additional_rules) if self.additional_rules else "Нет дополнительных указаний."

        chain_input = {
            "tone": self.tone,
            "preferred_person": self.preferred_person,
            "use_simplification": str(self.use_simplification).lower(),
            "avoid_jargon": str(self.avoid_jargon).lower(),
            "additional_rules": additional_rules,
            "original_text": text
        }

        chain = LLMChain(llm=self.llm, prompt=self.chat_prompt)
        return chain.run(chain_input)
