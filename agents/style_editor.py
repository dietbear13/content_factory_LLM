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

class StyleEditor:
    """
    Приводит текст к единому стилю (стилистика, орфография, пунктуация).
    Использует конфигурацию style_config.json.
    """
    def __init__(self, config_path=None):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if config_path is None:
            config_path = os.path.join(script_dir, "configs", "style_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Модель и её параметры
        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.0)
        self.top_p = self.config.get("top_p", 1.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.0)
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)

        # Остальные настройки
        self.style_prompt = self.config.get("style_prompt", "")
        self.tone = self.config.get("tone", "formal")
        self.preferred_person = self.config.get("preferred_person", "third_person")
        self.max_line_length = self.config.get("max_line_length", 120)
        self.additional_rules = self.config.get("additional_rules", [])
        self.grammar_check = self.config.get("grammar_check", {})
        self.use_simplification = self.config.get("use_simplification", False)
        self.avoid_jargon = self.config.get("avoid_jargon", False)

        # LLM
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty
        )

        # Подготовим Prompt
        self.system_message_template = """
Ты — профессиональный литературный редактор.
Тебе нужно привести текст к следующему стилю:
- Тон: {tone}
- Использовать лицо: {preferred_person}
- Макс длина строки: {max_line_length}
- Проверка грамматики: {grammar_options}
- Избегать жаргона: {avoid_jargon_flag}
- Упрощать текст (если слишком сложный): {use_simplification_flag}

Вот дополнительные правила:
{additional_rules_list}
        """

        self.human_message_template = """
Ниже приведён текст, который нужно отредактировать согласно указанным правилам.

"{original_text}"

Верни отредактированный текст, сохраняя смысл.
        """

        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_message_template),
            HumanMessagePromptTemplate.from_template(self.human_message_template)
        ])

    def run(self, text: str):
        """
        Принимает текст после фактчекинга, возвращает "отредактированный" вариант.
        """
        logging.info(f"[StyleEditor] Применение стилистической правки (тон: {self.tone})")

        grammar_options = (
            f"language={self.grammar_check.get('language', 'ru')}, "
            f"spell_check={self.grammar_check.get('enable_spell_check', True)}, "
            f"punctuation_check={self.grammar_check.get('enable_punctuation_check', True)}"
        )

        use_simplification_flag = str(self.use_simplification).lower()
        avoid_jargon_flag = str(self.avoid_jargon).lower()

        additional_rules_list = "\n".join(self.additional_rules) if self.additional_rules else "Нет дополнительных правил."

        chain_input = {
            "tone": self.tone,
            "preferred_person": self.preferred_person,
            "max_line_length": self.max_line_length,
            "grammar_options": grammar_options,
            "avoid_jargon_flag": avoid_jargon_flag,
            "use_simplification_flag": use_simplification_flag,
            "additional_rules_list": additional_rules_list,

            "original_text": text
        }

        chain = LLMChain(llm=self.llm, prompt=self.chat_prompt)
        result_text = chain.run(chain_input)

        return result_text
