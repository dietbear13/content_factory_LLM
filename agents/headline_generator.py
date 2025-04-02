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


class HeadlineGenerator:
    """
    Генерирует список заголовков (подтем) для статьи на заданную тему.
    Использует конфигурацию из headline_config.json.
    """

    def __init__(self, config_path=None):
        """
        При инициализации читаем JSON-конфиг и настраиваем "агент".
        """
        if config_path is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(script_dir, "configs", "headline_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Извлекаем ключевые поля из JSON
        self.base_prompt = self.config.get("base_prompt", "")
        self.max_headlines = self.config.get("max_headlines", 10)

        # Дополнительные поля, которые можем использовать в prompt:
        self.style = self.config.get("style", "creative")
        self.detail_level = self.config.get("detail_level", "advanced")
        # self.knowledge_base = self.config.get("knowledge_base", [])
        self.criteria = self.config.get("criteria", {})
        self.additional_guidelines = self.config.get("additional_guidelines", [])

        # Инициализация реального ChatOpenAI (использует OPENAI_API_KEY из env)
        self.llm = ChatOpenAI(
            temperature=0.1,
            model_name="gpt-4"  # или другой вариант, например "gpt-4"
        )

        # Подготовим шаблон (System + Human) для цепочки
        # System-message: задаёт контекст и правила поведения модели
        # Human-message: формируем конкретный запрос
        self.system_message_template = """
Ты — профессиональный автор и эксперт.
Твой стиль: {style}.
Уровень детализации: {detail_level}.

Учитывай следующее:
1) Избегай повторений, если {avoid_repetition} = true.
2) Старайся давать уникальный взгляд на тему, если {unique_perspective} = true.
3) Регистр языка: {language_register}.
4) Минимум слов в заголовке: {minimum_words_per_headline}.

Вот дополнительные указания:
{guidelines}

        """

# FIXME вынес из промта

        # Также у тебя есть knowledge_base (список документов):
        # {kb_info}

        self.human_message_template = """
Тема статьи: "{theme}"

Необходимо создать {num_headlines} заголовков.

Важные детали, если нужно учесть:
 - Использовать подзаголовки: {use_subheadings}

Ответь на русском языке. Верни только список заголовков, разделённых переносами строки.
        """

        # Создаём шаблон для Chain
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_message_template),
            HumanMessagePromptTemplate.from_template(self.human_message_template)
        ])

    def run(self, theme: str, num_headlines: int = 5):
        """
        Сгенерировать список заголовков на основе темы.
        num_headlines ограничивается max_headlines из конфига.
        Возвращает список строк.
        """
        # Уточняем число заголовков
        num_headlines = min(num_headlines, self.max_headlines)

        # Подготовим данные для заполнения prompt
        # Из self.criteria вытащим нужные поля
        avoid_repetition = self.criteria.get("avoid_repetition", False)
        unique_perspective = self.criteria.get("unique_perspective", False)
        use_subheadings = self.criteria.get("use_subheadings", False)
        language_register = self.criteria.get("language_register", "neutral")
        min_words = self.criteria.get("minimum_words_per_headline", 3)

        # Дополнительные указания объединим в одну строку
        guidelines = "\n".join(self.additional_guidelines) if self.additional_guidelines else "Нет особых указаний."

        # Сформируем строку для knowledge_base
        # Например, просто перечислим их названия
        # kb_info = "\n".join([f"- {doc['title']} (keywords: {', '.join(doc['keywords'])})"
        #                      for doc in self.knowledge_base])

        logging.info(f"[HeadlineGenerator] Генерация {num_headlines} заголовков для темы '{theme}'")

        # Создаём chain
        chain = LLMChain(
            llm=self.llm,
            prompt=self.chat_prompt
        )

        # Сформируем финальный словарь для format()
        chain_input = {
            "style": self.style,
            "detail_level": self.detail_level,
            "avoid_repetition": str(avoid_repetition).lower(),
            "unique_perspective": str(unique_perspective).lower(),
            "language_register": language_register,
            "minimum_words_per_headline": min_words,
            "guidelines": guidelines,
            # "kb_info": kb_info,

            "theme": theme,
            "num_headlines": num_headlines,
            "use_subheadings": str(use_subheadings).lower()
        }

        # Запускаем chain и получаем текст
        result_text = chain.run(chain_input)

        # Теперь нужно превратить результат в список заголовков
        # Допустим, модель вернёт список строк, разделённых переносами
        headlines = [line.strip() for line in result_text.split("\n") if line.strip()]

        return headlines
