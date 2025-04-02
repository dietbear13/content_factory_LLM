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

class ArticleAggregator:
    """
    Собирает все разделы (заголовок + текст) в одну целостную статью,
    добавляет вступление, заключение, содержание и пр.
    """
    def __init__(self, config_path=None):
        if config_path is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(script_dir, "configs", "aggregator_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Параметры модели
        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.2)
        self.top_p = self.config.get("top_p", 1.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.0)
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)

        # Параметры сборки статьи
        self.intro = self.config.get("intro", "Добро пожаловать в нашу статью...")
        self.outro = self.config.get("outro", "Спасибо за внимание!")
        self.use_table_of_contents = self.config.get("use_table_of_contents", True)
        self.toc_title = self.config.get("toc_title", "Содержание статьи")
        self.number_sections = self.config.get("number_sections", True)
        self.transitions = self.config.get("transitions", [])
        self.final_checklist = self.config.get("final_checklist", {})
        self.section_separator = self.config.get("section_separator", "\n\n---\n\n")
        self.heading_format = self.config.get("heading_format", "## {index}. {headline}")
        self.reserve_space_for_images = self.config.get("reserve_space_for_images", False)
        self.images_placeholder_text = self.config.get("images_placeholder_text", "Здесь будет изображение.")

        # Создаём LLM
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty
        )

        # Prompt
        self.system_message_template = """
Ты — агрегатор контента. Тебе даны готовые разделы статьи в виде пар (заголовок, текст).
Задача: собрать из них единую статью со вступлением и заключением.
Нужно ли включать содержание: {use_table_of_contents}.
Заголовок содержания: {toc_title}.
Нумеровать ли секции: {number_sections}.

Вот вступление (intro): "{intro}"
Вот заключение (outro): "{outro}"

Разделы нужно разделять этим разделителем:
{section_separator}

Формат заголовков: {heading_format}

Переходы между разделами (можно использовать по желанию):
{transitions_list}

final_checklist:
  - include_disclaimer = {include_disclaimer}
  - disclaimer_text = "{disclaimer_text}"
  - check_coherence = {check_coherence}
  - add_author_signature = {add_author_signature}

reserve_space_for_images = {reserve_space_for_images_flag}
images_placeholder_text = "{images_placeholder_text}"

Собери логичную и целостную статью. Учитывай все настройки.
        """

        self.human_message_template = """
Ниже перечислены секции статьи (в формате JSON). Каждая секция имеет "headline" и "content".

{sections_json}

Сформируй финальный текст статьи целиком.
Не изменяй содержимое "content" радикально, только адаптируй переходы и логическую последовательность.
Если нужно, добавь содержание (только если use_table_of_contents = true).

Верни итоговую статью на русском языке.
        """

        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_message_template),
            HumanMessagePromptTemplate.from_template(self.human_message_template)
        ])

    def run(self, sections):
        """
        Принимает список словарей вида [{"headline": str, "content": str}, ...]
        Возвращает единую статью (строку).
        """
        logging.info("[ArticleAggregator] Агрегация всех разделов в одну статью.")

        # Подготовим JSON-строку для секций
        # (В реальности можно упрощённо передавать, но JSON удобно для LLM)
        import json
        sections_json = json.dumps(sections, ensure_ascii=False, indent=2)

        # Разбираем final_checklist
        include_disclaimer = self.final_checklist.get("include_disclaimer", False)
        disclaimer_text = self.final_checklist.get("disclaimer_text", "")
        check_coherence = self.final_checklist.get("check_coherence", True)
        add_author_signature = self.final_checklist.get("add_author_signature", False)

        # Список transitions
        transitions_list = "\n".join(self.transitions) if self.transitions else "нет"

        reserve_space_for_images_flag = str(self.reserve_space_for_images).lower()

        chain_input = {
            "use_table_of_contents": str(self.use_table_of_contents).lower(),
            "toc_title": self.toc_title,
            "number_sections": str(self.number_sections).lower(),
            "intro": self.intro,
            "outro": self.outro,
            "section_separator": self.section_separator,
            "heading_format": self.heading_format,
            "transitions_list": transitions_list,

            "include_disclaimer": str(include_disclaimer).lower(),
            "disclaimer_text": disclaimer_text,
            "check_coherence": str(check_coherence).lower(),
            "add_author_signature": str(add_author_signature).lower(),

            "reserve_space_for_images_flag": reserve_space_for_images_flag,
            "images_placeholder_text": self.images_placeholder_text,

            "sections_json": sections_json
        }

        chain = LLMChain(llm=self.llm, prompt=self.chat_prompt)
        final_article = chain.run(chain_input)

        return final_article
