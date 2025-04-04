# file: agents/fact_compressor.py

import os
import json
import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain


class FactFilter:
    """
    Агент, который берёт "сырые" факты по теме,
    фильтрует, группирует по подзаголовкам и переформулирует.
    """

    def __init__(self, config_path=None):
        """
        Загружаем настройки (model_name, temperature и т.п.) из JSON-конфига,
        чтобы было аналогично другим агентам в проекте.
        """
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if config_path is None:
            config_path = os.path.join(script_dir, "configs", "fact_compressor_config.json")

        # Пример содержимого fact_compressor_config.json:
        # {
        #   "model_name": "gpt-4",
        #   "temperature": 0.2
        # }
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model_name = self.config.get("model_name", "gpt-4")
        self.temperature = self.config.get("temperature", 0.2)

        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature
        )

        self.system_prompt = """
Ты — помощник, который умеет фильтровать и сжимать факты.
Вот твоя задача:
1) Получить список "сырых фактов" (возможно, пересекающихся) и список подзаголовков.
2) Распределить эти факты по подзаголовкам (если факт явно не подходит — игнорируй).
3) Удалить дубли и сомнительные утверждения, переформулировать, чтобы избежать копирования исходных фраз.
4) Итог: для каждого подзаголовка дай список коротких, чётко сформулированных фактов.
Не выдумывай новые факты. Не добавляй комментарии.
Формат ответа:
{ "подзаголовок1": ["Факт1", "Факт2"], "подзаголовок2": [...] }
"""

        self.human_prompt = """
Сырые факты:
{raw_facts}

Подзаголовки:
{headlines}
"""

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template(self.human_prompt)
        ])

    def run(self, raw_facts: list[str], headlines: list[str]) -> dict:
        """
        :param raw_facts: список строк (фактов), собранных FactCollector'ом
        :param headlines: список подзаголовков (H2)
        :return: dict, где ключ = подзаголовок, значение = список фактов
        """
        logging.info("[FactFilter] Запуск фильтра и группировки фактов.")

        # Собираем все факты в одну строку
        combined_facts = "\n".join(f"- {f}" for f in raw_facts)
        # Собираем подзаголовки
        combined_headlines = "\n".join(f"- {h}" for h in headlines)

        chain_input = {
            "raw_facts": combined_facts,
            "headlines": combined_headlines
        }

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        response = chain.run(chain_input).strip()

        # Предполагается, что ответ будет в формате JSON.
        # Пробуем распарсить:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
            else:
                logging.warning("[FactFilter] Ответ не является словарём JSON.")
                return {}
        except Exception as e:
            logging.warning(f"[FactFilter] Не удалось распарсить JSON: {e}")
            return {}
