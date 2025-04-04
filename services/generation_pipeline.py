# services/generation_pipeline.py

import logging

from agents.content_generator import ContentGenerator
from agents.factchecking_editor import FactCheckingEditor
from agents.style_editor import StyleEditor
from agents.article_aggregator import ArticleAggregator
from agents.fact_compressor import FactFilter
from tools.collectors.fact_collector import fetch_articles_from_xmlriver, FactCollector


def generate_article(theme: str, edited_headlines: list[str]) -> str:
    logging.info("[Pipeline] Запуск генерации статьи")

    # 1. Получаем статьи и сырые факты
    articles = fetch_articles_from_xmlriver(theme, limit=6)
    collector = FactCollector()
    raw_facts = collector.collect_raw_facts(articles)

    # 2. Фильтруем и распределяем факты по заголовкам
    fact_filter = FactFilter()
    filtered_facts_dict = fact_filter.run(raw_facts, edited_headlines)

    # 3. Инициализация агентов
    cg = ContentGenerator()
    fce = FactCheckingEditor()
    se = StyleEditor()

    content_list = []

    # 4. Генерация контента для каждого заголовка
    for headline in edited_headlines:
        facts = filtered_facts_dict.get(headline, [])

        try:
            raw = cg.run_with_facts(
                headline=headline,
                global_theme=theme,
                example_text="",
                filtered_facts=facts
            )
            checked = fce.run(raw)
            polished = se.run(checked)
            content_list.append({"headline": headline, "content": polished})
        except Exception as e:
            logging.error(f"[Pipeline] Ошибка генерации блока '{headline}': {e}")
            content_list.append({"headline": headline, "content": f"Ошибка генерации: {e}"})

    # 5. Финальная сборка
    aggregator = ArticleAggregator()
    try:
        final_article = aggregator.run(content_list, theme=theme)
    except Exception as e:
        logging.error(f"[Pipeline] Ошибка при сборке статьи: {e}")
        final_article = "Не удалось собрать статью."

    return final_article, filtered_facts_dict
