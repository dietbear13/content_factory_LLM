# agents/article_aggregator.py

import logging
from tools.filters.text_cleaner import clean_text


class ArticleAggregator:
    """
    Собирает финальную статью из заголовков и текстов.
    Без использования LLM. Применяет фильтр текста (анти-клише) на последнем этапе.
    """

    def __init__(self):
        pass

    def run(self, sections: list[dict], theme: str = "") -> str:
        """
        :param sections: список блоков вида {"headline": str, "content": str}
        :param theme: тема статьи (h1)
        :return: финальный текст статьи
        """
        logging.info("[ArticleAggregator] Сборка финальной статьи в Python")

        lines = []

        if theme:
            lines.append(f"# {theme}\n")

        # Вставим абзац-вступление (первый параграф первого контента)
        if sections and sections[0].get("content"):
            intro = sections[0]["content"].strip().split("\n")[0]
            cleaned_intro = clean_text(intro)
            lines.append(cleaned_intro + "\n")

        for i, block in enumerate(sections, 1):
            headline = block.get("headline", "").strip()
            content = block.get("content", "").strip()

            if not headline or not content:
                continue

            lines.append(f"## {headline}\n")

            paragraphs = content.split("\n\n")  # <-- абзацы
            for para in paragraphs:
                cleaned_para = clean_text(para.strip())
                if cleaned_para:
                    lines.append(cleaned_para + "\n")

        final_text = "\n".join(lines).strip()
        return final_text
