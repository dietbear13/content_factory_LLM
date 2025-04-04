# test_parser.py

import os
import logging
from dotenv import load_dotenv

from tools.parsers.google_parser import parse_google_results
from tools.parsers.article_parser import get_article_html, parse_article_content

# Настройка логгирования
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Загрузка переменных среды
load_dotenv()

def test_parsers(query="искусственный интеллект", num_results=5):
    logging.info(f"🔍 Запрос к XMLriver: '{query}' (результатов: {num_results})")

    try:
        results = parse_google_results(query, limit=num_results)
    except Exception as e:
        logging.exception("❌ Исключение при вызове parse_google_results()")
        return

    if not results:
        logging.warning("❌ Ничего не получено от XMLriver. Проверь ключи или соединение.")
        return

    print("\n🔗 Результаты из XMLriver:\n")
    for i, item in enumerate(results, 1):
        print(f"{i}. {item['title']}\n   🔗 {item['url']}")

    print("\n📰 Пробуем загрузить и распарсить первую ссылку из списка...")

    first_url = results[0]['url']
    logging.info(f"🌐 Загружаем HTML по ссылке: {first_url}")

    html = get_article_html(first_url)
    if not html:
        logging.error("❌ Не удалось загрузить HTML. Возможно, сайт заблокировал запрос или вернул ошибку.")
        return

    logging.info("✅ HTML успешно загружен, начинаем парсинг содержимого...")

    content = parse_article_content(html)
    if not content:
        logging.warning("⚠️ Содержимое не было извлечено (возможно, необычная HTML-структура).")
    else:
        print("\n📄 Извлечённый контент:\n")
        print(content)


if __name__ == "__main__":
    test_parsers()
