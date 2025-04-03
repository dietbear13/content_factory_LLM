# tools/parsers/google_parser.py

import os
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()  # Загружаем переменные из .env

def parse_google_headlines(query: str, num_results: int = 10) -> list:
    """
    Делает запрос к XMLriver API и возвращает заголовки из поисковой выдачи Google.

    :param query: Поисковый запрос (тема статьи)
    :param num_results: Кол-во заголовков
    :return: Список строк — заголовков
    """
    user = os.getenv("XMLRIVER_USER")
    key = os.getenv("XMLRIVER_KEY")

    if not user or not key:
        raise ValueError("Не заданы XMLRIVER_USER или XMLRIVER_KEY в .env")

    url = f"https://xmlriver.com/search/xml?user={user}&key={key}&q={query}&maxresults={num_results}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        titles = []
        for result in root.findall(".//title"):
            title = result.text.strip()
            if title and query.lower() not in title.lower():
                titles.append(title)
        return titles

    except Exception as e:
        print(f"Ошибка при парсинге XMLriver: {e}")
        return []
