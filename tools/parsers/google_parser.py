import os
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()  # Загружаем переменные из .env


def parse_google_results(query: str, limit: int = 6) -> list[dict]:
    """
    Делает запрос к XMLriver API и возвращает список словарей:
    { "title": заголовок, "url": ссылка }

    :param query: Поисковый запрос
    :param limit: Сколько первых результатов отдать
    :return: список результатов
    """
    user = os.getenv("XMLRIVER_USER")
    key = os.getenv("XMLRIVER_KEY")

    if not user or not key:
        raise ValueError("Не заданы XMLRIVER_USER или XMLRIVER_KEY в .env")

    url = f"https://xmlriver.com/search/xml?user={user}&key={key}&query={query}"

    try:
        response = requests.get(url)
        print(response)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        results = []
        for item in root.findall(".//result"):

            title = item.findtext("title", "").strip()
            link = item.findtext("url", "").strip()
            print(link)

            if title and link:
                results.append({
                    "title": title,
                    "url": link
                })

            if len(results) >= limit:
                break

        return results

    except Exception as e:
        print(f"[XMLriver] Ошибка при запросе: {e}")
        return []
