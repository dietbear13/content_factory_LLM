# tools/parsers/article_parser.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from fake_useragent import UserAgent

HEADERS = {
    "User-Agent": UserAgent().chrome,
    "Accept-Language": "ru,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp",
    "Connection": "keep-alive"
}


def get_article_html(url: str, timeout: int = 10) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"[article_parser] Ошибка при запросе {url}: {e}")
        return ""


def parse_article_content(html: str) -> str:
    """
    Извлекает важное содержимое из HTML-страницы:
    абзацы, списки, цитаты, таблицы — в логичном порядке.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")

    if not body:
        return ""

    content_parts = []

    for tag in body.find_all(["p", "li", "blockquote", "table"], recursive=True):
        if tag.name == "p":
            text = tag.get_text(strip=True)
            if text:
                content_parts.append(text)

        elif tag.name == "li":
            text = tag.get_text(strip=True)
            if text:
                content_parts.append(f"• {text}")

        elif tag.name == "blockquote":
            text = tag.get_text(strip=True)
            if text:
                content_parts.append(f"Цитата: {text}")

        elif tag.name == "table":
            rows = tag.find_all("tr")
            for row in rows:
                cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
                if cols:
                    content_parts.append(" | ".join(cols))

    return "\n\n".join(content_parts)
