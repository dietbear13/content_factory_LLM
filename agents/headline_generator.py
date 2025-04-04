# agents/headline_generator.py

"""
Разбор пользовательского ввода вида:
"Что такое кинезиотейпы: История; Эффективность; Где купить"
- До двоеточия — это h1 (тема)
- После — список h2 (подтемы)
"""
from typing import Tuple, List, Any


def parse_theme_and_headlines(raw_input: str) -> tuple[str, list[str]]:
    """
    Принимает строку от пользователя и возвращает тему и список заголовков.
    """
    if ":" in raw_input:
        theme, raw_h2s = raw_input.split(":", 1)
        theme = theme.strip()
        h2_list = [h.strip() for h in raw_h2s.split(";") if h.strip()]
    else:
        theme = raw_input.strip()
        h2_list = []

    return theme, h2_list


def run(theme_input: str, num_headlines: int = 5) -> tuple[str, list[Any] | list | list[str]]:
    """
    Главная точка входа — возвращает заголовки, если они есть.
    Если пользователь не ввёл заголовки, список будет пустой (чтобы вызвать парсинг через Google).
    """
    theme, h2_list = parse_theme_and_headlines(theme_input)

    if not h2_list:
        # Здесь можно подключить парсер из tools/parsers/google_parser.py
        from tools.parsers.google_parser import parse_google_results
        try:
            h2_list = parse_google_results(query=theme, num_results=num_headlines)
        except Exception as e:
            print(f"Ошибка при вызове парсера: {e}")
            h2_list = []

    return theme, h2_list
