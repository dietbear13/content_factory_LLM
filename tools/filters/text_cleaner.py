# tools/filters/text_cleaner.py

import spacy
import re

nlp = spacy.load("ru_core_news_sm")

# Расширенный словарь Ильяхова-style
BLACKLIST = {
    "осуществлять": "делать",
    "реализация": "выполнение",
    "функционирование": "работа",
    "в рамках": "во время",
    "обеспечение": "обеспечить",
    "путем": "через",
    "данный": "этот",
    "осуществление": "действие",
    "в целях": "чтобы",
    "мероприятие": "действие",
    "внедрение": "введение",
    "комплексный": "всеобъемлющий",
    "оптимизация": "улучшение",
    "повышение эффективности": "улучшение",
    "вышеназванный": "",
    "принимаемые меры": "действия",
    "иметь возможность": "мочь",
    "недопущение": "предотвращение",
    "направлен на": "для",
    "реализуется": "делается",
    "позволяет": "даёт возможность"
}

# Шаблоны речевых конструкций
PHRASE_PATTERNS = [
    r"\bв рамках [^.,;:]+",
    r"\bс целью [^.,;:]+",
    r"\bимеет возможность [^.,;:]+",
    r"\bнаправлен на [^.,;:]+",
    r"\bреализуется посредством [^.,;:]+",
]


def clean_text(text: str) -> str:
    doc = nlp(text)
    new_tokens = []

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in BLACKLIST:
            replacement = BLACKLIST[lemma]
            new_tokens.append(replacement)
        else:
            new_tokens.append(token.text)

    clean_sentence = " ".join(new_tokens)

    # Удаляем/переписываем речевые конструкции
    for pattern in PHRASE_PATTERNS:
        clean_sentence = re.sub(pattern, "", clean_sentence, flags=re.IGNORECASE)

    # Удалим лишние пробелы
    clean_sentence = re.sub(r"\s{2,}", " ", clean_sentence)
    return clean_sentence.strip()
