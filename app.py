from flask import Flask, render_template, request, redirect, url_for, session
import logging

# Импорт агентов
from agents.headline_generator import HeadlineGenerator
from agents.content_generator import ContentGenerator
from agents.factchecking_editor import FactCheckingEditor
from agents.style_editor import StyleEditor
from agents.article_aggregator import ArticleAggregator

app = Flask(__name__)
# Для сессии (если нужно), установим секретный ключ. В реальном проекте храните его в .env:
app.secret_key =  "SUPER_SECRET_KEY_CHANGE_IT"

# Настраиваем логгирование
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


@app.route("/", methods=["GET"])
def index():
    """
    Главная страница: форма ввода темы статьи и параметров
    """
    return render_template("index.html")


@app.route("/generate_headlines", methods=["POST"])
def generate_headlines():
    """
    Обрабатывает форму: получает тему, генерирует список заголовков
    и перенаправляет на страницу редактирования заголовков.
    """
    theme = request.form.get("theme_input", "").strip()
    if not theme:
        return "Тема не может быть пустой!", 400

    # Примерные доп. параметры — глубина или кол-во заголовков (необязательно)
    num_headings = request.form.get("num_headings", "5")
    try:
        num_headings = int(num_headings)
    except ValueError:
        num_headings = 5  # значение по умолчанию

    # Генерация заголовков через HeadlineGenerator
    hg = HeadlineGenerator()
    try:
        headlines = hg.run(theme=theme, num_headlines=num_headings)
    except Exception as e:
        logging.error(f"Ошибка при генерации заголовков: {e}")
        return "Произошла ошибка при генерации заголовков", 500

    # Сохраняем заголовки в сессии, чтобы отредактировать или подтвердить
    session["theme"] = theme
    session["headlines"] = headlines

    return redirect(url_for("edit_headlines"))


@app.route("/edit_headlines", methods=["GET"])
def edit_headlines():
    """
    Страница, где пользователь может отредактировать предложенные заголовки.
    """
    headlines = session.get("headlines", [])
    theme = session.get("theme", "")
    return render_template("headlines.html", theme=theme, headlines=headlines)


@app.route("/finalize_headlines", methods=["POST"])
def finalize_headlines():
    """
    Получает отредактированные заголовки, запускает цепочку:
    - Для каждого заголовка генерирует контент
    - Фактчекинг
    - Стилистическая правка
    - Сборка финальной статьи
    Затем перенаправляет на страницу /result
    """
    # Считываем отредактированные заголовки из формы
    edited_headlines = request.form.getlist("headline")
    edited_headlines = [h.strip() for h in edited_headlines if h.strip()]

    # Сохраняем в сессии
    session["headlines"] = edited_headlines

    # Далее запускаем основную "генерацию" контента
    theme = session.get("theme", "Тема не найдена")
    content_list = []

    cg = ContentGenerator()
    fce = FactCheckingEditor()
    se = StyleEditor()

    # Генерируем текст отдельно для каждого заголовка
    for h in edited_headlines:
        try:
            raw_text = cg.run(headline=h, global_theme=theme)
            fact_checked_text = fce.run(raw_text)
            styled_text = se.run(fact_checked_text)
            content_list.append({
                "headline": h,
                "content": styled_text
            })
        except Exception as e:
            logging.error(f"Ошибка при обработке заголовка '{h}': {e}")
            content_list.append({
                "headline": h,
                "content": f"Ошибка при генерации текста: {e}"
            })

    # Агрегируем в одну статью
    aa = ArticleAggregator()
    try:
        final_article = aa.run(content_list)
    except Exception as e:
        logging.error(f"Ошибка при агрегации финальной статьи: {e}")
        final_article = "Не удалось собрать статью из-за ошибки."

    # Сохраняем финальную статью в сессии
    session["final_article"] = final_article

    return redirect(url_for("result"))


@app.route("/result", methods=["GET"])
def result():
    """
    Страница с финальной статьёй
    """
    article = session.get("final_article", "Статья не найдена.")
    return render_template("result.html", article=article)


if __name__ == "__main__":
    # Запуск Flask-приложения
    app.run(debug=True)
