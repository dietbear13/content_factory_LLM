# app.py

from flask import Flask, render_template, request, redirect, url_for, session
import logging

from agents.content_generator import ContentGenerator
from agents.factchecking_editor import FactCheckingEditor
from agents.style_editor import StyleEditor
from agents.article_aggregator import ArticleAggregator
from agents.headline_generator import run as parse_theme_input

app = Flask(__name__)
app.secret_key = "SUPER_SECRET_KEY_CHANGE_IT"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate_headlines", methods=["POST"])
def generate_headlines():
    raw_input = request.form.get("theme_input", "").strip()
    if not raw_input:
        return "Тема не может быть пустой!", 400

    try:
        num_headings = int(request.form.get("num_headings", "5"))
    except ValueError:
        num_headings = 5

    theme, headlines = parse_theme_input(raw_input, num_headings)

    session["theme"] = theme
    session["headlines"] = headlines

    return redirect(url_for("edit_headlines"))


@app.route("/edit_headlines", methods=["GET"])
def edit_headlines():
    theme = session.get("theme", "")
    headlines = session.get("headlines", [])
    return render_template("headlines.html", theme=theme, headlines=headlines)


@app.route("/finalize_headlines", methods=["POST"])
def finalize_headlines():
    theme = session.get("theme", "Тема не найдена")
    edited_headlines = [h.strip() for h in request.form.getlist("headline") if h.strip()]
    session["headlines"] = edited_headlines

    # --- NEW CODE: собираем сырые факты и фильтруем ---
    from tools.collectors.fact_collector import fetch_articles_from_xmlriver, FactCollector
    from agents.fact_compressor import FactFilter

    # 1) Получаем из Google тексты по всей теме (до двоеточия) — а не по каждому H2
    articles = fetch_articles_from_xmlriver(theme, limit=6)
    collector = FactCollector()
    raw_facts = collector.collect_raw_facts(articles)  # много "сырых" текстовых фрагментов

    # 2) Запускаем фильтратор:
    fact_filter = FactFilter()
    filtered_facts_dict = fact_filter.run(raw_facts, edited_headlines)
    # Сохраним в сессии, если захотим потом использовать повторно:
    session["filtered_facts_by_h2"] = filtered_facts_dict

    content_list = []

    cg = ContentGenerator()
    fce = FactCheckingEditor()
    se = StyleEditor()

    # 3) Для каждого заголовка используем только факты из filtered_facts_dict
    for headline in edited_headlines:
        # Берём факты, относящиеся к этому заголовку
        facts_for_this_headline = filtered_facts_dict.get(headline, [])

        try:
            # вместо старого cg.run(...) вызываем новую функцию
            raw = cg.run_with_facts(
                headline=headline,
                global_theme=theme,
                example_text="",
                filtered_facts=facts_for_this_headline
            )
            checked = fce.run(raw)
            polished = se.run(checked)
            content_list.append({"headline": headline, "content": polished})
        except Exception as e:
            logging.error(f"Ошибка в блоке '{headline}': {e}")
            content_list.append({"headline": headline, "content": f"Ошибка генерации: {e}"})

    aa = ArticleAggregator()
    try:
        final_article = aa.run(content_list, theme=theme)
    except Exception as e:
        logging.error(f"Ошибка агрегации: {e}")
        final_article = "Не удалось собрать статью."

    session["final_article"] = final_article
    return redirect(url_for("result"))


@app.route("/result", methods=["GET"])
def result():
    article = session.get("final_article", "Статья не найдена.")
    return render_template("result.html", article=article)


if __name__ == "__main__":
    app.run(debug=True)
