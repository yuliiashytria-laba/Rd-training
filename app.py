"""Рейтинг курсів: Streamlit-дашборд + Gemini-агент.

Дані: rd-training-course-db (PostgreSQL, read-only).
Рейтинг: байєсове середнє зірок з таблиці reviews.
Агент: Gemini пише саммері відгуків і відповідає на питання про рейтинг.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import psycopg
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

APP_TITLE = "🎓 Рейтинг курсів Coursera"
SUMMARY_CACHE = Path(__file__).parent / "summaries.json"
TOP_N = 10


# ---------- Налаштування ----------

def get_setting(name: str, default: str | None = None) -> str | None:
    """Читає значення з .env (локально) або зі Streamlit Secrets (у хмарі)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


DATABASE_URL = get_setting("DATABASE_URL")
GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
GEMINI_MODEL = get_setting("GEMINI_MODEL", "gemini-3.5-flash-lite")


# ---------- Дані з БД ----------

RANKING_SQL = """
WITH r AS (
    SELECT course_url,
           COUNT(*)                          AS n_reviews,
           AVG(stars)::float                 AS avg_stars,
           AVG((stars <= 2)::int)::float     AS pct_low
    FROM reviews
    GROUP BY course_url
),
prior AS (
    -- C = загальне середнє всіх відгуків, m = 25-й перцентиль кількості відгуків
    SELECT (SELECT AVG(stars) FROM reviews)::float                  AS c,
           PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY n_reviews)  AS m
    FROM r
)
SELECT r.course_url,
       TRIM(d.course)                       AS course,
       TRIM(d.specialization_name)          AS specialization,
       r.n_reviews,
       r.avg_stars,
       r.pct_low,
       d.review_score::float                AS spec_score,
       (r.n_reviews * r.avg_stars + prior.m * prior.c)
           / (r.n_reviews + prior.m)        AS bayes_score,
       prior.c                              AS global_avg,
       prior.m                              AS prior_weight
FROM r
JOIN dim_course d USING (course_url)
CROSS JOIN prior
"""

REVIEWS_SQL = """
SELECT stars, review_content
FROM reviews
WHERE course_url = %s AND review_content IS NOT NULL AND review_content <> ''
"""


def require_db() -> None:
    if not DATABASE_URL:
        st.error("Не задано DATABASE_URL. Додайте його у файл .env (див. .env.example).")
        st.stop()


@st.cache_data(ttl=3600, show_spinner="Завантажую рейтинг з бази…")
def load_ranking() -> pd.DataFrame:
    with psycopg.connect(DATABASE_URL) as conn:
        df = pd.read_sql(RANKING_SQL, conn)
    df = df.sort_values(["bayes_score", "n_reviews", "course"], ascending=[False, False, True])
    df["rank"] = range(1, len(df) + 1)
    df["label"] = df["course"] + " — " + df["specialization"].fillna("без спеціалізації")
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_reviews(course_url: str) -> pd.DataFrame:
    with psycopg.connect(DATABASE_URL) as conn:
        return pd.read_sql(REVIEWS_SQL, conn, params=(course_url,))


# ---------- Агент (Gemini) ----------

def gemini_client() -> genai.Client | None:
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


def load_summaries() -> dict:
    if SUMMARY_CACHE.exists():
        try:
            return json.loads(SUMMARY_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_summaries(cache: dict) -> None:
    SUMMARY_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


SUMMARY_PROMPT = """Ти аналітик освітніх курсів. Нижче наведено ВСІ відгуки студентів про курс
"{course}" (спеціалізація: {specialization}). Кожен рядок: кількість зірок (1–5) і текст.

Напиши саммері УКРАЇНСЬКОЮ мовою у форматі JSON з полями:
- "verdict": один рядок — загальне враження студентів;
- "pros": 2–3 короткі пункти — що студенти хвалять найчастіше;
- "cons": 2–3 короткі пункти — на що найчастіше скаржаться (якщо скарг майже немає, так і напиши одним пунктом).

Спирайся тільки на текст відгуків, не вигадуй фактів і не наводи чисел.

Відгуки:
{reviews}
"""


def generate_summary(row: pd.Series) -> dict:
    client = gemini_client()
    if client is None:
        raise RuntimeError("Не задано GEMINI_API_KEY у файлі .env.")
    reviews = load_reviews(row["course_url"])
    reviews_text = "\n".join(
        f"★{int(s)}: {str(t).strip()}" for s, t in zip(reviews["stars"], reviews["review_content"])
    )
    prompt = SUMMARY_PROMPT.format(
        course=row["course"], specialization=row["specialization"], reviews=reviews_text
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
    )
    data = json.loads(response.text)
    return {
        "verdict": str(data.get("verdict", "")).strip(),
        "pros": [str(x) for x in data.get("pros", [])],
        "cons": [str(x) for x in data.get("cons", [])],
        "model": GEMINI_MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def get_summary(row: pd.Series, force: bool = False) -> dict | None:
    """Повертає саммері з кешу або генерує нове й зберігає його."""
    cache = st.session_state.setdefault("summaries", load_summaries())
    key = row["course_url"]
    if key in cache and not force:
        return cache[key]
    try:
        cache[key] = generate_summary(row)
    except Exception as exc:  # мережа, ліміти, некоректний JSON
        st.error(f"Не вдалося згенерувати саммері для «{row['course']}»: {exc}")
        return None
    save_summaries(cache)
    return cache[key]


QA_PROMPT = """Ти аналітик-агент, який відповідає на питання про рейтинг онлайн-курсів.
Відповідай УКРАЇНСЬКОЮ, коротко й по суті. Використовуй ТІЛЬКИ дані нижче.
Якщо даних недостатньо для відповіді, прямо скажи про це.

Як рахується рейтинг: курси впорядковані за байєсовим середнім зірок
(bayes_score = (n × avg + m × C) / (n + m), C = {c:.4f} — загальне середнє, m = {m:.0f}).
rank 1 — найкращий курс, rank {total} — найнижчий. Усі курси мають високі оцінки, тому
«найгірші» означає «найнижчі серед добрих».

Таблиця рейтингу (CSV):
{table}

Готові саммері відгуків (є не для всіх курсів):
{summaries}

Питання користувача: {question}
"""


def answer_question(question: str, df: pd.DataFrame) -> str:
    client = gemini_client()
    if client is None:
        raise RuntimeError("Не задано GEMINI_API_KEY у файлі .env.")
    table = df[
        ["rank", "course", "specialization", "bayes_score", "avg_stars", "n_reviews", "pct_low", "spec_score"]
    ].round(4).to_csv(index=False)
    cache = st.session_state.setdefault("summaries", load_summaries())
    names = dict(zip(df["course_url"], df["label"]))
    summaries = "\n".join(
        f"- {names.get(url, url)}: {s['verdict']} | Хвалять: {'; '.join(s['pros'])} | Скарги: {'; '.join(s['cons'])}"
        for url, s in cache.items()
    ) or "(ще немає)"
    prompt = QA_PROMPT.format(
        c=df["global_avg"].iloc[0], m=df["prior_weight"].iloc[0], total=len(df),
        table=table, summaries=summaries, question=question,
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt, config=types.GenerateContentConfig(temperature=0.2)
    )
    return response.text


# ---------- Елементи інтерфейсу ----------

def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def ranking_table(df: pd.DataFrame) -> None:
    view = pd.DataFrame({
        "Місце": df["rank"],
        "Курс": df["course"],
        "Спеціалізація": df["specialization"],
        "Байєсова оцінка": df["bayes_score"].round(4),
        "Середнє ★": df["avg_stars"].round(4),
        "Відгуків": df["n_reviews"],
        "Частка 1–2★": (df["pct_low"] * 100).round(2).astype(str) + "%",
        "Рейтинг спеціалізації": df["spec_score"],
    })
    st.dataframe(view, hide_index=True, width="stretch")


def show_summary(summary: dict) -> None:
    st.markdown(f"**Висновок:** {summary['verdict']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**👍 Що хвалять**")
        st.markdown("\n".join(f"- {p}" for p in summary["pros"]) or "—")
    with col2:
        st.markdown("**👎 На що скаржаться**")
        st.markdown("\n".join(f"- {c}" for c in summary["cons"]) or "—")
    st.caption(f"Згенеровано моделлю {summary.get('model', '?')} · {summary.get('generated_at', '')}")


def course_block(row: pd.Series, total: int, key_prefix: str) -> None:
    cache = st.session_state.setdefault("summaries", load_summaries())
    title = f"#{row['rank']} · {row['course']} — {row['specialization']}"
    with st.expander(title, expanded=False):
        st.caption(
            f"Байєсова оцінка {row['bayes_score']:.4f} · середнє {row['avg_stars']:.4f}★ · "
            f"{row['n_reviews']} відгуків · 1–2★: {fmt_pct(row['pct_low'])} · "
            f"рейтинг спеціалізації {row['spec_score']}"
        )
        summary = cache.get(row["course_url"])
        if summary:
            show_summary(summary)
            if st.button("🔄 Оновити саммері", key=f"{key_prefix}_refresh_{row['course_url']}"):
                if get_summary(row, force=True):
                    st.rerun()
        elif st.button("✨ Згенерувати саммері", key=f"{key_prefix}_gen_{row['course_url']}"):
            if get_summary(row):
                st.rerun()


def top_list_tab(subset: pd.DataFrame, total: int, key_prefix: str) -> None:
    ranking_table(subset)
    cache = st.session_state.setdefault("summaries", load_summaries())
    missing = [r for _, r in subset.iterrows() if r["course_url"] not in cache]
    if missing and st.button(f"✨ Згенерувати відсутні саммері ({len(missing)})", key=f"{key_prefix}_all"):
        progress = st.progress(0.0, text="Агент читає відгуки…")
        for i, row in enumerate(missing, start=1):
            get_summary(row)
            progress.progress(i / len(missing), text=f"Готово {i} з {len(missing)}")
        st.rerun()
    st.markdown("#### Саммері відгуків")
    for _, row in subset.iterrows():
        course_block(row, total, key_prefix)


def scatter_chart(df: pd.DataFrame, selected_url: str | None) -> alt.Chart:
    plot = df.copy()
    top_urls = set(df.head(TOP_N)["course_url"])
    bottom_urls = set(df.tail(TOP_N)["course_url"])
    plot["Група"] = "Інші курси"
    plot.loc[plot["course_url"].isin(top_urls), "Група"] = f"Топ-{TOP_N}"
    plot.loc[plot["course_url"].isin(bottom_urls), "Група"] = f"Найгірші {TOP_N}"

    tooltip = [
        alt.Tooltip("course:N", title="Курс"),
        alt.Tooltip("specialization:N", title="Спеціалізація"),
        alt.Tooltip("rank:Q", title="Місце"),
        alt.Tooltip("avg_stars:Q", title="Середнє ★", format=".4f"),
        alt.Tooltip("bayes_score:Q", title="Байєсова оцінка", format=".4f"),
        alt.Tooltip("n_reviews:Q", title="Відгуків"),
    ]
    points = alt.Chart(plot).mark_circle(size=70, opacity=0.75).encode(
        x=alt.X("n_reviews:Q", title="Кількість відгуків"),
        y=alt.Y("avg_stars:Q", title="Середня оцінка, ★", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "Група:N",
            scale=alt.Scale(
                domain=[f"Топ-{TOP_N}", f"Найгірші {TOP_N}", "Інші курси"],
                range=["#2e9d5b", "#d9534f", "#9aa5b1"],
            ),
            legend=alt.Legend(orient="bottom", title=None),
        ),
        tooltip=tooltip,
    )
    avg_line = alt.Chart(pd.DataFrame({"y": [df["global_avg"].iloc[0]]})).mark_rule(
        strokeDash=[4, 4], color="#6c757d"
    ).encode(y="y:Q")
    layers = [points, avg_line]
    if selected_url:
        sel = plot[plot["course_url"] == selected_url]
        layers.append(
            alt.Chart(sel).mark_point(size=300, shape="diamond", filled=True, color="#f0ad4e",
                                      stroke="black", strokeWidth=1).encode(
                x="n_reviews:Q", y="avg_stars:Q", tooltip=tooltip
            )
        )
    return alt.layer(*layers).properties(height=420)


EXAMPLE_QUESTIONS = [
    "Який курс з Python має найвищий рейтинг?",
    "Які спеціалізації найчастіше трапляються в топ-10?",
    "Чим відрізняються курси з топу від найгірших?",
]


def qa_section(df: pd.DataFrame) -> None:
    st.subheader("🤖 Запитайте агента")
    st.caption("Агент відповідає на основі таблиці рейтингу та вже згенерованих саммері відгуків.")
    ask = None
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, width="stretch"):
            ask = q  # приклад одразу надсилається агенту
    with st.form("qa_form"):
        typed = st.text_input("Ваше питання", placeholder="Наприклад: які курси з фінансів у топі?")
        if st.form_submit_button("Запитати") and typed.strip():
            ask = typed.strip()
    question = ask
    if question:
        with st.spinner("Агент аналізує рейтинг…"):
            try:
                st.session_state["qa_answer"] = (question, answer_question(question, df))
            except Exception as exc:
                st.error(f"Агент не зміг відповісти: {exc}")
    if "qa_answer" in st.session_state:
        q, a = st.session_state["qa_answer"]
        st.markdown(f"**Питання:** {q}")
        st.info(a)


# ---------- Сторінка ----------

def main() -> None:
    st.set_page_config(page_title="Рейтинг курсів", page_icon="🎓", layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Рейтинг курсів за відгуками студентів (rd-training-course-db) · "
        f"саммері та відповіді: Gemini ({GEMINI_MODEL})"
    )
    require_db()
    if not GEMINI_API_KEY:
        st.warning("Не задано GEMINI_API_KEY — рейтинг працює, але агент (саммері та відповіді) вимкнений.")

    df = load_ranking()
    total = len(df)
    selected_url = st.session_state.get("selected_course_url")

    tab_overview, tab_top, tab_bottom, tab_search = st.tabs(
        ["📊 Огляд", f"🏆 Топ-{TOP_N}", f"📉 {TOP_N} найгірших", "🔎 Пошук курсу"]
    )

    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Курсів з відгуками", f"{total}")
        c2.metric("Загальне середнє", f"{df['global_avg'].iloc[0]:.3f} ★")
        c3.metric("Діапазон оцінок", f"{df['avg_stars'].min():.2f} – {df['avg_stars'].max():.2f} ★")
        st.markdown("#### Кількість відгуків × середня оцінка")
        st.caption(
            "Пунктир — загальне середнє. Байєсове середнє «підтягує» до нього курси з малою кількістю "
            "відгуків, тому вони не потрапляють у топ чи антитоп випадково."
            + (" Ромбом позначено курс, обраний у пошуку." if selected_url else "")
        )
        st.altair_chart(scatter_chart(df, selected_url), width="stretch")
        st.divider()
        qa_section(df)

    with tab_top:
        st.subheader(f"🏆 Топ-{TOP_N} курсів")
        top_list_tab(df.head(TOP_N), total, "top")

    with tab_bottom:
        st.subheader(f"📉 {TOP_N} курсів з найнижчим рейтингом")
        st.info(
            f"Усі курси мають високі оцінки ({df['avg_stars'].min():.2f}–{df['avg_stars'].max():.2f} ★), "
            "тож «найгірші» означає «найнижчі серед добрих»."
        )
        top_list_tab(df.tail(TOP_N).iloc[::-1], total, "bottom")

    with tab_search:
        st.subheader("🔎 Пошук курсу")
        labels = df.sort_values("course")["label"].tolist()
        label_to_url = dict(zip(df["label"], df["course_url"]))
        current = next((l for l, u in label_to_url.items() if u == selected_url), None)
        choice = st.selectbox(
            "Почніть вводити назву курсу",
            labels,
            index=labels.index(current) if current in labels else None,
            placeholder="Наприклад: Python, Finance, Data…",
        )
        new_url = label_to_url.get(choice) if choice else None
        if new_url != selected_url:
            st.session_state["selected_course_url"] = new_url
            st.rerun()
        if new_url:
            row = df[df["course_url"] == new_url].iloc[0]
            st.markdown(f"### #{row['rank']} з {total}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Байєсова оцінка", f"{row['bayes_score']:.4f}")
            m2.metric("Середнє ★", f"{row['avg_stars']:.4f}")
            m3.metric("Відгуків", f"{row['n_reviews']}")
            m4.metric("Частка 1–2★", fmt_pct(row["pct_low"]))
            st.caption(f"Спеціалізація: {row['specialization']} · рейтинг спеціалізації {row['spec_score']}")
            st.caption("Курс позначено ромбом на графіку у вкладці «Огляд».")
            if GEMINI_API_KEY:
                with st.spinner("Агент читає відгуки…"):
                    summary = get_summary(row)
                if summary:
                    show_summary(summary)
                    if st.button("🔄 Оновити саммері", key=f"search_refresh_{new_url}"):
                        if get_summary(row, force=True):
                            st.rerun()


if __name__ == "__main__":
    main()
