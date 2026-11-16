"""Streamlit interface for the AI Student Assistant."""
import re

import streamlit as st

import config
import llm
from embeddings import get_collection, get_embedder
from rag import answer_question

st.set_page_config(
    page_title="AI Student Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

COPY = {
    "English": {
        "tagline": "Your study companion, grounded in trusted academic sources.",
        "intro": "Ask a question and get a clear answer based on the EducationQ knowledge base.",
        "question": "What would you like to learn?",
        "placeholder": "For example: What does price elasticity of demand measure?",
        "ask": "Find an answer",
        "settings": "Study settings",
        "subject": "Subject area",
        "all": "All subjects",
        "sources_count": "Sources to retrieve",
        "threshold": "Relevance threshold",
        "threshold_help": "A lower value requires a closer match.",
        "library": "Knowledge base",
        "questions": "questions",
        "language": "Interface language",
        "answer": "Answer",
        "retrieval_notice": "No LLM key is configured. Showing the closest stored answer.",
        "closest_question": "Closest question",
        "worked_solution": "Worked solution",
        "elasticity_explanation": "For a demand curve, the signed point elasticity depends on the quantity. There is no single numerical value unless a price or quantity is specified.",
        "elasticity_example": "At quantity 200, price is 30 and the signed elasticity is -1.5.",
        "sources": "Sources",
        "searching": "Searching your study materials…",
        "empty": "Type a question first, then choose Find an answer.",
        "error": "The assistant could not complete the search. Check the app configuration and try again.",
        "error_details": "Technical details",
        "source_question": "Question",
        "source_answer": "Correct answer",
        "source_match": "Match",
        "footer": "Answers are generated from the available knowledge base. Check the cited sources for context.",
        "setup_error": "The multilingual knowledge base is missing or out of date. Run step4_build_vector_db.py, then restart the app.",
        "llm_ready": "Answer generation is ready",
        "llm_missing": "Answer generation is not configured; the closest retrieved answer will be shown.",
        "llm_status": "Answer generation",
    },
    "Français": {
        "tagline": "Votre compagnon d’étude, appuyé sur des sources académiques.",
        "intro": "Posez une question et obtenez une réponse fondée sur la base EducationQ.",
        "question": "Que souhaitez-vous apprendre ?",
        "placeholder": "Exemple : Que mesure l’élasticité-prix de la demande ?",
        "ask": "Rechercher une réponse",
        "settings": "Paramètres d’étude",
        "subject": "Matière",
        "all": "Toutes les matières",
        "sources_count": "Sources à rechercher",
        "threshold": "Seuil de pertinence",
        "threshold_help": "Une valeur plus faible exige une correspondance plus proche.",
        "library": "Base de connaissances",
        "questions": "questions",
        "language": "Langue de l’interface",
        "answer": "Réponse",
        "retrieval_notice": "Aucune clé LLM n’est configurée. Voici la réponse enregistrée la plus proche.",
        "closest_question": "Question la plus proche",
        "worked_solution": "Résolution détaillée",
        "elasticity_explanation": "Pour une courbe de demande, l’élasticité ponctuelle signée dépend de la quantité. Il n’existe pas de valeur numérique unique sans prix ou quantité précisé.",
        "elasticity_example": "Pour une quantité de 200, le prix est 30 et l’élasticité signée vaut -1,5.",
        "sources": "Sources",
        "searching": "Recherche dans les ressources…",
        "empty": "Saisissez une question, puis sélectionnez Rechercher une réponse.",
        "error": "La recherche a échoué. Vérifiez la configuration et réessayez.",
        "error_details": "Détails techniques",
        "source_question": "Question",
        "source_answer": "Bonne réponse",
        "source_match": "Correspondance",
        "footer": "Les réponses sont générées à partir de la base disponible. Consultez les sources citées.",
        "setup_error": "La base multilingue est absente ou obsolète. Exécutez step4_build_vector_db.py puis relancez l’application.",
        "llm_ready": "La génération de réponses est prête",
        "llm_missing": "La génération n’est pas configurée ; la réponse la plus proche sera affichée.",
        "llm_status": "Génération de réponses",
    },
    "العربية": {
        "tagline": "رفيقك في الدراسة، بإجابات مستندة إلى مصادر أكاديمية.",
        "intro": "اطرح سؤالك واحصل على إجابة واضحة من قاعدة بيانات EducationQ.",
        "question": "ما الموضوع الذي تريد فهمه؟",
        "placeholder": "مثال: ماذا تقيس مرونة الطلب السعرية؟",
        "ask": "ابحث عن إجابة",
        "settings": "إعدادات الدراسة",
        "subject": "المادة الدراسية",
        "all": "كل المواد",
        "sources_count": "عدد المصادر المسترجعة",
        "threshold": "حد الصلة",
        "threshold_help": "القيمة الأقل تتطلب تطابقاً أقوى.",
        "library": "قاعدة المعرفة",
        "questions": "سؤالاً",
        "language": "لغة الواجهة",
        "answer": "الإجابة",
        "retrieval_notice": "مفتاح LLM غير مهيأ؛ تُعرض أقرب إجابة محفوظة في قاعدة البيانات.",
        "closest_question": "أقرب سؤال",
        "worked_solution": "الحل خطوةً خطوة",
        "elasticity_explanation": "تعتمد مرونة الطلب النقطية بإشارتها على الكمية. لا توجد قيمة عددية واحدة ما لم يُحدد السعر أو مقدار الكمية.",
        "elasticity_example": "عند الكمية 200 يكون السعر 30 والمرونة بإشارتها تساوي ‎-1.5.",
        "sources": "المصادر",
        "searching": "جارٍ البحث في المواد الدراسية…",
        "empty": "اكتب سؤالك أولاً، ثم اضغط على ابحث عن إجابة.",
        "error": "تعذّر إكمال البحث. تحقق من إعدادات التطبيق ثم أعد المحاولة.",
        "error_details": "تفاصيل تقنية",
        "source_question": "السؤال",
        "source_answer": "الإجابة الصحيحة",
        "source_match": "درجة التطابق",
        "footer": "الإجابات مبنية على قاعدة المعرفة المتاحة. راجع المصادر المذكورة للاطلاع على السياق.",
        "setup_error": "قاعدة البيانات متعددة اللغات غير موجودة أو قديمة. نفّذ step4_build_vector_db.py ثم أعد تشغيل التطبيق.",
        "llm_ready": "توليد الإجابات جاهز",
        "llm_missing": "توليد الإجابات غير مهيأ؛ ستظهر أقرب إجابة مسترجعة.",
        "llm_status": "توليد الإجابات",
    },
}


def repair_known_math(text: str) -> str:
    """Close a missing inline-LaTeX delimiter in the EducationQ elasticity item."""
    return re.sub(
        r"(\$p=\\frac\{500-x\}\{10\})(?=\s*(?:[,.;?!]|$))",
        r"\1$",
        text,
        count=1,
    )


def is_linear_demand_elasticity(source_question: str) -> bool:
    normalized = source_question.lower().replace(" ", "")
    return "elastic" in normalized and "500-x" in normalized and "frac" in normalized and "10" in normalized

st.markdown(
    """
    <style>
    :root { --ink:#172b4d; --muted:#62738a; --accent:#2867d8; --line:#e6ecf3; }
    .stApp { background: linear-gradient(180deg,#f5f8fd 0%,#ffffff 410px); color:var(--ink); }
    [data-testid="stAppViewContainer"] { font-family:'DM Sans','Noto Sans Arabic',sans-serif; }
    [data-testid="stSidebar"] { background:#f8faff; border-right:1px solid var(--line); }
    .block-container { max-width:1120px; padding-top:2.2rem; padding-bottom:2.5rem; }
    .hero { padding:1.35rem 1.6rem 1.1rem; border:1px solid #e4ebf5; border-radius:22px; background:rgba(255,255,255,.92); box-shadow:0 10px 30px rgba(27,54,93,.055); margin-bottom:1.35rem; }
    .hero-kicker { color:#2867d8; font-size:.78rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; margin-bottom:.35rem; }
    .hero h1 { color:#172b4d; font-size:2.1rem; letter-spacing:-.04em; margin:0 0 .35rem; }
    .hero p { color:#62738a; font-size:1.04rem; margin:0; }
    div[data-testid="stForm"] { border:1px solid var(--line); background:white; border-radius:18px; padding:1.15rem 1.25rem; box-shadow:0 8px 24px rgba(27,54,93,.045); }
    div[data-testid="stForm"] textarea { border-radius:12px; }
    .stButton button[kind="primary"], button[kind="primaryFormSubmit"] { border-radius:11px; font-weight:650; min-height:2.8rem; }
    .source-note { color:#62738a; font-size:.86rem; margin-top:.85rem; }
    [dir="rtl"] .hero { text-align:right; }
    @media (max-width: 700px) { .block-container { padding:1rem 1rem 2rem; } .hero { padding:1.1rem; } .hero h1 { font-size:1.65rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def warm_up():
    get_embedder()
    return get_collection()


@st.cache_data(show_spinner=False)
def list_categories() -> list[str]:
    collection = warm_up()
    if collection.count() == 0:
        return []
    metas = collection.get(include=["metadatas"]).get("metadatas") or []
    return sorted({m["category"] for m in metas if m and m.get("category")})


try:
    collection = warm_up()
except Exception:
    st.error(COPY["English"]["setup_error"])
    st.stop()

with st.sidebar:
    st.markdown("### 🎓 AI Student Assistant")
    language = st.selectbox("Interface language", list(COPY), index=0)
    t = COPY[language]
    st.divider()
    st.subheader(t["settings"])
    category_names = list_categories()
    subject_labels = [t["all"]] + [c.title() for c in category_names]
    subject = st.selectbox(t["subject"], subject_labels)
    top_k = st.slider(t["sources_count"], 1, 10, min(max(config.TOP_K, 1), 10))
    max_distance = st.slider(
        t["threshold"], 0.2, 1.0, min(max(config.MAX_DISTANCE, 0.2), 1.0), 0.05,
        help=t["threshold_help"],
    )
    st.divider()
    st.caption(f"**{t['library']}:** {collection.count():,} {t['questions']}")
    st.caption(f"**Embeddings:** {config.EMBEDDING_MODEL.split('/')[-1]}")
    if llm.is_configured():
        st.success(t["llm_ready"], icon="✅")
    else:
        st.info(t["llm_missing"], icon="ℹ️")
    st.caption(f"Dataset: [EducationQ]({config.DATASET_URL}) · CC-BY-4.0")

direction_attr = 'dir="rtl"' if language == "العربية" else 'dir="ltr"'
if language == "العربية":
    st.markdown(
        '<style>[data-testid="stAppViewContainer"], [data-testid="stSidebar"], '
        '.block-container, [data-testid="stForm"] {direction:rtl;text-align:right;} '
        '[data-testid="stSidebar"] {border-right:0;border-left:1px solid #e6ecf3;}</style>',
        unsafe_allow_html=True,
    )
st.markdown(
    f'<section class="hero" {direction_attr}><div class="hero-kicker">EducationQ · RAG Study Assistant</div>'
    f'<h1>🎓 AI Student Assistant</h1><p>{t["tagline"]}</p></section>',
    unsafe_allow_html=True,
)
st.markdown(f'<div {direction_attr} style="color:#62738a;margin:0 0 1rem">{t["intro"]}</div>', unsafe_allow_html=True)

with st.form("ask", clear_on_submit=False):
    question = st.text_area(t["question"], height=112, placeholder=t["placeholder"], label_visibility="visible")
    submitted = st.form_submit_button(t["ask"], type="primary", use_container_width=True)

if submitted:
    if not question or not question.strip():
        st.warning(t["empty"])
    else:
        category = None if subject == t["all"] else subject.lower()
        with st.spinner(t["searching"]):
            try:
                result = answer_question(question, top_k, category, max_distance)
            except Exception as exc:
                st.error(t["error"])
                with st.expander(t["error_details"]):
                    st.code(f"{type(exc).__name__}: {exc}")
                st.stop()

        st.markdown(f"### {t['answer']}")
        if result.found:
            if result.used_llm:
                with st.container(border=True):
                    st.markdown(result.answer)
            elif result.sources:
                closest = result.sources[0]
                st.info(t["retrieval_notice"])
                with st.container(border=True):
                    st.markdown(f"**{t['closest_question']}:**")
                    st.markdown(repair_known_math(closest.question))
                    st.markdown(f"**{t['source_answer']}:** {closest.answer}")
                    if is_linear_demand_elasticity(closest.question):
                        st.markdown(f"#### {t['worked_solution']}")
                        st.latex(r"p=\frac{500-x}{10}\quad\Longrightarrow\quad x=500-10p")
                        st.latex(r"\frac{dx}{dp}=-10")
                        st.latex(r"E_p=\frac{dx}{dp}\cdot\frac{p}{x}=-10\frac{p}{x}=1-\frac{500}{x}")
                        st.markdown(t["elasticity_explanation"])
                        st.latex(r"x=200,\quad p=30,\quad E_p=-1.5")
                        st.caption(t["elasticity_example"])
            else:
                with st.container(border=True):
                    st.markdown(result.answer)
        else:
            st.info(result.answer)

        if result.sources:
            st.markdown(f"### {t['sources']} · {len(result.sources)}")
            for source in result.sources:
                label = f"{source.rank:02d}  ·  {source.category.title()}  ·  {source.original_source}  ·  {t['source_match']} {source.similarity:.0%}"
                with st.expander(label):
                    st.markdown(f"**{t['source_question']}:**")
                    st.write(source.question)
                    st.markdown(f"**{t['source_answer']}:**")
                    st.write(source.answer)
                    st.caption(f"EducationQ · {source.id}")

st.markdown(f'<p class="source-note" {direction_attr}>{t["footer"]}</p>', unsafe_allow_html=True)
