"""Interactive Week 4 vectorizer walkthrough. Run with: streamlit run <this file>."""
import base64
import html
import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.set_page_config(page_title="Text vectorization · Big Data", page_icon="◈", layout="wide")

EXAMPLES = {
    "Transport headlines": (
        "Bus fares rise\n"
        "Train fares rise\n"
        "Bus fares rise as train fares rise\n"
        "City plans new tram line"
    ),
    "Climate policy": (
        "Climate policy divides council\n"
        "Council debates climate policy\n"
        "Climate protests grow\n"
        "City announces a new tram line"
    ),
    "A word in every text": "Climate\nClimate",
}
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
CSS = """
<style>
:root { --ink:#173c43; --muted:#657a7e; --teal:#176b65; --line:#e1e9e7; --amber:#fff0b9; }
.stApp { background:#f6f8f7; color:var(--ink); }
.stAppHeader { display:none; }
[data-testid="stMainBlockContainer"] { max-width:1440px; padding-top:2.5rem; padding-bottom:2rem; }
[data-testid="stVerticalBlock"] { gap:.7rem; }
h1,h2,h3,p,label { color:var(--ink); }
h3 { font-size:1.13rem!important; font-weight:650!important; padding:.15rem 0!important; }
[data-testid="stCaptionContainer"] p { color:var(--muted); font-size:.85rem; }
[data-testid="stButton"] button, [data-testid="stPopover"] button {
    border:1px solid #dce5e3; border-radius:11px; background:white; color:var(--ink);
    min-height:2rem; padding:.25rem .75rem; box-shadow:none;
}
[data-testid="stButton"] button:hover { border-color:var(--teal); color:var(--teal); background:#edf5f1; }
[data-testid="stButton"] button[kind="primary"], button[kind="primaryFormSubmit"] {
    background:var(--teal); border-color:var(--teal); color:white;
}
[data-testid="stButton"] button[kind="primary"] p, button[kind="primaryFormSubmit"] p {
    color:white;
}
[data-testid="stButton"] button:focus-visible { outline:3px solid #d39a28; outline-offset:2px; }
[data-testid="stTabs"] [role="tablist"] { border-bottom:1px solid var(--line); gap:1.8rem; }
[data-testid="stTabs"] [role="tab"] { padding:.65rem .1rem; color:var(--muted); font-weight:600; }
[data-testid="stTabs"] [aria-selected="true"] { color:var(--teal); }
[role="tablist"] .react-aria-SelectionIndicator { background-color:#176b65!important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; border-color:var(--line); background:white; }
[class*="st-key-headline-"] { background:white; }
[data-testid="stTextArea"] textarea { background:white; color:var(--ink); border-color:var(--line); }
.eyebrow { font-size:11px; font-weight:750; letter-spacing:.16em; color:#65867d; margin-bottom:10px; }
.hero h1 { font-size:clamp(30px,3.6vw,46px); letter-spacing:-.045em; margin:0; line-height:1.13; font-weight:720; }
.hero p { margin:12px 0 10px; font-size:16px; color:var(--muted); }
.summary { display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 12px; }
.summary span { display:inline-flex; align-items:baseline; gap:7px; padding:8px 13px;
    background:white; border:1px solid var(--line); border-radius:10px; font-size:12px; color:var(--muted); }
.summary b { font-size:17px; font-weight:700; color:var(--ink); }
.doc-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
.doc-label { font-size:10px; font-weight:750; letter-spacing:.12em; color:#6a8588; }
.doc-count { font-size:11px; color:#7d8e91; }
.headline { font-size:17px; line-height:1.45; margin:0 0 6px; color:var(--ink); overflow-wrap:anywhere; }
.headline mark { background:#fff0b9; color:#173c43; border-radius:3px; padding:1px 2px; }
.section-kicker { color:#72928a; font-size:10px; font-weight:750; letter-spacing:.14em; margin:7px 0 7px; }
.matrix-wrap { overflow:auto; border:1px solid var(--line); border-radius:13px; background:white; margin:5px 0 10px; }
.dtm { width:100%; border-collapse:separate; border-spacing:0; font-size:13px; text-align:center; }
.dtm th { background:#f0f5f3; padding:15px 11px; color:#71868b; font-weight:550; white-space:nowrap; }
.dtm th.row-label { text-align:left; position:sticky; left:0; z-index:1; min-width:58px; }
.dtm td { min-width:43px; height:57px; padding:8px; border-top:1px solid #eaf0ed;
    font-variant-numeric:tabular-nums; color:var(--ink); }
.dtm .zero { color:#97a7aa; }
.dtm th.selected { background:#176b65; color:white; font-weight:700; }
.dtm td.selected { background:#fff3cd!important; color:#5e4815; font-weight:750; }
.dtm tr.focused th { color:var(--teal); box-shadow:inset 3px 0 var(--teal); }
.dtm td.active { outline:2px solid #c18c28; outline-offset:-3px; border-radius:5px; font-size:18px; }
.legend { display:flex; gap:18px; font-size:11px; color:#7c9195; margin:6px 0 14px; flex-wrap:wrap; }
.legend i { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }
.insight { background:#edf5f1; border:1px solid #d8e7df; border-radius:14px; padding:18px 20px; margin:10px 0; }
.insight h4 { font-size:19px; margin:0 0 7px; letter-spacing:-.02em; color:var(--ink); }
.insight p { font-size:13px; color:#52716c; margin:0; line-height:1.65; }
.insight .big-number { font-size:33px; font-weight:750; color:var(--teal); }
.word-stats { display:flex; flex-wrap:wrap; gap:26px; margin-top:13px; }
.word-stats span { color:#5b7c75; font-size:11px; }
.word-stats b { font-size:17px; display:block; color:#204d46; margin-bottom:2px; }
.note { color:#5f777b; font-size:12px; line-height:1.6; margin:9px 0; }
.formula { font-size:14px; color:#426963; background:#f0f5f3; border-radius:10px; padding:12px 15px; }
.learn-card { padding:20px; margin:0 0 12px; background:white; border:1px solid var(--line); border-radius:14px; }
.learn-card h4 { margin:0 0 6px; font-size:16px; color:var(--ink); }
.learn-card p { margin:0; color:#698085; font-size:13px; line-height:1.6; }
.animation { display:block; width:100%; max-width:850px; margin:0 auto; border:1px solid var(--line); border-radius:15px; }
.footer { border-top:1px solid var(--line); padding-top:15px; margin-top:20px; color:#8aa09d; font-size:11px; letter-spacing:.04em; }
@media(max-width:700px) {
    [data-testid="stMainBlockContainer"] { padding:1.5rem 1rem; }
    .hero p { font-size:14px; }
    .dtm th { padding:12px 9px; }
    .summary { gap:6px; }
}
</style>
"""
st.html(CSS)


def select_word(word, document):
    st.session_state.selected_word = word
    st.session_state.selected_document = document


def change_method(method):
    st.session_state.method = method


def load_example():
    st.session_state.draft_text = EXAMPLES[st.session_state.example]


def apply_texts():
    st.session_state.source_text = st.session_state.draft_text


def toggle_animation():
    st.session_state.animation_playing = not st.session_state.animation_playing


def highlighted_text(text, selected):
    """Use the same token pattern as the vectorizer and escape all user text."""
    pieces, last = [], 0
    for match in TOKEN_PATTERN.finditer(text):
        pieces.append(html.escape(text[last:match.start()]))
        word = html.escape(match.group())
        pieces.append(f"<mark>{word}</mark>" if match.group().lower() == selected else word)
        last = match.end()
    pieces.append(html.escape(text[last:]))
    return "".join(pieces)


def matrix_html(frame, selected, focused, weights=False, label="Document-term matrix"):
    """Keep the original column order, highlighting one word across every row."""
    max_value = max(float(frame.to_numpy().max()), 1e-12)
    headers = ['<th class="row-label" scope="col">Text</th>']
    for term in frame.columns:
        cls = ' class="selected"' if term == selected else ""
        headers.append(f'<th scope="col"{cls}>{html.escape(str(term))}</th>')
    rows = []
    for row_index, (doc_label, row) in enumerate(frame.iterrows()):
        row_class = ' class="focused"' if row_index == focused else ""
        cells = [f'<th class="row-label" scope="row">{html.escape(str(doc_label))}</th>']
        for term, value in row.items():
            classes = []
            if value == 0:
                classes.append("zero")
            if term == selected:
                classes.append("selected")
                if row_index == focused:
                    classes.append("active")
            opacity = 0.04 + 0.15 * float(value) / max_value if value else 0
            text = f"{value:.3f}" if weights else str(int(value))
            cells.append(
                f'<td class="{" ".join(classes)}" data-word="{html.escape(str(term), quote=True)}" '
                f'data-document="{html.escape(str(doc_label), quote=True)}" data-value="{value}" '
                f'style="background:rgba(23,107,101,{opacity:.3f})">{text}</td>'
            )
        rows.append(f"<tr{row_class}>{''.join(cells)}</tr>")
    return (
        '<div class="matrix-wrap"><table class="dtm" aria-label="'
        + html.escape(label, quote=True)
        + '"><thead><tr>' + "".join(headers) + "</tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


@st.cache_data(show_spinner=False)
def animation_image(path, playing):
    data = Path(path).read_bytes()
    if not playing:
        with Image.open(io.BytesIO(data)) as image:
            image.seek(image.n_frames - 1)
            poster = io.BytesIO()
            image.convert("RGB").save(poster, format="PNG")
            data = poster.getvalue()
    mime = "image/gif" if playing else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


for key, value in {
    "source_text": EXAMPLES["Transport headlines"],
    "draft_text": EXAMPLES["Transport headlines"],
    "selected_word": "fares",
    "selected_document": 0,
    "method": "Counts",
    "animation_playing": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

heading, edit = st.columns([5, 1], vertical_alignment="center")
with heading:
    st.html(
        '<div class="hero"><div class="eyebrow">BIG DATA · WEEK 04</div>'
        '<h1>Text vectorization</h1>'
        '<p>Click a word to trace it from a headline to the matrix.</p></div>'
    )
with edit:
    with st.popover("Edit texts", icon=":material/edit:", width="stretch"):
        st.markdown("### Make it your own")
        st.selectbox("Try an example", list(EXAMPLES), key="example", on_change=load_example)
        with st.form("edit-texts"):
            st.text_area("One text per line", key="draft_text", height=180, max_chars=3000)
            st.caption("Use a small collection of short texts, with one text on each line.")
            st.form_submit_button("Update matrix", on_click=apply_texts, type="primary", width="stretch")

sentences = [line.strip() for line in st.session_state.source_text.splitlines() if line.strip()]
if len(sentences) < 2:
    st.info("Add at least two texts using Edit texts to start exploring.")
    st.stop()
if len(sentences) > 12:
    st.info("Use up to twelve short texts so the example stays easy to inspect.")
    st.stop()

cv, tfidf_cv = CountVectorizer(), TfidfVectorizer()
try:
    count_matrix = cv.fit_transform(sentences)
    weight_matrix = tfidf_cv.fit_transform(sentences)
except ValueError as error:
    if "empty vocabulary" not in str(error):
        raise
    st.info("Add words with at least two characters. The default tokenizer skips single letters and punctuation.")
    st.stop()

terms = list(cv.get_feature_names_out())
labels = [chr(65 + i) for i in range(len(sentences))]
counts = pd.DataFrame(count_matrix.toarray(), index=labels, columns=terms)
weights = pd.DataFrame(weight_matrix.toarray(), index=labels, columns=tfidf_cv.get_feature_names_out())
document_frequency = counts.gt(0).sum(axis=0)
idf = pd.Series(tfidf_cv.idf_, index=tfidf_cv.get_feature_names_out())
zero_share = 1 - count_matrix.count_nonzero() / (counts.shape[0] * counts.shape[1])

if st.session_state.selected_word not in terms:
    st.session_state.selected_word = terms[0]
st.session_state.selected_document = min(st.session_state.selected_document, len(sentences) - 1)
selected = st.session_state.selected_word
focused = st.session_state.selected_document
safe_word = html.escape(selected)
st.html(
    '<div class="summary">'
    f'<span><b>{len(sentences)}</b> texts · rows</span>'
    f'<span><b>{len(terms)}</b> words · columns</span>'
    f'<span><b>{zero_share:.0%}</b> of cells are zero</span>'
    '</div>'
)

explore, compare, learn = st.tabs(["Explore the matrix", "Compare methods", "How it works"])

with explore:
    left, right = st.columns([1, 1.75], gap="large")
    with left:
        st.html('<div class="section-kicker">01 / THE TEXTS</div>')
        st.markdown("### Pick a word")
        st.caption("Click a word below any headline. Its column lights up on the right.")
        for i, sentence in enumerate(sentences):
            occurrences = list(TOKEN_PATTERN.finditer(sentence))
            with st.container(border=True, key=f"headline-{i}", gap="xsmall"):
                st.html(
                    f'<div class="doc-top"><span class="doc-label">TEXT {labels[i]}</span>'
                    f'<span class="doc-count">{len(occurrences)} tokens</span></div>'
                    f'<p class="headline">{highlighted_text(sentence, selected)}</p>'
                )
                with st.container(horizontal=True, gap=5):
                    for j, match in enumerate(occurrences[:24]):
                        token = match.group().lower()
                        st.button(
                            match.group(),
                            key=f"word-{i}-{j}",
                            type="primary" if token == selected else "secondary",
                            on_click=select_word,
                            args=(token, i),
                            help=f'Inspect "{token}" in text {labels[i]}',
                            width="content",
                        )
                if len(occurrences) > 24:
                    st.caption("Word buttons show the first 24 tokens. The matrix uses the whole text.")
                if not occurrences:
                    st.caption("No tokens of at least two characters in this text.")
    with right:
        st.html('<div class="section-kicker">02 / THE DOCUMENT-TERM MATRIX</div>')
        title_col, mode_col = st.columns([1.3, 1], vertical_alignment="center")
        with title_col:
            st.markdown("### A number for every word")
        with mode_col:
            with st.container(horizontal=True, gap=5, horizontal_alignment="right"):
                for mode in ["Counts", "TF-IDF"]:
                    st.button(mode, key=f"mode-{mode}", type="primary" if st.session_state.method == mode else "secondary",
                              on_click=change_method, args=(mode,), width="content")
        weighted = st.session_state.method == "TF-IDF"
        current = weights if weighted else counts
        st.html(matrix_html(current, selected, focused, weights=weighted))
        st.html(
            '<div class="legend"><span><i style="background:#fff0b9"></i>Selected word</span>'
            '<span><i style="background:#d9e9e1"></i>Higher value</span>'
            '<span>0 = word absent</span></div>'
        )
        value = float(current.loc[labels[focused], selected])
        shown = f"{value:.3f}" if weighted else str(int(value))
        occurrences = int(counts.loc[labels[focused], selected])
        description = (
            f'Its TF-IDF weight is {shown}. It occurs {occurrences} '
            f'{"time" if occurrences == 1 else "times"} in this text.'
            if weighted else
            f'It occurs {occurrences} {"time" if occurrences == 1 else "times"} in this text.'
        )
        if occurrences == 0:
            description = "This word is absent from this text, so its value is 0."
        st.html(
            f'<div class="insight"><div class="section-kicker">FOLLOWING THE WORD</div>'
            f'<h4>“{safe_word}” in text {labels[focused]}</h4>'
            f'<div class="big-number">{shown}</div><p>{description}</p>'
            '<div class="word-stats">'
            f'<span><b>{int(document_frequency[selected])} of {len(sentences)}</b>texts contain it</span>'
            f'<span><b>{int(counts[selected].sum())}</b>occurrences overall</span>'
            f'<span><b>{float(idf[selected]):.2f}</b>IDF weight</span>'
            '</div></div>'
        )
        st.caption(
            "Counts are the number of occurrences in each text." if not weighted else
            "TF-IDF combines counts with inverse document frequency, then normalises each row. "
            "Weights are shown to three decimals; they are not percentages."
        )
        with st.expander("Why do we count zero cells?"):
            st.write(
                f"{count_matrix.count_nonzero()} of {counts.size} cells contain a non-zero count; "
                f"the remaining {counts.size - count_matrix.count_nonzero()} are zero. "
                "A cell containing 2 still counts as one non-zero cell. "
                "For large collections, sparse storage avoids storing all those zeros."
            )
        with st.expander("What does the default tokenizer do?"):
            st.write(
                "It lowercases the texts and extracts tokens of at least two letters, digits, or underscores. "
                "Punctuation separates tokens. The examples here keep stopwords and use single words."
            )

with compare:
    st.markdown("### Same texts. Different numbers.")
    st.write(f'Follow “{selected}” in both tables. The words and column order stay the same.')
    count_col, weight_col = st.columns(2, gap="large")
    with count_col:
        st.html('<div class="section-kicker">COUNT VECTORIZER</div>')
        st.markdown("#### How often does it occur?")
        st.html(matrix_html(counts, selected, focused, label="Word counts"))
    with weight_col:
        st.html('<div class="section-kicker">TF-IDF VECTORIZER</div>')
        st.markdown("#### How is it weighted?")
        st.html(matrix_html(weights, selected, focused, weights=True, label="TF-IDF weights"))
    st.caption("Each table has its own colour scale. Compare the numbers across methods.")
    st.html(
        f'<div class="insight"><h4>Why does “{safe_word}” get this weight?</h4>'
        f'<p>It appears in <strong>{int(document_frequency[selected])} of {len(sentences)} texts</strong>, '
        f'giving it an IDF of <strong>{float(idf[selected]):.3f}</strong>. '
        'Words found in fewer texts get a larger IDF. The final TF-IDF weight also depends on '
        'the word count and normalisation within each row.</p></div>'
    )
    if document_frequency[selected] == len(sentences):
        st.info(
            f'“{selected}” is in every text. Its IDF is 1, not 0. '
            "It can still have a positive TF-IDF weight: common words are not automatically removed."
        )
    st.caption("TF-IDF weights are not probabilities, and each row need not add up to 1.")
    with st.expander("Inspect all document frequencies and IDF weights"):
        st.dataframe(pd.DataFrame({"Texts containing word": document_frequency, "IDF": idf}).round(3), width="stretch")
    with st.expander("Try a quick comparison"):
        st.write(
            "In the original transport example, “bus” and “fares” each occur once in text A. "
            "“Bus” occurs in fewer texts and therefore receives a larger IDF and TF-IDF weight. "
            "Try the “A word in every text” example from Edit texts to see why a common word need not get a zero weight."
        )

with learn:
    animation_col, explanation_col = st.columns([1.8, 1], gap="large")
    with animation_col:
        st.markdown("### Watch the matrix take shape")
        st.caption("A worked example using the two transport headlines from the slides.")
        st.button(
            "Pause animation" if st.session_state.animation_playing else "Play animation",
            key="animation-toggle",
            on_click=toggle_animation,
            icon=":material/pause:" if st.session_state.animation_playing else ":material/play_arrow:",
        )
        gif_path = Path(__file__).with_name("countvectorizer-illustration.gif")
        if gif_path.is_file():
            uri = animation_image(str(gif_path), st.session_state.animation_playing)
            st.html(
                f'<img class="animation" src="{uri}" alt="Two headlines become a four-word vocabulary, '
                'then a matrix with rows A and B containing word counts.">'
            )
        else:
            st.info("The illustrated example is also available in the Week 4 vectorizer notebook.")
    with explanation_col:
        st.html(
            '<div class="learn-card"><div class="section-kicker">01 / FIT</div>'
            '<h4>Learn the vocabulary</h4><p>Find the distinct tokens in the collection. '
            'Each token gets a column.</p></div>'
            '<div class="learn-card"><div class="section-kicker">02 / TRANSFORM</div>'
            '<h4>Fill in the numbers</h4><p>Count the vocabulary words in each text. '
            'The columns stay in the same order.</p></div>'
            '<div class="learn-card"><div class="section-kicker">03 / FIT_TRANSFORM</div>'
            '<h4>Do both in one call</h4><p>Learn the vocabulary and count words in the same collection.</p></div>'
        )
        st.code('cv = CountVectorizer()\ndtm = cv.fit_transform(texts)', language="python")

    st.markdown("### A few things to think about")
    questions = [
        ("Does a word found in every text get a TF-IDF weight of zero?",
         "No. With these defaults its IDF is 1. Its final weight depends on its count and row normalisation. "
         "If both texts consist only of “climate”, both weights are 1. Stopword removal is a separate choice."),
        ("What happens if you add a copy of a text?",
         "Words in that text become more common across the collection. If they were not already in every text, "
         "their IDF decreases. Final TF-IDF weights need not all decrease, because normalisation also changes."),
        ("When would you prefer counts?",
         "When raw frequency answers your research question, such as how often a politician mentions a topic. "
         "For other tasks, compare representations and evaluate what works with your data."),
        ("What can stemming or lemmatisation change?",
         "They can merge word forms into one column. The Porter stemmer turns “running” and “runs” into “run” "
         "but leaves “ran” unchanged. A suitable lemmatiser can map “ran” to “run” using context. "
         "Inspect the results and consider whether merging forms suits your research question."),
    ]
    for question, answer in questions:
        with st.expander(question):
            st.write(answer)

st.html('<div class="footer">BIG DATA AND AUTOMATED CONTENT ANALYSIS · UNIVERSITY OF AMSTERDAM</div>')
