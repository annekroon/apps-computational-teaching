import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="DTM Explorer", layout="wide")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📚 Text Vectorization Explorer")
st.write(
    "This tool shows you step by step how raw text is turned into numbers "
    "that a computer can work with. Try editing the sentences and switching "
    "between methods to see what changes."
)

# ── Step 1: Concept intro ─────────────────────────────────────────────────────
st.markdown("---")
st.header("Step 1 — What is vectorization?")
st.markdown(
    """
When we analyze text with a computer, we need to convert words into numbers.
The result is a **document-term matrix (DTM)**: a table where

- each **row** is a document (sentence, article, tweet, …)
- each **column** is a token (word)
- each **cell** contains a number representing how important that token is in that document

Below you can explore two ways of filling in those numbers.
"""
)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
**🔢 Count Vectorizer**
Simply counts how often each token appears in each document.
- Easy to interpret
- Treats all tokens equally
- Common words like *"the"* get high counts even though they carry little meaning
"""
    )
with col2:
    st.markdown(
        """
**📏 TF-IDF Vectorizer**
Weights each token by how *distinctive* it is:
- High score → frequent in *this* document, rare across *other* documents
- Low score → appears in almost every document (e.g. *"the"*)
- Better for comparing documents or finding key terms
"""
    )

# ── Step 2: Input ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("Step 2 — Enter your documents")
st.write("Each line is treated as one document. Try adding, removing, or editing sentences.")

default_text = (
    "The cat sat on the mat.\n"
    "The dog chased the cat.\n"
    "The dog and the cat became friends."
)
user_input = st.text_area("One sentence per line:", value=default_text, height=150)

sentences = [line.strip() for line in user_input.split("\n") if line.strip()]

if len(sentences) < 2:
    st.warning("Please enter at least two sentences.")
    st.stop()

doc_labels = [f"Doc {i+1}" for i in range(len(sentences))]

with st.expander("👀 See your documents"):
    for label, sent in zip(doc_labels, sentences):
        st.markdown(f"**{label}:** {sent}")

# ── Step 3: Choose method ─────────────────────────────────────────────────────
st.markdown("---")
st.header("Step 3 — Choose a vectorization method")

method = st.radio(
    "Which method do you want to use?",
    ("Count Vectorizer", "TF-IDF Vectorizer"),
    horizontal=True,
)

# ── Step 4: Build and show the matrix ────────────────────────────────────────
st.markdown("---")
st.header("Step 4 — The document-term matrix")

if method == "Count Vectorizer":
    vectorizer = CountVectorizer()
    st.info(
        "**Count Vectorizer:** each cell shows how many times that token appears "
        "in that document. The value is always a whole number."
    )
else:
    vectorizer = TfidfVectorizer()
    st.info(
        "**TF-IDF Vectorizer:** each cell shows a weighted score. "
        "Tokens that appear in every document (like *'the'*) get scores close to 0. "
        "Tokens that are distinctive to one document get higher scores."
    )

X = vectorizer.fit_transform(sentences)
tokens = vectorizer.get_feature_names_out()
matrix = pd.DataFrame(X.toarray(), columns=tokens, index=doc_labels)
matrix = matrix.round(3)

st.dataframe(matrix.style.background_gradient(cmap="Blues"), use_container_width=True)

st.caption(
    "💡 Rows are documents, columns are tokens. "
    "Darker cells = higher value. "
    "Scroll right if there are many tokens."
)

# ── Step 5: Observations ──────────────────────────────────────────────────────
st.markdown("---")
st.header("Step 5 — What do you notice?")

common_tokens = [t for t in tokens if matrix[t].gt(0).all()]
unique_tokens = [t for t in tokens if matrix[t].gt(0).sum() == 1]

if common_tokens:
    st.markdown(
        f"🔵 **Tokens that appear in every document:** {', '.join(f'*{t}*' for t in common_tokens)}  \n"
        f"→ In Count Vectorizer these get high counts. In TF-IDF their score is **0 or very low** "
        f"because they are not distinctive. This is exactly why we often remove stopwords."
    )

if unique_tokens:
    st.markdown(
        f"🟢 **Tokens that appear in only one document:** {', '.join(f'*{t}*' for t in unique_tokens)}  \n"
        f"→ In TF-IDF these get **higher scores** because they are distinctive."
    )

if not common_tokens and not unique_tokens:
    st.markdown("Try adding more sentences to see patterns emerge.")

# ── Step 6: Compare side by side ─────────────────────────────────────────────
st.markdown("---")
st.header("Step 6 — Compare Count vs TF-IDF side by side")
st.write(
    "The same documents, both methods at once. "
    "Notice how the values differ for tokens that appear in every document."
)

cv = CountVectorizer()
tfidf = TfidfVectorizer()

X_cv = cv.fit_transform(sentences)
X_tfidf = tfidf.fit_transform(sentences)

df_cv = pd.DataFrame(X_cv.toarray(), columns=cv.get_feature_names_out(), index=doc_labels)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out(), index=doc_labels).round(3)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Count Vectorizer**")
    st.dataframe(df_cv.style.background_gradient(cmap="Blues"), use_container_width=True)
with col_b:
    st.markdown("**TF-IDF Vectorizer**")
    st.dataframe(df_tfidf.style.background_gradient(cmap="Oranges"), use_container_width=True)

st.caption(
    "💡 Pick a token that appears in every document (e.g. *'the'*) and compare its value "
    "in both tables. Then pick a token that appears in only one document and do the same."
)

# ── Step 7: Reflection questions ─────────────────────────────────────────────
st.markdown("---")
st.header("Step 7 — Reflect")
st.write("Think about each question, then click to reveal the answer.")

questions = [
    (
        "1. Which tokens get the **highest** TF-IDF scores? Why?",
        "Tokens that appear frequently in one document but rarely (or never) in the others. "
        "They are distinctive — they tell you something specific about that document. "
        "For example, a word that only appears in Doc 2 will get a high TF-IDF score for Doc 2 and 0 for all others."
    ),
    (
        "2. What happens to the word *'the'* in TF-IDF compared to Count Vectorizer?",
        "In Count Vectorizer, *'the'* gets a high count because it appears many times across documents. "
        "In TF-IDF, its score is very low (close to 0) because it appears in *every* document — "
        "it carries no information about what makes one document different from another. "
        "This is exactly why stopword removal is useful."
    ),
    (
        "3. If you add a fourth sentence that shares many words with Doc 1, what do you expect to happen?",
        "Those shared words will now appear in more documents, so their IDF (inverse document frequency) "
        "will decrease — meaning their TF-IDF score will drop. "
        "Words that were distinctive to Doc 1 become less distinctive once another document uses them too."
    ),
    (
        "4. When would you prefer Count Vectorizer over TF-IDF?",
        "When raw frequency matters — for example, counting how often a politician mentions a topic, "
        "or when you are feeding counts into a model that expects integer input (like Naive Bayes).\n\n"
        "That said, there is no universal rule. In practice, it is often a matter of **trying both and "
        "evaluating which works better** for your specific data and task. Do not assume TF-IDF is always "
        "superior — sometimes raw counts perform just as well or better."
    ),
    (
        "5. What would happen if you applied stemming or lemmatization **before** vectorizing?",
        "Different surface forms of the same word would be merged into one column. "
        "For example, *'running'*, *'runs'*, and *'ran'* would all become *'run'* — "
        "one column instead of three. This reduces the size of the matrix and can improve "
        "the quality of the representation, especially for smaller corpora."
    ),
]

for question, answer in questions:
    st.markdown(question)
    with st.expander("💡 Show answer"):
        st.markdown(answer)
    st.markdown("")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("CCS2 · Week 2 · University of Amsterdam")
