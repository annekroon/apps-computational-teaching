import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

# Label documents
doc_labels = [f"Doc {i+1}" for i in range(len(sentences))]

# Show what was entered
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
    "💡 Tip: rows are documents, columns are tokens. "
    "Darker cells = higher value. "
    "Scroll right if there are many tokens."
)

# ── Step 5: Heatmap ───────────────────────────────────────────────────────────
st.markdown("---")
st.header("Step 5 — Visualize the matrix")

fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.6), len(sentences) + 1))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f" if method == "TF-IDF Vectorizer" else ".0f",
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor="white",
    cbar=True,
    ax=ax,
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_title(f"{method} — heatmap", fontsize=13, pad=12)
st.pyplot(fig)

# ── Step 6: Observations ──────────────────────────────────────────────────────
st.markdown("---")
st.header("Step 6 — What do you notice?")

# Automatically surface a few observations
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

# ── Step 7: Compare side by side ─────────────────────────────────────────────
st.markdown("---")
st.header("Step 7 — Compare Count vs TF-IDF side by side")
st.write(
    "The same documents, both methods at once. "
    "Notice how the values differ for tokens that appear in every document."
)

cv = CountVectorizer()
tfidf = TfidfVectorizer()

X_cv = cv.fit_transform(sentences)
X_tfidf = tfidf.fit_transform(sentences)

tokens_cv = cv.get_feature_names_out()
tokens_tfidf = tfidf.get_feature_names_out()

df_cv = pd.DataFrame(X_cv.toarray(), columns=tokens_cv, index=doc_labels)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tokens_tfidf, index=doc_labels).round(3)

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

# ── Step 8: Reflection questions ─────────────────────────────────────────────
st.markdown("---")
st.header("Step 8 — Reflect")
st.markdown(
    """
Think about these questions — they will come up in the tutorial:

1. Which tokens get the **highest** TF-IDF scores? Why?
2. What happens to the word *"the"* in TF-IDF compared to Count Vectorizer?
3. If you add a fourth sentence that shares many words with Doc 1, what do you expect to happen?
4. When would you prefer Count Vectorizer over TF-IDF?
5. What would happen if you applied stemming or lemmatization **before** vectorizing?
"""
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("CCS2 · Week 2 · University of Amsterdam")
