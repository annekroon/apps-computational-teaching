import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# App setup
st.set_page_config(page_title="DTM explorer", layout="wide")
st.title("📚 Text Vectorization Explorer")
st.write("Explore how Count Vectorizer and TF-IDF work using simple sentences.")

# Sidebar: Choose vectorization method
vectorizer_choice = st.sidebar.selectbox(
    "Choose a vectorization method:",
    ("Count Vectorizer", "TF-IDF")
)

# Input Section
st.header("✍️ Input your sentences")
default_text = """The cat sat on the mat.
The dog chased the cat.
The dog and the cat became friends."""
user_input = st.text_area("Enter one sentence per line:", value=default_text, height=150)

# Preprocess input into a list of sentences
sentences = [line.strip() for line in user_input.split('\n') if line.strip()]
if len(sentences) < 2:
    st.warning("Please enter at least two sentences to analyze.")
    st.stop()

# Explanation
st.markdown("""
### 🧠 What is vectorization?

Vectorization turns words or sentences into numbers so that we can analyze them with math and code.

- **Count Vectorizer** just counts how many times each word shows up.
- **TF-IDF Vectorizer** gives higher scores to words that are important (i.e., frequent in one doc but rare across others).
""")

# Vectorization
st.subheader(f"🔢 {vectorizer_choice} Matrix")

if vectorizer_choice == "Count Vectorizer":
    vectorizer = CountVectorizer()
    st.markdown("**Count Vectorizer Explanation**: This method simply counts how often each word appears in each sentence (document). No weighting or scaling is applied.")
else:
    vectorizer = TfidfVectorizer()
    st.markdown("**TF-IDF Vectorizer Explanation**: This method scores words based on how important they are. Words that appear a lot in one sentence but not others get higher scores.")

# Fit and transform sentences
X = vectorizer.fit_transform(sentences)
tokens = vectorizer.get_feature_names_out()
matrix = pd.DataFrame(X.toarray(), columns=tokens)

# Show matrix
st.dataframe(matrix.style.background_gradient(cmap='Blues'), use_container_width=True)

# Heatmap
st.subheader("📈 Term Matrix Heatmap")
fig, ax = plt.subplots(figsize=(10, len(sentences)))
sns.heatmap(matrix, annot=True, cmap="YlGnBu",
            xticklabels=tokens,
            yticklabels=[f"Doc {i+1}" for i in range(len(sentences))],
            cbar=False)
plt.xticks(rotation=45)
st.pyplot(fig)

# Concepts
st.header("📘 Learn the Concepts")
with st.expander("Click to learn what each method does"):
    st.markdown("""
    ### 🔢 Count Vectorizer  
    - Each sentence is treated as a document.
    - Words are turned into a matrix of **word counts**.
    - Example: If "dog" appears twice in a sentence, its value is 2.

    ### 📏 TF-IDF Vectorizer  
    - TF = Term Frequency → how often a word appears in a document.
    - IDF = Inverse Document Frequency → gives less weight to common words across documents.
    - Words like "the" will get **low scores**, while unique words get **higher scores**.
    - The result is a matrix of **weighted word importance**.
    """)

# Footer
st.markdown("---")
st.markdown("Made for CCS-2 students with Streamlit and sklearn")
