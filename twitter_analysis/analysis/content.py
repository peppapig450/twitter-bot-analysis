import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.language import Language

from bot_analysis import ContentAnalysisResult, sentiment_analyzer
from twitter_analysis.models.analysis_result import ContentAnalysisResult

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from spacy.language import Language


def preprocess_text(text: str, nlp: Language | None = None, min_word_length: int = 2) -> list[str]:
    """
    Preprocess text for NLP analysis with error handling.

    Args:
        text: Text to preprocess.
        nlp: Optional spaCy Language model for advanced processing.
        min_word_length: Minimum length for tokens to be included.

    Returns:
        List of preprocessed tokens.

    """
    if not isinstance(text, str) or not text.strip():
        return []

    if nlp is not None:
        try:
            doc = nlp(text.lower())
            return [
                token.lemma_
                for token in doc
                if not token.is_stop and token.is_alpha and len(token.text) > min_word_length
            ]
        except (ValueError, TypeError):
            logging.exception("Error preprocessing text with spaCy")
            return []
    # Fallback to basic tokenization if no NLP or NLTK
    return [w.lower() for w in text.split() if w.isalpha() and len(w) > min_word_length]


def analyze_content_dynamic(
    df: pd.DataFrame,
    nlp: Language | None = None,
    max_features: int = 100,
    n_components: int = 5,
    min_word_length: int = 2,
) -> ContentAnalysisResult:
    """
    Analyze text content in a DataFrame using TF-IDF and topic modeling.

    Args:
        df: DataFrame with a "text" column containing string data.
        nlp: Optional spaCy Language model for advanced NLP processing.
        max_features: Maximum number of features for TF-IDF vectorization.
        n_components: Number of topics for topic modeling.
        min_word_length: Minimum word length for tokenization.

    Returns:
        ContentAnalysisResult with:
            - top_terms: Top 10 terms by TF-IDF score.
            - topics: Topics extracted via LSA.
            - type_token_ratio: Unique tokens / total tokens.
            - avg_sentence_length: Average sentences per text.
            - text_similarity: Mean cosine similarity between texts (0 if single text).

    Notes:
        Returns default result if input is invalid or processing fails.
        Sentiment analysis is handled separately in sentiment.py.

    """
    default_result = ContentAnalysisResult()

    # Input validation
    if df.empty or "text" not in df.columns:
        logging.warning("DataFrame is empty or missing 'text column")
        return default_result

    if not pd.api.types.is_string_dtype(df["text"]):
        logging.warning("Converting 'text' column to string type")
        df["text"] = df["text"].astype(str)

    texts = df["text"].dropna().str.strip().tolist()
    if not texts:
        return default_result

    try:
        # Preprocess texts
        preprocessed = [tokens for text in texts if (tokens := preprocess_text(text, nlp, min_word_length))]
        if not preprocessed:
            return default_result

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", token_pattern=r"(?u)\b\w+\b"
        )
        tfidf_matrix: csr_matrix = cast(
            csr_matrix, vectorizer.fit_transform([" ".join(tokens) for tokens in preprocessed])
        )
        feature_names: NDArray[np.str_] = vectorizer.get_feature_names_out()

        # Top terms
        term_scores = tfidf_matrix.sum(axis=0).A1
        n_terms = min(10, len(feature_names))
        top_indices = term_scores.argsort()[::-1][:n_terms]
        top_terms = [str(feature_names[i]) for i in top_indices]

        # Topic modeling with TruncatedSVD (LSA)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(tfidf_matrix)
        topics = []
        for i, comp in enumerate(svd.components_):
            top_indices = comp.argsort()[-10:][::-1]
            top_terms: list[str] = [str(feature_names[idx]) for idx in top_indices]
            topics.append(f"Topic {i + 1}: {' '.join(top_terms)}")

        # Type-token ratio
        all_tokens = [token for tokens in preprocessed for token in tokens]
        type_token_ratio = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0

        # Average sentence length
        if nlp is not None:
            sentence_counts = [len(list(doc.sents)) for doc in nlp.pipe(texts, disable=["ner", "textcat"])]
        else:
            sentence_counts = [
                max(text.count(".") + text.count("!") + text.count("?"), 1) for text in texts
            ]
        avg_sentence_length = float(np.mean(sentence_counts)) if sentence_counts else 0.0

        # Text similarity
        if tfidf_matrix.shape[0] > 1:
            if tfidf_matrix.shape[0] > 1000:
                centroid = tfidf_matrix.mean(axis=0)
                similarities = cosine_similarity(tfidf_matrix, centroid)
                text_similarity = float(np.mean(similarities))
            else:
                similarity_matrix = cosine_similarity(tfidf_matrix)
                np.fill_diagonal(similarity_matrix, 0)
                text_similarity = float(similarity_matrix.mean())
        else:
            text_similarity = 0.0

        return ContentAnalysisResult(
            top_terms=top_terms,
            topics=topics,
            type_token_ratio=type_token_ratio,
            avg_sentence_length=avg_sentence_length,
            text_similarity=text_similarity,
        )

    except Exception:
        logging.exception("Content analysis failed")
        return default_result
