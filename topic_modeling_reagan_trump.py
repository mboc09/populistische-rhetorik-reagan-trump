from __future__ import annotations

import argparse
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import NMF, PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

TITLE_PREFIXES = (
    "Address",
    "Remarks",
    "Speech",
    "Statement",
    "Radio Address",
    "Question-and-Answer",
    "Debate",
    "Interview",
    "Exchange",
    "Conversation",
    "Announcement",
    "Campaign",
    "Trump Campaign Press Release",
    "Press Release",
)

KEEP_FOR_ANALYSIS_PREFIXES = (
    "Address",
    "Remarks",
    "Speech",
)

CUSTOM_STOPWORDS = {
    "applause", "laughter", "crowd", "cheers", "thank", "thanks", "lot", "lets", "ve",
    "dont", "didnt", "doesnt", "isnt", "cant", "wont", "im", "youre", "weve",
    "theyre", "thats", "theres", "ive", "id", "ill", "mr", "mrs", "ms",
    "people", "thing", "things", "year", "years", "day", "days", "way", "time",
    "going", "go", "know", "want", "like", "right", "said", "say", "says",
    "just", "make", "made", "come", "came", "look", "looking", "let", "well",
    "today", "tonight", "tomorrow", "yesterday", "everybody", "really", "guy",
    "good", "night", "friends", "friend", "fellow", "ladies", "gentlemen",
    "audience", "delegates", "chairman", "governor", "nominee", "convention",
    "welcome", "warm", "wonderful", "deep", "began", "express", "support",
    "america", "american", "americans", "country", "nation", "states", "united",
    "president", "presidential", "campaign", "republican", "democrat", "democratic",
    "nancy", "paul", "bush", "george", "dan", "quayle",
    "online", "gerhard", "peters", "john", "woolley", "presidency", "project",
    "https", "http", "www", "node",
}

STOPWORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS

DEFAULT_TOPIC_RANGE = list(range(4, 11))
DEFAULT_FINAL_K = {
    "Reagan": 5,
    "Trump": 8,
}

@dataclass
class TopicModelResult:
    speaker: str
    documents: pd.DataFrame
    vectorizer: TfidfVectorizer
    dtm: np.ndarray
    model: NMF
    doc_topic: np.ndarray
    topic_term: np.ndarray
    topic_terms: List[List[Tuple[str, float]]]
    topic_prevalence: np.ndarray
    evaluation: pd.DataFrame

def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def is_title_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) < 12 or len(s) > 180:
        return False
    if s.startswith("Donald J. Trump") or s.startswith("Ronald Reagan"):
        return False
    if s.startswith("The American Presidency Project"):
        return False
    if re.search(r"https?://", s):
        return False
    return s.startswith(TITLE_PREFIXES)


def clean_metadata(text: str, speaker_name: str | None = None) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bThe American Presidency Project\b", " ", text, flags=re.I)
    if speaker_name:
        text = re.sub(
            rf"{re.escape(speaker_name)}.*?Online by Gerhard Peters and John T\. Woolley.*",
            " ",
            text,
            flags=re.I,
        )
    text = re.sub(r"\bOnline by Gerhard Peters and John T\. Woolley\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def segment_corpus_from_titles(
    raw_text: str,
    speaker: str,
    min_words: int = 120,
) -> pd.DataFrame:
    """
    Zerlegt das Korpus anhand von Titellinien in Einzeldokumente.
    Alles vor dem ersten erkannten Titel wird ignoriert.
    """
    lines = raw_text.splitlines()
    title_positions = [i for i, line in enumerate(lines) if is_title_line(line)]

    docs = []
    if not title_positions:
        raise ValueError(f"Keine Titellinien im Korpus von {speaker} erkannt.")

    for idx, start in enumerate(title_positions):
        end = title_positions[idx + 1] if idx + 1 < len(title_positions) else len(lines)
        title = lines[start].strip()
        body = "\n".join(lines[start + 1:end]).strip()
        body = clean_metadata(body, speaker_name=speaker)

        if not title.startswith(KEEP_FOR_ANALYSIS_PREFIXES):
            continue

        if len(body.split()) < min_words:
            continue

        docs.append(
            {
                "speaker": speaker,
                "title": title,
                "text_raw": body,
                "n_words_raw": len(body.split()),
            }
        )

    df = pd.DataFrame(docs)
    if df.empty:
        raise ValueError(f"Nach Segmentierung blieb für {speaker} kein Dokument übrig.")
    return df.reset_index(drop=True)

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\b[a-z]\b", " ", text)
    tokens = []
    for tok in text.split():
        tok = tok.strip("'")
        if len(tok) < 3:
            continue
        if tok in STOPWORDS:
            continue
        if not re.match(r"^[a-z]+$", tok):
            continue
        tokens.append(tok)
    return " ".join(tokens)


def preprocess_documents(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text_clean"] = out["text_raw"].apply(normalize_text)
    out["n_words_clean"] = out["text_clean"].str.split().str.len()
    out = out[out["n_words_clean"] >= 50].reset_index(drop=True)
    if out.empty:
        raise ValueError("Nach Vorverarbeitung sind keine Dokumente mehr übrig.")
    return out

def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,          
        preprocessor=None,
        tokenizer=str.split,
        token_pattern=None,
        stop_words=None,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=4000,
        sublinear_tf=True,
    )


def topic_diversity(components: np.ndarray, feature_names: Sequence[str], top_n: int = 10) -> float:
    """
    Anteil einzigartiger Top-Terme über alle Topics hinweg.
    Höher = interpretierbarere, weniger redundante Topics.
    """
    chosen = []
    for comp in components:
        idx = np.argsort(comp)[::-1][:top_n]
        chosen.extend(feature_names[i] for i in idx)
    unique_ratio = len(set(chosen)) / max(len(chosen), 1)
    return float(unique_ratio)


def evaluate_topic_numbers(
    texts: Sequence[str],
    topic_range: Sequence[int],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    vectorizer = build_vectorizer()
    dtm = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    rows = []
    for k in topic_range:
        model = NMF(
            n_components=k,
            init="nndsvda",
            random_state=random_state,
            max_iter=800,
            alpha_W=0.0,
            alpha_H=0.0,
            l1_ratio=0.0,
        )
        W = model.fit_transform(dtm)
        H = model.components_

        error = model.reconstruction_err_
        diversity = topic_diversity(H, feature_names, top_n=10)

        # Balance: Wie gleichmäßig Topics über Dokumente verteilt sind
        dominant_topics = np.argmax(W, axis=1)
        counts = np.bincount(dominant_topics, minlength=k)
        balance = (counts > 0).sum() / k

        rows.append(
            {
                "k": k,
                "reconstruction_error": error,
                "topic_diversity": diversity,
                "topic_balance": balance,
            }
        )

    eval_df = pd.DataFrame(rows)

    err_scaled = (eval_df["reconstruction_error"] - eval_df["reconstruction_error"].min()) / (
        eval_df["reconstruction_error"].max() - eval_df["reconstruction_error"].min() + 1e-9
    )
    div_scaled = (eval_df["topic_diversity"] - eval_df["topic_diversity"].min()) / (
        eval_df["topic_diversity"].max() - eval_df["topic_diversity"].min() + 1e-9
    )
    bal_scaled = (eval_df["topic_balance"] - eval_df["topic_balance"].min()) / (
        eval_df["topic_balance"].max() - eval_df["topic_balance"].min() + 1e-9
    )

    eval_df["selection_score"] = (1 - err_scaled) * 0.5 + div_scaled * 0.3 + bal_scaled * 0.2
    return eval_df, vectorizer, dtm


def extract_topic_terms(
    model: NMF,
    feature_names: Sequence[str],
    top_n: int = 12,
) -> List[List[Tuple[str, float]]]:
    topic_terms = []
    for topic_idx, comp in enumerate(model.components_):
        idx = np.argsort(comp)[::-1][:top_n]
        topic_terms.append([(feature_names[i], float(comp[i])) for i in idx])
    return topic_terms


def fit_topic_model(
    speaker: str,
    documents: pd.DataFrame,
    n_topics: int,
    topic_range: Sequence[int] = DEFAULT_TOPIC_RANGE,
    random_state: int = 42,
) -> TopicModelResult:
    evaluation, vectorizer, dtm = evaluate_topic_numbers(
        documents["text_clean"].tolist(),
        topic_range=topic_range,
        random_state=random_state,
    )

    final_vectorizer = build_vectorizer()
    final_dtm = final_vectorizer.fit_transform(documents["text_clean"])
    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=random_state,
        max_iter=1000,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    W = model.fit_transform(final_dtm)
    H = model.components_

    topic_prevalence = W.sum(axis=0) / W.sum()

    topic_terms = extract_topic_terms(model, final_vectorizer.get_feature_names_out(), top_n=12)

    result = TopicModelResult(
        speaker=speaker,
        documents=documents.copy(),
        vectorizer=final_vectorizer,
        dtm=final_dtm,
        model=model,
        doc_topic=W,
        topic_term=H,
        topic_terms=topic_terms,
        topic_prevalence=topic_prevalence,
        evaluation=evaluation,
    )
    return result

def topic_label(topic_terms: List[Tuple[str, float]], n_terms: int = 4) -> str:
    return " | ".join(term for term, _ in topic_terms[:n_terms])


def get_top_documents_for_topic(
    result: TopicModelResult,
    topic_idx: int,
    n: int = 3,
    snippet_chars: int = 280,
) -> pd.DataFrame:
    scores = result.doc_topic[:, topic_idx]
    order = np.argsort(scores)[::-1][:n]
    rows = []
    for doc_idx in order:
        raw = result.documents.loc[doc_idx, "text_raw"]
        snippet = textwrap.shorten(raw.replace("\n", " "), width=snippet_chars, placeholder=" ...")
        rows.append(
            {
                "speaker": result.speaker,
                "topic_id": topic_idx + 1,
                "topic_label": topic_label(result.topic_terms[topic_idx]),
                "document_rank": len(rows) + 1,
                "document_title": result.documents.loc[doc_idx, "title"],
                "document_score": float(scores[doc_idx]),
                "snippet": snippet,
            }
        )
    return pd.DataFrame(rows)


def topic_summary_table(result: TopicModelResult) -> pd.DataFrame:
    rows = []
    for i, terms in enumerate(result.topic_terms):
        rows.append(
            {
                "speaker": result.speaker,
                "topic_id": i + 1,
                "prevalence": float(result.topic_prevalence[i]),
                "label": topic_label(terms),
                "top_terms": ", ".join(term for term, _ in terms),
            }
        )
    return pd.DataFrame(rows).sort_values("prevalence", ascending=False).reset_index(drop=True)


def compare_models(reagan: TopicModelResult, trump: TopicModelResult) -> pd.DataFrame:
    """
    Vergleicht Themen zwischen beiden Korpora über Cosine Similarity
    der Topic-Term-Matrizen. Dazu werden die Topic-Vektoren auf einen
    gemeinsamen Feature-Raum projiziert.
    """
    reagan_vocab = list(reagan.vectorizer.get_feature_names_out())
    trump_vocab = list(trump.vectorizer.get_feature_names_out())
    joint_vocab = sorted(set(reagan_vocab) | set(trump_vocab))
    joint_index = {term: i for i, term in enumerate(joint_vocab)}

    def project(topic_term: np.ndarray, vocab: Sequence[str]) -> np.ndarray:
        mat = np.zeros((topic_term.shape[0], len(joint_vocab)), dtype=float)
        for old_idx, term in enumerate(vocab):
            mat[:, joint_index[term]] = topic_term[:, old_idx]
        return mat

    reg_joint = project(reagan.topic_term, reagan_vocab)
    tr_joint = project(trump.topic_term, trump_vocab)

    sim = cosine_similarity(reg_joint, tr_joint)
    rows = []
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            rows.append(
                {
                    "reagan_topic_id": i + 1,
                    "reagan_label": topic_label(reagan.topic_terms[i]),
                    "trump_topic_id": j + 1,
                    "trump_label": topic_label(trump.topic_terms[j]),
                    "cosine_similarity": float(sim[i, j]),
                }
            )
    return pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False).reset_index(drop=True)


def save_k_selection_plot(result: TopicModelResult, outdir: Path) -> None:
    df = result.evaluation.sort_values("k")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["k"], df["reconstruction_error"], marker="o", label="Reconstruction Error")
    ax1.set_xlabel("Anzahl Topics (k)")
    ax1.set_ylabel("Reconstruction Error")
    ax1.set_title(f"{result.speaker}: Themenzahl-Evaluation")

    ax2 = ax1.twinx()
    ax2.plot(df["k"], df["selection_score"], marker="s", linestyle="--", label="Selection Score")
    ax2.set_ylabel("Selection Score")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"{result.speaker.lower()}_k_selection.png", dpi=180)
    plt.close(fig)


def save_topic_prevalence_plot(result: TopicModelResult, outdir: Path) -> None:
    order = np.argsort(result.topic_prevalence)[::-1]
    labels = [f"T{idx + 1}" for idx in order]
    values = result.topic_prevalence[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    ax.set_title(f"{result.speaker}: Topic-Prävalenz")
    ax.set_xlabel("Topics")
    ax.set_ylabel("Anteil am Dokument-Topic-Gewicht")
    for x, y in zip(labels, values):
        ax.text(x, y + 0.003, f"{y:.2%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f"{result.speaker.lower()}_topic_prevalence.png", dpi=180)
    plt.close(fig)


def save_top_terms_plot(result: TopicModelResult, outdir: Path, top_n: int = 10) -> None:
    n_topics = len(result.topic_terms)
    ncols = 2
    nrows = int(np.ceil(n_topics / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for topic_idx, ax in enumerate(axes):
        if topic_idx >= n_topics:
            ax.axis("off")
            continue

        terms = result.topic_terms[topic_idx][:top_n]
        labels = [term for term, _ in terms][::-1]
        weights = [weight for _, weight in terms][::-1]

        ax.barh(labels, weights)
        ax.set_title(f"{result.speaker} – Topic {topic_idx + 1}\n{topic_label(result.topic_terms[topic_idx], n_terms=4)}")
        ax.set_xlabel("Gewicht")

    fig.tight_layout()
    fig.savefig(outdir / f"{result.speaker.lower()}_top_terms.png", dpi=180)
    plt.close(fig)


def save_wordclouds(result: TopicModelResult, outdir: Path) -> None:
    if not WORDCLOUD_AVAILABLE:
        print(f"[Hinweis] wordcloud nicht installiert – Wordclouds für {result.speaker} werden übersprungen.")
        return

    n_topics = len(result.topic_terms)
    ncols = 2
    nrows = int(np.ceil(n_topics / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for topic_idx, ax in enumerate(axes):
        if topic_idx >= n_topics:
            ax.axis("off")
            continue

        freq = dict(result.topic_terms[topic_idx])
        wc = WordCloud(
            width=900,
            height=500,
            background_color="white",
            collocations=False,
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{result.speaker} – Topic {topic_idx + 1}")

    fig.tight_layout()
    fig.savefig(outdir / f"{result.speaker.lower()}_wordclouds.png", dpi=180)
    plt.close(fig)


def save_doc_topic_pca(result: TopicModelResult, outdir: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(result.doc_topic)
    dominant = np.argmax(result.doc_topic, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=dominant, alpha=0.75)
    ax.set_title(f"{result.speaker}: Dokumente im Topic-Raum (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(*scatter.legend_elements(), title="Dominantes Topic", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(outdir / f"{result.speaker.lower()}_doc_topic_pca.png", dpi=180)
    plt.close(fig)


def save_similarity_heatmap(
    comparison_df: pd.DataFrame,
    reagan_result: TopicModelResult,
    trump_result: TopicModelResult,
    outdir: Path,
) -> None:
    sim_matrix = comparison_df.pivot(
        index="reagan_topic_id",
        columns="trump_topic_id",
        values="cosine_similarity",
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(sim_matrix.values, aspect="auto")
    ax.set_title("Cosine Similarity: Reagan-Topics vs. Trump-Topics")
    ax.set_xlabel("Trump Topics")
    ax.set_ylabel("Reagan Topics")
    ax.set_xticks(np.arange(sim_matrix.shape[1]))
    ax.set_xticklabels([f"T{c}" for c in sim_matrix.columns])
    ax.set_yticks(np.arange(sim_matrix.shape[0]))
    ax.set_yticklabels([f"T{r}" for r in sim_matrix.index])

    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            ax.text(j, i, f"{sim_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity")
    fig.tight_layout()
    fig.savefig(outdir / "reagan_trump_topic_similarity.png", dpi=180)
    plt.close(fig)

def save_csv_outputs(
    reagan: TopicModelResult,
    trump: TopicModelResult,
    comparison: pd.DataFrame,
    outdir: Path,
) -> None:
    topic_table = pd.concat(
        [topic_summary_table(reagan), topic_summary_table(trump)],
        ignore_index=True,
    )
    topic_table.to_csv(outdir / "topic_summary.csv", index=False)

    comparison.to_csv(outdir / "topic_similarity.csv", index=False)

    reps = []
    for result in [reagan, trump]:
        for t in range(result.model.n_components):
            reps.append(get_top_documents_for_topic(result, t, n=3))
    pd.concat(reps, ignore_index=True).to_csv(outdir / "representative_documents.csv", index=False)

    reagan.documents.to_csv(outdir / "reagan_documents_segmented.csv", index=False)
    trump.documents.to_csv(outdir / "trump_documents_segmented.csv", index=False)

def run_pipeline(reagan_path: str, trump_path: str, outdir: str) -> None:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    reagan_raw = read_text(reagan_path)
    trump_raw = read_text(trump_path)

    reagan_docs = segment_corpus_from_titles(reagan_raw, speaker="Ronald Reagan")
    trump_docs = segment_corpus_from_titles(trump_raw, speaker="Donald J. Trump")

    reagan_docs = preprocess_documents(reagan_docs)
    trump_docs = preprocess_documents(trump_docs)

    reagan_result = fit_topic_model(
        speaker="Reagan",
        documents=reagan_docs,
        n_topics=DEFAULT_FINAL_K["Reagan"],
        topic_range=DEFAULT_TOPIC_RANGE,
    )
    trump_result = fit_topic_model(
        speaker="Trump",
        documents=trump_docs,
        n_topics=DEFAULT_FINAL_K["Trump"],
        topic_range=DEFAULT_TOPIC_RANGE,
    )

    comparison_df = compare_models(reagan_result, trump_result)

    save_csv_outputs(reagan_result, trump_result, comparison_df, outdir_path)

    save_k_selection_plot(reagan_result, outdir_path)
    save_k_selection_plot(trump_result, outdir_path)
    save_topic_prevalence_plot(reagan_result, outdir_path)
    save_topic_prevalence_plot(trump_result, outdir_path)
    save_top_terms_plot(reagan_result, outdir_path)
    save_top_terms_plot(trump_result, outdir_path)
    save_wordclouds(reagan_result, outdir_path)
    save_wordclouds(trump_result, outdir_path)
    save_doc_topic_pca(reagan_result, outdir_path)
    save_doc_topic_pca(trump_result, outdir_path)
    save_similarity_heatmap(comparison_df, reagan_result, trump_result, outdir_path)

    print("=" * 72)
    print("Topic Modeling erfolgreich abgeschlossen.")
    print(f"Ausgabeordner: {outdir_path.resolve()}")
    print("=" * 72)
    print("\nReagan – Themen:")
    print(topic_summary_table(reagan_result).to_string(index=False))
    print("\nTrump – Themen:")
    print(topic_summary_table(trump_result).to_string(index=False))
    print("\nÄhnlichste Topic-Paare:")
    print(comparison_df.head(10).to_string(index=False))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topic Modeling: Reagan vs. Trump")
    parser.add_argument("--reagan", default="reagan.txt", help="Pfad zu reagan.txt")
    parser.add_argument("--trump", default="trump.txt", help="Pfad zu trump.txt")
    parser.add_argument("--outdir", default="topic_model_outputs", help="Ausgabeordner")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        reagan_path=args.reagan,
        trump_path=args.trump,
        outdir=args.outdir,
    )
