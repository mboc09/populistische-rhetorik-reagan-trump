from __future__ import annotations

import argparse
import math
import os
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    "Press Release",
    "Trump Campaign Press Release",
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
    "nancy", "paul", "bush", "george", "dan", "quayle", "donald", "trump", "reagan",
    "online", "gerhard", "peters", "john", "woolley", "presidency", "project",
    "https", "http", "www", "node", "click", "watch",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS

DEFAULT_FINAL_K = {
    "Reagan": 5,
    "Trump": 8,
}

US_MARKERS = {
    "we", "us", "our", "ours", "ourselves", "families", "family", "workers",
    "citizens", "taxpayers", "voters", "children", "neighbors", "communities",
    "hardworking", "forgotten", "ordinary", "silent", "real", "folks",
    "patriots", "patriot", "households", "parents", "seniors", "men", "women",
    "american people", "working families", "our people", "our workers", "our citizens",
    "our families", "our nation", "our country", "the people",
}

ELITE_MARKERS = {
    "establishment", "elite", "elites", "bureaucrats", "bureaucrat", "washington",
    "politicians", "politician", "insiders", "administration", "media", "press",
    "commission", "swamp", "globalists", "special interests", "lobbyists", "ruling class",
    "big government", "federal", "regime", "party bosses", "dnc", "democrat party",
    "liberal elite", "deep state", "officials", "institutions", "commission on presidential debates",
}

OUTGROUP_MARKERS = {
    "soviet", "soviets", "communist", "communists", "communism", "china", "chinese", "iran", "isis",
    "terrorism", "terrorists", "cartels", "cartel", "illegal", "immigration", "immigrant",
    "aliens", "illegal aliens", "criminals", "crime", "enemy", "enemies", "adversaries",
    "socialism", "socialist", "radical", "radicals", "left", "far left", "biden", "kamala",
    "harris", "hillary", "clinton", "obama", "carter", "mondale", "bigots", "anti semitism",
}

ANTAGONISM_MARKERS = {
    "threat", "danger", "dangerous", "destroy", "destroying", "kill", "killing",
    "failure", "failed", "weak", "weakness", "lawlessness", "chaos", "crisis", "hoax",
    "corruption", "corrupt", "criminal", "crime", "misery", "fear", "radical", "radicals",
    "disaster", "catastrophe", "betrayal", "betray", "abandon", "surrender", "lie", "lies",
    "lie", "fraud", "fraudulent", "threaten", "compromised", "enemy", "enemies",
    "totalitarian", "socialism", "socialist", "unwise", "lawless", "illegally",
}

RADICALITY_MARKERS = {
    "criminal", "criminals", "corrupt", "corruption", "traitor", "treason", "hoax",
    "radical", "radicals", "evil", "thug", "thugs", "cartels", "terrorists", "terrorism",
    "lawlessness", "chaos", "invasion", "destroy", "destroying", "poison", "threat",
    "dangerous", "totalitarian", "communist", "socialist", "fake", "bogus", "disaster",
}

ENEMY_CATEGORY_LEXICA = {
    "elite_establishment": {
        "washington", "establishment", "elite", "elites", "bureaucrats", "insiders",
        "administration", "special interests", "globalists", "lobbyists", "federal", "swamp",
        "commission", "officials", "institutions", "media", "press",
    },
    "party_opponent": {
        "carter", "mondale", "biden", "kamala", "harris", "hillary", "clinton", "obama",
        "democrats", "democrat", "left", "far left", "liberals", "socialist", "socialism",
        "pelosi", "sanders", "squad",
    },
    "foreign_adversary": {
        "soviet", "soviets", "communist", "communists", "communism", "china", "chinese", "iran", "isis",
        "terrorism", "terrorists", "adversaries", "enemy", "enemies", "putin", "north korea",
    },
    "immigration_crime": {
        "illegal", "immigration", "immigrant", "aliens", "illegal aliens", "cartels", "cartel",
        "criminal", "criminals", "crime", "crime", "borders", "border", "lawlessness",
    },
    "moral_ideological_enemy": {
        "totalitarian", "bigots", "anti semitism", "intolerance", "socialism", "socialist",
        "radical", "radicals", "communist", "cancel culture", "racist",
    },
}

MONTH_PATTERN = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
DATE_RE = re.compile(rf"{MONTH_PATTERN}\s+\d{{1,2}},\s+(19[5-9]\d|20[0-2]\d)")

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
    text = re.sub(r"\[APP NOTE:.*?\]", " ", text, flags=re.I | re.S)
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



def infer_year(title: str, text: str, speaker: str) -> float:
    sample = f"{title}\n{text[:350]}"

    date_matches = [int(y) for y in DATE_RE.findall(sample)]
    if speaker == "Reagan":
        preferred_dates = [y for y in date_matches if y in (1979, 1980, 1983, 1984)]
        if preferred_dates:
            return float(preferred_dates[0])
    elif speaker == "Trump":
        preferred_dates = [y for y in date_matches if y in (2015, 2016, 2020, 2023, 2024)]
        if preferred_dates:
            return float(preferred_dates[0])

    years = [int(y) for y in YEAR_RE.findall(sample)]
    if not years:
        return np.nan

    if speaker == "Reagan":
        preferred = [y for y in years if y in (1979, 1980, 1983, 1984)]
        return float(preferred[0]) if preferred else np.nan
    if speaker == "Trump":
        preferred = [y for y in years if y in (2015, 2016, 2020, 2023, 2024)]
        return float(preferred[0]) if preferred else np.nan
    return np.nan



def assign_period(speaker: str, year: float) -> str:
    if np.isnan(year):
        return f"{speaker}_undated"
    y = int(year)
    if speaker == "Reagan":
        if y in (1979, 1980):
            return "Reagan_1980"
        if y in (1983, 1984):
            return "Reagan_1984"
        return f"Reagan_{y}"
    if speaker == "Trump":
        if y in (2015, 2016):
            return "Trump_2016"
        if y in (2019, 2020, 2021):
            return "Trump_2020_cycle"
        if y in (2023, 2024):
            return "Trump_2024"
        return f"Trump_{y}"
    return f"{speaker}_{y}"



def segment_corpus_from_titles(raw_text: str, speaker: str, min_words: int = 120) -> pd.DataFrame:
    lines = raw_text.splitlines()
    title_positions = [i for i, line in enumerate(lines) if is_title_line(line)]
    if not title_positions:
        raise ValueError(f"Keine Titellinien im Korpus von {speaker} erkannt.")

    docs = []
    for idx, start in enumerate(title_positions):
        end = title_positions[idx + 1] if idx + 1 < len(title_positions) else len(lines)
        title = lines[start].strip()
        body = "\n".join(lines[start + 1:end]).strip()
        body = clean_metadata(body, speaker_name=speaker)

        if not title.startswith(KEEP_FOR_ANALYSIS_PREFIXES):
            continue
        if len(body.split()) < min_words:
            continue

        year = infer_year(title, body, speaker)
        docs.append(
            {
                "speaker": speaker,
                "title": title,
                "text_raw": body,
                "n_words_raw": len(body.split()),
                "year": year,
                "period": assign_period(speaker, year),
            }
        )

    df = pd.DataFrame(docs)
    if df.empty:
        raise ValueError(f"Nach Segmentierung blieb für {speaker} kein Dokument übrig.")
    return df.reset_index(drop=True)

def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\b[a-z]\b", " ", text)
    tokens = []
    for tok in text.split():
        if len(tok) < 3:
            continue
        if tok in STOPWORDS:
            continue
        if not re.fullmatch(r"[a-z]+", tok):
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
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )



def get_topic_terms(model: NMF, feature_names: Sequence[str], top_n: int = 15) -> List[List[Tuple[str, float]]]:
    all_terms: List[List[Tuple[str, float]]] = []
    for topic_vec in model.components_:
        top_idx = topic_vec.argsort()[::-1][:top_n]
        all_terms.append([(feature_names[i], float(topic_vec[i])) for i in top_idx])
    return all_terms



def fit_topic_model(df: pd.DataFrame, speaker: str, n_topics: int) -> TopicModelResult:
    vectorizer = build_vectorizer()
    dtm = vectorizer.fit_transform(df["text_clean"])
    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=42,
        max_iter=800,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    doc_topic = model.fit_transform(dtm)
    topic_term = model.components_
    feature_names = vectorizer.get_feature_names_out()
    topic_terms = get_topic_terms(model, feature_names, top_n=20)
    topic_prevalence = doc_topic.sum(axis=0) / doc_topic.sum()

    return TopicModelResult(
        speaker=speaker,
        documents=df.copy(),
        vectorizer=vectorizer,
        dtm=dtm,
        model=model,
        doc_topic=doc_topic,
        topic_term=topic_term,
        topic_terms=topic_terms,
        topic_prevalence=topic_prevalence,
    )

def alpha_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())



def term_matches_lexicon(term: str, lexicon: set[str]) -> bool:
    term_norm = " ".join(alpha_tokens(term))
    if not term_norm:
        return False
    if term_norm in lexicon:
        return True
    toks = term_norm.split()
    return any(tok in lexicon for tok in toks)



def weighted_overlap(topic_terms: List[Tuple[str, float]], lexicon: set[str]) -> Tuple[float, List[str]]:
    hits = []
    score = 0.0
    for term, weight in topic_terms:
        if term_matches_lexicon(term, lexicon):
            score += float(weight)
            hits.append(term)
    return score, hits



def topic_populism_dataframe(result: TopicModelResult) -> pd.DataFrame:
    rows = []
    for idx, term_pairs in enumerate(result.topic_terms, start=1):
        us_score, us_hits = weighted_overlap(term_pairs, US_MARKERS)
        elite_score, elite_hits = weighted_overlap(term_pairs, ELITE_MARKERS)
        outgroup_score, outgroup_hits = weighted_overlap(term_pairs, OUTGROUP_MARKERS)
        antagonism_score, antagonism_hits = weighted_overlap(term_pairs, ANTAGONISM_MARKERS)
        radicality_score, radicality_hits = weighted_overlap(term_pairs, RADICALITY_MARKERS)

        them_score = elite_score + outgroup_score
        distinct_hit_count = len(set(us_hits + elite_hits + outgroup_hits + antagonism_hits))
        category_count = sum(x > 0 for x in [us_score, elite_score, outgroup_score, antagonism_score])

        composite = (
            1.0 * us_score
            + 1.4 * them_score
            + 1.2 * antagonism_score
            + 0.9 * radicality_score
            + 0.15 * category_count
        )

        label = " | ".join([term for term, _ in term_pairs[:4]])
        rows.append(
            {
                "speaker": result.speaker,
                "topic_id": idx,
                "topic_label": label,
                "prevalence": float(result.topic_prevalence[idx - 1]),
                "us_score": us_score,
                "them_score": them_score,
                "elite_score": elite_score,
                "outgroup_score": outgroup_score,
                "antagonism_score": antagonism_score,
                "radicality_seed_score": radicality_score,
                "distinct_hit_count": distinct_hit_count,
                "category_count": category_count,
                "populism_score": composite,
                "us_hits": ", ".join(us_hits),
                "elite_hits": ", ".join(elite_hits),
                "outgroup_hits": ", ".join(outgroup_hits),
                "antagonism_hits": ", ".join(antagonism_hits),
                "radicality_hits": ", ".join(radicality_hits),
            }
        )
    df = pd.DataFrame(rows).sort_values(["speaker", "populism_score"], ascending=[True, False])
    df["rank_within_speaker"] = df.groupby("speaker")["populism_score"].rank(method="first", ascending=False)
    return df.reset_index(drop=True)



def select_populist_topics(pop_df: pd.DataFrame) -> pd.DataFrame:
    selected_parts = []
    for speaker, grp in pop_df.groupby("speaker", sort=False):
        grp = grp.sort_values("populism_score", ascending=False).copy()
        score_threshold = max(0.25, float(grp["populism_score"].quantile(0.50)))
        them_threshold = float(grp["them_score"].quantile(0.60)) if len(grp) > 1 else 0.0
        mask = (
            (grp["populism_score"] >= score_threshold)
            & (grp["them_score"] > 0)
            & (grp["distinct_hit_count"] >= 1)
            & (
                (grp["us_score"] > 0)
                | (grp["antagonism_score"] > 0)
                | (grp["them_score"] >= max(0.15, them_threshold))
            )
        )
        chosen = grp.loc[mask].copy()

        if len(chosen) < 2:
            fallback = grp[grp["them_score"] > 0].copy().head(min(2, len(grp[grp["them_score"] > 0])) )
            chosen = pd.concat([chosen, fallback], ignore_index=True).drop_duplicates(subset=["speaker", "topic_id"])

        if chosen.empty:
            chosen = grp.head(1).copy()

        chosen = chosen.sort_values("populism_score", ascending=False).head(3).copy()
        chosen["selected_reason"] = np.where(
            chosen["rank_within_speaker"] == 1,
            "höchster Populismus-Score",
            "oberhalb Schwelle bzw. als starkes Them-/Antagonismus-Topic ergänzt",
        )
        selected_parts.append(chosen)
    return pd.concat(selected_parts, ignore_index=True)

def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]



def split_into_passages(text: str, min_words: int = 45, max_words: int = 180) -> List[str]:
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    passages: List[str] = []

    for para in raw_paragraphs:
        n_words = len(para.split())
        if n_words < min_words:
            continue
        if n_words <= max_words:
            passages.append(para)
            continue

        sents = split_sentences(para)
        if len(sents) <= 2:
            passages.append(para)
            continue

        window: List[str] = []
        window_words = 0
        for sent in sents:
            sent_words = len(sent.split())
            if window_words + sent_words > max_words and window_words >= min_words:
                passages.append(" ".join(window).strip())
                window = [sent]
                window_words = sent_words
            else:
                window.append(sent)
                window_words += sent_words
        if window_words >= min_words:
            passages.append(" ".join(window).strip())

    if not passages:
        sents = split_sentences(text)
        for i in range(0, max(1, len(sents) - 2)):
            chunk = " ".join(sents[i:i + 3]).strip()
            if len(chunk.split()) >= min_words:
                passages.append(chunk)
    return passages



def make_passage_dataframe(documents: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for doc_id, row in documents.reset_index(drop=True).iterrows():
        passages = split_into_passages(row["text_raw"])
        for passage_id, passage in enumerate(passages, start=1):
            clean = normalize_text(passage)
            if len(clean.split()) < 25:
                continue
            rows.append(
                {
                    "speaker": row["speaker"],
                    "document_id": doc_id,
                    "title": row["title"],
                    "year": row["year"],
                    "period": row["period"],
                    "passage_id": passage_id,
                    "passage_text": passage,
                    "passage_clean": clean,
                    "n_words_clean": len(clean.split()),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["speaker", "document_id", "title", "year", "period", "passage_id", "passage_text", "passage_clean", "n_words_clean"])
    return pd.DataFrame(rows)



def analyze_us_them(passage_text: str) -> Dict[str, object]:
    text = passage_text.lower().replace("’", "'")
    tokens = alpha_tokens(text)
    token_counter = Counter(tokens)

    def find_hits(lexicon: set[str]) -> List[str]:
        hits = []
        for seed in lexicon:
            if " " in seed:
                if seed in text:
                    hits.append(seed)
            else:
                if token_counter.get(seed, 0) > 0:
                    hits.append(seed)
        return sorted(set(hits))

    us_hits = find_hits(US_MARKERS)
    elite_hits = find_hits(ELITE_MARKERS)
    outgroup_hits = find_hits(OUTGROUP_MARKERS)
    antagonism_hits = find_hits(ANTAGONISM_MARKERS)
    radicality_hits = find_hits(RADICALITY_MARKERS)

    them_hits = sorted(set(elite_hits + outgroup_hits))
    enemy_categories = []
    for category, lexicon in ENEMY_CATEGORY_LEXICA.items():
        if any(seed in them_hits for seed in lexicon) or any((seed in text) for seed in lexicon if " " in seed):
            enemy_categories.append(category)
        else:
            if any(token_counter.get(tok, 0) > 0 for tok in lexicon if " " not in tok):
                enemy_categories.append(category)
    enemy_categories = sorted(set(enemy_categories))

    us_pronouns = sum(token_counter[p] for p in ["we", "us", "our", "ours"])
    them_pronouns = sum(token_counter[p] for p in ["they", "them", "their", "theirs", "those"])

    radicality_score = len(radicality_hits) + 0.5 * len([t for t in them_hits if t in {"terrorists", "cartels", "criminals", "communist", "socialist", "radicals"}])
    antagonism_density = (len(antagonism_hits) + them_pronouns + len(them_hits)) / max(1, math.sqrt(len(tokens)))

    return {
        "us_markers": ", ".join(us_hits),
        "them_markers": ", ".join(them_hits),
        "elite_markers": ", ".join(elite_hits),
        "outgroup_markers": ", ".join(outgroup_hits),
        "antagonism_markers": ", ".join(antagonism_hits),
        "radicality_markers": ", ".join(radicality_hits),
        "enemy_categories": ", ".join(enemy_categories),
        "enemy_category_count": len(enemy_categories),
        "us_pronouns": us_pronouns,
        "them_pronouns": them_pronouns,
        "radicality_score": float(radicality_score),
        "antagonism_density": float(antagonism_density),
    }



def score_populist_passages(result: TopicModelResult, selected_topics: pd.DataFrame) -> pd.DataFrame:
    topic_ids = selected_topics.loc[selected_topics["speaker"] == result.speaker, "topic_id"].tolist()
    if not topic_ids:
        return pd.DataFrame()

    passages = make_passage_dataframe(result.documents)
    if passages.empty:
        return passages

    X = result.vectorizer.transform(passages["passage_clean"])
    passage_topic = result.model.transform(X)

    rows = []
    topic_id_to_label = dict(zip(selected_topics["topic_id"], selected_topics["topic_label"]))
    topic_id_to_score = dict(zip(selected_topics["topic_id"], selected_topics["populism_score"]))

    for i, p_row in passages.iterrows():
        total_weight = float(passage_topic[i].sum())
        if total_weight <= 0:
            continue
        topic_probs = passage_topic[i] / total_weight
        pop_intensity = float(topic_probs[[t - 1 for t in topic_ids]].sum())
        dominant_topic_id = int(np.argmax(topic_probs) + 1)
        dominant_prob = float(topic_probs[dominant_topic_id - 1])

        analysis = analyze_us_them(p_row["passage_text"])
        rows.append(
            {
                **p_row.to_dict(),
                "dominant_topic_id": dominant_topic_id,
                "dominant_topic_label": " | ".join([term for term, _ in result.topic_terms[dominant_topic_id - 1][:4]]),
                "dominant_topic_prob": dominant_prob,
                "populist_topic_intensity": pop_intensity,
                "max_selected_topic_prob": float(max(topic_probs[t - 1] for t in topic_ids)),
                "selected_topic_ids": ", ".join(map(str, topic_ids)),
                **analysis,
            }
        )
        for t in topic_ids:
            rows[-1][f"topic_{t}_prob"] = float(topic_probs[t - 1])
            rows[-1][f"topic_{t}_label"] = topic_id_to_label[t]
            rows[-1][f"topic_{t}_populism_score"] = float(topic_id_to_score[t])

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["populist_topic_intensity", "max_selected_topic_prob"], ascending=False).reset_index(drop=True)
    return out



def collect_topic_exemplars(passages_df: pd.DataFrame, selected_topics: pd.DataFrame, max_per_topic: int = 4) -> pd.DataFrame:
    rows = []
    if passages_df.empty:
        return pd.DataFrame()

    for _, topic_row in selected_topics.iterrows():
        speaker = topic_row["speaker"]
        topic_id = int(topic_row["topic_id"])
        prob_col = f"topic_{topic_id}_prob"
        if prob_col not in passages_df.columns:
            continue
        subset = passages_df[passages_df["speaker"] == speaker].copy()
        subset = subset[subset[prob_col] > 0]
        subset = subset.sort_values([prob_col, "populist_topic_intensity", "radicality_score"], ascending=False)
        subset = subset.head(max_per_topic)
        if subset.empty:
            continue
        subset["focus_topic_id"] = topic_id
        subset["focus_topic_label"] = topic_row["topic_label"]
        subset["focus_topic_prob"] = subset[prob_col]
        rows.append(subset)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def shannon_entropy(values: Sequence[float]) -> float:
    vals = np.array([v for v in values if v > 0], dtype=float)
    if vals.size == 0:
        return 0.0
    probs = vals / vals.sum()
    return float(-(probs * np.log2(probs)).sum())



def compute_comparative_metrics(result: TopicModelResult, selected_topics: pd.DataFrame, passages_df: pd.DataFrame) -> Dict[str, object]:
    speaker_topics = selected_topics[selected_topics["speaker"] == result.speaker]
    topic_ids = speaker_topics["topic_id"].astype(int).tolist()
    topic_idx = [t - 1 for t in topic_ids]

    doc_topic = result.doc_topic.copy()
    doc_norm = doc_topic / np.maximum(doc_topic.sum(axis=1, keepdims=True), 1e-12)
    doc_pop_intensity = doc_norm[:, topic_idx].sum(axis=1) if topic_idx else np.zeros(len(result.documents))
    doc_dominant = np.argmax(doc_norm, axis=1) + 1 if len(result.documents) else np.array([])

    high_passages = passages_df[passages_df["speaker"] == result.speaker].copy()
    if not high_passages.empty:
        high_passages = high_passages[high_passages["populist_topic_intensity"] >= high_passages["populist_topic_intensity"].quantile(0.80)]
    category_counter = Counter()
    for cats in high_passages.get("enemy_categories", pd.Series(dtype=str)).fillna(""):
        for cat in [c.strip() for c in str(cats).split(",") if c.strip()]:
            category_counter[cat] += 1

    n_tokens = float(result.documents["n_words_clean"].sum())
    antagonistic_passages = high_passages[
        (high_passages["them_markers"].fillna("") != "")
        & (high_passages["antagonism_markers"].fillna("") != "")
    ]
    antagonistic_per_10k = 10000.0 * len(antagonistic_passages) / max(1.0, n_tokens)

    metrics = {
        "speaker": result.speaker,
        "n_documents": int(len(result.documents)),
        "n_selected_populist_topics": int(len(topic_ids)),
        "selected_topic_ids": ", ".join(map(str, topic_ids)),
        "share_documents_dominated_by_populist_topics": float(np.mean(np.isin(doc_dominant, topic_ids))) if len(doc_dominant) else 0.0,
        "mean_document_populist_intensity": float(doc_pop_intensity.mean()) if len(doc_pop_intensity) else 0.0,
        "median_document_populist_intensity": float(np.median(doc_pop_intensity)) if len(doc_pop_intensity) else 0.0,
        "share_high_populist_passages": float(len(high_passages) / max(1, len(passages_df[passages_df['speaker'] == result.speaker]))),
        "antagonistic_passages_per_10000_clean_words": float(antagonistic_per_10k),
        "mean_passage_radicality": float(high_passages["radicality_score"].mean()) if not high_passages.empty else 0.0,
        "mean_passage_antagonism_density": float(high_passages["antagonism_density"].mean()) if not high_passages.empty else 0.0,
        "enemy_category_diversity_count": int(len(category_counter)),
        "enemy_category_entropy": float(shannon_entropy(list(category_counter.values()))),
        "top_enemy_categories": ", ".join([f"{k}:{v}" for k, v in category_counter.most_common(5)]),
    }
    return metrics



def compute_period_metrics(passages_df: pd.DataFrame) -> pd.DataFrame:
    if passages_df.empty:
        return pd.DataFrame()

    rows = []
    for period, grp in passages_df.groupby("period", sort=True):
        if grp.empty:
            continue
        high = grp[grp["populist_topic_intensity"] >= grp["populist_topic_intensity"].quantile(0.80)]
        if high.empty:
            high = grp.nlargest(min(5, len(grp)), "populist_topic_intensity")
        cat_counter = Counter()
        for cats in high["enemy_categories"].fillna(""):
            for cat in [c.strip() for c in str(cats).split(",") if c.strip()]:
                cat_counter[cat] += 1
        rows.append(
            {
                "period": period,
                "speaker": grp["speaker"].iloc[0],
                "n_passages": int(len(grp)),
                "mean_populist_topic_intensity": float(grp["populist_topic_intensity"].mean()),
                "mean_radicality": float(high["radicality_score"].mean()) if not high.empty else 0.0,
                "mean_antagonism_density": float(high["antagonism_density"].mean()) if not high.empty else 0.0,
                "enemy_category_diversity_count": int(len(cat_counter)),
                "top_enemy_categories": ", ".join([f"{k}:{v}" for k, v in cat_counter.most_common(5)]),
            }
        )
    return pd.DataFrame(rows).sort_values(["speaker", "period"]).reset_index(drop=True)

def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()



def plot_populism_scores(pop_df: pd.DataFrame, selected_topics: pd.DataFrame, outdir: Path) -> None:
    for speaker, grp in pop_df.groupby("speaker", sort=False):
        grp = grp.sort_values("populism_score", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        bars = ax.barh(grp["topic_id"].astype(str), grp["populism_score"])
        selected_ids = set(selected_topics[selected_topics["speaker"] == speaker]["topic_id"].astype(int).tolist())
        for bar, topic_id in zip(bars, grp["topic_id"]):
            if int(topic_id) in selected_ids:
                bar.set_linewidth(2.5)
                bar.set_edgecolor("black")
        ax.set_title(f"{speaker}: Populismus-Score pro Topic")
        ax.set_xlabel("Populismus-Score")
        ax.set_ylabel("Topic-ID")
        savefig(outdir / f"{speaker.lower()}_populism_scores.png")



def plot_enemy_category_counts(passages_df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for speaker, grp in passages_df.groupby("speaker", sort=False):
        high = grp[grp["populist_topic_intensity"] >= grp["populist_topic_intensity"].quantile(0.80)]
        counter = Counter()
        for cats in high["enemy_categories"].fillna(""):
            for cat in [c.strip() for c in str(cats).split(",") if c.strip()]:
                counter[cat] += 1
        for cat, count in counter.items():
            rows.append({"speaker": speaker, "enemy_category": cat, "count": count})

    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="enemy_category", columns="speaker", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Feindbild-Kategorien in hoch-populistischen Passagen")
    ax.set_ylabel("Anzahl Passagen")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    savefig(outdir / "enemy_category_counts.png")



def plot_period_shift(period_df: pd.DataFrame, outdir: Path) -> None:
    if period_df.empty:
        return
    wanted = period_df[period_df["period"].isin(["Reagan_1980", "Trump_2016", "Trump_2024", "Trump_2020_cycle"])]
    if wanted.empty:
        wanted = period_df.copy()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(wanted["period"], wanted["mean_populist_topic_intensity"])
    ax.set_title("Verschiebung populistischer Intensität nach Wahlkampfperiode")
    ax.set_ylabel("Mittlere populistische Topic-Intensität")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    savefig(outdir / "period_populist_intensity.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(wanted["period"], wanted["mean_radicality"])
    ax.set_title("Radikalität hoch-populistischer Passagen nach Periode")
    ax.set_ylabel("Mittlere Radikalität")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    savefig(outdir / "period_radicality.png")



def plot_populist_topic_similarity(reagan_result: TopicModelResult, trump_result: TopicModelResult, selected_topics: pd.DataFrame, outdir: Path) -> None:
    r_ids = selected_topics[selected_topics["speaker"] == "Reagan"]["topic_id"].astype(int).tolist()
    t_ids = selected_topics[selected_topics["speaker"] == "Trump"]["topic_id"].astype(int).tolist()
    if not r_ids or not t_ids:
        return

    r_components = reagan_result.model.components_[np.array(r_ids) - 1]
    t_components = trump_result.model.components_[np.array(t_ids) - 1]

    def seed_profile(term_pairs: List[Tuple[str, float]]) -> np.ndarray:
        scores = []
        for lex in [US_MARKERS, ELITE_MARKERS, OUTGROUP_MARKERS, ANTAGONISM_MARKERS, RADICALITY_MARKERS]:
            score, _ = weighted_overlap(term_pairs, lex)
            scores.append(score)
        return np.array(scores, dtype=float)

    r_profiles = np.vstack([seed_profile(reagan_result.topic_terms[i - 1]) for i in r_ids])
    t_profiles = np.vstack([seed_profile(trump_result.topic_terms[i - 1]) for i in t_ids])
    sim = cosine_similarity(r_profiles, t_profiles)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(sim, aspect="auto")
    ax.set_xticks(range(len(t_ids)))
    ax.set_xticklabels([f"T{t}" for t in t_ids])
    ax.set_yticks(range(len(r_ids)))
    ax.set_yticklabels([f"R{r}" for r in r_ids])
    ax.set_title("Ähnlichkeit populistischer Topic-Profile (Seed-Raum)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(outdir / "populist_topic_similarity_seedspace.png")

def wrap_excerpt(text: str, width: int = 110, max_chars: int = 720) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …"
    return textwrap.fill(text, width=width)



def format_topic_row(row: pd.Series) -> str:
    return (
        f"- Topic {int(row['topic_id'])} ({row['topic_label']}): "
        f"Populismus={row['populism_score']:.3f}, Prävalenz={row['prevalence']:.3f}, "
        f"Us={row['us_score']:.3f}, Them={row['them_score']:.3f}, Antagonismus={row['antagonism_score']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Populistische Topic-Analyse für Reagan/Trump")
    parser.add_argument("--reagan", type=str, default="reagan.txt", help="Pfad zu reagan.txt")
    parser.add_argument("--trump", type=str, default="trump.txt", help="Pfad zu trump.txt")
    parser.add_argument("--outdir", type=str, default="populism_outputs", help="Ausgabeverzeichnis")
    parser.add_argument("--reagan-topics", type=int, default=DEFAULT_FINAL_K["Reagan"], help="Anzahl Reagan-Topics")
    parser.add_argument("--trump-topics", type=int, default=DEFAULT_FINAL_K["Trump"], help="Anzahl Trump-Topics")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reagan_docs = preprocess_documents(segment_corpus_from_titles(read_text(args.reagan), "Reagan"))
    trump_docs = preprocess_documents(segment_corpus_from_titles(read_text(args.trump), "Trump"))

    reagan_result = fit_topic_model(reagan_docs, "Reagan", args.reagan_topics)
    trump_result = fit_topic_model(trump_docs, "Trump", args.trump_topics)

    pop_df = pd.concat(
        [
            topic_populism_dataframe(reagan_result),
            topic_populism_dataframe(trump_result),
        ],
        ignore_index=True,
    )
    selected_topics = select_populist_topics(pop_df)

    reagan_passages = score_populist_passages(reagan_result, selected_topics)
    trump_passages = score_populist_passages(trump_result, selected_topics)
    all_passages = pd.concat([reagan_passages, trump_passages], ignore_index=True)

    exemplar_df = collect_topic_exemplars(all_passages, selected_topics, max_per_topic=4)

    comparative_df = pd.DataFrame(
        [
            compute_comparative_metrics(reagan_result, selected_topics, all_passages),
            compute_comparative_metrics(trump_result, selected_topics, all_passages),
        ]
    )
    period_df = compute_period_metrics(all_passages)

    enemy_rows = []
    for _, row in all_passages.iterrows():
        for cat in [c.strip() for c in str(row.get("enemy_categories", "")).split(",") if c.strip()]:
            enemy_rows.append(
                {
                    "speaker": row["speaker"],
                    "period": row["period"],
                    "title": row["title"],
                    "passage_id": row["passage_id"],
                    "enemy_category": cat,
                    "populist_topic_intensity": row["populist_topic_intensity"],
                    "radicality_score": row["radicality_score"],
                }
            )
    enemy_df = pd.DataFrame(enemy_rows)

    pop_df.to_csv(outdir / "populist_topic_scores.csv", index=False)
    selected_topics.to_csv(outdir / "populist_topics_selected.csv", index=False)
    all_passages.to_csv(outdir / "populist_passages.csv", index=False)
    exemplar_df.to_csv(outdir / "populist_topic_exemplars.csv", index=False)
    comparative_df.to_csv(outdir / "comparative_metrics.csv", index=False)
    period_df.to_csv(outdir / "period_metrics.csv", index=False)
    enemy_df.to_csv(outdir / "enemy_category_counts_long.csv", index=False)

    plot_populism_scores(pop_df, selected_topics, outdir)
    plot_enemy_category_counts(all_passages, outdir)
    plot_period_shift(period_df, outdir)
    plot_populist_topic_similarity(reagan_result, trump_result, selected_topics, outdir)

    summary = {
        "reagan_documents": len(reagan_docs),
        "trump_documents": len(trump_docs),
        "selected_populist_topics_reagan": selected_topics[selected_topics["speaker"] == "Reagan"]["topic_id"].tolist(),
        "selected_populist_topics_trump": selected_topics[selected_topics["speaker"] == "Trump"]["topic_id"].tolist(),
        "report": str(outdir / "report_populism.md"),
    }
    (outdir / "run_summary.txt").write_text(str(summary))
    print("Fertig.")
    print(summary)


if __name__ == "__main__":
    main()
