"""Mini-RAG: chunking + 5 retrievers (keyword, TF-IDF, BM25, LSA embeddings, hybrid) + extractive answering."""
import math
from collections import Counter

import numpy as np

from text import sentences, tokenize

CHUNK_SENTENCES, STRIDE = 2, 1         # overlapping 2-sentence windows
BM25_K1, BM25_B = 1.5, 0.75
RRF_K = 60                             # reciprocal rank fusion constant


def chunk(docs):
    """Split each document into overlapping windows of sentences, so a retrieved chunk is focused but keeps context."""
    chunks = []
    for d in docs:
        sents = sentences(d["text"])
        starts = range(0, max(1, len(sents) - CHUNK_SENTENCES + 1), STRIDE)
        for s in starts:
            chunks.append({"doc": d["id"], "title": d["title"], "text": " ".join(sents[s:s + CHUNK_SENTENCES])})
    return chunks


class Index:
    def __init__(self, docs, lsa_dims=24):
        self.chunks = chunk(docs)
        # the title is included in each chunk's searchable text: a cheap, very effective trick
        self.tokens = [tokenize(c["title"] + " " + c["text"]) for c in self.chunks]
        self.vocab = sorted({t for toks in self.tokens for t in toks})
        self.vid = {t: i for i, t in enumerate(self.vocab)}
        N = len(self.chunks)
        df = Counter(t for toks in self.tokens for t in set(toks))
        self.df = np.array([df[t] for t in self.vocab], float)
        self.idf = np.log((N + 1) / (self.df + 1)) + 1                       # smoothed TF-IDF idf
        self.bm25_idf = np.log(1 + (N - self.df + 0.5) / (self.df + 0.5))
        self.lengths = np.array([len(t) for t in self.tokens], float)
        self.avgdl = self.lengths.mean()
        self.tf = np.zeros((N, len(self.vocab)))
        for i, toks in enumerate(self.tokens):
            for t, c in Counter(toks).items():
                self.tf[i, self.vid[t]] = c
        self.tfidf = self._norm(self.tf * self.idf)
        self.fit_lsa(lsa_dims)

    @staticmethod
    def _norm(M):
        n = np.linalg.norm(M, axis=-1, keepdims=True)
        return M / np.where(n == 0, 1, n)

    def fit_lsa(self, k):
        """Latent Semantic Analysis: SVD of the TF-IDF matrix squeezes ~500 word dimensions into k 'topic'
        dimensions. Words that appear in similar chunks end up close together, giving a dense embedding."""
        U, S, Vt = np.linalg.svd(self.tfidf, full_matrices=False)
        self.k = k
        self.V = Vt[:k].T                                                    # (vocab, k): projects a query into topic space
        self.chunk_emb = self._norm(self.tfidf @ self.V)
        self.explained = float((S[:k] ** 2).sum() / (S ** 2).sum())

    def query_vec(self, q):
        v = np.zeros(len(self.vocab))
        for t, c in Counter(tokenize(q)).items():
            if t in self.vid:
                v[self.vid[t]] = c
        return v

    # ---------- the five retrievers: each returns one score per chunk ----------
    def score(self, q, method):
        qv = self.query_vec(q)
        if method == "keyword":                       # how many distinct query words appear in the chunk
            return ((self.tf > 0) & (qv > 0)).sum(1).astype(float)
        if method == "tfidf":                         # cosine similarity of TF-IDF vectors
            return self.tfidf @ self._norm(qv * self.idf)
        if method == "bm25":                          # the classic search-engine formula
            present = qv > 0
            tf = self.tf[:, present]
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[:, None] / self.avgdl)
            return (self.bm25_idf[present] * tf * (BM25_K1 + 1) / denom).sum(1)
        if method == "lsa":                           # cosine similarity of dense embeddings
            return self.chunk_emb @ self._norm(self._norm(qv * self.idf) @ self.V)
        if method == "hybrid":                        # reciprocal rank fusion of BM25 + LSA
            fused = np.zeros(len(self.chunks))
            for m in ("bm25", "lsa"):
                ranks = np.empty(len(self.chunks)); ranks[np.argsort(-self.score(q, m), kind="stable")] = np.arange(1, len(self.chunks) + 1)
                fused += 1 / (RRF_K + ranks)
            return fused
        raise ValueError(method)

    def search(self, q, method="hybrid", top=5):
        s = self.score(q, method)
        order = np.argsort(-s, kind="stable")
        return [(int(i), float(s[i])) for i in order[:top]]

    def rank_docs(self, q, method):
        """Document ranking = order in which each document's best chunk appears."""
        seen, out = set(), []
        for i, _ in self.search(q, method, top=len(self.chunks)):
            d = self.chunks[i]["doc"]
            if d not in seen:
                seen.add(d); out.append(d)
        return out

    def answer(self, q, method="hybrid"):
        """Extractive 'generation': the sentence in the top chunks that best matches the question,
        weighted by how rare (informative) each matched word is. An LLM would replace this step."""
        hits = self.search(q, method, top=3)
        qt = set(tokenize(q))
        best = None
        for rank, (i, _) in enumerate(hits):
            for s in sentences(self.chunks[i]["text"]):
                st = set(tokenize(s))
                score = sum(self.bm25_idf[self.vid[t]] for t in qt & st if t in self.vid) - 0.01 * rank
                if best is None or score > best[0] + 1e-9:
                    best = (score, s, i)
        return {"sentence": best[1], "chunk": best[2], "sources": hits}
