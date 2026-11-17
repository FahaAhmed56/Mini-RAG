// The same RAG engine as rag/text.py + rag/engine.py, running in the browser on the exported index.
const STOPWORDS = new Set(`a an the and or but if of to in on at by for with from as is are was were be been being am do does did
i me my we our you your he she it its they them their this that these those there here what which who whom when where why how
can could should would will shall may might must have has had not no yes so than then too very just also into about over
any all each every some such only own same other more most up out off again further once per get gets got`.split(/\s+/));

function stem(w) {
  if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.length > 5 && w.endsWith("ing")) return w.slice(0, -3);
  if (w.length > 4 && w.endsWith("ed")) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith("es") && "sxz".includes(w[w.length - 3])) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) return w.slice(0, -1);
  return w;
}
const tokenize = text => (text.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => !STOPWORDS.has(w)).map(stem);
const sentences = text => text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);

class Rag {
  constructor(index) {
    this.ix = index;
    this.vid = new Map(index.vocab.map((t, i) => [t, i]));
    this.N = index.chunks.length;
  }
  queryCounts(q) {                       // vocab index -> count, only words the index knows
    const m = new Map();
    for (const t of tokenize(q)) if (this.vid.has(t)) { const j = this.vid.get(t); m.set(j, (m.get(j) || 0) + 1); }
    return m;
  }
  static normalize(v) { const n = Math.hypot(...v); return n ? v.map(x => x / n) : v; }

  score(q, method) {
    const { ix } = this, qc = this.queryCounts(q);
    if (method === "keyword") return ix.tf.map(row => [...qc.keys()].filter(j => row[j]).length);
    if (method === "tfidf") {
      const qn = Math.hypot(...[...qc].map(([j, c]) => c * ix.idf[j])) || 1;
      return ix.tf.map(row => {
        let dot = 0, norm = 0;
        for (const j in row) { const w = row[j] * ix.idf[j]; norm += w * w; if (qc.has(+j)) dot += w * qc.get(+j) * ix.idf[j]; }
        return norm ? dot / Math.sqrt(norm) / qn : 0;
      });
    }
    if (method === "bm25") {
      const k1 = 1.5, b = 0.75;
      return ix.tf.map((row, i) => {
        let s = 0;
        for (const j of qc.keys()) {
          const tf = row[j] || 0;
          if (tf) s += ix.bm25_idf[j] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * ix.lengths[i] / ix.avgdl));
        }
        return s;
      });
    }
    if (method === "lsa") {
      const qv = new Array(ix.vocab.length).fill(0);
      for (const [j, c] of qc) qv[j] = c * ix.idf[j];
      const qn = Rag.normalize(qv);
      const emb = Rag.normalize(Array.from({ length: ix.k }, (_, d) => qn.reduce((s, v, j) => (v ? s + v * ix.V[j][d] : s), 0)));
      return ix.chunk_emb.map(e => e.reduce((s, v, d) => s + v * emb[d], 0));
    }
    if (method === "hybrid") {
      const fused = new Array(this.N).fill(0);
      for (const m of ["bm25", "lsa"]) this.order(this.score(q, m)).forEach((i, r) => (fused[i] += 1 / (60 + r + 1)));
      return fused;
    }
    throw new Error(method);
  }
  order(scores) { return scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]); }   // stable sort, like NumPy's
  search(q, method, top = 5) { const s = this.score(q, method); return this.order(s).slice(0, top).map(i => [i, s[i]]); }

  answer(q, method) {
    const hits = this.search(q, method, 3), qt = new Set(tokenize(q));
    let best = null;
    hits.forEach(([i], rank) => {
      for (const s of sentences(this.ix.chunks[i].text)) {
        const st = new Set(tokenize(s));
        let score = -0.01 * rank;
        for (const t of qt) if (st.has(t) && this.vid.has(t)) score += this.ix.bm25_idf[this.vid.get(t)];
        if (!best || score > best.score + 1e-9) best = { score, sentence: s, chunk: i };
      }
    });
    return { ...best, hits };
  }
}

if (typeof module !== "undefined") module.exports = { Rag, tokenize, stem, sentences };
