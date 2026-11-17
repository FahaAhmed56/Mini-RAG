const $ = id => document.getElementById(id);
const rag = new Rag(INDEX);
const NAMES = { keyword: "Keyword match", tfidf: "TF-IDF", bm25: "BM25", lsa: "LSA embeddings", hybrid: "Hybrid (BM25+LSA)" };
const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

$("nChunks").textContent = `${INDEX.chunks.length} chunks`;
$("chips").innerHTML = ["How big is a two bedroom apartment?", ...INDEX.examples].map(q => `<button class="chip">${esc(q)}</button>`).join("");
$("chips").addEventListener("click", e => { if (e.target.classList.contains("chip")) { $("q").value = e.target.textContent; ask(); } });
$("ask").addEventListener("submit", e => { e.preventDefault(); ask(); });
$("method").addEventListener("change", () => $("q").value.trim() && ask());

// highlight the question's words inside a passage
function highlight(text, qTokens) {
  return esc(text).replace(/[A-Za-z0-9]+/g, w => (qTokens.has(stem(w.toLowerCase())) && !STOPWORDS.has(w.toLowerCase()) ? `<mark>${w}</mark>` : w));
}

function ask() {
  const q = $("q").value.trim(); if (!q) return;
  const method = $("method").value, toks = tokenize(q), qset = new Set(toks);
  const { sentence, chunk, hits } = rag.answer(q, method);
  const top = Math.max(...hits.map(h => h[1])) || 1;

  $("answerCard").hidden = $("sourcesCard").hidden = false;
  if (!hits.length || hits[0][1] <= 0) {
    $("answer").innerHTML = "I couldn't find anything about that in the knowledge base.";
    $("cite").textContent = "";
  } else {
    $("answer").innerHTML = highlight(sentence, qset);
    $("cite").innerHTML = `Source: <b>${esc(INDEX.chunks[chunk].title)}</b>`;
    // low-confidence warning: most of the question's words are unknown to the knowledge base
    const known = toks.filter(t => rag.vid.has(t)).length;
    if (toks.length >= 2 && known / toks.length < 0.5)
      $("cite").innerHTML += `<span class="warn">⚠ Only ${known} of your ${toks.length} key words appear in the knowledge base, so this answer may be wrong. Try rephrasing.</span>`;
  }
  $("methodName").textContent = `· ${NAMES[method]}`;
  const matches = rag.search(q, method, 5).filter(([, s]) => s > 0);       // only passages that actually match
  $("sources").innerHTML = (matches.length ? "" : "<li>No passage matched any word of the question.</li>") + matches.map(([i, s]) => `
    <li>
      <div class="src-head"><b>${esc(INDEX.chunks[i].title)}</b><span class="score"><span class="bar" style="width:${Math.max(0, s / top) * 100}%"></span>${s.toFixed(3)}</span></div>
      <p>${highlight(INDEX.chunks[i].text, qset)}</p>
    </li>`).join("");

  $("tokens").innerHTML = toks.length ? toks.map(t => `<span class="tok ${rag.vid.has(t) ? "" : "unknown"}" title="${rag.vid.has(t) ? "in the knowledge base" : "not in the knowledge base: ignored"}">${esc(t)}</span>`).join("") : "–";
  $("mName").textContent = NAMES[method];

  const ref = INDEX.chunks[rag.search(q, method, 1)[0][0]].doc;
  $("compare").innerHTML = "<tr><th>Method</th><th>Top source</th></tr>" + Object.keys(NAMES).map(m => {
    const d = INDEX.chunks[rag.search(q, m, 1)[0][0]];
    return `<tr class="${m === method ? "sel" : ""}"><td>${NAMES[m]}</td><td class="${d.doc === ref ? "" : "diff"}">${esc(d.title)}</td></tr>`;
  }).join("");
}

$("evalTable").innerHTML = "<tr><th>Method</th><th>#1 right</th><th>Top 3</th><th>Paraphrase #1</th></tr>" +
  Object.keys(NAMES).map(m => {
    const r = INDEX.results[m];
    return `<tr><td>${NAMES[m]}</td><td>${Math.round(r.all["recall@1"] * 100)}%</td><td>${Math.round(r.all["recall@3"] * 100)}%</td><td>${Math.round(r.paraphrase["recall@1"] * 100)}%</td></tr>`;
  }).join("");
