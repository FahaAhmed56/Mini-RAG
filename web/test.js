// Run with: node web/test.js
const fs = require("fs"), path = require("path"), vm = require("vm"), assert = require("assert");
const { Rag, tokenize } = require("./rag.js");
const ctx = {}; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, "index_data.js"), "utf8") + "\nthis.I = INDEX;", ctx);
// copy the index out of the vm sandbox so arrays share prototypes with this file
const rag = new Rag(JSON.parse(JSON.stringify(ctx.I)));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "test_fixtures.json"), "utf8"));
let passed = 0; const test = (n, f) => { f(); passed++; console.log("  ✓ " + n); };

test("tokenizer + stemmer match Python on all 40 questions", () => fx.forEach(f => assert.deepStrictEqual(tokenize(f.q), f.tokens, f.q)));
for (const m of ["keyword", "tfidf", "bm25", "lsa", "hybrid"])
  test(`${m}: same top-3 chunks as Python for all 40 questions`, () =>
    fx.forEach(f => assert.deepStrictEqual(rag.search(f.q, m, 3).map(h => h[0]), f.top[m], `${m}: ${f.q}`)));
test("extractive answer sentence matches Python for all 40 questions", () =>
  fx.forEach(f => assert.strictEqual(rag.answer(f.q, "hybrid").sentence, f.answer, f.q)));
test("unknown words are ignored, not crashing", () => assert.strictEqual(rag.search("zzzz qqqq", "bm25", 1).length, 1));
console.log(`\nAll ${passed} tests passed ✓`);
