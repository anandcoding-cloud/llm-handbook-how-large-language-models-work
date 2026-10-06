// MIT License. See LICENSE-MIT-code.txt in this repository.
// TinyGPT: exact forward, backward (manual), gradient check, AdamW step.
const T = 3, D = 4, H = 2, DH = 2, HID = 8, V = 6;
const vocab = ["The", "cat", "sat", "dog", "ran", "END"];
const ids = [0, 1, 2];
const targets = [1, 2, 5];

// ---------- matrix helpers ----------
const zeros = (r, c) => Array.from({ length: r }, () => Array(c).fill(0));
const mm = (A, B) => { const r = A.length, k = B.length, c = B[0].length; const O = zeros(r, c); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) { let s = 0; for (let t = 0; t < k; t++) s += A[i][t] * B[t][j]; O[i][j] = s; } return O; };
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
const mul = (A, B) => A.map((r, i) => r.map((v, j) => v * B[i][j]));
const scale = (A, s) => A.map(r => r.map(v => v * s));
const clone = A => A.map(r => r.slice());
const cols = (A, a, b) => A.map(r => r.slice(a, b));
const hcat = (A, B) => A.map((r, i) => r.concat(B[i]));

// ---------- parameters ----------
function initParams() {
  const lcg = (() => { let s = 12345; return () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; }; })();
  const r1 = () => Math.round((lcg() * 0.8 - 0.4) * 10) / 10; // one decimal in [-0.4, 0.4]
  const rm = (r, c) => Array.from({ length: r }, () => Array.from({ length: c }, r1));
  return {
    E: [[0.2, 0.6, 0.4, 0.1], [0.7, 0.1, 0.2, 0.8], [0.5, 0.8, 0.9, 0.2], [0.9, 0.2, 0.4, 0.6], [0.3, 0.4, 0.7, 0.1], [0.1, 0.9, 0.3, 0.5]],
    P: [[0.1, 0.0, 0.2, 0.1], [0.0, 0.3, 0.1, 0.0], [0.2, 0.1, 0.0, 0.4]],
    WQ1: [[0.4, 0.1], [0.3, -0.2], [-0.2, 0.6], [0.5, 0.2]],
    WK1: [[0.3, -0.1], [0.2, 0.4], [-0.3, 0.2], [0.1, 0.5]],
    WV1: [[0.5, 0.1], [-0.2, 0.3], [0.4, -0.1], [0.2, 0.6]],
    WQ2: [[-0.3, 0.2], [0.5, 0.1], [0.1, -0.4], [0.2, 0.3]],
    WK2: [[0.4, 0.3], [-0.1, 0.2], [0.3, -0.2], [0.2, 0.1]],
    WV2: [[0.2, -0.3], [0.4, 0.5], [-0.1, 0.2], [0.3, 0.1]],
    WO: [[0.2, 0.1, -0.1, 0.3], [0.5, -0.2, 0.4, 0.1], [-0.3, 0.6, 0.2, -0.2], [0.1, 0.2, 0.3, 0.4]],
    g1: [[1, 1, 1, 1]], b1: [[0, 0, 0, 0]],
    W1: rm(D, HID), W2: rm(HID, D),
    g2: [[1, 1, 1, 1]], b2: [[0, 0, 0, 0]],
    Wh: rm(D, V),
  };
}
const names = ["E", "P", "WQ1", "WK1", "WV1", "WQ2", "WK2", "WV2", "WO", "g1", "b1", "W1", "W2", "g2", "b2", "Wh"];

// ---------- ops ----------
const EPS = 1e-5;
function lnFwd(X, g, b) { const Y = [], cache = []; X.forEach(row => { const mu = row.reduce((a, v) => a + v, 0) / row.length; const va = row.reduce((a, v) => a + (v - mu) ** 2, 0) / row.length; const sd = Math.sqrt(va + EPS); const xh = row.map(v => (v - mu) / sd); Y.push(xh.map((v, j) => v * g[0][j] + b[0][j])); cache.push({ xh, sd, mu }); }); return { Y, cache }; }
function lnBwd(dY, cache, g) { const dX = [], dg = [Array(g[0].length).fill(0)], db = [Array(g[0].length).fill(0)]; dY.forEach((dy, i) => { const { xh, sd } = cache[i]; const n = dy.length; const dxh = dy.map((v, j) => v * g[0][j]); const m1 = dxh.reduce((a, v) => a + v, 0) / n; const m2 = dxh.reduce((a, v, j) => a + v * xh[j], 0) / n; dX.push(dxh.map((v, j) => (v - m1 - xh[j] * m2) / sd)); dy.forEach((v, j) => { dg[0][j] += v * xh[j]; db[0][j] += v; }); }); return { dX, dg, db }; }
const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
const geluD = x => { const c = Math.sqrt(2 / Math.PI); const u = c * (x + 0.044715 * x ** 3); const th = Math.tanh(u); return 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * c * (1 + 3 * 0.044715 * x * x); };
function softmaxRows(S) { return S.map(row => { const m = Math.max(...row.filter(v => v > -Infinity)); const e = row.map(v => (v === -Infinity ? 0 : Math.exp(v - m))); const s = e.reduce((a, v) => a + v, 0); return e.map(v => v / s); }); }

function forward(p) {
  const c = {};
  c.tokE = ids.map(i => p.E[i].slice());
  c.posE = p.P.map(r => r.slice());
  c.X = add(c.tokE, c.posE);
  c.heads = [];
  for (const h of [1, 2]) {
    const Q = mm(c.X, p["WQ" + h]), K = mm(c.X, p["WK" + h]), Vv = mm(c.X, p["WV" + h]);
    const raw = mm(Q, tr(K));
    const scaled = scale(raw, 1 / Math.sqrt(DH));
    const masked = scaled.map((row, i) => row.map((v, j) => (j > i ? -Infinity : v)));
    const A = softmaxRows(masked);
    const O = mm(A, Vv);
    c.heads.push({ Q, K, V: Vv, raw, scaled, masked, A, O });
  }
  c.C = hcat(c.heads[0].O, c.heads[1].O);
  c.Attn = mm(c.C, p.WO);
  c.R1 = add(c.X, c.Attn);
  const ln1 = lnFwd(c.R1, p.g1, p.b1); c.H1 = ln1.Y; c.ln1c = ln1.cache;
  c.Z = mm(c.H1, p.W1);
  c.G = c.Z.map(r => r.map(gelu));
  c.F = mm(c.G, p.W2);
  c.R2 = add(c.H1, c.F);
  const ln2 = lnFwd(c.R2, p.g2, p.b2); c.H2 = ln2.Y; c.ln2c = ln2.cache;
  c.logits = mm(c.H2, p.Wh);
  c.probs = softmaxRows(c.logits);
  c.losses = c.probs.map((row, i) => -Math.log(row[targets[i]]));
  c.loss = c.losses.reduce((a, v) => a + v, 0) / T;
  return c;
}

function backward(p, c) {
  const g = {};
  const dlog = c.probs.map((row, i) => row.map((v, j) => (v - (j === targets[i] ? 1 : 0)) / T));
  c.dlogits = dlog;
  g.Wh = mm(tr(c.H2), dlog);
  const dH2 = mm(dlog, tr(p.Wh));
  c.dH2 = dH2;
  const l2 = lnBwd(dH2, c.ln2c, p.g2); g.g2 = l2.dg; g.b2 = l2.db;
  const dR2 = l2.dX; c.dR2 = dR2;
  let dH1 = clone(dR2);
  const dF = dR2;
  g.W2 = mm(tr(c.G), dF);
  const dG = mm(dF, tr(p.W2));
  const dZ = dG.map((r, i) => r.map((v, j) => v * geluD(c.Z[i][j])));
  g.W1 = mm(tr(c.H1), dZ);
  dH1 = add(dH1, mm(dZ, tr(p.W1)));
  c.dH1 = dH1;
  const l1 = lnBwd(dH1, c.ln1c, p.g1); g.g1 = l1.dg; g.b1 = l1.db;
  const dR1 = l1.dX; c.dR1 = dR1;
  let dX = clone(dR1);
  const dAttn = dR1;
  g.WO = mm(tr(c.C), dAttn);
  const dC = mm(dAttn, tr(p.WO)); c.dC = dC;
  [1, 2].forEach((h, hi) => {
    const hd = c.heads[hi];
    const dO = cols(dC, hi * DH, hi * DH + DH);
    const dA = mm(dO, tr(hd.V));
    const dV = mm(tr(hd.A), dO);
    const dS = hd.A.map((arow, i) => { const dot = arow.reduce((a, v, j) => a + v * dA[i][j], 0); return arow.map((v, j) => v * (dA[i][j] - dot)); });
    const dRaw = scale(dS, 1 / Math.sqrt(DH));
    const dQ = mm(dRaw, hd.K);
    const dK = mm(tr(dRaw), hd.Q);
    g["WQ" + h] = mm(tr(c.X), dQ); g["WK" + h] = mm(tr(c.X), dK); g["WV" + h] = mm(tr(c.X), dV);
    dX = add(dX, mm(dQ, tr(p["WQ" + h]))); dX = add(dX, mm(dK, tr(p["WK" + h]))); dX = add(dX, mm(dV, tr(p["WV" + h])));
    hd.dA = dA; hd.dS = dS; hd.dQ = dQ; hd.dK = dK; hd.dV = dV;
  });
  c.dX = dX;
  g.P = clone(dX);
  g.E = zeros(V, D);
  ids.forEach((id, i) => dX[i].forEach((v, j) => { g.E[id][j] += v; }));
  return g;
}

function gradCheck(p) {
  let maxRel = 0, worst = "";
  const c0 = forward(p); const g = backward(p, c0);
  for (const n of names) {
    for (let i = 0; i < p[n].length; i++) for (let j = 0; j < p[n][i].length; j++) {
      const h = 1e-5; const old = p[n][i][j];
      p[n][i][j] = old + h; const lp = forward(p).loss;
      p[n][i][j] = old - h; const lm = forward(p).loss;
      p[n][i][j] = old;
      const num = (lp - lm) / (2 * h); const an = g[n][i][j];
      const rel = Math.abs(num - an) / Math.max(1e-8, Math.abs(num) + Math.abs(an));
      if (rel > maxRel && Math.abs(num) + Math.abs(an) > 1e-7) { maxRel = rel; worst = `${n}[${i}][${j}] num=${num} an=${an}`; }
    }
  }
  return { maxRel, worst };
}

function adamwStep(p, g, state, step, lr = 0.05, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01) {
  const info = {};
  for (const n of names) {
    state[n] = state[n] || { m: zeros(p[n].length, p[n][0].length), v: zeros(p[n].length, p[n][0].length) };
    for (let i = 0; i < p[n].length; i++) for (let j = 0; j < p[n][i].length; j++) {
      const gr = g[n][i][j];
      const m = b1 * state[n].m[i][j] + (1 - b1) * gr;
      const v = b2 * state[n].v[i][j] + (1 - b2) * gr * gr;
      state[n].m[i][j] = m; state[n].v[i][j] = v;
      const mh = m / (1 - b1 ** step), vh = v / (1 - b2 ** step);
      const old = p[n][i][j];
      const nw = old - lr * mh / (Math.sqrt(vh) + eps) - lr * wd * old;
      info[`${n}[${i}][${j}]`] = { old, gr, m, v, mh, vh, nw };
      p[n][i][j] = nw;
    }
  }
  return info;
}

module.exports = { T, D, H, DH, HID, V, vocab, ids, targets, initParams, names, forward, backward, gradCheck, adamwStep, clone, mm, tr };

if (require.main === module) {
  const p = initParams();
  const c = forward(p);
  const f = (M, d = 3) => M.map(r => r.map(v => (v === -Infinity ? "-inf" : v.toFixed(d))).join("  ")).join("\n");
  console.log("X\n" + f(c.X));
  c.heads.forEach((h, i) => { console.log("Head", i + 1, "\nQ\n" + f(h.Q), "\nK\n" + f(h.K), "\nV\n" + f(h.V), "\nraw\n" + f(h.raw), "\nA\n" + f(h.A), "\nO\n" + f(h.O)); });
  console.log("C\n" + f(c.C), "\nAttn\n" + f(c.Attn), "\nR1\n" + f(c.R1), "\nH1\n" + f(c.H1), "\nZ\n" + f(c.Z), "\nG\n" + f(c.G), "\nF\n" + f(c.F), "\nR2\n" + f(c.R2), "\nH2\n" + f(c.H2), "\nlogits\n" + f(c.logits), "\nprobs\n" + f(c.probs));
  console.log("losses", c.losses.map(v => v.toFixed(4)), "loss", c.loss.toFixed(4));
  console.log("W1\n" + f(p.W1, 1), "\nW2\n" + f(p.W2, 1), "\nWh\n" + f(p.Wh, 1));
  const gc = gradCheck(p); console.log("gradcheck", gc);
  const g = backward(p, c);
  for (const n of names) { const a = g[n].flat(); console.log(n, "norm", Math.sqrt(a.reduce((s, v) => s + v * v, 0)).toExponential(3)); }
  const st = {}; const info = adamwStep(p, g, st, 1);
  const c2 = forward(p); console.log("loss after", c2.loss.toFixed(4), c2.losses.map(v => v.toFixed(4)));
}
