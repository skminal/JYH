/* ML for Engineers 双语精讲 — interactive demos.
   The pure maths helpers at the top are exported for node tests (see test-demos.js);
   everything after `if (typeof document === 'undefined')` drives the page. */
(function () {
  'use strict';

  // ---------------------------------------------------------------- pure helpers
  function mulberry32(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) { // Box–Muller
    let spare = null;
    return function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = rand();
      while (u <= 1e-12) u = rand();
      const v = rand();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
  }

  function erf(x) { // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    const sgn = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return sgn * y;
  }

  function cholesky(A, n) { // A row-major n×n, symmetric positive definite → lower L, or null
    const L = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * n + j];
        for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) {
          if (!(s > 0)) return null;
          L[i * n + i] = Math.sqrt(s);
        } else {
          L[i * n + j] = s / L[j * n + j];
        }
      }
    }
    return L;
  }

  function cholSolve(L, n, b) { // solve (L Lᵀ) x = b
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
      y[i] = s / L[i * n + i];
    }
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
      x[i] = s / L[i * n + i];
    }
    return x;
  }

  function traceInv(L, n) { // tr((L Lᵀ)⁻¹) = ‖L⁻¹‖²_F
    let tr = 0;
    const col = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      for (let i = j; i < n; i++) {
        let s = i === j ? 1 : 0;
        for (let k = j; k < i; k++) s -= L[i * n + k] * col[k];
        col[i] = s / L[i * n + i];
        tr += col[i] * col[i];
      }
    }
    return tr;
  }

  function kernel(kind, x, y, ell) {
    if (kind === 'lin') return 1 + x * y; // φ(x) = (1, x)ᵀ → ordinary ridge regression
    const r = Math.abs(x - y);
    switch (kind) {
      case 'gauss': return Math.exp(-r * r / (2 * ell * ell));
      case 'm12': return Math.exp(-r / ell);
      case 'm32': { const q = Math.sqrt(3) * r / ell; return (1 + q) * Math.exp(-q); }
      case 'm52': { const q = Math.sqrt(5) * r / ell; return (1 + q + q * q / 3) * Math.exp(-q); }
      default: return 0;
    }
  }

  // Kernel ridge regression: α = (K + λI)⁻¹ y,  f(x) = k(x)ᵀ α
  function krrFit(xs, ys, kind, ell, lam) {
    const n = xs.length;
    const A = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) A[i * n + j] = kernel(kind, xs[i], xs[j], ell) + (i === j ? lam : 0);
    }
    let L = cholesky(A, n);
    let jitter = 1e-10;
    while (!L && jitter < 10) {
      for (let i = 0; i < n; i++) A[i * n + i] += jitter;
      L = cholesky(A, n);
      jitter *= 10;
    }
    const alpha = cholSolve(L, n, ys);
    function predict(x) {
      let s = 0;
      for (let i = 0; i < n; i++) s += alpha[i] * kernel(kind, x, xs[i], ell);
      return s;
    }
    // effective degrees of freedom of the linear smoother S = K(K + λI)⁻¹: tr S = n − λ tr((K + λI)⁻¹)
    function dof() { return n - lam * traceInv(L, n); }
    return { alpha, predict, dof };
  }

  // One-hidden-layer MLP with scalar input/output, exactly the network on the slides:
  // a = W x + b,  z = σ(a),  ŷ = vᵀz + c,  ℓ = ½(ŷ − y)²
  function act(kind) {
    if (kind === 'relu') return { f: a => (a > 0 ? a : 0), df: a => (a > 0 ? 1 : 0) };
    return { f: Math.tanh, df: a => { const t = Math.tanh(a); return 1 - t * t; } };
  }

  function mlpCreate(h, kind, rand) {
    const g = gaussian(rand);
    const net = { h, kind, W: new Float64Array(h), b: new Float64Array(h), v: new Float64Array(h), c: 0 };
    for (let j = 0; j < h; j++) {
      net.W[j] = g() * 2.5;
      net.b[j] = g();
      net.v[j] = g() / Math.sqrt(h);
    }
    return net;
  }

  function mlpPredict(net, x) {
    const A = act(net.kind);
    let y = net.c;
    for (let j = 0; j < net.h; j++) y += net.v[j] * A.f(net.W[j] * x + net.b[j]);
    return y;
  }

  function zeroGrad(h) {
    return { W: new Float64Array(h), b: new Float64Array(h), v: new Float64Array(h), c: 0 };
  }

  // Forward pass + backpropagation for one sample; adds the gradients into `grad`, returns the loss.
  function mlpBackprop(net, x, y, grad) {
    const A = act(net.kind), h = net.h;
    const a = new Float64Array(h), z = new Float64Array(h);
    let yhat = net.c;
    for (let j = 0; j < h; j++) { // forward pass
      a[j] = net.W[j] * x + net.b[j];
      z[j] = A.f(a[j]);
      yhat += net.v[j] * z[j];
    }
    const dOut = yhat - y; // δ(o) = ŷ − y
    for (let j = 0; j < h; j++) {
      const dH = dOut * net.v[j] * A.df(a[j]); // δ(h) = (δ(o) v) ⊙ σ'(a)
      grad.v[j] += dOut * z[j]; // ∂ℓ/∂v = δ(o) z
      grad.W[j] += dH * x; // ∂ℓ/∂W = δ(h) xᵀ
      grad.b[j] += dH; // ∂ℓ/∂b = δ(h)
    }
    grad.c += dOut; // ∂ℓ/∂c = δ(o)
    return 0.5 * dOut * dOut;
  }

  // One mini-batch SGD update: θ ← θ − η · (1/|B|) Σ_{i∈B} ∇ℓ_i
  function sgdStep(net, X, Y, idx, lr) {
    const g = zeroGrad(net.h);
    let loss = 0;
    for (const i of idx) loss += mlpBackprop(net, X[i], Y[i], g);
    const s = lr / idx.length;
    for (let j = 0; j < net.h; j++) {
      net.W[j] -= s * g.W[j];
      net.b[j] -= s * g.b[j];
      net.v[j] -= s * g.v[j];
    }
    net.c -= s * g.c;
    return loss / idx.length;
  }

  function mlpLoss(net, X, Y) {
    let s = 0;
    for (let i = 0; i < X.length; i++) {
      const e = mlpPredict(net, X[i]) - Y[i];
      s += 0.5 * e * e;
    }
    return s / X.length;
  }

  // ---- Part 3: parametric regression helpers
  function cholJitter(A, n) {
    let L = cholesky(A, n), jitter = 1e-12;
    while (!L && jitter < 1e3) { for (let i = 0; i < n; i++) A[i * n + i] += jitter; L = cholesky(A, n); jitter *= 10; }
    return L;
  }
  function forwardSolve(L, n, b) { // L v = b
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * v[k]; v[i] = s / L[i * n + i]; }
    return v;
  }
  function backSolveT(L, n, b) { // Lᵀ v = b
    const v = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * v[k]; v[i] = s / L[i * n + i]; }
    return v;
  }
  // Ridge (λ > 0) or OLS (λ → 0) on design matrix rows Phi: β = (ΦᵀΦ + λI)⁻¹Φᵀy, with hat-matrix diagonal H_nn
  function ridgeFit(Phi, y, lam) {
    const N = Phi.length, p = Phi[0].length;
    const A = new Float64Array(p * p), b = new Float64Array(p);
    for (let n = 0; n < N; n++) {
      const r = Phi[n];
      for (let i = 0; i < p; i++) { b[i] += r[i] * y[n]; for (let j = 0; j < p; j++) A[i * p + j] += r[i] * r[j]; }
    }
    for (let i = 0; i < p; i++) A[i * p + i] += lam;
    const L = cholJitter(A, p);
    const beta = cholSolve(L, p, b);
    const yhat = Phi.map(r => r.reduce((acc, v, i) => acc + v * beta[i], 0));
    const H = Phi.map(r => { const v = forwardSolve(L, p, r); return v.reduce((acc, t) => acc + t * t, 0); }); // H_nn = φₙᵀ A⁻¹ φₙ
    return { beta, yhat, H, trH: H.reduce((acc, v) => acc + v, 0) };
  }
  // Training error, exact LOOCV via the hat matrix, and GCV
  function cvScores(fit, y) {
    const N = y.length;
    let se = 0, loo = 0;
    for (let n = 0; n < N; n++) { const r = y[n] - fit.yhat[n], d = 1 - fit.H[n]; se += r * r; loo += (r / d) * (r / d); }
    const den = 1 - fit.trH / N;
    return { train: se / N, loocv: loo / N, gcv: den > 1e-6 ? (se / N) / (den * den) : Infinity };
  }
  // Local regression at x0: Nadaraya–Watson (p = 0) and local linear (p = 1) with kernel weights
  function kweight(kind, d, h) {
    const u = Math.abs(d) / h;
    if (kind === 'tricube') return u < 1 ? Math.pow(1 - u * u * u, 3) : 0;
    return Math.exp(-u * u / 2);
  }
  function localFit(xs, ys, x0, h, kind) {
    let S0 = 0, S1 = 0, S2 = 0, T0 = 0, T1 = 0;
    for (let n = 0; n < xs.length; n++) {
      const d = xs[n] - x0, w = kweight(kind, d, h);
      S0 += w; S1 += w * d; S2 += w * d * d; T0 += w * ys[n]; T1 += w * d * ys[n];
    }
    const det = S0 * S2 - S1 * S1, ok = det > 1e-9 * S0 * S2 && S0 > 1e-12;
    return { nw: S0 > 1e-12 ? T0 / S0 : NaN, ll: ok ? (S2 * T0 - S1 * T1) / det : NaN, slope: ok ? (S0 * T1 - S1 * T0) / det : NaN, S0 };
  }
  // Bayesian linear regression: prior β ~ N(0, α⁻¹I), noise precision η.
  // Posterior N(m, S), S = A⁻¹, A = αI + ηΦᵀΦ, m = ηSΦᵀy; log evidence log p(y | α, η)
  function bayesFit(Phi, y, alpha, eta) {
    const N = Phi.length, p = Phi[0].length;
    const A = new Float64Array(p * p), b = new Float64Array(p);
    for (let n = 0; n < N; n++) {
      const r = Phi[n];
      for (let i = 0; i < p; i++) { b[i] += eta * r[i] * y[n]; for (let j = 0; j < p; j++) A[i * p + j] += eta * r[i] * r[j]; }
    }
    for (let i = 0; i < p; i++) A[i * p + i] += alpha;
    const L = cholJitter(A, p);
    const m = cholSolve(L, p, b);
    let sse = 0, mm = 0, logdetA = 0;
    for (let n = 0; n < N; n++) { const r = y[n] - Phi[n].reduce((acc, v, i) => acc + v * m[i], 0); sse += r * r; }
    for (let i = 0; i < p; i++) { mm += m[i] * m[i]; logdetA += 2 * Math.log(L[i * p + i]); }
    const gamma = p - alpha * traceInv(L, p); // effective number of parameters = Σ ηλᵢ/(α + ηλᵢ)
    const logEv = p / 2 * Math.log(alpha) + N / 2 * Math.log(eta) - eta / 2 * sse - alpha / 2 * mm - logdetA / 2 - N / 2 * Math.log(2 * Math.PI);
    return { m, L, p, sse, mm, gamma, logEv };
  }
  function predVar(fit, phi, eta) { const v = forwardSolve(fit.L, fit.p, phi); return 1 / eta + v.reduce((acc, t) => acc + t * t, 0); }
  // Type-II maximum likelihood: α ← γ / mᵀm, η ← (N − γ) / ‖y − Φm‖², iterated to a fixed point
  function evidenceOpt(Phi, y, alpha, eta, maxIter) {
    let k = 0;
    for (; k < (maxIter || 500); k++) {
      const f = bayesFit(Phi, y, alpha, eta);
      const a2 = f.gamma / f.mm, e2 = (y.length - f.gamma) / f.sse;
      const done = Math.abs(Math.log(a2 / alpha)) < 1e-9 && Math.abs(Math.log(e2 / eta)) < 1e-9;
      alpha = a2; eta = e2;
      if (done) break;
    }
    return { alpha, eta, iters: k + 1 };
  }

  // ---- Part 5: classification helpers
  const sigmoid = t => (t >= 0 ? 1 / (1 + Math.exp(-t)) : Math.exp(t) / (1 + Math.exp(t)));
  // Logistic regression with augmented inputs x̃ = (1, x₁, x₂); labels y ∈ {0, 1}
  function logisticNLL(X, y, th) { // −ℒ(θ) = −Σ [yᵢ θᵀxᵢ − log(1 + e^{θᵀxᵢ})]
    let s = 0;
    for (let i = 0; i < X.length; i++) {
      const t = X[i].reduce((a, v, k) => a + v * th[k], 0);
      s += (t > 0 ? t + Math.log1p(Math.exp(-t)) : Math.log1p(Math.exp(t))) - y[i] * t;
    }
    return s;
  }
  // One Newton–Raphson (IRLS) step: θ ← θ − (XᵀRX)⁻¹ Xᵀ(z − y), z = σ(Xθ), R = diag(z(1 − z))
  function newtonStep(X, y, th) {
    const p = th.length, g = new Float64Array(p), H = new Float64Array(p * p);
    for (let i = 0; i < X.length; i++) {
      const xi = X[i], z = sigmoid(xi.reduce((a, v, k) => a + v * th[k], 0)), r = z * (1 - z);
      for (let a = 0; a < p; a++) { g[a] += xi[a] * (z - y[i]); for (let b = 0; b < p; b++) H[a * p + b] += r * xi[a] * xi[b]; }
    }
    const L = cholJitter(H, p);
    const d = cholSolve(L, p, g);
    return { th: th.map((v, k) => v - d[k]), grad: g };
  }
  // SVM dual by SMO (Platt): max Σα − ½ΣΣ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ), 0 ≤ α ≤ C, Σαᵢyᵢ = 0; labels y ∈ {−1, +1}
  function svmTrain(K, y, C, opts) {
    opts = opts || {};
    const n = y.length, tol = opts.tol || 1e-4 / Math.max(1, C), rand = mulberry32(opts.seed || 1); // primal error ≈ C·Σξ, so the KKT tolerance must shrink as C grows
    const alpha = new Float64Array(n), f = new Float64Array(n); // f = Σ αⱼyⱼK(xⱼ, ·) + b at training points
    let b = 0, passes = 0, steps = 0;
    const kkt = i => { const g = y[i] * f[i]; return (alpha[i] < C - 1e-12 && g < 1 - tol) || (alpha[i] > 1e-12 && g > 1 + tol); };
    function takeStep(i, j) {
      if (i === j) return false;
      const Ei = f[i] - y[i], Ej = f[j] - y[j], ai = alpha[i], aj = alpha[j];
      let L, H;
      if (y[i] !== y[j]) { L = Math.max(0, aj - ai); H = Math.min(C, C + aj - ai); } else { L = Math.max(0, ai + aj - C); H = Math.min(C, ai + aj); }
      if (H - L < 1e-12) return false;
      const eta = 2 * K[i * n + j] - K[i * n + i] - K[j * n + j];
      if (eta >= -1e-12) return false;
      let ajn = Math.min(H, Math.max(L, aj - y[j] * (Ei - Ej) / eta));
      if (Math.abs(ajn - aj) < 1e-10 * (ajn + aj + 1e-10)) return false;
      const ain = ai + y[i] * y[j] * (aj - ajn);
      const b1 = b - Ei - y[i] * (ain - ai) * K[i * n + i] - y[j] * (ajn - aj) * K[i * n + j];
      const b2 = b - Ej - y[i] * (ain - ai) * K[i * n + j] - y[j] * (ajn - aj) * K[j * n + j];
      const bn = ain > 1e-12 && ain < C - 1e-12 ? b1 : ajn > 1e-12 && ajn < C - 1e-12 ? b2 : (b1 + b2) / 2;
      const di = y[i] * (ain - ai), dj = y[j] * (ajn - aj), db = bn - b;
      for (let k = 0; k < n; k++) f[k] += di * K[i * n + k] + dj * K[j * n + k] + db;
      alpha[i] = ain; alpha[j] = ajn; b = bn; steps++;
      return true;
    }
    const maxSteps = opts.maxSteps || 200000;
    while (passes < 5 && steps < maxSteps) {
      let changed = 0;
      for (let i = 0; i < n && steps < maxSteps; i++) {
        if (!kkt(i)) continue;
        const Ei = f[i] - y[i];
        let best = -1, gap = -1;
        for (let j = 0; j < n; j++) { const d = Math.abs(Ei - (f[j] - y[j])); if (j !== i && d > gap) { gap = d; best = j; } }
        if (takeStep(i, best)) { changed++; continue; }
        const start = Math.floor(rand() * n);
        for (let t = 0; t < n; t++) { if (takeStep(i, (start + t) % n)) { changed++; break; } }
      }
      passes = changed === 0 ? passes + 1 : 0;
    }
    let dual = 0, wnorm2 = 0, hinge = 0;
    for (let i = 0; i < n; i++) { dual += alpha[i]; for (let j = 0; j < n; j++) wnorm2 += alpha[i] * alpha[j] * y[i] * y[j] * K[i * n + j]; }
    dual -= wnorm2 / 2;
    for (let i = 0; i < n; i++) hinge += Math.max(0, 1 - y[i] * f[i]);
    return { alpha, b, f, steps, dual, primal: wnorm2 / 2 + C * hinge, wnorm2 };
  }

  const api = { mulberry32, gaussian, erf, cholesky, cholSolve, traceInv, kernel, krrFit, act, mlpCreate, mlpPredict, zeroGrad, mlpBackprop, sgdStep, mlpLoss,
    sigmoid, logisticNLL, newtonStep, svmTrain,
    ridgeFit, cvScores, localFit, kweight, bayesFit, predVar, evidenceOpt, forwardSolve, backSolveT };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof document === 'undefined') return;

  // ---------------------------------------------------------------- page plumbing
  const $ = id => document.getElementById(id);
  const reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const MONO = '11px "IBM Plex Mono", ui-monospace, Menlo, monospace';
  const SANS = '12px "IBM Plex Sans", "Noto Sans SC", "PingFang SC", sans-serif';

  let C = {};
  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const g = n => cs.getPropertyValue(n).trim();
    C = {
      ink: g('--ink'), ink2: g('--ink-2'), ink3: g('--ink-3'), rule: g('--rule'), surface: g('--surface'),
      accent: g('--accent'), c2: g('--c2'), c3: g('--c3'),
    };
  }

  const redrawers = [];
  function canvasView(canvas, draw) {
    const ctx = canvas.getContext('2d');
    const view = { canvas, ctx, w: 0, h: 0 };
    view.draw = function () {
      if (view.w <= 0) return;
      ctx.save();
      ctx.clearRect(0, 0, view.w, view.h);
      draw(view);
      ctx.restore();
    };
    function resize(drawNow) {
      const r = canvas.getBoundingClientRect();
      if (r.width < 2) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      view.w = r.width;
      view.h = r.height;
      canvas.width = Math.round(r.width * dpr);
      canvas.height = Math.round(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (drawNow !== false) view.draw();
    }
    if ('ResizeObserver' in window) new ResizeObserver(() => resize()).observe(canvas);
    else window.addEventListener('resize', () => resize());
    resize(false); // size only: each demo draws once its own state is ready
    redrawers.push(view.draw);
    return view;
  }

  function frame(view, xr, yr, m) {
    m = Object.assign({ l: 46, r: 12, t: 12, b: 30 }, m || {});
    const W = view.w, H = view.h, pw = W - m.l - m.r, ph = H - m.t - m.b;
    return {
      m, W, H, pw, ph, xr, yr,
      X: x => m.l + (x - xr[0]) / (xr[1] - xr[0]) * pw,
      Y: y => m.t + ph - (y - yr[0]) / (yr[1] - yr[0]) * ph,
      ix: px => xr[0] + (px - m.l) / pw * (xr[1] - xr[0]),
      iy: py => yr[0] + (m.t + ph - py) / ph * (yr[1] - yr[0]),
    };
  }

  function niceTicks(lo, hi, count) {
    const step0 = (hi - lo) / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const e = step0 / mag;
    const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function fmtNum(v) {
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e5 || a < 1e-3) return v.toExponential(0);
    return String(+v.toFixed(4));
  }

  const SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '.': '·' };
  const pow10 = k => '10' + String(k).split('').map(ch => SUP[ch] || ch).join('');
  function bigNum(v) {
    if (v < 1e5) return v < 10 ? v.toFixed(1) : Math.round(v).toLocaleString('en-US');
    const k = Math.floor(Math.log10(v));
    return (v / Math.pow(10, k)).toFixed(1) + '×' + pow10(k);
  }

  function axes(ctx, F, o) {
    o = o || {};
    const xt = o.xticks || niceTicks(F.xr[0], F.xr[1], Math.max(3, Math.round(F.pw / 80)));
    const yt = o.yticks || niceTicks(F.yr[0], F.yr[1], Math.max(3, Math.round(F.ph / 50)));
    const xf = o.xfmt || fmtNum, yf = o.yfmt || fmtNum;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = C.rule;
    ctx.fillStyle = C.ink3;
    ctx.font = MONO;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const t of xt) {
      const x = Math.round(F.X(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, F.m.t); ctx.lineTo(x, F.m.t + F.ph); ctx.stroke();
      ctx.fillText(xf(t), x, F.m.t + F.ph + 6);
    }
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const t of yt) {
      const y = Math.round(F.Y(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(F.m.l, y); ctx.lineTo(F.m.l + F.pw, y); ctx.stroke();
      ctx.fillText(yf(t), F.m.l - 6, y);
    }
    ctx.font = SANS;
    ctx.fillStyle = C.ink2;
    if (o.xlabel) { ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(o.xlabel, F.m.l + F.pw, F.H - 1); }
    if (o.ylabel) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.ylabel, F.m.l + 6, F.m.t + 4); }
    ctx.restore();
  }

  function clip(ctx, F) {
    ctx.beginPath();
    ctx.rect(F.m.l, F.m.t, F.pw, F.ph);
    ctx.clip();
  }

  function curve(ctx, F, xs, fn) {
    ctx.beginPath();
    xs.forEach((x, i) => { const px = F.X(x), py = F.Y(fn(x)); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
    ctx.stroke();
  }

  function poly(ctx, F, xs, ys) {
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) { const px = F.X(xs[i]), py = F.Y(ys[i]); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
    ctx.stroke();
  }

  function dots(ctx, F, xs, ys, r, color) {
    ctx.fillStyle = color;
    for (let i = 0; i < xs.length; i++) { ctx.beginPath(); ctx.arc(F.X(xs[i]), F.Y(ys[i]), r, 0, 2 * Math.PI); ctx.fill(); }
  }

  function legend(ctx, F, items) { // items: [label, color, dashed]
    ctx.save();
    ctx.font = SANS;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    let w = 0;
    for (const it of items) w = Math.max(w, ctx.measureText(it[0]).width);
    const x0 = F.m.l + F.pw - w - 34, y0 = F.m.t + 12;
    items.forEach((it, i) => {
      const y = y0 + i * 17;
      ctx.strokeStyle = it[1];
      ctx.lineWidth = 2.2;
      ctx.setLineDash(Array.isArray(it[2]) ? it[2] : (it[2] ? [5, 3] : []));
      ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + 20, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.ink2;
      ctx.fillText(it[0], x0 + 26, y);
    });
    ctx.restore();
  }

  const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));

  // ---------------------------------------------------------------- demo 1: least squares
  function demoLS() {
    const xs = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50];
    const ys = [108, 117, 128, 130, 135, 145, 159, 167, 175, 182, 190];
    const n = xs.length;
    const xb = xs.reduce((s, v) => s + v, 0) / n, yb = ys.reduce((s, v) => s + v, 0) / n;
    let sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sxy += (xs[i] - xb) * (ys[i] - yb); sxx += (xs[i] - xb) * (xs[i] - xb); }
    const aOpt = sxy / sxx, bOpt = yb - aOpt * xb;
    const E = (a, b) => { let s = 0; for (let i = 0; i < n; i++) { const r = ys[i] - a * xs[i] - b; s += r * r; } return s; };
    const Emin = E(aOpt, bOpt);
    const AR = [0, 4], BR = [40, 140];
    const aEl = $('ls-a'), bEl = $('ls-b');
    let a = +aEl.value, b = +bEl.value;

    const fit = canvasView($('ls-plot'), v => {
      const ctx = v.ctx, F = frame(v, [5, 55], [60, 240], { l: 40 });
      axes(ctx, F, { xlabel: '温度 x / °C', ylabel: '压强 y / kPa' });
      ctx.save(); clip(ctx, F);
      ctx.strokeStyle = C.c2; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
      for (let i = 0; i < n; i++) {
        ctx.beginPath(); ctx.moveTo(F.X(xs[i]), F.Y(ys[i])); ctx.lineTo(F.X(xs[i]), F.Y(a * xs[i] + b)); ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2.2;
      curve(ctx, F, [5, 55], x => a * x + b);
      dots(ctx, F, xs, ys, 3.6, C.ink);
      ctx.restore();
    });

    const GA = 64, GB = 48;
    const heat = new Float64Array(GA * GB);
    let hi = -Infinity;
    for (let i = 0; i < GA; i++) {
      for (let j = 0; j < GB; j++) {
        const aa = AR[0] + (i + 0.5) / GA * (AR[1] - AR[0]), bb = BR[0] + (j + 0.5) / GB * (BR[1] - BR[0]);
        const val = Math.log10(E(aa, bb));
        heat[i * GB + j] = val;
        hi = Math.max(hi, val);
      }
    }
    const lo = Math.log10(Emin);

    const surf = canvasView($('ls-surface'), v => {
      const ctx = v.ctx, F = frame(v, AR, BR, { l: 40 });
      const cw = F.pw / GA, ch = F.ph / GB;
      ctx.fillStyle = C.accent;
      for (let i = 0; i < GA; i++) {
        for (let j = 0; j < GB; j++) {
          const t = (heat[i * GB + j] - lo) / (hi - lo);
          ctx.globalAlpha = Math.max(0.04, Math.pow(1 - t, 2.4));
          ctx.fillRect(F.m.l + i * cw, F.m.t + F.ph - (j + 1) * ch, cw + 0.6, ch + 0.6);
        }
      }
      ctx.globalAlpha = 1;
      axes(ctx, F, { xlabel: '斜率 a', ylabel: '截距 b' });
      ctx.save(); clip(ctx, F);
      // optimum: ring
      ctx.strokeStyle = C.c2; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(F.X(aOpt), F.Y(bOpt), 7, 0, 2 * Math.PI); ctx.stroke();
      // current guess
      ctx.fillStyle = C.ink; ctx.strokeStyle = C.surface; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(F.X(a), F.Y(b), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.restore();
    });

    function update() {
      $('ls-a-out').textContent = a.toFixed(2);
      $('ls-b-out').textContent = b.toFixed(1);
      const e = E(a, b);
      $('ls-readout').innerHTML =
        '当前总平方误差 <b>E(a,b) = ' + e.toFixed(1) + ' kPa²</b>，相当于每个点平均偏差 √(E/N) ≈ ' + Math.sqrt(e / n).toFixed(2) + ' kPa' +
        '<br>最小二乘解 a* = ' + aOpt.toFixed(3) + ' kPa/°C，b* = ' + bOpt.toFixed(2) + ' kPa，此时 E* = ' + Emin.toFixed(1) + ' kPa²';
      fit.draw(); surf.draw();
    }
    function sync() { aEl.value = a; bEl.value = b; update(); }
    aEl.addEventListener('input', () => { a = +aEl.value; update(); });
    bEl.addEventListener('input', () => { b = +bEl.value; update(); });

    let dragging = false;
    const sc = $('ls-surface');
    function fromPointer(e) {
      const r = sc.getBoundingClientRect(), F = frame(surf, AR, BR, { l: 40 });
      a = Math.min(AR[1], Math.max(AR[0], F.ix(e.clientX - r.left)));
      b = Math.min(BR[1], Math.max(BR[0], F.iy(e.clientY - r.top)));
      sync();
    }
    sc.addEventListener('pointerdown', e => { dragging = true; try { sc.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ } fromPointer(e); });
    sc.addEventListener('pointermove', e => { if (dragging) fromPointer(e); });
    sc.addEventListener('pointerup', () => { dragging = false; });
    sc.addEventListener('pointercancel', () => { dragging = false; });

    function animateTo(a1, b1) {
      if (reduceMotion) { a = a1; b = b1; sync(); return; }
      const a0 = a, b0 = b, t0 = performance.now();
      function step(t) {
        const k = Math.min(1, (t - t0) / 700), e = 1 - Math.pow(1 - k, 3);
        a = a0 + (a1 - a0) * e; b = b0 + (b1 - b0) * e; sync();
        if (k < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }
    $('ls-solve').addEventListener('click', () => animateTo(aOpt, bOpt));
    $('ls-reset').addEventListener('click', () => animateTo(1.2, 110));
    update();
  }

  // ---------------------------------------------------------------- demo 2: three coins
  function demoCoin() {
    let rand = mulberry32(2026);
    const P = [1 / 8, 3 / 8, 3 / 8, 1 / 8];
    let counts, N, s1, s2;
    function reset() { counts = [0, 0, 0, 0]; N = 0; s1 = 0; s2 = 0; }
    function flip(k) {
      for (let t = 0; t < k; t++) {
        const x = (rand() < 0.5 ? 1 : 0) + (rand() < 0.5 ? 1 : 0) + (rand() < 0.5 ? 1 : 0);
        counts[x]++; N++; s1 += x; s2 += x * x;
      }
    }
    const view = canvasView($('coin-plot'), v => {
      const ctx = v.ctx;
      const maxF = Math.max(0.5, ...counts.map(c => (N ? c / N : 0)));
      const F = frame(v, [-0.6, 3.6], [0, Math.min(1, maxF * 1.15)], { l: 40 });
      axes(ctx, F, { xticks: [0, 1, 2, 3], xfmt: t => 'X=' + t, ylabel: '频率 / 概率' });
      const bw = Math.min(70, F.pw / 4.2 * 0.62);
      for (let k = 0; k < 4; k++) {
        const fr = N ? counts[k] / N : 0, cx = F.X(k);
        ctx.fillStyle = C.accent; ctx.globalAlpha = 0.8;
        ctx.fillRect(cx - bw / 2, F.Y(fr), bw, F.Y(0) - F.Y(fr));
        ctx.globalAlpha = 1;
        ctx.strokeStyle = C.c2; ctx.lineWidth = 2; ctx.setLineDash([5, 3]);
        ctx.strokeRect(cx - bw / 2 - 5, F.Y(P[k]), bw + 10, F.Y(0) - F.Y(P[k]));
        ctx.setLineDash([]);
      }
      legend(ctx, F, [['实验频率', C.accent, false], ['理论 p(x)', C.c2, true]]);
    });
    function update() {
      const mean = N ? s1 / N : 0, variance = N ? s2 / N - mean * mean : 0;
      $('coin-readout').innerHTML =
        '已做 <b>' + N.toLocaleString('en-US') + '</b> 次试验（每次抛 3 枚硬币）。' +
        '样本均值 = <b>' + mean.toFixed(3) + '</b>（理论 E[X] = 1.5）· 样本方差 = <b>' + variance.toFixed(3) + '</b>（理论 var(X) = 0.75）' +
        '<br>频率：' + counts.map((c, k) => 'X=' + k + ' → ' + (N ? (c / N).toFixed(3) : '—')).join('，') + '（理论 0.125, 0.375, 0.375, 0.125）';
      view.draw();
    }
    $('coin-1').addEventListener('click', () => { flip(1); update(); });
    $('coin-100').addEventListener('click', () => { flip(100); update(); });
    $('coin-10000').addEventListener('click', () => { flip(10000); update(); });
    $('coin-reset').addEventListener('click', () => { reset(); rand = mulberry32(Math.floor(Math.random() * 1e9)); update(); });
    reset(); flip(20); update();
  }

  // ---------------------------------------------------------------- demo 3: 1-D Gaussian
  function demoGauss1() {
    const mu = $('g-mu'), sg = $('g-sigma'), kk = $('g-k');
    const view = canvasView($('g-plot'), v => {
      const ctx = v.ctx, m = +mu.value, s = +sg.value, k = +kk.value;
      const pdf = x => Math.exp(-(x - m) * (x - m) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));
      const peak = pdf(m);
      const F = frame(v, [-8, 8], [0, Math.max(0.5, peak * 1.15)], { l: 40 });
      axes(ctx, F, { xlabel: 'x', ylabel: 'p(x)' });
      ctx.save(); clip(ctx, F);
      const lo = m - k * s, hi = m + k * s;
      ctx.beginPath(); ctx.moveTo(F.X(lo), F.Y(0));
      for (const x of linspace(lo, hi, 120)) ctx.lineTo(F.X(x), F.Y(pdf(x)));
      ctx.lineTo(F.X(hi), F.Y(0)); ctx.closePath();
      ctx.fillStyle = C.accent; ctx.globalAlpha = 0.2; ctx.fill(); ctx.globalAlpha = 1;
      if (peak > 1) {
        ctx.strokeStyle = C.c2; ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(F.m.l, F.Y(1)); ctx.lineTo(F.m.l + F.pw, F.Y(1)); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.c2; ctx.font = MONO; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText('p = 1', F.m.l + 6, F.Y(1) - 3);
      }
      ctx.strokeStyle = C.ink3; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(F.X(m), F.Y(0)); ctx.lineTo(F.X(m), F.Y(peak)); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2.2;
      curve(ctx, F, linspace(-8, 8, 400), pdf);
      ctx.restore();
    });
    function update() {
      const m = +mu.value, s = +sg.value, k = +kk.value;
      $('g-mu-out').textContent = m.toFixed(1);
      $('g-sigma-out').textContent = s.toFixed(2);
      $('g-k-out').textContent = k.toFixed(1);
      const peak = 1 / (s * Math.sqrt(2 * Math.PI));
      $('g-readout').innerHTML =
        'P(μ − kσ ≤ X ≤ μ + kσ) = erf(k/√2) = <b>' + (100 * erf(k / Math.SQRT2)).toFixed(2) + '%</b>（阴影面积）' +
        '<br>方差 σ² = ' + (s * s).toFixed(3) + '，峰值 p(μ) = 1/(σ√(2π)) = <b>' + peak.toFixed(3) + '</b>' +
        (peak > 1 ? ' — 密度大于 1 完全合法：它不是概率，曲线下总面积仍然是 1' : '');
      view.draw();
    }
    [mu, sg, kk].forEach(el => el.addEventListener('input', update));
    update();
  }

  // ---------------------------------------------------------------- demo 4: 2-D Gaussian
  function demoGauss2() {
    let seed = 11, Z = [];
    function sample() {
      const g = gaussian(mulberry32(seed));
      Z = [];
      for (let i = 0; i < 600; i++) Z.push([g(), g()]);
    }
    sample();
    const s1 = $('g2-s1'), s2 = $('g2-s2'), rh = $('g2-rho');
    function params() {
      const a = +s1.value, b = +s2.value, r = +rh.value;
      const p = a * a, q = r * a * b, w = b * b;
      const mid = (p + w) / 2, rad = Math.sqrt(((p - w) / 2) * ((p - w) / 2) + q * q);
      const l1 = mid + rad, l2 = mid - rad;
      let e1 = Math.abs(q) > 1e-12 ? [q, l1 - p] : (p >= w ? [1, 0] : [0, 1]);
      const nn = Math.hypot(e1[0], e1[1]);
      e1 = [e1[0] / nn, e1[1] / nn];
      return { a, b, r, p, q, w, l1, l2, e1, e2: [-e1[1], e1[0]], L: [[a, 0], [r * b, b * Math.sqrt(1 - r * r)]] };
    }
    const view = canvasView($('g2-plot'), v => {
      const ctx = v.ctx, P = params();
      const F = frame(v, [-5, 5], [-5, 5], { l: 34, r: 12, t: 18, b: 28 });
      axes(ctx, F, { xticks: [-4, -2, 0, 2, 4], yticks: [-4, -2, 0, 2, 4], xlabel: 'x₁', ylabel: 'x₂' });
      ctx.save(); clip(ctx, F);
      const L = P.L;
      ctx.fillStyle = C.ink3; ctx.globalAlpha = 0.55;
      for (const z of Z) {
        const x = L[0][0] * z[0], y = L[1][0] * z[0] + L[1][1] * z[1];
        ctx.beginPath(); ctx.arc(F.X(x), F.Y(y), 1.7, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.globalAlpha = 1;
      [1, 2, 3].forEach((D, i) => {
        ctx.strokeStyle = C.accent; ctx.lineWidth = i === 0 ? 2.4 : 1.5; ctx.globalAlpha = [1, 0.75, 0.5][i];
        ctx.beginPath();
        for (let k = 0; k <= 120; k++) {
          const t = 2 * Math.PI * k / 120, u0 = D * Math.cos(t), u1 = D * Math.sin(t);
          const x = L[0][0] * u0, y = L[1][0] * u0 + L[1][1] * u1;
          if (k) ctx.lineTo(F.X(x), F.Y(y)); else ctx.moveTo(F.X(x), F.Y(y));
        }
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      ctx.strokeStyle = C.c2; ctx.lineWidth = 2;
      [[P.e1, Math.sqrt(P.l1)], [P.e2, Math.sqrt(Math.max(P.l2, 0))]].forEach(([e, len], i) => {
        ctx.setLineDash(i ? [5, 4] : []);
        ctx.beginPath(); ctx.moveTo(F.X(-e[0] * len), F.Y(-e[1] * len)); ctx.lineTo(F.X(e[0] * len), F.Y(e[1] * len)); ctx.stroke();
      });
      ctx.setLineDash([]);
      ctx.restore();
    });
    function mat(rows) {
      return '<span class="mat">' + rows.map(r => '<span>' + r.map(x => '<span>' + (x >= 0 ? ' ' : '') + x.toFixed(2) + '</span>').join('') + '</span>').join('') + '</span>';
    }
    function update() {
      const P = params();
      $('g2-s1-out').textContent = P.a.toFixed(2);
      $('g2-s2-out').textContent = P.b.toFixed(2);
      $('g2-rho-out').textContent = P.r.toFixed(2);
      const det = P.p * P.w - P.q * P.q;
      const inside1 = Z.filter(z => z[0] * z[0] + z[1] * z[1] <= 1).length / Z.length;
      const inside2 = Z.filter(z => z[0] * z[0] + z[1] * z[1] <= 4).length / Z.length;
      const ang = Math.atan2(P.e1[1], P.e1[0]) * 180 / Math.PI;
      $('g2-readout').innerHTML =
        '<div class="mats"><div>Σ = ' + mat([[P.p, P.q], [P.q, P.w]]) + '</div>' +
        '<div>Σ⁻¹ = ' + mat([[P.w / det, -P.q / det], [-P.q / det, P.p / det]]) + '</div></div>' +
        '|Σ| = ' + det.toFixed(3) + ' · 特征值 λ₁ = ' + P.l1.toFixed(3) + '，λ₂ = ' + P.l2.toFixed(3) +
        ' · 长轴方向 ' + (((ang % 180) + 180) % 180).toFixed(0) + '°（橙色实线；虚线为短轴，半轴长 √λ）' +
        '<br>落在 Δ ≤ 1 内的样本 ' + (100 * inside1).toFixed(1) + '%（理论 39.3%），Δ ≤ 2 内 ' + (100 * inside2).toFixed(1) + '%（理论 86.5%）— 不随 Σ 变化';
      view.draw();
    }
    [s1, s2, rh].forEach(el => el.addEventListener('input', update));
    $('g2-resample').addEventListener('click', () => { seed++; sample(); update(); });
    update();
  }

  // ---------------------------------------------------------------- demo 5: minimax rates
  function demoRate() {
    const sEl = $('r-s'), dEl = $('r-d');
    const view = canvasView($('r-plot'), v => {
      const ctx = v.ctx, s = +sEl.value, d = +dEl.value;
      const F = frame(v, [1, 6], [-6, 0], { l: 44, r: 40 });
      axes(ctx, F, {
        xticks: [1, 2, 3, 4, 5, 6], yticks: [-6, -5, -4, -3, -2, -1, 0],
        xfmt: k => pow10(k), yfmt: k => (k === 0 ? '1' : pow10(k)), xlabel: '样本量 N', ylabel: 'E‖f̂ − f‖²（省略常数）',
      });
      ctx.save(); clip(ctx, F);
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(F.X(1), F.Y(-1)); ctx.lineTo(F.X(6), F.Y(-6)); ctx.stroke(); ctx.setLineDash([]);
      ctx.restore();
      ctx.font = MONO; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillStyle = C.ink2;
      ctx.fillText('N⁻¹', F.X(4.3) + 6, F.Y(-4.3) - 8);
      const ds = [1, 2, 5, 10, 20, 50].filter(x => x !== d);
      for (const dd of ds) {
        const r = 2 * s / (2 * s + dd);
        ctx.strokeStyle = C.ink3; ctx.globalAlpha = 0.6; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(F.X(1), F.Y(-r)); ctx.lineTo(F.X(6), F.Y(-6 * r)); ctx.stroke();
        ctx.globalAlpha = 1; ctx.fillStyle = C.ink3;
        ctx.fillText('d=' + dd, F.X(6) + 4, F.Y(-6 * r));
      }
      const r = 2 * s / (2 * s + d);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(F.X(1), F.Y(-r)); ctx.lineTo(F.X(6), F.Y(-6 * r)); ctx.stroke();
      ctx.fillStyle = C.accent;
      ctx.fillText('d=' + d, F.X(6) + 4, F.Y(-6 * r));
    });
    function update() {
      const s = +sEl.value, d = +dEl.value, r = 2 * s / (2 * s + d);
      $('r-s-out').textContent = s.toFixed(1);
      $('r-d-out').textContent = d;
      $('r-readout').innerHTML =
        '最优速率 N<sup>−2s/(2s+d)</sup> = <b>N<sup>−' + r.toFixed(3) + '</sup></b>' +
        '<br>想把均方误差减半，需要把数据量乘以 2<sup>1/' + r.toFixed(3) + '</sup> ≈ <b>' + bigNum(Math.pow(2, 1 / r)) + '</b> 倍；' +
        '减到 1/10 需要 <b>' + bigNum(Math.pow(10, 1 / r)) + '</b> 倍。（设定正确的参数模型：2 倍和 10 倍）';
      view.draw();
    }
    [sEl, dEl].forEach(el => el.addEventListener('input', update));
    update();
  }

  // ---------------------------------------------------------------- demo 6: kernel ridge regression
  function demoKRR() {
    const kEl = $('krr-kernel'), lEl = $('krr-ell'), lamEl = $('krr-lam'), nEl = $('krr-n'), noiseEl = $('krr-noise'), truthEl = $('krr-truth'), bvEl = $('krr-bv');
    const f = x => Math.sin(2 * Math.PI * x);
    const GRID = linspace(-0.02, 1.02, 260), ERRGRID = linspace(0, 1, 101), BVGRID = linspace(0, 1, 51);
    const NSETS = 20, NSWEEP = 12;
    let seed = 3, xs = [], ys = [], sets = [], sweep = [], bvSweep = [], cur = null, bv = null;
    function makeSet(s, n, noise) {
      const rand = mulberry32(s), g = gaussian(mulberry32(s + 1000));
      const X = [];
      for (let i = 0; i < n; i++) X.push(rand());
      X.sort((p, q) => p - q);
      return [X, X.map(x => f(x) + noise * g())];
    }
    function gen() { // the data set you see, plus 20 independent ones from the same distribution
      const n = +nEl.value, noise = +noiseEl.value;
      [xs, ys] = makeSet(seed, n, noise);
      sets = [];
      for (let j = 0; j < NSETS; j++) sets.push(makeSet(seed * 7919 + 101 * (j + 1), n, noise));
    }
    function errs(m) {
      let tr = 0, te = 0;
      for (let i = 0; i < xs.length; i++) { const e = m.predict(xs[i]) - ys[i]; tr += e * e; }
      for (const x of ERRGRID) { const e = m.predict(x) - f(x); te += e * e; }
      return [tr / xs.length, te / ERRGRID.length];
    }
    // E_D[(f̂ − f)²] = (E_D f̂ − f)² + E_D[(f̂ − E_D f̂)²], estimated from `count` independent data sets, averaged over x
    function biasVar(kind, ell, lam, count, grid) {
      const preds = [];
      for (let j = 0; j < count; j++) { const m = krrFit(sets[j][0], sets[j][1], kind, ell, lam); preds.push(grid.map(m.predict)); }
      const mean = grid.map((_, k) => preds.reduce((acc, p) => acc + p[k], 0) / count);
      let b2 = 0, va = 0;
      grid.forEach((x, k) => {
        b2 += (mean[k] - f(x)) * (mean[k] - f(x));
        let acc = 0;
        for (const p of preds) acc += (p[k] - mean[k]) * (p[k] - mean[k]);
        va += acc / count;
      });
      return { preds, mean, b2: b2 / grid.length, va: va / grid.length };
    }
    const ell = () => Math.pow(10, +lEl.value), lam = () => Math.pow(10, +lamEl.value);
    function computeSweep() {
      sweep = []; bvSweep = [];
      for (let i = 0; i <= 40; i++) {
        const le = -2.3 + i * 3 / 40;
        sweep.push([le].concat(errs(krrFit(xs, ys, kEl.value, Math.pow(10, le), lam()))));
      }
      if (bvEl.checked) {
        for (let i = 0; i <= 20; i++) {
          const le = -2.3 + i * 3 / 20, r = biasVar(kEl.value, Math.pow(10, le), lam(), NSWEEP, BVGRID);
          bvSweep.push([le, r.b2, r.va, r.b2 + r.va]);
        }
      }
    }
    function compute() {
      const m = krrFit(xs, ys, kEl.value, ell(), lam()), e = errs(m);
      cur = { m, tr: e[0], te: e[1], dof: m.dof() };
      bv = bvEl.checked ? biasVar(kEl.value, ell(), lam(), NSETS, ERRGRID) : null;
    }

    const view = canvasView($('krr-plot'), v => {
      const ctx = v.ctx, F = frame(v, [-0.02, 1.02], [-2.2, 2.2], { l: 38 });
      axes(ctx, F, { xlabel: 'x', ylabel: 'y' });
      ctx.save(); clip(ctx, F);
      if (bv) {
        ctx.strokeStyle = C.ink3; ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
        for (const p of bv.preds) poly(ctx, F, ERRGRID, p);
        ctx.globalAlpha = 1;
      }
      if (truthEl.checked) {
        ctx.strokeStyle = C.c3; ctx.lineWidth = 1.8; ctx.setLineDash([6, 4]);
        curve(ctx, F, GRID, f); ctx.setLineDash([]);
      }
      if (bv) {
        ctx.strokeStyle = C.c2; ctx.lineWidth = 2.6; ctx.setLineDash([2, 3]);
        poly(ctx, F, ERRGRID, bv.mean); ctx.setLineDash([]);
      }
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2.4;
      curve(ctx, F, GRID, cur.m.predict);
      dots(ctx, F, xs, ys, 3.4, C.ink);
      ctx.restore();
      const items = [['KRR 拟合', C.accent, false]];
      if (truthEl.checked) items.push(['真实 f(x)', C.c3, true]);
      if (bv) { items.push(['20 组数据的平均拟合', C.c2, [2, 3]]); items.push(['每组数据各自的拟合', C.ink3, false]); }
      legend(ctx, F, items);
    });
    const sv = canvasView($('krr-sweep'), v => {
      const ctx = v.ctx, F = frame(v, [-2.3, 0.7], [-4, 0.5], { l: 44 });
      axes(ctx, F, {
        xticks: [-2, -1, 0], xfmt: k => ({ '-2': '0.01', '-1': '0.1', '0': '1' })[String(k)],
        yticks: [-4, -3, -2, -1, 0], yfmt: k => (k === 0 ? '1' : pow10(k)), xlabel: '长度尺度 ℓ（对数刻度）', ylabel: '均方误差',
      });
      const lg = x => Math.log10(Math.max(x, 1e-5));
      const series = bv && bvSweep.length
        ? [[bvSweep, 1, C.c2, [5, 3], '偏差²'], [bvSweep, 2, C.c3, [2, 3], '方差'], [bvSweep, 3, C.accent, false, '偏差² + 方差']]
        : [[sweep, 1, C.c2, [5, 3], '训练误差'], [sweep, 2, C.accent, false, '与真实 f 的误差']];
      ctx.save(); clip(ctx, F);
      for (const [data, k, col, dash] of series) {
        ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.setLineDash(dash || []);
        poly(ctx, F, data.map(p => p[0]), data.map(p => lg(p[k])));
      }
      ctx.setLineDash([3, 3]); ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(F.X(+lEl.value), F.m.t); ctx.lineTo(F.X(+lEl.value), F.m.t + F.ph); ctx.stroke(); ctx.setLineDash([]);
      ctx.restore();
      legend(ctx, F, series.map(s => [s[4], s[2], s[3]]));
    });
    const NAMES = { gauss: '高斯核', m12: 'Matérn 1/2', m32: 'Matérn 3/2', m52: 'Matérn 5/2', lin: '线性核 1 + x·x′' };
    function update() {
      const isLin = kEl.value === 'lin';
      lEl.disabled = isLin;
      $('krr-ell-out').textContent = isLin ? '不适用' : ell().toPrecision(2);
      $('krr-lam-out').textContent = lam().toExponential(0);
      $('krr-n-out').textContent = nEl.value;
      $('krr-noise-out').textContent = (+noiseEl.value).toFixed(2);
      $('krr-readout').innerHTML =
        NAMES[kEl.value] + ' · 训练误差 <b>' + cur.tr.toExponential(2) + '</b> · 与真实函数的误差 ‖f̂ − f‖² ≈ <b>' + cur.te.toExponential(2) + '</b>' +
        '<br>有效参数个数 tr(S) = <b>' + cur.dof.toFixed(1) + '</b>（数据点 n = ' + xs.length + '）' +
        (isLin ? ' — 线性核只有两个特征 (1, x)，所以 tr(S) ≤ 2，怎么调都是直线' : '') +
        (bv ? '<br>20 组独立数据集（同样的 n 和噪声）：偏差² ≈ <b>' + bv.b2.toExponential(2) + '</b>，方差 ≈ <b>' + bv.va.toExponential(2) +
          '</b>，两者之和 ≈ ' + (bv.b2 + bv.va).toExponential(2) + '，等于各组拟合与真实函数误差的平均值' : '');
      view.draw(); sv.draw();
    }
    // coalesce slider events into one recomputation per frame: 1 = current fit, 2 = + sweep, 3 = + new data
    let pending = 0;
    function schedule(level) {
      if (!pending) requestAnimationFrame(() => {
        const L = pending; pending = 0;
        if (L >= 3) gen();
        if (L >= 2) computeSweep();
        compute(); update();
      });
      pending = Math.max(pending, level);
    }
    kEl.addEventListener('change', () => schedule(2));
    lamEl.addEventListener('input', () => schedule(2));
    bvEl.addEventListener('change', () => schedule(2));
    lEl.addEventListener('input', () => schedule(1));
    nEl.addEventListener('input', () => schedule(3));
    noiseEl.addEventListener('input', () => schedule(3));
    truthEl.addEventListener('change', () => view.draw());
    $('krr-resample').addEventListener('click', () => { seed++; schedule(3); });
    gen(); computeSweep(); compute(); update();
  }

  // ---------------------------------------------------------------- demo 7: functions drawn from a kernel (GP prior)
  function demoGP() {
    const kEl = $('gp-kernel'), lEl = $('gp-ell');
    const XS = linspace(0, 1, 161), n = XS.length;
    let seed = 21, Z = [], paths = [];
    function newZ() {
      const g = gaussian(mulberry32(seed));
      Z = [];
      for (let s = 0; s < 4; s++) Z.push(XS.map(() => g()));
    }
    function compute() { // K = L Lᵀ, f = L z with z ~ N(0, I)  ⇒  f ~ N(0, K)
      const kind = kEl.value, ell = Math.pow(10, +lEl.value);
      const A = new Float64Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) A[i * n + j] = kernel(kind, XS[i], XS[j], ell);
      let L = null, jitter = 1e-9;
      while (!L && jitter < 1) { for (let i = 0; i < n; i++) A[i * n + i] += jitter; L = cholesky(A, n); jitter *= 10; }
      paths = L ? Z.map(z => XS.map((_, i) => { let acc = 0; for (let k = 0; k <= i; k++) acc += L[i * n + k] * z[k]; return acc; })) : [];
    }
    const view = canvasView($('gp-plot'), v => {
      const ctx = v.ctx, F = frame(v, [0, 1], [-3.2, 3.2], { l: 34 });
      ctx.fillStyle = C.accent; ctx.globalAlpha = 0.08;
      ctx.fillRect(F.m.l, F.Y(2), F.pw, F.Y(-2) - F.Y(2));
      ctx.globalAlpha = 1;
      axes(ctx, F, { xlabel: 'x', ylabel: 'f(x)', yticks: [-3, -2, -1, 0, 1, 2, 3] });
      ctx.save(); clip(ctx, F);
      const cols = [C.accent, C.c2, C.c3, C.ink2];
      paths.forEach((p, s) => { ctx.strokeStyle = cols[s]; ctx.lineWidth = 1.8; poly(ctx, F, XS, p); });
      ctx.restore();
    });
    const NOTES = {
      gauss: '高斯核：无穷次可导，曲线极其光滑。平均每单位长度约 1/(πℓ) 次穿过 0',
      m52: 'Matérn 5/2：二阶可导，光滑，但比高斯核更有"棱角"',
      m32: 'Matérn 3/2：一阶可导，能看到明显的急转弯',
      m12: 'Matérn 1/2：连续但处处不可导，像布朗运动一样毛糙',
    };
    function update() {
      const ell = Math.pow(10, +lEl.value);
      $('gp-ell-out').textContent = ell.toPrecision(2);
      $('gp-readout').innerHTML = NOTES[kEl.value] + '。ℓ = ' + ell.toPrecision(2) + '：ℓ 越小，曲线变化越快。' +
        '<br>浅色带是 ±2：因为 k(x, x) = 1，每个点的 f(x) 都服从 N(0, 1)，约 95% 落在带内。';
      view.draw();
    }
    kEl.addEventListener('change', () => { compute(); update(); });
    lEl.addEventListener('input', () => { compute(); update(); });
    $('gp-resample').addEventListener('click', () => { seed++; newZ(); compute(); update(); });
    newZ(); compute(); update();
  }

  // ---------------------------------------------------------------- Part 3 shared data: the quartic from the slides
  const quartic = x => 3 - 2 * x + 3 * x * x - 5 * x * x * x + x * x * x * x;
  const fmtSig = v => (!isFinite(v) || Math.abs(v) > 1e9 ? '∞' : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(2) : v.toFixed(2));

  // ---------------------------------------------------------------- demo P3a: polynomial regression bench (overfitting, ridge, LOOCV, GCV)
  function demoPoly() {
    const mEl = $('poly-m'), lamEl = $('poly-lam'), nEl = $('poly-n'), sEl = $('poly-noise'), axisEl = $('poly-axis');
    const GRID = linspace(0, 4, 201);
    const basis = (x, M) => { const u = x / 2 - 1, r = [1]; for (let k = 1; k <= M; k++) r.push(r[k - 1] * u); return r; };
    let seed = 7, xs = [], ys = [], cur = null, sweep = [];
    const lamOf = v => (+v <= -7 ? 0 : Math.pow(10, +v));
    function gen() {
      const g = gaussian(mulberry32(seed));
      xs = linspace(0, 4, +nEl.value); // fixed, equally spaced design points
      ys = xs.map(x => quartic(x) + +sEl.value * g());
    }
    function evaluate(M, lam) {
      const fit = ridgeFit(xs.map(x => basis(x, M)), ys, lam > 0 ? lam : 1e-10);
      const cv = cvScores(fit, ys);
      let te = 0;
      for (const x of GRID) { const r = basis(x, M); let acc = 0; for (let i = 0; i <= M; i++) acc += r[i] * fit.beta[i]; te += (acc - quartic(x)) * (acc - quartic(x)); }
      const sig = +sEl.value;
      return { M, lam, fit, cv, test: te / GRID.length + sig * sig };
    }
    function bruteLoo(M, lam) {
      let acc = 0;
      for (let n = 0; n < xs.length; n++) {
        const X = xs.filter((_, i) => i !== n), Y = ys.filter((_, i) => i !== n);
        const fl = ridgeFit(X.map(x => basis(x, M)), Y, lam > 0 ? lam : 1e-10);
        const pr = basis(xs[n], M).reduce((a, v, i) => a + v * fl.beta[i], 0);
        acc += (ys[n] - pr) * (ys[n] - pr);
      }
      return acc / xs.length;
    }
    function computeSweep() {
      sweep = [];
      if (axisEl.value === 'M') for (let M = 0; M <= 12; M++) { const r = evaluate(M, lamOf(lamEl.value)); sweep.push([M, r.cv.train, r.test, r.cv.loocv, r.cv.gcv]); }
      else for (let i = 0; i <= 40; i++) { const lv = -6 + i * 0.25, r = evaluate(+mEl.value, Math.pow(10, lv)); sweep.push([lv, r.cv.train, r.test, r.cv.loocv, r.cv.gcv]); }
    }
    function compute() { cur = evaluate(+mEl.value, lamOf(lamEl.value)); cur.brute = bruteLoo(cur.M, cur.lam); }

    const view = canvasView($('poly-plot'), v => {
      const ctx = v.ctx, F = frame(v, [-0.05, 4.05], [-45, 20], { l: 40 });
      axes(ctx, F, { xlabel: 'x', ylabel: 'y' });
      ctx.save(); clip(ctx, F);
      ctx.strokeStyle = C.c3; ctx.lineWidth = 1.8; ctx.setLineDash([6, 4]); curve(ctx, F, GRID, quartic); ctx.setLineDash([]);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2.4;
      curve(ctx, F, linspace(-0.05, 4.05, 300), x => basis(x, cur.M).reduce((a, t, i) => a + t * cur.fit.beta[i], 0));
      dots(ctx, F, xs, ys, 3.6, C.ink);
      ctx.restore();
      legend(ctx, F, [['拟合 f_β(x)', C.accent, false], ['真实函数', C.c3, true]]);
    });
    const sv = canvasView($('poly-sweep'), v => {
      const ctx = v.ctx, byM = axisEl.value === 'M';
      const s2 = +sEl.value * +sEl.value, lo = Math.floor(Math.log10(Math.max(s2, 0.05)) - 1), hi = lo + 5;
      const F = frame(v, byM ? [0, 12] : [-6, 4], [lo, hi], { l: 44 });
      const yt = []; for (let k = lo; k <= hi; k++) yt.push(k);
      axes(ctx, F, {
        xticks: byM ? [0, 2, 4, 6, 8, 10, 12] : [-6, -4, -2, 0, 2, 4], xfmt: byM ? (t => String(t)) : (k => pow10(k)),
        yticks: yt, yfmt: k => (k === 0 ? '1' : pow10(k)), xlabel: byM ? '多项式阶数 M' : '正则化 λ（对数刻度）', ylabel: '均方误差',
      });
      const lg = x => (isFinite(x) ? Math.log10(Math.max(x, 1e-6)) : hi + 1);
      const series = [[1, C.c2, [5, 3], '训练误差'], [2, C.accent, false, '测试误差（期望）'], [3, C.c3, [2, 3], 'LOOCV'], [4, C.ink2, [8, 3, 2, 3], 'GCV']];
      ctx.save(); clip(ctx, F);
      for (const [k, col, dash] of series) { ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.setLineDash(dash || []); poly(ctx, F, sweep.map(p => p[0]), sweep.map(p => lg(p[k]))); }
      ctx.setLineDash([3, 3]); ctx.strokeStyle = C.ink2; ctx.lineWidth = 1;
      const xm = byM ? +mEl.value : Math.max(-6, +lamEl.value);
      if (byM || +lamEl.value > -7) { ctx.beginPath(); ctx.moveTo(F.X(xm), F.m.t); ctx.lineTo(F.X(xm), F.m.t + F.ph); ctx.stroke(); }
      ctx.setLineDash([]);
      ctx.restore();
      legend(ctx, F, series.map(q => [q[3], q[1], q[2]]));
    });
    function update() {
      const lam = lamOf(lamEl.value), M = cur.M, N = xs.length;
      $('poly-m-out').textContent = M;
      $('poly-lam-out').textContent = lam === 0 ? '0（OLS）' : lam.toExponential(0);
      $('poly-n-out').textContent = N;
      $('poly-noise-out').textContent = (+sEl.value).toFixed(1);
      const interp = Math.max(...cur.fit.H) > 1 - 1e-6;
      const coef = Array.from(cur.fit.beta).map((b, i) => 'β' + i + ' = ' + fmtSig(b)).join('，');
      $('poly-readout').innerHTML =
        '训练误差 <b>' + fmtSig(cur.cv.train) + '</b> · 测试误差（新数据上的期望 MSE）<b>' + fmtSig(cur.test) + '</b> · 噪声下限 σ² = ' + (+sEl.value * +sEl.value).toFixed(2) +
        '<br>LOOCV（帽子矩阵公式）<b>' + fmtSig(cur.cv.loocv) + '</b>，暴力重拟合 ' + N + ' 次得到 ' + fmtSig(cur.brute) +
        (interp ? '（H<sub>nn</sub> ≈ 1：模型穿过了每个点，留一误差没有意义）' : '，两者一致') +
        ' · GCV <b>' + fmtSig(cur.cv.gcv) + '</b> · 有效参数个数 tr(H) = <b>' + cur.fit.trH.toFixed(2) + '</b>（参数总数 M + 1 = ' + (M + 1) + '）' +
        (M + 1 > N && lam === 0 ? '<br><b class="warn-text">M + 1 > N：参数比数据还多，最小二乘解不唯一。</b>这里用极小的 λ 取了其中范数最小的一个。' : '') +
        '<br><span class="mono">' + coef + '</span>';
      view.draw(); sv.draw();
    }
    let pending = 0;
    function schedule(level) {
      if (!pending) requestAnimationFrame(() => { const L = pending; pending = 0; if (L >= 3) gen(); computeSweep(); compute(); update(); });
      pending = Math.max(pending, level);
    }
    [mEl, lamEl].forEach(el => el.addEventListener('input', () => schedule(2)));
    axisEl.addEventListener('change', () => schedule(2));
    [nEl, sEl].forEach(el => el.addEventListener('input', () => schedule(3)));
    $('poly-resample').addEventListener('click', () => { seed++; schedule(3); });
    gen(); computeSweep(); compute(); update();
  }

  // ---------------------------------------------------------------- demo P3b: ridge vs LASSO shrinkage (orthonormal design)
  function demoShrink() {
    const lEl = $('shrink-lam');
    const view = canvasView($('shrink-plot'), v => {
      const ctx = v.ctx, lam = +lEl.value, F = frame(v, [-3, 3], [-3, 3], { l: 34, r: 12, t: 14, b: 28 });
      axes(ctx, F, { xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2, 3], xlabel: '最小二乘系数 b = Φᵀy', ylabel: '正则化后的 β' });
      ctx.save(); clip(ctx, F);
      const B = linspace(-3, 3, 241);
      ctx.strokeStyle = C.ink3; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); curve(ctx, F, B, b => b); ctx.setLineDash([]);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2.4; curve(ctx, F, B, b => b / (1 + lam));
      ctx.strokeStyle = C.c2; ctx.lineWidth = 2.4; curve(ctx, F, B, b => Math.sign(b) * Math.max(Math.abs(b) - lam, 0));
      ctx.fillStyle = C.c2; ctx.globalAlpha = 0.1; ctx.fillRect(F.X(-lam), F.m.t, F.X(lam) - F.X(-lam), F.ph); ctx.globalAlpha = 1;
      ctx.restore();
      legend(ctx, F, [['OLS：β = b', C.ink3, true], ['岭回归：β = b/(1+λ)', C.accent, false], ['LASSO：软阈值', C.c2, false]]);
    });
    function update() {
      const lam = +lEl.value, b = [3, -0.5, 1.2];
      const r = b.map(x => (x / (1 + lam)).toFixed(2)), l = b.map(x => (Math.sign(x) * Math.max(Math.abs(x) - lam, 0)).toFixed(2));
      $('shrink-lam-out').textContent = lam.toFixed(2);
      $('shrink-readout').innerHTML = '例：b = (3, −0.5, 1.2)，λ = ' + lam.toFixed(2) + '<br>岭回归 β = (' + r.join(', ') + ')：<b>每个系数按同一比例缩小，永远不会正好为 0</b>' +
        '<br>LASSO β = (' + l.join(', ') + ')：<b>|b| ≤ λ 的系数被直接置零</b>（浅橙色带），其余的平移 λ';
      view.draw();
    }
    lEl.addEventListener('input', update);
    update();
  }

  // ---------------------------------------------------------------- demo P3c: Nadaraya–Watson vs local linear regression
  function demoLocal() {
    const hEl = $('loc-h'), xEl = $('loc-x'), kEl = $('loc-kernel');
    const rnd = mulberry32(23), g = gaussian(mulberry32(24));
    const xs = []; for (let i = 0; i < 40; i++) xs.push(4 * rnd()); xs.sort((p, q) => p - q);
    const ys = xs.map(x => quartic(x) + 2.5 * g());
    const GRID = linspace(0, 4, 241);
    const view = canvasView($('loc-plot'), v => {
      const ctx = v.ctx, h = Math.pow(10, +hEl.value), x0 = +xEl.value, kind = kEl.value;
      const F = frame(v, [-0.05, 4.05], [-45, 20], { l: 40 });
      axes(ctx, F, { xlabel: 'x', ylabel: 'y' });
      ctx.save(); clip(ctx, F);
      const fits = GRID.map(x => localFit(xs, ys, x, h, kind));
      ctx.strokeStyle = C.c3; ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]); curve(ctx, F, GRID, quartic); ctx.setLineDash([]);
      [['nw', C.c2], ['ll', C.accent]].forEach(([key, col]) => {
        ctx.strokeStyle = col; ctx.lineWidth = 2.3; ctx.beginPath(); let on = false;
        GRID.forEach((x, i) => { const y = fits[i][key]; if (!isFinite(y)) { on = false; return; } if (on) ctx.lineTo(F.X(x), F.Y(y)); else { ctx.moveTo(F.X(x), F.Y(y)); on = true; } });
        ctx.stroke();
      });
      // kernel weights at x*: bars along the bottom
      const w = xs.map(x => kweight(kind, x - x0, h)), wmax = Math.max(...w, 1e-12);
      ctx.fillStyle = C.ink3; ctx.globalAlpha = 0.55;
      xs.forEach((x, i) => { const hh = 0.22 * F.ph * w[i] / wmax; ctx.fillRect(F.X(x) - 1.5, F.m.t + F.ph - hh, 3, hh); });
      ctx.globalAlpha = 1;
      dots(ctx, F, xs, ys, 3.2, C.ink);
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(F.X(x0), F.m.t); ctx.lineTo(F.X(x0), F.m.t + F.ph); ctx.stroke(); ctx.setLineDash([]);
      const r = localFit(xs, ys, x0, h, kind), span = Math.max(1.5 * h, 0.25);
      if (isFinite(r.ll)) {
        ctx.strokeStyle = C.accent; ctx.lineWidth = 1.6; ctx.setLineDash([7, 4]);
        ctx.beginPath(); ctx.moveTo(F.X(x0 - span), F.Y(r.ll - r.slope * span)); ctx.lineTo(F.X(x0 + span), F.Y(r.ll + r.slope * span)); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(F.X(x0), F.Y(r.ll), 5, 0, 2 * Math.PI); ctx.fill();
      }
      if (isFinite(r.nw)) { ctx.fillStyle = C.c2; ctx.beginPath(); ctx.arc(F.X(x0), F.Y(r.nw), 5, 0, 2 * Math.PI); ctx.fill(); }
      ctx.restore();
      legend(ctx, F, [['局部线性', C.accent, false], ['Nadaraya–Watson', C.c2, false], ['真实函数', C.c3, true]]);
    });
    function update() {
      const h = Math.pow(10, +hEl.value), x0 = +xEl.value, r = localFit(xs, ys, x0, h, kEl.value);
      let s1 = 0, s2 = 0; xs.forEach(x => { const w = kweight(kEl.value, x - x0, h); s1 += w; s2 += w * w; });
      $('loc-h-out').textContent = h.toPrecision(2);
      $('loc-x-out').textContent = x0.toFixed(2);
      $('loc-readout').innerHTML = '在 x* = ' + x0.toFixed(2) + '：真实值 f(x*) = <b>' + quartic(x0).toFixed(2) + '</b> · 局部线性 β̂₀ = <b>' + fmtSig(r.ll) + '</b> · Nadaraya–Watson = <b>' + fmtSig(r.nw) + '</b>' +
        '<br>有效样本数 (Σw)²/Σw² ≈ ' + (s2 > 0 ? (s1 * s1 / s2).toFixed(1) : '0') + '（灰色竖条是每个点的权重 k<sub>h</sub>(x<sub>n</sub> − x*)，虚线是 x* 处拟合出的局部直线）';
      view.draw();
    }
    [hEl, xEl].forEach(el => el.addEventListener('input', update));
    kEl.addEventListener('change', update);
    update();
  }

  // ---------------------------------------------------------------- demo P3d: Bayesian linear regression + evidence maximisation
  function demoBayes() {
    const aEl = $('bayes-alpha'), eEl = $('bayes-eta'), sEl = $('bayes-samples');
    const rnd = mulberry32(41), g = gaussian(mulberry32(42));
    const xs = []; for (let i = 0; i < 9; i++) xs.push(1.3 * rnd()); for (let i = 0; i < 9; i++) xs.push(2.6 + 1.4 * rnd()); xs.sort((p, q) => p - q);
    const ys = xs.map(x => quartic(x) + 2 * g());
    const CENTRES = linspace(0, 4, 11), WIDTH = 0.45;
    const rbf = x => [1].concat(CENTRES.map(c => Math.exp(-(x - c) * (x - c) / (2 * WIDTH * WIDTH))));
    const Phi = xs.map(rbf), GRID = linspace(-0.05, 4.05, 200), GPHI = GRID.map(rbf);
    let fit = null, Z = [], note = '', aVal = Math.pow(10, +aEl.value), eVal = Math.pow(10, +eEl.value);
    const zg = gaussian(mulberry32(77));
    for (let s = 0; s < 5; s++) Z.push(Array.from({ length: 12 }, () => zg()));
    const alpha = () => aVal, eta = () => eVal;
    function compute() { fit = bayesFit(Phi, ys, alpha(), eta()); }
    const view = canvasView($('bayes-plot'), v => {
      const ctx = v.ctx, F = frame(v, [-0.05, 4.05], [-55, 30], { l: 40 }), e = eta();
      axes(ctx, F, { xlabel: 'x', ylabel: 'y' });
      ctx.save(); clip(ctx, F);
      const mean = GPHI.map(r => r.reduce((a, t, i) => a + t * fit.m[i], 0)), sd = GPHI.map(r => Math.sqrt(predVar(fit, r, e)));
      ctx.fillStyle = C.accent; ctx.globalAlpha = 0.16; ctx.beginPath();
      GRID.forEach((x, i) => { const px = F.X(x), py = F.Y(mean[i] + 2 * sd[i]); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
      for (let i = GRID.length - 1; i >= 0; i--) ctx.lineTo(F.X(GRID[i]), F.Y(mean[i] - 2 * sd[i]));
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
      if (sEl.checked) {
        ctx.strokeStyle = C.ink3; ctx.lineWidth = 1; ctx.globalAlpha = 0.7;
        for (const z of Z) { const d = backSolveT(fit.L, fit.p, z), beta = fit.m.map((m, i) => m + d[i]); poly(ctx, F, GRID, GPHI.map(r => r.reduce((a, t, i) => a + t * beta[i], 0))); }
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = C.c3; ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]); curve(ctx, F, GRID, quartic); ctx.setLineDash([]);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2.4; poly(ctx, F, GRID, mean);
      dots(ctx, F, xs, ys, 3.4, C.ink);
      ctx.restore();
      const items = [['预测均值 mᵀφ(x)', C.accent, false], ['真实函数', C.c3, true]];
      if (sEl.checked) items.push(['后验抽样的函数', C.ink3, false]);
      legend(ctx, F, items);
    });
    function update() {
      const a = alpha(), e = eta();
      $('bayes-alpha-out').textContent = a.toExponential(1);
      $('bayes-eta-out').textContent = e.toPrecision(2);
      const sdGap = Math.sqrt(predVar(fit, rbf(1.95), e)), sdData = Math.sqrt(predVar(fit, rbf(0.6), e));
      $('bayes-readout').innerHTML =
        'α = ' + a.toExponential(2) + '，η = ' + e.toPrecision(3) + '（真实噪声精度 1/2² = 0.25）· 等价岭回归 λ = α/η = <b>' + (a / e).toExponential(2) + '</b>' +
        '<br>有效参数个数 γ = <b>' + fit.gamma.toFixed(2) + '</b>（共 ' + fit.p + ' 个基函数）· log 证据 log p(y | α, η) = <b>' + fit.logEv.toFixed(2) + '</b>' +
        '<br>预测标准差：数据空档 x = 1.95 处 <b>' + sdGap.toFixed(2) + '</b>，数据密集的 x = 0.6 处 <b>' + sdData.toFixed(2) + '</b>' + (note ? '<br>' + note : '');
      view.draw();
    }
    [aEl, eEl].forEach(el => el.addEventListener('input', () => { note = ''; aVal = Math.pow(10, +aEl.value); eVal = Math.pow(10, +eEl.value); compute(); update(); }));
    sEl.addEventListener('change', () => view.draw());
    $('bayes-opt').addEventListener('click', () => {
      const r = evidenceOpt(Phi, ys, alpha(), eta(), 500);
      const la = Math.min(+aEl.max, Math.max(+aEl.min, Math.log10(r.alpha))), le = Math.min(+eEl.max, Math.max(+eEl.min, Math.log10(r.eta)));
      aEl.value = la; eEl.value = le;
      aVal = Math.pow(10, la); eVal = Math.pow(10, le); // exact optimum unless it lies outside the slider range
      note = '<b class="ok-text">证据最大化 ' + r.iters + ' 次迭代收敛</b>：α = ' + r.alpha.toExponential(2) + '，η = ' + r.eta.toPrecision(3) + '。完全只用训练数据自动选出了超参数，没有做交叉验证。';
      compute(); update();
    });
    compute(); update();
  }

  // ---------------------------------------------------------------- Part 5 shared drawing helpers
  // draw the line a·x₁ + b·x₂ + c = 0 inside the frame's data box
  function drawLine(ctx, F, a, b, c) {
    const [x0, x1] = F.xr, [y0, y1] = F.yr, pts = [];
    if (Math.abs(b) > 1e-12) { for (const x of [x0, x1]) { const y = -(a * x + c) / b; if (y >= y0 - 1e-9 && y <= y1 + 1e-9) pts.push([x, y]); } }
    if (Math.abs(a) > 1e-12) { for (const y of [y0, y1]) { const x = -(b * y + c) / a; if (x >= x0 - 1e-9 && x <= x1 + 1e-9) pts.push([x, y]); } }
    if (pts.length < 2) return;
    let best = [pts[0], pts[1]], dmax = -1;
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) { const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]); if (d > dmax) { dmax = d; best = [pts[i], pts[j]]; } }
    ctx.beginPath(); ctx.moveTo(F.X(best[0][0]), F.Y(best[0][1])); ctx.lineTo(F.X(best[1][0]), F.Y(best[1][1])); ctx.stroke();
  }
  // marching squares: G[j][i] is the value at (gx[i], gy[j])
  function contour(ctx, F, G, gx, gy, level) {
    ctx.beginPath();
    for (let j = 0; j < gy.length - 1; j++) {
      for (let i = 0; i < gx.length - 1; i++) {
        const v = [G[j][i] - level, G[j][i + 1] - level, G[j + 1][i + 1] - level, G[j + 1][i] - level];
        const c = [[gx[i], gy[j]], [gx[i + 1], gy[j]], [gx[i + 1], gy[j + 1]], [gx[i], gy[j + 1]]], pts = [];
        for (let e = 0; e < 4; e++) {
          const a = v[e], b = v[(e + 1) % 4];
          if ((a < 0) !== (b < 0)) { const t = a / (a - b), p = c[e], q = c[(e + 1) % 4]; pts.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
        }
        for (let k = 0; k + 1 < pts.length; k += 2) { ctx.moveTo(F.X(pts[k][0]), F.Y(pts[k][1])); ctx.lineTo(F.X(pts[k + 1][0]), F.Y(pts[k + 1][1])); }
      }
    }
    ctx.stroke();
  }
  function classDots(ctx, F, X, isPos) {
    X.forEach((x, i) => {
      const px = F.X(x[0]), py = F.Y(x[1]);
      if (isPos[i]) { ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(px, py, 4.2, 0, 2 * Math.PI); ctx.fill(); }
      else { ctx.fillStyle = C.c2; ctx.fillRect(px - 3.8, py - 3.8, 7.6, 7.6); }
    });
  }

  // ---------------------------------------------------------------- demo P5a: sigmoid and log-odds
  function demoSigmoid() {
    const wEl = $('sig-w'), bEl = $('sig-b'), XS = linspace(-6, 6, 300);
    const view = canvasView($('sig-plot'), v => {
      const ctx = v.ctx, w = +wEl.value, b = +bEl.value, half = v.h / 2;
      const top = frame(v, [-6, 6], [-0.05, 1.05], { l: 40, t: 10, b: half + 6 });
      const bot = frame(v, [-6, 6], [-8, 8], { l: 40, t: half + 14, b: 28 });
      axes(ctx, top, { xticks: [-6, -4, -2, 0, 2, 4, 6], xfmt: () => '', yticks: [0, 0.5, 1], ylabel: 'p(x) = σ(wx + b)' });
      axes(ctx, bot, { xticks: [-6, -4, -2, 0, 2, 4, 6], yticks: [-8, -4, 0, 4, 8], xlabel: 'x', ylabel: '对数几率 log p/(1−p) = wx + b' });
      const x0 = Math.abs(w) > 1e-9 ? -b / w : null;
      [top, bot].forEach((F, k) => {
        ctx.save(); clip(ctx, F);
        ctx.strokeStyle = C.ink3; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        const yref = k === 0 ? 0.5 : 0;
        ctx.beginPath(); ctx.moveTo(F.m.l, F.Y(yref)); ctx.lineTo(F.m.l + F.pw, F.Y(yref)); ctx.stroke();
        if (x0 !== null) { ctx.strokeStyle = C.ink2; ctx.beginPath(); ctx.moveTo(F.X(x0), F.m.t); ctx.lineTo(F.X(x0), F.m.t + F.ph); ctx.stroke(); }
        ctx.setLineDash([]);
        ctx.strokeStyle = k === 0 ? C.accent : C.c2; ctx.lineWidth = 2.4;
        curve(ctx, F, XS, k === 0 ? (x => sigmoid(w * x + b)) : (x => w * x + b));
        ctx.restore();
      });
    });
    function update() {
      const w = +wEl.value, b = +bEl.value, p1 = sigmoid(w + b);
      $('sig-w-out').textContent = w.toFixed(1); $('sig-b-out').textContent = b.toFixed(1);
      $('sig-readout').innerHTML = (Math.abs(w) > 1e-9 ? '决策边界 p = ½ 在 x = −b/w = <b>' + (-b / w).toFixed(2) + '</b>（对数几率 = 0 的地方）' : 'w = 0：概率处处相同，没有决策边界') +
        ' · 边界处的斜率 dp/dx = w/4 = ' + (w / 4).toFixed(2) +
        '<br>在 x = 1：对数几率 w + b = ' + (w + b).toFixed(2) + '，几率 odds = e<sup>w+b</sup> = ' + fmtSig(Math.exp(w + b)) + '，概率 p = σ(w + b) = <b>' + p1.toFixed(3) + '</b>';
      view.draw();
    }
    [wEl, bEl].forEach(el => el.addEventListener('input', update));
    update();
  }

  // ---------------------------------------------------------------- demo P5b: 2-D logistic regression by Newton–Raphson (IRLS)
  function demoLogit() {
    const sepEl = $('logit-sep'), DOM = [-4, 4];
    const g = gaussian(mulberry32(31)), Z = [];
    for (let i = 0; i < 60; i++) Z.push([g(), g()]);
    let X = [], A = [], y = [], th = [0, 0, 0], hist = [], gdNote = '';
    function gen() {
      const s = +sepEl.value;
      X = Z.map((z, i) => { const c = i % 2; return [(c ? 1 : -1) * s / 2 + 0.85 * z[0], (c ? 1 : -1) * s / 4 + 0.85 * z[1]]; });
      y = X.map((_, i) => i % 2); A = X.map(x => [1, x[0], x[1]]);
    }
    function record() {
      const st = newtonStep(A, y, th);
      hist.push({ m: hist.length, nll: logisticNLL(A, y, th), gn: Math.hypot(...st.grad), tn: Math.hypot(...th), th: th.slice() });
    }
    function reset() { th = [0, 0, 0]; hist = []; gdNote = ''; record(); }
    function step() { th = newtonStep(A, y, th).th; record(); }
    function gdCompare() { // plain gradient descent from θ = 0 with a safe step 1/(¼ λmax(XᵀX))
      const last = hist[hist.length - 1]; if (last.gn > 1e-6) { gdNote = ''; return; }
      const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; A.forEach(a => { for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M[i][j] += a[i] * a[j]; });
      let v = [1, 1, 1], lam = 0; for (let k = 0; k < 60; k++) { const u = M.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]); lam = Math.hypot(...u); v = u.map(t => t / lam); }
      const eta = 4 / lam, target = last.nll + 1e-6; let t = [0, 0, 0], k = 0;
      for (; k < 200000; k++) { if (logisticNLL(A, y, t) <= target) break; const gr = newtonStep(A, y, t).grad; t = t.map((q, i) => q - eta * gr[i]); }
      gdNote = '对比：从 θ = 0 出发，步长 η = 4/λ<sub>max</sub>(XᵀX) 的梯度下降要 <b>' + (k >= 200000 ? '超过 200,000' : k.toLocaleString('en-US')) + '</b> 步才能达到同样精度；Newton 只用了 ' + last.m + ' 步。';
    }
    const view = canvasView($('logit-plot'), v => {
      const ctx = v.ctx, F = frame(v, DOM, DOM, { l: 34, r: 12, t: 18, b: 28 });
      const N = 56, cw = F.pw / N, ch = F.ph / N;
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const x1 = DOM[0] + (i + 0.5) * 8 / N, x2 = DOM[0] + (j + 0.5) * 8 / N, p = sigmoid(th[0] + th[1] * x1 + th[2] * x2);
        ctx.fillStyle = p >= 0.5 ? C.accent : C.c2; ctx.globalAlpha = 0.28 * Math.abs(2 * p - 1);
        ctx.fillRect(F.m.l + i * cw, F.m.t + F.ph - (j + 1) * ch, cw + 0.6, ch + 0.6);
      }
      ctx.globalAlpha = 1;
      axes(ctx, F, { xticks: [-4, -2, 0, 2, 4], yticks: [-4, -2, 0, 2, 4], xlabel: 'x₁', ylabel: 'x₂' });
      ctx.save(); clip(ctx, F);
      if (Math.hypot(th[1], th[2]) > 1e-9) {
        ctx.strokeStyle = C.ink; ctx.lineWidth = 2.4; drawLine(ctx, F, th[1], th[2], th[0]);
        ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
        drawLine(ctx, F, th[1], th[2], th[0] - Math.log(9)); drawLine(ctx, F, th[1], th[2], th[0] + Math.log(9));
        ctx.setLineDash([]);
      }
      classDots(ctx, F, X, y.map(v => v === 1));
      ctx.restore();
    });
    function update() {
      const s = +sepEl.value, last = hist[hist.length - 1];
      $('logit-sep-out').textContent = s.toFixed(1);
      let correct = 0; A.forEach((a, i) => { if ((th[0] + th[1] * a[1] + th[2] * a[2] > 0 ? 1 : 0) === y[i]) correct++; });
      const rows = hist.slice(-7).map(h => '<tr><td class="num">' + h.m + '</td><td class="num">' + h.nll.toFixed(6) + '</td><td class="num">' + h.gn.toExponential(1) + '</td><td class="num">' + h.tn.toFixed(2) + '</td></tr>').join('');
      const sepWarn = last.tn > 25 || last.nll < 1e-3;
      $('logit-readout').innerHTML =
        '第 <b>' + last.m + '</b> 次迭代 · θ = (b, w₁, w₂) = (' + th.map(t => t.toFixed(3)).join(', ') + ') · 训练准确率 ' + correct + '/' + A.length +
        '<div class="tbl" style="margin-top:6px"><table><thead><tr><th class="num">m</th><th class="num">NLL −ℒ(θ)</th><th class="num">‖∇‖</th><th class="num">‖θ‖</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
        (sepWarn ? '<b class="warn-text">数据线性可分：NLL 能无限接近 0，‖θ‖ 每一步都在变大，MLE 不存在。</b>实际中要加正则化（例如 Part 3 的岭惩罚 λ‖θ‖²）。' : (gdNote ? gdNote : ''));
      view.draw();
    }
    $('logit-step').addEventListener('click', () => { step(); if (hist[hist.length - 1].gn < 1e-6) gdCompare(); update(); });
    $('logit-run').addEventListener('click', () => { for (let k = 0; k < 40 && hist[hist.length - 1].gn > 1e-10; k++) step(); gdCompare(); update(); });
    $('logit-reset').addEventListener('click', () => { reset(); update(); });
    sepEl.addEventListener('input', () => { gen(); reset(); update(); });
    gen(); reset(); update();
  }

  // ---------------------------------------------------------------- figure P5c: margin-based losses
  function demoLosses() {
    canvasView($('loss-plot'), v => {
      const ctx = v.ctx, F = frame(v, [-2.5, 3], [0, 4], { l: 34 }), G = linspace(-2.5, 3, 400);
      axes(ctx, F, { xticks: [-2, -1, 0, 1, 2, 3], yticks: [0, 1, 2, 3, 4], xlabel: '间隔 γ = y f(x)', ylabel: '损失 ℓ(γ)' });
      ctx.save(); clip(ctx, F);
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(F.X(-2.5), F.Y(1)); ctx.lineTo(F.X(0), F.Y(1)); ctx.lineTo(F.X(0), F.Y(0)); ctx.lineTo(F.X(3), F.Y(0)); ctx.stroke();
      ctx.lineWidth = 2.4;
      ctx.strokeStyle = C.accent; curve(ctx, F, G, gm => Math.max(0, 1 - gm));
      ctx.strokeStyle = C.c2; ctx.setLineDash([6, 3]); curve(ctx, F, G, gm => Math.log1p(Math.exp(-gm)) / Math.LN2); ctx.setLineDash([]);
      ctx.strokeStyle = C.c3; ctx.setLineDash([2, 3]); curve(ctx, F, G, gm => Math.exp(-gm)); ctx.setLineDash([]);
      ctx.restore();
      legend(ctx, F, [['0–1 损失', C.ink2, false], ['hinge：max(0, 1−γ)（SVM）', C.accent, false], ['logistic：log₂(1+e^−γ)（逻辑回归）', C.c2, true], ['指数：e^−γ（boosting）', C.c3, [2, 3]]]);
    });
  }

  // ---------------------------------------------------------------- demo P5d: soft-margin / kernel SVM (SMO)
  function demoSVM() {
    const dEl = $('svm-data'), kEl = $('svm-kernel'), cEl = $('svm-c'), lEl = $('svm-ell'), DOM = [-3.2, 3.2];
    const GX = linspace(DOM[0], DOM[1], 71);
    let seed = 13, X = [], y = [], model = null, G = null;
    function gen() {
      const r = mulberry32(seed), g = gaussian(mulberry32(seed + 500)), kind = dEl.value;
      X = []; y = [];
      for (let i = 0; i < 40; i++) {
        const c = i % 2 ? 1 : -1;
        if (kind === 'ring') { const rad = (c > 0 ? 2.1 : 0.85) + 0.28 * g(), ang = 2 * Math.PI * r(); X.push([rad * Math.cos(ang), rad * Math.sin(ang)]); }
        else { const sep = kind === 'sep' ? 1.35 : 0.8, sd = kind === 'sep' ? 0.45 : 0.75; X.push([c * sep + sd * g(), c * sep * 0.5 + sd * g()]); }
        y.push(c);
      }
    }
    const ell = () => Math.pow(10, +lEl.value), Cv = () => Math.pow(10, +cEl.value);
    const kf = (a, b) => (kEl.value === 'lin' ? a[0] * b[0] + a[1] * b[1] : Math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * ell() * ell())));
    function train() {
      const n = X.length, K = new Float64Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = kf(X[i], X[j]);
      model = svmTrain(K, y, Cv(), { seed: 3 });
      const sv = []; for (let i = 0; i < n; i++) if (model.alpha[i] > 1e-8) sv.push(i);
      G = GX.map(x2 => GX.map(x1 => sv.reduce((acc, i) => acc + model.alpha[i] * y[i] * kf(X[i], [x1, x2]), model.b)));
    }
    const view = canvasView($('svm-plot'), v => {
      const ctx = v.ctx, F = frame(v, DOM, DOM, { l: 34, r: 12, t: 18, b: 28 });
      const N = GX.length - 1, cw = F.pw / N, ch = F.ph / N;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const fv = (G[j][i] + G[j][i + 1] + G[j + 1][i] + G[j + 1][i + 1]) / 4;
        ctx.fillStyle = fv >= 0 ? C.accent : C.c2; ctx.globalAlpha = Math.min(0.26, 0.1 + 0.08 * Math.abs(fv));
        ctx.fillRect(F.X(GX[i]), F.Y(GX[j + 1]), cw + 0.6, ch + 0.6);
      }
      ctx.globalAlpha = 1;
      axes(ctx, F, { xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [-3, -2, -1, 0, 1, 2, 3], xlabel: 'x₁', ylabel: 'x₂' });
      ctx.save(); clip(ctx, F);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2.4; contour(ctx, F, G, GX, GX, 0);
      ctx.strokeStyle = C.ink2; ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]); contour(ctx, F, G, GX, GX, 1); contour(ctx, F, G, GX, GX, -1); ctx.setLineDash([]);
      classDots(ctx, F, X, y.map(t => t > 0));
      const Cc = Cv();
      X.forEach((x, i) => {
        const a = model.alpha[i]; if (a <= 1e-8) return;
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8; ctx.setLineDash(a >= Cc - 1e-8 ? [3, 2] : []);
        ctx.beginPath(); ctx.arc(F.X(x[0]), F.Y(x[1]), 8, 0, 2 * Math.PI); ctx.stroke();
      });
      ctx.setLineDash([]);
      ctx.restore();
    });
    function update() {
      const Cc = Cv(), lin = kEl.value === 'lin', n = X.length;
      lEl.disabled = lin;
      $('svm-c-out').textContent = fmtSig(Cc);
      $('svm-ell-out').textContent = lin ? '不适用' : ell().toPrecision(2);
      let onM = 0, bound = 0, err = 0;
      for (let i = 0; i < n; i++) { const a = model.alpha[i]; if (a > 1e-8 && a < Cc - 1e-8) onM++; else if (a >= Cc - 1e-8) bound++; if (y[i] * model.f[i] <= 0) err++; }
      $('svm-readout').innerHTML =
        '支持向量 <b>' + (onM + bound) + '</b> / ' + n + '：正好在间隔线上（0 &lt; α &lt; C，实线圈）' + onM + ' 个，α = C（虚线圈，在间隔内或被分错）' + bound + ' 个 · 训练错误 ' + err + ' 个' +
        (lin ? ' · 间隔宽度 2/‖w‖ = <b>' + (2 / Math.sqrt(model.wnorm2)).toFixed(3) + '</b>' : '') +
        '<br>原问题目标 ½‖w‖² + CΣξᵢ = ' + model.primal.toFixed(4) + '，对偶目标 = ' + model.dual.toFixed(4) + '，相对差 ' + Math.abs((model.primal - model.dual) / Math.max(1e-12, Math.abs(model.primal))).toExponential(0) + '：两者相等（强对偶），说明 SMO 求到了最优解。';
      view.draw();
    }
    let pending = 0;
    function schedule(level) {
      if (!pending) requestAnimationFrame(() => { const L = pending; pending = 0; if (L >= 2) gen(); train(); update(); });
      pending = Math.max(pending, level);
    }
    [cEl, lEl].forEach(el => el.addEventListener('input', () => schedule(1)));
    kEl.addEventListener('change', () => schedule(1));
    dEl.addEventListener('change', () => schedule(2));
    $('svm-resample').addEventListener('click', () => { seed++; schedule(2); });
    gen(); train(); update();
  }

  // ---------------------------------------------------------------- demo 8: one-hidden-layer MLP + SGD
  function demoMLP() {
    const hEl = $('mlp-h'), lrEl = $('mlp-lr'), bsEl = $('mlp-bs'), actEl = $('mlp-act'), unitsEl = $('mlp-units');
    const runBtn = $('mlp-run');
    const N = 40, f = x => Math.sin(2 * Math.PI * x);
    const rd = mulberry32(99), gd = gaussian(mulberry32(199));
    const X = [], Y = [];
    for (let i = 0; i < N; i++) { const x = rd(); X.push(x); Y.push(f(x) + 0.12 * gd()); }
    const U = X.map(x => 2 * x - 1); // inputs rescaled to [−1, 1]
    const shRand = mulberry32(12345);
    const GRID = linspace(-0.02, 1.02, 220);
    let net, epoch, hist, running = false, diverged = false, initSeed = 5;

    function reset() {
      net = mlpCreate(+hEl.value, actEl.value, mulberry32(initSeed));
      epoch = 0; diverged = false;
      hist = [[0, mlpLoss(net, U, Y)]];
    }
    function shuffle() {
      const o = Array.from({ length: N }, (_, i) => i);
      for (let i = N - 1; i > 0; i--) { const j = Math.floor(shRand() * (i + 1)); const t = o[i]; o[i] = o[j]; o[j] = t; }
      return o;
    }
    function runEpoch() { // shuffle → split into mini-batches → SGD step on each
      const bs = Math.min(+bsEl.value, N), lr = Math.pow(10, +lrEl.value);
      const order = shuffle();
      for (let p = 0; p < N; p += bs) sgdStep(net, U, Y, order.slice(p, p + bs), lr);
      epoch++;
      const L = mlpLoss(net, U, Y);
      if (!isFinite(L) || L > 1e4) { diverged = true; running = false; }
      hist.push([epoch, isFinite(L) ? L : 1e4]);
      if (hist.length > 3000) hist = hist.filter((_, i) => i % 2 === 0 || i === hist.length - 1);
    }

    const fv = canvasView($('mlp-plot'), v => {
      const ctx = v.ctx, F = frame(v, [-0.02, 1.02], [-2, 2], { l: 38 });
      axes(ctx, F, { xlabel: 'x', ylabel: 'y' });
      ctx.save(); clip(ctx, F);
      ctx.strokeStyle = C.c3; ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]);
      curve(ctx, F, GRID, f); ctx.setLineDash([]);
      if (unitsEl.checked) {
        const A = act(net.kind);
        ctx.strokeStyle = C.ink3; ctx.lineWidth = 1; ctx.globalAlpha = 0.55;
        for (let j = 0; j < net.h; j++) curve(ctx, F, GRID, x => net.v[j] * A.f(net.W[j] * (2 * x - 1) + net.b[j]));
        ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2.4;
      curve(ctx, F, GRID, x => mlpPredict(net, 2 * x - 1));
      dots(ctx, F, X, Y, 3.2, C.ink);
      ctx.restore();
      legend(ctx, F, [['网络输出 ŷ', C.accent, false], ['真实 f(x)', C.c3, true]]);
    });
    const lv = canvasView($('mlp-loss'), v => {
      const ctx = v.ctx;
      const xmax = Math.max(50, hist[hist.length - 1][0]);
      const logs = hist.map(p => Math.log10(Math.max(p[1], 1e-5)));
      const ymin = Math.min(-2, Math.floor(Math.min(...logs))), ymax = Math.max(0, Math.ceil(Math.max(...logs)));
      const yt = [];
      for (let k = ymin; k <= ymax; k++) yt.push(k);
      const F = frame(v, [0, xmax], [ymin, ymax], { l: 44 });
      axes(ctx, F, { yticks: yt, yfmt: k => (k === 0 ? '1' : pow10(k)), xlabel: 'epoch', ylabel: '训练损失 (1/n)Σ½(ŷ−y)²' });
      ctx.save(); clip(ctx, F);
      ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
      ctx.beginPath();
      hist.forEach((p, i) => { const px = F.X(p[0]), py = F.Y(logs[i]); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
      ctx.stroke();
      ctx.restore();
    });
    function update() {
      const bs = Math.min(+bsEl.value, N), h = net.h;
      $('mlp-h-out').textContent = hEl.value;
      $('mlp-lr-out').textContent = Math.pow(10, +lrEl.value).toPrecision(2);
      $('mlp-readout').innerHTML =
        'epoch <b>' + epoch + '</b> · 训练损失 <b>' + hist[hist.length - 1][1].toExponential(2) + '</b>' +
        ' · 参数个数 3·' + h + ' + 1 = ' + (3 * h + 1) + '（W, b, v 各 ' + h + ' 个，c 1 个）· 每个 epoch 做 ' + Math.ceil(N / bs) + ' 次 SGD 更新' +
        (diverged ? '<br><b class="warn-text">训练发散了：学习率太大，损失爆炸。调小学习率后点「重置」。</b>' : '');
      runBtn.textContent = running ? '暂停 Pause' : '开始训练 Train';
      fv.draw(); lv.draw();
    }
    function loop() {
      if (!running) return;
      for (let k = 0; k < 3 && running; k++) runEpoch();
      update();
      if (running) requestAnimationFrame(loop);
    }
    runBtn.addEventListener('click', () => {
      if (diverged) reset();
      running = !running;
      update();
      if (running) requestAnimationFrame(loop);
    });
    $('mlp-step').addEventListener('click', () => { if (diverged) reset(); running = false; runEpoch(); update(); });
    $('mlp-reset').addEventListener('click', () => { running = false; initSeed++; reset(); update(); });
    [hEl, actEl].forEach(el => el.addEventListener(el.tagName === 'SELECT' ? 'change' : 'input', () => { running = false; reset(); update(); }));
    [lrEl, bsEl].forEach(el => el.addEventListener(el.tagName === 'SELECT' ? 'change' : 'input', update));
    unitsEl.addEventListener('change', () => fv.draw());
    reset(); update();
  }

  // ---------------------------------------------------------------- contents highlight
  function tocSpy() {
    if (!('IntersectionObserver' in window)) return;
    const links = Array.from(document.querySelectorAll('.toc a[href^="#"]'));
    const map = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (!e.isIntersecting) return;
        const a = map.get(e.target.id);
        if (!a) return;
        links.forEach(l => l.classList.remove('active'));
        a.classList.add('active');
      });
    }, { rootMargin: '-10% 0px -80% 0px' });
    map.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
  }

  // ---------------------------------------------------------------- per-viewer "got it" ticks (browser storage only)
  function progress() {
    const KEY = 'mlfe-done-v1';
    let done = new Set();
    try { done = new Set(JSON.parse(localStorage.getItem(KEY) || '[]')); } catch (_) { /* storage unavailable: ticks just won't persist */ }
    const secs = Array.from(document.querySelectorAll('section.sec[id]'));
    const counter = $('toc-progress');
    function save() { try { localStorage.setItem(KEY, JSON.stringify(Array.from(done))); } catch (_) { /* ignore */ } }
    function paint() {
      secs.forEach(s => {
        const a = document.querySelector('.toc a[href="#' + s.id + '"]');
        if (a) a.classList.toggle('done', done.has(s.id));
      });
      if (counter) counter.textContent = '已掌握 ' + secs.filter(s => done.has(s.id)).length + ' / ' + secs.length + ' 节';
    }
    secs.forEach(s => {
      const head = s.querySelector('.sec-head');
      if (!head) return;
      const lab = document.createElement('label');
      lab.className = 'done-toggle';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.id = 'done-' + s.id;
      cb.checked = done.has(s.id);
      lab.htmlFor = cb.id;
      lab.append(cb, document.createTextNode('这一节我已经掌握了'));
      cb.addEventListener('change', () => { if (cb.checked) done.add(s.id); else done.delete(s.id); save(); paint(); });
      head.appendChild(lab);
    });
    paint();
  }

  readColors();
  [demoLS, demoCoin, demoGauss1, demoGauss2, demoPoly, demoShrink, demoLocal, demoBayes, demoRate, demoKRR, demoGP, demoMLP, demoSigmoid, demoLogit, demoLosses, demoSVM, tocSpy, progress].forEach(fn => {
    try { fn(); } catch (err) { console.error(fn.name, err); }
  });
  const redraw = () => { readColors(); redrawers.forEach(d => d()); };
  if (window.matchMedia) {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', redraw);
  }
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(redraw);
})();
