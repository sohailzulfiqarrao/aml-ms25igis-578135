/* Four more algorithms for the ML hub landing page (mounted by ml-viz.js after ml-algos.js):
     tsne    t-SNE of the 87 observations of Assignment 1 (eight predictors, real data), coloured by evapotranspiration
     forest  a random forest of 24 trees on two interleaved moons: votes, smoothing and held-out accuracy
     svm     a soft-margin linear SVM (SMO) at four values of C: margin, support vectors
     cv      five-fold cross-validation of the Assignment 1 regression, computed live: it reproduces the report
   The algorithms themselves are plain functions (window.MLVIZ_PURE) so they can be checked without a browser. */
(function () {
   'use strict';
   var TAU = Math.PI * 2;

   /* ---------------------------------------------------------------- the maths ---------------------------------------------------------------- */
   function rng(seed) {
      var s = seed >>> 0;
      return function () {
         s = (s + 0x6D2B79F5) >>> 0;
         var t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
         return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
   }
   function gauss(r) { var u = 0; while (u === 0) { u = r(); } return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * r()); }

   /* t-SNE: perplexity-calibrated affinities in the input space, Student-t affinities in the map, gradient descent with
      momentum and adaptive gains, and early exaggeration for the first 80 iterations. */
   function tsneInit(X, perp, seed) {
      var n = X.length, d = X[0].length, i, j, k, r = rng(seed), D = new Float64Array(n * n), P = new Float64Array(n * n), logU = Math.log(perp);
      for (i = 0; i < n; i++) { for (j = 0; j < n; j++) { var s = 0; for (k = 0; k < d; k++) { var q = X[i][k] - X[j][k]; s += q * q; } D[i * n + j] = s; } }
      for (i = 0; i < n; i++) {
         var beta = 1, bmin = -Infinity, bmax = Infinity, row = new Float64Array(n), sumP = 1, tries;
         for (tries = 0; tries < 60; tries++) {
            sumP = 0; var sumDP = 0;
            for (j = 0; j < n; j++) { if (j === i) { row[j] = 0; continue; } row[j] = Math.exp(-D[i * n + j] * beta); sumP += row[j]; sumDP += D[i * n + j] * row[j]; }
            if (sumP < 1e-300) { sumP = 1e-300; }
            var diff = Math.log(sumP) + beta * sumDP / sumP - logU;
            if (Math.abs(diff) < 1e-5) { break; }
            if (diff > 0) { bmin = beta; beta = bmax === Infinity ? beta * 2 : (beta + bmax) / 2; } else { bmax = beta; beta = bmin === -Infinity ? beta / 2 : (beta + bmin) / 2; }
         }
         for (j = 0; j < n; j++) { P[i * n + j] = row[j] / sumP; }
      }
      var Ps = new Float64Array(n * n);
      for (i = 0; i < n; i++) { for (j = 0; j < n; j++) { if (i !== j) { Ps[i * n + j] = Math.max((P[i * n + j] + P[j * n + i]) / (2 * n), 1e-12); } } }
      var Y = new Float64Array(2 * n); for (i = 0; i < 2 * n; i++) { Y[i] = gauss(r) * 0.01; }
      var gains = new Float64Array(2 * n); gains.fill(1);
      return { n: n, P: Ps, Y: Y, dY: new Float64Array(2 * n), gains: gains, grad: new Float64Array(2 * n), num: new Float64Array(n * n), it: 0, kl: 0 };
   }
   function tsneStep(S, lr) {
      var n = S.n, P = S.P, Y = S.Y, num = S.num, g = S.grad, Z = 0, i, j, ex = S.it < 80 ? 4 : 1, mom = S.it < 80 ? 0.5 : 0.8;
      for (i = 0; i < n; i++) { for (j = i + 1; j < n; j++) { var dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1], v = 1 / (1 + dx * dx + dy * dy); num[i * n + j] = v; num[j * n + i] = v; Z += 2 * v; } }
      var kl = 0;
      for (i = 0; i < n; i++) {
         var gx = 0, gy = 0;
         for (j = 0; j < n; j++) {
            if (i === j) { continue; }
            var q = num[i * n + j] / Z, pij = P[i * n + j], m = (ex * pij - q) * num[i * n + j];
            gx += m * (Y[2 * i] - Y[2 * j]); gy += m * (Y[2 * i + 1] - Y[2 * j + 1]);
            kl += pij * Math.log(pij / Math.max(q, 1e-12));
         }
         g[2 * i] = 4 * gx; g[2 * i + 1] = 4 * gy;
      }
      var cx = 0, cy = 0;
      for (i = 0; i < 2 * n; i++) {
         var same = (g[i] > 0) === (S.dY[i] > 0);
         S.gains[i] = same ? Math.max(S.gains[i] * 0.8, 0.01) : S.gains[i] + 0.2;
         S.dY[i] = mom * S.dY[i] - lr * S.gains[i] * g[i]; Y[i] += S.dY[i];
         if (i % 2 === 0) { cx += Y[i]; } else { cy += Y[i]; }
      }
      cx /= n; cy /= n; for (i = 0; i < n; i++) { Y[2 * i] -= cx; Y[2 * i + 1] -= cy; }
      S.it++; S.kl = kl;
   }
   function neighbourAgreement(Y, cls, k) {                                         // share of each point's k nearest map neighbours that have its colour
      var n = cls.length, same = 0, i, j;
      for (i = 0; i < n; i++) {
         var d = [];
         for (j = 0; j < n; j++) { if (j !== i) { d.push([(Y[2 * i] - Y[2 * j]) * (Y[2 * i] - Y[2 * j]) + (Y[2 * i + 1] - Y[2 * j + 1]) * (Y[2 * i + 1] - Y[2 * j + 1]), j]); } }
         d.sort(function (a, b) { return a[0] - b[0]; });
         for (j = 0; j < k; j++) { if (cls[d[j][1]] === cls[i]) { same++; } }
      }
      return same / (n * k);
   }

   /* a classification tree on two features (x, y) with Gini splits; mtry = 1 gives the random forest its diversity */
   function giniOf(P, idx) { if (!idx.length) { return 0; } var a = 0; idx.forEach(function (q) { if (P[q].c === 1) { a++; } }); var f = a / idx.length; return 2 * f * (1 - f); }
   function growTree(P, idx, depth, maxD, r) {
      var n1 = 0; idx.forEach(function (q) { if (P[q].c === 1) { n1++; } });
      var n0 = idx.length - n1;
      if (depth >= maxD || n1 === 0 || n0 === 0 || idx.length < 4) { return { leaf: n1 > n0 ? 1 : 0 }; }
      var parent = giniOf(P, idx), feats = r() < 0.5 ? ['x', 'y'] : ['y', 'x'], best = null;
      for (var fi = 0; fi < feats.length && !best; fi++) {                                     // one random feature per split; the other only if it cannot split
         var f = feats[fi], s = idx.slice().sort(function (u, v) { return P[u][f] - P[v][f]; }), left1 = 0;
         for (var k = 1; k < s.length; k++) {
            if (P[s[k - 1]].c === 1) { left1++; }
            if (P[s[k]][f] === P[s[k - 1]][f]) { continue; }
            var nl = k, nr = s.length - k, fl = left1 / nl, fr = (n1 - left1) / nr, g = (nl * 2 * fl * (1 - fl) + nr * 2 * fr * (1 - fr)) / s.length;
            if (g < parent - 1e-9 && (!best || g < best.g)) { best = { f: f, th: (P[s[k]][f] + P[s[k - 1]][f]) / 2, g: g }; }
         }
      }
      if (!best) { return { leaf: n1 > n0 ? 1 : 0 }; }
      var L = [], R = []; idx.forEach(function (q) { (P[q][best.f] < best.th ? L : R).push(q); });
      return { f: best.f, th: best.th, l: growTree(P, L, depth + 1, maxD, r), r: growTree(P, R, depth + 1, maxD, r) };
   }
   function treePredict(t, x, y) { while (t.f !== undefined) { t = (t.f === 'x' ? x : y) < t.th ? t.l : t.r; } return t.leaf; }

   /* a linear soft-margin SVM by simplified SMO; snaps records (w, b) after every pass that changed something */
   function smo(X, y, C, seed, maxIter) {
      var n = X.length, a = new Float64Array(n), b = 0, r = rng(seed), tol = 1e-3, calm = 0, iter = 0, snaps = [], i, j;
      function dot(p, q) { return X[p][0] * X[q][0] + X[p][1] * X[q][1]; }
      function fx(i0) { var s = b; for (var k = 0; k < n; k++) { if (a[k] !== 0) { s += a[k] * y[k] * dot(k, i0); } } return s; }
      function wb() { var w0 = 0, w1 = 0; for (var k = 0; k < n; k++) { w0 += a[k] * y[k] * X[k][0]; w1 += a[k] * y[k] * X[k][1]; } return { w: [w0, w1], b: b }; }
      while (calm < 6 && iter < maxIter) {
         var changed = 0;
         for (i = 0; i < n; i++) {
            var Ei = fx(i) - y[i];
            if (!((y[i] * Ei < -tol && a[i] < C) || (y[i] * Ei > tol && a[i] > 0))) { continue; }
            j = Math.floor(r() * (n - 1)); if (j >= i) { j++; }
            var Ej = fx(j) - y[j], ai = a[i], aj = a[j], L, H;
            if (y[i] !== y[j]) { L = Math.max(0, aj - ai); H = Math.min(C, C + aj - ai); } else { L = Math.max(0, ai + aj - C); H = Math.min(C, ai + aj); }
            if (L === H) { continue; }
            var eta = 2 * dot(i, j) - dot(i, i) - dot(j, j); if (eta >= 0) { continue; }
            var ajn = Math.min(H, Math.max(L, aj - y[j] * (Ei - Ej) / eta)); if (Math.abs(ajn - aj) < 1e-5) { continue; }
            var ain = ai + y[i] * y[j] * (aj - ajn);
            var b1 = b - Ei - y[i] * (ain - ai) * dot(i, i) - y[j] * (ajn - aj) * dot(i, j), b2 = b - Ej - y[i] * (ain - ai) * dot(i, j) - y[j] * (ajn - aj) * dot(j, j);
            a[i] = ain; a[j] = ajn; b = (ain > 0 && ain < C) ? b1 : ((ajn > 0 && ajn < C) ? b2 : (b1 + b2) / 2); changed++;
         }
         iter++; if (changed === 0) { calm++; } else { calm = 0; snaps.push(wb()); }
      }
      var fin = wb(), sv = [];
      for (i = 0; i < n; i++) { if (y[i] * (fin.w[0] * X[i][0] + fin.w[1] * X[i][1] + fin.b) <= 1 + 1e-3) { sv.push(i); } }
      return { w: fin.w, b: fin.b, sv: sv, snaps: snaps, iter: iter };
   }

   /* ordinary least squares with an intercept, by the normal equations */
   function ols(X, y) {
      var n = X.length, m = X[0].length + 1, A = [], v = [], i, j, k;
      for (i = 0; i < m; i++) { A[i] = new Float64Array(m + 1); }
      for (k = 0; k < n; k++) {
         var row = [1].concat(X[k]);
         for (i = 0; i < m; i++) { for (j = 0; j < m; j++) { A[i][j] += row[i] * row[j]; } A[i][m] += row[i] * y[k]; }
      }
      for (i = 0; i < m; i++) { A[i][i] += 1e-10; }
      for (i = 0; i < m; i++) {
         var piv = i; for (j = i + 1; j < m; j++) { if (Math.abs(A[j][i]) > Math.abs(A[piv][i])) { piv = j; } }
         var tmp = A[i]; A[i] = A[piv]; A[piv] = tmp;
         for (j = i + 1; j < m; j++) { var f = A[j][i] / A[i][i]; for (k = i; k <= m; k++) { A[j][k] -= f * A[i][k]; } }
      }
      var w = new Float64Array(m);
      for (i = m - 1; i >= 0; i--) { var s = A[i][m]; for (j = i + 1; j < m; j++) { s -= A[i][j] * w[j]; } w[i] = s / A[i][i]; }
      return w;
   }
   function crossValidate(D) {                                                                    // D: { folds: [fold of each row], X: rows of predictors, y }
      var K = 0, i, f; D.folds.forEach(function (v) { if (v + 1 > K) { K = v + 1; } });
      var out = { r2: [], rmse: [], size: [] };
      for (f = 0; f < K; f++) {
         var trX = [], trY = [], teX = [], teY = [];
         for (i = 0; i < D.y.length; i++) { if (D.folds[i] === f) { teX.push(D.X[i]); teY.push(D.y[i]); } else { trX.push(D.X[i]); trY.push(D.y[i]); } }
         var w = ols(trX, trY), my = 0, ss = 0, st = 0;
         teY.forEach(function (v) { my += v; }); my /= teY.length;
         teX.forEach(function (row, k) { var p = w[0]; for (var q = 0; q < row.length; q++) { p += w[q + 1] * row[q]; } ss += (teY[k] - p) * (teY[k] - p); st += (teY[k] - my) * (teY[k] - my); });
         out.r2.push(1 - ss / st); out.rmse.push(Math.sqrt(ss / teY.length)); out.size.push(teY.length);
      }
      out.mean = out.r2.reduce(function (a, b) { return a + b; }, 0) / K;
      out.std = Math.sqrt(out.r2.reduce(function (a, b) { return a + (b - out.mean) * (b - out.mean); }, 0) / K);
      return out;
   }

   window.MLVIZ_PURE = { rng: rng, gauss: gauss, tsneInit: tsneInit, tsneStep: tsneStep, neighbourAgreement: neighbourAgreement, growTree: growTree, treePredict: treePredict, smo: smo, ols: ols, crossValidate: crossValidate };

   /* ------------------------------------------------------------------ the panels ------------------------------------------------------------------ */
   window.MLVIZ_EXTRA2 = function (A) {
      var COL = A.COL, clamp = A.clamp, ease = A.ease, easeIO = A.easeIO, lerp = A.lerp, rgba = A.rgba, MONO = A.MONO, TEXT = A.TEXT, GRID = A.GRID, AXIS = A.AXIS, reduce = A.reduce;
      var FONT = 'ui-monospace, "Cascadia Code", Consolas, monospace', PURE = window.MLVIZ_PURE;
      function square(x0, y0, w, h) { var s = Math.max(8, Math.min(w, h)); return { x: x0 + (w - s) / 2, y: y0 + (h - s) / 2, s: s }; }
      function note(c, text, W, H) {                                                              // a small legend with its own dark backing, bottom right
         c.font = '10.5px ' + FONT; var w = c.measureText(text).width + 12, x = W - w - 6, y = H - 24;
         c.fillStyle = 'rgba(5,10,22,.82)'; c.fillRect(x, y, w, 18); c.strokeStyle = 'rgba(148,163,184,.3)'; c.lineWidth = 1; c.strokeRect(x + 0.5, y + 0.5, w - 1, 17);
         c.fillStyle = 'rgba(203,213,225,.95)'; c.textAlign = 'left'; c.fillText(text, x + 6, y + 13); c.font = MONO;
      }
      function mix2(a, b, t) { var n1 = parseInt(a.slice(1), 16), n2 = parseInt(b.slice(1), 16); return 'rgb(' + Math.round(lerp(n1 >> 16, n2 >> 16, t)) + ',' + Math.round(lerp((n1 >> 8) & 255, (n2 >> 8) & 255, t)) + ',' + Math.round(lerp(n1 & 255, n2 & 255, t)) + ')'; }

      /* ============================ t-SNE on the real data ============================ */
      function tsne(card) {
         var D = window.ML_COLLINEAR, E = window.ML_NDVI_ET;
         if (!D || !E) { return null; }
         var p = A.base(card), n = D.rows.length, K = D.names.length, i, j;
         var X = D.rows.map(function (r) { return r.slice(); });
         for (j = 0; j < K; j++) {                                                                  // standardise every predictor
            var m = 0, v = 0; for (i = 0; i < n; i++) { m += X[i][j]; } m /= n;
            for (i = 0; i < n; i++) { v += (X[i][j] - m) * (X[i][j] - m); } v = Math.sqrt(v / n) || 1;
            for (i = 0; i < n; i++) { X[i][j] = (X[i][j] - m) / v; }
         }
         var et = E.map(function (r) { return r[1]; }), sorted = et.slice().sort(function (a, b) { return a - b; }), t1 = sorted[Math.floor(n / 3)], t2 = sorted[Math.floor(2 * n / 3)];
         var cls = et.map(function (v) { return v < t1 ? 0 : (v < t2 ? 1 : 2); }), COLS = [COL.violet, COL.amber, COL.green];
         var TOTAL = 400, RATE = 70, LEAD = 0.7, S, t, hold, agree;
         p.reset = function () { S = PURE.tsneInit(X, 12, 7); t = 0; hold = 0; agree = 0; p.bx = null; p.finished = false; };
         p.update = function (dt) {
            t += dt;
            var target = Math.min(TOTAL, Math.floor(Math.max(0, t - LEAD) * RATE)), guard = 0;
            while (S.it < target && guard++ < 40) { PURE.tsneStep(S, 60); if (S.it % 10 === 0 || S.it === TOTAL) { agree = PURE.neighbourAgreement(S.Y, cls, 5); } }
            if (S.it >= TOTAL) { p.finished = true; hold += dt; if (hold > 3 && !reduce) { p.reset(); } }
         };
         p.draw = function () {
            p.begin();
            var c = p.ctx, W = p.W, H = p.H, Y = S.Y, mnx = 1e9, mxx = -1e9, mny = 1e9, mxy = -1e9;
            for (i = 0; i < n; i++) { mnx = Math.min(mnx, Y[2 * i]); mxx = Math.max(mxx, Y[2 * i]); mny = Math.min(mny, Y[2 * i + 1]); mxy = Math.max(mxy, Y[2 * i + 1]); }
            var tgt = [mnx, mxx, mny, mxy];
            if (!p.bx) { p.bx = tgt.slice(); } else { for (i = 0; i < 4; i++) { p.bx[i] += (tgt[i] - p.bx[i]) * 0.25; } }
            var bw = Math.max(p.bx[1] - p.bx[0], 1e-6), bh = Math.max(p.bx[3] - p.bx[2], 1e-6), sc = Math.min((W - 44) / bw, (H - 58) / bh), ox = W / 2 - (p.bx[0] + p.bx[1]) / 2 * sc, oy = (H - 22) / 2 - (p.bx[2] + p.bx[3]) / 2 * sc;
            c.strokeStyle = GRID; c.lineWidth = 1; c.strokeRect(10.5, 10.5, W - 21, H - 36);
            for (i = 0; i < n; i++) {
               var x = ox + Y[2 * i] * sc, y = oy + Y[2 * i + 1] * sc;
               c.fillStyle = rgba(COLS[cls[i]], 0.25); c.beginPath(); c.arc(x, y, 7, 0, TAU); c.fill();
               c.fillStyle = COLS[cls[i]]; c.beginPath(); c.arc(x, y, 3.6, 0, TAU); c.fill();
            }
            c.textAlign = 'left'; c.fillStyle = TEXT;
            [['low ET', 0], ['mid', 1], ['high ET', 2]].forEach(function (e, k) { var lx = 16 + k * 78; c.fillStyle = COLS[e[1]]; c.beginPath(); c.arc(lx, H - 12, 4, 0, TAU); c.fill(); c.fillStyle = TEXT; c.fillText(e[0], lx + 9, H - 8); });
            p.say('iter', String(S.it)); p.say('kl', S.it ? S.kl.toFixed(2) : '--'); p.say('same', S.it ? Math.round(agree * 100) + '%' : '--');
            p.say('phase', S.it >= TOTAL ? 'done' : (S.it < 80 ? 'exaggerating' : 'settling'));
         };
         return p;
      }

      /* ================================ random forest ================================= */
      function forest(card) {
         var p = A.base(card), r = PURE.rng(41), GX = 34, GY = 21, B = 24, i, b;
         function moons(m, jit) {
            var out = [];
            for (var k = 0; k < m; k++) { var th = Math.PI * k / (m - 1); out.push({ x: 0.13 + (Math.cos(th) + 1) / 3 * 0.72 + PURE.gauss(r) * jit, y: 0.74 - (Math.sin(th) + 0.4) / 1.5 * 0.56 + PURE.gauss(r) * jit, c: 0 }); }
            for (k = 0; k < m; k++) { var th2 = Math.PI * k / (m - 1); out.push({ x: 0.13 + (1 - Math.cos(th2) + 1) / 3 * 0.72 + PURE.gauss(r) * jit, y: 0.74 - (0.4 - Math.sin(th2) + 0.4) / 1.5 * 0.56 + PURE.gauss(r) * jit, c: 1 }); }
            return out;
         }
         var train = moons(50, 0.075), test = moons(40, 0.075), trees = [], cell = [], cum = [], tAcc = [], fAcc = [];
         for (b = 0; b < B; b++) {
            var idx = []; for (i = 0; i < train.length; i++) { idx.push(Math.floor(r() * train.length)); }                   // bootstrap sample
            var tr = PURE.growTree(train, idx, 0, 5, r); trees.push(tr);
            var g = new Uint8Array(GX * GY); for (var gy = 0; gy < GY; gy++) { for (var gx = 0; gx < GX; gx++) { g[gy * GX + gx] = PURE.treePredict(tr, (gx + 0.5) / GX, (gy + 0.5) / GY); } }
            cell.push(g);
            var cg = new Uint8Array(GX * GY); for (i = 0; i < cg.length; i++) { cg[i] = (b ? cum[b - 1][i] : 0) + g[i]; } cum.push(cg);
            var ok = 0; test.forEach(function (pt) { if (PURE.treePredict(tr, pt.x, pt.y) === pt.c) { ok++; } }); tAcc.push(ok / test.length);
            var okF = 0; test.forEach(function (pt) { var v = 0; for (var q = 0; q <= b; q++) { v += PURE.treePredict(trees[q], pt.x, pt.y); } if ((v * 2 > b + 1 ? 1 : 0) === pt.c) { okF++; } }); fAcc.push(okF / test.length);
         }
         var meanTree = tAcc.reduce(function (a, v) { return a + v; }, 0) / B;
         var LEAD = 0.5, SLOT = 0.4, T_END = LEAD + B * SLOT + 0.4, CL = [COL.green, COL.violet], t, hold;
         p.reset = function () { t = 0; hold = 0; p.finished = false; };
         p.update = function (dt) { t += dt; if (t >= T_END) { p.finished = true; hold += dt; if (hold > 3 && !reduce) { p.reset(); } } };
         p.draw = function () {
            p.begin();
            var c = p.ctx, W = p.W, H = p.H, sq = square(8, 8, W - 16, H - 16), pos = reduce ? B : clamp((t - LEAD) / SLOT, 0, B), done = Math.min(B, Math.floor(pos)), local = pos - done;
            function SX(u) { return sq.x + u * sq.s; } function SY(v) { return sq.y + v * sq.s; }
            var cw = sq.s / GX, ch = sq.s / GY, nT = Math.max(done, 1);
            for (var gy = 0; gy < GY; gy++) {
               for (var gx = 0; gx < GX; gx++) {
                  var k = gy * GX + gx, pr = done ? cum[done - 1][k] / done : 0.5, st = Math.abs(pr - 0.5) * 2;
                  c.fillStyle = rgba(pr >= 0.5 ? CL[1] : CL[0], done ? 0.06 + 0.3 * st : 0.03); c.fillRect(SX(0) + gx * cw, SY(0) + gy * ch, cw + 0.5, ch + 0.5);
                  if (done < B && local < 0.9 && !reduce) { c.fillStyle = rgba(cell[done][k] ? CL[1] : CL[0], 0.3 * (1 - local)); c.fillRect(SX(0) + gx * cw, SY(0) + gy * ch, cw + 0.5, ch + 0.5); }   // the tree being added shows its own blocky regions
               }
            }
            c.strokeStyle = GRID; c.lineWidth = 1; c.strokeRect(sq.x + 0.5, sq.y + 0.5, sq.s - 1, sq.s - 1);
            train.forEach(function (pt) { c.fillStyle = CL[pt.c]; c.strokeStyle = 'rgba(255,255,255,.6)'; c.lineWidth = 1; c.beginPath(); c.arc(SX(pt.x), SY(pt.y), 3.3, 0, TAU); c.fill(); c.stroke(); });
            test.forEach(function (pt) { c.strokeStyle = rgba(CL[pt.c], 0.9); c.lineWidth = 1.4; c.beginPath(); c.arc(SX(pt.x), SY(pt.y), 2.6, 0, TAU); c.stroke(); });
            note(c, 'filled: training, ring: held out', W, H);
            var shown = done || (reduce ? B : 0);
            p.say('trees', (reduce ? B : done) + '/' + B);
            p.say('tree', shown ? Math.round(tAcc[Math.max(0, shown - 1)] * 100) + '%' : '--');
            p.say('forest', shown ? Math.round(fAcc[shown - 1] * 100) + '%' : '--');
            p.say('avg', Math.round(meanTree * 100) + '%');
         };
         return p;
      }

      /* ================================== SVM margin ================================== */
      function svm(card) {
         var p = A.base(card), r = PURE.rng(8), pts = [], i, CS = [100, 4, 0.8, 0.25];
         for (i = 0; i < 30; i++) { pts.push({ x: clamp(0.7 + PURE.gauss(r) * 0.095, 0.03, 0.97), y: clamp(0.33 + PURE.gauss(r) * 0.095, 0.03, 0.97), c: 1 }); }
         for (i = 0; i < 30; i++) { pts.push({ x: clamp(0.3 + PURE.gauss(r) * 0.095, 0.03, 0.97), y: clamp(0.67 + PURE.gauss(r) * 0.095, 0.03, 0.97), c: -1 }); }
         pts.push({ x: 0.43, y: 0.46, c: 1 }); pts.push({ x: 0.57, y: 0.54, c: -1 });                                                    // two awkward points near the middle
         var X = pts.map(function (q) { return [q.x, q.y]; }), Yl = pts.map(function (q) { return q.c; });
         var fits = CS.map(function (C, k) { return PURE.smo(X, Yl, C, 5 + k, 400); });
         var first = fits[0].snaps.length ? fits[0].snaps : [{ w: fits[0].w, b: fits[0].b }];
         var stride = Math.max(1, Math.ceil(first.length / 34)), train = first.filter(function (_, k) { return k % stride === 0; }); train.push({ w: fits[0].w, b: fits[0].b });
         var T_TRAIN = 0.4 + train.length * 0.07, HOLD = 1.2, MOVE = 1.3, T_END = T_TRAIN + (CS.length - 1) * (MOVE + HOLD) + HOLD, t, hold;
         function cur() {                                                                                                                  // the (w, b) at the current time, and which C it belongs to
            if (reduce) { return { w: fits[0].w, b: fits[0].b, k: 0, snap: -1 }; }
            if (t < T_TRAIN) { var si = clamp(Math.floor((t - 0.4) / 0.07), 0, train.length - 1); return { w: train[si].w, b: train[si].b, k: 0, snap: si }; }
            var u = t - T_TRAIN, seg = MOVE + HOLD, k = Math.min(CS.length - 2, Math.floor(u / seg)), loc = u - k * seg;
            if (u >= (CS.length - 1) * seg) { var L = fits[CS.length - 1]; return { w: L.w, b: L.b, k: CS.length - 1, snap: -1 }; }
            var f = easeIO(loc / MOVE), a = fits[k], b = fits[k + 1];
            return { w: [lerp(a.w[0], b.w[0], f), lerp(a.w[1], b.w[1], f)], b: lerp(a.b, b.b, f), k: f > 0.5 ? k + 1 : k, snap: -1 };
         }
         p.reset = function () { t = 0; hold = 0; p.finished = false; };
         p.update = function (dt) { t += dt; if (t >= T_END) { p.finished = true; hold += dt; if (hold > 2.2 && !reduce) { p.reset(); } } };
         p.draw = function () {
            p.begin();
            var c = p.ctx, W = p.W, H = p.H, sq = square(8, 8, W - 16, H - 16), s = cur(), w = s.w, b = s.b, nw = Math.sqrt(w[0] * w[0] + w[1] * w[1]) || 1e-6;
            function SX(u) { return sq.x + u * sq.s; } function SY(v) { return sq.y + v * sq.s; }
            function f(x, y) { return w[0] * x + w[1] * y + b; }
            var GX = 30, GY = 19, cw = sq.s / GX, ch = sq.s / GY;
            for (var gy = 0; gy < GY; gy++) { for (var gx = 0; gx < GX; gx++) { var v = f((gx + 0.5) / GX, (gy + 0.5) / GY), inM = Math.abs(v) <= 1; c.fillStyle = v >= 0 ? rgba(COL.cyan, inM ? 0.2 : 0.07) : rgba(COL.violet, inM ? 0.2 : 0.07); c.fillRect(SX(0) + gx * cw, SY(0) + gy * ch, cw + 0.5, ch + 0.5); } }
            c.save(); c.beginPath(); c.rect(sq.x, sq.y, sq.s, sq.s); c.clip();
            function line(off, dash, col, wd) {                                                                                           // w.x + b = off, drawn across the box
               c.setLineDash(dash); c.strokeStyle = col; c.lineWidth = wd; c.beginPath();
               if (Math.abs(w[1]) > Math.abs(w[0])) { c.moveTo(SX(0), SY((off - b) / w[1])); c.lineTo(SX(1), SY((off - b - w[0]) / w[1])); } else { c.moveTo(SX((off - b) / w[0]), SY(0)); c.lineTo(SX((off - b - w[1]) / w[0]), SY(1)); }
               c.stroke(); c.setLineDash([]);
            }
            line(1, [5, 4], rgba(COL.cyan, 0.9), 1.4); line(-1, [5, 4], rgba(COL.violet, 0.9), 1.4); c.shadowColor = '#fff'; c.shadowBlur = 8; line(0, [], '#ffffff', 2.4); c.shadowBlur = 0;
            c.restore();
            var nsv = 0, ok = 0;
            pts.forEach(function (pt) {
               var m = pt.c * f(pt.x, pt.y), isSv = m <= 1 + 1e-3; if (isSv) { nsv++; } if (m > 0) { ok++; }
               c.fillStyle = pt.c > 0 ? COL.cyan : COL.violet; c.strokeStyle = 'rgba(255,255,255,.6)'; c.lineWidth = 1; c.beginPath(); c.arc(SX(pt.x), SY(pt.y), 3.6, 0, TAU); c.fill(); c.stroke();
               if (isSv && s.snap < 0) { c.strokeStyle = COL.amber; c.lineWidth = 2; c.beginPath(); c.arc(SX(pt.x), SY(pt.y), 7.5, 0, TAU); c.stroke(); }
            });
            note(c, 'amber ring: support vector', W, H);
            p.say('c', s.snap >= 0 ? 'training...' : 'C = ' + CS[s.k]); p.say('margin', (2 / nw).toFixed(2)); p.say('sv', s.snap >= 0 ? '--' : String(nsv)); p.say('acc', Math.round(ok / pts.length * 100) + '%');
         };
         return p;
      }

      /* ============================ five-fold cross-validation ============================ */
      function cv(card) {
         var D = window.ML_CV;
         if (!D) { return null; }
         var p = A.base(card), res = PURE.crossValidate(D), K = res.r2.length, total = D.y.length;
         var SLOT = 1.7, LEAD = 0.5, T_END = LEAD + K * SLOT + 1.6, t, hold;
         p.reset = function () { t = 0; hold = 0; p.finished = false; };
         p.update = function (dt) { t += dt; if (t >= T_END) { p.finished = true; hold += dt; if (hold > 3 && !reduce) { p.reset(); } } };
         p.draw = function () {
            p.begin();
            var c = p.ctx, W = p.W, H = p.H, top = 26, bot = 40, rowH = (H - top - bot) / K, xL = 56, xR = W * 0.52, bx0 = W * 0.58, bx1 = W - 52, tt = reduce ? 1e9 : t, f, k;
            c.textAlign = 'left'; c.fillStyle = TEXT; c.fillText('observations, grouped by fold', xL, 14); c.fillText('R² on the held-out fold', bx0, 14);
            var doneN = 0, running = 0, lastR2 = null;
            for (f = 0; f < K; f++) {
               var y0 = top + f * rowH, start = LEAD + f * SLOT, loc = tt - start, finishedRound = loc >= SLOT * 0.9, activeRound = loc >= 0 && !finishedRound;
               c.fillStyle = activeRound ? '#fff' : TEXT; c.textAlign = 'right'; c.fillText('round ' + (f + 1), xL - 8, y0 + rowH / 2 + 4);
               var x = xL;
               for (k = 0; k < K; k++) {                                                                                                  // one segment per fold, widths in proportion to the fold sizes
                  var segW = (xR - xL) * res.size[k] / total, isTest = k === f, a = loc < 0 ? 0.35 : (isTest ? 1 : 0.5);
                  c.fillStyle = isTest ? rgba(COL.amber, loc < 0 ? 0.25 : 0.95) : rgba(COL.cyan, loc < 0 ? 0.2 : a * 0.6); c.fillRect(x + 1, y0 + 5, segW - 2, rowH - 10);
                  if (isTest && activeRound) { c.strokeStyle = '#fff'; c.lineWidth = 1.4; c.strokeRect(x + 1.5, y0 + 5.5, segW - 3, rowH - 11); }
                  x += segW;
               }
               if (activeRound && loc > 0.45 && loc < SLOT * 0.7) {                                                                       // "fit on the other four folds": a sweep across the training segments
                  var sw = (loc - 0.45) / (SLOT * 0.7 - 0.45), sx = xL + (xR - xL) * sw; c.strokeStyle = rgba(COL.green, 0.9); c.lineWidth = 2; c.beginPath(); c.moveTo(sx, y0 + 3); c.lineTo(sx, y0 + rowH - 3); c.stroke();
               }
               var grow = reduce ? 1 : clamp((loc - SLOT * 0.55) / (SLOT * 0.4), 0, 1);
               if (grow > 0) {
                  var wpx = (bx1 - bx0) * res.r2[f] * ease(grow);
                  c.fillStyle = rgba(COL.green, 0.9); c.shadowColor = rgba(COL.green, 0.5); c.shadowBlur = 8; c.fillRect(bx0, y0 + rowH * 0.3, Math.max(1, wpx), rowH * 0.4); c.shadowBlur = 0;
                  c.fillStyle = '#e8eefb'; c.textAlign = 'left'; c.fillText((res.r2[f] * ease(grow)).toFixed(4), bx0 + wpx + 6, y0 + rowH / 2 + 4);
               }
               if (grow >= 1) { doneN++; running += res.r2[f]; lastR2 = res.r2[f]; }
            }
            c.strokeStyle = AXIS; c.lineWidth = 1; c.beginPath(); c.moveTo(bx0, top); c.lineTo(bx0, top + K * rowH); c.stroke();
            var fin = reduce ? 1 : clamp((tt - LEAD - K * SLOT) / 0.9, 0, 1);
            if (fin > 0) {                                                                                                                 // the mean and one standard deviation either side
               var mx = bx0 + (bx1 - bx0) * res.mean, sdw = (bx1 - bx0) * res.std;
               c.fillStyle = rgba(COL.amber, 0.14 * fin); c.fillRect(mx - sdw, top - 4, sdw * 2, K * rowH + 8);
               c.setLineDash([5, 4]); c.strokeStyle = rgba(COL.amber, 0.95 * fin); c.lineWidth = 1.6; c.beginPath(); c.moveTo(mx, top - 4); c.lineTo(mx, top + K * rowH + 4); c.stroke(); c.setLineDash([]);
               c.fillStyle = rgba(COL.amber, fin); c.textAlign = 'center'; c.fillText('mean ' + res.mean.toFixed(4) + ' ± ' + res.std.toFixed(4), W / 2, H - 12);
            }
            p.say('round', Math.min(K, doneN + (doneN < K && tt >= LEAD ? 1 : 0)) + '/' + K); p.say('fold', lastR2 === null ? '--' : lastR2.toFixed(4)); p.say('mean', doneN ? (running / doneN).toFixed(4) : '--');
         };
         return p;
      }

      return { tsne: tsne, forest: forest, svm: svm, cv: cv };
   };
})();
