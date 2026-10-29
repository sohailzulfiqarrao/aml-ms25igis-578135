/* More machine learning algorithms, running live on the ML hub landing page (loaded before ml-viz.js, which mounts them).
     dendrogram  Ward hierarchical clustering: the merges and the dendrogram that records them (Assignment 2)
     dbscan      density-based clustering on two moons: core, border and noise points (Assignment 2)
     gmm         a Gaussian mixture fitted by expectation maximisation (Assignment 2)
     pca         principal components of the eight predictors of Assignment 1 (real data)
     overfit     polynomial fits of rising degree to noisy data: under- and over-fitting
     knn         k nearest neighbours and its decision regions for k = 1, 5 and 15
     tree        a decision tree grown split by split with the Gini criterion
     logistic    logistic regression learning a decision boundary by gradient descent
     gd          gradient descent, momentum and Adam rolling down the same loss surface
   Every algorithm is run for real by the code that draws it; only the point clouds are made up (fixed seeds, so they never
   change), except PCA, which uses the data of Assignment 1. */
window.MLVIZ_EXTRA = function (A) {
   'use strict';
   var COL = A.COL, TAU = A.TAU, clamp = A.clamp, ease = A.ease, easeIO = A.easeIO, lerp = A.lerp, rgba = A.rgba, MONO = A.MONO, TEXT = A.TEXT, GRID = A.GRID, AXIS = A.AXIS, reduce = A.reduce;
   var PAL = [COL.green, COL.cyan, COL.amber, COL.violet];
   var FONT = 'ui-monospace, "Cascadia Code", Consolas, monospace';
   function d2(a, b) { return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y); }
   function mix(hexes, w) {                                   // blend colours by weights
      var r = 0, g = 0, b = 0, i;
      for (i = 0; i < hexes.length; i++) { var n = parseInt(hexes[i].slice(1), 16); r += (n >> 16) * w[i]; g += ((n >> 8) & 255) * w[i]; b += (n & 255) * w[i]; }
      return 'rgb(' + Math.round(r) + ',' + Math.round(g) + ',' + Math.round(b) + ')';
   }
   function shuffle(arr, r) { for (var i = arr.length - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)), t = arr[i]; arr[i] = arr[j]; arr[j] = t; } return arr; }
   // a square drawing box in the canvas: unit coordinates (0..1) map into it
   function square(x0, y0, w, h) { var s = Math.max(8, Math.min(w, h)); return { x: x0 + (w - s) / 2, y: y0 + (h - s) / 2, s: s }; }

   /* ============================ hierarchical (Ward) ========================= */
   function dendrogram(card) {
      var p = A.base(card), r = A.rng(4), blobs = [[0.22, 0.3], [0.72, 0.26], [0.5, 0.76]], pts = [], i, j, n;
      blobs.forEach(function (b) { for (j = 0; j < 8; j++) { pts.push({ x: clamp(b[0] + A.gauss(r) * 0.075, 0.04, 0.96), y: clamp(b[1] + A.gauss(r) * 0.075, 0.04, 0.96) }); } });
      n = pts.length;
      var nodes = [], active = [], merges = [], nid = n;
      pts.forEach(function (pt, k) { nodes[k] = { id: k, size: 1, h: 0, cx: pt.x, cy: pt.y, leaves: [k] }; active.push(nodes[k]); });
      while (active.length > 1) {                                           // Ward linkage on cluster centroids, like scipy: sqrt(2 na nb / (na + nb)) * distance
         var best = 1e9, ai = 0, bi = 1;
         for (i = 0; i < active.length; i++) { for (j = i + 1; j < active.length; j++) { var a = active[i], b = active[j], dd = Math.sqrt(2 * a.size * b.size / (a.size + b.size)) * Math.sqrt((a.cx - b.cx) * (a.cx - b.cx) + (a.cy - b.cy) * (a.cy - b.cy)); if (dd < best) { best = dd; ai = i; bi = j; } } }
         var L = active[ai], R = active[bi], s = L.size + R.size;
         var node = { id: nid++, left: L, right: R, size: s, h: best, cx: (L.cx * L.size + R.cx * R.size) / s, cy: (L.cy * L.size + R.cy * R.size) / s, leaves: L.leaves.concat(R.leaves) };
         nodes[node.id] = node; merges.push(node); active.splice(bi, 1); active[ai] = node;
      }
      var root = active[0], HMAX = root.h, order = [];
      (function walk(nd) { if (nd.left) { walk(nd.left); walk(nd.right); } else { order.push(nd.id); } })(root);
      (function pos(nd) { if (!nd.left) { nd.px = order.indexOf(nd.id); return nd.px; } nd.px = (pos(nd.left) + pos(nd.right)) / 2; return nd.px; })(root);
      var K = 3, cutH = (merges[n - K - 1].h + merges[n - K].h) / 2, group = [], g = 0;
      (function cut(nd) { if (!nd.left || nd.h <= cutH) { nd.leaves.forEach(function (lf) { group[lf] = g; }); nd.group = g; g++; } else { cut(nd.left); cut(nd.right); } })(root);
      var T0 = 0.6, MS = 0.27, GROW = 0.5, T_CUT = T0 + (n - 1) * MS + 0.5, CUTD = 1.6, T_END = T_CUT + CUTD + 0.2, t, hold;
      p.reset = function () { t = 0; hold = 0; p.finished = false; };
      p.update = function (dt) { t += dt; if (t >= T_END) { p.finished = true; hold += dt; if (hold > 2.8 && !reduce) { p.reset(); } } };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, sq = square(10, 12, W * 0.4 - 12, H - 24), done = 0, lastH = 0, i2;
         var cp = reduce ? 1 : ease((t - T_CUT) / CUTD), tt = reduce ? 1e9 : t;
         function SX(u) { return sq.x + u * sq.s; } function SY(v) { return sq.y + v * sq.s; }
         var dx0 = W * 0.43, dx1 = W - 12, dy0 = 26, dy1 = H - 16;
         function DX(k) { return dx0 + (k + 0.5) / n * (dx1 - dx0); } function DY(h) { return dy1 - Math.sqrt(Math.max(0, h) / HMAX) * (dy1 - dy0 - 6); }          // square-root height scale: early merges stay readable
         c.strokeStyle = GRID; c.lineWidth = 1; c.strokeRect(sq.x + 0.5, sq.y + 0.5, sq.s - 1, sq.s - 1);
         var hot = {};
         for (i2 = 0; i2 < merges.length; i2++) {
            var m = merges[i2], q = clamp((tt - T0 - i2 * MS) / GROW, 0, 1); if (q <= 0) { continue; }
            if (q >= 1) { done++; lastH = m.h; }
            if (q < 1) { m.leaves.forEach(function (lf) { hot[lf] = 1; }); }
            var lx = DX(m.left.px), rx = DX(m.right.px), ly = DY(m.left.h), ry = DY(m.right.h), hy = DY(m.h), e = ease(q);
            var below = cp > 0 && m.h <= cutH, col = below ? PAL[m.group !== undefined ? m.group % 4 : 0] : 'rgba(226,232,240,.85)';
            if (below) { var gi = group[m.leaves[0]]; col = PAL[gi % 4]; }
            c.strokeStyle = q < 1 ? COL.amber : col; c.lineWidth = q < 1 ? 2.2 : 1.6; c.beginPath();
            c.moveTo(lx, ly); c.lineTo(lx, lerp(ly, hy, e)); c.moveTo(rx, ry); c.lineTo(rx, lerp(ry, hy, e));
            if (q > 0.5) { var k2 = ease((q - 0.5) / 0.5); c.moveTo(lx, hy); c.lineTo(lerp(lx, rx, k2), hy); }
            c.stroke();
            if (q < 1 && !reduce) { c.strokeStyle = rgba(COL.amber, 0.7 * (1 - q)); c.lineWidth = 1.2; c.beginPath(); c.moveTo(SX(m.left.cx), SY(m.left.cy)); c.lineTo(SX(m.right.cx), SY(m.right.cy)); c.stroke(); }
         }
         for (i2 = 0; i2 < n; i2++) {                                                                     // the leaves and the points
            var lg = cp > 0.3 ? PAL[group[i2] % 4] : (hot[i2] ? COL.amber : COL.cyan);
            c.fillStyle = lg; c.beginPath(); c.arc(SX(pts[i2].x), SY(pts[i2].y), hot[i2] ? 4.4 : 3.6, 0, TAU); c.fill();
            c.fillStyle = lg; c.globalAlpha = 0.8; c.beginPath(); c.arc(DX(order.indexOf(i2)), dy1 + 5, 2, 0, TAU); c.fill(); c.globalAlpha = 1;
         }
         if (cp > 0) {                                                                                    // the cut
            var cy = lerp(DY(HMAX), DY(cutH), cp);
            c.setLineDash([5, 4]); c.strokeStyle = rgba(COL.amber, 0.95); c.lineWidth = 1.5; c.beginPath(); c.moveTo(dx0 - 6, cy); c.lineTo(dx1, cy); c.stroke(); c.setLineDash([]);
            c.fillStyle = COL.amber; c.textAlign = 'left'; c.fillText('cut: ' + (cp > 0.95 ? 'k = ' + K : 'height ' + lerp(HMAX, cutH, cp).toFixed(2)), dx0, cy - 5);
         }
         c.fillStyle = TEXT; c.textAlign = 'left'; c.fillText('merge height (square-root scale)', dx0 - 6, 11);
         p.say('merges', done + '/' + merges.length); p.say('height', lastH ? lastH.toFixed(2) : '--'); p.say('clusters', String(cp > 0.95 ? K : n - done));
      };
      return p;
   }

   /* ================================= DBSCAN ================================= */
   function dbscan(card) {
      var p = A.base(card), r = A.rng(12), pts = [], i, n, EPS = 0.07, MINP = 5;
      for (i = 0; i < 56; i++) { var th = Math.PI * i / 55; pts.push({ x: 0.16 + (Math.cos(th) + 1) / 3 * 0.68 + A.gauss(r) * 0.012, y: 0.8 - (Math.sin(th) + 0.5) / 1.5 * 0.58 + A.gauss(r) * 0.012 }); }
      for (i = 0; i < 56; i++) { var th2 = Math.PI * i / 55; pts.push({ x: 0.16 + (1 - Math.cos(th2) + 1) / 3 * 0.68 + A.gauss(r) * 0.012, y: 0.8 - (0.5 - Math.sin(th2) + 0.5) / 1.5 * 0.58 + A.gauss(r) * 0.012 }); }
      for (i = 0; i < 14; i++) { pts.push({ x: 0.05 + r() * 0.9, y: 0.05 + r() * 0.9 }); }                       // 14 stray points
      n = pts.length;
      var order = shuffle(pts.map(function (_, k) { return k; }), r), lab = pts.map(function () { return 0; }), events = [], cid = 0;
      function neigh(k) { var out = []; for (var q = 0; q < n; q++) { if (d2(pts[k], pts[q]) <= EPS * EPS) { out.push(q); } } return out; }
      order.forEach(function (o) {
         if (lab[o] !== 0) { return; }
         var nb = neigh(o);
         if (nb.length < MINP) { lab[o] = -1; events.push({ i: o, noise: true }); return; }
         cid++; lab[o] = cid; events.push({ i: o, c: cid, core: true });
         var queue = nb.slice();
         while (queue.length) {
            var q = queue.shift();
            if (lab[q] === -1) { lab[q] = cid; events.push({ i: q, c: cid, core: false }); continue; }
            if (lab[q] !== 0) { continue; }
            lab[q] = cid; var nb2 = neigh(q);
            if (nb2.length >= MINP) { events.push({ i: q, c: cid, core: true }); queue = queue.concat(nb2); } else { events.push({ i: q, c: cid, core: false }); }
         }
      });
      var LEAD = 0.7, STEP = 0.05, T_END = LEAD + events.length * STEP + 0.3, t, hold;
      p.reset = function () { t = 0; hold = 0; p.finished = false; };
      p.update = function (dt) { t += dt; if (t >= T_END) { p.finished = true; hold += dt; if (hold > 2.6 && !reduce) { p.reset(); } } };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, sq = square(8, 8, W - 16, H - 16), k = reduce ? events.length : clamp(Math.floor((t - LEAD) / STEP), 0, events.length);
         function SX(u) { return sq.x + u * sq.s; } function SY(v) { return sq.y + v * sq.s; }
         var state = pts.map(function () { return null; }), clusters = 0, noise = 0, e;
         for (i = 0; i < k; i++) { e = events[i]; state[e.i] = e; if (e.c > clusters) { clusters = e.c; } }
         for (i = 0; i < k; i++) { if (events[i].noise && state[events[i].i] === events[i]) { noise++; } }
         for (i = 0; i < n; i++) {
            var s = state[i], col = s ? (s.noise ? COL.grey || '#64748b' : PAL[(s.c - 1) % 4]) : '#cbd5e1', a = s ? (s.noise ? 0.55 : (s.core ? 0.95 : 0.6)) : 0.35;
            c.fillStyle = rgba(col.charAt(0) === '#' ? col : '#64748b', a); c.beginPath(); c.arc(SX(pts[i].x), SY(pts[i].y), s && s.core ? 3.6 : 3, 0, TAU); c.fill();
         }
         if (k > 0 && k <= events.length && !reduce) {                                                      // the point being examined and its eps neighbourhood
            e = events[Math.min(k, events.length) - 1]; var pt = pts[e.i];
            c.setLineDash([4, 4]); c.strokeStyle = e.noise ? 'rgba(148,163,184,.7)' : rgba(PAL[(e.c - 1) % 4], 0.95); c.lineWidth = 1.4;
            c.beginPath(); c.arc(SX(pt.x), SY(pt.y), EPS * sq.s, 0, TAU); c.stroke(); c.setLineDash([]);
            c.fillStyle = '#fff'; c.beginPath(); c.arc(SX(pt.x), SY(pt.y), 2.2, 0, TAU); c.fill();
         }
         c.fillStyle = TEXT; c.textAlign = 'right'; c.fillText('dashed circle = eps', W - 10, H - 10);
         p.say('clusters', String(clusters)); p.say('noise', String(noise)); p.say('params', 'eps ' + EPS + ' · min ' + MINP);
      };
      return p;
   }

   /* ============================ Gaussian mixture (EM) ======================= */
   function gmm(card) {
      var p = A.base(card), r = A.rng(31), pts = [], i, k, K = 3;
      [{ m: [0.26, 0.3], s: [0.085, 0.03], a: 0.5, w: 52 }, { m: [0.7, 0.36], s: [0.035, 0.09], a: -0.4, w: 52 }, { m: [0.48, 0.76], s: [0.09, 0.035], a: 0.15, w: 52 }].forEach(function (cp) {
         for (i = 0; i < cp.w; i++) { var u = A.gauss(r) * cp.s[0], v = A.gauss(r) * cp.s[1]; pts.push({ x: clamp(cp.m[0] + u * Math.cos(cp.a) - v * Math.sin(cp.a), 0.03, 0.97), y: clamp(cp.m[1] + u * Math.sin(cp.a) + v * Math.cos(cp.a), 0.03, 0.97) }); }
      });
      var n = pts.length;
      function pdf(x, y, m, S) { var det = S[0] * S[2] - S[1] * S[1], dx = x - m[0], dy = y - m[1], q = (S[2] * dx * dx - 2 * S[1] * dx * dy + S[0] * dy * dy) / det; return Math.exp(-0.5 * q) / (TAU * Math.sqrt(det)); }
      var mu = [[0.4, 0.5], [0.5, 0.55], [0.55, 0.45]], cov = [[0.03, 0, 0.03], [0.03, 0, 0.03], [0.03, 0, 0.03]], pi = [1 / 3, 1 / 3, 1 / 3], snaps = [];
      function estep() {                                                                  // responsibilities and the log-likelihood
         var R = [], ll = 0;
         for (var q = 0; q < n; q++) { var row = [], tot = 0; for (var c2 = 0; c2 < K; c2++) { var v = pi[c2] * pdf(pts[q].x, pts[q].y, mu[c2], cov[c2]); row.push(v); tot += v; } ll += Math.log(tot + 1e-300); R.push(row.map(function (v2) { return v2 / (tot + 1e-300); })); }
         return { R: R, ll: ll };
      }
      var prev = -1e18;
      for (var it = 0; it < 22; it++) {
         var E = estep();
         snaps.push({ mu: mu.map(function (m) { return m.slice(); }), cov: cov.map(function (s) { return s.slice(); }), pi: pi.slice(), R: E.R, ll: E.ll });
         if (Math.abs(E.ll - prev) < 0.02 && it > 3) { break; }
         prev = E.ll;
         for (var c3 = 0; c3 < K; c3++) {                                                // M step
            var Nk = 0, mx = 0, my = 0, q2;
            for (q2 = 0; q2 < n; q2++) { Nk += E.R[q2][c3]; mx += E.R[q2][c3] * pts[q2].x; my += E.R[q2][c3] * pts[q2].y; }
            mx /= Nk; my /= Nk; var sxx = 0, sxy = 0, syy = 0;
            for (q2 = 0; q2 < n; q2++) { var rr = E.R[q2][c3], ddx = pts[q2].x - mx, ddy = pts[q2].y - my; sxx += rr * ddx * ddx; sxy += rr * ddx * ddy; syy += rr * ddy * ddy; }
            mu[c3] = [mx, my]; cov[c3] = [sxx / Nk + 1e-4, sxy / Nk, syy / Nk + 1e-4]; pi[c3] = Nk / n;
         }
      }
      var STEP = 0.55, LEAD = 0.6, T_END = LEAD + (snaps.length - 1) * STEP + 0.6, t, hold;
      p.reset = function () { t = 0; hold = 0; p.finished = false; };
      p.update = function (dt) { t += dt; if (t >= T_END) { p.finished = true; hold += dt; if (hold > 2.8 && !reduce) { p.reset(); } } };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, sq = square(8, 8, W - 16, H - 16);
         function SX(u) { return sq.x + u * sq.s; } function SY(v) { return sq.y + v * sq.s; }
         var pos = reduce ? snaps.length - 1 : clamp((t - LEAD) / STEP, 0, snaps.length - 1), a = Math.floor(pos), b = Math.min(snaps.length - 1, a + 1), f = easeIO(pos - a);
         var S0 = snaps[a], S1 = snaps[b], cols = [COL.green, COL.cyan, COL.amber];
         c.strokeStyle = GRID; c.lineWidth = 1; c.strokeRect(sq.x + 0.5, sq.y + 0.5, sq.s - 1, sq.s - 1);
         for (i = 0; i < n; i++) { var w = [0, 1, 2].map(function (q) { return lerp(S0.R[i][q], S1.R[i][q], f); }); c.fillStyle = mix(cols, w); c.beginPath(); c.arc(SX(pts[i].x), SY(pts[i].y), 3, 0, TAU); c.fill(); }
         for (k = 0; k < K; k++) {                                                                         // each component as its 1 and 2 sigma ellipses
            var m = [lerp(S0.mu[k][0], S1.mu[k][0], f), lerp(S0.mu[k][1], S1.mu[k][1], f)], S = [lerp(S0.cov[k][0], S1.cov[k][0], f), lerp(S0.cov[k][1], S1.cov[k][1], f), lerp(S0.cov[k][2], S1.cov[k][2], f)];
            var mid = (S[0] + S[2]) / 2, dif = Math.sqrt(Math.pow((S[0] - S[2]) / 2, 2) + S[1] * S[1]), l1 = mid + dif, l2 = Math.max(1e-6, mid - dif), ang = 0.5 * Math.atan2(2 * S[1], S[0] - S[2]);
            [1, 2].forEach(function (sg) { c.strokeStyle = rgba(cols[k], sg === 1 ? 0.95 : 0.5); c.lineWidth = sg === 1 ? 2 : 1.2; c.beginPath(); c.ellipse(SX(m[0]), SY(m[1]), Math.sqrt(l1) * sg * sq.s, Math.sqrt(l2) * sg * sq.s, ang, 0, TAU); c.stroke(); });
            c.fillStyle = '#fff'; c.strokeStyle = cols[k]; c.lineWidth = 2; c.beginPath(); c.arc(SX(m[0]), SY(m[1]), 4.5, 0, TAU); c.fill(); c.stroke();
         }
         var shown = Math.round(pos);
         p.say('iter', String(shown)); p.say('ll', lerp(S0.ll, S1.ll, f).toFixed(1)); p.say('state', pos >= snaps.length - 1.05 ? 'converged' : 'E step, M step');
      };
      return p;
   }

   /* ================================== PCA =================================== */
   function pca(card) {
      var D = window.ML_COLLINEAR;
      if (!D) { return null; }
      var p = A.base(card), names = D.names, K = names.length, n = D.rows.length, i, j;
      var cols = names.map(function (nm, c) { return D.rows.map(function (row) { return row[c]; }); });
      function corr(a, b) { var ma = A.mean(a), mb = A.mean(b), sab = 0, saa = 0, sbb = 0, q; for (q = 0; q < n; q++) { sab += (a[q] - ma) * (b[q] - mb); saa += (a[q] - ma) * (a[q] - ma); sbb += (b[q] - mb) * (b[q] - mb); } return sab / Math.sqrt(saa * sbb); }
      var M = [], V = [];
      for (i = 0; i < K; i++) { M[i] = []; V[i] = []; for (j = 0; j < K; j++) { M[i][j] = i === j ? 1 : corr(cols[i], cols[j]); V[i][j] = i === j ? 1 : 0; } }
      for (var sweep = 0; sweep < 60; sweep++) {                                                  // Jacobi eigenvalue method on the correlation matrix
         var off = 0; for (i = 0; i < K; i++) { for (j = 0; j < K; j++) { if (i !== j) { off += M[i][j] * M[i][j]; } } } if (off < 1e-20) { break; }
         for (var a = 0; a < K - 1; a++) { for (var b = a + 1; b < K; b++) {
            if (Math.abs(M[a][b]) < 1e-14) { continue; }
            var th = (M[b][b] - M[a][a]) / (2 * M[a][b]), tt = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1)), cs = 1 / Math.sqrt(tt * tt + 1), sn = tt * cs, q;
            for (q = 0; q < K; q++) { var akp = M[q][a], akq = M[q][b]; M[q][a] = cs * akp - sn * akq; M[q][b] = sn * akp + cs * akq; }
            for (q = 0; q < K; q++) { var apk = M[a][q], aqk = M[b][q]; M[a][q] = cs * apk - sn * aqk; M[b][q] = sn * apk + cs * aqk; }
            for (q = 0; q < K; q++) { var vp = V[q][a], vq = V[q][b]; V[q][a] = cs * vp - sn * vq; V[q][b] = sn * vp + cs * vq; }
         } }
      }
      var pcs = []; for (i = 0; i < K; i++) { var vec = []; for (j = 0; j < K; j++) { vec.push(V[j][i]); } pcs.push({ ev: M[i][i], vec: vec }); }
      pcs.sort(function (x, y) { return y.ev - x.ev; });
      var tot = 0; pcs.forEach(function (pc) { tot += pc.ev; });
      pcs.forEach(function (pc) { pc.share = pc.ev / tot; if (pc.vec[0] < 0) { pc.vec = pc.vec.map(function (v) { return -v; }); } });         // sign: NDVI positive
      var cum = [], run = 0; pcs.forEach(function (pc) { run += pc.share; cum.push(run); });
      var n90 = cum.findIndex(function (v) { return v >= 0.9; }) + 1;
      var T_BARS = 0.4, T_CUM = 2.2, T_LOAD = 3.4, T_SWAP = 6.4, T_END = 8.2, t, hold;
      p.reset = function () { t = 0; hold = 0; p.finished = false; };
      p.update = function (dt) { t += dt; if (t >= T_END) { p.finished = true; hold += dt; if (hold > 3 && !reduce) { p.reset(); } } };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, tt = reduce ? 1e9 : t;
         var x0 = 40, x1 = W * 0.47, y0 = 26, y1 = H - 40, bw = (x1 - x0) / K;
         function SY(v) { return y1 - v * (y1 - y0); }
         c.lineWidth = 1; c.fillStyle = TEXT; c.textAlign = 'right';
         [0, 0.5, 1].forEach(function (g) { c.strokeStyle = GRID; c.beginPath(); c.moveTo(x0, SY(g)); c.lineTo(x1, SY(g)); c.stroke(); c.fillText(Math.round(g * 100) + '%', x0 - 5, SY(g) + 4); });
         c.textAlign = 'left'; c.fillText('variance explained', x0, y0 - 11);
         for (i = 0; i < K; i++) {
            var q = reduce ? 1 : ease((tt - T_BARS - i * 0.12) / 0.6), h = pcs[i].share * q;
            c.fillStyle = rgba(COL.cyan, 0.9); c.shadowColor = rgba(COL.cyan, 0.5); c.shadowBlur = 8; c.fillRect(x0 + i * bw + bw * 0.16, SY(h), bw * 0.68, (y1 - SY(h))); c.shadowBlur = 0;
            c.fillStyle = TEXT; c.textAlign = 'center'; c.fillText('PC' + (i + 1), x0 + (i + 0.5) * bw, y1 + 14);
            if (q > 0.9) { c.fillStyle = '#e8eefb'; c.fillText(Math.round(pcs[i].share * 100) + '', x0 + (i + 0.5) * bw, SY(h) - 5); }
         }
         var cq = reduce ? 1 : clamp((tt - T_CUM) / 1.2, 0, 1);
         if (cq > 0) {                                                                                  // the cumulative line, and the 90% mark
            c.strokeStyle = COL.amber; c.lineWidth = 2.2; c.shadowColor = rgba(COL.amber, 0.6); c.shadowBlur = 8; c.beginPath();
            for (i = 0; i < K; i++) { var xx = x0 + (i + 0.5) * bw, yy = SY(cum[i]); if (i / (K - 1) > cq) { var pr = (cq - (i - 1) / (K - 1)) * (K - 1); c.lineTo(lerp(x0 + (i - 0.5) * bw, xx, pr), lerp(SY(cum[i - 1]), yy, pr)); break; } if (i === 0) { c.moveTo(xx, yy); } else { c.lineTo(xx, yy); } }
            c.stroke(); c.shadowBlur = 0;
            c.setLineDash([4, 4]); c.strokeStyle = rgba(COL.green, 0.7); c.lineWidth = 1; c.beginPath(); c.moveTo(x0, SY(0.9)); c.lineTo(x1, SY(0.9)); c.stroke(); c.setLineDash([]);
            c.fillStyle = COL.green; c.textAlign = 'right'; c.fillText('90%', x1, SY(0.9) - 4);
         }
         // loadings of PC1, crossfading to PC2
         var lx0 = W * 0.52, lx1 = W - 12, zero = lx0 + (lx1 - lx0 + 36) / 2 + 22, half = (lx1 - zero), ly0 = 34, rowH = (H - 40 - ly0) / K;
         var sw = reduce ? 0 : easeIO((tt - T_SWAP) / 0.9), la = reduce ? 1 : ease((tt - T_LOAD) / 0.8), shownPc = sw > 0.5 ? 1 : 0;
         c.fillStyle = TEXT; c.textAlign = 'left'; c.fillText('loadings of PC' + (shownPc + 1) + ' (' + (pcs[shownPc].share * 100).toFixed(1) + '%)', lx0, y0 - 11);
         c.strokeStyle = AXIS; c.lineWidth = 1; c.beginPath(); c.moveTo(zero, ly0 - 6); c.lineTo(zero, ly0 + K * rowH); c.stroke();
         for (i = 0; i < K; i++) {
            var val = lerp(pcs[0].vec[i], pcs[1].vec[i], sw) * la, yc = ly0 + (i + 0.5) * rowH, wpx = val / 0.7 * half * 0.92;
            c.fillStyle = TEXT; c.textAlign = 'right'; c.fillText(names[i], lx0 + 34, yc + 4);
            if (la > 0.01) { c.fillStyle = val >= 0 ? rgba(COL.cyan, 0.9) : rgba(COL.violet, 0.9); c.fillRect(Math.min(zero, zero + wpx), yc - rowH * 0.3, Math.abs(wpx), rowH * 0.6); c.fillStyle = '#e8eefb'; c.textAlign = val >= 0 ? 'left' : 'right'; c.fillText(val.toFixed(2), zero + wpx + (val >= 0 ? 4 : -4), yc + 4); }
         }
         p.say('pc1', (pcs[0].share * 100).toFixed(1) + '%'); p.say('pc2', (cum[1] * 100).toFixed(1) + '%'); p.say('n90', String(n90));
      };
      return p;
   }

   /* ============================ overfitting (polynomials) =================== */
   function overfit(card) {
      var p = A.base(card), r = A.rng(17), i, N = 14, DEG = [1, 2, 3, 4, 6, 8, 10];
      function truth(x) { return Math.sin(TAU * x); }
      var train = [], test = [];
      for (i = 0; i < N; i++) { var x = (i + 0.3 + r() * 0.5) / N; train.push({ x: x, y: truth(x) + A.gauss(r) * 0.22 }); }
      for (i = 0; i < 60; i++) { var x2 = r(); test.push({ x: x2, y: truth(x2) + A.gauss(r) * 0.22 }); }
      function fit(d) {                                                                           // least squares on powers of 2x - 1 (normal equations)
         var m = d + 1, G = [], bvec = [], a, b, q;
         for (a = 0; a < m; a++) { G[a] = []; for (b = 0; b < m; b++) { var s = 0; for (q = 0; q < N; q++) { s += Math.pow(2 * train[q].x - 1, a + b); } G[a][b] = s + (a === b ? 1e-9 : 0); } var sy = 0; for (q = 0; q < N; q++) { sy += train[q].y * Math.pow(2 * train[q].x - 1, a); } bvec[a] = sy; }
         for (a = 0; a < m; a++) {
            var piv = a; for (b = a + 1; b < m; b++) { if (Math.abs(G[b][a]) > Math.abs(G[piv][a])) { piv = b; } }
            var tmp = G[a]; G[a] = G[piv]; G[piv] = tmp; var tb = bvec[a]; bvec[a] = bvec[piv]; bvec[piv] = tb;
            for (b = a + 1; b < m; b++) { var f = G[b][a] / G[a][a]; for (q = a; q < m; q++) { G[b][q] -= f * G[a][q]; } bvec[b] -= f * bvec[a]; }
         }
         var w = []; for (a = m - 1; a >= 0; a--) { var sum = bvec[a]; for (b = a + 1; b < m; b++) { sum -= G[a][b] * w[b]; } w[a] = sum / G[a][a]; }
         return function (x) { var v = 0, tp = 1, u = 2 * x - 1; for (var k = 0; k < m; k++) { v += w[k] * tp; tp *= u; } return v; };
      }
      function rmse(f, set) { var s = 0; set.forEach(function (pt) { s += Math.pow(f(pt.x) - pt.y, 2); }); return Math.sqrt(s / set.length); }
      var NS = 90, fits = DEG.map(function (d) { var f = fit(d), ys = []; for (var k = 0; k <= NS; k++) { ys.push(f(k / NS)); } return { d: d, ys: ys, tr: rmse(f, train), te: rmse(f, test) }; });
      var bestTe = fits.reduce(function (m, f) { return f.te < m.te ? f : m; }, fits[0]);
      var STEP = 1.2, LEAD = 0.5, T_END = LEAD + (DEG.length - 1) * STEP + 0.9, t, hold;
      p.reset = function () { t = 0; hold = 0; p.finished = false; };
      p.update = function (dt) { t += dt; if (t >= T_END) { p.finished = true; hold += dt; if (hold > 2.8 && !reduce) { p.reset(); } } };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, pos = reduce ? DEG.length - 1 : clamp((t - LEAD) / STEP, 0, DEG.length - 1), a = Math.floor(pos), b = Math.min(DEG.length - 1, a + 1), f = easeIO(pos - a);
         var px0 = 34, px1 = W * 0.62, py0 = 16, py1 = H - 28, YR = 2.1;
         function X(x) { return px0 + x * (px1 - px0); } function Y(y) { return (py0 + py1) / 2 - y / YR * (py1 - py0) / 2; }
         c.save(); c.beginPath(); c.rect(px0, py0, px1 - px0, py1 - py0); c.clip();
         c.strokeStyle = GRID; c.lineWidth = 1; [-1, 0, 1].forEach(function (g) { c.beginPath(); c.moveTo(px0, Y(g)); c.lineTo(px1, Y(g)); c.stroke(); });
         c.setLineDash([4, 4]); c.strokeStyle = rgba(COL.green, 0.6); c.lineWidth = 1.4; c.beginPath(); for (i = 0; i <= NS; i++) { var tx = i / NS; (i === 0 ? c.moveTo : c.lineTo).call(c, X(tx), Y(truth(tx))); } c.stroke(); c.setLineDash([]);
         var over = fits[a].d > bestTe.d && fits[a].te > bestTe.te * 1.6, colr = over ? '#f87171' : COL.amber;
         c.strokeStyle = colr; c.lineWidth = 2.6; c.shadowColor = rgba(colr, 0.6); c.shadowBlur = 9; c.beginPath();
         for (i = 0; i <= NS; i++) { var yy = lerp(fits[a].ys[i], fits[b].ys[i], f); (i === 0 ? c.moveTo : c.lineTo).call(c, X(i / NS), Y(yy)); } c.stroke(); c.shadowBlur = 0;
         train.forEach(function (pt) { c.fillStyle = COL.cyan; c.beginPath(); c.arc(X(pt.x), Y(pt.y), 3.6, 0, TAU); c.fill(); });
         c.restore();
         c.strokeStyle = AXIS; c.beginPath(); c.moveTo(px0, py0); c.lineTo(px0, py1); c.lineTo(px1, py1); c.stroke();
         c.fillStyle = TEXT; c.textAlign = 'left'; c.fillText('x', px1 - 8, py1 + 14); c.fillText('green dashes: the true curve', px0 + 6, py1 + 14);
         // error against degree, filled in as the degree rises
         var ex0 = W * 0.7, ex1 = W - 12, ey0 = py0, ey1 = py1, LO = Math.log(0.06), HI = Math.log(12);
         function EX(k) { return ex0 + (k + 0.5) / DEG.length * (ex1 - ex0); } function EY(v) { return ey1 - (Math.log(clamp(v, 0.06, 12)) - LO) / (HI - LO) * (ey1 - ey0); }
         c.strokeStyle = AXIS; c.beginPath(); c.moveTo(ex0, ey0); c.lineTo(ex0, ey1); c.lineTo(ex1, ey1); c.stroke();
         [0.1, 1, 10].forEach(function (g) { c.strokeStyle = GRID; c.beginPath(); c.moveTo(ex0, EY(g)); c.lineTo(ex1, EY(g)); c.stroke(); c.fillStyle = TEXT; c.textAlign = 'right'; c.fillText(String(g), ex0 - 3, EY(g) + 4); });
         c.textAlign = 'center'; c.fillStyle = TEXT; DEG.forEach(function (d, k) { c.fillText(String(d), EX(k), ey1 + 14); });
         c.textAlign = 'left'; c.fillText('error (RMSE) by degree', ex0, ey0 - 4);
         [['tr', COL.green], ['te', COL.cyan]].forEach(function (s) {
            c.strokeStyle = s[1]; c.lineWidth = 2; c.beginPath();
            for (var k = 0; k <= a; k++) { (k === 0 ? c.moveTo : c.lineTo).call(c, EX(k), EY(fits[k][s[0]])); }
            if (f > 0 && b > a) { c.lineTo(lerp(EX(a), EX(b), f), lerp(EY(fits[a][s[0]]), EY(fits[b][s[0]]), f)); }
            c.stroke();
            for (k = 0; k <= a; k++) { c.fillStyle = s[1]; c.beginPath(); c.arc(EX(k), EY(fits[k][s[0]]), 3, 0, TAU); c.fill(); }
         });
         var fa = fits[a];
         p.say('degree', String(fa.d)); p.say('train', fa.tr.toFixed(2)); p.say('test', fa.te.toFixed(2));
         p.say('verdict', fa.te > bestTe.te * 1.6 ? (fa.d < bestTe.d ? 'underfit' : 'overfit') : (fa.d === bestTe.d ? 'best fit' : (fa.d < bestTe.d ? 'underfit' : 'fits')));
      };
      return p;
   }

   /* ================================== k-NN ================================== */
   function knn(card) {
      var p = A.base(card), r = A.rng(5), pts = [], i, j, KS = [1, 5, 15];
      [[0.3, 0.38, 0], [0.4, 0.7, 0], [0.7, 0.55, 1], [0.62, 0.28, 1]].forEach(function (cl) { for (i = 0; i < 20; i++) { pts.push({ x: clamp(cl[0] + A.gauss(r) * 0.1, 0.03, 0.97), y: clamp(cl[1] + A.gauss(r) * 0.1, 0.03, 0.97), c: cl[2] }); } });
      var n = pts.length, CLS = [COL.green, COL.violet], CN = ['A', 'B'];
      function vote(x, y, k) { var ds = pts.map(function (pt, q) { return { d: (pt.x - x) * (pt.x - x) + (pt.y - y) * (pt.y - y), q: q }; }).sort(function (u, v) { return u.d - v.d; }).slice(0, k), v0 = 0; ds.forEach(function (e) { if (pts[e.q].c === 0) { v0++; } }); return { nb: ds, a: v0, b: k - v0 }; }
      var GX = 30, GY = 19, grids = KS.map(function (k) { var g = []; for (var gy = 0; gy < GY; gy++) { for (var gx = 0; gx < GX; gx++) { g.push(vote((gx + 0.5) / GX, (gy + 0.5) / GY, k).a / k); } } return g; });
      var HOLDK = 4, t;
      p.reset = function () { t = 0; p.finished = false; };
      p.update = function (dt) { t += dt; if (t >= 4.4) { p.finished = true; } };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, sq = square(8, 8, W - 16, H - 16), kIdx = Math.floor(t / HOLDK) % KS.length, k = KS[kIdx], pk = (kIdx + KS.length - 1) % KS.length, fade = ease((t % HOLDK) / 0.6);
         if (t < HOLDK) { pk = kIdx; fade = 1; }
         function SX(u) { return sq.x + u * sq.s; } function SY(v) { return sq.y + v * sq.s; }
         var cw = sq.s / GX, ch = sq.s / GY;
         for (var gy = 0; gy < GY; gy++) { for (var gx = 0; gx < GX; gx++) { var pr = lerp(grids[pk][gy * GX + gx], grids[kIdx][gy * GX + gx], fade), cl = pr >= 0.5 ? 0 : 1, st = Math.abs(pr - 0.5) * 2; c.fillStyle = rgba(CLS[cl], 0.07 + 0.22 * st); c.fillRect(SX(0) + gx * cw, SY(0) + gy * ch, cw + 0.5, ch + 0.5); } }
         pts.forEach(function (pt) { c.fillStyle = CLS[pt.c]; c.beginPath(); c.arc(SX(pt.x), SY(pt.y), 3.4, 0, TAU); c.fill(); });
         var qx = reduce ? 0.5 : 0.5 + 0.34 * Math.sin(t * 0.8), qy = reduce ? 0.46 : 0.5 + 0.3 * Math.sin(t * 1.15 + 1), v = vote(qx, qy, k), pred = v.a >= v.b ? 0 : 1;
         v.nb.forEach(function (e) { c.strokeStyle = rgba(CLS[pts[e.q].c], 0.85); c.lineWidth = 1.4; c.beginPath(); c.moveTo(SX(qx), SY(qy)); c.lineTo(SX(pts[e.q].x), SY(pts[e.q].y)); c.stroke(); });
         c.setLineDash([4, 4]); c.strokeStyle = 'rgba(226,232,240,.8)'; c.lineWidth = 1.2; c.beginPath(); c.arc(SX(qx), SY(qy), Math.sqrt(v.nb[v.nb.length - 1].d) * sq.s, 0, TAU); c.stroke(); c.setLineDash([]);
         c.fillStyle = '#fff'; c.strokeStyle = CLS[pred]; c.lineWidth = 3; c.beginPath(); c.arc(SX(qx), SY(qy), 6, 0, TAU); c.fill(); c.stroke();
         p.say('k', String(k)); p.say('votes', 'A ' + v.a + ' : B ' + v.b); p.say('pred', 'class ' + CN[pred]);
      };
      return p;
   }

   /* ============================== decision tree ============================= */
   function tree(card) {
      var p = A.base(card), r = A.rng(3), pts = [], i;
      [[0.25, 0.3, 0, 22], [0.25, 0.74, 0, 22], [0.74, 0.3, 0, 22], [0.74, 0.74, 1, 30]].forEach(function (cl) { for (i = 0; i < cl[3]; i++) { pts.push({ x: clamp(cl[0] + A.gauss(r) * 0.095, 0.03, 0.97), y: clamp(cl[1] + A.gauss(r) * 0.095, 0.03, 0.97), c: cl[2] }); } });
      var n = pts.length, CLS = [COL.green, COL.violet];
      function gini(idx) { if (!idx.length) { return 0; } var a = 0; idx.forEach(function (q) { if (pts[q].c === 0) { a++; } }); var f = a / idx.length; return 2 * f * (1 - f); }
      function best(idx) {
         var bs = null;
         ['x', 'y'].forEach(function (f) {
            var sorted = idx.slice().sort(function (u, v) { return pts[u][f] - pts[v][f]; });
            for (var k = 1; k < sorted.length; k++) {
               if (pts[sorted[k]][f] === pts[sorted[k - 1]][f]) { continue; }
               var th = (pts[sorted[k]][f] + pts[sorted[k - 1]][f]) / 2, L = sorted.slice(0, k), R = sorted.slice(k), g = (L.length * gini(L) + R.length * gini(R)) / sorted.length;
               if (!bs || g < bs.g) { bs = { f: f, th: th, g: g, L: L, R: R }; }
            }
         });
         return bs;
      }
      var root = { rect: [0, 0, 1, 1], idx: pts.map(function (_, k) { return k; }), depth: 0, id: 0 }, queue = [root], splits = [], MAXD = 3;
      while (queue.length) {
         var nd = queue.shift();
         if (nd.depth >= MAXD || gini(nd.idx) === 0 || nd.idx.length < 5) { continue; }
         var bs = best(nd.idx); if (!bs) { continue; }
         nd.split = bs; var rc = nd.rect;
         nd.kids = [{ rect: bs.f === 'x' ? [rc[0], rc[1], bs.th, rc[3]] : [rc[0], rc[1], rc[2], bs.th], idx: bs.L, depth: nd.depth + 1, parent: nd }, { rect: bs.f === 'x' ? [bs.th, rc[1], rc[2], rc[3]] : [rc[0], bs.th, rc[2], rc[3]], idx: bs.R, depth: nd.depth + 1, parent: nd }];
         splits.push(nd); queue.push(nd.kids[0], nd.kids[1]);
      }
      var LEAD = 0.6, STEP = 1.0, T_END = LEAD + splits.length * STEP + 0.4, t, hold;
      p.reset = function () { t = 0; hold = 0; p.finished = false; };
      p.update = function (dt) { t += dt; if (t >= T_END) { p.finished = true; hold += dt; if (hold > 2.8 && !reduce) { p.reset(); } } };
      // tree layout: levels top to bottom, x positions by in-order of the final tree
      var leafCount = 0;
      (function lay(nd) { if (nd.kids) { lay(nd.kids[0]); lay(nd.kids[1]); nd.lx = (nd.kids[0].lx + nd.kids[1].lx) / 2; } else { nd.lx = leafCount++; } })(root);
      function maj(idx) { var a = 0; idx.forEach(function (q) { if (pts[q].c === 0) { a++; } }); return a >= idx.length - a ? 0 : 1; }
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, sq = square(8, 8, W * 0.52 - 10, H - 16), shown = reduce ? splits.length : clamp(Math.floor((t - LEAD) / STEP), 0, splits.length);
         function SX(u) { return sq.x + u * sq.s; } function SY(v) { return sq.y + v * sq.s; }
         var leaves = [], correct = 0, depth = 0;
         (function collect(nd, d) {                                                  // which nodes exist so far: a node is split when its split index < shown
            var si = splits.indexOf(nd);
            if (si >= 0 && si < shown) { nd.on = true; depth = Math.max(depth, d + 1); collect(nd.kids[0], d + 1); collect(nd.kids[1], d + 1); } else { nd.on = false; leaves.push(nd); }
         })(root, 0);
         leaves.forEach(function (lf) { var rc = lf.rect, m = maj(lf.idx); c.fillStyle = rgba(CLS[m], 0.16); c.fillRect(SX(rc[0]), SY(rc[1]), (rc[2] - rc[0]) * sq.s, (rc[3] - rc[1]) * sq.s); lf.idx.forEach(function (q) { if (pts[q].c === m) { correct++; } }); });
         c.strokeStyle = GRID; c.lineWidth = 1; c.strokeRect(sq.x + 0.5, sq.y + 0.5, sq.s - 1, sq.s - 1);
         pts.forEach(function (pt) { c.fillStyle = CLS[pt.c]; c.beginPath(); c.arc(SX(pt.x), SY(pt.y), 3.5, 0, TAU); c.fill(); });
         splits.forEach(function (nd, si) {                                          // the split lines, growing in
            if (si > shown || (si === shown && reduce)) { return; }
            var q = si < shown ? 1 : ease((t - LEAD - si * STEP) / 0.6); if (q <= 0) { return; }
            var rc = nd.rect, b = nd.split; c.strokeStyle = COL.amber; c.lineWidth = 2.2; c.shadowColor = rgba(COL.amber, 0.7); c.shadowBlur = 8; c.beginPath();
            if (b.f === 'x') { var y1 = lerp(rc[1], rc[3], q); c.moveTo(SX(b.th), SY((rc[1] + rc[3]) / 2)); c.lineTo(SX(b.th), SY(lerp((rc[1] + rc[3]) / 2, rc[1], q))); c.moveTo(SX(b.th), SY((rc[1] + rc[3]) / 2)); c.lineTo(SX(b.th), SY(lerp((rc[1] + rc[3]) / 2, rc[3], q))); }
            else { c.moveTo(SX((rc[0] + rc[2]) / 2), SY(b.th)); c.lineTo(SX(lerp((rc[0] + rc[2]) / 2, rc[0], q)), SY(b.th)); c.moveTo(SX((rc[0] + rc[2]) / 2), SY(b.th)); c.lineTo(SX(lerp((rc[0] + rc[2]) / 2, rc[2], q)), SY(b.th)); }
            c.stroke(); c.shadowBlur = 0;
         });
         // the tree itself, on the right
         var tx0 = W * 0.56, tx1 = W - 14, ty0 = 30, ty1 = H - 24, LV = MAXD;
         function TX(nd) { return tx0 + (nd.lx + 0.5) / leafCount * (tx1 - tx0); } function TY(d) { return ty0 + d / LV * (ty1 - ty0); }
         c.fillStyle = TEXT; c.textAlign = 'left'; c.fillText('the tree', tx0, 14);
         (function drawNode(nd, d) {
            var visible = nd === root || (nd.parent && nd.parent.on); if (!visible) { return; }
            var x = TX(nd), y = TY(d);
            if (nd.kids && nd.on) { [0, 1].forEach(function (k) { var ch = nd.kids[k]; c.strokeStyle = 'rgba(148,163,184,.5)'; c.lineWidth = 1.4; c.beginPath(); c.moveTo(x, y); c.lineTo(TX(ch), TY(d + 1)); c.stroke(); }); }
            var m = maj(nd.idx), pur = 1 - gini(nd.idx) * 2;
            c.fillStyle = rgba(CLS[m], 0.35 + 0.6 * pur); c.strokeStyle = nd.on ? COL.amber : CLS[m]; c.lineWidth = 1.8; c.beginPath(); c.arc(x, y, nd.on ? 9 : 7, 0, TAU); c.fill(); c.stroke();
            if (nd.on) { c.fillStyle = '#e8eefb'; c.textAlign = 'center'; c.font = '9px ' + FONT; c.fillText(nd.split.f + '<' + nd.split.th.toFixed(2), x, y - 13); c.font = MONO; }
            if (nd.kids && nd.on) { drawNode(nd.kids[0], d + 1); drawNode(nd.kids[1], d + 1); }
         })(root, 0);
         p.say('splits', shown + '/' + splits.length); p.say('depth', String(depth)); p.say('acc', Math.round(correct / n * 100) + '%');
      };
      return p;
   }

   /* ============================ logistic regression ========================= */
   function logistic(card) {
      var p = A.base(card), r = A.rng(23), pts = [], i;
      for (i = 0; i < 32; i++) { pts.push({ x: clamp(0.34 + A.gauss(r) * 0.12, 0.03, 0.97), y: clamp(0.62 + A.gauss(r) * 0.12, 0.03, 0.97), c: 0 }); }
      for (i = 0; i < 32; i++) { pts.push({ x: clamp(0.66 + A.gauss(r) * 0.12, 0.03, 0.97), y: clamp(0.38 + A.gauss(r) * 0.12, 0.03, 0.97), c: 1 }); }
      var n = pts.length, CLS = [COL.green, COL.violet], w0, w1, b0, steps, acc, hold, t;
      function sig(z) { return 1 / (1 + Math.exp(-z)); }
      function lossAcc() { var L = 0, ok = 0; pts.forEach(function (pt) { var pr = sig(w0 * (pt.x - 0.5) * 6 + w1 * (pt.y - 0.5) * 6 + b0); L += -(pt.c ? Math.log(pr + 1e-9) : Math.log(1 - pr + 1e-9)); if ((pr >= 0.5) === (pt.c === 1)) { ok++; } }); return { L: L / n, acc: ok / n }; }
      p.reset = function () { w0 = 1.2; w1 = 1.6; b0 = 0.8; steps = 0; acc = 0; hold = 0; t = 0; p.finished = false; };
      p.update = function (dt) {
         t += dt;
         if (t > 0.6 && steps < 120) { acc += dt * 30; var g = 0; while (acc >= 1 && g++ < 5) { acc -= 1; var d0 = 0, d1 = 0, db = 0; pts.forEach(function (pt) { var u = (pt.x - 0.5) * 6, v = (pt.y - 0.5) * 6, e = sig(w0 * u + w1 * v + b0) - pt.c; d0 += e * u; d1 += e * v; db += e; }); w0 -= 0.35 * d0 / n; w1 -= 0.35 * d1 / n; b0 -= 0.35 * db / n; steps++; } }
         if (steps >= 120) { p.finished = true; hold += dt; if (hold > 2.8 && !reduce) { p.reset(); } }
      };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, sq = square(8, 8, W - 16, H - 16), GX = 34, GY = 21, cw = sq.s / GX, ch = sq.s / GY;
         function SX(u) { return sq.x + u * sq.s; } function SY(v) { return sq.y + v * sq.s; }
         for (var gy = 0; gy < GY; gy++) { for (var gx = 0; gx < GX; gx++) { var pr = sig(w0 * ((gx + 0.5) / GX - 0.5) * 6 + w1 * ((gy + 0.5) / GY - 0.5) * 6 + b0), cl = pr >= 0.5 ? 1 : 0, st = Math.abs(pr - 0.5) * 2; c.fillStyle = rgba(CLS[cl], 0.05 + 0.3 * st); c.fillRect(SX(0) + gx * cw, SY(0) + gy * ch, cw + 0.5, ch + 0.5); } }
         // the boundary: w0 u + w1 v + b0 = 0
         c.save(); c.beginPath(); c.rect(sq.x, sq.y, sq.s, sq.s); c.clip();
         c.strokeStyle = '#fff'; c.lineWidth = 2.4; c.shadowColor = '#ffffff'; c.shadowBlur = 8; c.beginPath();
         if (Math.abs(w1) > 1e-6) { [0, 1].forEach(function (xx, k) { var v = (-b0 - w0 * (xx - 0.5) * 6) / (w1 * 6) + 0.5; (k === 0 ? c.moveTo : c.lineTo).call(c, SX(xx), SY(v)); }); }
         c.stroke(); c.shadowBlur = 0; c.restore();
         pts.forEach(function (pt) { c.fillStyle = CLS[pt.c]; c.strokeStyle = 'rgba(255,255,255,.6)'; c.lineWidth = 1; c.beginPath(); c.arc(SX(pt.x), SY(pt.y), 3.8, 0, TAU); c.fill(); c.stroke(); });
         var la = lossAcc(); p.say('step', String(steps)); p.say('loss', la.L.toFixed(3)); p.say('acc', Math.round(la.acc * 100) + '%');
      };
      return p;
   }

   /* ======================= gradient descent on a loss surface =============== */
   function gd(card) {
      var p = A.base(card), PHI = 0.5, CS = Math.cos(PHI), SN = Math.sin(PHI), XR = 3.4, YR = 2.2, A1 = 1, A2 = 30;
      function f(x, y) { var u = x * CS + y * SN, v = -x * SN + y * CS; return 0.5 * (A1 * u * u + A2 * v * v); }
      function grad(x, y) { var u = x * CS + y * SN, v = -x * SN + y * CS, gu = A1 * u, gv = A2 * v; return [gu * CS - gv * SN, gu * SN + gv * CS]; }
      var S0 = [-2.8 * CS - 1.1 * SN, -2.8 * SN + 1.1 * CS], STEPS = 90, paths = {};
      (function sim() {
         var x = S0[0], y = S0[1], a = [[x, y]], g, s;
         for (s = 0; s < STEPS; s++) { g = grad(x, y); x -= 0.06 * g[0]; y -= 0.06 * g[1]; a.push([x, y]); } paths.gd = a;                        // plain gradient descent
         x = S0[0]; y = S0[1]; var vx = 0, vy = 0; a = [[x, y]];
         for (s = 0; s < STEPS; s++) { g = grad(x, y); vx = 0.9 * vx - 0.02 * g[0]; vy = 0.9 * vy - 0.02 * g[1]; x += vx; y += vy; a.push([x, y]); } paths.mom = a;   // momentum 0.9
         x = S0[0]; y = S0[1]; var mx = 0, my = 0, sx = 0, sy = 0; a = [[x, y]];
         for (s = 1; s <= STEPS; s++) {                                                                                                        // Adam
            g = grad(x, y); mx = 0.9 * mx + 0.1 * g[0]; my = 0.9 * my + 0.1 * g[1]; sx = 0.999 * sx + 0.001 * g[0] * g[0]; sy = 0.999 * sy + 0.001 * g[1] * g[1];
            var m1x = mx / (1 - Math.pow(0.9, s)), m1y = my / (1 - Math.pow(0.9, s)), s1x = sx / (1 - Math.pow(0.999, s)), s1y = sy / (1 - Math.pow(0.999, s));
            x -= 0.25 * m1x / (Math.sqrt(s1x) + 1e-8); y -= 0.25 * m1y / (Math.sqrt(s1y) + 1e-8); a.push([x, y]);
         }
         paths.adam = a;
      })();
      var bg = null, bgKey = '', t, hold, LEAD = 0.6, RATE = 22, T_END = LEAD + STEPS / RATE + 0.3;
      p.reset = function () { t = 0; hold = 0; p.finished = false; };
      p.update = function (dt) { t += dt; if (t >= T_END) { p.finished = true; hold += dt; if (hold > 3 && !reduce) { p.reset(); } } };
      function ensureBg(W, H) {                                                                       // filled contour bands, drawn once per size
         var key = W + 'x' + H + 'x' + p.dpr; if (bgKey === key && bg) { return; }
         var cw = 240, chh = Math.round(240 * H / W), cv = document.createElement('canvas'); cv.width = cw; cv.height = chh; var g = cv.getContext('2d'), im = g.createImageData(cw, chh);
         for (var yy = 0; yy < chh; yy++) { for (var xx = 0; xx < cw; xx++) {
            var x = ((xx + 0.5) / cw * 2 - 1) * XR, y = ((yy + 0.5) / chh * 2 - 1) * YR, lv = Math.log(1 + f(x, y)) * 2.2, band = Math.floor(lv), edge = lv - band < 0.07 ? 1 : 0, shade = Math.max(0, 1 - lv / 9), idx = (yy * cw + xx) * 4;
            var base = [9 + 14 * shade, 22 + 34 * shade, 44 + 52 * shade]; im.data[idx] = base[0] + edge * 28; im.data[idx + 1] = base[1] + edge * 52; im.data[idx + 2] = base[2] + edge * 70; im.data[idx + 3] = 255;
         } }
         g.putImageData(im, 0, 0); bg = cv; bgKey = key;
      }
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H; ensureBg(W, H);
         c.imageSmoothingEnabled = true; c.drawImage(bg, 0, 0, W, H);
         function SX(x) { return (x / XR + 1) / 2 * W; } function SY(y) { return (y / YR + 1) / 2 * H; }
         var step = reduce ? STEPS : clamp((t - LEAD) * RATE, 0, STEPS), whole = Math.floor(step), frac = step - whole;
         var opts = [['gd', COL.cyan, 'gradient descent'], ['mom', COL.amber, 'momentum'], ['adam', COL.green, 'Adam']], vals = {};
         c.fillStyle = '#fff'; c.beginPath(); c.arc(SX(0), SY(0), 4, 0, TAU); c.fill(); c.strokeStyle = '#fff'; c.lineWidth = 1.4; c.beginPath(); c.moveTo(SX(0) - 8, SY(0)); c.lineTo(SX(0) + 8, SY(0)); c.moveTo(SX(0), SY(0) - 8); c.lineTo(SX(0), SY(0) + 8); c.stroke();
         opts.forEach(function (o, oi) {
            var path = paths[o[0]], col = o[1]; c.strokeStyle = col; c.lineWidth = 1.8; c.shadowColor = rgba(col, 0.6); c.shadowBlur = 6; c.beginPath(); c.moveTo(SX(path[0][0]), SY(path[0][1]));
            for (var k = 1; k <= whole; k++) { c.lineTo(SX(path[k][0]), SY(path[k][1])); }
            var hx = path[whole][0], hy = path[whole][1];
            if (whole < STEPS) { hx = lerp(path[whole][0], path[whole + 1][0], frac); hy = lerp(path[whole][1], path[whole + 1][1], frac); c.lineTo(SX(hx), SY(hy)); }
            c.stroke(); c.shadowBlur = 0; c.fillStyle = col; c.strokeStyle = '#fff'; c.lineWidth = 1.5; c.beginPath(); c.arc(SX(hx), SY(hy), 5, 0, TAU); c.fill(); c.stroke();
            vals[o[0]] = f(hx, hy);
            c.fillStyle = col; c.textAlign = 'left'; c.fillText('● ' + o[2], 10, 16 + oi * 14);
         });
         c.fillStyle = '#fff'; c.textAlign = 'left'; c.fillText('minimum', SX(0) + 10, SY(0) + 4); c.fillStyle = TEXT; c.textAlign = 'right'; c.fillText('start', SX(S0[0]) + 34, SY(S0[1]) - 8);
         function fv(v) { return v < 0.001 ? '0.000' : v < 10 ? v.toFixed(3) : v.toFixed(1); }
         p.say('gd', fv(vals.gd)); p.say('mom', fv(vals.mom)); p.say('adam', fv(vals.adam));
      };
      return p;
   }

   return { dendrogram: dendrogram, dbscan: dbscan, gmm: gmm, pca: pca, overfit: overfit, knn: knn, tree: tree, logistic: logistic, gd: gd };
};
