/* Machine learning in motion: four small models running live on the ML hub landing page.
     regression  gradient descent fitting a line to the real NDVI / evapotranspiration data of Assignment 1
     kmeans      Lloyd's algorithm on three clouds of points (an illustration)
     collinearity the correlation matrix and VIF pruning of Assignment 1: drop EVI, then LAI (real data)
     loss        training and validation loss with early stopping (an illustration)
   Every number on screen is computed by the code that draws it. A panel animates only while it is on screen and
   the tab is visible; with prefers-reduced-motion each panel is drawn once, in its finished state. */
(function () {
   'use strict';
   var DATA = window.ML_NDVI_ET;
   var cards = document.querySelectorAll('.viz-card');
   if (!cards.length || !DATA) { return; }

   var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
   var TAU = Math.PI * 2;
   var COL = { green: '#34d399', cyan: '#22d3ee', amber: '#fbbf24', violet: '#a78bfa', grey: '#64748b' };
   var TEXT = 'rgba(148,163,184,.9)', GRID = 'rgba(148,163,184,.13)', AXIS = 'rgba(148,163,184,.42)';
   var MONO = '11px ui-monospace, "Cascadia Code", Consolas, monospace';

   /* ---------- small helpers ---------- */
   function rng(seed) {                                   // mulberry32: the same seed always gives the same numbers
      var s = seed >>> 0;
      return function () {
         s = (s + 0x6D2B79F5) >>> 0;
         var t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
         return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
   }
   function gauss(r) { var u = 0; while (u === 0) { u = r(); } return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * r()); }
   function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
   function ease(t) { return 1 - Math.pow(1 - clamp(t, 0, 1), 3); }
   function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) { s += a[i]; } return s / a.length; }
   function lerp(a, b, t) { return a + (b - a) * t; }
   function rgba(hex, a) { var n = parseInt(hex.slice(1), 16); return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')'; }

   function base(card) {
      var p = { card: card, canvas: card.querySelector('canvas'), W: 1, H: 1, dpr: 1, visible: false, finished: false, started: false, stats: {} };
      p.ctx = p.canvas.getContext('2d');
      [].forEach.call(card.querySelectorAll('[data-s]'), function (el) { p.stats[el.getAttribute('data-s')] = el; });
      p.say = function (k, v) { var el = p.stats[k]; if (el && el.textContent !== v) { el.textContent = v; } };
      p.resize = function () {
         var r = p.canvas.getBoundingClientRect();
         p.dpr = Math.min(window.devicePixelRatio || 1, 2);
         p.W = Math.max(1, Math.round(r.width)); p.H = Math.max(1, Math.round(r.height));
         p.canvas.width = Math.round(p.W * p.dpr); p.canvas.height = Math.round(p.H * p.dpr);
      };
      p.begin = function () { p.ctx.setTransform(p.dpr, 0, 0, p.dpr, 0, 0); p.ctx.clearRect(0, 0, p.W, p.H); p.ctx.font = MONO; };
      return p;
   }

   /* =============================== regression =============================== */
   function regression(card) {
      var p = base(card), n = DATA.length;
      var xs = DATA.map(function (d) { return d[0]; }), ys = DATA.map(function (d) { return d[1]; });
      var mx = mean(xs), my = mean(ys), sx = 0, sy = 0, sst = 0, i;
      for (i = 0; i < n; i++) { sx += (xs[i] - mx) * (xs[i] - mx); sy += (ys[i] - my) * (ys[i] - my); }
      sst = sy; sx = Math.sqrt(sx / n); sy = Math.sqrt(sy / n);
      var zx = xs.map(function (v) { return (v - mx) / sx; }), zy = ys.map(function (v) { return (v - my) / sy; });
      var XMIN = 0.15, XMAX = 0.8, YMIN = 2, YMAX = 20;
      var t, w, b, steps, acc, hold;

      p.reset = function () { t = 0; w = -0.35; b = 0.95; steps = 0; acc = 0; hold = 0; p.finished = false; };
      function pred(x) { return my + sy * (w * (x - mx) / sx + b); }
      function step() {
         var dw = 0, db = 0, lr = 0.05;
         for (var k = 0; k < n; k++) { var e = w * zx[k] + b - zy[k]; dw += e * zx[k]; db += e; }
         w -= lr * 2 * dw / n; b -= lr * 2 * db / n; steps++;
      }
      p.update = function (dt) {
         t += dt;
         if (t > 1.2 && steps < 130) { acc += dt * 26; var guard = 0; while (acc >= 1 && guard++ < 6) { acc -= 1; step(); } }
         if (steps >= 130) { p.finished = true; hold += dt; if (hold > 3.2 && !reduce) { p.reset(); } }
      };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, ml = 38, mr = 14, mt = 10, mb = 38;
         function px(x) { return ml + (x - XMIN) / (XMAX - XMIN) * (W - ml - mr); }
         function py(y) { return H - mb - (y - YMIN) / (YMAX - YMIN) * (H - mt - mb); }
         c.lineWidth = 1; c.fillStyle = TEXT; c.textAlign = 'center';
         [0.2, 0.4, 0.6, 0.8].forEach(function (g) { c.strokeStyle = GRID; c.beginPath(); c.moveTo(px(g), mt); c.lineTo(px(g), H - mb); c.stroke(); c.fillText(g.toFixed(1), px(g), H - mb + 14); });
         c.textAlign = 'right';
         [5, 10, 15, 20].forEach(function (g) { c.strokeStyle = GRID; c.beginPath(); c.moveTo(ml, py(g)); c.lineTo(W - mr, py(g)); c.stroke(); c.fillText(String(g), ml - 6, py(g) + 4); });
         c.strokeStyle = AXIS; c.beginPath(); c.moveTo(ml, mt); c.lineTo(ml, H - mb); c.lineTo(W - mr, H - mb); c.stroke();
         c.textAlign = 'center'; c.fillText('NDVI', ml + (W - ml - mr) / 2, H - 5);
         c.textAlign = 'left'; c.fillText('evapotranspiration, mm', ml + 8, mt + 13);

         var la = reduce ? 1 : ease((t - 1.0) / 0.6), sse = 0;
         for (i = 0; i < n; i++) {
            var a = reduce ? 1 : ease((t - i / n * 1.3) / 0.3), yh = pred(xs[i]);
            sse += (ys[i] - yh) * (ys[i] - yh);
            if (la > 0) { c.strokeStyle = 'rgba(251,191,36,' + (0.3 * la * a).toFixed(3) + ')'; c.lineWidth = 1; c.beginPath(); c.moveTo(px(xs[i]), py(ys[i])); c.lineTo(px(xs[i]), py(yh)); c.stroke(); }
            c.fillStyle = 'rgba(34,211,238,' + (0.9 * a).toFixed(3) + ')'; c.beginPath(); c.arc(px(xs[i]), py(ys[i]), 3.3, 0, TAU); c.fill();
         }
         if (la > 0) {
            var g = c.createLinearGradient(px(XMIN), 0, px(XMAX), 0);
            g.addColorStop(0, rgba(COL.green, la)); g.addColorStop(1, rgba(COL.cyan, la));
            c.strokeStyle = g; c.lineWidth = 2.8; c.shadowColor = rgba(COL.green, 0.6 * la); c.shadowBlur = 12;
            c.beginPath(); c.moveTo(px(XMIN), py(pred(XMIN))); c.lineTo(px(XMAX), py(pred(XMAX))); c.stroke(); c.shadowBlur = 0;
         }
         var r2 = 1 - sse / sst;
         p.say('r2', (steps === 0 && !reduce ? '--' : r2.toFixed(3)));
         p.say('slope', (sy * w / sx).toFixed(1));
         p.say('step', String(steps));
      };
      return p;
   }

   /* ================================ k-means ================================= */
   function kmeans(card) {
      var p = base(card), r = rng(8), blobs = [[0.24, 0.32], [0.74, 0.3], [0.5, 0.76]], pts = [], i, j, k;
      for (i = 0; i < 3; i++) { for (j = 0; j < 42; j++) { pts.push({ x: clamp(blobs[i][0] + gauss(r) * 0.085, 0.03, 0.97), y: clamp(blobs[i][1] + gauss(r) * 0.085, 0.04, 0.96), a: j / 42 * 0.9 + i * 0.05 }); } }
      var N = pts.length;

      function d2(a, b) { return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y); }
      function lloyd(seed) {                                     // run the whole algorithm once and record every step
         var rr = rng(seed * 7919 + 13), cents = [], used = {}, c;
         while (cents.length < 3) { var q = Math.floor(rr() * N); if (!used[q]) { used[q] = 1; cents.push({ x: pts[q].x, y: pts[q].y }); } }
         var steps = [], prev = null, assign;
         for (var it = 0; it < 12; it++) {
            assign = pts.map(function (pt) { var best = 0, bd = 9; for (c = 0; c < 3; c++) { var d = d2(pt, cents[c]); if (d < bd) { bd = d; best = c; } } return best; });
            var inertia = 0; for (i = 0; i < N; i++) { inertia += d2(pts[i], cents[assign[i]]); }
            var same = prev && prev.every(function (v, idx) { return v === assign[idx]; });
            steps.push({ assign: assign, cents: cents.map(function (q2) { return { x: q2.x, y: q2.y }; }), inertia: inertia * 100, done: !!same });
            if (same) { break; }
            var moved = [0, 1, 2].map(function (cc) { var sx = 0, sy = 0, cnt = 0; for (i = 0; i < N; i++) { if (assign[i] === cc) { sx += pts[i].x; sy += pts[i].y; cnt++; } } return cnt ? { x: sx / cnt, y: sy / cnt } : cents[cc]; });
            steps[steps.length - 1].next = moved; cents = moved; prev = assign;
         }
         return steps;
      }
      var runs = [], seed = 1;
      while (runs.length < 4 && seed < 60) { var s = lloyd(seed++); if (s.length >= 4) { runs.push(s); } }   // pick starts that take a few rounds, so there is something to watch
      if (!runs.length) { runs.push(lloyd(1)); }
      var run, ri = 0, t, hold;
      var ASSIGN = 0.8, UPDATE = 0.95, LEAD = 0.9;

      p.reset = function () { run = runs[ri % runs.length]; ri++; t = 0; hold = 0; p.finished = false; };
      p.update = function (dt) {
         t += dt;
         var total = LEAD + run.length * (ASSIGN + UPDATE);
         if (t >= total - UPDATE) { p.finished = true; hold += dt; if (hold > 2.6 && !reduce) { p.reset(); } }
      };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, m = 14;
         function px(x) { return m + x * (W - 2 * m); }
         function py(y) { return m + y * (H - 2 * m); }
         c.strokeStyle = GRID; c.lineWidth = 1;
         for (var g = 1; g < 4; g++) { c.beginPath(); c.moveTo(px(g / 4), m); c.lineTo(px(g / 4), H - m); c.moveTo(m, py(g / 4)); c.lineTo(W - m, py(g / 4)); c.stroke(); }
         var cols = [COL.green, COL.cyan, COL.amber];
         // where are we in the run?
         var tt = Math.max(0, t - LEAD), it = Math.min(run.length - 1, Math.floor(tt / (ASSIGN + UPDATE))), local = tt - it * (ASSIGN + UPDATE);
         if (t >= LEAD + run.length * (ASSIGN + UPDATE) - UPDATE) { it = run.length - 1; local = ASSIGN + UPDATE; }
         var st = run[it], prevSt = run[Math.max(0, it - 1)];
         var assignQ = t < LEAD ? 0 : ease(local / ASSIGN);                                   // colours fade from the previous round to this one
         var moveQ = t < LEAD ? 0 : ease((local - ASSIGN) / UPDATE);                            // centroids glide to the new means
         var from = st.cents, to = st.next || st.cents;
         var cur = from.map(function (f, idx) { return { x: lerp(f.x, to[idx].x, moveQ), y: lerp(f.y, to[idx].y, moveQ) }; });
         if (reduce || (st.done && local >= ASSIGN)) { cur = st.cents; }
         for (i = 0; i < N; i++) {
            var pt = pts[i], a = reduce ? 1 : ease((t - pt.a) / 0.35);
            var colNow = cols[st.assign[i]];
            if (t < LEAD) { colNow = COL.grey; }                                              // before the first assignment nothing has a cluster
            else if (assignQ < 1) { var pc = it === 0 ? COL.grey : cols[prevSt.assign[i]]; colNow = assignQ < 0.5 ? pc : cols[st.assign[i]]; }
            if (t >= LEAD && assignQ > 0 && assignQ < 1) {                                     // a faint line to the centroid being chosen
               c.strokeStyle = rgba(cols[st.assign[i]], 0.12 * Math.sin(assignQ * Math.PI)); c.lineWidth = 1;
               c.beginPath(); c.moveTo(px(pt.x), py(pt.y)); c.lineTo(px(from[st.assign[i]].x), py(from[st.assign[i]].y)); c.stroke();
            }
            c.fillStyle = rgba(colNow, 0.88 * a); c.beginPath(); c.arc(px(pt.x), py(pt.y), 3.2, 0, TAU); c.fill();
         }
         for (var q = 0; q < 3; q++) {
            var cx = px(cur[q].x), cy = py(cur[q].y), pulse = reduce ? 0 : 0.5 + 0.5 * Math.sin(t * 5 + q);
            var pop = reduce ? 1 : ease(t / 0.7);
            c.strokeStyle = rgba(cols[q], 0.35 * pop); c.lineWidth = 2; c.beginPath(); c.arc(cx, cy, 12 + pulse * 3, 0, TAU); c.stroke();
            c.strokeStyle = rgba('#ffffff', 0.95 * pop); c.fillStyle = rgba(cols[q], pop); c.lineWidth = 2;
            c.beginPath(); c.arc(cx, cy, 7, 0, TAU); c.fill(); c.stroke();
            c.beginPath(); c.moveTo(cx - 4, cy); c.lineTo(cx + 4, cy); c.moveTo(cx, cy - 4); c.lineTo(cx, cy + 4); c.strokeStyle = '#04101e'; c.lineWidth = 1.6; c.stroke();
         }
         p.say('iter', t < LEAD ? '0' : String(it + 1));
         p.say('inertia', t < LEAD ? '--' : st.inertia.toFixed(1));
         p.say('state', (st.done && local >= ASSIGN) || reduce && p.finished ? 'converged' : (t < LEAD ? 'random start' : (local < ASSIGN ? 'assign' : 'update')));
      };
      return p;
   }

   /* ============================ multicollinearity =========================== */
   function collinearity(card) {
      var D = window.ML_COLLINEAR;
      if (!D) { return null; }
      var p = base(card), names = D.names, K = names.length, n = D.rows.length, i, j;
      var cols = names.map(function (nm, c) { return D.rows.map(function (row) { return row[c]; }); });

      function corr(a, b) {
         var ma = mean(a), mb = mean(b), sab = 0, saa = 0, sbb = 0, k;
         for (k = 0; k < n; k++) { sab += (a[k] - ma) * (b[k] - mb); saa += (a[k] - ma) * (a[k] - ma); sbb += (b[k] - mb) * (b[k] - mb); }
         return sab / Math.sqrt(saa * sbb);
      }
      var R = [];
      for (i = 0; i < K; i++) { R[i] = []; for (j = 0; j < K; j++) { R[i][j] = i === j ? 1 : corr(cols[i], cols[j]); } }

      function inverse(m) {                                     // Gauss-Jordan, with pivoting
         var s = m.length, a = m.map(function (row, r) { return row.concat(row.map(function (v, c) { return r === c ? 1 : 0; })); }), r, c, k, f, d, piv, tmp;
         for (k = 0; k < s; k++) {
            piv = k; for (r = k + 1; r < s; r++) { if (Math.abs(a[r][k]) > Math.abs(a[piv][k])) { piv = r; } }
            tmp = a[k]; a[k] = a[piv]; a[piv] = tmp;
            d = a[k][k]; for (c = 0; c < 2 * s; c++) { a[k][c] /= d; }
            for (r = 0; r < s; r++) { if (r !== k) { f = a[r][k]; for (c = 0; c < 2 * s; c++) { a[r][c] -= f * a[k][c]; } } }
         }
         return a.map(function (row) { return row.slice(s); });
      }
      function vifs(active) {                                    // VIF of predictor i = the i-th diagonal entry of the inverse correlation matrix
         var idx = [], out = [];
         for (var q = 0; q < K; q++) { out[q] = 0; if (active[q]) { idx.push(q); } }
         var inv = inverse(idx.map(function (a) { return idx.map(function (b) { return R[a][b]; }); }));
         idx.forEach(function (a, k) { out[a] = inv[k][k]; });
         return out;
      }
      var EVI = names.indexOf('EVI'), LAI = names.indexOf('LAI');
      var all = names.map(function () { return true; }), noEvi = all.slice(), noBoth = all.slice();
      noEvi[EVI] = false; noBoth[EVI] = false; noBoth[LAI] = false;
      var V0 = vifs(all), V1 = vifs(noEvi), V2 = vifs(noBoth);
      var T_BARS = 2.5, T_S1 = 4.7, T_S2 = 6.9, T_END = 8.5, DUR = 1.1, t, hold;

      p.reset = function () { t = 0; hold = 0; p.finished = false; };
      p.update = function (dt) {
         t += dt;
         if (t >= T_END) { p.finished = true; hold += dt; if (hold > 3 && !reduce) { p.reset(); } }
      };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, topM = 46, botM = 26, rowH = Math.min(31, (H - topM - botM) / K), x0 = 44;
         var heatW = K * rowH, xb = x0 + heatW + 26, barW = Math.max(40, W - xb - 40), VMAX = 32;
         function bx(v) { return xb + Math.min(v, VMAX) / VMAX * barW; }
         var k1 = reduce ? 1 : ease((t - T_S1) / DUR), k2 = reduce ? 1 : ease((t - T_S2) / DUR);
         var grow = reduce ? 1 : ease((t - T_BARS) / 1.2);
         var rowA = names.map(function (nm, q) { return q === EVI ? 1 - 0.86 * k1 : (q === LAI ? 1 - 0.86 * k2 : 1); });
         var v = names.map(function (nm, q) { return lerp(lerp(V0[q], V1[q], k1), V2[q], k2); });

         // column and row names
         c.fillStyle = TEXT; c.font = '10px ui-monospace, "Cascadia Code", Consolas, monospace';
         for (i = 0; i < K; i++) {
            c.globalAlpha = rowA[i]; c.textAlign = 'right'; c.fillText(names[i], x0 - 6, topM + (i + 0.5) * rowH + 3.5);
            c.save(); c.translate(x0 + (i + 0.5) * rowH + 3.5, topM - 6); c.rotate(-Math.PI / 2); c.textAlign = 'left'; c.fillText(names[i], 0, 0); c.restore();
            if (rowA[i] < 0.9) { c.strokeStyle = rgba(COL.amber, 0.8 * (1 - rowA[i])); c.lineWidth = 1.2; var wdt = c.measureText(names[i]).width; c.beginPath(); c.moveTo(x0 - 6 - wdt, topM + (i + 0.5) * rowH); c.lineTo(x0 - 6, topM + (i + 0.5) * rowH); c.stroke(); }
         }
         c.globalAlpha = 1;

         // correlation matrix: cyan for positive, violet for negative, strength as opacity
         for (i = 0; i < K; i++) {
            for (j = 0; j < K; j++) {
               var delay = (i + j) / (2 * K) * 1.3, a = (reduce ? 1 : ease((t - delay) / 0.45)) * Math.min(rowA[i], rowA[j]);
               if (a <= 0) { continue; }
               var r = R[i][j], mag = Math.abs(r), cx = x0 + j * rowH, cy = topM + i * rowH;
               c.fillStyle = rgba('#0c1a33', a); c.fillRect(cx + 1, cy + 1, rowH - 2, rowH - 2);
               c.fillStyle = rgba(r >= 0 ? COL.cyan : COL.violet, a * (0.1 + 0.9 * Math.pow(mag, 1.4))); c.fillRect(cx + 1, cy + 1, rowH - 2, rowH - 2);
               if (i !== j && mag >= 0.8) { c.fillStyle = rgba('#ffffff', 0.92 * a); c.font = '8.5px ui-monospace, Consolas, monospace'; c.textAlign = 'center'; c.fillText(r.toFixed(2), cx + rowH / 2, cy + rowH / 2 + 3); }
            }
         }
         // the block that moves together
         var blockA = reduce ? 0 : ease((t - 1.8) / 0.5) * (1 - k1) * (0.65 + 0.35 * Math.sin(t * 6));
         if (blockA > 0) {
            c.strokeStyle = rgba(COL.amber, blockA); c.lineWidth = 2; c.shadowColor = rgba(COL.amber, 0.7 * blockA); c.shadowBlur = 10;
            c.strokeRect(x0 + 0.5, topM + 0.5, 3 * rowH - 1, 3 * rowH - 1); c.shadowBlur = 0;
         }

         // VIF bars, one per row, with the usual limits of 5 and 10
         c.font = MONO; c.textAlign = 'center'; c.fillStyle = TEXT; c.fillText('VIF', xb + barW / 2, topM - 22);
         [5, 10].forEach(function (lim) {
            c.setLineDash([4, 4]); c.strokeStyle = lim === 5 ? rgba(COL.green, 0.55) : rgba('#f87171', 0.6); c.lineWidth = 1;
            c.beginPath(); c.moveTo(bx(lim), topM - 6); c.lineTo(bx(lim), topM + heatW + 4); c.stroke(); c.setLineDash([]);
            c.fillStyle = lim === 5 ? COL.green : '#f87171'; c.textAlign = 'center'; c.fillText(String(lim), bx(lim), topM + heatW + 16);
         });
         c.strokeStyle = AXIS; c.beginPath(); c.moveTo(xb, topM - 4); c.lineTo(xb, topM + heatW + 4); c.stroke();
         var worst = 0;
         for (i = 0; i < K; i++) {
            var val = v[i] * grow, yc = topM + (i + 0.5) * rowH, col = v[i] < 5 ? COL.green : (v[i] < 10 ? COL.amber : '#f87171');
            if (v[i] * 1 > worst && rowA[i] > 0.5) { worst = v[i]; }
            if (val < 0.05) { continue; }
            c.globalAlpha = Math.max(0.25, rowA[i]);
            c.fillStyle = rgba(col, 0.9); c.shadowColor = rgba(col, 0.5); c.shadowBlur = 8;
            c.fillRect(xb + 1, yc - rowH * 0.28, Math.max(1, bx(val) - xb - 1), rowH * 0.56); c.shadowBlur = 0;
            c.fillStyle = '#e8eefb'; c.textAlign = 'left'; c.fillText(val.toFixed(1), bx(val) + 5, yc + 3.5);
            c.globalAlpha = 1;
         }
         var removed = k2 > 0.5 ? 'EVI, LAI' : (k1 > 0.5 ? 'EVI' : 'none');
         p.say('removed', removed);
         p.say('maxvif', t < T_BARS && !reduce ? '--' : worst.toFixed(1));
         p.say('n', String(K - (k1 > 0.5 ? 1 : 0) - (k2 > 0.5 ? 1 : 0)));
      };
      return p;
   }

   /* ============================== training loss ============================= */
   function loss(card) {
      var p = base(card), E = 60, r = rng(33), train = [], val = [], e, best = 0;
      for (e = 0; e <= E; e++) {
         train.push(0.055 + 0.95 * Math.exp(-e / 11) + gauss(r) * 0.006);
         val.push(0.13 + 0.88 * Math.exp(-e / 11) + (e > 33 ? 0.0046 * Math.pow(e - 33, 1.12) : 0) + gauss(r) * 0.011);
      }
      for (e = 1; e <= E; e++) { if (val[e] < val[best]) { best = e; } }
      var DUR = 5.2, t, hold;
      p.reset = function () { t = 0; hold = 0; p.finished = false; };
      p.update = function (dt) {
         t += dt;
         if (t >= DUR) { p.finished = true; hold += dt; if (hold > 3 && !reduce) { p.reset(); } }
      };
      p.draw = function () {
         p.begin();
         var c = p.ctx, W = p.W, H = p.H, ml = 40, mr = 14, mt = 14, mb = 38, YMAX = 1.1;
         function px(x) { return ml + x / E * (W - ml - mr); }
         function py(y) { return H - mb - y / YMAX * (H - mt - mb); }
         c.lineWidth = 1; c.fillStyle = TEXT; c.textAlign = 'center';
         [0, 20, 40, 60].forEach(function (g) { c.strokeStyle = GRID; c.beginPath(); c.moveTo(px(g), mt); c.lineTo(px(g), H - mb); c.stroke(); c.fillText(String(g), px(g), H - mb + 14); });
         c.textAlign = 'right';
         [0, 0.5, 1].forEach(function (g) { c.strokeStyle = GRID; c.beginPath(); c.moveTo(ml, py(g)); c.lineTo(W - mr, py(g)); c.stroke(); c.fillText(g.toFixed(1), ml - 6, py(g) + 4); });
         c.strokeStyle = AXIS; c.beginPath(); c.moveTo(ml, mt); c.lineTo(ml, H - mb); c.lineTo(W - mr, H - mb); c.stroke();
         c.textAlign = 'center'; c.fillText('epoch', ml + (W - ml - mr) / 2, H - 5);
         var cur = reduce ? E : clamp(t / DUR, 0, 1) * E, whole = Math.floor(cur), frac = cur - whole;
         function line(arr, colour, width) {
            c.strokeStyle = colour; c.lineWidth = width; c.lineJoin = 'round'; c.beginPath(); c.moveTo(px(0), py(arr[0]));
            for (var k = 1; k <= whole; k++) { c.lineTo(px(k), py(arr[k])); }
            if (whole < E) { c.lineTo(px(cur), py(lerp(arr[whole], arr[whole + 1], frac))); }
            c.stroke();
         }
         c.shadowColor = rgba(COL.green, 0.5); c.shadowBlur = 8; line(train, COL.green, 2.4); c.shadowColor = rgba(COL.cyan, 0.5); line(val, COL.cyan, 2.4); c.shadowBlur = 0;
         if (cur >= best) {                                                       // early stopping: keep the weights from the best validation epoch
            var a = reduce ? 1 : ease((cur - best) / 5);
            c.setLineDash([5, 5]); c.strokeStyle = rgba(COL.amber, 0.85 * a); c.lineWidth = 1.5; c.beginPath(); c.moveTo(px(best), mt); c.lineTo(px(best), H - mb); c.stroke(); c.setLineDash([]);
            c.fillStyle = rgba(COL.amber, a); c.beginPath(); c.arc(px(best), py(val[best]), 4.5, 0, TAU); c.fill();
            c.textAlign = 'right'; c.fillStyle = rgba(COL.amber, a); c.fillText('early stop: best val at epoch ' + best, px(best) - 8, mt + 12);
         }
         c.textAlign = 'left';                                                    // legend, in the empty area on the right
         c.strokeStyle = COL.green; c.lineWidth = 2.4; c.beginPath(); c.moveTo(W - mr - 92, py(0.82)); c.lineTo(W - mr - 72, py(0.82)); c.stroke();
         c.fillStyle = COL.green; c.fillText('train', W - mr - 66, py(0.82) + 4);
         c.strokeStyle = COL.cyan; c.beginPath(); c.moveTo(W - mr - 92, py(0.7)); c.lineTo(W - mr - 72, py(0.7)); c.stroke();
         c.fillStyle = COL.cyan; c.fillText('validation', W - mr - 66, py(0.7) + 4);
         var shown = Math.min(E, whole);
         p.say('epoch', String(shown)); p.say('train', train[shown].toFixed(3)); p.say('val', val[shown].toFixed(3));
      };
      return p;
   }

   /* ---------- wiring ---------- */
   var makers = { regression: regression, kmeans: kmeans, collinearity: collinearity, loss: loss };
   var panels = [], running = false, last = 0;

   function frame(now) {
      var dt = (now - last) / 1000, any = false;
      if (!(dt > 0)) { dt = 0.016; } if (dt > 0.05) { dt = 0.05; }   // a frame stamp can precede the start time: never step backwards
      last = now;
      panels.forEach(function (p) { if (p.visible) { any = true; p.update(dt); p.draw(); } });
      if (any && !document.hidden) { requestAnimationFrame(frame); } else { running = false; }
   }
   function kick() { if (!running && !reduce && !document.hidden) { running = true; last = performance.now(); requestAnimationFrame(frame); } }
   function settle(p) { var guard = 0; p.reset(); while (!p.finished && guard++ < 3000) { p.update(0.05); } p.draw(); }

   [].forEach.call(cards, function (card) {
      var make = makers[card.getAttribute('data-viz')];
      if (!make) { return; }
      var p = make(card); if (!p) { return; } p.card = card; card._p = p; panels.push(p);
      p.resize(); p.reset();
      if (reduce) { settle(p); } else { p.draw(); }
      var btn = card.querySelector('.viz-replay');
      if (btn) { btn.addEventListener('click', function () { p.reset(); p.visible = true; kick(); }); }
   });

   if ('IntersectionObserver' in window && !reduce) {
      var io = new IntersectionObserver(function (entries) {
         entries.forEach(function (en) {
            var p = en.target._p;
            if (en.isIntersecting && !p.visible) { p.visible = true; if (!p.started) { p.started = true; p.reset(); } kick(); }
            else if (!en.isIntersecting) { p.visible = false; }
         });
      }, { threshold: 0.2 });
      panels.forEach(function (p) { io.observe(p.card); });
   } else if (!reduce) {
      panels.forEach(function (p) { p.visible = true; });
      kick();
   }
   document.addEventListener('visibilitychange', function () { if (!document.hidden) { kick(); } });

   var timer;
   function relayout() { clearTimeout(timer); timer = setTimeout(function () { panels.forEach(function (p) { p.resize(); p.draw(); }); }, 100); }
   if ('ResizeObserver' in window) { panels.forEach(function (p) { new ResizeObserver(relayout).observe(p.canvas); }); }
   else { window.addEventListener('resize', relayout); }
})();
