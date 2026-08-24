<script>
const D = __PAYLOAD__;
const NS = "http://www.w3.org/2000/svg";
const MODEL_VAR = {qwen30b:"--m1", oss120b:"--m2", "kimi27-sglang":"--m3"};
const el = (n, a = {}) => { const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };

/* ---- tooltip ---- */
const tip = document.getElementById("tip");
function hook(node, text){
  node.addEventListener("mouseenter", e => { tip.textContent = text; tip.style.opacity = 1; });
  node.addEventListener("mousemove", e => {
    tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top  = (e.clientY - tip.offsetHeight - 10) + "px";
  });
  node.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
}

/* ---- verdict tally ---- */
(() => {
  const p = D.pooled, tot = p.win + p.tie + p.lose;
  const bar = document.getElementById("tally");
  [["win","var(--pos)",p.win],["tie","var(--mid)",p.tie],["lose","var(--neg)",p.lose]]
    .forEach(([k,c,v],i) => {
      const d = document.createElement("div");
      d.style.cssText = `background:${c};flex:${v} 0 0;${i===1?"color:var(--ink-2)":""}`;
      if (v / tot > 0.12) d.textContent = v;
      hook(d, `${k}: ${v} of ${tot} kernels (${(100*v/tot).toFixed(0)}%)`);
      bar.appendChild(d);
    });
  document.getElementById("abrows").innerHTML = D.ab.map(a => `
    <tr><td>${a.model} <span class="tag">${a.lang}</span></td>
    <td>${a.points.length}</td><td>${a.win}</td><td>${a.lose}</td><td>${a.tie}</td>
    <td>${a.p.toFixed(3)}</td></tr>`).join("");
})();

/* ---- paired scatters (log-log) ---- */
(() => {
  const LO = 0.9, HI = 40, W = 300, H = 236, M = {t:10, r:12, b:34, l:38};
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const lg = v => Math.log(v), a0 = lg(LO), a1 = lg(HI);
  const px = v => M.l + (lg(v) - a0) / (a1 - a0) * iw;
  const py = v => M.t + ih - (lg(v) - a0) / (a1 - a0) * ih;
  const TICKS = [1, 2, 5, 10, 20];

  D.ab.forEach(ab => {
    const wrap = document.createElement("div");
    wrap.className = "panel";
    const col = `var(${MODEL_VAR[ab.model]})`;
    wrap.innerHTML = `<div class="ptitle"><h3>${ab.model} &middot; ${ab.lang}</h3></div>
      <div class="pstat"><b>${ab.win}</b> better &nbsp;<b>${ab.lose}</b> worse &nbsp;<b>${ab.tie}</b> tied
      &nbsp;&middot;&nbsp; p = <b>${ab.p.toFixed(3)}</b></div>`;
    const s = el("svg", {viewBox:`0 0 ${W} ${H}`, role:"img",
      "aria-label":`${ab.model} ${ab.lang}: best speedup per kernel, without skills versus with skills`});

    // coarse-grid region: everything outside the [<=5x, <=5x] square
    const f = px(5), g = py(5);
    s.appendChild(el("path", {d:`M${M.l},${M.t} H${M.l+iw} V${M.t+ih} H${f} V${g} H${M.l} Z`,
      fill:"var(--band)"}));
    s.appendChild(el("line", {x1:f, y1:M.t, x2:f, y2:M.t+ih, class:"gridline"}));
    s.appendChild(el("line", {x1:M.l, y1:g, x2:M.l+iw, y2:g, class:"gridline"}));

    TICKS.forEach(t => {
      s.appendChild(el("line", {x1:px(t), y1:M.t+ih, x2:px(t), y2:M.t+ih+4, class:"axis"}));
      const tx = el("text", {x:px(t), y:M.t+ih+15, class:"tick", "text-anchor":"middle"});
      tx.textContent = t + "x"; s.appendChild(tx);
      s.appendChild(el("line", {x1:M.l-4, y1:py(t), x2:M.l, y2:py(t), class:"axis"}));
      const ty = el("text", {x:M.l-7, y:py(t)+3, class:"tick", "text-anchor":"end"});
      ty.textContent = t + "x"; s.appendChild(ty);
    });
    s.appendChild(el("line", {x1:M.l, y1:M.t+ih, x2:M.l+iw, y2:M.t+ih, class:"axis"}));
    s.appendChild(el("line", {x1:M.l, y1:M.t, x2:M.l, y2:M.t+ih, class:"axis"}));
    s.appendChild(el("line", {x1:px(LO), y1:py(LO), x2:px(HI), y2:py(HI), class:"diag"}));

    const xl = el("text", {x:M.l+iw/2, y:H-3, class:"axlabel", "text-anchor":"middle"});
    xl.textContent = "no skills"; s.appendChild(xl);
    const yl = el("text", {x:-(M.t+ih/2), y:11, class:"axlabel", "text-anchor":"middle",
      transform:"rotate(-90)"});
    yl.textContent = "with skills"; s.appendChild(yl);

    ab.points.forEach(p => {
      const d = Math.abs(p.sk - p.no) < 1e-9;
      const c = el("circle", {cx:px(p.no), cy:py(p.sk), r:d ? 3.2 : 4.2,
        fill:d ? "var(--mid)" : col, "fill-opacity":d ? 1 : .82,
        stroke:d ? "var(--rule-2)" : "var(--surface)", class:"dot"});
      hook(c, `${p.k}  ${p.no.toFixed(2)}x -> ${p.sk.toFixed(2)}x`);
      s.appendChild(c);
    });
    wrap.appendChild(s);
    document.getElementById("scatters").appendChild(wrap);
  });
})();

/* ---- outcome composition ---- */
(() => {
  const KEYS = [["ok","--good"],["incorrect","--critical"],["build_error","--serious"],
                ["score_error","--warning"]];
  const host = document.getElementById("outcomes");
  const rows = D.arms.slice().sort((a,b) =>
    (a.lang > b.lang ? 1 : a.lang < b.lang ? -1 : 0) || a.model.localeCompare(b.model) ||
    (a.skills - b.skills));
  const LW = 186, BH = 26, GAP = 12, W = 720;
  const H = rows.length * (BH + GAP);
  const s = el("svg", {viewBox:`0 0 ${W} ${H + 22}`, role:"img",
    "aria-label":"share of judged calls by outcome for each arm"});
  rows.forEach((r, i) => {
    const y = i * (BH + GAP);
    const label = `${r.model}/${r.lang}${r.skills ? " +skills" : ""}`;
    const t = el("text", {x:LW - 10, y:y + BH/2 + 4, class:"axlabel", "text-anchor":"end"});
    t.textContent = label; t.setAttribute("style","fill:var(--ink);font-size:11.5px");
    s.appendChild(t);
    let x = LW;
    const iw = W - LW - 52;
    KEYS.forEach(([k, v]) => {
      const share = r.status[k] / r.total;
      if (share <= 0) return;
      const w = Math.max(share * iw, share > 0 ? 1.6 : 0);
      const rect = el("rect", {x, y, width:Math.max(w - 2, 1), height:BH, rx:2,
        fill:`var(${v})`, "fill-opacity":.9});
      hook(rect, `${label} - ${k}: ${r.status[k]} calls (${(100*share).toFixed(1)}%)`);
      s.appendChild(rect); x += w;
    });
    const other = (r.status.timeout + r.status.overfit) / r.total;
    if (other > 0){
      const w = Math.max(other * iw, 1.6);
      const rect = el("rect", {x, y, width:Math.max(w - 2, 1), height:BH, rx:2,
        fill:"var(--ink-3)", "fill-opacity":.7});
      hook(rect, `${label} - timeout/overfit: ${r.status.timeout + r.status.overfit} calls`);
      s.appendChild(rect);
    }
    const pct = el("text", {x:W - 46, y:y + BH/2 + 4, class:"tick", "text-anchor":"start"});
    pct.textContent = (100 * r.status.ok / r.total).toFixed(0) + "% ok";
    pct.setAttribute("style","fill:var(--ink-2);font-size:10.5px");
    s.appendChild(pct);
  });
  host.appendChild(s);
})();

/* ---- arm table ---- */
(() => {
  document.getElementById("armrows").innerHTML = D.arms.map(a => {
    const pc = k => (100 * a.status[k] / a.total).toFixed(1);
    const coarse = a.median > 5;
    return `<tr>
      <td>${a.model}<span class="tag">${a.lang}</span>${a.skills ? '<span class="tag">skills</span>' : ""}</td>
      <td>${a.job}</td><td>${a.kernels}</td><td>${a.subs}</td>
      <td>${pc("ok")}</td><td>${pc("incorrect")}</td>
      <td>${a.median.toFixed(3)}${coarse ? '<span class="tag" title="in the coarse-grid region">coarse</span>' : ""}</td>
    </tr>`;
  }).join("");
})();
</script>
