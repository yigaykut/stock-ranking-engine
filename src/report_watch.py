"""Izleme listesi panosu — gunluk pozisyon takibi.

Ana gorsel: her hisse icin bir FIYAT EKSENI seridi.
    stop ──────── giris ── simdi ──────── kisa hedef ──── uzun hedef
Boylece "neredeyim, nereye gidiyorum, nerede yanlisim" tek bakista okunur.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .report import _clean
from .theme import SERIES, hero_svg, sigil_svg

RISK_COLOR = {
    "GUVENLI":     SERIES[5],   # jade
    "IZLE":        SERIES[1],   # cyan
    "DIKKAT":      SERIES[2],   # gold
    "YUKSEK_RISK": SERIES[7],   # amber
    "SAT":         SERIES[0],   # crimson
}
RISK_GLYPH = {"GUVENLI": "●", "IZLE": "◐", "DIKKAT": "◑", "YUKSEK_RISK": "◕", "SAT": "✕"}


def _payload(results: list[dict], history: list[dict]) -> dict:
    rows = []
    for r in results:
        if not r.get("ok"):
            rows.append({"ticker": r["ticker"], "ok": False,
                         "reason": r.get("reason", "bilinmiyor")})
            continue

        a = r["analysis"]
        pos = r["position"]
        st, lt = a.get("short_term", {}), a.get("long_term", {})
        stops = a.get("stops", {})

        hist = [h for h in history if h.get("ticker") == r["ticker"]]
        spark = []
        for h in hist[-60:]:
            try:
                spark.append({"d": h["date"], "p": float(h["price"])})
            except (TypeError, ValueError, KeyError):
                pass

        rows.append({
            "ok": True,
            "ticker": r["ticker"],
            "name": r.get("name"),
            "sector": r.get("sector"),
            "currency": r.get("currency", "USD"),
            "price": a.get("price"),
            "entry": a.get("entry_price"),
            "addedDate": pos.get("added_date"),
            "quantity": pos.get("quantity"),
            "note": pos.get("note") or "",
            "pnlPct": a.get("pnl_pct"),
            "riskLevel": a.get("risk_level"),
            "riskLevelTr": a.get("risk_level_tr"),
            "action": a.get("action_tr"),
            "signals": a.get("signals", []),
            "shortTarget": st.get("target") if st.get("available") else None,
            "shortUpside": st.get("upside_pct") if st.get("available") else None,
            "shortMethod": st.get("method"),
            "shortCandidates": st.get("candidates", []),
            "resistances": st.get("next_resistances", []),
            "longTarget": lt.get("target") if lt.get("available") else None,
            "longUpside": lt.get("upside_pct") if lt.get("available") else None,
            "longCandidates": lt.get("candidates", []),
            "analystHigh": lt.get("analyst_high"),
            "analystLow": lt.get("analyst_low"),
            "stop": stops.get("active_stop"),
            "stopMethod": stops.get("active_method"),
            "stopWhy": stops.get("initial_why"),
            "stopDistPct": stops.get("distance_pct"),
            "stopPnlPct": a.get("stop_pnl_pct"),
            "initialStop": stops.get("initial_stop"),
            "chandelier": stops.get("chandelier_stop"),
            "riskReward": a.get("risk_reward"),
            "technical": a.get("technical", {}),
            "spark": spark,
        })

    return {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y.%m.%d · %H:%M UTC"),
        "rows": rows,
        "riskColors": RISK_COLOR,
        "riskGlyphs": RISK_GLYPH,
    }


_CSS = """
.wl-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
  background:var(--rule-2);border:1px solid var(--rule-2);margin-top:-1px}
.card{background:linear-gradient(160deg,rgba(30,16,19,.72),rgba(10,4,6,.5));
  border:1px solid var(--rule-2);margin-bottom:14px;position:relative;
  clip-path:polygon(0 0,calc(100% - 18px) 0,100% 18px,100% 100%,18px 100%,0 calc(100% - 18px))}
.card::before{content:"";position:absolute;top:0;right:0;width:18px;height:18px;
  background:linear-gradient(225deg,var(--rl,var(--crimson)) 50%,transparent 51%);opacity:.7}
.card-head{display:grid;grid-template-columns:1fr auto auto;gap:18px;align-items:center;
  padding:16px 20px;border-bottom:1px solid var(--rule-2)}
.ch-id .tk{font:400 27px/1 var(--disp);letter-spacing:.02em}
.ch-id .nm{font:500 10px/1.5 var(--mono);letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink-3);margin-top:4px}
.ch-px{text-align:right}
.ch-px .v{font:400 27px/1 var(--disp)}
.ch-px .s{font:500 9.5px/1.5 var(--mono);letter-spacing:.13em;color:var(--ink-3);margin-top:4px}
.badge{display:inline-flex;align-items:center;gap:7px;padding:8px 13px;border:1px solid;
  font:600 10px/1 var(--mono);letter-spacing:.17em;text-transform:uppercase;white-space:nowrap}
.pnl{font:400 22px/1 var(--disp)}
.axis-wrap{padding:22px 20px 14px}
.axis{position:relative;height:56px}
.axis .track{position:absolute;left:0;right:0;top:24px;height:5px;background:var(--track);
  outline:1px solid rgba(160,140,140,.09)}
.axis .fill{position:absolute;top:24px;height:5px}
.axis .mk{position:absolute;top:8px;width:2px;height:37px;transform:translateX(-1px)}
.axis .lab{position:absolute;top:-4px;transform:translateX(-50%);white-space:nowrap;
  font:500 9px/1.35 var(--mono);letter-spacing:.09em;text-align:center;text-transform:uppercase}
.axis .lab b{display:block;font:600 11.5px/1.3 var(--mono);letter-spacing:.02em;color:var(--ink)}
.axis .lab.below{top:auto;bottom:-16px}
.sig{padding:0 20px 18px}
.sig-item{display:grid;grid-template-columns:26px 1fr;gap:12px;padding:9px 0;
  border-top:1px solid var(--rule-2);align-items:start}
.sig-item .sv{font:600 9px/1.7 var(--mono);letter-spacing:.1em;text-align:center;
  border:1px solid;padding:1px 0}
.sig-item .st{font-size:12.5px;color:var(--ink)}
.sig-item .sd{font-size:11.5px;color:var(--ink-3);margin-top:3px;line-height:1.5}
.act{margin:0 20px 18px;padding:11px 14px;border-left:2px solid var(--rl,var(--crimson));
  background:rgba(240,72,58,.06);font-size:12.5px;color:var(--ink-2)}
/* Yalnizca dogrudan cocuk <b> baslik olur; metin ici vurgular satir ici kalir. */
.act > b{color:var(--ink);display:block;font:600 9.5px/1 var(--mono);letter-spacing:.19em;
  text-transform:uppercase;margin-bottom:6px}
.act b{color:var(--ink);font-weight:600}
.tech{display:grid;grid-template-columns:repeat(auto-fill,minmax(104px,1fr));gap:1px;
  background:var(--rule-2);border-top:1px solid var(--rule-2)}
.tech div{background:var(--surface);padding:9px 12px}
.tech .k{font:600 8.5px/1 var(--mono);letter-spacing:.14em;color:var(--ink-3);text-transform:uppercase}
.tech .v{font:400 15px/1 var(--mono);margin-top:6px;color:var(--ink)}
.empty{border:1px dashed var(--rule);padding:34px 24px;text-align:center;color:var(--ink-2)}
.empty code{font:500 12px/1.9 var(--mono);color:var(--crimson);display:block;margin-top:10px}
.spark{height:30px;width:100%;display:block;margin-top:10px}
.spark path{fill:none;stroke:var(--rl,var(--crimson));stroke-width:1.4}
@media (max-width:720px){
  .card-head{grid-template-columns:1fr;gap:12px}
  .axis .lab{font-size:8px}
}
"""


def build_html(results: list[dict], history: list[dict],
               title: str = "IZLEME LISTESI / POZISYON TAKIBI") -> str:
    data = _clean(_payload(results, history))
    data_json = json.dumps(data, ensure_ascii=False, allow_nan=False)

    # Ana panonun stilini yeniden kullan, izleme listesine ozel stilleri ekle
    from .report import _CSS as BASE_CSS

    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{BASE_CSS}{_CSS}</style>
<div class="hero" style="min-height:min(52vh,360px)">
  {hero_svg()}
  <div class="hero-inner">
    <div class="eyebrow">Gunluk pozisyon ve risk takibi</div>
    <h1>IZLEME<em>LISTESI</em></h1>
    <div class="hero-meta">
      <span>GUNCELLEME <b id="gen"></b></span>
      <span>POZISYON <b id="m-cnt"></b></span>
      <span>RISKTE <b id="m-risk"></b></span>
    </div>
  </div>
</div>

<div class="wrap">
  <div class="wl-tiles" id="tiles"></div>

  <section>
    <div class="sec-head"><span class="sec-num">I</span><h2>Pozisyonlar</h2></div>
    <div class="sec-rule"></div>
    <p class="sec-note">En riskliden guvenliye siralanmistir. Fiyat ekseni soldan saga:
      <b>stop</b> &rarr; <b>giris</b> &rarr; <b>guncel</b> &rarr; <b>kisa vadeli hedef</b>
      &rarr; <b>uzun vadeli hedef</b>.</p>
    <div id="cards"></div>
  </section>

  <div class="disclaimer">
    <b>Uyari</b>
    Hedefler ve stop seviyeleri istatistiksel/teknik hesaplamalardir, gelecek
    fiyat taahhudu degildir. Yatirim tavsiyesi degildir; kararlarindan yalnizca
    sen sorumlusun.
  </div>
  <footer>
    <span>Gunluk calistir &middot; python run.py watch update</span>
    <span id="foot-gen"></span>
  </footer>
</div>
<script>
const D = {data_json};
const RC = D.riskColors, RG = D.riskGlyphs;
const f2 = v => v == null ? '—' : Number(v).toFixed(2);
const f1 = v => v == null ? '—' : Number(v).toFixed(1);
const pct = v => v == null ? '—' : (v >= 0 ? '+' : '') + Number(v).toFixed(1) + '%';
const ok = D.rows.filter(r => r.ok);

document.getElementById('gen').textContent = D.generatedAt;
document.getElementById('foot-gen').textContent = D.generatedAt;
document.getElementById('m-cnt').textContent = ok.length;
document.getElementById('m-risk').textContent =
  ok.filter(r => ['SAT','YUKSEK_RISK'].includes(r.riskLevel)).length;

/* ---------- ozet ---------- */
(function tiles() {{
  const held = ok.filter(r => r.entry != null);
  const avg = held.length ? held.reduce((a,r)=>a+(r.pnlPct||0),0)/held.length : null;
  const t = [
    ['Pozisyon', ok.length, 'Σ'],
    ['Ortalama K/Z', avg == null ? '—' : pct(avg), 'Δ'],
    ['Satis sinyali', ok.filter(r=>r.riskLevel==='SAT').length, '✕'],
    ['Yuksek risk', ok.filter(r=>r.riskLevel==='YUKSEK_RISK').length, '◕'],
    ['Guvenli', ok.filter(r=>r.riskLevel==='GUVENLI').length, '●'],
  ];
  document.getElementById('tiles').innerHTML = t.map(([k,v,g]) =>
    `<div class="tile"><span class="glyph">${{g}}</span>
      <div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join('');
}})();

/* ---------- fiyat ekseni ---------- */
function axis(r) {{
  const pts = [
    {{k:'stop',   v:r.stop,        lab:'STOP',       col:RC.SAT}},
    {{k:'entry',  v:r.entry,       lab:'GIRIS',      col:'var(--ink-2)'}},
    {{k:'price',  v:r.price,       lab:'GUNCEL',     col:'var(--ink)'}},
    {{k:'st',     v:r.shortTarget, lab:'KISA HEDEF', col:RC.IZLE}},
    {{k:'lt',     v:r.longTarget,  lab:'UZUN HEDEF', col:RC.GUVENLI}},
  ].filter(p => p.v != null);
  if (pts.length < 2) return '';

  const vals = pts.map(p => p.v);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.12 || hi * 0.05;
  lo -= pad; hi += pad;
  const x = v => ((v - lo) / (hi - lo)) * 100;

  const px = x(r.price);
  const stopX = r.stop != null ? x(r.stop) : 0;
  // stop -> guncel arasi: elde tutulan bolge
  const fill = `<i class="fill" style="left:${{stopX}}%;width:${{Math.max(0,px-stopX)}}%;
      background:linear-gradient(90deg,${{RC.SAT}}44,${{RC[r.riskLevel]}})"></i>`;

  const marks = pts.map(p => `<i class="mk" style="left:${{x(p.v)}}%;background:${{p.col}}"></i>`).join('');
  // Etiketler ust/alt siraya bolunur ki komsu isaretler ust uste binmesin
  const labs = pts.map(p => {{
    const below = (p.k === 'entry' || p.k === 'lt');
    const val = `<b>${{f2(p.v)}}</b>`;
    const body = below ? val + p.lab : p.lab + val;
    return `<span class="lab${{below ? ' below' : ''}}" `
         + `style="left:${{x(p.v)}}%;color:${{p.col}}">${{body}}</span>`;
  }}).join('');

  return `<div class="axis-wrap"><div class="axis">
    <i class="track"></i>${{fill}}${{marks}}${{labs}}</div></div>`;
}}

function sparkline(r) {{
  if (!r.spark || r.spark.length < 3) return '';
  const ps = r.spark.map(s => s.p);
  const lo = Math.min(...ps), hi = Math.max(...ps);
  const rng = (hi - lo) || 1;
  const d = ps.map((p,i) =>
    `${{(i/(ps.length-1))*100}},${{28 - ((p-lo)/rng)*26}}`).join('L');
  return `<svg class="spark" viewBox="0 0 100 30" preserveAspectRatio="none">
    <path d="M${{d}}"/></svg>`;
}}

/* Stop ve hedef aciklamalari — her parcasi bagimsiz, eksik veri kirmaz */
function detailLines(r) {{
  const out = [];
  if (r.stop != null) {{
    out.push(`Stop <b>${{f2(r.stop)}}</b> (${{r.stopMethod}}) — guncel fiyatin `
           + `%${{f1(r.stopDistPct)}} altinda. ${{r.stopWhy || ''}}`);
  }}
  if (r.shortTarget != null) {{
    out.push(`Kisa vadeli hedef <b>${{f2(r.shortTarget)}}</b> (${{pct(r.shortUpside)}}, `
           + `1-3 ay, yontem: ${{r.shortMethod}})`);
  }}
  if (r.longTarget != null) {{
    out.push(`Uzun vadeli hedef <b>${{f2(r.longTarget)}}</b> (${{pct(r.longUpside)}}, 12 ay)`);
  }}
  if (!out.length) return '';
  return `<div style="color:var(--ink-3);margin-top:8px;line-height:1.7">`
       + out.join('<br>') + `</div>`;
}}

/* ---------- kartlar ---------- */
function render() {{
  const host = document.getElementById('cards');
  if (!ok.length) {{
    host.innerHTML = `<div class="empty">Izleme listesi bos.
      Hisse eklemek icin:<code>python run.py watch add NVDA --price 219.40</code></div>`;
    return;
  }}

  host.innerHTML = ok.map(r => {{
    const col = RC[r.riskLevel] || RC.IZLE;
    const pnlCls = r.pnlPct == null ? '' : (r.pnlPct >= 0 ? 'pos' : 'neg');

    const sigs = r.signals.length ? `<div class="sig">` + r.signals.map(s => {{
      const sc = s.siddet >= 5 ? RC.SAT : s.siddet >= 4 ? RC.YUKSEK_RISK
               : s.siddet >= 3 ? RC.DIKKAT : RC.IZLE;
      return `<div class="sig-item">
        <span class="sv" style="color:${{sc}};border-color:${{sc}}">${{s.siddet}}</span>
        <div><div class="st">${{s.baslik}}</div><div class="sd">${{s.aciklama}}</div></div>
      </div>`;
    }}).join('') + `</div>` : '';

    const t = r.technical || {{}};
    const tech = [
      ['Fiyat', f2(r.price)], ['1 gun', pct(t.change_1d_pct)],
      ['5 gun', pct(t.change_5d_pct)], ['21 gun', pct(t.change_21d_pct)],
      ['RSI 14', f1(t.rsi14)], ['ATR %', f1(t.atr_pct)],
      ['MA50', f2(t.ma50)], ['MA150', f2(t.ma150)], ['MA200', f2(t.ma200)],
      ['52h zirve', f2(t['52w_high'])], ['52h dip', f2(t['52w_low'])],
      ['Risk/Getiri', r.riskReward == null ? '—' : r.riskReward + 'x'],
    ].map(([k,v]) => `<div><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join('');

    const entryLine = r.entry != null
      ? `<div class="ch-px"><div class="v ${{pnlCls}}">${{pct(r.pnlPct)}}</div>
         <div class="s">GIRIS ${{f2(r.entry)}} · ${{r.addedDate || ''}}</div></div>`
      : `<div class="ch-px"><div class="v" style="color:var(--ink-3)">IZLEME</div>
         <div class="s">POZISYON YOK · ${{r.addedDate || ''}}</div></div>`;

    return `<div class="card" style="--rl:${{col}}">
      <div class="card-head">
        <div class="ch-id"><div class="tk">${{r.ticker}}</div>
          <div class="nm">${{r.name || ''}} · ${{r.sector || ''}}</div>
          ${{sparkline(r)}}</div>
        ${{entryLine}}
        <span class="badge" style="color:${{col}};border-color:${{col}}">
          ${{RG[r.riskLevel] || ''}} ${{r.riskLevel.replace('_',' ')}}</span>
      </div>
      ${{axis(r)}}
      <div class="act"><b>Ne yapmali</b>${{r.action}}${{detailLines(r)}}</div>
      ${{sigs}}
      <div class="tech">${{tech}}</div>
    </div>`;
  }}).join('');
}}
render();
</script>"""


def write_html(results: list[dict], history: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(results, history), encoding="utf-8")
    return path
