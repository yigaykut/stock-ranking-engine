"""Asistan (chat) ve izleme listesine ekleme arayuzu.

TAMAMEN UCRETSIZ VE CEVRIMDISI CALISIR.
Ucretli bir dil modeli servisi kullanilmaz. Asistan, panoya zaten gomulu olan
veriden (skorlar, faktorler, hedefler, gerekceler) cevap uretir.

Bunun bir dil modeline gore AVANTAJI: uydurma (halusinasyon) yapamaz.
Her cevap, hesaplanmis bir sayiya veya config'deki gerekce metnine dayanir.
DEZAVANTAJI: serbest sohbet edemez, sadece tanidigi soru kaliplarini anlar.
Tanimadigi soruda ne yapabilecegini soyler.

Bu dosya saf metin (f-string degil) tutulur; JS suslu parantezleri boylece
kacislanmak zorunda kalmaz.
"""

CSS = """
/* ================= ASISTAN ================= */
.chat{display:grid;grid-template-columns:1fr;gap:0;border:1px solid var(--rule-2)}
.chat-log{min-height:230px;max-height:460px;overflow-y:auto;padding:18px;
  background:var(--surface);display:flex;flex-direction:column;gap:14px}
.msg{max-width:88%;font-size:13.5px;line-height:1.65}
.msg.me{align-self:flex-end;text-align:right}
.msg.me .b{display:inline-block;background:rgba(240,72,58,.14);
  border:1px solid var(--crimson-dim);padding:9px 14px;color:var(--ink)}
.msg.bot .b{display:block;border-left:2px solid var(--crimson);padding:2px 0 2px 14px;
  color:var(--ink-2)}
.msg.bot b{color:var(--ink)}
.msg .who{font:600 8.5px/1 var(--mono);letter-spacing:.2em;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:7px}
.msg table{border-collapse:collapse;margin:9px 0;font-size:12px;width:100%}
.msg td,.msg th{padding:4px 9px;border-bottom:1px solid var(--rule-2);text-align:right}
.msg td:first-child,.msg th:first-child{text-align:left}
.msg th{font:600 8.5px/1 var(--mono);letter-spacing:.13em;color:var(--ink-3);
  text-transform:uppercase}
.msg .num{font-family:var(--mono)}
.chat-in{display:grid;grid-template-columns:1fr auto;border-top:1px solid var(--rule-2)}
.chat-in input{border:0;background:var(--plane);padding:15px 18px;color:var(--ink);
  font:400 13.5px/1 var(--body);letter-spacing:0;text-transform:none;outline:none}
.chat-in input::placeholder{color:var(--ink-3)}
.chat-in button{border:0;border-left:1px solid var(--rule-2);background:var(--plane);
  color:var(--crimson);padding:0 24px;cursor:pointer;
  font:600 10px/1 var(--mono);letter-spacing:.19em;text-transform:uppercase}
.chat-in button:hover{background:rgba(240,72,58,.1)}
.chips{display:flex;flex-wrap:wrap;gap:7px;padding:13px 18px;border-top:1px solid var(--rule-2);
  background:var(--plane)}
.chip{border:1px solid var(--rule-2);background:transparent;color:var(--ink-3);cursor:pointer;
  padding:7px 12px;font:500 10px/1 var(--mono);letter-spacing:.1em}
.chip:hover{border-color:var(--crimson);color:var(--ink)}

/* ================= IZLEME LISTESINE EKLE ================= */
.addbtn{border:1px solid var(--rule-2);background:transparent;color:var(--ink-3);
  cursor:pointer;padding:5px 9px;font:600 9.5px/1 var(--mono);letter-spacing:.1em}
.addbtn:hover{border-color:var(--crimson);color:var(--crimson)}
.addbtn.on{background:var(--crimson);border-color:var(--crimson);color:#fff}
.basket{position:fixed;right:18px;bottom:18px;z-index:80;width:min(400px,calc(100vw - 36px));
  background:#150a0d;border:1px solid var(--crimson-dim);
  box-shadow:0 16px 54px rgba(0,0,0,.72);display:none}
.basket.on{display:block}
.basket h4{margin:0;padding:14px 17px;border-bottom:1px solid var(--rule-2);
  font:400 17px/1 var(--disp);letter-spacing:.03em;text-transform:uppercase;
  display:flex;justify-content:space-between;align-items:center}
.basket h4 span{font:600 9px/1 var(--mono);letter-spacing:.16em;color:var(--ink-3);cursor:pointer}
.basket h4 span:hover{color:var(--crimson)}
.basket-list{max-height:210px;overflow-y:auto}
.brow{display:grid;grid-template-columns:1fr 92px 26px;gap:9px;align-items:center;
  padding:9px 17px;border-bottom:1px solid var(--rule-2)}
.brow .bt{font:600 12px/1.3 var(--mono);letter-spacing:.05em}
.brow .bp{font:400 9px/1.4 var(--mono);color:var(--ink-3);margin-top:3px}
.brow input{width:100%;background:var(--plane);border:1px solid var(--rule-2);
  color:var(--ink);padding:6px 8px;font:400 11px/1 var(--mono);outline:none;text-align:right}
.brow input:focus{border-color:var(--crimson)}
.brow .x{background:none;border:0;color:var(--ink-3);cursor:pointer;font-size:15px;padding:0}
.brow .x:hover{color:var(--crimson)}
.basket-cmd{padding:14px 17px;border-top:1px solid var(--rule-2)}
.basket-cmd code{display:block;background:var(--plane);border:1px solid var(--rule-2);
  padding:11px 13px;font:400 10.5px/1.65 var(--mono);color:var(--ink-2);
  word-break:break-all;max-height:78px;overflow-y:auto}
.basket-act{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.basket-act button{border:1px solid var(--crimson-dim);background:transparent;
  color:var(--crimson);cursor:pointer;padding:10px;
  font:600 9.5px/1 var(--mono);letter-spacing:.13em;text-transform:uppercase}
.basket-act button:hover{background:rgba(240,72,58,.12)}
.basket-hint{font-size:11px;color:var(--ink-3);margin-top:9px;line-height:1.6}
.fab{position:fixed;right:18px;bottom:18px;z-index:79;background:var(--crimson);
  border:0;color:#fff;cursor:pointer;padding:14px 19px;
  font:600 10px/1 var(--mono);letter-spacing:.15em;text-transform:uppercase;
  box-shadow:0 10px 34px rgba(240,72,58,.4);display:none}
.fab.on{display:block}
"""


HTML = """
  <section>
    <div class="sec-head"><span class="sec-num">IV</span><h2>Asistan</h2></div>
    <div class="sec-rule"></div>
    <p class="sec-note">Hisseler, parametreler ve siralama hakkinda soru sor.
      Tamamen cevrimdisi calisir ve yalnizca bu sayfadaki hesaplanmis veriden
      cevap verir &mdash; bu yuzden uydurma cevap uretemez.</p>
    <div class="chat">
      <div class="chat-log" id="chatLog"></div>
      <div class="chips" id="chatChips"></div>
      <form class="chat-in" id="chatForm" autocomplete="off">
        <input id="chatIn" placeholder="Ornek: LQDT nasil?  /  en iyi 5  /  40 kurali nedir?">
        <button type="submit">Sor</button>
      </form>
    </div>
  </section>

  <button class="fab" id="fab">Secilenler (<span id="fabN">0</span>)</button>
  <div class="basket" id="basket">
    <h4>Izleme Listesi <span id="basketClose">KAPAT</span></h4>
    <div class="basket-list" id="basketList"></div>
    <div class="basket-cmd">
      <code id="basketCmd"></code>
      <div class="basket-act">
        <button id="copyCmd">Komutu kopyala</button>
        <button id="dlJson">JSON indir</button>
      </div>
      <div class="basket-hint">Komutu terminale yapistir, ya da JSON'u indirip
        <b>python run.py watch import &lt;dosya&gt;</b> calistir. Fiyat alanini bos
        birakirsan hisse yalnizca izlenir, pozisyon acilmis sayilmaz.</div>
    </div>
  </div>
"""


JS = r"""
/* =====================================================================
   IZLEME LISTESI SEPETI  (tarayicida saklanir)
   ===================================================================== */
const BASKET_KEY = 'invest_basket_v1';
let basket = {};
try { basket = JSON.parse(localStorage.getItem(BASKET_KEY) || '{}'); } catch (e) { basket = {}; }

function saveBasket() {
  try { localStorage.setItem(BASKET_KEY, JSON.stringify(basket)); } catch (e) {}
}

function basketCmd() {
  const ks = Object.keys(basket);
  if (!ks.length) return '';
  // Ayni fiyat bilgisine sahip olanlari tek komutta grupla
  const withPrice = ks.filter(k => basket[k].price);
  const noPrice = ks.filter(k => !basket[k].price);
  const lines = [];
  if (noPrice.length) lines.push('python run.py watch add ' + noPrice.join(','));
  for (const k of withPrice) {
    lines.push('python run.py watch add ' + k + ' --price ' + basket[k].price);
  }
  return lines.join('\n');
}

function renderBasket() {
  const ks = Object.keys(basket);
  document.getElementById('fabN').textContent = ks.length;
  document.getElementById('fab').classList.toggle('on', ks.length > 0 && !basketOpen);
  document.getElementById('basket').classList.toggle('on', basketOpen && ks.length > 0);

  document.getElementById('basketList').innerHTML = ks.map(function (k) {
    const row = (typeof DATA !== 'undefined')
      ? DATA.rows.find(function (r) { return r.ticker === k; }) : null;
    const px = row ? row.price : null;
    return '<div class="brow"><div><div class="bt">' + k + '</div>' +
      '<div class="bp">' + (px != null ? 'guncel ' + Number(px).toFixed(2) : '') + '</div></div>' +
      '<input data-t="' + k + '" placeholder="alis fiyati" value="' +
      (basket[k].price || '') + '">' +
      '<button class="x" data-x="' + k + '" title="cikar">&times;</button></div>';
  }).join('');

  document.getElementById('basketCmd').textContent = basketCmd() || '—';

  document.querySelectorAll('.brow input').forEach(function (inp) {
    inp.oninput = function () {
      const v = inp.value.trim();
      basket[inp.dataset.t].price = v;
      saveBasket();
      document.getElementById('basketCmd').textContent = basketCmd() || '—';
    };
  });
  document.querySelectorAll('.brow .x').forEach(function (b) {
    b.onclick = function () { toggleBasket(b.dataset.x, true); };
  });
  syncAddButtons();
}

let basketOpen = false;
function toggleBasket(ticker, forceRemove) {
  if (basket[ticker] || forceRemove) delete basket[ticker];
  else basket[ticker] = { price: '' };
  saveBasket();
  if (!Object.keys(basket).length) basketOpen = false;
  renderBasket();
}

function syncAddButtons() {
  document.querySelectorAll('.addbtn').forEach(function (b) {
    const on = !!basket[b.dataset.add];
    b.classList.toggle('on', on);
    b.textContent = on ? '✓ EKLI' : '+ EKLE';
  });
}

document.getElementById('fab').onclick = function () { basketOpen = true; renderBasket(); };
document.getElementById('basketClose').onclick = function () { basketOpen = false; renderBasket(); };

document.getElementById('copyCmd').onclick = function () {
  const t = basketCmd();
  if (!t) return;
  navigator.clipboard.writeText(t).then(function () {
    const b = document.getElementById('copyCmd');
    b.textContent = 'Kopyalandi';
    setTimeout(function () { b.textContent = 'Komutu kopyala'; }, 1600);
  }).catch(function () {});
};

document.getElementById('dlJson').onclick = function () {
  const out = Object.keys(basket).map(function (k) {
    const o = { ticker: k };
    if (basket[k].price) o.entry_price = Number(basket[k].price);
    return o;
  });
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'watchlist_ekle.json';
  a.click();
  URL.revokeObjectURL(a.href);
};

/* =====================================================================
   ASISTAN
   ===================================================================== */
const LOG = document.getElementById('chatLog');

/* Turkce'yi arama icin sadelestir: i/I noktalari, sapkalar, buyuk-kucuk */
function norm(s) {
  return (s || '').toLocaleLowerCase('tr')
    .replace(/ı/g, 'i').replace(/İ/g, 'i')
    .replace(/[ç]/g, 'c').replace(/[ğ]/g, 'g').replace(/[ö]/g, 'o')
    .replace(/[ş]/g, 's').replace(/[ü]/g, 'u').replace(/[âå]/g, 'a')
    .replace(/\s+/g, ' ').trim();
}

function say(who, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + who;
  d.innerHTML = '<div class="who">' + (who === 'me' ? 'SEN' : 'ASISTAN') +
                '</div><div class="b">' + html + '</div>';
  LOG.appendChild(d);
  LOG.scrollTop = LOG.scrollHeight;
}

const nf = function (v, d) { return v == null ? '—' : Number(v).toFixed(d == null ? 1 : d); };
const sgn = function (v) { return v == null ? '—' : (v >= 0 ? '+' : '') + Number(v).toFixed(1) + '%'; };

function findRow(q) {
  const n = norm(q);
  const up = q.toUpperCase();
  // once tam sembol eslesmesi
  let hit = DATA.rows.find(function (r) {
    return new RegExp('(^|[^A-Z])' + r.ticker + '([^A-Z]|$)').test(up);
  });
  if (hit) return hit;
  // sonra sirket adi
  return DATA.rows.find(function (r) {
    const nm = norm(r.name || '');
    return nm.length > 3 && n.indexOf(nm.split(' ')[0]) >= 0;
  });
}

function factorTable(row, list, title) {
  let h = '<table><tr><th>' + title + '</th><th>Puan</th><th>Agirlik</th></tr>';
  list.forEach(function (f) {
    h += '<tr><td>' + f.name + '</td><td class="num">' + nf(f.score, 0) +
         '</td><td class="num">' + nf(f.weight, 1) + '</td></tr>';
  });
  return h + '</table>';
}

function describe(row) {
  const av = row.factors.filter(function (f) { return f.available; });
  const strong = av.slice().sort(function (a, b) { return b.score - a.score; }).slice(0, 4);
  const weak = av.slice().sort(function (a, b) { return a.score - b.score; }).slice(0, 3);

  let h = '<b>' + row.ticker + ' — ' + (row.name || '') + '</b><br>' +
    'Siralamada <b>' + row.rank + '.</b> sirada, toplam etki puani <b>' +
    nf(row.total, 1) + '</b>/100. Fiyat <span class="num">' + nf(row.price, 2) +
    ' ' + row.currency + '</span>, sektor: ' + (row.sector || '—') + '.';

  if (row.penaltiesHit && row.penaltiesHit.length) {
    h += '<br><br><b>Ceza aldi:</b> ' + row.penaltiesHit.map(function (p) {
      return p.name_tr + ' (' + p.points + ')';
    }).join(', ') + '.';
  }
  if (row.lowConfidence) {
    h += '<br><br><b>Dikkat:</b> veri kapsamasi %' + nf(row.coverage * 100, 0) +
         ' — skor guven duzeltmesiyle notre cekildi.';
  }

  h += '<br>' + factorTable(row, strong, 'En guclu yanlari');
  h += factorTable(row, weak, 'En zayif yanlari');
  h += '<br><span style="color:var(--ink-3)">Getiri: 1 ay ' + sgn(row.returns['1m']) +
       ' &middot; 3 ay ' + sgn(row.returns['3m']) + ' &middot; 12 ay ' +
       sgn(row.returns['12m']) + '</span>';
  h += '<br><br>Bu hisseyi izleme listesine eklemek icin tablodaki satirinda ' +
       '<b>+ EKLE</b> butonuna bas.';
  return h;
}

function topList(rows, title, valFn, unit) {
  let h = '<b>' + title + '</b><table><tr><th>#</th><th>Sembol</th><th>Fiyat</th><th>' +
          (unit || 'Puan') + '</th></tr>';
  rows.forEach(function (r, i) {
    h += '<tr><td>' + (i + 1) + '</td><td><b>' + r.ticker + '</b></td>' +
         '<td class="num">' + nf(r.price, 2) + '</td>' +
         '<td class="num">' + nf(valFn(r), 1) + '</td></tr>';
  });
  return h + '</table>';
}

/* Parametre sozlugu — config'deki gerekcelerden uretilir (tek dogruluk kaynagi) */
function explainFactor(q) {
  const n = norm(q);
  const af = (DATA.diagnostics.active_factors || []);
  let best = null, bestLen = 0;
  af.forEach(function (f) {
    const words = norm(f.name_tr).split(/[^a-z0-9]+/).filter(function (w) { return w.length > 3; });
    words.forEach(function (w) {
      if (n.indexOf(w) >= 0 && w.length > bestLen) { best = f; bestLen = w.length; }
    });
    // id ile de eslesebilsin
    const idw = f.id.replace(/_/g, ' ');
    if (n.indexOf(norm(idw)) >= 0 && idw.length > bestLen) { best = f; bestLen = idw.length; }
  });
  if (!best) return null;

  let h = '<b>' + best.name_tr + '</b><br>Etki puani (agirlik): <b>' +
    nf(best.weight, 1) + '</b> &middot; kapsama %' + nf(best.coverage * 100, 0) +
    ' &middot; aile: ' + best.category + '<br><br>' +
    (best.rationale_tr || '').replace(/\n/g, ' ');

  const top = DATA.rows.filter(function (r) {
    const f = r.factors.find(function (x) { return x.id === best.id; });
    return f && f.available;
  }).sort(function (a, b) {
    const fa = a.factors.find(function (x) { return x.id === best.id; });
    const fb = b.factors.find(function (x) { return x.id === best.id; });
    return fb.score - fa.score;
  }).slice(0, 5);

  if (top.length) {
    h += '<br>' + topList(top, 'Bu parametrede en iyiler', function (r) {
      return r.factors.find(function (x) { return x.id === best.id; }).score;
    });
  }
  h += '<br><span style="color:var(--ink-3)">Tum parametrelerin detayli anlatimi: ' +
       'docs/PARAMETRELER.md</span>';
  return h;
}

function countAsked(q) {
  const m = norm(q).match(/(\d+)/);
  const n = m ? parseInt(m[1], 10) : 5;
  return Math.max(1, Math.min(20, n));
}

function scoreOf(row, id) {
  const f = row.factors.find(function (x) { return x.id === id; });
  return (f && f.available) ? f.score : null;
}

function byFactor(id, n) {
  return DATA.rows
    .filter(function (r) { return scoreOf(r, id) != null; })
    .sort(function (a, b) { return scoreOf(b, id) - scoreOf(a, id); })
    .slice(0, n);
}

/* Konu -> faktor id eslemesi */
const TOPICS = [
  { keys: ['ucuz', 'degerleme', 'deger', 'f/k', 'fk orani'], id: 'valuation_composite', t: 'En ucuz (degerleme)' },
  { keys: ['teknik', 'trend'], id: 'trend_structure', t: 'Teknik trend yapisi en guclu' },
  { keys: ['kirilim', 'sikisma', 'vcp', 'patlamaya'], id: 'breakout_setup', t: 'Kirilim kurulumu en iyi' },
  { keys: ['buyume', 'ciro', 'satis'], id: 'revenue_scaling', t: 'Ciro buyumesi en yuksek' },
  { keys: ['potansiyel', 'hedef', 'yukselme'], id: 'analyst_upside', t: 'Hedef fiyat potansiyeli en yuksek' },
  { keys: ['kucuk', 'olcek', 'katlanma'], id: 'size_opportunity', t: 'Olcek firsati en iyi' },
  { keys: ['kesfedilmemis', 'bilinmeyen', 'radar'], id: 'undiscovered', t: 'En az kesfedilmis' },
  { keys: ['guvenli', 'risk', 'saglam', 'oynaklik'], id: 'risk_drawdown', t: 'Riski en dusuk' },
  { keys: ['kalite', 'karli', 'karlilik'], id: 'quality_profitability', t: 'Karliligi en yuksek' },
  { keys: ['nakit', 'borc', 'saglik'], id: 'financial_health', t: 'Finansal sagligi en iyi' },
  { keys: ['analist', 'tavsiye'], id: 'eps_revision_momentum', t: 'Analist revizyonu en guclu' },
  { keys: ['momentum', 'hizli'], id: 'price_momentum_12_1', t: 'Momentumu en guclu' },
  { keys: ['asama', 'weinstein', 'baz'], id: 'stage2_breakout', t: 'Asama 2 kirilimi en taze' },
  { keys: ['hacim', 'toplama', 'obv'], id: 'volume_accumulation', t: 'Hacim onayi en guclu' },
  { keys: ['40 kural', 'rule of 40', 'kirk kural'], id: 'rule_of_40', t: '40 kuralinda en iyi' }
];

function answer(q) {
  const n = norm(q);

  if (!n) return 'Bir soru yaz.';

  if (/^(yardim|help|ne yapabilir|nasil kullan|komut)/.test(n)) {
    return '<b>Sunlari sorabilirsin:</b><table>' +
      '<tr><td>LQDT nasil?</td><td>tek hisse analizi</td></tr>' +
      '<tr><td>en iyi 10</td><td>siralama</td></tr>' +
      '<tr><td>en ucuz 5</td><td>degerlemeye gore</td></tr>' +
      '<tr><td>kirilim kurulumu en iyi</td><td>parametreye gore siralama</td></tr>' +
      '<tr><td>teknoloji hisseleri</td><td>sektore gore</td></tr>' +
      '<tr><td>50 dolar alti</td><td>fiyata gore</td></tr>' +
      '<tr><td>LQDT vs PRG</td><td>karsilastirma — farki hangi parametre yaratiyor</td></tr>' +
      '<tr><td>40 kurali nedir</td><td>parametre aciklamasi</td></tr>' +
      '<tr><td>kac hisse tarandi</td><td>tarama ozeti</td></tr></table>';
  }

  if (/(kac hisse|tarama ozet|ozet|istatistik|kac tane)/.test(n) && !/hisse (var|bul)/.test(n)) {
    const d = DATA.diagnostics;
    let h = '<b>Tarama ozeti</b><table>' +
      '<tr><td>Taranan evren</td><td class="num">' + (d.universe_size || '—') + '</td></tr>' +
      '<tr><td>Skorlanan</td><td class="num">' + DATA.totalScored + '</td></tr>' +
      '<tr><td>Listede gosterilen</td><td class="num">' + DATA.rows.length + '</td></tr>' +
      '<tr><td>Aktif parametre</td><td class="num">' + (d.active_factors || []).length + '</td></tr>' +
      '</table>';
    if ((d.auto_disabled || []).length) {
      h += 'Devre disi kalan: ' + d.auto_disabled.map(function (x) {
        return '<b>' + x.name_tr + '</b> (kapsama %' + nf(x.coverage * 100, 1) + ')';
      }).join(', ') + '. Agirliklari diger parametrelere dagitildi.';
    }
    return h;
  }

  /* --- karsilastirma --- */
  const cmp = q.toUpperCase().match(/\b([A-Z]{1,6})\b[^A-Z]+(?:VS|ILE|VE|\/)[^A-Z]*\b([A-Z]{1,6})\b/);
  if (cmp) {
    const a = DATA.rows.find(function (r) { return r.ticker === cmp[1]; });
    const b = DATA.rows.find(function (r) { return r.ticker === cmp[2]; });
    if (a && b) {
      let h = '<b>' + a.ticker + ' vs ' + b.ticker + '</b><table><tr><th>Kategori</th><th>' +
        a.ticker + '</th><th>' + b.ticker + '</th></tr>';
      h += '<tr><td>Toplam puan</td><td class="num">' + nf(a.total, 1) +
           '</td><td class="num">' + nf(b.total, 1) + '</td></tr>';
      h += '<tr><td>Fiyat</td><td class="num">' + nf(a.price, 2) +
           '</td><td class="num">' + nf(b.price, 2) + '</td></tr>';
      const cats = Object.keys(a.categoryScores || {});
      cats.forEach(function (c) {
        h += '<tr><td>' + c + '</td><td class="num">' + nf(a.categoryScores[c], 0) +
             '</td><td class="num">' + nf(b.categoryScores[c], 0) + '</td></tr>';
      });
      h += '</table>';

      /* --- FARKI YARATAN PARAMETRELER ---
         "Hangisi onde" sorusunun cevabi tek basina bir sayidir ve hicbir sey
         ogretmez. Asil soru "NEDEN onde": farki yaratan parametreler
         katkilarinin farkina gore siralanir. Katki = puan x agirlik, yani
         zaten toplama giren buyuklugun ta kendisi; bu yuzden asagidaki
         listenin toplami puan farkini aciklar. */
      const bmap = {};
      (b.factors || []).forEach(function (f) { bmap[f.id] = f; });
      const diffs = (a.factors || []).map(function (fa) {
        const fb = bmap[fa.id];
        if (!fb) return null;
        const ca = fa.contribution || 0, cb = fb.contribution || 0;
        return { name: fa.name, d: ca - cb, sa: fa.score, sb: fb.score,
                 ok: fa.available && fb.available };
      }).filter(function (x) { return x && x.ok && Math.abs(x.d) > 0.05; });
      diffs.sort(function (x, y) { return Math.abs(y.d) - Math.abs(x.d); });

      if (diffs.length) {
        h += '<b>Farki yaratan parametreler</b><table><tr><th>Parametre</th>' +
             '<th>' + a.ticker + '</th><th>' + b.ticker + '</th><th>Katki farki</th></tr>';
        diffs.slice(0, 6).forEach(function (x) {
          const lehte = x.d > 0 ? a.ticker : b.ticker;
          h += '<tr><td>' + x.name + '</td><td class="num">' + nf(x.sa, 0) +
               '</td><td class="num">' + nf(x.sb, 0) + '</td><td class="num">' +
               (x.d > 0 ? '+' : '') + nf(x.d, 1) + ' <span class="muted">' +
               lehte + '</span></td></tr>';
        });
        h += '</table>';
      }

      /* Eksik veri, karsilastirmayi sessizce carpitir: bir hissede olculemeyen
         parametrenin agirligi digerlerine dagitiliyor. Soylenmezse okuyucu
         iki sayiyi esit kosullarda sanir. */
      const covA = Math.round((a.coverage || 0) * 100), covB = Math.round((b.coverage || 0) * 100);
      if (Math.abs(covA - covB) >= 10) {
        h += '<b>Dikkat:</b> kapsama farki buyuk (' + a.ticker + ' %' + covA +
             ', ' + b.ticker + ' %' + covB + '). Daha az verisi olan hissenin ' +
             'puani notre (50) dogru cekilir; iki puan esit kosullarda uretilmedi. ';
      }
      const pa = (a.penaltiesHit || []).length, pb = (b.penaltiesHit || []).length;
      if (pa || pb) {
        h += '<b>Ceza:</b> ' + a.ticker + ' ' + pa + ' adet (' + nf(a.penalty, 1) +
             '), ' + b.ticker + ' ' + pb + ' adet (' + nf(b.penalty, 1) + '). ';
      }

      const win = a.total >= b.total ? a : b;
      const gap = Math.abs(a.total - b.total);
      /* Fark gunluk gurultunun altindaysa "onde" demek yaniltici olur (bulgu K2). */
      const noise = (DATA.noise && DATA.noise.median_abs_change) || 0;
      h += 'Toplam puanda <b>' + win.ticker + '</b> onde (' + nf(gap, 1) + ' puan fark).';
      if (noise && gap < noise) {
        h += ' Ancak bu fark gunluk tipik oynamanin (&plusmn;' + nf(noise, 1) +
             ') altinda: <b>ikisi ayirt edilemez</b> kabul edilmeli.';
      }
      return h;
    }
  }

  /* --- tek hisse --- */
  const row = findRow(q);
  if (row && !/en (iyi|ucuz|guclu|yuksek|dusuk)/.test(n)) return describe(row);

  /* --- sektor --- */
  const secs = Array.from(new Set(DATA.rows.map(function (r) { return r.sector; }).filter(Boolean)));
  const secHit = secs.find(function (s) {
    const t = norm(s);
    return n.indexOf(t) >= 0 || (t.indexOf('technology') >= 0 && /teknoloji/.test(n)) ||
      (t.indexOf('healthcare') >= 0 && /saglik/.test(n)) ||
      (t.indexOf('financial') >= 0 && /(finans|banka)/.test(n)) ||
      (t.indexOf('energy') >= 0 && /enerji/.test(n)) ||
      (t.indexOf('industrial') >= 0 && /sanayi/.test(n)) ||
      (t.indexOf('real estate') >= 0 && /(gayrimenkul|emlak)/.test(n)) ||
      (t.indexOf('basic materials') >= 0 && /(hammadde|malzeme)/.test(n));
  });
  if (secHit) {
    const rs = DATA.rows.filter(function (r) { return r.sector === secHit; }).slice(0, countAsked(q) || 8);
    if (rs.length) return topList(rs, secHit + ' sektorunde listedekiler', function (r) { return r.total; });
    return secHit + ' sektorunde listede hisse yok.';
  }

  /* --- fiyat filtresi --- */
  const pm = n.match(/(\d+(?:[.,]\d+)?)\s*(dolar|usd|\$)?\s*(alti|altinda|asagi|dan az|den az)/);
  if (pm) {
    const lim = parseFloat(pm[1].replace(',', '.'));
    const rs = DATA.rows.filter(function (r) { return r.price != null && r.price < lim; }).slice(0, 12);
    if (!rs.length) return 'Listede ' + lim + ' dolarin altinda hisse yok.';
    return topList(rs, lim + ' dolar altindakiler', function (r) { return r.total; });
  }
  const pm2 = n.match(/(\d+(?:[.,]\d+)?)\s*(dolar|usd|\$)?\s*(ustu|uzerinde|yukari|dan fazla|den fazla)/);
  if (pm2) {
    const lim = parseFloat(pm2[1].replace(',', '.'));
    const rs = DATA.rows.filter(function (r) { return r.price != null && r.price > lim; }).slice(0, 12);
    if (!rs.length) return 'Listede ' + lim + ' dolarin uzerinde hisse yok.';
    return topList(rs, lim + ' dolar uzerindekiler', function (r) { return r.total; });
  }

  /* --- parametreye gore siralama --- */
  for (let i = 0; i < TOPICS.length; i++) {
    const t = TOPICS[i];
    const matched = t.keys.some(function (k) { return n.indexOf(k) >= 0; });
    if (matched && /(en |siral|hangi|liste|goster|bul)/.test(n)) {
      const rs = byFactor(t.id, countAsked(q));
      if (rs.length) {
        return topList(rs, t.t, function (r) {
          return r.factors.find(function (x) { return x.id === t.id; }).score;
        });
      }
    }
  }

  /* --- parametre aciklamasi --- */
  if (/(nedir|ne demek|ne olcer|acikla|neden|nasil hesap)/.test(n)) {
    const ex = explainFactor(q);
    if (ex) return ex;
  }
  const ex2 = explainFactor(q);
  if (ex2 && n.split(' ').length <= 5) return ex2;

  /* --- genel siralama --- */
  if (/(en iyi|en yuksek|top|birinci|kazanan|onerdigin|tavsiye)/.test(n)) {
    return topList(DATA.rows.slice(0, countAsked(q)), 'Toplam etki puani en yuksek',
                   function (r) { return r.total; });
  }
  if (/(en kotu|en dusuk|sonuncu)/.test(n)) {
    const rs = DATA.rows.slice().reverse().slice(0, countAsked(q));
    return topList(rs, 'Listedeki en dusuk puanlilar', function (r) { return r.total; });
  }

  /* --- anlasilmadi --- */
  return 'Bunu anlayamadim. Su anda sunlari yapabiliyorum:<br><br>' +
    '&bull; <b>Tek hisse:</b> "LQDT nasil?" &mdash; sembolu yaz yeter<br>' +
    '&bull; <b>Siralama:</b> "en iyi 10", "en ucuz 5"<br>' +
    '&bull; <b>Parametreye gore:</b> "kirilim kurulumu en iyi", "buyumesi en yuksek"<br>' +
    '&bull; <b>Sektor / fiyat:</b> "teknoloji hisseleri", "50 dolar alti"<br>' +
    '&bull; <b>Karsilastirma:</b> "LQDT vs PRG"<br>' +
    '&bull; <b>Aciklama:</b> "40 kurali nedir"<br><br>' +
    '<span style="color:var(--ink-3)">Not: Bu asistan cevrimdisi ve kural tabanlidir; ' +
    'yalnizca bu sayfadaki hesaplanmis veriden cevap verir.</span>';
}

document.getElementById('chatForm').onsubmit = function (e) {
  e.preventDefault();
  const inp = document.getElementById('chatIn');
  const q = inp.value.trim();
  if (!q) return;
  say('me', q.replace(/[<>]/g, ''));
  inp.value = '';
  setTimeout(function () { say('bot', answer(q)); }, 90);
};

const CHIPS = ['en iyi 5', 'en ucuz 5', 'kirilim kurulumu en iyi', 'buyumesi en yuksek',
               'teknoloji hisseleri', '40 kurali nedir', 'tarama ozeti', 'yardim'];
document.getElementById('chatChips').innerHTML = CHIPS.map(function (c) {
  return '<button class="chip">' + c + '</button>';
}).join('');
document.querySelectorAll('.chip').forEach(function (b) {
  b.onclick = function () {
    document.getElementById('chatIn').value = b.textContent;
    document.getElementById('chatForm').dispatchEvent(new Event('submit'));
  };
});

say('bot', 'Merhaba. Bu listedeki <b>' + DATA.rows.length + '</b> hisse ve <b>' +
  (DATA.diagnostics.active_factors || []).length + '</b> parametre hakkinda soru sorabilirsin.<br>' +
  'Ornek: <b>' + (DATA.rows[0] ? DATA.rows[0].ticker : 'LQDT') + ' nasil?</b> ya da <b>en ucuz 5</b>.<br><br>' +
  '<span style="color:var(--ink-3)">Cevrimdisi calisir, ucret gerektirmez ve yalnizca ' +
  'bu sayfadaki hesaplanmis veriden cevap verir.</span>');

renderBasket();
"""
