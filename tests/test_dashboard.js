/**
 * Pano arayuz testleri — GERCEK DOM uzerinde (jsdom).
 *
 * Neden saplama (stub) degil de gercek DOM:
 * Onceki saplama testinde `querySelectorAll` daima bos dizi donuyordu. Bu
 * yuzden `syncAddButtons()` icindeki dongu hic calismiyor, dolayisiyla
 * `basket` degiskenine bildiriminden once erisildigi (temporal dead zone)
 * ReferenceError ASLA tetiklenmiyordu. Tarayicida ise script komple
 * duruyor, sohbet formu baglanmiyor ve sayfa basa atiyordu.
 *
 * Gercek DOM bu sinif hatalari yakalar.
 *
 * Calistir:  node tests/test_dashboard.js [pano.html]
 */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const file = process.argv[2] ||
  path.join(__dirname, '..', 'output', 'dashboard.html');

if (!fs.existsSync(file)) {
  console.error('Pano bulunamadi: ' + file + '\nOnce: python run.py');
  process.exit(1);
}

let fails = 0;
const ok = (name, cond, extra) => {
  if (cond) console.log('  OK    ' + name + (extra ? '  ' + extra : ''));
  else { console.log('  HATA  ' + name + (extra ? '  ' + extra : '')); fails++; }
};

const errors = [];
const dom = new JSDOM(fs.readFileSync(file, 'utf8'), {
  runScripts: 'dangerously',
  pretendToBeVisual: true,
  // Gercek bir origin sart: jsdom, about:blank gibi opak kaynaklarda
  // localStorage erisiminde DOMException firlatiyor.
  url: 'http://localhost/dashboard.html',
});
dom.virtualConsole.on('jsdomError', e => errors.push(e.message));
dom.window.addEventListener('error', e => errors.push(String(e.error || e.message)));

const { window } = dom;
const doc = window.document;
const $ = s => doc.querySelector(s);
const $$ = s => Array.from(doc.querySelectorAll(s));

console.log('\n=== PANO TESTLERI (gercek DOM) ===');

// 1. Script hatasiz yuklendi mi — en kritik kontrol
ok('script hatasiz yuklendi', errors.length === 0,
   errors.length ? errors[0].split('\n')[0].slice(0, 110) : '');

// 2. Ana bolumler doldu mu
ok('siralama grafigi cizildi', $$('#bars .row').length > 0,
   $$('#bars .row').length + ' satir');
ok('agirlik listesi doldu', $$('#weights .wrow').length > 0,
   $$('#weights .wrow').length + ' parametre');
ok('tablo doldu', $$('#tbl tbody tr').length > 0,
   $$('#tbl tbody tr').length + ' satir');
ok('ozet kutucuklari doldu', $$('#tiles .tile').length > 0);

// 3. EKLE butonlari — hem grafikte hem tabloda
const barBtns = $$('#bars .addbtn');
const tblBtns = $$('#tbl .addbtn');
ok('grafikte EKLE butonu var', barBtns.length > 0, barBtns.length + ' adet');
ok('tabloda EKLE butonu var', tblBtns.length > 0, tblBtns.length + ' adet');
ok('EKLE butonu satir sayisiyla ayni', barBtns.length === $$('#bars .row').length);

// 4. EKLE butonu gercekten calisiyor mu
if (barBtns.length) {
  const tk = barBtns[0].dataset.add;
  barBtns[0].dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  const stored = JSON.parse(window.localStorage.getItem('invest_basket_v1') || '{}');
  ok('EKLE tiklamasi sepete ekliyor', !!stored[tk], tk);
  ok('EKLE butonu isaretleniyor', barBtns[0].classList.contains('on'));
  ok('sepet paneli sayaci guncelleniyor', $('#fabN').textContent === '1');

  const cmd = $('#basketCmd').textContent || '';
  ok('komut uretiliyor', cmd.indexOf('watch add ' + tk) >= 0, cmd.slice(0, 60));

  // ikinci tik -> cikarma
  barBtns[0].dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  const after = JSON.parse(window.localStorage.getItem('invest_basket_v1') || '{}');
  ok('ikinci tik sepetten cikariyor', !after[tk]);
}

// 5. EKLE'ye basmak satir detayini acmamali
if (tblBtns.length) {
  const before = $$('#tbl tbody tr.det').length;
  tblBtns[0].dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  ok('EKLE satir detayini acmiyor', $$('#tbl tbody tr.det').length === before);
  tblBtns[0].dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
}

// 6. Satira tiklayinca detay aciliyor mu
const row = $('#tbl tbody tr');
if (row) {
  row.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  const det = $('#tbl tbody tr.det');
  ok('satir tiklamasi detay aciyor', !!det);
  ok('detayda parametreler var', det && det.querySelectorAll('.fitem').length > 5,
     det ? det.querySelectorAll('.fitem').length + ' parametre' : '');
}

// 7. SOHBET — en kritik kismi: cevap geliyor mu, sayfa submit ediyor mu
const form = $('#chatForm');
const input = $('#chatIn');
ok('sohbet formu var', !!form && !!input);

if (form && input) {
  const before = $$('#chatLog .msg').length;
  ok('acilis mesaji goruntulendi', before > 0, before + ' mesaj');

  // Formun native submit yapip yapmadigini yakala
  let defaultPrevented = null;
  const ev = new window.Event('submit', { bubbles: true, cancelable: true });
  input.value = 'en iyi 5';
  form.dispatchEvent(ev);
  defaultPrevented = ev.defaultPrevented;

  ok('form native submit yapmiyor (sayfa basa atmaz)', defaultPrevented === true);

  // Cevap asenkron (setTimeout 90ms)
  setTimeout(() => {
    const msgs = $$('#chatLog .msg');
    ok('soru gunlukte gorunuyor', msgs.length >= before + 1);
    const bots = $$('#chatLog .msg.bot');
    const last = bots[bots.length - 1];
    ok('asistan cevap verdi', msgs.length >= before + 2,
       last ? last.textContent.replace(/\s+/g, ' ').slice(0, 55) : 'cevap yok');
    ok('cevap tablo iceriyor', last && last.innerHTML.indexOf('<table') >= 0);
    ok('cevapta undefined/NaN yok', last && !/undefined|NaN/.test(last.innerHTML));
    ok('girdi alani temizlendi', input.value === '');

    // Hazir soru butonlari
    const chip = $('.chip');
    if (chip) {
      const n = $$('#chatLog .msg').length;
      chip.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
      setTimeout(() => {
        ok('hazir soru butonu calisiyor', $$('#chatLog .msg').length > n);
        finish();
      }, 220);
    } else { finish(); }
  }, 260);
} else {
  finish();
}

function finish() {
  // 8. Tema ve icerik saglik kontrolleri
  ok('charset tanimli', !!doc.querySelector('meta[charset]'));
  ok('sayfada "AI" ibaresi yok', !/\bAI\b/.test(doc.body.textContent));
  ok('yatirim tavsiyesi uyarisi var',
     doc.body.textContent.indexOf('yatirim tavsiyesi degildir') >= 0);

  console.log(fails ? '\n' + fails + ' TEST BASARISIZ\n' : '\nTUM PANO TESTLERI GECTI\n');
  process.exit(fails ? 1 : 0);
}
