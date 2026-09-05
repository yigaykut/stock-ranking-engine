# Kısa Vade — Kurulumlar ve Ölçülmüş Güven

## Neden ayrı bir sistem

Sistemin geri kalanı tek bir soru soruyor: **"bugün hangi hisse diğerlerinden
daha iyi duruyor?"** Çapraz kesitsel, 21 günlük ufuklu, 28 parametreli bir
sıralama.

Kısa vade bambaşka bir soru soruyor: **"bu hissede şu anda, önümüzdeki 3-10
günde işe yarayabilecek bir kurulum var mı?"**

İkisi karıştırılmamalı ve bu yüzden ayrı bir bölümde duruyor. Uzun vadeli skor
bir **sıralamadır** — her hissenin bir yeri vardır. Kısa vadeli çıktı bir **olay
tespitidir** — ya vardır ya yoktur. Aynı tabloya konsalar ikisi de anlamını
kaybederdi: bir hisse sıralamada 400. olup bugün tertemiz bir kurulum
gösterebilir, ve bu bir çelişki değildir.

```
python run.py kisa           # bugünkü kurulumlar
python run.py kisa kalibre   # geçmişten güven değerlerini ölç
python run.py kisa panel     # model eğitim kümesini dışa aktar
```

Günlük tarama `kisa tara`'yı kendisi çalıştırır (pano yazılmadan **önce**).
Kalibrasyon elle çalıştırılır — bütün önbelleği geziyor, her sabah koşmasının
anlamı yok.

---

## Kurulumlar

On iki tane. Onu giriş tarafı, ikisi izleme listesinin çıkış tarafı için.

| Kimlik | Ne arıyor | Ufuk |
|---|---|---:|
| `boga_yutan` | Dün düşüş, bugün onu tamamen kapsayan yükseliş mumu; kısa bir geri çekilmenin ardından | 5g |
| `cekic` | Uzun alt fitil, küçük gövde, günün üst yarısında kapanış; 10 günün dibine yakın | 5g |
| `nr7_ic_bar` | Son 7 günün en dar menzili ve önceki barın içinde | 10g |
| `hacimli_kirilim` | 20 günün zirvesini medyan hacmin 1.5 katıyla kırmak | 10g |
| `ma20_geri_cekilme` | Yükselen MA20'ye değip üstünde kapanmak | 5g |
| `bollinger_sikismasi` | Bant genişliği son 120 günün en dar %10'unda | 10g |
| `rsi2_asiri_satim` | RSI(2) < 10, fiyat MA200 üstünde | 3g |
| `bosluk_dolumu` | %3+ aşağı boşlukla açılıp boşluğu kapatarak dönmek | 3g |
| `uc_gun_geri_cekilme` | Yükselen trendde üst üste üç düşen kapanış | 5g |
| `hacim_kurumasi` | Yükselen MA50 desteğinde hacmin medyanın %60'ının altına inmesi | 10g |
| `yutan_ayi` | Zirve civarında dünün yükseliş mumunu yutan düşüş mumu | 5g |
| `dagitim_gunu` | Zirve civarında yüksek hacimli düşüş günü | 5g |

Hiçbiri yeni değil. Hepsi mum formasyonu, oynaklık sıkışması ve Connors tipi
ortalamaya dönüş literatüründen; nesnel olarak tanımlanabilir ve günlük barla
hesaplanabilir olmaları dışında ortak özellikleri yok. **Yeni olan, her birinin
bu evrende ölçülüyor olması.**

Her dedektör iki şey döndürür:

- **var** — o barda kurulum oluştu mu
- **güç** — [0, 1] arasında, kurulum ne kadar *temiz* oluştu

**Güç bir olasılık değildir.** Yüksek güç "daha çok kazandırır" demek değil,
"kalıba daha çok benziyor" demektir. Olasılık kalibrasyondan gelir.

---

## Güven — iddia değil, sayım

"Bu kurulum işe yarar mı" sorusunun cevabı dedektörde **yoktur**. Cevap bir
sayımdır: geçmişte bu kurulum kaç kez oluştu, kaçında önümüzdeki N günde endeks
geçildi.

Ham oranı olduğu gibi yazmak üç ayrı yerde yanıltır.

### 1. Taban oranı

Yükselen bir piyasada rastgele bir hisse günü bile endeksi yarıya yakın
oranda geçer. **%53 tutturan bir kurulum iyi değildir.** Her kova kendi ufkunun
taban oranıyla birlikte raporlanır ve asıl sayı:

```
edge = p - taban
```

Panoda ikisi yan yana durur; tek başına "güven" göstermek yanıltıcı olurdu.

### 2. Küçük örneklem

7 gözlemde 5 kazanç %71 eder ve hiçbir şey anlatmaz. İki katmanlı çözüm:

**Wilson aralığı.** Normal yaklaşım (`p ± z·√(p(1-p)/n)`) küçük n'de ve p uçlara
yakınken [0,1] dışına taşar; 0/10 için **sıfır genişlikte** bir aralık verir ki
bu açıkça yanlıştır. Wilson ikisinde de doğru davranır:

```
wilson(0, 10)  -> [0.000, 0.278]      sifir genislikte DEGIL
wilson(10, 10) -> [0.722, 1.000]      alt sinir 1 DEGIL
```

**Büzme (empirical Bayes).** Kova, taban orana 30 sanal gözlem kadar çekilir:

```
buzulmus(5, 7, taban=0.52)      = 0.56    ham %71 -> %56
buzulmus(700, 1000, taban=0.52) = 0.69    ham %70 -> %69
```

n büyüdükçe ham orana yakınsar. Yani veri biriktikçe sistem kendiliğinden daha
iddialı olur, ama önce değil.

### 3. Gözlemler bağımsız değil — en önemlisi

İki ayrı kırılım var:

- Aynı gün 50 hissede aynı kurulum oluşur. Hepsi **aynı piyasa gününü** yaşar.
  Bu 50 gözlem değil, kabaca 1 gözlemdir.
- Etiket N günlük ileri getiri, sinyaller günlük. Ardışık N günün sonuçları
  **aynı geleceği paylaşır**.

İkisi birlikte etkin örneklem büyüklüğünü düşürür:

```
n_etkin ≈ farklı gün sayısı / ufuk
```

Aralıklar bu sayıyla hesaplanır. Sonuç her zaman daha **geniş** bir aralıktır —
düzeltme sistemi daha ihtiyatlı yapar, daha iddialı değil.

Ne kadar fark ettiği ölçüldü. Kenar **ekilmemiş** sentetik seride:

```
n=298   n_etkin=30.2   p=0.7359   taban=0.6129   aralik=[0.5907, 0.8821]
-> alt sinir tabani asmiyor: "gurultuden ayirt edilemiyor"
```

298 ham sinyal, 30 etkin. Düzeltme olmasaydı %73'lük bir güven gösterip
**olmayan bir kenar uyduracaktı.** Aynı test, kenar ekilmiş seride onu buluyor
(`edge +0.30`, alt sınır 0.887 > taban 0.662). Yani modül hem gerçek kenarı
görüyor hem de olmayanı uydurmuyor — ikisi ayrı test.

---

## "Hangi durumlarda daha iyi çalışıyor"

Her sinyal, oluştuğu ortamla birlikte kaydedilir:

| Eksen | Kovalar |
|---|---|
| oynaklık | sakin (ATR% < 2), orta, oynak (> 4.5) |
| likidite | ince (< 2M$), orta, kalın (> 20M$) |
| trend konumu | ma200_altı, yakın (< %15), uzak |

Kalibrasyon her ekseni **ayrı ayrı** kırar. Çapraz çarpım (3×3×3 = 27 kova)
bilerek yapılmıyor: her kova boş kalır ve **ölçülemeyen kırılımın değeri
yoktur.**

Güven sorgusu en özel kovadan genele iner: sinyalin kendi koşuluna ait kova
ölçüm eşiğini geçiyorsa o kullanılır, geçmiyorsa kurulumun geneline düşülür.
Hiçbiri yoksa cevap **"bilinmiyor"**dur. Boşluğu doldurmak için sayı
uydurulmaz.

---

## İleriye bakış — iki katmanlı kilit

Bir dedektör yanlışlıkla geleceğe bakarsa geçmişte mükemmel çalışır, canlıda
hiçbir şey üretmez, ve bunu fark etmek çok zordur: kod çalışır, sayılar makul
görünür, kalibrasyon parlak sonuçlar verir.

**Davranış testi.** Seri t tarihinde kesilip yeniden hesaplanınca t'deki sinyal
aynı çıkmalı. Ayrıca t'den *sonraki* barları 3 katına çıkarmak t'deki sinyali
değiştirmemeli.

**Kaynak testi.** Dosyada negatif `shift`, `center=True` veya `bfill`
geçmemeli. Yorumlar ve metin sabitleri `tokenize` ile ayıklanarak aranır —
düz metin araması modülün kendi açıklamasındaki "`center=True` YOKTUR" cümlesine
takılıyordu.

Etiketleme tarafında `shift(-ufuk)` **vardır ve olmalıdır** — o ileriye bakış
değil, ölçülen sonucun ta kendisidir. Kritik olan dedektörlerin onu görmemesi.

---

## Ölü dedektör vakası (05.09.2026)

`bollinger_sikismasi` şunu yapıyordu:

```python
ust, orta, alt = ind.bollinger(c, 20, 2.0)   # bollinger BES deger dondurur
```

Her çağrıda `ValueError` fırlıyordu, `tespit()` istisnayı yutuyordu, dedektör
boş seri döndürüyordu. Sonuç: **hiç tetiklenmemiş bir kurulum, hiç tetiklenmeyen
bir kurulumdan ayırt edilemiyordu.** İlk kalibrasyon koşusu 12 değil 11 kurulumu
ölçtü ve bunu söyleyen hiçbir şey yoktu.

İki ders, ikisi de koda geçti:

1. **İstisna yutulmuyor.** Hatalar `kisa_vade.HATALAR`'da birikiyor. Tarama tek
   bozuk hisse yüzünden durmuyor ama bozuk dedektör görünür oluyor. Test: uzun
   bir seride hiçbir dedektör hata vermemeli, ve hiçbiri **tamamen ölü**
   olmamalı.

2. **Fikstür fazla temizdi.** Yeni üreteç hacim patlaması/kuraklığı ve boşluk
   içeriyor, ve açılışı **önceki** kapanıştan türetiyor. Bugünün kapanışından
   türetince boşluksuz günlerde `Open == Close` oluyordu, hiçbir bar "yükseliş
   mumu" sayılmıyordu ve iki yutan-mum dedektörü yapısal olarak hiç
   tetiklenemiyordu.

---

## Meta-etiket paneli — sıradaki model için

```
python run.py kisa panel   ->   data/kisa_vade_panel.csv
```

Her kurulum oluşumu için bir satır:

- **kimlik**: ticker, tarih, kurulum, yön
- **özellikler** (sinyal gününe kadar): güç, ATR%, bant genişliği, dolar hacim,
  MA200/MA50 uzaklığı, RSI(14), hacim oranı, 5 ve 20 günlük getiri, gövde oranı,
  ve kova adları
- **etiketler** (sinyal gününden sonra): 3/5/10 gün fazla getiri ve kazanç 0/1

Neden kalibrasyondan ayrı bir çıktı: kova bazlı kalibrasyon **insanın okuyup
karar vereceği özet**, ve kovalar bilerek kaba. Bir model kovaya ihtiyaç duymaz,
ham sayılardan kendi eşiklerini öğrenir. İkisi aynı dedektörlerden, aynı
özelliklerden, aynı etiketten geliyor — yani model, kalibrasyonun ölçtüğünden
**başka bir dünyayı** öğrenmiyor.

Literatürdeki adı **meta-etiketleme**: birincil sinyal kalıptan gelir, ikincil
model o sinyalin tutup tutmayacağını kestirir. Yani model "hangi hisse
yükselecek" değil, **"bu kurulum bu ortamda tutar mı"** sorusunu öğrenir — çok
daha dar ve çok daha öğrenilebilir bir soru.

Model henüz **yok**. Bu dosya onun eğitim kümesi, ve kalibrasyon o gelene kadar
zaten çalışan bir cevap veriyor.

---

## Sınırlar — dürüst liste

- **Kalibrasyon geçmişi önbellekle sınırlı**: 2 yıllık günlük bar. Uzun bir
  düşüş rejimi bu pencerede yok, dolayısıyla hiçbir güven değeri "piyasa
  düşerken ne olur" sorusuna cevap vermiyor.
- **Hayatta kalma yanlılığı**: önbellekte bugün kote olan hisseler var. Kote
  dışı kalmış hisselerin kurulumları sayılmıyor.
- **İşlem maliyeti yok**: fazla getiri brüt. Alış-satış farkı, komisyon ve
  kayma düşülmemiş. İnce likiditede bu fark, ölçülen kenardan büyük olabilir.
- **Kapanış fiyatından giriş varsayılıyor**: sinyal kapanışta oluşuyor, giriş
  de kapanışta sayılıyor. Gerçekte ertesi açılışa kalır.
- **Hiçbir şey otomatik yapılmıyor.** Sistem sinyal üretir ve sayımı gösterir.
  Karar kullanıcınındır ve **bu bir yatırım tavsiyesi değildir.**
