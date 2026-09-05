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

## İlk gerçek ölçüm (05.09.2026)

2641 hisse, ~2 yıllık günlük bar, kazanç tanımı "endeksten iyi".

**Taban oranları: 3g %48 · 5g %48 · 10g %47.**

Bu tek başına dikkate değer: rastgele bir küçük/orta ölçekli hisse günü, SPY'ı
**yarıdan az** oranda geçiyor. Medyan hisse endeksin altında kalıyor — endeks
birkaç dev hisseyle taşınıyor. Yani "kazanma oranı %50" bile taban oranın
üstünde bir sonuçtur.

### 330 kova, 328'i ölçüldü, **hiçbiri tabanı aşmıyor**

Tek bir kovanın bile alt güven sınırı taban oranın üstünde değil. Ölçülen
hiçbir kurulum, "rastgele bir gün" olmaktan ayırt edilemiyor.

### Ama işaretlerde bir desen var

| Kurulum | 3g | 5g | 10g | ort |
|---|---:|---:|---:|---:|
| rsi2_asiri_satim | +0.033 | +0.027 | +0.029 | **+0.030** |
| uc_gun_geri_cekilme | +0.027 | +0.024 | +0.020 | **+0.024** |
| boga_yutan | −0.001 | +0.014 | +0.015 | +0.010 |
| dagitim_gunu | +0.008 | +0.005 | +0.010 | +0.007 |
| yutan_ayi | −0.009 | −0.000 | −0.001 | −0.003 |
| bollinger_sikismasi | −0.007 | −0.006 | +0.001 | −0.004 |
| hacimli_kirilim | −0.003 | −0.008 | −0.008 | −0.006 |
| nr7_ic_bar | −0.011 | −0.009 | −0.002 | −0.007 |
| ma20_geri_cekilme | −0.012 | −0.007 | −0.003 | −0.007 |
| bosluk_dolumu | −0.023 | −0.018 | −0.011 | −0.017 |
| cekic | −0.028 | −0.017 | −0.012 | −0.019 |
| hacim_kurumasi | −0.023 | −0.026 | −0.018 | **−0.022** |

Üstteki iki sıra **ortalamaya dönüş** kurulumu: aşırı satım ve geri çekilme.
Alttaki sıralar **kırılım ve trend** kurulumları. Ayrım temiz.

Ve bu, uzun vade tarafındaki bulguyla **aynı yöne bakıyor**: orada da trend
ailesinin tamamı ölçüm penceresinin ikinci yarısında gücünü kaybediyordu
(bkz. [OGRENME.md](OGRENME.md), zaman/rejim kırılımı). İki bağımsız ölçüm,
farklı yöntemlerle, aynı şeyi söylüyor.

**Ama bu bir sonuç değil, bir hipotez.** Üç sebeple:

1. Hiçbiri anlamlılık eşiğini geçmiyor. En büyük edge +0.033, aralığı
   0.43–0.59 ve taban 0.48 aralığın içinde.
2. Üç ufkun işaretleri **bağımsız değil** — örtüşen pencerelerden geliyorlar.
   "12 kurulumun 10'unda üç ufuk da aynı yönde" cümlesi kulağa güçlü geliyor
   ama o 10, bağımsız 10 gözlem değil.
3. Tek bir 2 yıllık pencere, ve o pencerede uzun bir düşüş rejimi yok.

### En kötü kovalar da tutarlı

```
hacimli_kirilim  10g  oynaklik=sakin   edge -0.062
cekic             5g  oynaklik=sakin   edge -0.047
cekic             3g  oynaklik=sakin   edge -0.047
hacimli_kirilim   5g  oynaklik=sakin   edge -0.044
```

Sakin piyasada kırılım. Oynaklık genişlemesi olmadan olan kırılım, kırılım
sayılmıyor — kalıp gerçekleşiyor ama arkasında hareket yok. Yine anlamlı değil,
yine yönü mantıklı.

### Pratikte ne anlama geliyor

Panoda **hiçbir satır yıldız almıyor.** Sistem her kurulumu buluyor, her biri
için sayımı gösteriyor ve hiçbiri için "bu ayırt edilebilir" demiyor. Tasarım
böyle: eşiği geçen bir şey çıkana kadar hepsi eşit derecede kanıtsız.

İşlem maliyeti de düşülmedi. Ölçülen en büyük kenar %3.3; ince likiditede
alış-satış farkı tek başına bunu yiyebilir.

---

## Hangi zaman dilimi? — karar ve gerekçesi

Soru "1 dakika mı, 5 mi, 15 mi, 1 saat mi" gibi görünüyor ama aslında tek bir
şeye bağlı: **kaç farklı gün veri var.**

Bar sayısı yanıltır. 1 dakikalık veride bir hissede 2.730 bar olur — ama hepsi
7 günden gelir. Kalibrasyonun etkin örneklem formülü zaten bunu söylüyor: aynı
günün barları tek bir piyasa günüdür. Üstüne ileri getiri örtüşmesi de binince
7 günden ölçülecek bir şey kalmaz.

| Aralık | Sağlayıcı sınırı | Farklı gün | Ölçülebilir mi |
|---|---|---:|---|
| 1 dk | 7 gün | ~7 | **hayır** — etkin örneklem 1-2 |
| 5 dk | 60 gün | ~60 | sınırda |
| 15 dk | 60 gün | ~60 | sınırda |
| 30 dk | 60 gün | ~60 | sınırda |
| **1 saat** | **730 gün** | **~500** | **evet** |
| 1 gün | sınırsız | ~500 | evet (zaten var) |

**Birincil aralık: 1 saat.** Üç sebeple:

1. **Tek çok yıllık gün içi çözünürlük.** 500 farklı gün, ~3.500 bar. 5 ve 15
   dakika 60 günle sınırlı; o pencerede tek bir rejim var ve ölçüm o rejime
   ait çıkar.
2. **Derin öğrenme için kesit sayısı.** Günlükte ~500 zaman noktası var,
   saatlikte ~3.500. Aynı takvim aralığından **7 kat fazla çapraz kesit.** Ve
   bu, `AttnRanker`'ın zaten kurulu olduğu şekil: bir zaman noktasındaki
   kesiti küme olarak işlemek.
3. **Maliyet/kenar oranı.** 1-5 dakikada tipik hareket, makas + komisyonun
   yanında küçük kalır. Saatlik ufuklarda (2-10 bar ≈ 2 saat - 1.5 gün)
   hareket büyüklüğü, gerçek bir kenarın maliyeti aşabileceği mertebede.

**Kontrol grubu: günlük.** Zaten elimizde ve aynı dedektörler orada ölçüldü.
Saatlikte çıkan bir sonuç günlükte de bir iz bırakıyor mu — bu, tek başına
saatlik sonuçtan daha güvenilir bir kanıt.

**15 dakika şimdilik dışarıda ama kapı açık.** 60 günlük pencere kayan bir
pencere: bugünden itibaren **arşivlemeye başlarsak** altı ay sonra ~120 günlük
gerçek bir 15 dakikalık geçmişimiz olur. Sistem zaten bu disiplinle çalışıyor
(günlük görüntü biriktir, sonra eğit); aynısı 15 dakikaya uygulanabilir.

### Sınır varsayılmıyor, ölçülüyor

Yukarıdaki "730 gün" belgelenmiş bir sayı ama **doğrulanmadı** — ölçmeye
çalıştığım sırada hız sınırına takıldım. Bu yüzden koda gömülmedi.

`src/intraday.py` her başarılı çekimde gerçekte ne geldiğini
`data/intraday_kapsam.json`'a yazıyor: kaç bar, kaç farklı gün, hangi tarih
aralığı. Son on ölçüm saklanıyor, yani sınır zamanla değişirse görülüyor. Aralık
seçimi bu ölçülen sayılara bakıyor, benim yazdığım tabloya değil.

```
python run.py intraday kapsam
```

Bir de şu bulundu: `Ticker.history()` bu yfinance sürümünde her çağrıda
`TypeError` fırlatıyor. `yf.download` aynı veriyi döndürüyor ve çalışıyor. Gün
içi yolu bu yüzden ayrı bir modülde.

---

## Test havuzu — "benzer durumdaki şirketler"

2.641 karışık hissede kısa vadeli kenar ölçmek, kurulumu değil **şirket farkını**
ölçmektir. Günde 2 milyon dolarlık bir biyoteknoloji ile 50 milyon dolarlık bir
bankayı aynı tabloya koyup "bu kurulum %52 tutturuyor" demek, iki farklı dünyanın
ortalamasını almaktır; o ortalama ikisini de tarif etmez.

**Sektör bir ağırlık değil, şart.** Sektör karışırsa ölçülen şey sektör
rotasyonu olur ve her şeyi bastırır.

Kalan dört eksen ağırlıklı uzaklıkla eşleşiyor:

| Eksen | Ağırlık | Neden |
|---|---:|---|
| log piyasa değeri | 0.30 | Sektörden sonra çapraz kesitte en güçlü belirleyici |
| log dolar hacim | 0.30 | Ölçülen kenarın işlem edilebilir olup olmadığını bu belirler |
| oynaklık (ATR%) | 0.25 | Getirilerin büyüklüğü karşılaştırılabilir olsun |
| log fiyat | 0.15 | Düşük fiyatta tik boyutu ve makas kenardan büyük olabilir |

Likiditeye büyüklükle **eşit** ağırlık vermek bilinçli: kısa vade tarafının tüm
iddiası "ölçülen kenar maliyeti aşıyor mu". Günlük 500 bin dolarlık hisseyle 50
milyon dolarlığı aynı havuza koyarsan o soruyu soramazsın bile.

### Neyin dışarıda bırakıldığı daha önemli

Havuz **yapısal** niteliklerle kuruluyor. Trend, momentum, "MA200 üstünde
olanlar" gibi **durum** nitelikleri bilerek kullanılmıyor.

Sebep: bunlar zamanla değişir ve bugünkü değerleri, ölçeceğimiz dönemin
sonucuyla kısmen aynı şeyden beslenir. "Yükselen trendde olan hisseler" kümesi,
o trendin devam ettiği dönemde seçilmiş olur — sessiz bir ileriye bakıştır.

Trend bir **seçim ölçütü** değil, bir **koşul etiketidir**. Kalibrasyon zaten
her sinyali kendi ortamıyla kaydediyor.

### Ölçülen sonuç

6 havuz × 25 hisse = 150 sembol. Homojenlik iddia edilmiyor, ölçülüyor — her
eksende havuzun çeyrekler arası genişliği evrenin kaçta kaçı:

| Sektör | mcap | hacim | ATR% | fiyat | medyan mcap | medyan ATR% |
|---|---:|---:|---:|---:|---:|---:|
| Financial Services | 0.15x | 0.19x | **0.09x** | 0.18x | ~930M$ | %2.50 |
| Healthcare | 0.24x | 0.19x | 0.27x | 0.15x | ~1.7Mr$ | %4.59 |
| Technology | 0.28x | 0.25x | 0.30x | 0.47x | ~5.4Mr$ | %4.27 |
| Industrials | 0.26x | 0.31x | 0.13x | 0.22x | ~5.5Mr$ | %3.48 |
| Consumer Cyclical | 0.21x | 0.28x | 0.14x | 0.31x | ~4.4Mr$ | %3.79 |
| Real Estate | 0.32x | 0.22x | 0.10x | 0.36x | ~3.5Mr$ | %2.04 |

Hepsi evrenin yarısından dar, çoğu üçte birinden. Eşleme çalışmış.

Havuzun pratik bir işi daha var: **2.755 hisse için saatlik veri çekmek hız
sınırına çarpar, 150 hisse için çarpmaz.** Havuz olmadan saatlik ölçüm zaten
mümkün değil.

```
python run.py havuz
```

---

## Saatlik ölçüm (05.09.2026)

Havuzun 147 sembolü, saatlik bar. Ölçülen kapsam: medyan **5.082 bar / 730
farklı gün**, 2023-10-09 → 2026-09-04. Yani ~3 yıl.

Bu, zaman dilimi kararının doğrulanması: 15 dakikada 60 gün, 1 dakikada 7 gün
alabiliyorken saatlikte 730 gün var.

### Ölçümün keskinliği gerçekten arttı

| | günlük | saatlik |
|---|---:|---:|
| etkin örneklem (tipik) | 40–160 | **490–500** |
| aralık genişliği | ±0.12 | **±0.04** |

Beklenen kazanç bu: daha çok bağımsız gün → daha dar aralık → gerçek bir kenar
varsa görünme şansı yüksek.

### Ve hiçbir şey görünmüyor

| Kurulum | Yön | 3b | 7b | 21b | ort |
|---|---|---:|---:|---:|---:|
| rsi2_asiri_satim | long | +0.011 | −0.001 | −0.011 | −0.000 |
| uc_gun_geri_cekilme | long | +0.010 | −0.000 | −0.013 | −0.001 |
| yutan_ayi | short | −0.013 | −0.001 | +0.006 | −0.003 |
| nr7_ic_bar | long | −0.013 | −0.014 | −0.037 | −0.021 |
| bollinger_sikismasi | long | −0.016 | −0.026 | −0.023 | −0.022 |
| hacim_kurumasi | long | −0.011 | −0.021 | −0.036 | −0.023 |
| ma20_geri_cekilme | long | −0.017 | −0.020 | −0.032 | −0.023 |
| boga_yutan | long | −0.023 | −0.025 | −0.022 | −0.023 |
| cekic | long | −0.028 | −0.030 | −0.029 | −0.029 |
| hacimli_kirilim | long | −0.031 | −0.029 | −0.033 | −0.031 |
| dagitim_gunu | short | −0.038 | −0.021 | −0.045 | −0.034 |

279 kovadan **1 tanesi** tabanı aşıyor. Çoklu test hesabı: %95 aralıkla 279
kova test edilince hiç kenar olmasa bile şansa ~7 kova geçer. **1, şans
beklentisinin altında** — yani geçen o tek kova da kanıt sayılmaz. Bu satır
artık çıktının kendisinde.

Günlükteki "ortalamaya dönüş pozitif / kırılım negatif" ayrımı saatlikte
**silinmiş**: `rsi2` ve `uc_gun` sıfıra oturuyor. Ölçüm keskinleşince desen
kayboldu — bu, desenin gürültü olduğuna dair günlük tablodan daha güçlü bir
işaret.

### Bir hipotez: kısa vadede geri verme

Neredeyse hepsi negatif, ve **en negatif olanlar bir hareketin ARDINDAN
tetiklenenler**: hacimli kırılım −0.031, çekiç −0.029, dağıtım günü −0.034.
Saatlik ölçekte kısa vadeli geri dönüşle tutarlı: yeni hareket eden biraz geri
veriyor. Hepsi aralıkların içinde, yani kanıt değil — ama not edilmeye değer.

### Yön hatası — düzeltilmeden önce ve sonra

İlk saatlik koşuda `dagitim_gunu` **+0.034** vermişti ve tabanı aşan iki
kovadan biriydi. Sebep: bütün kurulumlar "endeksi geçti mi" diye sayılıyordu,
çıkış sinyalleri dahil. Bir dağıtım günü sonrası hissenin endeksi geçmesi,
kurulumun **çalıştığını değil çalışmadığını** gösterir.

Düzeltme sonrası aynı kurulum **−0.034**, ve listedeki en kötü satır. İşaret
tam tersine döndü.

Artık kazanç yöne göre tanımlı: uzun tarafta "endeksi geçti", kısa tarafta
"endeksin altında kaldı", taban oranı da tümleyen. Test: aynı dedektör iki
yönde kayıtlı → +0.323 ve −0.323.

Bu hatanın önemi büyüklüğünden değil türünden: sistem bir çıkış sinyalinin
başarısızlığını başarı olarak raporluyordu, ve sayı makul göründüğü için
kendiliğinden fark edilmezdi.

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
