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
| dagitim_gunu (kısa) | −0.008 | −0.005 | −0.010 | −0.008 |
| yutan_ayi (kısa) | +0.009 | −0.000 | +0.000 | +0.003 |
| bollinger_sikismasi | −0.007 | −0.005 | +0.002 | −0.004 |
| hacimli_kirilim | −0.003 | −0.008 | −0.008 | −0.006 |
| nr7_ic_bar | −0.011 | −0.009 | −0.002 | −0.007 |
| ma20_geri_cekilme | −0.012 | −0.007 | −0.003 | −0.007 |
| bosluk_dolumu | −0.023 | −0.018 | −0.011 | −0.017 |
| cekic | −0.028 | −0.017 | −0.012 | −0.019 |
| hacim_kurumasi | −0.023 | −0.026 | −0.018 | **−0.022** |

Üstteki iki sıra **ortalamaya dönüş** kurulumu: aşırı satım ve geri çekilme.
Alttaki sıralar **kırılım ve trend** kurulumları. Ayrım temiz.

(Tablo 05.09'da yön düzeltmesinden sonra yenilendi: kısa taraf
kurulumlarında kazanç artık "endeksin **altında** kaldı" demek.
`dagitim_gunu` +0.007'den −0.008'e, `yutan_ayi` −0.003'ten +0.003'e döndü.
Uzun taraf satırları değişmedi.)

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

## Meta-model — ilk ölçüm (06.09.2026)

Saatlik panel: 336.731 satır, %99.96 etiketli, 33 özellik (10 sayısal + kova
adları + kurulum kimliği one-hot). 4 katmanlı ileri yürüyüş, arındırma +
tampon.

**Taban çizgisi 0.5 değil, kova bazlı kalibrasyonun kendisi.**

| Ufuk | Satır | Gün | Brier model | Brier kova | Brier sabit | AUC model | AUC kova | t |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3b | 270.196 | 582 | 0.25185 | **0.24950** | 0.24970 | 0.506 | 0.512 | **−8.39** |
| 7b | 270.019 | 581 | 0.25367 | **0.24922** | 0.24939 | 0.500 | 0.511 | **−7.41** |
| 21b | 270.444 | 580 | 0.25508 | **0.24805** | 0.24841 | 0.499 | 0.512 | **−8.04** |

Model kovadan iyi değil — **kovadan belirgin biçimde kötü.** t değerleri −7
ile −8 arası, yani fark gürültü değil.

### Güvenilirlik eğrisi — asıl anlatan tablo

Tahmin edilen olasılık → gerçekleşen oran, on dilim:

```
ufuk  3b:  40->47  44->48  46->48  47->47  48->48  49->48  50->48  52->49  54->48  59->49
ufuk  7b:  36->47  42->47  45->47  46->47  48->47  49->48  50->47  52->48  54->47  60->47
ufuk 21b:  32->45  39->46  42->46  44->45  46->45  47->45  49->46  51->45  54->45  62->45
```

Model tahminini **%32 ile %62 arasında** oynatıyor. Gerçekleşen oran ise
**%45–49 arasında sabit**. Yani model, olmayan bir şeyden yayılım üretiyor:
"bu sinyal %62 tutar" dediği satırlarla "%32 tutar" dediği satırlar
gerçekte aynı oranda tutuyor.

AUC 0.499–0.506. 0.50 sıralama gücünün hiç olmaması demek.

### Kova da neredeyse hiçbir şey katmıyor

Dikkat: kova Brier'i 0.24950, sabit taban oranı 0.24970. Fark **0.0002**.
Yani kova bazlı kalibrasyon da tek bir sayı söylemekten neredeyse ayırt
edilemiyor. Zaten kalibrasyonun kendisi bunu söylüyordu (282 kovadan 0'ı
tabanı aşıyor); Brier bunu ikinci bir yoldan doğruluyor.

### Modelin kovadan KÖTÜ çıkması neden beklenen

Öğrenilecek bir şey yokken model eğitim gürültüsüne uyar, ve bu örneklem
dışında Brier'e maloluyor. Dürüst bir değerlendirmenin doğru davranışı bu.
"Model biraz daha iyi" çıksaydı, önce ölçümde bir sızıntı arardım.

### Bu "derin öğrenme çalışmıyor" demek değil

Makine çalışıyor: ekilmiş kenarı bulduğunu ayrı bir testte gösteriyor
(sentetik seride edge +0.30, alt sınır 0.887 > taban 0.662). Cümlenin doğru
hali şu:

> Bu 12 kurulumda, bu 33 özellikte, bu 147 hisselik havuzda, bu 730 günlük
> pencerede öğrenilecek bir yapı yok.

Bunların her biri değiştirilebilir. Sonuç, bir sonraki denemenin nereye
bakması gerektiğini söylüyor: kurulumlar ve özellikler, veri miktarı değil —
çünkü etkin örneklem 40'tan 700'e çıkarken kenar 0.030'dan 0.002'ye düştü.

---

## Grafiğin şeklini modele vermek (06.09.2026)

Skaler özellikler sinyal anını tarif ediyor: ATR%, RSI, hacim oranı, MA
uzaklığı. "Şu an neredeyiz" diyorlar, **"buraya nasıl geldik"** demiyorlar.
Oysa grafik okumanın iddiası tam da o yol.

`src/dizi.py` her sinyale önceki **24 barı** ekliyor, yedi ölçekten bağımsız
kanalla: log getiri, gövde/fitil oranları, barın menzili ve hacmi hissenin
kendi 50 barlık medyanına göre, kapanışın son 20 bardaki konumu. Küçük bir 1-B
evrişim pencereyi okuyup çıktısını skaler özelliklere ekliyor.

Pencere sinyal barında **bitiyor**; etiket ondan sonra başlıyor. Tam penceresi
olmayan satırlar sıfırla doldurulmuyor, atılıyor.

Bunun için panele tam zaman damgası eklemek gerekti — saatlik panelde yalnızca
tarih vardı ve aynı günün iki sinyali ayırt edilemiyordu.

### İlk deneme: çok daha kötü

| Ufuk | Brier dizi | Brier kova | AUC dizi | t |
|---:|---:|---:|---:|---:|
| 3b | 0.26846 | 0.24950 | 0.507 | −17.79 |
| 7b | 0.28671 | 0.24922 | 0.498 | −21.56 |
| 21b | 0.30942 | 0.24805 | 0.502 | −23.27 |

Güvenilirlik eğrisi sebebi söylüyordu (21 bar):

```
tahmin :   6   18   28   36   43   49   55   64   75   91
gercek :  43   45   45   47   47   47   47   46   45   44
```

Satır başına 168 ek girdi verilince model eğitim penceresindeki şekilleri
ezberledi ve ezberini kesinlik olarak dışarı verdi. AUC baştan sona 0.50:
**muazzam özgüven, sıfır ayırt etme.**

### Eksik olan iki şey

**Erken durdurma yoktu.** Eğitim sabit sayıda devir koşuyordu. Artık eğitim
penceresinin son %15'i doğrulama olarak ayrılıyor — *zamana göre*, rastgele
değil. Rastgele bölmek sızıntı olurdu: aynı gün iki tarafa da düşer.

**Çıktı kalibrasyonu yoktu.** Aynı doğrulama diliminde tek parametreli bir
Platt ölçeklemesi, tahminleri verinin desteklediği aralığa çekiyor.

Sentetik doğrulama: 0.02–0.92 aralığındaki uydurma güven 0.456–0.489'a iniyor,
Brier 0.314 → 0.249 (sabit taban 0.2494). Gerçek sinyal olan veride ise
AUC 0.746 → 0.746 korunuyor. **Uydurma güveni siliyor, sinyal icat etmiyor.**

### Düzeltmelerden sonra

| Ufuk | Skaler | Dizi | Kova | Sabit | AUC-sk | AUC-dizi | AUC-kova | t-dizi |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3b | 0.24969 | 0.24968 | **0.24950** | 0.24970 | 0.501 | 0.505 | **0.512** | −1.86 |
| 7b | 0.24948 | 0.24929 | **0.24922** | 0.24939 | 0.495 | 0.503 | **0.511** | +0.09 |
| 21b | 0.24878 | 0.24891 | **0.24805** | 0.24841 | 0.491 | 0.491 | **0.512** | −1.86 |

Dizi modeli 0.309'dan 0.249'a geldi — felaket tamamen düzeldi. Ama **skalerle
aynı yere** geldi: fark ±0.0002. Grafiğin şekli hiçbir şey katmadı.

Güvenilirlik artık dürüst: tahminler %41–50, gerçekleşen %44–48. Model
"bilmiyorum" diyor, ve bu doğru cevap.

7 barlık ufukta dizi modeli t = +0.09 ile kovayla **başa baş**. Hiçbir varyant
kovayı geçmedi.

### En yüksek AUC hâlâ kovada

0.511–0.512. Kova aslında sadece "hangi kurulum + hangi koşul" demek. Yani var
olan minicik sıralama bilgisi **kurulum kimliğinden** geliyor; ne skaler ne
dizi modeli bunun üstüne bir şey koyabiliyor.

### Ölçümlerin oku hep aynı yöne bakıyor

| Ölçüm | Etkin n | Kenar / AUC |
|---|---:|---|
| Günlük kova (rsi2) | 40 | edge +0.030 |
| Saatlik kova (rsi2) | 700 | edge +0.002 |
| Skaler model | — | AUC 0.50 |
| Dizi model | — | AUC 0.50 |

Ölçüm her keskinleştiğinde ve model her zenginleştiğinde cevap sıfıra
yaklaştı. Gerçek bir kenar olsaydı tersi olurdu.

### Nereye bakılmalı — dürüst liste

1. **Etiket değişebilir.** Şu anki soru "N bar sonra endeksi geçti mi".
   Alternatif: "önce +%X mi geldi yoksa −%Y mi" (üç bariyer). Bu *farklı bir
   soru* ve bu kısıtlar içinde en somut sonraki adım.
2. **Kurulumlar fazla bilinen kurulumlar.** On ikisi de yayımlanmış, yaygın
   kalıplar. Likit ABD orta ölçeklilerinde işleyen bir kenar taşısalardı
   şaşırtıcı olurdu.
3. **Evren bilerek homojen.** Kenar daha ince ya da daha oynak isimlerde
   olabilir — ama orada maliyet ısırır.
4. **Haberler dışarıda.** Bilinçli tercih; ulaşılabilecek tavanı sınırlıyor ve
   bu ölçüm "grafik yetmiyor" ile "haber eksik" arasını ayıramaz.

---

## Göstergeleri genişletmek ve etiketi değiştirmek (06.09.2026)

İki ayrı deneme, bilerek ayrı koşuldu.

### 1. Gösterge kütüphanesinin tamamı: 33 → 111 özellik

`indicators.py`'de 24 fonksiyon vardı, dördü kullanılıyordu. Hepsi girdi
oldu, üstüne 16 mum formasyonu bayrağı ve yapısal özellikler (tepe/dip
uzaklıkları, üst üste yükselen bar sayısı, sıkışma).

| Ufuk | Brier 33 | Brier 111 | AUC 33 | AUC 111 | t 33 | t 111 |
|---:|---:|---:|---:|---:|---:|---:|
| 3b | 0.24969 | 0.24960 | 0.501 | **0.508** | −1.96 | **−0.98** |
| 7b | 0.24948 | 0.24936 | 0.495 | **0.502** | −0.75 | **−0.21** |
| 21b | 0.24878 | 0.24824 | 0.491 | **0.498** | −1.70 | **−0.82** |

Üç ölçütte de küçük ama tutarlı bir iyileşme. İlk kez ok sıfırdan uzağa
gitti. Hiçbiri anlamlılık eşiğini geçmiyor.

Dizi modeli (24 barlık pencere) 111 özellikle de skalerin gerisinde:
AUC 0.507 / 0.498 / 0.491. Dört varyantta da grafiğin şekli bir şey katmadı.

### 2. Etiketi değiştirmek — ve yakalanan yanlış keşif

Yeni etiket: sinyalden itibaren ileri yürü, `+1.5 ATR` hedefe mi önce değdi
`−1.0 ATR` stopa mı. Süre dolarsa cevap yok (NaN).

İlk sonuç çarpıcıydı:

| Ufuk | Kapsama | Brier model | Brier sabit | AUC | t |
|---:|---:|---:|---:|---:|---:|
| **3b** | **%54** | **0.22789** | 0.22978 | **0.555** | **+3.69** |
| 7b | %89 | 0.24020 | 0.24010 | 0.511 | +0.16 |
| 21b | %100 | 0.24230 | 0.24168 | 0.500 | −1.42 |

Güvenilirlik eğrisi de ilk kez gerçekten izliyordu: 28→28, 31→30, 33→33,
35→35, 40→41, 43→42. Monoton, 15 puanlık aralıkta 1-2 puan sapma.

**Ama kenar, seçilimin en güçlü olduğu yerde çıkıyordu.** 3 barda satırların
yalnızca %54'ü etiketli — o kadar kısa sürede fiyat hiçbir bariyere değmiyor.
Model, *hızlı ve kararlı hareket eden* satırlar üzerinde ölçülüyordu. Canlıda
bir işlemin hızlı çözüleceği önceden bilinemez.

### Kesin test: kapsamayı %100'e çıkar

`bariyertam` etiketi çözülmeyen satırları da dikey bariyerdeki getirinin
işaretiyle etiketliyor. Aynı satırlar, aynı özellikler, aynı katmanlar — tek
fark seçilimin kalkması.

| Ufuk | KOŞULLU AUC | KOŞULLU t | **TAM AUC** | **TAM t** |
|---:|---:|---:|---:|---:|
| 3b | 0.555 | +3.69 | **0.506** | **+0.53** |
| 7b | 0.511 | +0.16 | 0.502 | −1.17 |
| 21b | 0.500 | −1.42 | 0.501 | −1.53 |

**AUC 0.555 → 0.506. Kenar seçilimdi.**

Güvenilirlik eğrisi de çöküyor:

```
kosullu:  28->28  31->30  33->33  35->35  40->41  43->42
tam    :  45->46  46->46  46->46  46->46  48->47  49->48
```

Koşullu ölçümde model 28-43 aralığında konuşup tutturuyordu. Tam ölçümde
45-49'a sıkışıyor — yani "bilmiyorum" diyor.

### Bu testin kendisi bulgudur

t = +3.69 rapor edilip bırakılsaydı, yanlış bir keşif olurdu. Sayı gerçekti,
tekrarlanabilirdi, kalibrasyonu düzgündü — ama **sorduğu soru canlıda
sorulamayacak bir soruydu.**

Kural olarak: bir etiket satırların bir kısmını boş bırakıyorsa, o boşluk
rastgele değildir. Boş kalanlar sistematik olarak farklı satırlardır ve
ölçüm o farkı kenar sanabilir.

---

## Akran-göreli ölçüm (06.09.2026)

Buraya kadarki her ölçüm AUC ~0.50 verdi. Sebebi ararken iki yapısal boşluk
çıktı: kısa vade tarafı çapraz kesiti hiç kullanmıyordu (her özellik hisse
bazında mutlaktı — "RSI 32", hiçbir yerde "bugün akranlarına göre nerede"
sorulmuyordu) ve bütün ufuklar 3-21 bardı. Dördü de kapatıldı:

- **Akran grupları** (`havuz.tam_gruplar`): evrenin tamamı sektör × piyasa
  değeri terzili ile 32 gruba ayrıldı, 2056 sembol, grup başına 21-132 üye.
- **Çapraz kesitsel özellikler**: 20 göstergenin (grup, zaman) içindeki
  yüzdelik sırası, `x_` önekiyle.
- **Akran-arındırılmış etiket**: `akran_Ng = fazla_Ng − (grup × zaman
  ortalaması)`. Piyasa ve sektör hareketi çıkıyor, idiyosinkratik kısım
  kalıyor — çapraz kesitsel bir modelin tahmin edebileceği tek parça o.
- **Uzun ufuklar**: 21 / 42 / 63 gün.

Panel: 384.004 satır, 133 sütun. Model 128 genişlik × 3 katman, en fazla 200
devir, erken durdurma karar veriyor.

### İlk sonuç yanlıştı — ve neden yanlış olduğu

İlk çalıştırma 63 günde üst dilim için maliyet düşülmüş **%10.4**, t=8.49
verdi. Planın kendi notu "%1.5 üstü çıkarsa sızıntı ararım" diyordu. Üç ayrı
sebep vardı ve üçü de aynı yöne, modelin lehine, çalışıyordu.

**1. Arındırma birimi.** Eğitim ile test arasındaki boşluk ufku *işlem günü*
sayıp *takvim günü* olarak çıkarıyordu. 63 işlem günü ≈ 91 takvim günü,
atılan 68 gündü; eğitim etiketlerinin yaklaşık iki haftası test penceresinin
içinde kalıyordu. Boşluk ufukla büyüdüğü için 21 barda makul, 63 barda absürt
görünüyordu.

**2. Newey-West gecikmesi.** `4·(T/100)^(2/9)` kural-ı kaidesi verinin
*miktarına* bakar, *üretimine* değil. 63 günlük ileri getiride ardışık iki
günün 62 günü ortaktır; seri ~63 gecikmeye kadar korelasyonlu, kural ise 4
seçiyordu. Saf gürültüde bu, t=9.08'i t=2.08 yapıyor — test bunu artık
kilitliyor. `faktor_zaman`'ın kendi docstring'i bu tuzağı zaten yazıyordu,
`dilim_getirisi` onu görmezden gelmişti.

**3. Asıl mesele: ortalama uydurmaydı.** İlk ikisi düzeltildikten sonra t
8.49'dan 3.26'ya indi ama getiri %8.9'da kaldı. Sorun sızıntı değil etiketin
kendisiydi. `akran_63g`: ortalama +%0.56, **medyan −%2.11**, maksimum +%2479,
ve satırların en üst %1'i toplamın **%402'sini** taşıyor — yani geri kalan her
şey negatif. Tipik hisse akranına kaybediyor, artı ortalama bir avuç isimden
geliyor ve o isimlerin çoğu düzeltilmemiş ters bölünme. Güvenilirlik tablosu
bunu zaten söylüyordu: üst dilim **%49** isabetle ortalama +%8.9 getiriyordu.

Çözüm: her günün kesiti kendi 1. ve 99. yüzdeliğinde kırpılıyor. Yalnız aynı
günün verisini kullanır, ileriye bakmaz, ve taban oranını üst dilimle aynı
kuralla budar. Ortalamanın yanına **medyan** da yazılıyor; ikisi ayrışırsa
getiriyi birkaç ismin taşıdığı tabloda görünür.

### Düzeltilmiş sonuç

Üst dilim %10, akran-fazla getiri, gün bazında Newey-West t (gecikme = ufuk):

| Ufuk | Taban | 0 bp | 10 bp | 20 bp | t(10bp) | Medyan | AUC |
|-----:|------:|-----:|------:|------:|--------:|-------:|----:|
| 21 | −0.170% | +0.755% | **+0.655%** | +0.555% | **+2.52** | +0.457% | 0.528 |
| 42 | −0.030% | +2.501% | **+2.401%** | +2.301% | **+2.53** | −0.118% | 0.531 |
| 63 | −0.035% | +4.818% | **+4.718%** | +4.618% | **+3.98** | −0.203% | 0.508 |

Taban medyanları sırasıyla −%0.764, −%1.130, −%1.706. Yani sinyal satırları
akranlarına ortalamada kaybediyor; modelin işi en az kaybedeni seçmek.

**Null kontrolü.** `--karistir 42` etiketi ve getiriyi gün *içinde*
karıştırır; özellikler, tarihler ve çapraz kesit yapısı olduğu gibi kalır.
Sonuç: 3 ufkun **0'ı** geçiyor, üç üst dilim de negatif (−%0.62, −%1.01,
−%1.18), t −1.26 … −0.24. Ölçüm düzeneği kendi başına kenar üretmiyor.

### Ne iddia edilebilir, ne edilemez

- **21 gün — sağlam.** Hem ortalama hem medyan pozitif, medyan farkı tabana
  göre +%1.22, t=2.52, AUC 0.528, null kontrolünde temiz.
- **42 gün — kısmen.** AUC 0.531 ve dilimler tek yönlü (−%2.14 → +%2.50), ama
  medyan hâlâ −%0.12: tipik seçim akranına az da olsa kaybediyor, ortalamayı
  sağ kuyruk taşıyor. Tabandan iyi, mutlak olarak pozitif değil.
- **63 gün — hayır.** t en yüksek olan bu ama AUC 0.508 ve dilimler tek yönlü
  değil (3. dilim +%2.34, 6. dilim −%2.24). Buradaki +%4.72 sıralama gücü
  değil, budamadan sonra bile kalan kuyruk.

Üç ufuk denendi; birinde t≥2 çıkması tek başına az şey söyler. Buradaki
resmi tutan şey ikisinde birden t≈2.5 olması, 42'de dilimlerin tek yönlü
gitmesi, AUC'nin 0.52'yi geçmesi ve null kontrolünün temiz gelmesi.

---

## 21 güne odaklanma ve kenarın nereden geldiği (06.09.2026)

Üç ufuk içinde 21 gün en sağlamıydı, o yüzden üzerine gidildi. Dört kaldıraç
denendi ve **ayrı ayrı** ölçüldü.

### 1. Üst dilim gün bazında seçilmeli — bu bir ayar değil, düzeltme

`dilim_getirisi` üst %10'u **havuzlanmış** test kümesinden seçiyordu. Bu,
"üç yılın bütün sinyalleri içinde hangileri en yüksek puan aldı" sorusudur.
Model puanının seviyesi katmanlar ve rejimler arasında kaydığı için bu soru
seçimleri birkaç tarihe yığıyor — sentetik testte 60 günün 28'ine. Oysa
işleyebileceğiniz tek şey **bugünün adayları içinden bugünün en iyileri**.

Etkisi büyük ve ters yönde:

| | Getiri (10bp) | t | Medyan | Gün |
|---|---:|---:|---:|---:|
| Havuzlanmış seçim | +0.655% | +2.52 | +0.457% | 186 |
| Gün bazında seçim | +0.810% | **+0.77** | −0.132% | 372 |

Yani daha önce raporlanan t=2.52'nin bir kısmı hisse seçimi değil **tarih
seçimi**ymiş. Doğru ölçütte getiri yükseliyor ama anlamlılık kayboluyor.

### 2. Tohum topluluğu ve 3. sıralama hedefi

Gürültülü bir hedefte ağ her tohumda başka bir yere düşüyor; tohumlar arası
fark, aranan kenarla aynı mertebede. `--tohum-sayisi` birkaç tane eğitip
ortalamasını alıyor. `--siralama` ise eğitim hedefini 0/1 yerine **aynı gün
ateşlenen sinyaller içindeki yüzdelik sıra** yapıyor ve erken durdurmayı gün
içi sıra korelasyonuna bağlıyor — BCE ve Brier, kimsenin işlem yapmadığı
dağılım ortasına emek harcıyordu.

| Varyant | Getiri (10bp) | t | Medyan | AUC |
|---|---:|---:|---:|---:|
| 1 tohum, 0/1 hedef | +0.810% | +0.77 | −0.132% | 0.528 |
| 5 tohum | +0.974% | +0.52 | −0.143% | 0.530 |
| 5 tohum + sıralama | +0.864% | **+2.25** | −0.267% | 0.523 |

Topluluk getiriyi ve AUC'yi yükseltip t'yi düşürüyor: kazancı daha az güne
yığıyor. Sıralama hedefi tam tersini yapıyor — getiri benzer, günden güne
tutarlılık belirgin artıyor. İkisi birlikte t'yi 0.77'den 2.25'e taşıyor.

### 4. Ama üst dilim neyden oluşuyor?

Buraya kadar sayı iyileşiyordu. Asıl soru şuydu: model **neyi** seçiyor?
Rapora bir satır eklendi — üst dilimin ortanca özellikleri, genelin
ortancasına bölünerek:

```
ufuk 21: dolar_hacim=0.368  atr_pct=1.136  g_ma200_uzaklik=1.201
```

Model sistematik olarak **en az likit** isimleri seçiyor. Ve paneli dolar
hacmine göre beşe bölünce sebep ortaya çıkıyor:

| DV kovası | Ortanca DV | Ortalama akran-fazla | Medyan |
|---|---:|---:|---:|
| 0 (en ince) | 1.6M | **+1.594%** | +0.265% |
| 1 | 5.5M | +0.139% | −0.209% |
| 2 | 14.3M | −0.378% | −0.888% |
| 3 | 32.4M | −0.767% | −1.240% |
| 4 (en likit) | 78.7M | **−1.554%** | −2.340% |

Beş kovada da tek yönlü. Bu temiz bir gradyan ve yönü yanlış tarafa bakıyor:
önbellek **bugün kote olan** şirketleri tutuyor, sessizce ölenlerin çoğu
küçüktü, dolayısıyla mikro-kap bandında kalan şey kazananlar — kaybedenler
silinmiş halde. Hayatta kalma yanlılığı içeriden tam böyle görünür.

### Kenar likit bir bantta duruyor mu? — Hayır

`--min-hacim` ile taban konulduğunda:

| Evren | Satır | Getiri (10bp) | t | Medyan | AUC | DV eğilimi |
|---|---:|---:|---:|---:|---:|---:|
| Tümü | 384k | +0.864% | +2.25 | −0.267% | 0.523 | 0.368 |
| ≥ $5M/gün | 215k | **−0.484%** | −0.79 | −1.072% | 0.516 | 0.675 |
| ≥ $20M/gün | 123k | **−0.656%** | −1.06 | −1.496% | 0.506 | 0.955 |

Mikro-kaplar çıkınca kenar yok oluyor. Eğilim oranının 0.368 → 0.675 → 0.955
diye kapanması mekanizmayı da gösteriyor: ölçülen +%0.864, hisse seçiminin
değil likidite tarafına yaslanmanın getirisiydi.

**Sonuç:** 21 günde, işlem yapılabilir bir evrende, bu özelliklerden ölçülebilir
bir kenar çıkmıyor. Bu bir başarısızlık değil bir cevap — ve bu oturumdaki her
ölçüm keskinleştikçe aynı yöne, sıfıra doğru gitti.

### Kanıtlanamayan kısım

Hayatta kalma yanlılığı **en güçlü açıklama**, ispat değil. İspatı için kote
dışı kalmış hisselerin barları gerekir; yfinance onları vermiyor. Gradyanın
tek yönlü olması, kenarın $5M üstünde yok olması ve modelin 0.368'lik eğilimi
birlikte güçlü bir dolaylı kanıt oluşturuyor, o kadar.

---

## On yıllık veri — ve 21 günün cevabı (07.09.2026)

Önceki bölümdeki bütün ölçümler 501 günlük bardan çıktı. 21 günlük ufukta bu,
örtüşmeyen yaklaşık 24 dönem demek. t=2.25 gibi bir değerin bu kadar az
bağımsız dönem üzerinde ne kadar oynak olduğunu görmenin tek yolu daha uzun
bir geçmişti.

### Veriyi getirmek — yfinance devre dışı

06.09.2026 itibarıyla yfinance 1.0'ın bütün fiyat yolları (`yf.download` da,
`Ticker.history` de) `'NoneType' object is not subscriptable` ile düşüyor;
kütüphanenin çerez/crumb adımı `None` dönüyor. Aynı anda Yahoo'nun chart ucu
düz bir HTTP isteğine sorunsuz cevap veriyor — yani engellenmiş değiliz,
arızalı olan aradaki kütüphane. `src/gecmis.py` o yüzden uca doğrudan
bağlanıyor.

Doğrudan bağlanmak bölünme düzeltmesini de bize bırakıyor ve asıl sessiz
bozulma riski orada: Yahoo ham OHLC'yi düzeltmeden verip `adjclose`'u ayrı
koyuyor, düzeltilmemiş bir 2'ye 1 bölünme ise göstergelerin gerçek sandığı
%50'lik bir çöküş oluyor. Dört fiyat sütunu da `adjclose/close` oranıyla
ölçekleniyor, hacme dokunulmuyor. AAPL, MSFT ve NVDA'da örtüşen 501 barın
tamamı mevcut iki yıllık önbellekle **son hanesine kadar** aynı çıktı.

Sonuç: **2.616 hisse × 10 yıl** (2016-09-06 → 2026-09-04), ortanca 2.514 bar.
Panel 384.004 satırdan **1.872.659** satıra, 487 günden **2.494** güne çıktı.

### Sonuç: 2 yılda görünen şey 10 yılda yok

Yapılandırma birebir aynı tutuldu (5 tohum, sıralama hedefi, gün bazında
dilim, 4 katman) — değişen tek şey veri.

**Tüm evren:**

| Veri | Satır | Gün | Üst dilim (10bp) | t | Medyan | AUC | DV eğilimi |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 yıl | 24.820 | 372 | +0.864% | **+2.25** | −0.267% | 0.523 | 0.368 |
| 10 yıl | 118.978 | **1.978** | **+0.093%** | **−0.75** | −0.185% | 0.514 | 0.189 |

**Likit evren (≥ $5M/gün):**

| Veri | Satır | Gün | Üst dilim (10bp) | t | Medyan | AUC |
|---|---:|---:|---:|---:|---:|---:|
| 2 yıl | 15.426 | 278 | −0.484% | −0.79 | −1.072% | 0.516 |
| 10 yıl | 72.737 | **1.977** | **−0.671%** | −4.09 | −0.699% | 0.513 |

10 yıllık likit evrende dilimler tamamen düz:

```
-0.57  -0.81  -0.73  -0.81  -0.70  -0.59  -0.61  -0.56  -0.54  -0.57
```

En düşük dilimle en yüksek dilim arasında fark yok. Sıralama diye bir şey yok.

### Ne öğrendik

**21 günde, bu özelliklerden, işlem yapılabilir bir evrende ölçülebilir bir
kenar çıkmıyor.** İki yılda görünen t=2.25, beş kat veriyle t=−0.75'e döndü.
Bu, modelin kötü ayarlanmış olmasından değil; iki yılda 24 bağımsız dönem
üzerinde hesaplanan bir t değerinin bu kadar oynak olmasından.

İki küçük ayrıntı dürüstlük adına:

- **AUC 0.50'nin bir tık üstünde ve orada kalıyor** (0.514 / 0.513). 1977 gün
  üzerinde bu muhtemelen şanstan ayırt edilebilir. Yani modelin cılız bir
  sıralama gücü *var*; ama maliyet ve negatif taban oranını aşmaya yetmiyor.
  "Hiçbir şey öğrenmedi" ile "işe yarar bir şey öğrendi" arasında bir yerde,
  ve karar açısından ikincisi değil.
- **Sinyal satırları akranlarına kaybediyor.** Likit evrende taban −%0.649.
  Yani bu 12 kurulum, likit isimlerde ortalamada negatif bir işaret. Modelin
  işi iyi olanı seçmek değil, en az kötü olanı seçmekmiş — ve onu da
  yapamıyor.

### Bu oturumun deseni

Her ölçüm keskinleştikçe sonuç sıfıra doğru gitti; hiçbiri istisna olmadı.

| Ölçüm | Sonuç |
|---|---|
| Günlük kovalar, rsi2 | +0.030 (n_eff 40) |
| Saatlik kovalar | +0.002 (n_eff 700) |
| Bariyer etiketi, %54 kapsama | AUC 0.555, t=+3.69 |
| Aynısı, %100 kapsama | AUC 0.506, t=+0.53 |
| Akran etiketi, havuzlanmış dilim, 2 yıl | +0.655%, t=+2.52 |
| Gün bazında dilim | +0.810%, t=+0.77 |
| + sıralama hedefi | +0.864%, t=+2.25 |
| Likidite tabanı ≥ $5M | −0.484%, t=−0.79 |
| 10 yıl, tüm evren | +0.093%, t=−0.75 |
| 10 yıl, likit | −0.671%, t=−4.09 |

Bu bir başarısızlık listesi değil, bir ölçüm disiplininin çalıştığının kaydı.
Her satırda kenarı büyüten şey bir seçim yanlılığı, bir sızıntı ya da az
veriydi; düzeltildiğinde kenar gitti.

---

## Kesite geçmek — ve kenarın gerçekten neye yaslandığı (07.09.2026)

Buraya kadar her ölçüm **kurulum satırları** üzerindeydi: on iki dedektörden
biri ateşlediğinde oluşan barlar. O havuzun akran-göreli tabanı likit
isimlerde **−%0.649**. Yani model hiçbir zaman iyi hisse seçmiyordu; zaten
kötü olan bir havuzun içinden en az kötüyü seçiyordu, ve uzun-only sayı
büyük ölçüde havuz hakkında bir cümleydi.

Bunu gerektiren hiçbir şey yoktu. Akran sıralamaları ve akran-arındırılmış
etiket zaten her hisse için her gün hesaplanıyordu — hesaplanmak zorundaydı,
akran sıralaması bu demek — ve birkaç bin satır dışında hepsi atılıyordu.
`capraz_yaz` onları kendi tablosu olarak yazıyor; soru "bugün işlem gören iki
bin hisseden hangileri önümüzdeki ay sektörünü geçer" hâline geliyor.

21 günlük ufukta ardışık günler aynı bahsin bir gün kaymış hâli olduğu için
her 5. gün tutuluyor: **864.035 satır · 2.029 hisse · 503 gün**, on yıl.

### Yol boyunca düzeltilen üç şey

**Etiket, grup ortalamasından medyanına geçti.** 2018-07-05'te bir akran
grubunun 21 günlük ortalama fazla getirisi **+%5697**, medyanı **+%6.2**
çıkıyordu — QUBT'nin düzeltilmemiş bir hareketi. Etiket "fazla getiri eksi
grup ortalaması" olduğu için o gün o gruptaki *diğer her hisse* −%5697
etiketi alıyordu. Hücrelerin **%12.6'sında** ortalama medyandan beş puandan
fazla sapıyor. `akranmed_{N}g` medyana göre; ikisi de yazılıyor, çünkü etiket
değiştirmek aşağıdaki her sayıyı değiştirir ve bunun ayrı ölçülmesi gerekir.

**Grup kimliği özellik olmaktan çıkarıldı.** Etiketin grup ortalaması zaten
alınmış, dolayısıyla bir grup kuklasının tanım gereği tahmin edecek bir şeyi
yok. Yine de verildiğinde kuklalar özellik tablosunun tepesine t>8 ile
çıktı: demean edilmiş bir dağılımın çarpıklığı beceri diye okunuyordu.

**Newey-West gecikmesi gözlem sayar oldu, işlem günü değil.** İkisi ancak
günde bir satır varken aynı şeydir; her 5. günü tutan tabloda 21 günlük ufuk
beş satıra yayılır ve 21 kullanmak hata payını dört katına çıkarırdı. Günlük
tablodaki davranış değişmedi.

### Sonuç: dilimler ilk kez sıralanıyor

Ufuk 21, 399 gün, %10'luk üst dilim, 10bp maliyet düşülmüş:

| Koşu | Üst dilim | t | Taban | Tabana göre | AUC | DV eğilimi |
|---|---:|---:|---:|---:|---:|---:|
| Kesit, medyan etiket, ≥$5M | **+1.293%** | **+5.12** | +0.398% | +0.89% | 0.520 | 0.671 |
| Null kontrol (etiket karıştırıldı) | +0.172% | +1.50 | +0.398% | −0.23% | 0.502 | 0.856 |
| Taban ≥$20M | +0.788% | +2.03 | +0.060% | +0.73% | 0.517 | 0.807 |
| Ortalama etiket (medyan yerine) | −0.040% | −0.77 | −0.908% | +0.87% | 0.517 | 0.660 |

Dilimler uçtan uca monoton — bu araştırmada ilk kez:

```
-0.15  0.00  -0.04  0.08  0.20  0.31  0.51  0.78  0.89  1.39
```

Null kontrolde tamamen düz (`0.30 0.43 0.46 0.46 0.42 0.37 0.35 0.49 0.43
0.27`), AUC 0.502, üst−alt farkı −%0.238. Katman AUC'leri 0.508 / 0.516 /
0.516 / 0.533 — tek şanslı katman değil. Üst−alt farkı +%1.091, t=+2.69,
günlerin %59'unda pozitif.

Ortalama etiket satırı, etiket değişikliğinin ne yapıp ne yapmadığını
gösteriyor: tabana göre kenar neredeyse aynı (+%0.87'ye karşı +%0.89), ama
seviyeler aykırı değerlerce −%0.9'a çekilmiş. Medyan etiket kenarı
**yaratmadı**, ölçümü okunur hâle getirdi.

### Ama kenar neydi

Bu noktada sayı iyiydi. Asıl soru, üç turdur olduğu gibi, yine "model **neyi**
seçiyor" idi. İki cevap aynı yere çıktı.

**Birincisi, kendi başına duran özellikler.** 39 sütundan yalnızca dördü
|t|≥3 veriyor ve üçü aynı ekseni ölçüyor:

```
dolar_hacim      IC -0.0335   t -7.28
x_dolar_hacim    IC -0.0491   t -6.16
x_amihud         IC +0.0496   t +5.09
x_roc5           IC -0.0115   t -3.16     <- tek istisna
```

**İkincisi, düz ortalamanın seçtikleri.** Ağın yanında ayarlanacak hiçbir
şeyi olmayan bir temel çizgi çalışıyor: eğitim katmanında kendi başına duran
sütunları bul, ters bakanları çevir, sıralamalarını ortala. Getirisi ağınkine
eşit (+%1.207 / t=5.60'a karşı +%1.293 / t=5.12) — yani üç katman, dropout,
erken durdurma ve beş tohumlu topluluk hiçbir şey kazandırmıyor. Ve dört
katmanın **dördünde de** yalnızca şu üçü ortak:

```
kat 1: +x_roc126, +x_mom_6_1, +x_dip252_uzaklik, -dolar_hacim, -x_dolar_hacim,
       +x_amihud, +x_ma200_egim63, +x_roc252, +x_mom_12_1, ...
kat 2: -dolar_hacim, -x_dolar_hacim, +x_adx, +x_amihud, +x_ma200_egim63
kat 3: -dolar_hacim, -x_dolar_hacim, +x_amihud, -x_roc5, -x_roc10, -x_ma20_uzaklik
kat 4: -dolar_hacim, -x_dolar_hacim, +x_amihud, -x_roc5, -x_roc10, ...
```

Gerisi katmana özel — birinci katmanın bütün momentum sütunları bir daha hiç
görünmüyor.

### Belirleyici test: ekseni al, ne kalıyor

`--ozellik-disla dolar_hacim,amihud` bu sütunları modele hiç vermiyor; eğilim
tanısı onları yine de raporluyor, yani "söylenmediğinde de ince isimlere
yaslanıyor mu" sorusu cevaplanabilir kalıyor.

| | Üst dilim | t | Tabana göre | DV eğilimi |
|---|---:|---:|---:|---:|
| Likidite sütunları var | +1.293% | +5.12 | +0.89% | 0.671 |
| **Yok** | **+0.206%** | **+1.01** | **−0.19%** | **1.001** |

Eğilim oranı 0.671'den **1.001**'e gidiyor: söylenmediğinde model ince
isimlere hiç yaslanmıyor. Ve dilimler yalnızca düzleşmiyor, **tersine
dönüyor**:

```
0.58  0.57  0.52  0.38  0.37  0.37  0.37  0.28  0.23  0.31
```

En düşük puanlanan dilim en çok kazanıyor. Üst−alt farkı −%0.422.

**Yani +%1.293'ün tamamı likidite ekseniydi.** Ve bu veri kümesinde likidite
ekseni hayatta kalma yanlılığından ayrılamıyor: önbellek bugün kote olan
şirketleri tutuyor, sessizce ölenlerin çoğu küçüktü, ve bir önceki turda
ölçülen gradyan (en ince beşte bir +%1.59, en likit −%1.55) tam da hayatta
kalma yanlılığının ürettiği şekil. İkisi içeriden aynı görünüyor; ayıran tek
şey ekseni almaktı ve alındığında geriye bir şey kalmadı.

### Boş çıkan taraf: uzun hafıza

Bu tur için gösterge setine on beş uzun-hafıza sütunu eklendi — 3/6/12 aylık
getiri, son ayı çıkarılmış 12-1 ve 6-1 momentum, 52 hafta tepe/dip uzaklığı,
uzun ve düşüş oynaklığı, getiri çarpıklığı, ayın en iyi günü, Amihud etkisi,
200 günlük ortalamanın eğimi, hacmin kendi çeyreğine oranı. Gerekçe sağlamdı:
en uzun geriye bakış 200 barlık ortalamaydı ve 21 günlük ufukta literatürdeki
kesitsel etkiler ay ölçeğinde.

**Hiçbiri tutmadı.** Gün içi sıra korelasyonunda on beşinin de |t|<1:

```
x_en_iyi_gun21   -0.0088  -0.98      x_mom_12_1       +0.0041  +0.49
x_ma200_egim63   +0.0085  +0.98      x_dip252_uzaklik +0.0050  +0.49
x_getiri_carpiklik -0.0045 -0.90     x_tepe252_uzaklik +0.0045 +0.45
x_mom_6_1        +0.0050  +0.66      x_roc126         +0.0021  +0.27
x_roc63          -0.0047  -0.57      x_roc252         +0.0009  +0.11
```

12-1 momentum — kesitsel literatürün en çok belgelenmiş etkisi — bu evrende
ve bu dönemde IC +0.0041, t=+0.49. Kazanç momentumdan gelmiyor. Sütunlar
kaldı çünkü yokluklarını göstermek de bir ölçüm, ama hiçbirini taşımıyorlar.

### Geriye kalan tek dürüst pozitif

`x_roc5`: kısa vadeli tersine dönüş. IC −0.0115, t=−3.16, 399 gün, üçüncü ve
dördüncü katmanda düz ortalamanın seçtikleri arasında. Likidite dışında
|t|≥3'ü geçen tek sütun. Gerçek ama küçük — tek başına 10bp'yi karşılamıyor,
ve tersine dönüş tam da işlem maliyetinin en çok yediği stratejidir.

### Bu turun cevabı

Oran yükseltilemedi. Kesite geçmek doğru soruydu ve ilk kez sıralı bir dilim
yapısı üretti; ama o sıralamanın tamamı, işlem yapılabilirliği ölçtüğümüz
eksende duruyor ve o eksen elimizdeki veriyle hayatta kalma yanlılığından
ayrılamıyor.

Kalıcı olarak eklenenler: kesit tablosu ve `--kaynak capraz`, medyan akran
etiketi, üst−alt farkı, özellik bazında IC tablosu, düz ortalama temel
çizgisi, `--ozellik-disla`. Sonuncusu bu turu bitiren testti; ilk üçü olmadan
sayı iyi görünüyordu ve neden iyi göründüğü görünmüyordu.

---

## Şirket haberi: 8-K bildirimleri (08.09.2026)

Bir formasyon, şirket hakkında o sırada çıkan bir haberle birlikte bambaşka
bir şey ifade edebilir. Modelin bugüne kadar bu farkı görecek hiçbir sütunu
yoktu.

### Neden haber başlığı değil de SEC bildirimi

On yıllık geriye dönük bir ölçümde asıl zorluk haberin kendisi değil, zaman
damgası güvenilir bir kayıt. 8-K, şirketin "maddi bir olay oldu" diye kendi
yaptığı duyuru ve üç noktada kazınmış başlıklardan üstün:

- **Kabul zamanı kaynağından ve saniye hassasiyetinde.** Haber arşivlerinde
  yayın saati çoğu zaman yaklaşık, bazen sonradan düzeltilmiş.
- **Şirkete CIK ile bağlı.** "Apple" geçen her makaleyi toplamak başka bir
  şey.
- **Kote dışı kalmış şirketlerin bildirimleri duruyor.** Fiyat önbelleğinin
  baştan sona taşıdığı hayatta kalma yanlılığı burada yok.

Karşılığında kaybedilen şey **metin**: 8-K "yönetici ayrıldı" der, iyi mi
kötü mü demez. Yani olayın *türünü* biliyoruz, *tonunu* bilmiyoruz.

Toplanan: **2.789 şirket, 238.401 bildirim**, 2016-01 → 2026-09. CIK
bulunamayan 6 sembol, sıfır hata.

### Bütün risk tek bir yerde: bildirimin hangi güne yazıldığı

8-K'ların çoğu kapanıştan sonra dosyalanıyor, bilanço (2.02) bildirimlerinin
neredeyse tamamı öyle. Dosyalama gününe yazarsak o günün kapanışında
hesaplanan bir özellik henüz olmamış bir olayı görmüş olur — ve en büyük
fiyat hareketi tam onun ertesinde gerçekleşir. Yani hata ölçülebilir ve
tamamen sahte bir kenar üretir.

Bu yüzden her şey `etkin_gun` ile hizalanıyor: kabul saati New York'ta
16:00'dan önceyse dosyalama günü, sonraysa bir sonraki gün.

İlk sürüm bunu **tehlikeli yönde** yanlış yapıyordu. Yaz saatini görmezden
gelip sabit UTC-5 kullanıyor, yorumunda da bunun "temkinli" olduğu
yazıyordu. Tam tersiymiş: temmuzda 20:30 UTC gerçekte 16:30 EDT'dir,
kapanıştan sonra, ama UTC-5 ile 15:30 hesaplanıp aynı güne yazılıyordu.
Gerçek New York saatine geçince AAPL'ın bilanço bildirimlerinin kayma oranı
%30'dan %100'e çıktı. Test artık bir kış ve bir yaz vakasını birlikte
kilitliyor.

### Tanımlayıcı: olay çevresinde ne oluyor

Kovalar sıfıra karşı değil, **hiç bildirimi olmayan satırlara karşı, gün
eşleşmeli** ölçülüyor. Etiket akran medyanına göre arındırılmış ve sağa
çarpık, yani her kova pozitif ortalama verir; sıfıra karşı test etmek
kimsenin sormadığı bir soruyu cevaplar.

**Kesit paneli (855.919 satır, 499 gün):**

| Son bildirimden bu yana | N | Budanmış | Fark | t |
|---|---:|---:|---:|---:|
| olay günü | 34.156 | 0.633% | −0.423% | −0.79 |
| 1-4 bar sonra | 116.968 | 0.675% | −0.381% | −0.71 |
| 5-20 bar sonra | 305.175 | 0.661% | −0.329% | −0.57 |
| 21-62 bar sonra | 255.027 | 0.741% | −0.005% | −0.01 |
| bildirim yok | 144.593 | 0.840% | taban | — |

Hiçbiri anlamlı. Türe göre bakıldığında tek çıkan `sonuc` (bilanço):
−%0.888, t=−2.47 — ve **negatif**. `kotasyon` ham ortalamada %25.06
görünüyor ama budanmışı %6.09; 2.844 satırda birkaç isim taşıyor.

### Asıl soru: aynı kurulum, bildirim varken ve yokken

Sinyal paneli (1.861.215 satır). Her kurulum kendi içinde, son 5 barda
bildirim olan ve olmayan satırlara bölünüyor:

| Kurulum | Olaylı | Olaysız | Fark | t |
|---|---:|---:|---:|---:|
| nr7_ic_bar | 0.604% | 0.897% | −0.730% | −1.72 |
| rsi2_asiri_satim | 0.613% | 0.814% | −0.678% | **−2.41** |
| **yutan_ayi** | 0.680% | 0.546% | **+0.470%** | **+2.08** |
| cekic | 0.225% | 0.550% | −0.415% | −1.55 |
| hacim_kurumasi | 0.252% | 0.816% | −0.394% | −1.43 |
| boga_yutan | 0.773% | 0.894% | −0.391% | −1.35 |
| ma20_geri_cekilme | 0.440% | 0.559% | −0.332% | **−2.02** |
| uc_gun_geri_cekilme | 0.741% | 0.832% | −0.205% | −0.85 |
| bosluk_dolumu | 0.798% | 1.654% | −0.167% | −0.19 |
| dagitim_gunu | 0.679% | 1.002% | −0.164% | −0.62 |
| hacimli_kirilim | 1.000% | 1.012% | −0.065% | −0.22 |
| bollinger_sikismasi | 0.481% | 0.805% | −0.037% | −0.12 |

**On iki kurulumun on biri, bildirim varken daha kötü.** Tek istisna
`yutan_ayi` ve o zaten bir **short** formasyonu — yani "hisse için kötü",
"kurulum için iyi" demek. Yön kendi içinde tutarlı.

İşaret testi: 11/12 aynı yönde → p=0.0063; short'un işareti çevrilince 12/12
→ p=0.0005. **Ama kurulumlar bağımsız değil** — aynı günlerde ve aynı
isimlerde üst üste biniyorlar, dolayısıyla bu bir üst sınır, kanıt değil.

Sinyal satırlarında olay sütunlarının kendi gün-içi sıra korelasyonu da aynı
yöne bakıyor:

```
olay_denetci   IC -0.0176  t -3.51      olay_regfd     IC -0.0093  t -2.84
olay_yonetim   IC -0.0081  t -2.92      olay_sozlesme  IC -0.0067  t -2.35
```

Dördü de negatif: denetçi değişikliği, yönetici değişikliği, Reg FD
açıklaması ve önemli sözleşme, hepsi kurulumun ardından akranlarına göre
**daha düşük** getiriye işaret ediyor.

### Katman-dışı: model bunu kullanabiliyor mu

Hayır.

| Panel | Olay sütunlarıyla | Olaysız |
|---|---:|---:|
| Kesit, ≥$5M | +1.212% / t=4.49 | +1.293% / t=5.12 |
| Sinyal, ≥$20M | +0.280% / t=0.49 | +0.432% / t=0.79 |

İkisinde de **biraz daha kötü**, ikisinde de fark anlamsız. Kesitte 13 olay
sütununun en güçlüsü |t|=1.30; |t|≥3'ü geçen 15 sütunun yalnızca 1'i olay
sütunu ve listenin tepesi hâlâ likidite.

### Ne öğrendik

İlişki gerçek ve yönü tutarlı: **şirket bir şey açıkladıysa uzun bir
formasyon daha az güvenilir.** Ama etkisi, 21 günlük ufukta *hangi hisseyi
alacağını* değiştirecek kadar büyük değil.

Fark şurada: tanımlayıcı tablo ortalama bir ilişki görüyor, model ise bugün
bir sıralama üretmek zorunda. IC değerleri 0.007-0.018 mertebesinde; bu, bir
güven düşürücü olarak anlamlı ama bir seçici olarak değersiz. Ve modelin
koruyacağı bir kenarı olmadığı için — sinyal paneli t=0.49 ile zaten
sıfırda — güven düşürmenin tutunacak bir yeri yok.

Aynı sebeple "son 5 günde bildirim varsa uzun kurulumu atla" şeklinde bir
veto kuralı da denenmedi: kesit panelinde o farkın kendisi −%0.080, t=−0.54.

### Kalıcı olarak eklenenler

`src/olay.py` (EDGAR çekici, kapanış-sonrası hizalama, bar bazında
özellikler, etki tablosu), `run.py olay cek|ekle|etki`, ve 13 olay sütunu.
Olay sütunları panel derlemesinin **içinde değil**, sonradan birleştirmeyle
ekleniyor: yalnızca (hisse, tarih)'e bağlılar, dolayısıyla her yeni bildirim
için 2600 hissenin 93 göstergesini yeniden hesaplamak gerekmiyor — kırk
dakika yerine bir dakika.

---

## Hayatta kalma yanlılığı — ölçüldü (10.09.2026)

Bu araştırmadaki her pozitif sonuç aynı yere çıktı: kenar en ince
isimlerde. Kesitsel modelde dolar hacmi ve amihud çıkarılınca dilimler
tersine döndü. İki açıklama vardı ve içeriden aynı görünüyorlardı — likidite
primi, ya da önbelleğin yalnızca bugün kote olan şirketleri tutması.

Artık kimin öldüğünü bilen bir kaynak var.

### Önce: düzeltilemiyor

Yahoo kote dışı sembollere **404** dönüyor (TWTR, ATVI, CERN, XLNX, ZNGA,
SIVB, FRC denendi). Daha kötüsü, bazılarına 200 dönüyor ama bunlar **yeniden
kullanılmış semboller**: SBNY 2024-08'de, BBBY 2026-07'de başlıyor, ikisi de
bambaşka şirket. Körlemesine çekmek yanlılığı düzeltmek değil, ölmüş bir
şirketin yerine yenisinin fiyatlarını koymak olurdu.

### Ölçüm

Kohort: 2016 sonunda XBRL ile bilanço raporlayan her şirket (`Assets
CY2016Q4I`, tek istekte 6.591 şirket). Hayatta kalan: 2025 sonunda hâlâ
raporlayan. Büyüklük ölçüsü gelir değil **toplam varlık** — geliri olmayan
biyoteknoloji gelir çerçevesinde hiç görünmüyor ve mikro-kap bandı tam
orası.

| Kova | N | Ortanca varlık | Hâlâ raporluyor | Fiyat önbelleğinde |
|---:|---:|---:|---:|---:|
| 0 | 660 | 0 M$ | **23%** | **1%** |
| 1 | 659 | 1 M$ | 32% | 2% |
| 2 | 659 | 14 M$ | 47% | 7% |
| 3 | 659 | 61 M$ | 48% | 14% |
| 4 | 659 | 202 M$ | 44% | 20% |
| 5 | 659 | 517 M$ | 51% | 32% |
| 6 | 659 | 1.14 Mr$ | 57% | 39% |
| 7 | 659 | 2.32 Mr$ | 55% | 38% |
| 8 | 659 | 5.58 Mr$ | 65% | 36% |
| 9 | 659 | 24.14 Mr$ | **78%** | 22% |

Kohortun tamamında on yılda **%50** raporlamayı bırakmış; bugün fiyat
önbelleğinde olan yalnızca **%21**.

### Ne anlama geliyor

Model yalnızca son sütunu görüyor. En küçük kovada o sütun **%1**. Yani o
banttaki "kenar", tanım gereği hayatta kalanların kendisidir: en ince beşte
birde ölçülen +%1.59'un karşısında, o bandın dörtte üçünün kaybolmuş olması
duruyor.

Bu, likidite ekseninin neden çıkarıldığında dilimleri tersine çevirdiğini de
açıklıyor — model likidite primi bulmuyordu, hayatta kalmayı buluyordu.

### Dürüstlük notu: yanlılık tek yönlü değil

"Raporlamayı bırakmak" ölmek demek değil. Satın alınmak da bir çıkış ve
genellikle **primli**. Yani önbellek hem batanları hem satın alınanları
siliyor; ikisi ters yönlerde. Mikro-kap bandında batmaların baskın olması
beklenir ama bu ölçüm onu ayırmıyor. `cikis_nedeni()` son bildirimlerdeki
2.01 (devralma tamamlandı) ve 3.01 (kotasyon uyumsuzluğu) maddelerine bakıp
ayırmayı deniyor; ancak yalnızca bir zamanlar evrende olup sonradan düşen
semboller için cevap verebiliyor, hiç girmemişler için bildirim indirmek
gerekiyor.

Ayrıca kohort XBRL raporlayanlarla sınırlı: 2016'da XBRL'e hiç girmemiş çok
küçük şirketler ve yabancı ihraççılar dışarıda.

---

## Ufuk kısaltmak: hipotez ve sonucu (10.09.2026)

Likidite ekseni çıkarıldığında 21 günde dilimler tersine dönüyordu. Geriye
kalan tek dürüst sinyal 5 günlük tersine dönüştü (`x_roc5`, t=−3.16) ve
tersine dönüş kısa ufuklu bir etkidir — yani belki 21 gün yanlış ufuktu.

### Önce ucuz kontrol

Ufuk seçmek için seksen model eğitmeye gerek yok. `meta --sadece-ic` her
özelliğin gün içi sıra korelasyonunu ufuk ufuk basıyor; yirmi saniye, 0.17 GB.

| Özellik | u=3 | u=5 | u=10 | u=21 |
|---|---:|---:|---:|---:|
| `x_roc5` (tersine dönüş) | −3.0 | −3.2 | −3.5 | −3.5 |
| `x_bb_pct` | −2.9 | −3.3 | −3.4 | −3.2 |
| `x_ma20_uzaklik` | −2.8 | −3.4 | −3.3 | −2.9 |
| `x_amihud` (likidite) | +4.0 | +4.5 | +5.9 | **+8.3** |
| `x_dolar_hacim` | −3.7 | −6.3 | −7.9 | **−10.8** |

|t|≥3 geçen sütun: u=3'te 4, **u=5'te 9**, u=10'da 9, u=21'de 5.

Hipotez yanlıştı: tersine dönüş **her ufukta aynı güçte**, seyrelmiyor.
Değişen tek şey kirlilik — likidite ekseni ufukla büyüyor. Bu, hayatta kalma
yanlılığının olması gereken şekli: uzun pencere, ölüme daha çok zaman.

Yani kısa ufuk sinyali güçlendirmiyor, onu boğan şeyi zayıflatıyor. 21 günde
likidite tersine dönüşü 3'e 1 eziyor; 5 günde ikisi denk. Denemeye değer
hâle gelen şey buydu.

### Sonuç

Ufuk 5, 400 gün, %10'luk üst dilim, 10bp:

| Koşu | Üst dilim | t | Taban | Tabana göre | AUC | DV eğilimi |
|---|---:|---:|---:|---:|---:|---:|
| u=5, tam | +0.240% | +3.43 | +0.138% | +0.102% | 0.510 | 0.79 |
| **u=5, likiditesiz** | **+0.106%** | **1.38** | +0.138% | **−0.032%** | 0.506 | **1.013** |
| u=21, tam | +1.293% | +5.12 | +0.398% | +0.895% | 0.520 | 0.671 |
| u=21, likiditesiz | +0.206% | 1.01 | +0.398% | −0.192% | 0.503 | 1.001 |

u=5 likiditesiz dilimler:

```
0.18  0.10  0.12  0.15  0.11  0.10  0.11  0.18  0.13  0.21
```

**Düz.** 21 günde tersine dönüyordu, burada dönmüyor — bu kadarı iyileşme,
çünkü model artık ters tahmin etmiyor. Ama kenar yok: üst dilim tabanın
altında, üst−alt farkı −%0.148 (t=−1.17), bileşik +%0.135 (t=1.65).

### Cevap

**Likidite ekseni olmadan hiçbir ufukta kenar yok.** 21 günde ters, 5 günde
düz. Kısa ufka geçmek kirliliği azalttı, altından bir şey çıkarmadı.

Tersine dönüş ailesi (`roc5, bb_pct, rsi7, ma20_uzaklik, dip20, donchian20`)
tek tek |t|≈3 veriyor ve altısı da aynı şeyi ölçüyor: "yakın zamanda düştü".
Ama model bunu bir dilim farkına çeviremiyor — IC'ler 0.013-0.018
mertebesinde ve bu, günlük bir sıralamada seçim yapmaya yetmiyor.

### Maliyet notu

u=5, u=21'e göre ~4 kat devir. Tablodaki 10bp her tutuş için düşülüyor, yani
u=5'in yılda ~50, u=21'in ~12 kez ödemesi zaten hesapta. Yine de: u=5 tam
koşusunun tabana göre +%0.102'si, u=21'in +%0.895'inin yıllığa çevrilmiş
hâline yakın duruyor ve ikisi de aynı likidite eksenine yaslanıyor.

---

## Piyasanın yönü tahmin edilebiliyor mu (11.09.2026)

Kesitsel modelin etiketi akran-göreli, yani piyasayı **tanım gereği** aradan
çıkarıyor. "Sektöründen %2 iyi" ile "hesabın büyüdü" arasındaki fark tam
olarak bu: sektör %12 düştüyse, akranını %2 geçen bir portföy %10 kaybeder.

`regime.py` bugünün rejimini etiketliyor ama bilerek tahmin yapmıyor
("piyasayı zamanlamak için değil, bağlam vermek için"). Bu ölçüm tahminin
mümkün olup olmadığını soruyor.

### Önce örneklem tavanı

On yıl, 21 günlük örtüşmeyen dönem cinsinden **~118 gözlem**. Bu sayı bir
daha artmaz — piyasa zamanlamasında örneklem takvimle sınırlı, evreni
büyütmek işe yaramıyor. Kesitte iki bin hisse × beş yüz günle çalışmaya
alışmış bir disiplin için bu çok az, ve burada |t|=2 civarı bir sonuç kanıt
değildir.

### Sonuç: hiçbir şey

SPY'ın 21 gün sonrası, 2.492 gün, 14 özellik:

| Özellik | rho | t | rho(düşüş) |
|---|---:|---:|---:|
| vix | +0.173 | +1.84 | +0.102 |
| ma200_uzaklik | −0.169 | −1.68 | −0.141 |
| tepe_uzaklik | −0.178 | −1.68 | −0.103 |
| getiri63 | −0.136 | −1.63 | −0.088 |
| genislik_ayrisma | +0.112 | +1.54 | +0.001 |
| oynaklik20 | +0.160 | +1.45 | +0.134 |
| **genislik** | −0.057 | **−0.74** | −0.144 |
| genislik_degisim10 | −0.032 | +0.18 | −0.037 |

Hiçbiri |t|=2'yi geçmiyor. En iyisi VIX, +1.84.

İşaretler literatürle tutarlı — yüksek VIX sonrası daha iyi getiri
(oynaklık risk primi), 200 günlük ortalamadan uzaklık negatif (ortalamaya
dönüş) — ama 118 dönemle bunların hiçbiri ayırt edilemiyor.

**Genişlik de eklemedi.** En ayırt edici özellik olacağını umuyordum: endeks
birkaç dev hisseyle ayakta durabilir, hisselerin yüzde kaçı kendi
ortalamasının üstünde sorusu duramaz. 2.598 hisseden on yıllık seri
hesaplandı; getiriye karşı t=−0.74.

Tek kayda değer ayrıntı: genişlik, **düşüş bayrağıyla** en güçlü ilişkiyi
veren özellik (rho=−0.144, diğerlerinin hepsinden yüksek). Yani genişlik
düşükken %5'ten fazla düşüş biraz daha olası. Ama rho 0.14 ve getiriye karşı
t=−0.74; bir karar dayandırılmaz.

### Bunun pratik anlamı

"Piyasa düşerse yine kaybederim" sorunu **tahminle çözülmüyor.** Ölçüm bunu
söylüyor ve bu, literatürün de söylediği şey.

Çözülebileceği yer başka: tahmin etmek yerine **maruziyeti kaldırmak**.
Model akran-göreli bir sıralama üretiyor; o sıralamanın karşılığını almanın
yolu piyasa yönünü bilmek değil, portföyü piyasaya nötr kurmak — üst dilimi
alıp alt dilimi satmak. O sayı zaten ölçülüyor (üst−alt), ve bu projede
ölçüldüğünde yine likidite eksenine dayandığı çıktı: u=21 tam +%1.091
(t=2.69), likidite çıkarılınca −%0.422 (t=−1.03).

Yani iki ayrı duvar aynı yere çıkıyor: kesitte kalan tek şey likidite
ekseni, ve o eksen hayatta kalma yanlılığından ayrılamıyor.

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
- **Olayın tonu bilinmiyor.** 8-K "yönetici ayrıldı" der, iyi mi kötü mü
  demez. Ton için başlık metni gerekiyor ve o, on yıl geriye ücretsiz olarak
  bulunamıyor. Yahoo'nun haber ucu ayakta ama yalnızca güncel haber veriyor,
  yani canlı tarama için kullanılabilir, geriye dönük ölçüm için değil.
- **Ağ, düz ortalamayı geçmiyor.** Üç katman, dropout, erken durdurma ve beş
  tohumlu topluluk, ayarlanacak hiçbir şeyi olmayan bir eşit-ağırlık
  ortalamasıyla aynı getiriyi veriyor (+%1.207 / t=5.60'a karşı +%1.293 /
  t=5.12), $20M evreninde ondan geride kalıyor (+%1.075 / t=3.20'ye karşı
  +%0.788 / t=2.03). Bu evrende ve bu ufukta cevap daha çok parametre değil.
- **Akran-göreli etiket cebe girmez.** `akranmed_{N}g` "grubunun medyanını ne
  kadar geçti" ölçer; uzun-only bir portföy ham getiri kazanır. Cebe giren
  yakın karşılık üst−alt farkıdır.
- **Hayatta kalma yanlılığı artık sayıyla belli**: 2016 kohortunun en küçük
  onda birinde bugün hâlâ raporlayan %23, fiyat önbelleğinde olan %1. En
  büyük onda birinde %78. Giderilemiyor (Yahoo kote dışı sembollere 404
  dönüyor, 200 dönenler yeniden kullanılmış semboller), ama artık büyüklüğü
  tahmin değil ölçüm.
- **Hayatta kalma yanlılığı ölçüldü ama giderilemedi**: kote dışı kalmış
  hisselerin barları elde yok. Mikro-kap bandındaki fazla getirinin ne
  kadarının bundan geldiği bilinmiyor; yalnızca yönü ve büyüklüğü belli.
- **Kuyruk budandı, yok olmadı**: günlük kesit 1/99 yüzdelikte kırpılıyor.
  Bu, ortalamayı birkaç ismin taşımasını engeller ama uç hareketlerin gerçek
  olduğu durumlarda getiriyi de az gösterir. Kırpma eşiği bir tercihtir,
  ölçülmüş bir sabit değil.
- **Düzeltilmemiş bölünmeler**: +%2479 gibi değerler ters bölünme artığı.
  Budama bunları etkisizleştiriyor ama kaynaktan temizlemiyor.
- **Üç ufuk denendi**: birinde t≥2 çıkması tek başına çoklu-test karşısında
  zayıftır. Resmi tutan şey ikisinde birden çıkması ve null kontrolü.
- **Hiçbir şey otomatik yapılmıyor.** Sistem sinyal üretir ve sayımı gösterir.
  Karar kullanıcınındır ve **bu bir yatırım tavsiyesi değildir.**
