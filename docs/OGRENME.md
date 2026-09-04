# Öğrenme Sistemi — Kendi Kendini Besleyen Döngü

Bu belge, sisteme eklenen derin öğrenme katmanını ve geri besleme döngüsünü
anlatır: nasıl çalışır, neden bu şekilde tasarlandı, ve **ne zaman güvenilir
hale gelir.**

---

## Önce en önemli şey

**Bugün bu sistemle model eğitilemez ve bu kasıtlıdır.**

```
$ python run.py ml status
  anlık görüntü : 2   (gereken 60)
  veri aralığı  : 1 gün  (gereken 120)
  İLERLEME      : %0.8
  eğitime hazır : hayır
```

Az veriyle eğitilen bir model, **güvenilir görünen ama tamamen gürültüye
uydurulmuş** tahminler üretir. Finansal veride bu, sistemin en büyük riskidir:
model kendinden emin bir sayı verir, sayı yanlıştır, ve yanlış olduğu ancak
para kaybedildikten sonra anlaşılır.

Bu yüzden sistem, veri yeterli olana kadar eğitimi **reddeder**. `--force` ile
zorlanabilir ama sonuçların güvenilmez olduğu açıkça söylenir.

---

## Sorunun cevabı: aralıklı mı, sürekli mi?

Sorduğun soru doğru soruydu. Cevap **her ikisi de değil** — üç ayrı ritim var
ve her biri farklı bir şeye bağlı:

| Adım | Ritim | Neden |
|------|-------|-------|
| **Veri toplama** | Günde bir, piyasa kapandıktan sonra | Faktörlerin çözünürlüğü günlük. Temel veri çeyreklik, teknik göstergeler günlük bar üzerinden. Gün içi toplamak **sinyal eklemez**, yalnızca API yükünü ve yarım-bar gürültüsünü artırır |
| **Etiketleme** | Sürekli, kendiliğinden | Bir günün etiketi 21 işlem günü sonra olgunlaşır. Sistem her çalıştığında ufku dolan satırları otomatik etiketler |
| **Yeniden eğitim** | Periyodik (5 taramada bir) | Her gün yeniden eğitmek anlamsız: 1 günlük yeni veri model ağırlıklarını anlamlı değiştirmez, ama aşırı uyum riskini ve hesap maliyetini artırır |

**Sürekli (streaming) öğrenme neden yanlış olurdu:** Hedef 21 günlük ileri
getiri. Gün içinde model güncellemek, henüz sonucu belli olmamış tahminlere
göre ayar yapmak demektir. Ayrıca gün içi fiyat gürültüsü, günlük faktör
sinyalinden kat kat büyüktür — model gürültüyü öğrenir.

---

## Döngü

```
   ┌──────────────────────────────────────────────────────────┐
   │                                                          │
   ▼                                                          │
1. TOPLA        günlük tarama → feature store                 │
   │            (run.py daily — her iş günü)                   │
   ▼                                                          │
2. ETİKETLE     ufku dolan satırlara ileri getiri             │
   │            (otomatik, 21 işlem günü sonra)                │
   ▼                                                          │
3. EĞİT         sızıntısız ileri yürüyüşlü bölmelerle         │
   │            (5 taramada bir, otomatik)                     │
   ▼                                                          │
4. DEĞERLE      OOS: IC, ICIR, ilk-dilim getiri farkı         │
   │                                                          │
   ▼                                                          │
5. TERFİ ✓/✗    yalnızca eşikleri VE taban çizgisini geçerse  │
   │                                                          │
   ▼                                                          │
6. UYGULA       `model_score` parametresi olarak skora girer  │
   │            ağırlık = ölçülen beceriyle orantılı           │
   ▼                                                          │
7. İZLE         canlı tahmin vs gerçekleşen → bozulma ────────┘
```

---

## Dört ay beklemeden başlamak — geçmişe dönük panel

Feature store günde bir anlık görüntü büyüyor, kapı 120 gün içinde 60 görüntü
istiyor. Sıfırdan başlayan bir sistemde bu **dört ay** demek. O süre boyunca
GRU'nun işe yarayıp yaramadığını bile öğrenemezsin.

Ama önbellekte her hisse için **iki yıllık günlük fiyat verisi zaten var.**
Fiyattan türetilen faktörler geçmişteki herhangi bir gün için yeniden
hesaplanabilir: seriyi o güne kadar kes, aynı fonksiyonu çağır.

```bash
python run.py history                    # ~1 saat, tüm önbellek
python run.py ml train --pretrain --models ridge,mlp,seq
```

### Neden vektörize yazılmadı

`backfill.py`, `factors.f_*` fonksiyonlarını **kesilmiş bir DataFrame ile**
çağırıyor. Aynı hesabı vektörize (tüm seri için tek seferde) yazmak yaklaşık
50 kat hızlı olurdu. Yazılmadı, çünkü o durumda eğitim özelliği ile canlı
özellik iki ayrı kod yolundan gelir ve aralarında sessizce fark oluşabilirdi.
Model bir şeyi öğrenip başka bir şeyi görürdü. Hız için doğruluk feda edilmiyor.

### Kesintiye dayanıklılık

Tüm evren için bu iş bir saati buluyor. Bir saatlik hesabı kesintide çöpe atan
bir tasarım kabul edilemez — üretimde bir kere oldu, iş öldürüldü ve 40 dakika
gitti.

Her 150 sembolde sonuç diske yazılıyor ve işlenen semboller kaydediliyor. Aynı
komutu tekrar çalıştırınca kaldığı yerden devam ediyor. Yarım kalmış bir çalışma
da kullanılabilir panel bırakıyor: anlık görüntü dosyaları biriken tüm
yığınlardan yeniden üretiliyor, sadece içindeki hisse sayısı az oluyor.
`--restart` baştan başlatır.

Sıralama önemli: önce veri, sonra işaretleme. Ters olsaydı kesinti anında sembol
"işlendi" görünür ama satırları kaybolurdu.

`--merge-only`, üretim **devam ederken** eldeki yığınları panele çevirir. 3000
hisse hesaplanırken 300 hisseyle eğitim denemesi yapabilirsin.

### Etiketleme ağa çıkmamalı

Etiket geçmiş fiyattan hesaplanır; ufku dolmuş bir anlık görüntü için gereken
fiyat zaten önbellekte vardır. Ama `label_forward_returns` normal `fetch`
yolunu kullanıyordu ve onun TTL'i 6 saat — birkaç yüz sembol birkaç yüz ağ
isteğine dönüşüyordu. Bir eğitim çalışması bu yüzden hiç hesap yapmadan
dakikalarca ağ bekledi.

Artık önce uzun ömürlü önbellek deneniyor (7 gün), ağ yalnızca hiç kayıt
olmayan semboller için kullanılıyor.

### Ne dahil değil ve neden

Yalnızca **11 fiyat/hacim faktörü** üretiliyor. Temel veriler — F/K, analist
notu, EPS revizyonu, kısa pozisyon oranı, kurumsal sahiplik — Yahoo'dan sadece
*bugünkü* haliyle geliyor, geçmiş değerleri yok. Bunları geriye taşımak,
"şirketin bugün bilinen kârlılığını bir yıl önceki güne yazmak" olurdu. Bu
klasik geleceğe bakış hatasıdır ve modeli gerçek dışı başarılı gösterir. Bu
yüzden hiç dahil edilmiyorlar.

### Yanlılık — dürüstçe

1. **Hayatta kalma yanlılığı.** Önbellekteki evren *bugün kote olan* hisseler.
   Geçen yıl içinde çöküp kote dışı kalanlar burada yok. Bu panelde ölçülen
   başarı, gerçekte elde edebileceğinin üzerindedir.
2. **Evren yanlılığı.** Tarama dönüşümlü olduğu için önbellek, evrenin rastgele
   bir örneği değil en son taranan dilimidir.

Bu yüzden **bu panel şampiyon üretemez.** `promotion_check`, `pretrain`
bayrağı taşıyan sonucu — sayılar ne kadar iyi görünürse görünsün — reddeder.
Ön eğitim modeli ayrı dosyaya yazılır, iki depo hiç birleşmez.

Panelin amacı **mimari seçimi:** bu problemde hangi model türü sinyal
yakalayabiliyor? Modelin skora dokunma kararı hâlâ gerçek ileriye dönük anlık
görüntüleri bekliyor.

Testler: `python tests/test_backfill.py` (34 test). En kritiği: bir anlık
görüntünün tarihinden *sonraki* tüm fiyat barlarını üçe katlıyoruz ve o
görüntünün hiçbir değeri değişmiyor — yani geçmişe dönük panel geleceği
okumuyor.

### İlk gerçek ölçüm (14.08.2026)

276 hisse, 73 anlık görüntü, 2025-09 → 2026-07:

| Model | IC | ICIR | Katman | İlk-dilim farkı |
|-------|---:|-----:|-------:|----------------:|
| ridge | −0.081 | −1.03 | 2 | +0.0067 |
| mlp   | −0.055 | −0.98 | 2 | +0.0004 |
| seq   | −0.054 | −0.93 | 2 | +0.0001 |

**Bu sonuçtan model seçilmez ve seçilmemeli.** İki katman hiçbir şey kanıtlamaz.
`--splits 4` istendi ama 2 kuruldu: her katman için test penceresinden geriye
21+5 gün arındırılıyor ve en az 20 eğitim günü gerekiyor; 73 görüntü ancak bu
kadarına yetiyor.

Yine de not edilmeye değer iki şey var:

1. **Üçü de negatif ve birbirine yakın.** Rastgele olsaydı işaretler karışırdı.
   Bu pencerede (Eyl 2025 – Tem 2026) fiyat/momentum faktörleri 21 günlük göreli
   getiriyi *ters* tahmin etmiş görünüyor — kısa vadeli ortalamaya dönüş olabilir.
2. **IC negatif ama ilk-dilim farkı pozitif.** Yani uçlar ile orta birbiriyle
   çelişiyor. Küçük örneklem gürültüsünün klasik işareti.

Daha fazla katman için daha fazla *gün* gerekiyor (hisse değil): `--step 1` ile
73 yerine ~220 görüntü üretilebilir, bedeli 3 kat hesap.

Sistemin doğru davrandığının kanıtı: üç aday da kayıt defterine yazıldı, hiçbiri
şampiyon olmadı, `model_score` ağırlığı 0 kaldı.

### Çoklu ufuk ölçümü (18.08.2026)

Aynı panel, üç ufuk, üç model + topluluk:

| Ufuk | ridge | mlp | seq | **topluluk** | Katman |
|-----:|------:|----:|----:|-------------:|-------:|
| 5 gün | −0.0273 | −0.0172 | −0.0051 | **−0.0094** | 4 |
| 21 gün | −0.0533 | −0.0321 | +0.0020 | **−0.0618** | 2 |
| 63 gün | — | — | — | — | **0** |

Üç şey öğretti:

**1. Kısa ufuk daha çok katman veriyor.** 5 günde 4 katman kuruldu, 21 günde 2.
Sebep mekanik: her katman için test penceresinden geriye `ufuk + tampon` gün
arındırılıyor. Ufuk küçüldükçe aynı veriden daha fazla bağımsız ölçüm çıkıyor.
"Kanıt istiyorsan önce kısa ufka bak" demenin veri tarafındaki karşılığı bu.

**2. 63 günlük ufuk bu veriyle ÖLÇÜLEMİYOR.** Hiçbir katman kurulamadı — 73
görüntü, 3 işlem günü aralıkla, 63+5 günlük arındırma ve en az 20 eğitim günü
şartıyla tek bir geçerli bölme bile üretmiyor. Bu bir hata değil, bir **sınır
bildirimi**: uzun ufuk ölçmek istiyorsan daha fazla *gün* gerekiyor, daha fazla
hisse değil. `--step 1` ile ~220 görüntü üretilebilir.

**3. Topluluk her zaman kazanmıyor.** 21 günlük ufukta topluluk (−0.062)
üyelerinin hepsinden kötü. Sentetik testte topluluk üyelerini geçiyordu; burada
geçmiyor. Sebep: sentetik testte üyelerin hataları bağımsızdı, burada üçü de
aynı yöne (negatif) sapıyor. Bağımsız hata yapmayan tahmincileri ortalamak
ortak sapmayı yok etmez — **pekiştirir.** Topluluğun ICIR'i −3.34, yani iki
katmanda da tutarlı biçimde ters yönde.

Bu, topluluk özelliğinin gereksiz olduğu anlamına gelmiyor; terfi kapısının
işini yaptığı anlamına geliyor. Topluluk da diğerleri gibi bir **aday**, ve
diğerleri gibi reddedildi.

---

### Parametre bazlı ölçüm (17.08.2026)

Model seviyesinden bir kat aşağısı: **hangi parametre işe yarıyor?** Aynı 73
tarih, 188.777 etiketli satır, 21 günlük ufuk:

| Parametre | IC | ICIR |
| ----------- | ---: | -----: |
| momentum_persistence | +0.0262 | 0.25 |
| stage2_breakout | +0.0174 | 0.17 |
| trend_structure | +0.0157 | 0.16 |
| chart_position | +0.0127 | 0.14 |
| volume_accumulation | +0.0086 | 0.14 |
| risk_drawdown | +0.0031 | 0.02 |
| price_momentum_12_1 | −0.0027 | −0.03 |
| relative_strength | −0.0040 | −0.04 |
| technical_oscillators | −0.0054 | −0.06 |
| nominal_price_fit | −0.0173 | −0.17 |
| breakout_setup | −0.0234 | −0.22 |

Ölçüt |IC| > 0.03. **Hiçbiri geçmiyor.** İki gözlem:

1. **`breakout_setup` ters yönde ve config'in en ağır dört parametresinden
   biri.** Bu pencerede kırılım
   kurulumu güçlü olan hisseler, zayıf olanlardan daha kötü performans
   göstermiş. Tek bir 11 aylık pencere kanıt değildir, ama en yüksek ağırlıklı
   dördüncü parametre için not edilmesi gereken bir bulgu.
2. **`nominal_price_fit` de negatif.** Config'in kendi notu zaten "nominal
   fiyatın gelecek getiriyle istatistiksel ilişkisi yoktur" diyordu; ölçüm
   bunu doğruluyor. Ağırlığı zaten 1.0.

Bu tablo artık panoda. **Hiçbir ağırlık otomatik değiştirilmiyor** — ölçüm bir
öneridir, değişiklik kullanıcının kararıdır. Otomatik ağırlık güncellemesi,
11 aylık tek bir pencereye aşırı uydurma riski taşır.

Uyarı: bu ölçüm geçmişe dönük panelden yapıldı; hayatta kalma yanlılığı taşır
ve yalnızca fiyat türevi 11 parametreyi kapsar. Temel veri parametrelerinin IC
ölçümü, `data/fundamentals` arşivi birikince mümkün olacak.

---

## Güvenlik freni — sistemin en önemli parçası

Modelin skora etkisi, **ölçülen OOS becerisiyle orantılıdır.** Kanıt yoksa
ağırlık sıfırdır.

```python
suggested_weight(icir, ic):
    ic <= 0.02  →  0.0          # beceri yok, etki yok
    ICIR 0.35   →  0.9          # zayıf kanıt, küçük etki
    ICIR 1.20   →  12.0         # güçlü kanıt, üst sınır
```

Üst sınır **12** — 27 parametrenin toplamı ~137 olduğuna göre modelin payı en
fazla ~%8. Model, insan tarafından okunabilir parametrelerin **yerini almaz**;
onlara ek bir görüş katar.

Bu tasarım, *"model kurdum, artık ona güveniyorum"* hatasını yapısal olarak
imkânsız kılar.

### Terfi eşikleri

| Eşik | Değer | Neden |
|------|------:|-------|
| IC ortalaması | > 0.02 | Kantitatif finansta 0.03 zayıf-ama-kullanılabilir kabul edilir |
| ICIR | > 0.30 | Tutarlılık ölçüsü; altı gürültüden ayırt edilemez |
| Katman sayısı | ≥ 3 | Tek pencerede iyi sonuç şans olabilir |
| Pozitif katman oranı | ≥ %60 | Bazı dönemlerde çalışıp bazılarında çökmemeli |
| Taban çizgisini geçme | +0.005 IC | **Ridge'i geçemeyen derin model kullanılmaz** |

Son madde kritik: karmaşıklık bedava değildir. Basit bir ridge regresyonu kadar
iyi olan bir sinir ağı, daha fazla aşırı uyum riski ve daha az şeffaflık
demektir.

---

## Sızıntı önleme — burası en kolay hata yapılan yer

Finansal panel verisinde naif çapraz doğrulama, modeli **olduğundan iyi
gösterir.** Üç ayrı sebepten:

### 1. Örtüşen etiketler

21 günlük ileri getiri, ardışık günlerin etiketlerini ~%95 örtüştürür.
`t` günü için etiket `t+21`'e kadarki fiyatı kullanır. Eğitim setinde `t`,
test setinde `t+3` varsa, **eğitim etiketinin içinde test döneminin fiyatı
vardır** — doğrudan sızıntı.

**Çözüm — arındırma (purge):** Test başlangıcından geriye doğru `horizon`
kadar günlük eğitim verisi atılır.

### 2. Seri korelasyon

Arındırma sonrası bile sınır bölgesindeki örnekler benzer piyasa rejimini
paylaşır.

**Çözüm — embargo:** Arındırmanın üstüne ek tampon gün (varsayılan 5).

### 3. Geleceğe bakış

Özellikler o gün **bilinebilir** olmalı. Feature store her günün anlık
görüntüsünü ayrı sakladığı için bu yapısal olarak sağlanır.

### Bölmenin şeması

```
  eğitim ──────────────────┤  ARINDIRMA + EMBARGO  ├────── test ──────
                            └── horizon + 5 gün ──┘
```

Bu, `tests/test_ml.py` içinde otomatik doğrulanır: eğitim ve test tarihleri
kesişmemeli ve aralarında en az `horizon + embargo` gün olmalı.

---

## Modeller

Üç model artı bir topluluk. **Hepsi aynı arayüzü paylaşır.**

### RidgeRanker — taban çizgisi

Kapalı formül ridge regresyon, saf numpy. Bağımlılığı yok, saniyeler içinde
eğitilir.

**Rolü kritik:** Derin modeller bunu OOS'ta geçemiyorsa kullanılmaz.

### MLPRanker — çapraz kesitsel sinir ağı

İki katmanlı MLP (LayerNorm + GELU + Dropout). Parametreler arası **doğrusal
olmayan etkileşimleri** yakalar — örneğin *"ucuzluk yalnızca trend sağlamken
işe yarıyor"* gibi.

### SeqRanker — dizi modeli (GRU)

Girdi: her hisse için son 10 günün parametre dizisi.

**Derin öğrenmenin bu problemde gerçek katkı verdiği yer burasıdır.** Çapraz
kesitsel modeller yalnızca *"bugün neye benziyor"* sorusunu görür. Dizi modeli
*"son 10 günde nasıl değişti"* sorusunu da görür:

- Skorun yükselerek mi yoksa düşerek mi bu seviyeye geldiği
- Kırılım kurulumunun olgunlaşması
- Momentumun hızlanması veya sönmesi

**Bu iddia test edildi.** Sinyalin *yalnızca değişimde* olduğu sentetik bir
veri kümesinde:

```
çapraz kesitsel IC = 0.308
dizi modeli    IC = 0.757
```

Yani dizi modeli, çapraz kesitsel modelin **göremediği** örüntüyü yakalıyor.

#### Canlı çalışma — artık mümkün

Uzun süre bu modelin bir eksiği vardı: ölçülebiliyordu ama **kullanılamıyordu.**
`predict_live`, dizi modeli gördüğünde doğrudan `None` dönüyordu; çünkü tek bir
günün satırı dizi değildir.

Pencere artık feature store'dan kuruluyor: o sembolün önceki anlık görüntüleri +
bugünün canlı satırı. Sıralama **eğitimdekiyle birebir aynı** olmak zorunda —
önce gün içi çapraz kesitsel yüzdelik, sonra pencereleme. Ters çevirirsen model
eğitimde gördüğünden başka bir şey görür ve tahminleri sessizce anlamsızlaşır;
hiçbir hata mesajı almazsın.

Geçmişi yetersiz olan hisse için tahmin **üretilmez.** Kısa pencereyi aynı
satırın tekrarıyla doldurup uydurma bir dizi vermek mümkündü ama yanlış güven
üretirdi. O hisse parametreyi "eksik" olarak alır, kapsama mekanizması gerisini
halleder. Görüş yokken görüş uydurmamak doğrusu.

Performans notu: tüm depo değil, yalnızca son `window` kadar dosya okunur
(`ml.load_recent_snapshots`). Bir yıl biriktiğinde feature store yüzlerce MB
olur; her taramada baştan okumak taramaya dakikalar eklerdi.

### Topluluk — şampiyon-hepsini-alır neden yanlıştı

Üç model eğitiliyor, biri seçiliyor, ikisi çöpe gidiyordu. Oysa birbirinden
**bağımsız hatalar** yapan tahmincilerin ortalaması tek tek hepsinden daha
kararlıdır — ve terfi kapısındaki asıl zorluk IC'nin büyüklüğü değil
tutarlılığı (ICIR) olduğu için bu doğrudan işe yarar.

İki tasarım kararı önemli:

**Ham tahminler değil, YÜZDELİK SIRALAR harmanlanır.** Ridge'in çıktı ölçeği
ile sinir ağının ölçeği farklıdır; ham ortalama alınsaydı büyük ölçekli olan
diğerini ezerdi. Sıralar gün içinde hesaplanır, çünkü karşılaştırma çapraz
kesitseldir.

**Hizalama indeksle değil ANAHTARLA yapılır.** Dizi modeli, pencere kadar
geçmişi olmayan satırları düşürür; yani modeller aynı satır kümesi üzerinde
çalışmaz. İndeksle harmanlansaydı farklı hisselerin tahminleri toplanır ve
sonuç sessizce saçma olurdu. `(tarih, sembol)` ile hizalanır, yalnızca tüm
modellerde bulunan satırlar kullanılır.

Sentetik testte (iki bağımsız zayıf tahminci): topluluk IC 0.703, üyeler 0.600
ve 0.544. `tests/test_topluluk.py` bunu ve satır sırası değiştiğinde sonucun
değişmemesi gerektiğini doğrular.

Topluluk şampiyon olursa tek bir `.pkl` yoktur: kayıt defterinde `members`
listesi saklanır ve canlı tahmin üyeleri diskten tek tek yükleyip aynı yüzdelik
ortalamasını alır. **Eğitimde nasıl harmanlandıysa canlıda da aynen öyle** —
aksi halde ölçülen beceri ile üretilen tahmin aynı şeyin ölçümü olmaz.

### Modelin gördüğü şey — ham seri özellikleri

Model bugüne kadar yalnızca 28 skoru görüyordu. Yani **insan hipotezleriyle
sınırlıydı**: bir sinir ağından "insanın düşünmediği bir şey" öğrenmesini
beklemek, ona yalnızca insanın düşündüğü büyüklükleri vererek mümkün değil.

Artık her anlık görüntüye 20 ham büyüklük yazılıyor (`series_*`): yedi farklı
gecikmede getiri, 21 ve 63 günlük gerçekleşmiş oynaklık ve oranı, hacim
oranları, hareketli ortalamalara **ATR cinsinden** uzaklık, çarpıklık, basıklık,
düşüş ve 52 haftalık bant konumu.

ATR cinsinden uzaklık kullanmanın sebebi: yüzde uzaklık, farklı oynaklıktaki
hisseleri karşılaştırılamaz kılar. %10 uzaklık sakin bir hissede çok, dalgalı
bir hissede hiçbir şeydir.

Bunlar **parametre değildir**: skorlamaya girmez, ağırlıkları yoktur, panoda
görünmez. Yalnızca feature store'a yazılır ve eğitimde kullanılır.

### Çoklu ufuk

`--horizons 5,21,63` her ufku ayrı eğitir. Aynı veriden üç kat kanıt çıkar ama
asıl kazanç bu değil: **sinyalin ömrü** görünür hale gelir.

- Kısa ufukta güçlü, uzun ufukta zayıf → kısa vadeli etki (momentum, haber)
- Uzun ufukta güçlü, kısada zayıf → değerleme etkisi
- İkisinde de yok → sinyal yok

Bu ayrım, hangi parametrenin **neden** çalıştığını anlamanın en kestirme yolu.

---

## Kayıp fonksiyonu neden sıralama?

Hedef **sıralamadır, seviye değil.** Eğitim **gün bazlı** yapılır — her yığın
bir günün tüm hisseleridir. Sıralama ancak aynı gün içindeki hisseler arasında
anlamlıdır.

Ham getiriyi MSE ile kestirmek, birkaç aykırı hissenin modeli ele geçirmesine
yol açar: %300 sıçrayan tek bir hisse, kaybın büyük kısmını oluşturur ve model
o tek örneği açıklamaya çalışır.

### Düzeltme (04.09.2026) — "Spearman vekili" değildi

Kayıp, tahmin ile **ham getiri** arasındaki korelasyondu ve koda "Spearman'a
türevlenebilir vekil" diye yazılmıştı. Bu doğru değildi: ham getiriyle alınan
korelasyon **Pearson**'dur. Ortalama çıkarıp standart sapmaya bölmek **ölçeği**
düzeltir, **çarpıklığı** düzeltmez — %300 sıçrayan hisse standartlaştırmadan
sonra da +8 sigma'da durur ve o günün kaybının büyük kısmını tek başına
belirler. Yani modülün kaçınmak için yazıldığı sorun, hafifleyerek de olsa
duruyordu.

Şimdi **hedef gün içinde sıraya çevriliyor** (`_to_rank_tensor`, [-1, 1]
aralığına yayılmış). En iyi hisse artık "%300 yapan" değil, "1. sıradaki".

Tam Spearman değil ve olamaz: tahmin tarafını da sıralamak gerekirdi ama
`argsort` türevlenemez, gradyan ölürdü. Yani

> kayıp = −Pearson(tahmin, hedefin sırası)

Ölçülen fark gerçek Spearman'a ~0.02 (`tests/test_kayip.py`, 4. bölüm).
Aykırı değer sorunu ise tamamen çözülüyor: en yüksek getirili hissenin
getirisini %3'ten %300'e çıkarmak — sırası değişmediği için — sıra kaybını
**hiç** oynatmıyor (ölçülen fark 0.00e+00), aynı değişiklik Pearson kaybını
0.0534 oynatıyor.

Sentetik ağır kuyruklu hedefte üç tohum ortalaması: sıra kaybı IC +0.4634,
Pearson kaybı +0.4580. Küçük ama tutarlı. Gerçek panelde ölçüm ayrı.

Karşılaştırma yapılabilsin diye eski davranış duruyor: `models.RANK_TARGET`.

---

## Doğruluk kanıtı — bugün

Gerçek veri birikmesi ay alır, ama **boru hattının doğru olup olmadığını bugün
bilmemiz gerekiyor.** `tests/test_ml.py` sentetik veriyle 29 test çalıştırır:

```bash
python tests/test_ml.py
python tests/test_backfill.py
```

| Test | Sonuç |
|------|-------|
| Bilinen sinyali buluyor mu | IC = 0.605 ✓ |
| **Sinyalsiz veride sıfır** (aşırı uyum) | IC = 0.001 ✓ |
| Arındırma sızıntıyı engelliyor mu | Boşluk ≥ horizon+embargo ✓ |
| Terfi kapısı gürültüyü reddediyor mu | 6/6 senaryo ✓ |
| Dizi modeli zaman örüntüsünü görüyor mu | 0.757 vs 0.308 ✓ |
| Kanıt yoksa ağırlık sıfır mı | 0.0 ✓ |
| Uçtan uca döngü kapanıyor mu | Eğit→terfi→canlı tahmin ✓ |
| Dizi modeli **canlıda** çalışıyor mu | IC = 0.678 ✓ |
| Geçmişi olmayan hisse kapsam dışı mı | 0 sızıntı ✓ |
| Geçmişe dönük panel geleceği okuyor mu | Barlar ×3 → değişim yok ✓ |
| Ön eğitim şampiyon üretebiliyor mu | IC 0.25'te bile **hayır** ✓ |

İkinci satır en önemlisi: **sinyalsiz veride model sıfır buluyor.** Aşırı uyum
yapsaydı orada da yüksek IC çıkardı.

---

## Kullanım

```bash
# Durum — ne kadar veri var, model var mı
python run.py ml status

# Eğit ve değerlendir (terfi etmeden)
python run.py ml evaluate

# Eğit ve eşikleri geçeni şampiyon yap
python run.py ml train --promote

# Yalnızca belirli modeller (ridge her zaman dahil edilir)
python run.py ml train --models mlp,seq --promote

# Farklı ufuk
python run.py ml train --horizon 42 --promote

# Beklemeden mimari denemek: geçmişe dönük panel üret, üzerinde eğit
python run.py history
python run.py ml train --pretrain --models ridge,mlp,seq
```

Günlük döngü bunu **otomatik** yapar:

```bash
python run.py daily --universe smallcap,midcap,wsb --workers 4
```

3. adımda veri yeterliyse yeniden eğitir, değilse ilerlemeyi bildirir.
`--no-train` ile kapatılabilir, `--retrain-every N` ile sıklık ayarlanır.

---

## Dosya düzeni

```
src/dataset.py         panel kurulumu, sızıntısız bölme, dizi kurulumu, hazırlık kapısı
src/backfill.py        önbellekteki fiyattan geçmişe dönük anlık görüntü üretimi
src/models.py          RidgeRanker · MLPRanker · SeqRanker (ortak arayüz)
src/training.py        ileri yürüyüşlü eğitim, değerlendirme, terfi kapısı, canlı tahmin
data/feature_store/    gerçek (ileriye dönük) anlık görüntüler — şampiyon buradan çıkar
data/backfill_store/   geçmişe dönük panel — yalnızca ön eğitim, şampiyon çıkamaz
data/models/           eğitilmiş modeller + registry.json (şampiyon/aday geçmişi)
tests/test_ml.py       29 sentetik doğruluk testi
tests/test_backfill.py 34 geçmişe dönük panel testi
```

---

## Sınırlar — dürüst liste

| Sınır | Etkisi |
|-------|--------|
| **Hayatta kalan yanlılığı** | Evren bugün kote olanlardan; batmış şirketler eğitim setinde yok → sonuçlar yukarı yanlı olacak |
| **Rejim değişimi** | Model geçmiş rejimi öğrenir; piyasa yapısı değişirse bozulur. ICIR izlemesi bunu yakalar ama gecikmeyle |
| **Küçük veri** | 2400 hisse × N gün, derin öğrenme ölçeğinde küçüktür. Bu yüzden modeller kasıtlı olarak küçük tutuldu (64 gizli birim) |
| **Dizi modeli canlıda kapalı** | Geçmiş pencere gerektirir; yeterli anlık görüntü biriktiğinde açılacak |
| **Tek varlık sınıfı** | Yalnızca ABD hisseleri |

---

## Ne zaman güvenilir olur

| Aşama | Gereken | Ne olur |
|-------|---------|---------|
| Eğitim mümkün | 30 anlık görüntü, 41 gün | `ml train` çalışır ama sonuçlar zayıf |
| **Doğrulama anlamlı** | **60 anlık görüntü, 120 gün** | İleri yürüyüş katmanları bağımsızlaşır |
| Güvenilir | 120+ görüntü, 250+ gün | Birden fazla piyasa rejimi görülmüş olur |

Şu anki ilerleme panonun en üstünde ve `python run.py ml status` çıktısında.

> Bu sistem yatırım tavsiyesi vermez. Model çıktısı da dahil olmak üzere tüm
> skorlar istatistiksel göstergelerdir.
