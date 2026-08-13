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

Üç model, artan karmaşıklık sırasıyla. **Hepsi aynı arayüzü paylaşır.**

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

> **Not:** Dizi modeli şu an yalnızca değerlendirmede kullanılır. Canlı tahmin
> için her hissenin geçmiş penceresi gerekir; bu, yeterli anlık görüntü
> biriktiğinde etkinleşecek.

---

## Kayıp fonksiyonu neden sıralama?

Hedef **sıralamadır, seviye değil.** Kayıp fonksiyonu çapraz kesitsel
sıralamayı optimize eder (Spearman'a türevlenebilir vekil).

Ham getiriyi MSE ile kestirmek, birkaç aykırı hissenin modeli ele geçirmesine
yol açar: %300 sıçrayan tek bir hisse, kaybın büyük kısmını oluşturur ve model
o tek örneği açıklamaya çalışır. Sıralama kaybı bundan etkilenmez.

Eğitim **gün bazlı** yapılır — her yığın bir günün tüm hisseleridir. Sıralama
ancak aynı gün içindeki hisseler arasında anlamlıdır.

---

## Doğruluk kanıtı — bugün

Gerçek veri birikmesi ay alır, ama **boru hattının doğru olup olmadığını bugün
bilmemiz gerekiyor.** `tests/test_ml.py` sentetik veriyle 22 test çalıştırır:

```bash
python tests/test_ml.py
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
src/dataset.py    panel kurulumu, sızıntısız bölme, dizi kurulumu, hazırlık kapısı
src/models.py     RidgeRanker · MLPRanker · SeqRanker (ortak arayüz)
src/training.py   ileri yürüyüşlü eğitim, değerlendirme, terfi kapısı, canlı tahmin
data/models/      eğitilmiş modeller + registry.json (şampiyon/aday geçmişi)
tests/test_ml.py  22 sentetik doğruluk testi
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
