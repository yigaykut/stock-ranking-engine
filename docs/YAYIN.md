# GitHub Pages Yayını — Kurulum

Sistem hazır. Kalan tek adım GitHub kimlik doğrulaması — bunu **sen** yapmalısın,
parolan/token'ın bende olmamalı.

## 1. Giriş yap (bir kez)

```bash
gh auth login
```

Sırayla seç: **GitHub.com** → **HTTPS** → **Y** (git kimliği için kullan) →
**Login with a web browser** → çıkan kodu tarayıcıya gir.

## 2. Depoyu oluştur (bir kez)

```bash
gh repo create hisse-pano --public --description "Sifreli hisse siralama panosu"
```

> **Public olması şart:** GitHub Pages ücretsiz planda yalnızca public repoda
> çalışır. İçeriğin AES-256 şifreli olduğu için sorun değil — parolasız
> açılmıyor.

## 3. Parolayı ayarla (bir kez)

```bash
setx DASHBOARD_PASSWORD "en-az-5-rastgele-kelime-sec"
setx HISSE_REPO "yigaykut/hisse-pano"
```

Yeni bir terminal aç (setx mevcut oturumu etkilemez).

## 4. İlk yayın

```bash
python run.py publish     # şifreli sürümü üret
python run.py deploy --repo yigaykut/hisse-pano
```

## 5. GitHub Pages'i aç (bir kez)

Depo sayfası → **Settings** → **Pages** → Source: **Deploy from a branch** →
Branch: **main** / **(root)** → **Save**

Birkaç dakika sonra adresin hazır:

```
https://yigaykut.github.io/hisse-pano/
```

## Bundan sonra

Hiçbir şey yapman gerekmiyor. `HISSE_REPO` ve `DASHBOARD_PASSWORD` ayarlıysa
günlük görev (07:00) taramayı yapar, şifreler ve siteyi otomatik günceller.

---

## Sızıntı koruması

Yayın **proje dizininden değil**, ayrı bir `publish/` dizininden yapılır.
Oraya yalnızca şifreli dosyalar kopyalanır — düz metin oraya hiç girmez.

Üstüne gönderim öncesi her dosya taranır. Şu durumlarda yayın **durdurulur**:

| Kontrol | Davranış |
|---|---|
| Şifreleme belirteci yok | Durdurulur |
| Düz metin izi bulundu (`const DATA`, ticker adları, "Toplam Etki Puani"…) | Durdurulur |
| PBKDF2 turu < 100.000 | Durdurulur |

Test edildi: düz metin pano reddediliyor, kurcalanmış dosya yakalanıyor,
zayıflatılmış şifreleme reddediliyor.

Ayrıca proje kökünde `.gitignore` var — `data/`, `logs/`, düz metin panolar ve
`watchlist.json` (pozisyonların) hiçbir koşulda gönderilemez.

## Repoya ne gider

```
yigaykut/hisse-pano  (public)
├── index.html          ← AES-256-GCM şifreli pano
├── watchlist.html      ← AES-256-GCM şifreli izleme listesi
├── .nojekyll
└── yayin_bilgisi.json  ← yalnızca tarih ve şifreleme bilgisi
```

**Gitmeyenler:** düz metin pano, `data/` (önbellek, geçmiş, anlık görüntüler),
`watchlist.json` (pozisyonların), `logs/`, kaynak kod.
