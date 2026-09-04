"""Model havuzu — hepsi ayni arayuzu paylasir: fit / predict / save / load.

Dort model, artan karmasiklik sirasiyla:

  RidgeRanker : numpy-only, her zaman calisir. TABAN CIZGISI.
                Karmasik modeller bunu gecemiyorsa kullanilmaz.
  MLPRanker   : torch, capraz kesitsel. Dogrusal olmayan etkilesimleri yakalar.
  SeqRanker   : torch GRU, DIZI. "Son 10 gunde nasil degisti" sorusunu gorur —
                derin ogrenmenin bu problemde gercek katki verdigi yer.
  AttnRanker  : torch dikkat (attention), KUME. Gunun tum hisselerini birlikte
                gorur: "bu hisse, bugunku alternatiflere GORE nerede?" Ilk uc
                model her hisseyi tek basina puanlar; bu, goreliligi kayip
                fonksiyonuna birakmak yerine mimariye koyar.

Ortak tasarim kararlari
-----------------------
* HEDEF SIRALAMADIR, seviye degil. Kayip fonksiyonu capraz kesitsel siralamayi
  optimize eder (Spearman'a vekil). Ham getiriyi MSE ile kestirmek, birkac
  aykiri hissenin modeli ele gecirmesine yol acar.
* Egitim GUN BAZLI yapilir: her yigin bir gunun tum hisseleridir. Boylece
  siralama kaybi anlamli sekilde hesaplanabilir.
* Erken durdurma her zaman ZAMANSAL bir dogrulama diliminde yapilir.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

try:                                    # torch istege bagli
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                       # pragma: no cover
    _TORCH = False


# =============================================================================
#  Ortak yardimcilar
# =============================================================================
def _day_batches(dates: np.ndarray) -> list[np.ndarray]:
    """Gun bazli indeks yiginlari (capraz kesitsel kayip icin)."""
    out = []
    for d in np.unique(dates):
        idx = np.flatnonzero(dates == d)
        if len(idx) >= 8:               # siralama icin anlamli en az buyukluk
            out.append(idx)
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman korelasyonu (scipy'siz)."""
    if len(a) < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 1e-12 else float("nan")


class BaseModel:
    name = "base"
    needs_sequence = False

    def fit(self, X, y, dates, val=None) -> dict:      # pragma: no cover
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:                # pragma: no cover
        raise NotImplementedError

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "BaseModel":
        with path.open("rb") as fh:
            return pickle.load(fh)


# =============================================================================
#  1) Taban cizgisi — Ridge (numpy)
# =============================================================================
class RidgeRanker(BaseModel):
    """Kapali formul ridge regresyon. Bagimliligi yok, saniyeler icinde egitilir.

    TABAN CIZGISI ROLU: Derin modeller bunu OOS'ta gecemiyorsa, ek karmasiklik
    bedava degil zararlidir — daha fazla asiri uyum riski, daha az seffaflik.
    """
    name = "ridge"

    def __init__(self, alpha: float = 10.0):
        self.alpha = float(alpha)
        self.w: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, X, y, dates, val=None) -> dict:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd

        # Hedefi gun icinde standartlastir -> siralamaya vekil
        yy = y.copy()
        for d in np.unique(dates):
            m = dates == d
            s = yy[m].std()
            yy[m] = (yy[m] - yy[m].mean()) / (s if s > 1e-12 else 1.0)

        n_f = Z.shape[1]
        A = Z.T @ Z + self.alpha * np.eye(n_f)
        self.w = np.linalg.solve(A, Z.T @ yy)
        return {"model": self.name, "features": n_f, "rows": len(y)}

    def predict(self, X) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        return (Z @ self.w).astype(np.float32)


# =============================================================================
#  Torch ortak egitim dongusu
# =============================================================================
# Kayip fonksiyonunda hedefin SIRAYA cevrilip cevrilmeyecegi.
# True: Spearman vekili (varsayilan)  ·  False: eski Pearson vekili.
# Karsilastirma yapabilmek icin sabit olarak duruyor (bkz. tests/test_kayip.py).
RANK_TARGET = True


def _to_rank_tensor(t: "torch.Tensor") -> "torch.Tensor":
    """Bir gunun hedeflerini [-1, 1] araligina yayilmis SIRALARA cevirir.

    Hedef parametrelerden bagimsiz oldugu icin burada turev sorunu yok;
    argsort'un turevlenemez olmasi onemli degil.
    """
    n = t.numel()
    if n < 2:
        return t * 0
    r = torch.empty(n, dtype=t.dtype, device=t.device)
    r[t.argsort()] = torch.arange(n, dtype=t.dtype, device=t.device)
    return (r / (n - 1)) * 2.0 - 1.0


def _rank_loss(pred: "torch.Tensor", target: "torch.Tensor",
               rank_target: bool | None = None) -> "torch.Tensor":
    """Capraz kesitsel siralama kaybi.

    Gun icinde hem tahmini hem hedefi standartlastirip negatif korelasyonu
    dondurur.

    HEDEF SIRAYA CEVRILIR (04.09.2026). Onceki surum ham getiriyle korelasyon
    aliyordu ve buna "Spearman vekili" diyordu; oysa ham getiriyle alinan
    korelasyon PEARSON'dur. Ortalama cikarip standart sapmaya bolmek olcegi
    duzeltir, CARPIKLIGI duzeltmez: %300 sicrayan tek bir hisse
    standartlastirmadan sonra da ~+8 sigma'da durur ve o gunun kaybinin buyuk
    kismini tek basina belirler. Modul zaten "aykiri getiriler modeli ele
    gecirmesin" diye siralama kaybi kullaniyordu; hedefi siraya cevirmek o
    amaci gercekten yerine getiriyor -- en iyi hisse artik "1. sirada", "%300
    yapan" degil.

    TAM SPEARMAN DEGIL: yalnizca HEDEF siraya cevrilir, tahmin tarafi surekli
    kalir. Tahmini de siralamak gerekirdi ama argsort turevlenemez, gradyan
    olurdu. Yani kayip = -Pearson(tahmin, hedefin sirasi). Aykiri getiri
    sorununu tamamen cozer (hedef tarafinda aykiri deger kalmaz), gercek
    Spearman'a da yakin durur -- olculen fark ~0.02 (tests/test_kayip.py).

    Bu, siralama ogrenmesi yazininin capraz kesitsel hisse siralamasinda
    noktasal (MSE/MAE) kayiplar yerine siralama-farkindali kayiplari onermesiyle
    de ayni yonde.
    """
    if rank_target is None:
        rank_target = RANK_TARGET
    tgt = _to_rank_tensor(target) if rank_target else target
    p = pred - pred.mean()
    t = tgt - tgt.mean()
    # unbiased=False: kovaryans .mean() ile (n'e bolerek) hesaplaniyor; standart
    # sapma da ayni bolenle alinmazsa sonuc (n-1)/n kadar kayar. Gunler farkli
    # buyuklukte oldugu icin bu kayma her gun biraz farkli olur -- kucuk ama
    # gereksiz bir tutarsizlik. Bu haliyle kayip tam olarak
    # -Pearson(tahmin, hedefin sirasi).
    ps, ts = p.std(unbiased=False), t.std(unbiased=False)
    if ps < 1e-8 or ts < 1e-8:
        return (p * 0).sum()
    return -(p * t).mean() / (ps * ts)


def _train_torch(model, batches, X_t, y_t, val_batches, val_X, val_y,
                 epochs: int, lr: float, weight_decay: float,
                 patience: int, seed: int) -> dict:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_ic, bad = None, -np.inf, 0
    history = []

    for ep in range(epochs):
        model.train()
        np.random.shuffle(batches)
        tot = 0.0
        for idx in batches:
            opt.zero_grad()
            out = model(X_t[idx]).squeeze(-1)
            loss = _rank_loss(out, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())

        # --- dogrulama: gun bazli ortalama IC
        model.eval()
        with torch.no_grad():
            ics = []
            for idx in val_batches:
                p = model(val_X[idx]).squeeze(-1).cpu().numpy()
                ic = spearman(p, val_y[idx])
                if np.isfinite(ic):
                    ics.append(ic)
            val_ic = float(np.mean(ics)) if ics else float("nan")

        history.append({"epoch": ep + 1,
                        "train_loss": round(tot / max(1, len(batches)), 5),
                        "val_ic": None if not np.isfinite(val_ic) else round(val_ic, 5)})

        if np.isfinite(val_ic) and val_ic > best_ic + 1e-5:
            best_ic, bad = val_ic, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs_run": len(history), "best_val_ic": None if not np.isfinite(best_ic)
            else round(float(best_ic), 5), "history": history[-12:]}


# =============================================================================
#  2) MLP — capraz kesitsel
# =============================================================================
class MLPRanker(BaseModel):
    """Iki katmanli MLP. Parametreler arasi dogrusal olmayan etkilesimleri
    yakalar (orn. "ucuzluk yalnizca trend saglamken ise yariyor")."""
    name = "mlp"

    def __init__(self, hidden: int = 64, dropout: float = 0.3, lr: float = 1e-3,
                 epochs: int = 200, patience: int = 20, weight_decay: float = 1e-3,
                 seed: int = 7):
        if not _TORCH:
            raise ImportError("torch kurulu degil — MLPRanker kullanilamaz")
        self.cfg = dict(hidden=hidden, dropout=dropout, lr=lr, epochs=epochs,
                        patience=patience, weight_decay=weight_decay, seed=seed)
        self.net = None
        self.mu = self.sd = None

    def _build(self, n_f: int):
        h = self.cfg["hidden"]
        return nn.Sequential(
            nn.Linear(n_f, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(self.cfg["dropout"]),
            nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.GELU(),
            nn.Dropout(self.cfg["dropout"]),
            nn.Linear(h // 2, 1),
        )

    def fit(self, X, y, dates, val=None) -> dict:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd

        if val is None:                       # son %20 tarih dogrulamaya
            uniq = np.array(sorted(set(dates)))
            cut = uniq[int(len(uniq) * 0.8)] if len(uniq) > 4 else uniq[-1]
            vm = dates >= cut
        else:
            vm = val
        tm = ~vm

        self.net = self._build(Z.shape[1])
        X_t = torch.tensor(Z[tm]); y_t = torch.tensor(y[tm])
        vX = torch.tensor(Z[vm]); vy = y[vm]
        info = _train_torch(self.net, _day_batches(dates[tm]), X_t, y_t,
                            _day_batches(dates[vm]), vX, vy,
                            self.cfg["epochs"], self.cfg["lr"],
                            self.cfg["weight_decay"], self.cfg["patience"],
                            self.cfg["seed"])
        info.update({"model": self.name, "features": Z.shape[1], "rows": int(tm.sum())})
        return info

    def predict(self, X) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float32) - self.mu) / self.sd
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(Z)).squeeze(-1).cpu().numpy()

    def __getstate__(self):
        s = self.__dict__.copy()
        s["net"] = None if self.net is None else self.net.state_dict()
        s["_n_f"] = None if self.mu is None else len(self.mu)
        return s

    def __setstate__(self, s):
        state = s.pop("net"); n_f = s.pop("_n_f", None)
        self.__dict__.update(s)
        self.net = None
        if state is not None and n_f:
            self.net = self._build(n_f)
            self.net.load_state_dict(state)
            self.net.eval()


# =============================================================================
#  3) GRU — dizi modeli
# =============================================================================
class SeqRanker(BaseModel):
    """GRU tabanli dizi modeli.

    Girdi: her hisse icin son `window` gunun ozellik dizisi.
    Bu model, capraz kesitsel modellerin GOREMEDIGI seyi gorur: parametrelerin
    ZAMAN ICINDEKI HAREKETI. Ornegin skorun son bir haftada yukselmesi ile
    ayni skora dusarak gelmis olmak cok farkli seylerdir.
    """
    name = "seq"
    needs_sequence = True

    def __init__(self, hidden: int = 48, layers: int = 1, dropout: float = 0.25,
                 lr: float = 1e-3, epochs: int = 150, patience: int = 15,
                 weight_decay: float = 1e-3, window: int = 10, seed: int = 7):
        if not _TORCH:
            raise ImportError("torch kurulu degil — SeqRanker kullanilamaz")
        self.cfg = dict(hidden=hidden, layers=layers, dropout=dropout, lr=lr,
                        epochs=epochs, patience=patience, weight_decay=weight_decay,
                        window=window, seed=seed)
        self.net = None
        self.mu = self.sd = None

    def _build(self, n_f: int):
        c = self.cfg

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(n_f, c["hidden"], num_layers=c["layers"],
                                  batch_first=True,
                                  dropout=c["dropout"] if c["layers"] > 1 else 0.0)
                self.head = nn.Sequential(
                    nn.LayerNorm(c["hidden"]), nn.Dropout(c["dropout"]),
                    nn.Linear(c["hidden"], 1))

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1])       # son adimin gizli durumu

        return Net()

    def fit(self, X, y, dates, val=None) -> dict:
        X = np.asarray(X, dtype=np.float32)       # (n, window, f)
        y = np.asarray(y, dtype=np.float32)
        flat = X.reshape(-1, X.shape[-1])
        self.mu, self.sd = flat.mean(0), flat.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd

        if val is None:
            uniq = np.array(sorted(set(dates)))
            cut = uniq[int(len(uniq) * 0.8)] if len(uniq) > 4 else uniq[-1]
            vm = dates >= cut
        else:
            vm = val
        tm = ~vm

        self.net = self._build(Z.shape[-1])
        info = _train_torch(self.net, _day_batches(dates[tm]),
                            torch.tensor(Z[tm]), torch.tensor(y[tm]),
                            _day_batches(dates[vm]), torch.tensor(Z[vm]), y[vm],
                            self.cfg["epochs"], self.cfg["lr"],
                            self.cfg["weight_decay"], self.cfg["patience"],
                            self.cfg["seed"])
        info.update({"model": self.name, "features": Z.shape[-1],
                     "window": self.cfg["window"], "rows": int(tm.sum())})
        return info

    def predict(self, X) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float32) - self.mu) / self.sd
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(Z)).squeeze(-1).cpu().numpy()

    def __getstate__(self):
        s = self.__dict__.copy()
        s["net"] = None if self.net is None else self.net.state_dict()
        s["_n_f"] = None if self.mu is None else len(self.mu)
        return s

    def __setstate__(self, s):
        state = s.pop("net"); n_f = s.pop("_n_f", None)
        self.__dict__.update(s)
        self.net = None
        if state is not None and n_f:
            self.net = self._build(n_f)
            self.net.load_state_dict(state)
            self.net.eval()


# =============================================================================
#  4) Capraz kesitsel dikkat (attention) — gunun TUM hisselerini birlikte gorur
# =============================================================================
class AttnRanker(BaseModel):
    """Bir gunun hisselerini KUME olarak isleyen dikkat modeli.

    NEDEN BU MIMARI
    ---------------
    Ridge, MLP ve GRU'nun ucu de her hisseyi TEK BASINA puanliyor; capraz
    kesitsellik yalnizca kayip fonksiyonunda var. Oysa siralama dogasi geregi
    GORECELIDIR: "momentumu %8" bilgisi tek basina bir sey ifade etmez, o gun
    piyasadaki diger hisselerin momentumu %2 ise guclu, %20 ise zayiftir.

    Bu model gunun hisselerini bir KUME olarak alir ve her hissenin gosterimini
    ayni gunun digerlerine bakarak gunceller (self-attention). Yani "bugun bu
    hisse, bugunku alternatiflere GORE ne durumda" sorusunu mimarinin kendisi
    soruyor -- sonradan normalizasyonla degil.

    PERMUTASYON ESDEGERLIGI
    Konum kodlamasi (positional encoding) BILEREK YOK. Bir gunun hisseleri
    sirali bir dizi degil, sirasiz bir kumedir; hisseleri karistirmak ciktiyi
    ayni sekilde karistirmali, degistirmemeli. Konum kodlamasi eklemek modele
    anlamsiz bir sira bilgisi ogretirdi. `test_attn.py` bunu dogruluyor.

    MALIYET
    Dikkat, gun buyuklugunde O(n^2). Egitimde gunler `max_set` buyuklugunde
    parcalara bolunur: hem hesabi sinirlar hem de her parca gunun rastgele bir
    alt kumesi oldugu icin duzenlilestirme (regularization) etkisi yapar.
    Cikarimda gun butun halinde islenir (~2800 hisse, tek ileri gecis).

    UYARI: Bu model taban cizgisini gecmek ZORUNDA degil. Terfi kapisi
    (promote) digerleriyle ayni; kanit yoksa skorlamaya girmez.
    """
    name = "attn"
    needs_sequence = False
    needs_dates = True          # predict gun sinirlarini bilmek zorunda

    def __init__(self, d_model: int = 64, heads: int = 4, layers: int = 2,
                 dropout: float = 0.2, lr: float = 8e-4, epochs: int = 150,
                 patience: int = 15, weight_decay: float = 1e-3,
                 max_set: int = 256, seed: int = 11):
        if not _TORCH:
            raise ImportError("torch kurulu degil — AttnRanker kullanilamaz")
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) heads'e ({heads}) bolunmeli")
        self.cfg = dict(d_model=d_model, heads=heads, layers=layers,
                        dropout=dropout, lr=lr, epochs=epochs, patience=patience,
                        weight_decay=weight_decay, max_set=max_set, seed=seed)
        self.net = None
        self.mu = self.sd = None

    # -- ag ----------------------------------------------------------------
    def _build(self, n_f: int):
        c = self.cfg

        class Block(nn.Module):
            """On-normlu (pre-norm) dikkat + ileri besleme bloku."""
            def __init__(self):
                super().__init__()
                self.n1 = nn.LayerNorm(c["d_model"])
                self.att = nn.MultiheadAttention(c["d_model"], c["heads"],
                                                 dropout=c["dropout"],
                                                 batch_first=True)
                self.n2 = nn.LayerNorm(c["d_model"])
                self.ff = nn.Sequential(
                    nn.Linear(c["d_model"], c["d_model"] * 2), nn.GELU(),
                    nn.Dropout(c["dropout"]),
                    nn.Linear(c["d_model"] * 2, c["d_model"]))
                self.drop = nn.Dropout(c["dropout"])

            def forward(self, h):
                q = self.n1(h)
                a, _ = self.att(q, q, q, need_weights=False)
                h = h + self.drop(a)
                h = h + self.drop(self.ff(self.n2(h)))
                return h

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.inp = nn.Sequential(
                    nn.Linear(n_f, c["d_model"]), nn.GELU(),
                    nn.LayerNorm(c["d_model"]))
                self.blocks = nn.ModuleList([Block() for _ in range(c["layers"])])
                self.head = nn.Sequential(
                    nn.LayerNorm(c["d_model"]), nn.Dropout(c["dropout"]),
                    nn.Linear(c["d_model"], 1))

            def forward(self, x):
                # x: (n, f) tek gunun capraz kesiti -> (n, 1)
                # KONUM KODLAMASI YOK: kume, dizi degil.
                flat = (x.dim() == 2)
                if flat:
                    x = x.unsqueeze(0)              # (1, n, f)
                h = self.inp(x)
                for b in self.blocks:
                    h = b(h)
                out = self.head(h)                  # (1, n, 1)
                return out.squeeze(0) if flat else out

        return Net()

    # -- egitim ------------------------------------------------------------
    def _capped_batches(self, dates: np.ndarray, rng: np.random.Generator
                        ) -> list[np.ndarray]:
        """Gun yiginlarini `max_set` buyuklugunde parcalara boler.

        Dikkat O(n^2) oldugu icin 2000 hisselik bir gun tek yigin olarak agir.
        Parcalar gunun rastgele alt kumeleri; her biri gecerli bir capraz kesit
        ornegi. Bolme her cagrida yeniden karistirilir.
        """
        cap = int(self.cfg["max_set"])
        out: list[np.ndarray] = []
        for d in np.unique(dates):
            idx = np.flatnonzero(dates == d)
            if len(idx) < 8:                        # siralama icin cok kucuk
                continue
            rng.shuffle(idx)
            for s in range(0, len(idx), cap):
                part = idx[s:s + cap]
                if len(part) >= 8:
                    out.append(part)
        return out

    def fit(self, X, y, dates, val=None) -> dict:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd

        # Dogrulama dilimi ZAMANSAL: son %20 tarih.
        if val is None:
            uniq = np.array(sorted(set(dates)))
            cut = uniq[int(len(uniq) * 0.8)] if len(uniq) > 4 else uniq[-1]
            vm = dates >= cut
        else:
            vm = val
        tm = ~vm

        rng = np.random.default_rng(self.cfg["seed"])
        self.net = self._build(Z.shape[1])
        info = _train_torch(self.net,
                            self._capped_batches(dates[tm], rng),
                            torch.tensor(Z[tm]), torch.tensor(y[tm]),
                            self._capped_batches(dates[vm], rng),
                            torch.tensor(Z[vm]), y[vm],
                            self.cfg["epochs"], self.cfg["lr"],
                            self.cfg["weight_decay"], self.cfg["patience"],
                            self.cfg["seed"])
        info.update({"model": self.name, "features": Z.shape[1],
                     "rows": int(tm.sum()), "max_set": self.cfg["max_set"],
                     "params": sum(p.numel() for p in self.net.parameters())})
        return info

    # -- cikarim -----------------------------------------------------------
    def predict(self, X, dates=None) -> np.ndarray:
        """Gun bazli tahmin.

        `dates` verilmezse TUM girdi tek bir gun kabul edilir. Canli tahmin
        yollari zaten tek gunun satirlarini veriyor, orada dogru. Coklu tarih
        iceren bir matris dates olmadan verilirse gunler birbirine karisir --
        bu yuzden walk_forward `needs_dates` bayragina bakip tarihleri geciyor.
        """
        Z = ((np.asarray(X, dtype=np.float32) - self.mu) / self.sd)
        self.net.eval()
        out = np.zeros(len(Z), dtype=np.float32)
        with torch.no_grad():
            if dates is None:
                return self.net(torch.tensor(Z)).squeeze(-1).cpu().numpy()
            dates = np.asarray(dates)
            for d in np.unique(dates):
                m = dates == d
                out[m] = self.net(torch.tensor(Z[m])).squeeze(-1).cpu().numpy()
        return out

    # -- kalicilik ---------------------------------------------------------
    def __getstate__(self):
        s = self.__dict__.copy()
        s["net"] = None if self.net is None else self.net.state_dict()
        s["_n_f"] = None if self.mu is None else len(self.mu)
        return s

    def __setstate__(self, s):
        state = s.pop("net"); n_f = s.pop("_n_f", None)
        self.__dict__.update(s)
        self.net = None
        if state is not None and n_f:
            self.net = self._build(n_f)
            self.net.load_state_dict(state)
            self.net.eval()


# =============================================================================
#  Kayit defteri
# =============================================================================
AVAILABLE: dict[str, Any] = {"ridge": RidgeRanker}
if _TORCH:
    AVAILABLE["mlp"] = MLPRanker
    AVAILABLE["seq"] = SeqRanker
    AVAILABLE["attn"] = AttnRanker


def torch_available() -> bool:
    return _TORCH


def describe() -> dict:
    return {
        "torch": _TORCH,
        "torch_version": (torch.__version__ if _TORCH else None),
        "models": sorted(AVAILABLE),
    }
