"""Yedekleme — yeniden uretilemeyen tek varligin korunmasi.

NE YEDEKLENIR
-------------
Kodun tamami git'te; kaybolsa yeniden yazilir. Onbellek de yeniden cekilebilir.
Ama SU UC SEY yeniden uretilemez:

  data/feature_store  : her gunun anlik goruntusu. Gecmise donuk uretilemez --
                        Yahoo dunun temel verisini vermiyor. Aralik'ta
                        dolacak sayacin biriktirdigi seyin tamami burada.
  data/fundamentals   : gunluk temel veri arsivi. Ayni sebep.
  data/paper          : kagit uzerinde defter (sistemin karnesi).
  data/models         : egitilmis modeller + terfi kayit defteri.

Bunlar OneDrive icinde duruyor, yani senkron var. Ama senkron yedek DEGILDIR:
sessiz bir bozulma, yanlislikla silme veya bir betigin uzerine yazmasi aninda
tum kopyalara yayilir. Bu komut, o ana kadarki hali AYRI ve SIFRELI bir
arsivde dondurur.

SIFRELEME
---------
Pano yayiniyla ayni yontem: PBKDF2-SHA256 (600.000 tur) + AES-256-GCM. Parola
ayni ortam degiskeninden (DASHBOARD_PASSWORD) okunur veya sorulur; hicbir yere
YAZILMAZ. Parola kaybedilirse arsiv acilamaz -- baska bir kurtarma yolu yoktur.
"""
from __future__ import annotations

import io
import json
import secrets
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .publish import NONCE_BYTES, PBKDF2_ITERATIONS, SALT_BYTES, _derive

ROOT = Path(__file__).resolve().parents[1]
BACKUP_DIR = ROOT / "yedek"

# (yol, zorunlu mu)
SOURCES: tuple[tuple[str, bool], ...] = (
    ("data/feature_store", True),
    ("data/fundamentals", False),
    ("data/paper", False),
    ("data/models", False),
    ("data/scan_log.json", False),
    ("data/universe_history.json", False),
    ("data/kote_disi.json", False),
    ("data/rejim_gecmisi.json", False),
    ("data/faktor_ic.json", False),
    ("data/watchlist.json", False),
    ("data/watch_history.csv", False),
    ("config/weights.yaml", False),
)

MAGIC = b"HSYEDEK1"


def _collect() -> tuple[bytes, dict]:
    """Kaynaklari tek bir zip'e toplar. Doner: (zip baytlari, ozet)."""
    buf = io.BytesIO()
    summary: dict = {"files": 0, "bytes": 0, "parts": [], "missing": []}

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel, required in SOURCES:
            src = ROOT / rel
            if not src.exists():
                (summary["missing"]).append(rel)
                if required:
                    raise FileNotFoundError(
                        f"Yedeklenecek zorunlu kaynak yok: {rel}")
                continue

            files = [src] if src.is_file() else sorted(
                p for p in src.rglob("*") if p.is_file())
            n = size = 0
            for p in files:
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
                size += p.stat().st_size
            summary["files"] += n
            summary["bytes"] += size
            summary["parts"].append({"path": rel, "files": n,
                                     "mb": round(size / 1024 / 1024, 2)})
    return buf.getvalue(), summary


def create(password: str, out_dir: Path | None = None,
           label: str = "") -> dict:
    """Sifreli yedek arsivi olusturur.

    Dosya duzeni (hepsi tek dosyada, kendi kendini tanimlar):
        MAGIC(8) | baslik uzunlugu(4, big-endian) | JSON baslik | sifreli govde
    """
    raw, summary = _collect()

    salt = secrets.token_bytes(SALT_BYTES)
    nonce = secrets.token_bytes(NONCE_BYTES)
    key = _derive(password, salt)
    body = AESGCM(key).encrypt(nonce, raw, None)

    header = {
        "v": 1,
        "kdf": "PBKDF2-SHA256",
        "iter": PBKDF2_ITERATIONS,
        "cipher": "AES-256-GCM",
        "salt": salt.hex(),
        "nonce": nonce.hex(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plain_bytes": len(raw),
        "summary": summary,
    }
    hb = json.dumps(header, ensure_ascii=False).encode("utf-8")

    out_dir = out_dir or BACKUP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"yedek_{stamp}{('_' + label) if label else ''}.hsy"
    path = out_dir / name

    with path.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(len(hb).to_bytes(4, "big"))
        fh.write(hb)
        fh.write(body)

    return {"ok": True, "path": str(path),
            "mb": round(path.stat().st_size / 1024 / 1024, 2),
            "plain_mb": round(len(raw) / 1024 / 1024, 2),
            "summary": summary}


def inspect(path: Path) -> dict:
    """Arsivin basligini okur — PAROLA GEREKTIRMEZ, govdeyi acmaz."""
    with path.open("rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            return {"ok": False, "reason": "bu bir yedek dosyasi degil"}
        n = int.from_bytes(fh.read(4), "big")
        header = json.loads(fh.read(n).decode("utf-8"))
    return {"ok": True, **header}


def restore(path: Path, password: str, target: Path,
            overwrite: bool = False) -> dict:
    """Arsivi acar. Varsayilan olarak MEVCUT DOSYALARIN UZERINE YAZMAZ.

    Geri yukleme, yedeklemenin yarisidir: denenmemis bir yedek yedek degildir.
    Bu yuzden komut varsayilan olarak ayri bir dizine acar; boylece yedegin
    saglamligi, calisan kurulumu riske atmadan dogrulanabilir.
    """
    with path.open("rb") as fh:
        if fh.read(len(MAGIC)) != MAGIC:
            return {"ok": False, "reason": "bu bir yedek dosyasi degil"}
        n = int.from_bytes(fh.read(4), "big")
        header = json.loads(fh.read(n).decode("utf-8"))
        body = fh.read()

    key = _derive(password, bytes.fromhex(header["salt"]))
    try:
        raw = AESGCM(key).decrypt(bytes.fromhex(header["nonce"]), body, None)
    except Exception:
        return {"ok": False, "reason": "parola yanlis veya arsiv bozuk"}

    target.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            dest = target / name
            if dest.exists() and not overwrite:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            written += 1

    return {"ok": True, "written": written, "skipped": skipped,
            "target": str(target), "created_at": header.get("created_at")}


def latest(out_dir: Path | None = None) -> Path | None:
    d = out_dir or BACKUP_DIR
    if not d.exists():
        return None
    files = sorted(d.glob("yedek_*.hsy"))
    return files[-1] if files else None


def prune(keep: int = 8, out_dir: Path | None = None) -> list[str]:
    """En yeni `keep` yedegi birakir, digerlerini siler."""
    d = out_dir or BACKUP_DIR
    if not d.exists():
        return []
    files = sorted(d.glob("yedek_*.hsy"))
    removed = []
    for p in files[:-keep] if keep > 0 else []:
        try:
            p.unlink()
            removed.append(p.name)
        except OSError:
            pass
    return removed
