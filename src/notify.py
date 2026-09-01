"""Bildirimler — siteyi acmadan haberdar olmak.

NEDEN
-----
Gunluk is zaten her gun calisiyor ve izleme listesindeki pozisyonlar icin satis
sinyali uretiyor. Ama bu bilgi yalnizca panoda duruyordu: kullanicinin haberi
olmasi icin siteyi acmasi gerekiyordu. Bir stop seviyesinin kirilmasi, gorulmek
icin ziyaret bekleyemez.

KANALLAR
--------
  dosya     : her zaman acik. output/uyarilar.json + logs/uyarilar.log
  windows   : yerel masaustu bildirimi (ek kurulum gerektirmez)
  telegram  : YALNIZCA TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID ortam
              degiskenleri tanimliysa. Bu degerler kodda TUTULMAZ, sorulmaz ve
              hicbir yere yazilmaz; kullanici kendisi tanimlar, tanimlamazsa
              kanal sessizce kapali kalir.

TEKRAR ETMEME
-------------
Ayni uyari her gun yeniden gonderilirse bildirim degerini kaybeder ve
kullanici hepsini susturur. Her uyarinin bir kimligi vardir ve gonderilmis
kimlikler `data/uyari_gecmisi.json` icinde tutulur; ayni uyari
`REPEAT_AFTER_DAYS` gun gecmeden tekrar gonderilmez.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
LOGS = ROOT / "logs"
HISTORY = DATA / "uyari_gecmisi.json"

REPEAT_AFTER_DAYS = 5

SEVERITY_ORDER = {"kritik": 0, "yuksek": 1, "orta": 2, "bilgi": 3}


# =============================================================================
#  Uyari uretimi
# =============================================================================
def build_alerts(watch_results: list[dict] | None = None,
                 ranking=None, previous_top: set[str] | None = None,
                 top_n: int = 10) -> list[dict]:
    """Gunun uyarilarini uretir. Hicbir sey gondermez, yalnizca listeler."""
    alerts: list[dict] = []

    # --- 1) Izleme listesi: satis sinyali / yuksek risk
    for r in (watch_results or []):
        a = (r or {}).get("analysis") or {}
        if not a.get("available"):
            continue
        tk = r.get("ticker", "?")
        level = a.get("risk_level")
        if level in ("SAT", "YUKSEK_RISK"):
            sig = (a.get("signals") or [])[:3]
            alerts.append({
                "id": f"risk:{tk}:{level}",
                "severity": "kritik" if level == "SAT" else "yuksek",
                "title": f"{tk} — {a.get('risk_level_tr', level)}",
                "body": (a.get("action_tr") or "") + "\n" +
                        "\n".join(f"  [{s.get('siddet')}] {s.get('baslik')}" for s in sig),
                "ticker": tk,
            })

        # --- 2) Stop seviyesi kirildi
        stops = a.get("stops") or {}
        px, stop = a.get("price"), stops.get("active_stop")
        if px and stop and px <= stop:
            alerts.append({
                "id": f"stop:{tk}",
                "severity": "kritik",
                "title": f"{tk} — stop seviyesi kirildi",
                "body": f"Fiyat {px:.2f}, aktif stop {stop:.2f}.",
                "ticker": tk,
            })

        # --- 3) Bilanco cok yakin (pozisyon boyutu karari)
        er = a.get("earnings") or {}
        if er.get("available") and er.get("days") is not None and 0 <= er["days"] <= 2:
            alerts.append({
                "id": f"bilanco:{tk}:{er.get('date')}",
                "severity": "orta",
                "title": f"{tk} — bilanco {er['days']} gun sonra",
                "body": er.get("note_tr", ""),
                "ticker": tk,
            })

    # --- 4) Ilk 10'a yeni giren isim
    if ranking is not None and len(ranking) and previous_top is not None:
        head = ranking.head(top_n)
        for _, row in head.iterrows():
            tk = str(row["ticker"])
            if tk in previous_top:
                continue
            alerts.append({
                "id": f"yeni_top:{tk}:{datetime.now(timezone.utc):%Y-%W}",
                "severity": "bilgi",
                "title": f"{tk} ilk {top_n}'e girdi",
                "body": (f"Puan {row.get('total_score')}. "
                         f"Sektor: {row.get('sector', '?')}."),
                "ticker": tk,
            })

    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 9))
    return alerts


# =============================================================================
#  Tekrar filtresi
# =============================================================================
def _history() -> dict:
    if not HISTORY.exists():
        return {}
    try:
        return json.loads(HISTORY.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _remember(ids: list[str]) -> None:
    hist = _history()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i in ids:
        hist[i] = now
    # Cok eski kayitlar dosyayi sisirmesin
    cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    hist = {k: v for k, v in hist.items() if v >= cutoff}
    DATA.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(hist, ensure_ascii=False, indent=0, sort_keys=True),
                       encoding="utf-8")


def filter_new(alerts: list[dict]) -> list[dict]:
    """Yakin zamanda gonderilmis uyarilari eler."""
    hist = _history()
    cutoff = datetime.now(timezone.utc) - timedelta(days=REPEAT_AFTER_DAYS)
    out = []
    for a in alerts:
        prev = hist.get(a["id"])
        if prev:
            try:
                if datetime.fromisoformat(prev) > cutoff:
                    continue
            except ValueError:
                pass
        out.append(a)
    return out


# =============================================================================
#  Kanallar
# =============================================================================
def _channel_file(alerts: list[dict]) -> bool:
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "alerts": alerts}
    (OUT / "uyarilar.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (LOGS / "uyarilar.log").open("a", encoding="utf-8") as fh:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        for a in alerts:
            fh.write(f"[{stamp}] {a['severity'].upper():8} {a['title']}\n")
    return True


def _channel_windows(alerts: list[dict]) -> bool:
    """Yerel masaustu bildirimi — ek paket kurulumu gerektirmez."""
    if os.name != "nt" or not alerts:
        return False
    top = alerts[:3]
    text = " | ".join(a["title"] for a in top)
    if len(alerts) > 3:
        text += f" (+{len(alerts) - 3})"
    # Tirnak isaretleri PowerShell'i bozmasin
    text = text.replace("'", " ").replace("`", " ")[:220]
    ps = (
        "[reflection.assembly]::LoadWithPartialName('System.Windows.Forms') > $null;"
        "$n = New-Object System.Windows.Forms.NotifyIcon;"
        "$n.Icon = [System.Drawing.SystemIcons]::Information;"
        "$n.BalloonTipTitle = 'Hisse Siralama';"
        f"$n.BalloonTipText = '{text}';"
        "$n.Visible = $true; $n.ShowBalloonTip(15000); Start-Sleep -Seconds 6;"
        "$n.Dispose()"
    )
    try:
        import subprocess
        subprocess.run(["powershell", "-NoProfile", "-WindowStyle", "Hidden",
                        "-Command", ps], capture_output=True, timeout=40)
        return True
    except Exception:
        return False


def _channel_telegram(alerts: list[dict]) -> bool:
    """Telegram — yalnizca kullanici kendi ortam degiskenlerini tanimladiysa.

    Token ve chat id KODDA TUTULMAZ, sorulmaz, hicbir dosyaya yazilmaz.
    Tanimli degilse kanal sessizce kapalidir.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat or not alerts:
        return False

    lines = [f"*Hisse Siralama* — {datetime.now():%d.%m.%Y %H:%M}", ""]
    for a in alerts[:10]:
        lines.append(f"[{a['severity'].upper()}] {a['title']}")
        if a.get("body"):
            lines.append(a["body"].strip())
        lines.append("")
    if len(alerts) > 10:
        lines.append(f"... ve {len(alerts) - 10} uyari daha")

    try:
        import requests
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": "\n".join(lines)[:4000]},
                          timeout=20)
        return r.status_code == 200
    except Exception:
        return False


CHANNELS = {"dosya": _channel_file, "windows": _channel_windows,
            "telegram": _channel_telegram}


def send(alerts: list[dict], channels: list[str] | None = None,
         remember: bool = True) -> dict:
    """Uyarilari secili kanallara gonderir."""
    alerts = filter_new(alerts)
    if not alerts:
        return {"sent": 0, "channels": {}, "reason": "yeni uyari yok"}

    names = channels or ["dosya", "windows", "telegram"]
    used = {}
    for n in names:
        fn = CHANNELS.get(n)
        if not fn:
            continue
        try:
            used[n] = bool(fn(alerts))
        except Exception:
            used[n] = False

    if remember:
        _remember([a["id"] for a in alerts])
    return {"sent": len(alerts), "channels": used}


def load_alerts() -> dict | None:
    p = OUT / "uyarilar.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
