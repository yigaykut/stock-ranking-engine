"""Tek bakista sistem durumu: otomasyon calisiyor mu, sayac nerede, model var mi?

Panoda "%2.5" yazisini gorup "bunu ben mi baslatmaliyim?" diye sormak dogal bir
tepki. Bu betik o sorunun tum parcalarini tek yerde cevaplar:

    python scripts/durum.py

Hicbir sey degistirmez, yalnizca okur.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import dataset as ds        # noqa: E402
from src import report               # noqa: E402
from src import training as tr       # noqa: E402

TASK = "HisseSiralama_Gunluk"
BAR = "=" * 70


def _task_info() -> dict | None:
    """Windows gorev zamanlayicidan son/sonraki calisma bilgisi."""
    # Tetik SAYISI yaniltici olabilir: bir tetik "her N saatte bir tekrarla"
    # ile tanimliysa Triggers.Count onu 1 sayar. Kullaniciya anlamli olan sey
    # gunde KAC KEZ calisacagidir, o yuzden tekrar araligindan hesaplaniyor.
    # 27.08.2026'dan beri tekrar yok (gunde tek tetik) ve bu hesap 1 donduruyor;
    # kod yine de duruyor, cunku tekrar geri eklenirse dogru sayiyi verir.
    # ($rep atamasi hash literalinden ONCE gelmeli.)
    ps = (
        "$t = Get-ScheduledTask -TaskName '%s' -ErrorAction Stop;"
        "$i = Get-ScheduledTaskInfo -TaskName '%s';"
        "$ts = [System.Xml.XmlConvert];"
        "$rep = ($t.Triggers | ForEach-Object {"
        "  if ($_.Repetition.Interval -and $_.Repetition.Duration) {"
        "    [int]($ts::ToTimeSpan($_.Repetition.Duration).TotalMinutes /"
        "          $ts::ToTimeSpan($_.Repetition.Interval).TotalMinutes) + 1"
        "  } else { 1 } } | Measure-Object -Sum).Sum;"
        "$exe = ($t.Actions | Select-Object -First 1).Execute;"
        "[pscustomobject]@{state=$t.State.ToString();"
        "triggers=$rep;"
        "exe=$exe;"
        "last=$i.LastRunTime.ToString('yyyy-MM-dd HH:mm');"
        "result=$i.LastTaskResult;"
        "next=$i.NextRunTime.ToString('yyyy-MM-dd HH:mm')} | ConvertTo-Json"
    ) % (TASK, TASK)
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except Exception:
        return None


def main() -> int:
    print(BAR)
    print("SISTEM DURUMU")
    print(BAR)

    # --- 1) Otomasyon
    print("\n[1] GUNLUK OTOMASYON")
    t = _task_info()
    if not t:
        print("    KAYITLI GOREV YOK — gunluk tarama kendiliginden calismaz.")
        print("    Kurmak icin: scripts/gorev_kur.ps1 (veya Gorev Zamanlayici)")
    else:
        ok = t.get("result") == 0
        print(f"    gorev        : {TASK}  [{t.get('state')}]")
        kez = t.get("triggers")
        print(f"    tetik        : gunde {kez} kez"
              + ("" if kez != 1 else "  (kapaliysa ilk acilista telafi)"))
        print(f"    son calisma  : {t.get('last')}  "
              f"({'basarili' if ok else 'HATA kodu ' + str(t.get('result'))})")
        if not ok and str(t.get("result")) == "3221225786":
            print("                   ^ pencere kapatilmis (Ctrl+C). Tarama"
                  " yarida kalir ve gun kilidi dusmez.")
        # Gorev artik gunluk.bat'i wscript uzerinden penceresiz cagiriyor.
        # Dogrudan .bat cagiran eski kurulumda ekranda 30-40 dakika duran bir
        # konsol aciliyor; kapatilirsa tarama yarida kaliyor (bkz. ustteki
        # 3221225786 notu). Bu yuzden hangi kurulumun gecerli oldugu yazilir.
        exe = str(t.get("exe") or "")
        sessiz = "wscript" in exe.lower()
        print("    pencere      : "
              + ("yok (wscript ile penceresiz)" if sessiz else
                 "VAR — konsol aciliyor, kapatilirsa tarama yarim kalir;"
                 " duzeltmek icin scripts/gorev_kur.ps1"))
        print(f"    sonraki      : {t.get('next')}")
        print("    Not: ayni gun ikinci calisma is yapmaz; gun kilidi devrede.")

    mark = ROOT / "logs" / "son_basari.txt"
    today = datetime.now().strftime("%Y-%m-%d")
    if mark.exists():
        stamp = mark.read_text(encoding="utf-8").strip()
        print(f"    bugun        : {'TAMAMLANDI' if stamp == today else 'henuz yok'}"
              f"  (son basarili gun: {stamp})")
    else:
        print("    bugun        : henuz basarili calisma yok")

    # --- 2) Sayac
    print("\n[2] DOGRULAMA SAYACI")
    r = ds.readiness(21)
    v = report._validation_state()
    print(f"    anlik goruntu : {r['snapshots']} / {r['need_snapshots']}")
    print(f"    veri araligi  : {r['span_days']} / {r['need_span_days']} gun")
    print(f"    ilerleme      : %{r['progress_pct']}")
    if v.get("eta"):
        print(f"    tahmini bitis : {v['eta']}  ({v['days_left']} gun)")
    print("    Ilerlemesi icin YAPMAN GEREKEN BIR SEY YOK — her gunluk tarama")
    print("    bir goruntu ekler. Tek sart: bilgisayarin gun icinde acik olmasi.")

    # --- 3) Ogrenme
    print("\n[3] OGRENME")
    champ = tr.champion()
    if champ:
        print(f"    sampiyon     : {champ['model']}  agirlik {champ.get('weight')}")
        print(f"    IC {champ.get('ic')}  ICIR {champ.get('icir')}")
    else:
        print("    sampiyon     : yok — kanit olmadan model skorlamaya girmez")
    if r["ready_to_train"]:
        print("    egitim       : veri yeterli, gunluk is 5 taramada bir dener")
    else:
        print("    egitim       : sayac dolunca KENDILIGINDEN baslar")

    pt = v.get("pretrain")
    if pt:
        durum = "uretim suruyor" if pt.get("partial") else "tamamlandi"
        print(f"\n    Beklemeden deneme paneli hazir ({durum}):")
        print(f"      {pt['snapshots']} goruntu, {pt['tickers']} hisse, "
              f"{pt['first_date']} -> {pt['last_date']}")
        print("      python run.py ml train --pretrain")
        print("      (bu panel sampiyon URETEMEZ — hayatta kalma yanliligi)")
    else:
        print("\n    Beklemeden denemek icin: python run.py history")

    # --- 4) Karne
    print("\n[4] KARNE (kagit uzerinde defter)")
    try:
        from src import paper                       # noqa: PLC0415
        s = paper.load_summary() or {}
        for etiket, key in (("gercek taramalar", "live"), ("gecmis panel", "panel")):
            part = s.get(key) or {}
            if part.get("ok"):
                t = part.get("t_stat")
                anlam = ("gurultuden ayirt EDILEMEZ" if t is None or abs(t) < 2
                         else "gurultuden ayirt edilebilir")
                print(f"    {etiket:<17}: {part['cohorts']} kohort, endeks farki "
                      f"%{part['excess_pct']:+.2f}, t={t} -> {anlam}")
            else:
                print(f"    {etiket:<17}: {part.get('reason', 'veri yok')}")
        if not s:
            print("    defter bos — python run.py paper build --panel")
    except Exception as exc:
        print(f"    okunamadi ({exc})")

    # --- 5) Veri saglami
    print("\n[5] VERI")
    try:
        from src import fundamentals                # noqa: PLC0415
        fi = fundamentals.info()
        if fi.get("snapshots"):
            print(f"    temel veri arsivi : {fi['snapshots']} gun, {fi['mb']} MB "
                  f"({fi['first_date']} -> {fi['last_date']})")
        else:
            print("    temel veri arsivi : bos (ilk taramada olusur)")
    except Exception:
        pass
    try:
        from src import delisting                   # noqa: PLC0415
        di = delisting.info()
        print(f"    kote disi takibi  : {di['confirmed']} kesinlesti, "
              f"{di['pending']} izleniyor")
    except Exception:
        pass
    try:
        from src import backup                      # noqa: PLC0415
        last = backup.latest()
        if last:
            yas = (datetime.now().timestamp() - last.stat().st_mtime) / 86400
            print(f"    son yedek         : {last.name}  ({yas:.0f} gun once)")
        else:
            print("    son yedek         : YOK — python run.py backup")
    except Exception:
        pass

    print("\n" + BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
