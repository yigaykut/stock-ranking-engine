#!/usr/bin/env python
"""Hisse yatirim skorlama sistemi — komut satiri arayuzu.

Ornekler
--------
  # S&P 500 + WSB'de en cok anilanlari tara, ilk 40'i panoda goster
  python run.py --universe sp500,wsb --top 40

  # Hizli deneme (20 hisse)
  python run.py --universe sp500 --limit 20

  # WSB parametresini tamamen devre disi birak
  python run.py --disable reddit_wsb_attention

  # Nominal fiyat kriterini agirlastir, degerlemeyi hafiflet
  python run.py --weight nominal_price_fit=5 --weight valuation_composite=4

  # Kendi listen
  python run.py --universe file --symbols-file my_list.txt

  # ML: biriken anlik goruntuleri etiketle ve agirliklari veriden ogren
  python run.py --learn-weights --horizon 21
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _force_utf8_console() -> None:
    """Konsol cikisini UTF-8'e sabitler.

    Windows'ta varsayilan konsol kod sayfasi cp1254 (Turkce). ASCII disi TEK bir
    karakter iceren print, tum sureci UnicodeEncodeError ile dusuruyordu:
    15-17 Agustos'ta gunluk is tam da bunun yuzunden her calismada 1 dondu, gun
    isaretlenmedi ve 8 tetigin hepsi ayni taramayi bastan yapti. Yani bir SUS
    KARAKTERI, otomasyonun tamamini bozdu. Kod tarafinda ayrica ASCII disi
    karakter testi var (tests/test_kodlama.py), ama bu satir ikinci savunmadir:
    metin ne olursa olsun surec dusmez.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


_force_utf8_console()

from src import factors, ml, report, scanlog, scoring, universe  # noqa: E402
from src.providers import cache, reddit_wsb, yahoo              # noqa: E402

OUT = ROOT / "output"

# Yedek fiyat kaynagina bir turda gonderilecek azami istek. Sinir olmasaydi
# devre kesici sonrasi 450+ istek gidebilirdi; ikinci kaynagi da yakmak,
# birincisini kaybetmekten daha kotu bir durum yaratir.
FALLBACK_LIMIT = 300


# ---------------------------------------------------------------- yardimcilar
def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def apply_filters(records: list[dict], cfg: dict, enabled: bool,
                  wsb_data: dict, pinned: set[str] | None = None
                  ) -> tuple[list[dict], list[dict]]:
    """Cok gevsek on eleme. Amac anlamsiz enstrumanlari atmak, hisse elemek degil.

    pinned: izleme listesindeki semboller. Bunlar HICBIR filtreye takilmaz —
    kullanici bir hisseyi listesine aldiysa, hacmi dustu diye gozden kaybolmasi
    kabul edilemez; tam tersine o durumu gormesi gerekir.
    """
    if not enabled:
        return records, []

    pinned = pinned or set()
    f = cfg.get("filters", {}) or {}
    kept, dropped = [], []

    for r in records:
        if not r.get("ok"):
            dropped.append({"ticker": r["ticker"], "reason": r.get("reason", "veri yok")})
            continue

        if r["ticker"] in pinned:
            kept.append(r)          # izleme listesi filtrelerden muaf
            continue

        price = r.get("price") or 0
        mcap = r.get("market_cap")
        dv = r.get("avg_dollar_volume")
        reason = None

        if f.get("min_price") and price < f["min_price"]:
            reason = f"fiyat {price:.2f} < {f['min_price']}"
        elif f.get("max_price") and price > f["max_price"]:
            reason = f"fiyat {price:.2f} > {f['max_price']}"
        elif f.get("min_market_cap") and mcap is not None and mcap < f["min_market_cap"]:
            reason = f"piyasa degeri dusuk ({mcap:,.0f})"
        elif f.get("max_market_cap") and mcap is not None and mcap > f["max_market_cap"]:
            reason = f"piyasa degeri yuksek ({mcap:,.0f}) — yukselen sirket araniyor"
        elif f.get("min_avg_dollar_volume") and dv is not None and dv < f["min_avg_dollar_volume"]:
            reason = f"gunluk dolar hacmi dusuk ({dv:,.0f})"
        elif f.get("require_wsb") and r["ticker"] not in wsb_data:
            reason = "WSB'de anilmamis (require_wsb acik)"
        elif f.get("require_analyst_buy"):
            rec = (r.get("meta", {}).get("analyst_consensus", {}) or {}).get("recommendation_mean")
            if rec is None or rec > 2.5:
                reason = "analist tavsiyesi Al seviyesinde degil (require_analyst_buy acik)"

        (dropped if reason else kept).append(
            {"ticker": r["ticker"], "reason": reason} if reason else r)

    return kept, dropped


def write_status(ok: bool, detail: dict | None = None, error: str | None = None) -> None:
    """Son calismanin sonucunu kaydeder (bulgu O7).

    Pano bu dosyadan verinin yasini okur ve eskiyse uyarir; boylece gunluk is
    sessizce basarisiz oldugunda kullanici eski veriye bakarken bunu fark eder.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": bool(ok),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finished_at_local": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "detail": detail or {},
    }
    if error:
        payload["error"] = str(error)[:500]
    try:
        (OUT / "run_status.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def fetch_one(ticker: str, period: str, use_cache: bool):
    try:
        return ticker, yahoo.fetch(ticker, period=period, use_cache=use_cache)
    except yahoo.RateLimited as exc:
        return ticker, {"_rate_limited": str(exc)}
    except Exception as exc:
        return ticker, {"_error": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------------------------- komutlar
def _previous_snapshot_rows() -> int:
    """BUGUNDEN ONCEKI en son anlik goruntunun satir sayisi (yoksa 0)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    files = sorted(p for p in ml.FEATURE_STORE.glob("snapshot_*.csv")
                   if p.stem[len("snapshot_"):][:10] < today)
    if not files:
        return 0
    try:
        with files[-1].open(encoding="utf-8") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except OSError:
        return 0


def _previous_top(n: int) -> set[str]:
    """Onceki taramanin ilk N sembolu (cekim onceligi icin)."""
    prev, _ = ml.previous_snapshot()
    if prev is None or "total_score" not in prev.columns:
        return set()
    top = (prev.sort_values("total_score", ascending=False, na_position="last")
               .head(n))
    return set(top["ticker"].astype(str))


def guard_universe(tickers: list[str], breakdown: dict, sources: list[str],
                   args: argparse.Namespace) -> tuple[list[str], dict]:
    """Coken evreni tespit eder ve mumkunse kurtarir.

    OLAY: 16 Agustos'ta api.nasdaq.com erisilemedi. Kotasyon kaynagi sessizce
    bos liste dondu, evren kullanicinin izleme listesindeki 4 hisseye cokta ve
    tarama BASARIYLA tamamlandi -- 4 hisselik bir siralama uretip panonun
    uzerine yazdi. Sitede haftalardir birikmis siralama gitti, yerine "sadece
    benim ekledigim hisseler" kaldi.

    Sessiz cokusun uc ayagi vardi ve ucu de burada kapatiliyor:
      1. Bos kaynak hata sayilmiyordu      -> asagidaki esik kontrolu
      2. Bayat onbellek kullanilmiyordu    -> universe.us_listings
      3. Cokmus tarama panoyu eziyordu     -> kurtarilamazsa iptal

    Doner: (semboller, bilgi). bilgi["abort"] True ise cagiran taraf DURMALI.
    """
    info: dict = {"abort": False, "recovered": False, "reason": None}

    # Kucuk evren BEKLENEN durumlarda kontrol calismaz: elle sembol dosyasi,
    # --limit ile kucultulmus deneme taramasi, tek endeks.
    expects_large = any(s.strip().lower() in universe.PRESETS for s in sources)
    if not expects_large or args.limit:
        return tickers, info

    listings = breakdown.get("_listings") or {}
    threshold = max(scanlog.MIN_SANE_UNIVERSE, 0)
    if len(tickers) >= threshold and listings.get("ok", True):
        return tickers, info

    print(f"\n      UYARI: evren cokmus gorunuyor ({len(tickers)} sembol). "
          f"Kotasyon kaynagi: {listings.get('source', 'bilinmiyor')}")

    rescue, day = scanlog.last_universe()
    if rescue:
        info.update(recovered=True, reason="kotasyon kaynagi erisilemedi",
                    recovered_from=day, recovered_count=len(rescue))
        print(f"      KURTARMA: {day} tarihli evren kaydi kullaniliyor "
              f"({len(rescue)} sembol). Fiyat verisi onbellekten gelecek.")
        return rescue, info

    info.update(abort=True, reason="evren cokmus, kurtarma kaydi da yok")
    print("\nHATA: evren olusturulamadi ve kurtarilacak gecmis kayit da yok.\n"
          "      Tarama IPTAL edildi - eksik bir siralama panonun uzerine\n"
          "      yazilmasin diye. Kaynak erisilebilir olunca tekrar dene.",
          file=sys.stderr)
    write_status(ok=False, error="evren cokmus (kotasyon kaynagi erisilemedi)")
    return tickers, info


def cmd_scan(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    t0 = time.time()

    # --- 1) Evren
    sources = [s.strip() for s in args.universe.split(",") if s.strip()]
    print(f"[1/6] Evren olusturuluyor: {', '.join(sources)}")
    tickers, breakdown = universe.build(sources, wsb_top=args.wsb_top,
                                        symbols_file=args.symbols_file, limit=args.limit,
                                        min_mcap=args.min_mcap, max_mcap=args.max_mcap)

    lst = breakdown.pop("_listings", None)
    if lst and lst.get("source") in ("onbellek", "bayat_onbellek"):
        print(f"      kotasyon listesi onbellekten ({lst['count']} sembol, "
              f"{lst['age_hours']} saat once)")

    tickers, uni_guard = guard_universe(tickers, dict(breakdown, _listings=lst),
                                        sources, args)
    if uni_guard["abort"]:
        return 1
    if uni_guard["recovered"]:
        breakdown["kurtarilan_evren"] = uni_guard["recovered_count"]

    # --- Izleme listesi HER ZAMAN evrene dahil ---------------------------------
    # Liste gunluk yenilenir ve siralama degisir; ama kullanicinin sectigi
    # hisseler evrenden dusse bile taranmaya ve gosterilmeye devam eder.
    from src import watchlist as _wl
    pinned = {p["ticker"] for p in _wl.load()}
    if pinned:
        added = sorted(pinned - set(tickers))
        tickers = tickers + added
        breakdown["izleme_listesi"] = len(pinned)
        breakdown["final_unique"] = len(tickers)
        if added:
            print(f"      izleme listesinden eklenen: {len(added)} ({', '.join(added[:8])}"
                  f"{'...' if len(added) > 8 else ''})")

    # --- Kote disi takibi (bulgu Y3) -----------------------------------------
    # DIKKAT: kontrol, band suzgecinden gecmis evrene degil kotasyon
    # beslemesinin TAMAMINA karsi yapilir. Piyasa degeri 20 milyar dolari asan
    # bir sirket de evrenden duser -- ama o batmadi, tam tersi oldu. Ikisi
    # karistirilirsa yanlilik duzelmez, tersine cevrilir.
    try:
        from src import delisting as _dl
        if any(s.strip().lower() in universe.PRESETS for s in sources):
            all_rows, _ = universe.us_listings()
            dl_info = _dl.update({s for s, _ in all_rows})
            if dl_info.get("ok") and dl_info["newly_confirmed"]:
                print(f"      KOTE DISI kesinlesti: "
                      f"{', '.join(dl_info['newly_confirmed'][:8])}")
    except Exception as exc:
        print(f"      UYARI: kote disi takibi guncellenemedi ({exc})")

    uni_info = scanlog.record_universe(tickers)
    if uni_info["disappeared_count"]:
        print(f"      {uni_info['disappeared_count']} sembol {uni_info['compared_to']} "
              f"listesinde vardi, bugun yok (kote disi olabilir)")

    # --- DONUSUMLU TARAMA (bulgu K3) -----------------------------------------
    # Hiz siniri yuzunden evrenin tamami tek seferde cekilemiyor. Her gun ayni
    # sirayla taranirsa listenin sonu HIC gorulmez. "En uzun suredir taranmamis
    # once" sirasi, birkac gunde tum evreni dolasmayi saglar.
    cov = scanlog.coverage_stats(tickers)
    universe_all = list(tickers)              # batch kesiminden ONCEKI tam evren
    universe_full = len(tickers)
    # Onceki taramanin ust dilimi: butcenin ilk sirasi bunlarin (bkz.
    # order_by_staleness). Karar bu isimler uzerinden veriliyor, taze olmalilar.
    priority = _previous_top(max(args.top, 40) * 3)
    tickers = scanlog.order_by_staleness(tickers, pinned, priority)
    batched = False
    if args.batch:
        keep = max(len(pinned), int(args.batch))
        if keep < len(tickers):
            batched = True
            print(f"      donusumlu tarama: {universe_full} sembolden en bayat "
                  f"{keep} tanesi bu turda taranacak")
            tickers = tickers[:keep]
    print(f"      evren kapsamasi: %{cov['coverage_pct']} daha once tarandi, "
          f"{cov['never_scanned']} sembol hic taranmadi")
    if not tickers:
        print("HATA: evren bos. --universe / --symbols-file kontrol et.", file=sys.stderr)
        write_status(ok=False, error="evren bos")
        return 1
    print(f"      {breakdown}")
    for src in sources:
        if breakdown.get(src, 0) == 0:
            print(f"      UYARI: '{src}' kaynagi hic sembol dondurmedi (sayfa duzeni "
                  f"degismis veya erisilemiyor olabilir); diger kaynaklarla devam ediliyor")

    # --- 2) Reddit WSB
    print("[2/6] Reddit r/wallstreetbets verisi cekiliyor...")
    try:
        wsb_data = reddit_wsb.load(use_cache=not args.no_cache)
        print(f"      {len(wsb_data)} sembol icin anma verisi bulundu")
    except Exception as exc:
        wsb_data = {}
        print(f"      UYARI: WSB verisi alinamadi ({exc}); bu parametre devre disi kalacak")

    # --- 3) Endeks
    print(f"[3/6] Karsilastirma endeksi ({args.benchmark}) cekiliyor...")
    bench = yahoo.fetch_benchmark(args.benchmark, period=args.period, use_cache=not args.no_cache)
    bench_close = bench["Close"] if bench is not None and "Close" in bench else None
    if bench_close is None:
        print("      UYARI: endeks alinamadi; goreli guc parametresi bos kalacak")

    # --- 4) Fiyat + temel veri (paralel)
    print(f"[4/6] {len(tickers)} hisse icin veri cekiliyor (paralel: {args.workers})...")
    bundles: dict[str, dict] = {}
    errors: list[str] = []
    done = 0

    no_data: list[str] = []
    failed: list[str] = []          # yedek kaynakta denenecekler
    rate_limited = 0
    aborted = False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, tk, args.period, not args.no_cache): tk for tk in tickers}
        for fut in as_completed(futs):
            if fut.cancelled():
                continue          # devre kesici tarafindan iptal edildi
            try:
                tk, bundle = fut.result()
            except CancelledError:
                continue
            done += 1

            # --- Devre kesici: Yahoo bizi yavaslatiyorsa denemeye devam etmek
            # yasagi uzatmaktan baska ise yaramaz. Elde olanla devam ederiz.
            if "_rate_limited" in bundle:
                rate_limited += 1
                if rate_limited >= 25 and not aborted:
                    aborted = True
                    print(f"\n      DURDURULDU: Yahoo hiz siniri uyguluyor "
                          f"({rate_limited} ardisik ret).")
                    for f in futs:
                        f.cancel()
                continue

            if "_error" in bundle:
                errors.append(f"{tk}: {bundle['_error']}")
                failed.append(tk)
            elif bundle.get("history") is None:
                # Cekim istisna atmadi ama fiyat serisi gelmedi. Bunu "yetersiz
                # gecmis" saymak yaniltici olur — cogunlukla hiz siniridir.
                no_data.append(tk)
                failed.append(tk)
            else:
                bundles[tk] = bundle
            if done % 25 == 0 or done == len(tickers):
                print(f"      {done}/{len(tickers)}", end="\r", flush=True)

    # --- 4b) YEDEK FIYAT KAYNAGI ---------------------------------------------
    # Yahoo'nun basarisiz oldugu sembollerde fiyat tarafi kurtarilir. Yalnizca
    # BASARISIZLARDA denenir: butce boylece kendiliginden sinirli kalir
    # (~360 istek) ve saglam cekimler ikinci kez sorgulanmaz.
    #
    # Temel veri Yahoo onbelleginden korunur; yalnizca fiyat serisi tazelenir.
    # Ayrintili gerekce ve duzeltilmemis seri uyarisi: src/providers/nasdaq.py
    # Yahoo'nun kendi basari orani, yedek kaynaktan ONCE olculur: aksi halde
    # oran sisirilir ve gercek kaynak sagligini gizler.
    yahoo_ok = len(bundles)

    # Devre kesici devreye girdiginde kalan istekler IPTAL edilir; bu semboller
    # hic denenmedigi icin `failed` listesine de girmez. Oysa yedek kaynaga en
    # cok ihtiyac duyulan an tam olarak budur. Ilk gercek calismada bu yuzden
    # yedek kaynak hic devreye girmedi (446 sembol iptal, `failed` bos).
    fallback_used = 0
    cand: list[str] = []
    if not args.no_fallback:
        pool = tickers if aborted else failed
        cand = [t for t in pool if t not in bundles][:FALLBACK_LIMIT]

    if cand:
        from src.providers import nasdaq as _nq
        print(f"      yedek kaynak deneniyor: {len(cand)} sembol "
              f"(fiyat serisi; temel veri onbellekten)")

        def _fb(tk: str):
            base = yahoo.fetch_cached(tk, period=args.period,
                                      max_age_seconds=30 * 24 * 3600)
            try:
                return tk, _nq.as_bundle(tk, args.period, base)
            except Exception:
                return tk, None

        with ThreadPoolExecutor(max_workers=max(2, args.workers)) as pool:
            for fut in as_completed([pool.submit(_fb, t) for t in cand]):
                try:
                    tk, b = fut.result()
                except Exception:
                    continue
                if b is not None:
                    bundles[tk] = b
                    fallback_used += 1
        print(f"      yedek kaynaktan kurtarilan: {fallback_used} hisse")

    # Basarili cekimleri kaydet -> bir sonraki tur bunlari sona atar
    if bundles:
        scanlog.record(list(bundles.keys()))

    # Cekim basari orani, GERI DOLDURMADAN ONCE hesaplanir; aksi halde
    # onbellekten gelenler oranı sisirir ve %100'u asar.
    fetched_live = len(bundles)
    ok_rate = 100.0 * fetched_live / max(1, len(tickers))
    yahoo_rate = 100.0 * yahoo_ok / max(1, len(tickers))

    # --- Onbellekten geri doldurma -------------------------------------------
    # Donusumlu tarama bu turda evrenin bir dilimini cekti. Daha once cekilmis
    # hisseleri de skorlamaya katiyoruz: ag maliyeti YOK (yalnizca onbellek
    # okumasi) ama siralama her turda evrenin daha buyuk bir kismini kapsar.
    backfilled = 0
    # Her hissenin verisinin kac gunluk oldugu. Canli cekilenler 0.0.
    # Bu olmadan pano "bugunun siralamasi" gibi gorunur ama satirlarin bir
    # kismi haftalik eski fiyattan gelir; asagida sayilip panoya yaziliyor.
    data_age: dict[str, float] = {tk: 0.0 for tk in bundles}
    if not args.no_backfill:
        max_age = args.backfill_days * 24 * 3600
        for tk in universe_all:
            if tk in bundles:
                continue
            hit = cache.peek("yahoo", f"{tk}:{args.period}")
            if hit is None:
                continue
            b, age = hit
            if age > max_age:
                continue
            if b is not None and b.get("history") is not None:
                bundles[tk] = b
                data_age[tk] = age / 86400.0
                backfilled += 1
        if backfilled:
            print(f"      onbellekten eklendi: {backfilled} hisse "
                  f"(ag istegi yok) -> toplam {len(bundles)} hisse skorlanacak")

    # --- Piyasa rejimi -------------------------------------------------------
    # Skoru DEGISTIRMEZ; siralamanin hangi ortamda uretildigini kaydeder ve
    # panoda soyler. Rejim etiketi bugun yazilmazsa sonradan uretilemez
    # (evren ve kapsama degisiyor), bu yuzden her taramada kaydedilir.
    regime_state = {}
    try:
        from src import regime as _rg
        regime_state = _rg.compute(bench_close, _rg.breadth(bundles))
        _rg.record(regime_state)
        print(f"      piyasa rejimi: {regime_state.get('label_tr')}"
              + (f" · genislik %{regime_state['breadth_pct']}"
                 if regime_state.get("breadth_pct") is not None else ""))
    except Exception as exc:
        print(f"      UYARI: rejim hesaplanamadi ({exc})")

    # --- Temel verinin gunluk arsivi -----------------------------------------
    # Yahoo gecmise donuk temel veri VERMEZ. Bugun saklanmayan alan, alti ay
    # sonra hicbir yerden bulunamaz. Bu yuzden arsivleme skorlamadan once ve
    # kosulsuz yapilir; hatasi taramayi durdurmaz.
    try:
        from src import fundamentals as _fund
        fund_rows = [_fund.extract(tk, b) for tk, b in bundles.items()]
        fund_path = _fund.save_snapshot(fund_rows)
        if fund_path:
            print(f"      temel veri arsivi: {len(fund_rows)} hisse -> "
                  f"{fund_path.name}")
    except Exception as exc:
        print(f"      UYARI: temel veri arsivlenemedi ({exc})")

    print(f"      {fetched_live}/{len(tickers)} basarili (%{ok_rate:.0f}"
          + (f"; Yahoo %{yahoo_rate:.0f} + yedek {fallback_used}"
             if fallback_used else "") + ")"
          + (f", {len(no_data)} veri donmedi" if no_data else "")
          + (f", {rate_limited} hiz siniri" if rate_limited else "")
          + (f", {len(errors)} hata" if errors else ""))

    if aborted or rate_limited:
        print(f"\n      Yahoo hiz siniri devrede. Bu NORMALDIR: ~2800 hisse x ~12 istek\n"
              f"      ucretsiz uctan tek seferde cekilemez.\n"
              f"      TARAMA KALDIGI YERDEN DEVAM EDER — basarili cekimler onbellekte\n"
              f"      (data/cache), tekrar calistirdiginda yalnizca eksikler denenir.\n"
              f"      Onerilen: 30-60 dakika bekle, sonra ayni komutu tekrar calistir.")
    elif ok_rate < 70:
        print(f"      UYARI: cekim basari orani dusuk (%{ok_rate:.0f}).\n"
              f"      Daha az --workers ile tekrar calistir.")

    if not bundles:
        print("\nHATA: hic veri cekilemedi. Yahoo hiz siniri aktif olabilir; "
              "bir sure sonra tekrar dene.", file=sys.stderr)
        return 1

    # --- 5) Faktorler
    print("[5/6] Faktorler hesaplaniyor...")
    records = []
    for tk, bundle in bundles.items():
        try:
            records.append(factors.compute_all(tk, bundle, bench_close, wsb_data.get(tk)))
        except Exception as exc:
            errors.append(f"{tk} (faktor): {type(exc).__name__}: {exc}")

    kept, dropped = apply_filters(records, cfg, not args.no_filters, wsb_data, pinned)
    print(f"      {len(kept)} hisse skorlamaya girdi, {len(dropped)} elendi")

    # --- 6) Skorlama
    print("[6/6] Skorlaniyor...")
    disabled = set(x.strip() for x in (args.disable or "").split(",") if x.strip())
    overrides = {}
    for w in (args.weight or []):
        k, _, v = w.partition("=")
        try:
            overrides[k.strip()] = float(v)
        except ValueError:
            print(f"      UYARI: agirlik gecersiz, atlaniyor: {w}")

    if args.use_learned:
        lp = OUT / "learned_weights.json"
        if lp.exists():
            learned = json.loads(lp.read_text(encoding="utf-8")).get("weights", {})
            overrides.update({k: v for k, v in learned.items() if k not in overrides})
            print(f"      ogrenilmis agirliklar uygulandi ({len(learned)} faktor)")
        else:
            print("      UYARI: learned_weights.json yok, uzman agirliklari kullaniliyor")

    scorer = scoring.Scorer(cfg, disabled=disabled, weight_overrides=overrides,
                            pinned=pinned)
    result, diag = scorer.score(kept)
    if result.empty:
        print(f"HATA: {diag.get('error')}", file=sys.stderr)
        write_status(ok=False, error=str(diag.get("error")))
        return 1

    # --- OGRENILEN MODEL: geri beslemenin skora dondugu nokta ----------------
    # Iki gecisli: once model olmadan skorlanir (modelin girdisi bu skorlardir),
    # sonra tahmin bir parametre olarak eklenip yeniden skorlanir.
    # Sampiyon yoksa veya agirligi 0 ise hicbir sey degismez.
    from src import training as _tr
    champ = _tr.champion()
    diag["model"] = {"champion": None, "applied": False}
    if champ and float(champ.get("weight") or 0) > 0:
        try:
            feat = ml.to_feature_matrix(result, [f["id"] for f in cfg["factors"]])
            preds = _tr.predict_live(feat, champ.get("feature_names") or [])
        except Exception as exc:
            preds = None
            diag["model"]["error"] = f"{type(exc).__name__}: {exc}"

        if preds:
            for r in kept:
                r["raw"]["model_score"] = preds.get(r["ticker"])
            overrides["model_score"] = float(champ["weight"])
            scorer = scoring.Scorer(cfg, disabled=disabled, weight_overrides=overrides,
                                    pinned=pinned)
            result, diag2 = scorer.score(kept)
            diag2["model"] = {"champion": champ["model"], "applied": True,
                              "weight": champ["weight"], "ic": champ.get("ic"),
                              "icir": champ.get("icir"),
                              "covered": len(preds)}
            diag2["universe_size"] = diag.get("universe_size")
            diag = {**diag, **diag2}
            print(f"      ogrenilen model uygulandi: {champ['model']} "
                  f"(agirlik {champ['weight']}, IC {champ.get('ic')})")
        else:
            diag["model"]["reason"] = "tahmin uretilemedi (parametre seti degismis olabilir)"
            print("      UYARI: sampiyon model tahmin uretemedi, modelsiz devam ediliyor")
    elif champ:
        diag["model"]["champion"] = champ["model"]
        diag["model"]["reason"] = "agirlik 0 — henuz kanitlanmis beceri yok"

    # --- Onceki gune gore degisim (liste sabit degil, hareketi gorunur olmali)
    result, delta_info = ml.compute_deltas(result)
    diag["deltas"] = delta_info
    if delta_info.get("compared_to"):
        print(f"      onceki tarama {delta_info['compared_to']} ile karsilastirildi: "
              f"{delta_info['new_count']} yeni, {delta_info['moved_up']} yukari, "
              f"{delta_info['moved_down']} asagi")

    # Iki sayi ayri raporlanir: bu turda denenen (batch) ve evrenin tamami.
    # Tek sayi gosterilseydi donusumlu tarama, evren kucukmus gibi gorunurdu.
    diag["universe_size"] = len(tickers)          # bu turda denenen
    diag["universe_full"] = universe_full          # evrenin tamami
    diag["batched"] = batched
    diag["universe_coverage_pct"] = cov["coverage_pct"]
    diag["never_scanned"] = cov["never_scanned"]
    diag["delisted_candidates"] = uni_info.get("disappeared_count", 0)
    diag["fetched_ok"] = fetched_live
    diag["backfilled"] = backfilled
    diag["scored_universe"] = len(bundles)
    diag["fetch_no_data"] = len(no_data)
    diag["fetch_rate_limited"] = rate_limited
    diag["fetch_aborted"] = aborted

    # --- Veri tazeligi -------------------------------------------------------
    # yahoo.fetch_cached'in docstring'i "eskiyen satir bayat isaretlenir" diyordu
    # ama isaretleme HIC YAZILMAMISTI. Geri doldurma penceresi genisletildigi
    # icin (5 -> 12 gun) bu sayilar artik panoda gorunmek zorunda: aksi halde
    # iki haftalik fiyattan uretilmis bir sira "bugunun siralamasi" gibi okunur.
    scored_ages = [data_age.get(str(t), 0.0) for t in result["ticker"]] \
        if "ticker" in result else []
    if scored_ages:
        ages_sorted = sorted(scored_ages)
        diag["data_age"] = {
            "fresh_today": sum(1 for a in scored_ages if a < 1.0),
            "stale_over_3d": sum(1 for a in scored_ages if a >= 3.0),
            "stale_over_7d": sum(1 for a in scored_ages if a >= 7.0),
            "median_days": round(ages_sorted[len(ages_sorted) // 2], 1),
            "max_days": round(ages_sorted[-1], 1),
        }
        d = diag["data_age"]
        print(f"      veri tazeligi: {d['fresh_today']} hisse bugun cekildi, "
              f"{d['stale_over_3d']} hisse 3+ gunluk, {d['stale_over_7d']} hisse "
              f"7+ gunluk (medyan {d['median_days']} gun)")
    diag["fetch_success_rate"] = round(ok_rate / 100, 3)
    diag["yahoo_success_rate"] = round(yahoo_rate / 100, 3)
    diag["fallback_used"] = fallback_used
    diag["regime"] = regime_state
    # Ornekleri kirpiyoruz ama SAYIMLARI tam tutuyoruz — aksi halde evrenin
    # neden daraldigi gorunmez oluyor.
    reason_counts: dict[str, int] = {}
    for d in dropped:
        key = (d.get("reason") or "bilinmiyor").split("(")[0].split("—")[0].strip()
        reason_counts[key] = reason_counts.get(key, 0) + 1
    diag["dropped_total"] = len(dropped)
    diag["dropped_by_reason"] = dict(sorted(reason_counts.items(), key=lambda x: -x[1]))
    diag["dropped"] = dropped[:50]
    diag["errors_total"] = len(errors)
    diag["errors"] = errors[:50]

    print(f"\n  Eleme dokumu ({len(dropped)} hisse):")
    for k, v in list(diag["dropped_by_reason"].items())[:6]:
        print(f"    {v:5d}  {k}")

    # --- Cikti guvenligi: yarim bir tarama panonun uzerine yazmasin ----------
    # Ikinci savunma hatti. guard_universe evren tarafini tutuyor; bu kontrol
    # ise "evren dogru ama cekim neredeyse tamamen basarisiz" durumunu
    # yakaliyor. Pano, birikmis en degerli ciktidir; yerine 4 satirlik bir
    # liste yazmaktansa dunku panoyu birakmak her zaman daha iyidir.
    prev_rows = _previous_snapshot_rows()
    if prev_rows >= 200 and len(result) < 0.4 * prev_rows and not args.limit:
        print(f"\nHATA: bu tarama yalnizca {len(result)} hisse skorladi; onceki "
              f"tarama {prev_rows} idi.\n"
              f"      Pano GUNCELLENMEDI - eksik liste, dolu listenin uzerine\n"
              f"      yazilmaz. Veri kaynagi duzelince tekrar calistir.",
              file=sys.stderr)
        write_status(ok=False, detail={"scored": len(result), "previous": prev_rows},
                     error="tarama cok eksik, pano korundu")
        return 1

    # --- Ciktilar
    OUT.mkdir(parents=True, exist_ok=True)

    # SIRA ONEMLI: anlik goruntu, PANODAN ONCE kaydedilir.
    # Pano ust seridinde "dogrulama icin kac goruntu birikti" yaziyor ve bu
    # sayiyi feature store'dan okuyor. Kayit sonraya birakilirsa pano her gun
    # BUGUNU SAYMAZ — kalici olarak bir eksik gosterir ve kullanici sayaci
    # ilerlemiyor sanir. (compute_deltas zaten daha yukarida calisti ve
    # yalnizca bugunden ONCEKI goruntulere bakar; bu sira onu etkilemez.)
    factor_ids = [f["id"] for f in cfg["factors"]]
    feat = ml.to_feature_matrix(result, factor_ids)
    snap_path = ml.save_snapshot(feat)

    # Kagit uzerinde defter: bugunun ilk N'i, gercek getirisi olculmek uzere
    # kaydedilir. Panodan ONCE, cunku pano defterin ozetini gosteriyor.
    try:
        from src import paper
        paper.record_live(result, top_n=paper.DEFAULT_TOP_N)
        # Degerleme de burada yapilir: pano defterin OZETINI gosteriyor, yani
        # ozet panodan once hazir olmali. (Ayni sira hatasi sayacta bir kez
        # yapildi ve pano her gun bir eksik gosterdi.)
        pinfo = paper.refresh(horizon=21)
        pl = (pinfo.get("live") or {})
        if pl.get("ok"):
            print(f"      kagit defter: {pl['cohorts']} kohort, endeks farki "
                  f"%{pl['excess_pct']}")
    except Exception as exc:                      # defter kritik yol degil
        print(f"      UYARI: kagit defter guncellenemedi ({exc})")

    # Kisa vadeli kurulumlar da PANODAN ONCE uretilmeli -- ayni sira kurali:
    # pano kisa_vade.json'u okuyor, dosya panodan sonra yazilirsa pano her gun
    # bir gun eskisini gosterir. Ayrica burada zaten elimizde olan `bundles`
    # kullaniliyor; ayri bir onbellek turu gereksiz olurdu.
    _kisa_vade_uret(bundles)

    html_path = report.write_html(result, diag, OUT / "dashboard.html", top_n=args.top)
    csv_path = report.write_csv(result, OUT / "ranking.csv")
    llm_path = ml.export_for_llm(result, diag, OUT / "llm_export.json", top_n=args.top)

    (OUT / "diagnostics.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # --- Calisma durumu (bulgu O7: sessiz basarisizlik) -----------------------
    # Gunluk is zamanlayiciyla calisip basarisiz olursa pano eski veriyle acik
    # kalir ve bunu belli eden bir sey olmaz. Bu dosya son durumu kayit altina
    # alir; pano da yasini gosterir.
    write_status(ok=True, detail={
        "universe": len(tickers),
        "fetched": fetched_live,
        "backfilled": backfilled,
        "fetch_success_rate": round(ok_rate / 100, 3),
        "scored": int(len(result)),
        "rate_limited": rate_limited,
        "aborted": aborted,
        "universe_coverage_pct": cov["coverage_pct"],
    })

    # --- Konsol ozeti
    print("\n" + "=" * 74)
    print("PARAMETRE ETKI PUANLARI (yuksekten dusuge)")
    print("=" * 74)
    for f in diag["active_factors"]:
        print(f"  {f['weight']:5.1f}  {f['name_tr'][:52]:<52} kapsama %{f['coverage']*100:.0f}")
    if diag["auto_disabled"]:
        print("\n  OTOMATIK DEVRE DISI (veri kapsamasi yetersiz):")
        for d in diag["auto_disabled"]:
            print(f"    - {d['name_tr']}: %{d['coverage']*100:.1f}")

    print("\n" + "=" * 74)
    print(f"EN YUKSEK TOPLAM ETKI PUANI — ILK {min(args.top, 20)}")
    print("=" * 74)
    print(f"  {'#':>3} {'SEMBOL':<8} {'PUAN':>6} {'FIYAT':>9} {'12A%':>7}  SEKTOR")
    for _, r in result.head(min(args.top, 20)).iterrows():
        r12 = (r.get("returns") or {}).get("12m")
        print(f"  {int(r['rank']):>3} {r['ticker']:<8} {r['total_score']:>6.1f} "
              f"{(r['price'] or 0):>9.2f} {(r12*100 if r12 is not None else 0):>7.1f}"
              f"  {(r.get('sector') or '')[:26]}")

    print("\n" + "=" * 74)
    print(f"Sure: {time.time()-t0:.0f} sn")
    print(f"  Pano (ac):      {html_path}")
    print(f"  Siralama CSV:   {csv_path}")
    print(f"  LLM JSON:       {llm_path}")
    print(f"  ML anlik goru.: {snap_path}")
    print(f"  Tani:           {OUT / 'diagnostics.json'}")
    return 0


FACTOR_IC = ROOT / "data" / "faktor_ic.json"


def _zaman_analizi(labeled: "pd.DataFrame", cfg: dict, factor_ids: list,
                   label_col: str, horizon: int, source: str) -> None:
    """Parametre gucunu zamana ve piyasa rejimine gore kirar, diske yazar.

    Rejim etiketi endeksin KENDI fiyat gecmisinden gecmise donuk uretilir
    (regime.labels_for_dates). Boylece gecmise donuk panelin tarihleri de
    canli gunlerle ayni kuralla etiketlenir. Endeks gecmisi yoksa analiz
    rejimsiz yapilir -- anlamlilik ve zayiflama kismi yine calisir.
    """
    from src import faktor_zaman as fz
    from src import regime as rg

    try:
        dates = sorted(labeled["snapshot_date"].dropna().astype(str)
                       .str.slice(0, 10).unique())
        rejim = {}
        try:
            bench = yahoo.fetch_benchmark("SPY", "2y", use_cache=True)
            close = bench["Close"] if bench is not None and "Close" in bench else None
            rejim = rg.labels_for_dates(close, dates)
        except Exception:
            rejim = {}

        weights = {f["id"]: float(f.get("weight", 0)) for f in cfg["factors"]}
        yonler = {f["id"]: str(f.get("direction", "higher_better"))
                  for f in cfg["factors"]}
        payload = fz.analyze(labeled, factor_ids, label_col, horizon,
                             weights=weights, rejim=rejim, directions=yonler,
                             source=source)
        fz.save(payload)
        print()
        fz.print_table(payload)
    except Exception as e:                       # olcum, taramayi dusurmemeli
        print(f"Zaman/rejim analizi yapilamadi: {e}", file=sys.stderr)


def _save_factor_ic(ic: "pd.DataFrame", cfg: dict, horizon: int, labeled: int,
                    dates: int, source: str) -> None:
    """Parametre bazli IC tablosunu kalici olarak saklar (pano bunu okur).

    Yaninda YAPILANDIRILMIS agirlik da tasinir: kullanicinin gormesi gereken
    sey tek basina IC degil, "bu parametreye verdigim agirlik olculen gucuyle
    uyumlu mu" karsilastirmasidir. Agirlik ONERISI uretilir ama HICBIR SEY
    otomatik degistirilmez -- degisiklik kullanicinin onayiyla yapilir.
    """
    weights = {f["id"]: float(f.get("weight", 0)) for f in cfg["factors"]}
    rows = []
    for r in ic.to_dict("records"):
        fid = r["factor"]
        w = weights.get(fid, 0.0)
        icv = r.get("ic_mean") or 0.0
        if icv <= 0.005:
            oneri = "agirlik dusurulmeli veya kaldirilmali"
        elif icv >= 0.05 and w < 5:
            oneri = "agirlik artirilabilir"
        else:
            oneri = "mevcut agirlik makul"
        rows.append({**r, "weight": w, "suggestion_tr": oneri})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon": horizon,
        "source": source,
        "labeled_rows": labeled,
        "dates": dates,
        "factors": rows,
        "note_tr": ("Panel kaynakli olcumler hayatta kalma yanliligi tasir ve "
                    "yalnizca fiyat turevi parametreleri kapsar."
                    if source == "panel" else
                    "Gercek taramalardan olculmustur."),
    }
    try:
        FACTOR_IC.parent.mkdir(parents=True, exist_ok=True)
        FACTOR_IC.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\n  Parametre IC tablosu kaydedildi: {FACTOR_IC.name}")
    except OSError:
        pass


def cmd_learn(args: argparse.Namespace) -> int:
    """Biriken anlik goruntuleri ileri getiriyle etiketle ve agirliklari ogren."""
    cfg = load_config(Path(args.config))
    factor_ids = [f["id"] for f in cfg["factors"]]

    # --pretrain: gecmise donuk panelden olc. Canli magaza dolana kadar
    # "hangi parametre gercekten calisiyor" sorusuna bugun cevap verebilmenin
    # tek yolu bu. Yanlilik tasir, cikti da bunu isaretler.
    store = None
    if getattr(args, "pretrain", False):
        from src import backfill as _bf
        store = _bf.BACKFILL_STORE

    snaps = ml.load_all_snapshots(store)
    if snaps.empty:
        print("HATA: feature store bos. Once birkac kez 'python run.py' calistir.\n"
              "      Anlamli sonuc icin en az 2-3 ay, haftada bir tarama onerilir.", file=sys.stderr)
        return 1

    dates = sorted(snaps["snapshot_date"].dropna().unique())
    print(f"{len(snaps)} satir, {len(dates)} farkli tarih ({dates[0]} -> {dates[-1]})")

    print(f"Ileri getiriler etiketleniyor (ufuk: {args.horizon} gun)...")
    labeled = ml.label_forward_returns(snaps, horizon_days=args.horizon,
                                       use_cache=not args.no_cache)
    label_col = f"fwd_return_{args.horizon}d_excess"
    n_lab = int(labeled[label_col].notna().sum())
    print(f"{n_lab} satir etiketlendi (gerisi icin ufuk henuz dolmadi)")

    if n_lab < 30:
        print("\nUYARI: etiketli veri cok az; sonuclar guvenilir degil.\n"
              "Daha fazla tarama biriktir, sonra tekrar calistir.", file=sys.stderr)

    yonler = {f["id"]: str(f.get("direction", "higher_better"))
              for f in cfg["factors"]}
    ic = ml.information_coefficients(labeled, factor_ids, label_col, yonler)
    if not ic.empty:
        print("\n" + "=" * 66)
        print("BILGI KATSAYILARI (IC) — faktorlerin gercek ongoru gucu")
        print("=" * 66)
        print(f"  {'FAKTOR':<32} {'IC':>8} {'ICIR':>8} {'DONEM':>7}")
        for _, r in ic.iterrows():
            print(f"  {r['factor'][:32]:<32} {r['ic_mean']:>8.4f} "
                  f"{(r['icir'] if r['icir'] is not None else float('nan')):>8.2f} {r['periods']:>7}")
        print("\n  Yorum: |IC|>0.03 zayif-kullanilabilir, >0.05 iyi, >0.10 cok iyi")

        # Panonun okudugu KALICI dosya. learned_weights.json yalnizca agirlik
        # ogrenmesi basariliysa yaziliyordu; oysa IC tablosu kendi basina en
        # degerli cikti -- "hangi parametre ise yariyor" sorusunun cevabi.
        _save_factor_ic(ic, cfg, args.horizon, n_lab, len(dates),
                        "panel" if store is not None else "canli")

    # --- Zaman/rejim kirilimi: ortalamanin gizledigi her sey
    #     Tek bir IC ortalamasi "bu parametre calisiyor mu" sorusuna cevap
    #     vermez; ortusen etiketler yuzunden siradan t degeri de sisik cikar.
    #     Ayrintili gerekce: src/faktor_zaman.py.
    _zaman_analizi(labeled, cfg, factor_ids, label_col, args.horizon,
                   "panel" if store is not None else "canli")

    weights = ml.learn_weights(labeled, factor_ids, label_col, method=args.method)
    if not weights:
        print("\nAgirlik ogrenilemedi (yetersiz veri veya pozitif IC yok).", file=sys.stderr)
        print("IC tablosu yine de kaydedildi.", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": args.method, "horizon_days": args.horizon,
        "labeled_rows": n_lab, "snapshot_dates": len(dates),
        "weights": weights,
        "information_coefficients": ic.to_dict("records") if not ic.empty else [],
    }
    path = OUT / "learned_weights.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print("OGRENILEN AGIRLIKLAR")
    print("=" * 66)
    for k, v in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {v:6.2f}  {k}")
    print(f"\nKaydedildi: {path}")
    print("Kullanmak icin: python run.py --use-learned")
    return 0


def _latest_scores() -> dict[str, float]:
    """Son taramadaki toplam etki puanlarini okur (giris puani kaydetmek icin)."""
    p = OUT / "ranking.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        return {str(r["ticker"]): float(r["total_score"])
                for _, r in df.iterrows()
                if pd.notna(r.get("total_score"))}
    except Exception:
        return {}


def cmd_watch(args: argparse.Namespace) -> int:
    from src import report_watch, watchlist

    action = args.action

    # ---------------------------------------------------------------- ekle
    if action == "add":
        if not args.ticker:
            print("HATA: sembol gerekli. Ornek: run.py watch add NVDA --price 219.40",
                  file=sys.stderr)
            return 1
        scores = _latest_scores()
        added_any = False
        for tk in [t.strip().upper() for t in args.ticker.split(",") if t.strip()]:
            _, is_new = watchlist.add(
                tk, entry_price=args.price, quantity=args.qty, note=args.note or "",
                score_at_entry=scores.get(tk))
            added_any = True
            s = scores.get(tk)
            print(f"  {'eklendi' if is_new else 'guncellendi'}: {tk}"
                  + (f"  giris {args.price}" if args.price else "  (pozisyon yok, izleniyor)")
                  + (f"  giris puani {s:.1f}" if s is not None else ""))
        if added_any:
            print(f"\nListe: {watchlist.WATCHLIST}")
            print("Gunluk analiz icin: python run.py watch update")
        return 0

    # ------------------------------------------------------- panodan ice aktar
    if action == "import":
        src = Path(args.ticker or "")
        if not src.exists():
            print(f"HATA: dosya bulunamadi: {src}\n"
                  f"      Panodaki 'JSON indir' butonuyla olusturulan dosyayi ver.",
                  file=sys.stderr)
            return 1
        try:
            items = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"HATA: dosya okunamadi ({exc})", file=sys.stderr)
            return 1
        if isinstance(items, dict):
            items = items.get("positions", [])

        scores = _latest_scores()
        n_new = 0
        for it in items:
            tk = str(it.get("ticker", "")).strip().upper()
            if not tk:
                continue
            _, is_new = watchlist.add(
                tk, entry_price=it.get("entry_price"), quantity=it.get("quantity"),
                note=it.get("note", ""), score_at_entry=scores.get(tk))
            n_new += int(is_new)
            print(f"  {'eklendi' if is_new else 'guncellendi'}: {tk}"
                  + (f"  giris {it['entry_price']}" if it.get("entry_price") else ""))
        print(f"\n{len(items)} kayit islendi ({n_new} yeni).")
        print("Gunluk analiz icin: python run.py watch update")
        return 0

    # ---------------------------------------------------------------- sil
    if action == "remove":
        if not args.ticker:
            print("HATA: sembol gerekli.", file=sys.stderr)
            return 1
        for tk in [t.strip().upper() for t in args.ticker.split(",") if t.strip()]:
            print(f"  {'silindi' if watchlist.remove(tk) else 'bulunamadi'}: {tk}")
        return 0

    # ---------------------------------------------------------------- listele
    positions = watchlist.load()
    if action == "list":
        if not positions:
            print("Izleme listesi bos.\n"
                  "  Ekle: python run.py watch add NVDA --price 219.40")
            return 0
        print(f"{'SEMBOL':<8} {'GIRIS':>10} {'ADET':>8} {'EKLENME':>12}  NOT")
        for p in positions:
            ep = f"{p['entry_price']:.2f}" if p.get("entry_price") else "-"
            q = f"{p['quantity']:g}" if p.get("quantity") else "-"
            print(f"{p['ticker']:<8} {ep:>10} {q:>8} {p.get('added_date',''):>12}  "
                  f"{p.get('note','')}")
        return 0

    # ---------------------------------------------------------------- guncelle
    if not positions:
        print("Izleme listesi bos — guncellenecek pozisyon yok.\n"
              "  Ekle: python run.py watch add NVDA --price 219.40", file=sys.stderr)
        return 1

    # Guncel toplam puanlari son taramadan tasi (skor bozulmasi tespiti icin)
    scores = _latest_scores()
    for p in positions:
        if p["ticker"] in scores:
            p["score_now"] = scores[p["ticker"]]

    print(f"[1/3] {len(positions)} pozisyon icin taze veri cekiliyor...")
    results = watchlist.update(positions, use_cache=args.use_cache)
    good = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    print(f"      {len(good)} basarili" + (f", {len(bad)} hatali" if bad else ""))
    for r in bad:
        print(f"      ! {r['ticker']}: {r.get('reason')}")

    print("[2/3] Gecmise kaydediliyor...")
    hist_path = watchlist.append_history(watchlist.to_history_rows(results))
    history = watchlist.load_history()

    print("[3/3] Pano olusturuluyor...")
    OUT.mkdir(parents=True, exist_ok=True)
    html = report_watch.write_html(results, history, OUT / "watchlist.html")

    # --- konsol ozeti
    print("\n" + "=" * 78)
    print("IZLEME LISTESI — RISK SIRASIYLA")
    print("=" * 78)
    print(f"  {'SEMBOL':<7} {'FIYAT':>9} {'K/Z':>8} {'STOP':>9} "
          f"{'KISA H.':>9} {'UZUN H.':>9}  DURUM")
    for r in good:
        a = r["analysis"]
        st = a.get("short_term", {})
        lt = a.get("long_term", {})
        pnl = a.get("pnl_pct")
        print(f"  {r['ticker']:<7} {a['price']:>9.2f} "
              f"{(f'{pnl:+.1f}%' if pnl is not None else '-'):>8} "
              f"{(a.get('stops') or {}).get('active_stop', float('nan')):>9.2f} "
              f"{(st.get('target') if st.get('available') else float('nan')):>9.2f} "
              f"{(lt.get('target') if lt.get('available') else float('nan')):>9.2f}"
              f"  {a['risk_level']}")

    alerts = [r for r in good
              if r["analysis"]["risk_level"] in ("SAT", "YUKSEK_RISK")]
    if alerts:
        print("\n" + "=" * 78)
        print("DIKKAT GEREKTIRENLER")
        print("=" * 78)
        for r in alerts:
            a = r["analysis"]
            print(f"\n  {r['ticker']} — {a['risk_level_tr']}")
            print(f"     {a['action_tr']}")
            for s in a["signals"][:4]:
                print(f"       [{s['siddet']}] {s['baslik']}")

    # --- Bildirim: stop kirilmasi ve satis sinyali ziyaret bekleyemez -------
    try:
        from src import notify
        prev_top = _previous_top(10)
        ranking = None
        rp = OUT / "ranking.csv"
        if rp.exists():
            try:
                ranking = pd.read_csv(rp)
            except Exception:
                ranking = None
        built = notify.build_alerts(good, ranking=ranking, previous_top=prev_top)
        sent = notify.send(built)
        if sent.get("sent"):
            kanal = ", ".join(k for k, v in sent["channels"].items() if v) or "yok"
            print(f"\n  Bildirim: {sent['sent']} uyari gonderildi ({kanal})")
    except Exception as exc:
        print(f"  UYARI: bildirim gonderilemedi ({exc})")

    print("\n" + "=" * 78)
    print(f"  Pano:   {html}")
    print(f"  Gecmis: {hist_path}")
    return 0


def run_stage(name: str, fn, essential: bool, degraded: list[str]) -> int:
    """Bir gunluk is adimini yalitilmis calistirir.

    NEDEN: 15-17 Agustos'ta OGRENME adimindaki KOZMETIK bir print satiri
    (bozuk bir karakter yuzunden) istisna atti ve o gune ait BASARIYLA
    tamamlanmis taramanin tamamini gecersiz kildi -- cikis kodu 1 oldu, gun
    isaretlenmedi, sekiz tetigin hepsi ayni taramayi bastan yapti ve Yahoo
    kotasini bosa harcadi.

    Ders: bir adimin cokusu, digerlerinin urettigi degeri silmemeli. Yalnizca
    ZORUNLU adimlarin (tarama) basarisizligi gunu basarisiz yapar; digerleri
    gurultuyle raporlanir ama gunu tekrar tetiklemez.
    """
    import traceback

    try:
        return int(fn() or 0)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        traceback.print_exc()
        etiket = "ZORUNLU" if essential else "ikincil"
        print(f"\nUYARI: '{name}' adimi coktu ({etiket}): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        degraded.append(name)
        return 1 if essential else 0


def cmd_daily(args: argparse.Namespace) -> int:
    """Gunluk tam dongu: taze tarama + izleme listesi analizi.

    Gorev Zamanlayici'ya baglanacak tek komut budur.

    Adimlar birbirinden YALITILMISTIR: ayrintisi run_stage'de.
    """
    from src import watchlist

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 78)
    print(f"GUNLUK CALISMA — {stamp}")
    print("=" * 78)

    degraded: list[str] = []

    # Bayat fiyatla gunluk karar verilmez: once bos/bozuk kayitlari at.
    purged = run_stage("onbellek temizligi",
                       lambda: cache.purge_invalid("yahoo"), False, degraded)
    if purged:
        print(f"[on hazirlik] {purged} bos/bozuk onbellek kaydi temizlendi\n")

    print(">>> 1/3  TARAMA\n")
    rc = run_stage("tarama", lambda: cmd_scan(args), True, degraded)
    if rc != 0:
        print("\nUYARI: tarama basarisiz; izleme listesi yine de guncellenecek.",
              file=sys.stderr)

    rc2 = 0
    if not watchlist.load():
        print("\n>>> 2/3  IZLEME LISTESI — bos, atlandi")
        print("     Panodan '+ EKLE' ile hisse sec, sonra:")
        print("     python run.py watch add <SEMBOL> --price <FIYAT>")
    else:
        print("\n" + ">>> 2/3  IZLEME LISTESI\n")
        watch_args = argparse.Namespace(action="update", ticker=None, price=None,
                                        qty=None, note="", use_cache=False)
        rc2 = run_stage("izleme listesi", lambda: cmd_watch(watch_args),
                        False, degraded)

    run_stage("ogrenme dongusu", lambda: _daily_learning(args), False, degraded)
    run_stage("haftalik yedek", _weekly_backup, False, degraded)

    print("\n" + "=" * 78)
    print("GUNLUK CALISMA TAMAMLANDI" if not degraded
          else f"GUNLUK CALISMA TAMAMLANDI ({len(degraded)} adim eksik)")
    if degraded:
        print(f"  Eksik adimlar: {', '.join(degraded)}")
    print(f"  Tarama panosu : {OUT / 'dashboard.html'}")
    print(f"  Izleme panosu : {OUT / 'watchlist.html'}")
    return rc or rc2


def _weekly_backup() -> int:
    """Haftada bir sifreli yedek alir.

    Feature store yeniden uretilemez ve her gun buyuyor. Yedek elle
    calistirilmaya birakilirsa alinmaz -- gunluk isin bir parcasi olmali.
    Her gun almak gereksiz (gunde ~2 MB), haftada bir yeterli.

    Parola ortam degiskeninde yoksa SESSIZCE atlanir: gunluk is parola
    soramaz (etkilesimli degil) ve bu yuzden gunu basarisiz saymamali.
    """
    import os

    from src import backup as bk

    pw = os.environ.get("DASHBOARD_PASSWORD")
    if not pw:
        return 0

    last = bk.latest()
    if last:
        age_days = (time.time() - last.stat().st_mtime) / 86400
        if age_days < 7:
            return 0

    print("\n>>> HAFTALIK YEDEK\n")
    r = bk.create(pw)
    print(f"     {r['path']}  ({r['mb']} MB, {r['summary']['files']} dosya)")
    removed = bk.prune(keep=8)
    if removed:
        print(f"     eski yedek silindi: {len(removed)} dosya")
    return 0


def _daily_learning(args: argparse.Namespace) -> int:
    """Gunluk isin ogrenme adimi.

    Kendi kendini besleyen kisim: veri yeterliyse periyodik olarak yeniden
    egitir ve yalnizca esikleri gecen modeli terfi ettirir. Yetersizse
    sessizce ilerlemeyi bildirir — her gun bosuna egitmez.
    """
    from src import dataset as _ds
    from src import training as _tr

    ready = _ds.readiness(getattr(args, "horizon", 21))
    print("\n" + ">>> 3/3  OGRENME DONGUSU\n")
    if not ready["ready_to_train"]:
        print(f"     Veri birikiyor: {ready['snapshots']}/{ready['need_snapshots']} "
              f"anlik goruntu, {ready['span_days']}/{ready['need_span_days']} gun "
              f"(%{ready['progress_pct']})")
        print("     Egitim, esik asilinca kendiliginden baslayacak.")
        from src import backfill as _bf
        _bm = _bf.info()
        if _bm:
            print(f"     Beklerken: gecmise donuk panel hazir ({_bm['snapshots']} "
                  f"goruntu) -> 'python run.py ml train --pretrain'")
        else:
            print("     Beklemeden mimari denemek icin: python run.py history")
    elif args.no_train:
        print("     --no-train verildi, egitim atlandi.")
    else:
        champ = _tr.champion()
        last = (champ or {}).get("promoted_at", "")[:10]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        due = (not champ) or (last < today and
                              ready["snapshots"] % max(1, args.retrain_every) == 0)
        if not due:
            print(f"     Sampiyon guncel ({champ['model']}, agirlik {champ['weight']}). "
                  f"Yeniden egitim {args.retrain_every} taramada bir.")
        else:
            # Parametre IC tablosunu da tazele: pano bunu okuyor ve canli
            # veriyle olculdugu anda panel kaynakli (yanlili) surumun yerini
            # almasi gerekiyor.
            try:
                learn_args = argparse.Namespace(
                    config=str(ROOT / "config" / "weights.yaml"),
                    horizon=getattr(args, "horizon", 21), method="ic",
                    pretrain=False, no_cache=False)
                cmd_learn(learn_args)
            except Exception as exc:
                print(f"     UYARI: parametre IC tablosu tazelenemedi ({exc})")

            print(f"     Yeniden egitim basliyor (ufuk {getattr(args,'horizon',21)})...")
            train_args = argparse.Namespace(
                ml_action="train", models=None, horizon=getattr(args, "horizon", 21),
                splits=5, embargo=5, window=10, promote=True, force=False,
                no_cache=False, pretrain=False, min_rows=30,
                horizons=None, no_ensemble=False)
            cmd_ml(train_args)
    return 0


def _print_paper(s: dict, baslik: str) -> None:
    if not s.get("ok"):
        print(f"  {baslik}: {s.get('reason')}"
              + (f" ({s['bekleyen']} pozisyon bekliyor)" if s.get("bekleyen") else ""))
        return
    print(f"\n  {baslik}")
    print(f"    donem        : {s['first_date']} -> {s['last_date']}  "
          f"({s['cohorts']} kohort, {s['positions']} pozisyon)")
    print(f"    ortalama     : %{s['mean_pct']:+.2f}   "
          f"(SPY %{s['bench_mean_pct']:+.2f})" if s.get("bench_mean_pct") is not None
          else f"    ortalama     : %{s['mean_pct']:+.2f}")
    if s.get("excess_pct") is not None:
        print(f"    ENDEKS FARKI : %{s['excess_pct']:+.2f}   "
              f"(pozisyonlarin %{s['excess_positive_pct']}'i endeksi yendi)")
    print(f"    isabet       : %{s['hit_rate_pct']} pozitif  "
          f"(kazanan ort %{s['avg_win_pct']}, kaybeden ort %{s['avg_loss_pct']})")
    print(f"    dagilim      : en iyi %{s['best_pct']}, en kotu %{s['worst_pct']}, "
          f"std %{s['std_pct']}")
    if s.get("t_stat") is not None:
        guc = "anlamli" if abs(s["t_stat"]) >= 2 else "gurultuden ayirt EDILEMEZ"
        print(f"    t (kohort)   : {s['t_stat']}  -> {guc}")
    if s.get("bias_warning"):
        print(f"    UYARI: {s['bias_warning']}")


def cmd_paper(args: argparse.Namespace) -> int:
    """Kagit uzerinde portfoy defteri — sistemin kendi karnesi."""
    from src import paper

    if args.paper_action in ("build",):
        print("Defter dolduruluyor...")
        r1 = paper.build_from_feature_store(top_n=args.top)
        print(f"  gercek anlik goruntuler : "
              + (f"{r1['dates']} tarih, {r1['added']} pozisyon" if r1.get("ok")
                 else r1.get("reason")))
        if args.panel:
            cfg = load_config(Path(args.config))
            print("  geriye donuk panel siralaniyor (birkac dakika surebilir)...")
            r2 = paper.build_from_panel(cfg, top_n=args.top)
            print(f"  gecmise donuk panel     : "
                  + (f"{r2['dates']} tarih, {r2['added']} pozisyon" if r2.get("ok")
                     else r2.get("reason")))
        else:
            print("  gecmise donuk panel     : atlandi (--panel ile eklenir)")

    if args.paper_action in ("build", "mark"):
        print("\nPozisyonlar degerleniyor (onbellekten, ag istegi yok)...")
        m = paper.mark()
        if not m.get("ok"):
            print(f"BASARISIZ: {m.get('reason')}", file=sys.stderr)
            return 1
        print(f"  {m['positions']} pozisyon degerlendi"
              + (f", {m['missing_price']} sembolun fiyati onbellekte yok"
                 if m["missing_price"] else ""))

    out = paper.refresh(horizon=args.horizon)
    if not out.get("marked", {}).get("ok", True) and "live" not in out:
        print(f"BASARISIZ: {out.get('reason')}", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print(f"KAGIT UZERINDE DEFTER — ilk {args.top}, {args.horizon} islem gunu tutma")
    print("=" * 78)
    _print_paper(out.get("live") or {}, "GERCEK TARAMALAR (yanlilik yok)")
    _print_paper(out.get("panel") or {}, "GECMISE DONUK PANEL (ust sinir)")
    print(f"\n  Defter: {paper.COHORTS}")
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    """Onbellekteki fiyat gecmisinden gecmise donuk anlik goruntu uretir."""
    from src import backfill as bf

    if args.merge_only:
        meta = bf.materialize(step=args.step, horizon=args.horizon)
        if not meta.get("ok"):
            print(f"BASARISIZ: {meta.get('reason')}", file=sys.stderr)
            return 1
        print(f"Birikmis yiginlar panele cevrildi: {meta['snapshots']} anlik "
              f"goruntu, {meta['rows']} satir, {meta['tickers_used']} hisse")
        print(f"  aralik: {meta['first_date']} -> {meta['last_date']}")
        print("  (uretim devam ediyorsa bitince bu dosyalar tazelenir)")
        return 0

    tickers = sorted(scanlog.load().keys())
    if args.limit:
        tickers = tickers[: args.limit]

    print("=" * 74)
    print("GECMISE DONUK ANLIK GORUNTU URETIMI")
    print("=" * 74)
    print(f"  sembol           : {len(tickers)}")
    print(f"  izgara           : her {args.step} islem gununde bir, "
          f"en fazla {args.snapshots} goruntu")
    print(f"  faktor           : {len(bf.PIT_FACTORS)} (yalnizca fiyat/hacim turevi)")
    print("  temel veri       : DAHIL DEGIL — gecmise donuk bilinemez, "
          "eklemek gelecege bakis olurdu")
    print()

    t0 = time.time()
    try:
        meta = bf.build(step=args.step, max_snapshots=args.snapshots,
                        horizon=args.horizon, workers=args.workers,
                        tickers=tickers, resume=not args.restart)
    except KeyboardInterrupt:
        print("\n  Kesildi. Islenen semboller kaydedildi; ayni komut kaldigi "
              "yerden devam eder.", file=sys.stderr)
        return 130
    if not meta.get("ok"):
        print(f"BASARISIZ: {meta.get('reason')}", file=sys.stderr)
        return 1

    print()
    print(f"  anlik goruntu    : {meta['snapshots']}")
    print(f"  toplam satir     : {meta['rows']:,}".replace(",", "."))
    print(f"  hisse (kullanildi/atlandi): {meta['tickers_used']} / "
          f"{meta['tickers_skipped']}")
    print(f"  goruntu basina   : ~{meta['rows_per_snapshot_median']} hisse")
    print(f"  tarih araligi    : {meta['first_date']} -> {meta['last_date']}")
    print(f"  sure             : {time.time() - t0:.0f} sn")
    print(f"  depo             : {meta['store']}")
    if not meta.get("complete"):
        print()
        print(f"  YARIM: {meta['tickers_remaining']} sembol kaldi. Panel su "
              f"haliyle kullanilabilir; ayni komutu tekrar calistirinca "
              f"kaldigi yerden devam eder.")
    print()
    print("  UYARI: " + meta["bias_warning"])
    print()
    print("  Simdi: python run.py ml train --pretrain --models ridge,mlp,seq")
    return 0


def cmd_ml(args: argparse.Namespace) -> int:
    """Derin ogrenme / geri besleme dongusu komutlari."""
    from src import dataset as ds
    from src import models as mz
    from src import training as tr

    # --- COKLU UFUK ---------------------------------------------------------
    # Tek ufuk (21 gun) yalnizca bir soruyu soruyordu. Ayni veriden 5, 21 ve 63
    # gunluk ufuklari birden olcmek uc kat kanit uretir ve daha onemlisi
    # SINYALIN OMRUNU gosterir: 5 gunde var olup 63 gunde kaybolan bir sinyal
    # kisa vadeli bir etkidir; tersi degerleme etkisidir. Bu ayrim, hangi
    # parametrenin neden calistigini anlamanin en kestirme yolu.
    if getattr(args, "horizons", None) and args.ml_action in ("train", "evaluate"):
        hs = []
        for part in str(args.horizons).split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= 250:
                hs.append(int(part))
        if not hs:
            print("HATA: --horizons gecersiz (orn. 5,21,63)", file=sys.stderr)
            return 1

        rows = []
        for h in sorted(set(hs)):
            print("\n" + "#" * 74)
            print(f"# UFUK {h} ISLEM GUNU")
            print("#" * 74)
            sub_args = argparse.Namespace(**{**vars(args), "horizons": None,
                                             "horizon": h})
            rc = cmd_ml(sub_args)
            rows.append((h, rc))

        print("\n" + "=" * 74)
        print("UFUK KARSILASTIRMASI")
        print("=" * 74)
        for h, rc in rows:
            print(f"  {h:>3} gun : {'tamamlandi' if rc == 0 else 'basarisiz'}")
        print("\n  Sinyal kisa ufukta guclu, uzun ufukta zayifsa: kisa vadeli bir")
        print("  etki (momentum/haber). Tersi ise degerleme etkisidir. Ikisinde de")
        print("  yoksa sinyal yoktur.")
        return 0 if all(rc == 0 for _, rc in rows) else 1

    action = args.ml_action
    store = None
    if getattr(args, "pretrain", False):
        from src import backfill as bf
        store = bf.BACKFILL_STORE

    # ------------------------------------------------------------- durum
    if action == "status":
        ready = ds.readiness(args.horizon)
        champ = tr.champion()
        print("=" * 74)
        print("OGRENME SISTEMI DURUMU")
        print("=" * 74)
        print(f"  torch                : {'var ' + (mz.describe()['torch_version'] or '') if mz.torch_available() else 'YOK'}")
        print(f"  kullanilabilir model : {', '.join(mz.describe()['models'])}")
        print()
        print(f"  anlik goruntu        : {ready['snapshots']}  (gereken {ready['need_snapshots']})")
        print(f"  veri araligi         : {ready['span_days']} gun  (gereken {ready['need_span_days']})")
        print(f"  ufuk                 : {ready['horizon']} islem gunu")
        print(f"  etiketlenebilir gun  : {ready['labelable_days']}")
        print(f"  ILERLEME             : %{ready['progress_pct']}")
        print()
        print(f"  egitime hazir        : {'EVET' if ready['ready_to_train'] else 'hayir'}")
        print(f"  dogrulamaya hazir    : {'EVET' if ready['ready_to_validate'] else 'hayir'}")
        if not ready["ready_to_train"]:
            print(f"\n  Eksik: {ready['missing_snapshots']} anlik goruntu, "
                  f"{ready['missing_days']} gun")
            print("  Her is gunu 'python run.py daily' calistir; sayac kendiliginden ilerler.")

        from src import backfill as _bf
        bmeta = _bf.info()
        print()
        if bmeta:
            print(f"  ON EGITIM PANELI     : {bmeta['snapshots']} goruntu, "
                  f"{bmeta['rows']} satir")
            print(f"    aralik             : {bmeta['first_date']} -> {bmeta['last_date']}")
            print(f"    faktor             : {len(bmeta['factors'])} (fiyat turevi)")
            print("    Beklemeden mimari denemek icin: "
                  "python run.py ml train --pretrain")
        else:
            print("  ON EGITIM PANELI     : yok")
            print("    Onbellekteki 2 yillik fiyat gecmisinden gecmise donuk panel")
            print("    uretilebilir: python run.py history")

        print()
        if champ:
            print(f"  SAMPIYON MODEL       : {champ['model']} (ufuk {champ['horizon']})")
            print(f"    IC {champ.get('ic')}  ICIR {champ.get('icir')}  "
                  f"katman {champ.get('folds')}")
            print(f"    skor agirligi      : {champ.get('weight')}")
            print(f"    terfi tarihi       : {champ.get('promoted_at')}")
        else:
            print("  SAMPIYON MODEL       : yok — hicbir model esikleri gecmedi")
            print("    Model, kanit olmadan skorlamaya KATILMAZ (guvenlik freni).")
        return 0

    # ------------------------------------------------------------- egitim
    if action in ("train", "evaluate"):
        ready = ds.readiness(args.horizon, store=store)
        if store is not None:
            from src import backfill as bf
            bmeta = bf.info()
            if bmeta is None:
                print("Gecmise donuk panel yok. Once: python run.py history",
                      file=sys.stderr)
                return 1
            print(f"ON EGITIM MODU — {bmeta['snapshots']} gecmise donuk goruntu, "
                  f"{bmeta['first_date']} -> {bmeta['last_date']}")
            print("Bu modda SAMPIYON SECILMEZ (hayatta kalma yanliligi).\n")
        if not ready["ready_to_train"] and not args.force:
            print("EGITIM ENGELLENDI — yetersiz veri.\n", file=sys.stderr)
            print(f"  anlik goruntu : {ready['snapshots']} / {ready['need_snapshots']}")
            print(f"  veri araligi  : {ready['span_days']} / {ready['need_span_days']} gun")
            print(f"  ilerleme      : %{ready['progress_pct']}")
            print("\n  Az veriyle egitilen model, guvenilir GORUNEN ama tamamen")
            print("  gurultuye uydurulmus tahminler uretir. Sistemin en buyuk riski budur.")
            print("\n  Her is gunu 'python run.py daily' calistir.")
            print("  Yine de denemek istersen: --force (sonuclar guvenilmez)")
            return 1

        names = ([n.strip() for n in args.models.split(",") if n.strip()]
                 if args.models else list(mz.AVAILABLE))
        # Taban cizgisi her zaman once ve her zaman dahil
        names = ["ridge"] + [n for n in names if n != "ridge"]

        # Panel BIR KEZ kurulur, butun modeller ayni paneli kullanir.
        # Etiketleme onbellekte olmayan semboller icin aga cikiyor; her model
        # icin panel ayri kurulunca ayni ag isi model sayisi kadar tekrarliyordu
        # (dort modelde saatler). Panel modele bagli degil; modele ozel tek is
        # dizi modellerinin pencereye cevrilmesi ve o walk_forward icinde.
        from src import dataset as _dsp
        print()
        print("Panel kuruluyor (etiketleme ag onbellegine bagli, "
              "ilk kurulum uzun surebilir)...", flush=True)
        ortak_panel, panel_bilgi = _dsp.load_panel(
            horizon=args.horizon, use_cache=not args.no_cache, store=store,
            min_rows_per_date=args.min_rows)
        if ortak_panel is None:
            print(f"HATA: panel kurulamadi - {panel_bilgi.get('reason')}",
                  file=sys.stderr)
            return 1
        print(f"  {len(ortak_panel.y)} etiketli satir, "
              f"{panel_bilgi.get('dates_usable')} kullanilabilir tarih, "
              f"{panel_bilgi.get('features')} ozellik", flush=True)

        results: dict[str, dict] = {}
        for name in names:
            if name not in mz.AVAILABLE:
                print(f"  ! bilinmeyen model atlandi: {name}")
                continue
            print(f"\n>>> {name} egitiliyor (ufuk {args.horizon}, "
                  f"{args.splits} katman, embargo {args.embargo})...", flush=True)
            res = tr.walk_forward(name, horizon=args.horizon, n_splits=args.splits,
                                  embargo=args.embargo, window=args.window,
                                  use_cache=not args.no_cache, store=store,
                                  min_rows_per_date=args.min_rows,
                                  force=args.force,
                                  collect_predictions=not args.no_ensemble,
                                  panel=ortak_panel, panel_info=panel_bilgi)
            results[name] = res
            if not res.get("ok"):
                print(f"    BASARISIZ: {res.get('reason')}")
                continue
            print(f"    IC {res['ic_mean']}  ICIR {res['icir']}  "
                  f"katman {res['folds']} ({res['positive_folds']} pozitif)")
            print(f"    ilk-dilim getiri farki: {res['top_decile_spread']}")
            if res["folds"] < args.splits:
                # Sessizce az katman kurmak, sonucu oldugundan guvenilir
                # gosterir. Neden kurulamadigi soylenmeli.
                snaps = (res.get("panel") or {}).get("dates_usable", "?")
                print(f"    NOT: {args.splits} katman istendi, {res['folds']} "
                      f"kuruldu. Her katman icin test penceresinden geriye "
                      f"{args.horizon}+{args.embargo} gun arindiriliyor ve "
                      f"en az 20 egitim gunu gerekiyor; {snaps} anlik goruntu "
                      f"bu kadarina yetiyor. Daha fazla katman = daha fazla gun.")

        # --- Topluluk: uyelerin yuzdelik siralarinin ortalamasi -------------
        # Tek bir modeli secip digerlerini atmak, farkli hatalar yapan
        # tahmincilerin birbirini duzeltme imkanini bosa harciyordu.
        if not args.no_ensemble:
            ens = tr.ensemble(results)
            if ens.get("ok"):
                results["topluluk"] = ens
                print(f"\n>>> topluluk ({'+'.join(ens['members'])}) degerlendirildi")
                print(f"    IC {ens['ic_mean']}  ICIR {ens['icir']}  "
                      f"katman {ens['folds']} ({ens['positive_folds']} pozitif)")
            elif len([r for r in results.values() if r.get("ok")]) >= 2:
                print(f"\n>>> topluluk kurulamadi: {ens.get('reason')}")

        # Tahmin dizileri yalnizca topluluk icindi; ozet ciktilarda tasima.
        for r in results.values():
            r.pop("_predictions", None)

        base = results.get("ridge")
        print("\n" + "=" * 74)
        print("DEGERLENDIRME")
        print("=" * 74)
        print(f"  {'MODEL':<8} {'IC':>9} {'ICIR':>8} {'KATMAN':>7} {'DILIM FARKI':>13}  TERFI")
        best_name, best_dec = None, None
        for name, res in results.items():
            if not res.get("ok"):
                print(f"  {name:<9.9s}{'—':>9} {'—':>8} {'—':>7} {'—':>13}  hayir")
                continue
            dec = tr.promotion_check(res, baseline=base if name != "ridge" else None)
            tr.record_candidate(res, dec)
            print(f"  {name:<9.9s}{res['ic_mean']:>9.4f} {(res['icir'] or 0):>8.2f} "
                  f"{res['folds']:>7} {(res['top_decile_spread'] or 0):>13.4f}  "
                  f"{'EVET' if dec['promote'] else 'hayir'}")
            if dec["promote"] and (best_dec is None or
                                   (res["ic_mean"] or 0) > (results[best_name]["ic_mean"] or 0)):
                best_name, best_dec = name, dec

        if best_name is None:
            if store is not None:
                print("\n  On egitim modunda terfi YAPILMAZ — bu beklenen sonuc.")
                print("  Yukaridaki IC degerleri mimari karsilastirmasi icindir:")
                print("  hangi model turu bu problemde sinyal yakalayabiliyor?")
                print("  Gercek terfi, canli anlik goruntular birikince olur.")
                return 0
            print("\n  Hicbir model terfi esiklerini gecmedi.")
            for name, res in results.items():
                if res.get("ok"):
                    d = tr.promotion_check(res, baseline=base if name != "ridge" else None)
                    print(f"    {name}: {'; '.join(d['reasons'])}")
            print("\n  Bu KOTU bir sonuc degil — sistem, kanitlanmamis bir modeli")
            print("  skorlamaya sokmayi reddetti. Veri biriktikce tekrar dene.")
            return 0

        print(f"\n  EN IYI: {best_name}  onerilen skor agirligi "
              f"{best_dec['suggested_weight']}")
        if action == "train" and args.promote:
            entry = tr.promote(results[best_name], best_dec)
            print(f"  TERFI EDILDI -> sampiyon: {entry['model']} "
                  f"(agirlik {entry['weight']})")
            print("  Bir sonraki taramada 'model_score' parametresi devreye girecek.")
        else:
            print("  Terfi icin: python run.py ml train --promote")
        return 0

    return 1


def cmd_backup(args: argparse.Namespace) -> int:
    """Yeniden uretilemeyen veriyi sifreli arsivde dondurur."""
    from src import backup as bk
    from src import publish as pub

    if args.backup_action == "list":
        d = Path(args.dir) if args.dir else bk.BACKUP_DIR
        files = sorted(d.glob("yedek_*.hsy")) if d.exists() else []
        if not files:
            print(f"Yedek yok: {d}")
            return 0
        print("=" * 74)
        print(f"YEDEKLER — {d}")
        print("=" * 74)
        for p in files:
            meta = bk.inspect(p)
            mb = p.stat().st_size / 1024 / 1024
            print(f"  {p.name:<34} {mb:7.1f} MB  {meta.get('created_at', '?')[:16]}"
                  f"  {meta.get('summary', {}).get('files', '?')} dosya")
        return 0

    if args.backup_action == "restore":
        src = Path(args.file) if args.file else bk.latest(
            Path(args.dir) if args.dir else None)
        if not src or not src.exists():
            print("HATA: acilacak yedek bulunamadi (--file ile belirt).",
                  file=sys.stderr)
            return 1
        target = Path(args.target) if args.target else ROOT / "yedek_acilan"
        pw = pub.get_password()
        if not pw:
            print("HATA: parola alinamadi.", file=sys.stderr)
            return 1
        print(f"Aciliyor: {src.name} -> {target}")
        r = bk.restore(src, pw, target, overwrite=args.overwrite)
        if not r.get("ok"):
            print(f"BASARISIZ: {r['reason']}", file=sys.stderr)
            return 1
        print(f"  {r['written']} dosya yazildi"
              + (f", {r['skipped']} atlandi (zaten var)" if r["skipped"] else ""))
        print(f"  Yedek tarihi: {r.get('created_at')}")
        print("\n  NOT: calisan kuruluma DEGIL, ayri bir dizine acildi.")
        print("  Icerigi kontrol edip elle tasiyabilirsin.")
        return 0

    # --- olustur
    pw = pub.get_password()
    if not pw:
        print("HATA: parola alinamadi. DASHBOARD_PASSWORD ortam degiskenini "
              "ayarla veya sorulunca gir.", file=sys.stderr)
        return 1

    print("Yedek olusturuluyor...")
    try:
        r = bk.create(pw, Path(args.dir) if args.dir else None, label=args.label)
    except FileNotFoundError as exc:
        print(f"BASARISIZ: {exc}", file=sys.stderr)
        return 1

    s = r["summary"]
    print("=" * 74)
    print("YEDEK OLUSTURULDU")
    print("=" * 74)
    for part in s["parts"]:
        print(f"  {part['path']:<28} {part['files']:>6} dosya  {part['mb']:>8.2f} MB")
    if s["missing"]:
        print(f"  (yok, atlandi: {', '.join(s['missing'])})")
    print(f"\n  dosya   : {r['path']}")
    print(f"  boyut   : {r['mb']} MB  (sikistirilmamis {r['plain_mb']} MB)")
    print("  sifreli : AES-256-GCM + PBKDF2-SHA256 (600.000 tur)")
    print("\n  Parola kaybedilirse arsiv ACILAMAZ. Baska kurtarma yolu yok.")

    removed = bk.prune(keep=args.keep, out_dir=Path(args.dir) if args.dir else None)
    if removed:
        print(f"\n  Eski yedek silindi ({args.keep} tanesi tutuluyor): "
              f"{', '.join(removed)}")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    """Panolari sifreleyip tek dosyalik, herhangi bir yere konabilecek hale getirir."""
    from src import publish as pub

    targets = []
    for name, title in (("dashboard.html", "Hisse Siralama — Sifreli"),
                        ("watchlist.html", "Izleme Listesi — Sifreli")):
        src = OUT / name
        if src.exists():
            targets.append((src, OUT / f"secure_{name}", title))

    if not targets:
        print("HATA: yayinlanacak pano yok. Once 'python run.py daily' calistir.",
              file=sys.stderr)
        return 1

    pw = pub.get_password(confirm=args.set_password)
    if not pw:
        print("HATA: parola alinamadi.\n"
              "  Etkilesimli: python run.py publish\n"
              "  Otomatik   : DASHBOARD_PASSWORD ortam degiskenini ayarla",
              file=sys.stderr)
        return 1

    st = pub.password_strength(pw)
    if st["level"] == "ZAYIF" and not args.force:
        print(f"DURDURULDU: parola gucu {st['level']} (~{st['bits']} bit) — {st['note']}",
              file=sys.stderr)
        print("  Sifreli metin dosyanin icinde oldugu icin saldirgan sinirsiz\n"
              "  deneme yapabilir. En az 16 karakter veya 5 rastgele kelime kullan.\n"
              "  Yine de devam etmek icin: --force", file=sys.stderr)
        return 1

    print(f"Parola gucu: {st['level']} (~{st['bits']} bit, {st['length']} karakter)")
    for src, out, title in targets:
        info = pub.encrypt_html(src, pw, out, title=title)
        print(f"  {src.name:20s} -> {out.name:26s} "
              f"{info['source_kb']} KB -> {info['output_kb']} KB")

    print(f"\n{len(targets)} dosya sifrelendi (AES-256-GCM, "
          f"PBKDF2 {pub.PBKDF2_ITERATIONS:,} tur).")
    print("Bu dosyalari HERHANGI bir yere koyabilirsin — parola olmadan okunamaz.")
    print("Cozme islemi tamamen tarayicida yapilir; parola hicbir yere gonderilmez.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Panolari yerel agda yayinlar (telefondan/tabletten bakmak icin)."""
    import http.server
    import socket
    import socketserver

    root = OUT
    if not (root / "dashboard.html").exists():
        print("HATA: pano yok. Once 'python run.py daily' calistir.", file=sys.stderr)
        return 1

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def end_headers(self):
            # Pano her calistirmada degisir; tarayici eski surumu gostermesin
            self.send_header("Cache-Control", "no-store, must-revalidate")
            super().end_headers()

        def log_message(self, fmt, *a):
            pass

    ip = "127.0.0.1"
    if args.lan:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        except Exception:
            ip = "0.0.0.0"

    bind = "0.0.0.0" if args.lan else "127.0.0.1"
    page = "secure_dashboard.html" if (root / "secure_dashboard.html").exists() \
        else "dashboard.html"

    with socketserver.TCPServer((bind, args.port), Handler) as httpd:
        print(f"Pano yayinda:  http://{ip}:{args.port}/{page}")
        if args.lan:
            print("  Ayni Wi-Fi agindaki cihazlardan bu adrese girebilirsin.")
            if page.startswith("secure_"):
                print("  Sifreli surum yayinlaniyor — parola sorulacak.")
            else:
                print("  UYARI: sifresiz surum yayinlaniyor. Once "
                      "'python run.py publish' calistirmani oneririm.")
        print("Durdurmak icin Ctrl+C")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDurduruldu.")
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    """Sifreli panolari GitHub Pages'e gonderir."""
    from src import deploy

    try:
        info = deploy.build()
    except deploy.LeakDetected as exc:
        print(f"YAYIN DURDURULDU — {exc}", file=sys.stderr)
        print("\n  Bu bir guvenlik korumasidir: sifrelenmemis icerik herkese acik\n"
              "  bir depoya gonderilmek uzereydi. Once 'python run.py publish'\n"
              "  calistirip sifreli surumu uret.", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print("Yayin dizini hazir (yalnizca sifreli dosyalar):")
    for f in info["files"]:
        print(f"  {f['file']:24s} {f['kb']:>7} KB   PBKDF2 {f['iterations']:,} tur")

    if args.build_only:
        print(f"\nDizin: {info['dir']}")
        print("Gondermek icin: python run.py deploy --repo kullanici/depo")
        return 0

    if not args.repo:
        print("\nHATA: --repo gerekli (ornek: --repo yigaykut/hisse-pano)",
              file=sys.stderr)
        return 1

    print(f"\n{args.repo} deposuna gonderiliyor...")
    res = deploy.git_push(args.repo, branch=args.branch)
    for line in res["log"][-4:]:
        print("  " + line.replace("\n", "\n  "))

    if not res["ok"]:
        print("\nGONDERIM BASARISIZ.", file=sys.stderr)
        print("  Once kimlik dogrulamasi gerekebilir. Su komutu SEN calistir:\n"
              "    gh auth login\n"
              "  Sonra bu komutu tekrar dene.", file=sys.stderr)
        return 1

    user, _, name = args.repo.partition("/")
    print(f"\nGonderildi: {res['url']}")
    print(f"Site adresi: https://{user}.github.io/{name}/")
    print("  (GitHub Pages ilk kurulumda birkac dakika surebilir)")
    return 0


def cmd_clear_cache(args: argparse.Namespace) -> int:
    if args.invalid_only:
        n = cache.purge_invalid("yahoo")
        print(f"{n} bos/bozuk onbellek kaydi silindi. "
              f"Bir sonraki taramada bu hisseler yeniden denenecek.")
        return 0
    n = cache.clear(args.namespace)
    print(f"{n} onbellek dosyasi silindi.")
    return 0


def _kisa_vade_uret(bundles: dict) -> None:
    """Gunluk taramanin bir parcasi olarak kisa vade sinyallerini yazar.

    Kritik yol DEGIL: patlarsa tarama devam eder. Uzun vadeli siralama bu
    dosyaya hicbir sekilde bagli degil.
    """
    try:
        from src import kalibrasyon as kb
        from src import kisa_vade as kv

        tablo = kv.tara(bundles)
        kalib = kb.yukle()
        satirlar = []
        for r in tablo.to_dict("records"):
            g = kb.guven(kalib, r["kurulum"], r["ufuk"],
                         {"oynaklik": r.get("oynaklik"),
                          "likidite": r.get("likidite"),
                          "trend_konumu": r.get("trend_konumu")})
            satirlar.append({**r, "guven": g})
        satirlar.sort(key=lambda x: (
            0 if (x["guven"].get("ayirt_edilebilir")) else 1,
            -(x["guven"].get("edge") if x["guven"].get("edge") is not None else -1),
            -x["guc"]))

        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "kisa_vade.json").write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kalibrasyon_tarihi": (kalib or {}).get("generated_at"),
            "sinyal_sayisi": len(satirlar),
            "sinyaller": satirlar,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

        olculen = sum(1 for x in satirlar if x["guven"]["durum"] == "olculdu")
        ayirt = sum(1 for x in satirlar if x["guven"].get("ayirt_edilebilir"))
        ek = (f", {olculen} tanesi olculmus, {ayirt} tanesi tabandan ayirt "
              f"edilebilir" if kalib else " (kalibrasyon yok, guven uretilemedi)")
        print(f"      kisa vade: {len(satirlar)} kurulum{ek}")
    except Exception as exc:                      # kritik yol degil
        print(f"      UYARI: kisa vade sinyalleri uretilemedi ({exc})")


def _onbellekten_bundles(period: str = "2y", max_gun: int = 30,
                         limit: int | None = None) -> dict:
    """Ag istegi YAPMADAN, onbellekteki gunluk barlari toplar.

    Kisa vade taramasi gunluk taramadan SONRA calisir; onbellek zaten
    tazedir. Ayrica ag istegi eklemek, gunluk taramanin hiz siniri
    butcesini yer -- sistemin en kirilgan kaynagi o.
    """
    from src import watchlist as _wl
    from src.providers import cache as _cache
    out: dict = {}
    try:
        evren, _ = universe.build(["smallcap", "midcap", "wsb"])
    except Exception:
        evren = []
    try:
        izleme = [str(p.get("ticker")) for p in _wl.load() if p.get("ticker")]
    except Exception:
        izleme = []
    semboller = list(dict.fromkeys(list(evren) + izleme))
    if limit:
        semboller = semboller[:limit]
    azami = max_gun * 24 * 3600
    for tk in semboller:
        hit = _cache.peek("yahoo", f"{tk}:{period}")
        if not hit:
            continue
        b, yas = hit
        if yas > azami or not b or b.get("history") is None:
            continue
        out[tk] = b
    return out


def _kisa_bundles(args: argparse.Namespace) -> dict:
    """Frekansa gore bar kaynagi.

    Gunluk: tum evrenin onbellegi (2700+ hisse).
    Gun ici: YALNIZCA havuz. Iki sebep -- gun ici veri havuz disinda zaten
    cekilmiyor, ve cekilseydi hiz sinirina carpardi.
    """
    from src import havuz as hv
    from src import intraday as idy

    if args.frekans == "1d":
        b = _onbellekten_bundles(args.period, args.cache_days, args.limit)
        if not b:
            print("HATA: onbellekte gunluk bar yok. Once 'python run.py' calistir.",
                  file=sys.stderr)
        return b

    semboller = hv.semboller()
    if not semboller:
        print("HATA: havuz yok. Once: python run.py havuz", file=sys.stderr)
        return {}
    if args.limit:
        semboller = semboller[:args.limit]
    out = {}
    for sym in semboller:
        h = idy.oku(sym, args.frekans)           # ONBELLEK; ag istegi YOK
        if h is not None and len(h) >= kisa_min_bar():
            out[sym] = {"history": h}
    if not out:
        print(f"HATA: {args.frekans} barlari yok. Once: "
              f"python run.py intraday cek --interval {args.frekans}",
              file=sys.stderr)
    return out


def kisa_min_bar() -> int:
    from src import kisa_vade as kv
    return kv.MIN_BAR


def _kisa_bench(args: argparse.Namespace):
    """Karsilastirma endeksi. Yoksa acikca uyarir.

    Endekssiz olcum, yukselen piyasada her kurulumu iyi gosterir; sessizce
    dusulmesi gereken bir seye degil, gorulmesi gereken bir eksige benziyor.
    """
    frekans = getattr(args, "frekans", "1d")
    try:
        if frekans != "1d":
            # GUN ICI OLCUMDE ENDEKS DE GUN ICI OLMALI. Gunluk endeksi
            # saatlik barlara yaymak, gun icinde endeks getirisini SIFIR
            # yapar ve "endeksten iyi" olcusu sessizce "yukari gitti"ye
            # doner. Ayrica gunluk onbellek 2 yil, saatlik veri 3 yil --
            # aradaki fark etiketsiz satir olarak kaybolur (%68 etiketli).
            from src import intraday as idy
            h = idy.cek(args.benchmark, frekans)
            if h is not None and "Close" in h:
                return h["Close"]
        else:
            bd = yahoo.fetch_benchmark(args.benchmark, args.period,
                                       use_cache=True)
            if bd is not None and "Close" in bd:
                return bd["Close"]
    except Exception:
        pass
    print("  UYARI: endeks gecmisi yok, olcum ENDEKSSIZ yapilacak "
          "(yukselen piyasada her kurulum iyi gorunur)", file=sys.stderr)
    return None


def cmd_kisa(args: argparse.Namespace) -> int:
    """Kisa vadeli kurulum taramasi ve kalibrasyonu.

    Uzun vadeli siralamadan AYRI durur ve ayri dosyaya yazar. Ikisini ayni
    tabloya koymak, iki farkli soruyu tek cevaba sikistirmak olurdu: uzun
    vadeli skor bir SIRALAMA, buradaki cikti bir OLAY tespiti.
    """
    from src import kalibrasyon as kb
    from src import kisa_vade as kv

    eylem = getattr(args, "kisa_action", "tara")

    if eylem == "kalibre":
        print("=" * 74)
        print("KISA VADE KALIBRASYONU")
        print("=" * 74)
        print("Onbellekteki gunluk barlar taraniyor (ag istegi yok)...",
              flush=True)
        bundles = _kisa_bundles(args)
        if not bundles:
            return 1
        print(f"  {len(bundles)} hisse · frekans {args.frekans} · "
              f"ufuklar {kv.ufuklar(args.frekans)} bar", flush=True)

        bench = _kisa_bench(args)

        def ilerleme(i, islenen):
            print(f"      {i} sembol tarandi ({islenen} kullanildi)", flush=True)

        payload = kb.kur(bundles, bench, ufuklar=kv.ufuklar(args.frekans),
                         min_bar=kv.MIN_BAR, ilerleme=ilerleme,
                         frekans=args.frekans)
        yol = kb.kaydet(payload)
        _kalibrasyon_tablosu(payload)
        print()
        print(f"Kaydedildi: {yol}")
        return 0

    if eylem == "panel":
        print("=" * 74)
        print("KISA VADE META-ETIKET PANELI")
        print("=" * 74)
        bundles = _kisa_bundles(args)
        if not bundles:
            return 1
        print(f"  {len(bundles)} hisse · frekans {args.frekans}", flush=True)
        bench = _kisa_bench(args)

        def ilerleme(i, islenen):
            print(f"      {i} sembol tarandi ({islenen} kullanildi)", flush=True)

        ozet = kb.panel(bundles, bench, ufuklar=kv.ufuklar(args.frekans),
                        min_bar=kv.MIN_BAR, ilerleme=ilerleme,
                        frekans=args.frekans)
        if not ozet.get("ok"):
            print(f"HATA: {ozet.get('reason')}", file=sys.stderr)
            return 1
        print()
        print(f"  satir      : {ozet['satir']:,}")
        print(f"  hisse      : {ozet['hisse']}")
        print(f"  kurulum    : {ozet['kurulum']}")
        print(f"  tarih      : {ozet['tarih_araligi'][0]} -> "
              f"{ozet['tarih_araligi'][1]}")
        print(f"  ozellik    : {len(ozet['ozellikler'])} sutun")
        print(f"  etiket     : {', '.join(ozet['etiketler'])}")
        print(f"  etiketli   : "
              + ", ".join(f"{k} %{100*v:.0f}"
                          for k, v in ozet["etiketli_oran"].items()))
        print()
        print("  Bu tablo bir MODEL EGITIM KUMESIDIR, karar tablosu degil.")
        print("  Ozellikler sinyal gunune kadar, etiketler sinyal gununden")
        print("  SONRASINI olcer; ikisi hicbir yerde karismaz.")
        print()
        print(f"  Kaydedildi: {ozet['yol']}")
        return 0

    # --- bugunku tarama
    kalib = kb.yukle(frekans=args.frekans)
    bundles = _kisa_bundles(args)
    if not bundles:
        return 1

    tablo = kv.tara(bundles, frekans=args.frekans)
    if tablo.empty:
        print("Bugun hicbir kurulum olusmadi.")
        return 0

    satirlar = []
    for r in tablo.to_dict("records"):
        g = kb.guven(kalib, r["kurulum"], r["ufuk"],
                     {"oynaklik": r.get("oynaklik"),
                      "likidite": r.get("likidite"),
                      "trend_konumu": r.get("trend_konumu")})
        satirlar.append({**r, "guven": g})

    # Siralama: once olculmus VE tabandan ayirt edilebilir olanlar.
    # Guc'e gore siralamak yanlis olurdu -- guc kurulumun ne kadar temiz
    # olustugunu soyler, ise yarayip yaramadigini degil.
    def anahtar(x):
        g = x["guven"]
        return (0 if g.get("ayirt_edilebilir") else 1,
                -(g.get("edge") if g.get("edge") is not None else -1),
                -x["guc"])
    satirlar.sort(key=anahtar)

    _kisa_tablo(satirlar, kalib)

    # CIKTI DOSYASI FREKANSA GORE AYRI. Panonun okudugu dosya gunluk olan;
    # saatlik tarama onu EZMEMELI. Ilk surumde ayni dosyaya yaziliyordu ve
    # saatlik bir kosu, panoyu saatlik sinyallerle doldurup gunluk olanlari
    # siliyordu -- ayni kategoriden hatayi ufuk arsivlerinde de yapmistik.
    OUT.mkdir(parents=True, exist_ok=True)
    yol = (OUT / "kisa_vade.json" if args.frekans == "1d"
           else OUT / f"kisa_vade_{args.frekans}.json")
    yol.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frekans": args.frekans,
        "kalibrasyon_tarihi": (kalib or {}).get("generated_at"),
        "sinyal_sayisi": len(satirlar),
        "sinyaller": satirlar,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(f"Kaydedildi: {yol}")
    return 0


def _kisa_tablo(satirlar: list, kalib: "dict | None") -> None:
    print("=" * 98)
    print("KISA VADELI KURULUMLAR")
    print("=" * 98)
    if not kalib:
        print("  KALIBRASYON YOK — guven degeri uretilemiyor.")
        print("  Once: python run.py kisa kalibre")
    print()
    baslik = (f"  {'SEMBOL':<8}{'KURULUM':<24}{'YON':<7}{'UFUK':>5}{'GUC':>6}"
              f"{'GUVEN':>7}{'TABAN':>7}{'ARALIK':>14}{'ETKIN N':>9}  ORTAM")
    print(baslik)
    print("  " + "-" * 94)
    for r in satirlar[:40]:
        g = r["guven"]
        if g["durum"] == "olculdu":
            guven = f"%{100 * g['p']:.0f}"
            taban = f"%{100 * g['taban']:.0f}"
            aralik = f"%{100 * g['alt']:.0f}-%{100 * g['ust']:.0f}"
            netkin = f"{g['n_etkin']:.0f}"
            isaret = "*" if g.get("ayirt_edilebilir") else " "
        else:
            guven, taban, aralik, netkin = "-", "-", g["durum"], "-"
            isaret = " "
        print(f"{isaret} {r['ticker']:<8}{r['ad_tr'][:24]:<24}{r['yon']:<7}"
              f"{r['ufuk']:>5}{r['guc']:>6.2f}{guven:>7}{taban:>7}{aralik:>14}"
              f"{netkin:>9}  {r['oynaklik']}/{r['likidite']}")
    if len(satirlar) > 40:
        print(f"  ... toplam {len(satirlar)} sinyal (tamami kisa_vade.json icinde)")
    print()
    print("  * = alt guven siniri taban oranin USTUNDE, yani fark gurultuden")
    print("      ayirt edilebiliyor. Yildizsiz satirlar icin bunu soyleyemeyiz.")
    print("  GUVEN : gecmiste bu kurulum olustugunda endeksi gecme orani.")
    print("  TABAN : ayni donemde rastgele bir gunun endeksi gecme orani.")
    print("  Onemli olan ikisi arasindaki FARK; tek basina GUVEN degil.")
    print("  Bu bir yatirim tavsiyesi degildir.")


def _kalibrasyon_tablosu(payload: dict) -> None:
    kovalar = [k for k in payload.get("kovalar", []) if k["kosul"] == "*"]
    print()
    print("=" * 94)
    print("KURULUM BASARIMI — genel kova")
    print("=" * 94)
    print(f"  hisse {payload['hisse']} · kazanc tanimi: {payload['kazanc_tanimi']}")
    tabanlar = ", ".join(f"{u}g %{100 * v:.0f}"
                         for u, v in payload["taban"].items())
    print(f"  taban oranlari: {tabanlar}")
    print()
    print(f"  {'KURULUM':<24}{'UFUK':>5}{'N':>8}{'ETKIN':>7}{'P':>7}"
          f"{'TABAN':>7}{'EDGE':>8}{'ARALIK':>14}  DURUM")
    print("  " + "-" * 90)
    for k in sorted(kovalar, key=lambda x: (x["kurulum"], x["ufuk"])):
        aralik = f"{k['alt']:.2f}-{k['ust']:.2f}"
        print(f"  {k['kurulum'][:24]:<24}{k['ufuk']:>5}{k['n']:>8}"
              f"{k['n_etkin']:>7.0f}{k['p']:>7.2f}{k['taban']:>7.2f}"
              f"{k['edge']:>+8.3f}{aralik:>14}  {k['durum']}")
    print()
    for n in payload.get("notlar_tr", []):
        print("  * " + n)


def cmd_havuz(args: argparse.Namespace) -> int:
    """Benzer sirketlerden test havuzu kurar.

    Kisa vade olcumunun en buyuk tehlikesi, olculen farkin kurulumdan degil
    SIRKET FARKINDAN gelmesi. Havuz bunu kesiyor. Ayrica pratik bir isi daha
    var: 2755 hisse icin saatlik veri cekmek hiz sinirina carpar, 150 hisse
    icin carpmaz -- yani havuz olmadan saatlik olcum zaten mumkun degil.
    """
    from src import havuz as hv

    bundles = _onbellekten_bundles(args.period, args.cache_days, args.limit)
    if not bundles:
        print("HATA: onbellekte gunluk bar yok. Once 'python run.py' calistir.",
              file=sys.stderr)
        return 1

    d = hv.kur(bundles, boyut=args.boyut, en_fazla_sektor=args.sektor_sayisi,
               min_fiyat=args.min_fiyat, min_dv=args.min_hacim)
    if not d.get("ok"):
        print(f"HATA: {d.get('reason')}", file=sys.stderr)
        return 1

    print("=" * 88)
    print("TEST HAVUZLARI")
    print("=" * 88)
    print(f"  evren {d['evren']} -> olculebilir {d['uygun']} -> "
          f"{d['havuz_sayisi']} havuz x {args.boyut} hisse")
    print(f"  agirliklar: "
          + ", ".join(f"{k}={v}" for k, v in d["agirlik"].items()))
    print()
    print(f"  {'SEKTOR':<24}{'UYE':>5}{'ADAY':>6}   "
          f"{'MCAP':>8}{'HACIM':>10}{'ATR%':>7}   DARALMA (evrene gore)")
    print("  " + "-" * 84)
    for h in d["havuzlar"]:
        md = h["dagilim"]
        dar = " ".join(f"{k.replace('log_', '')[:4]} {v:.2f}x"
                       for k, v in h["daralma"].items())
        print(f"  {h['sektor'][:24]:<24}{h['boyut']:>5}{h['aday_havuzu']:>6}   "
              f"{md['log_mcap']['havuz']['medyan']:>8.2f}"
              f"{md['log_dolar_hacim']['havuz']['medyan']:>10.2f}"
              f"{md['atr_pct']['havuz']['medyan'] * 100:>7.2f}   {dar}")
    print()
    print("  MCAP ve HACIM 10 tabanina gore logaritma (8.97 = ~930M$).")
    print("  DARALMA: havuzun ceyrekler arasi genisligi / evrenin. Kucuk = dar.")
    print()
    for h in d["havuzlar"]:
        print(f"  [{h['sektor']}]")
        for i in range(0, len(h["uyeler"]), 12):
            print("    " + ", ".join(h["uyeler"][i:i + 12]))
    print()
    for n in d.get("notlar_tr", []):
        print("  * " + n)

    yol = hv.kaydet(d)
    print()
    print(f"Kaydedildi: {yol}")
    print(f"Toplam {len(hv.semboller(d))} sembol.")
    return 0


def cmd_intraday(args: argparse.Namespace) -> int:
    """Gun ici veri: saglayicinin sinirlarini olc, havuzu cek.

    Zaman dilimi karari BURADAN cikar: hangi aralikta kac FARKLI GUN veri
    geliyor. Bar sayisi degil gun sayisi belirleyici -- ayni gunun barlari
    tek bir piyasa gunudur.
    """
    from src import havuz as hv
    from src import intraday as idy

    eylem = getattr(args, "intraday_action", "kapsam")

    if eylem == "kapsam":
        print("=" * 78)
        print("GUN ICI VERI KAPSAMI")
        print("=" * 78)
        print(f"  Olcum sembolu: {args.symbol}  (amac veri toplamak degil, "
              f"sinirlari ogrenmek)")
        print()
        k = idy.kapsam_olc(args.symbol, use_cache=not args.no_cache)
        print(f"  {'ARALIK':<8}{'ISTENEN':<9}{'BAR':>8}{'GUN':>6}"
              f"{'BAR/GUN':>9}{'ILK':>13}{'SON':>13}  DURUM")
        print("  " + "-" * 74)
        for i, o in k["araliklar"].items():
            if o.get("hata"):
                print(f"  {i:<8}{'':<9}{'':>8}{'':>6}{'':>9}{'':>13}{'':>13}"
                      f"  {o['hata']}")
                continue
            durum = ("olculebilir" if o.get("olculebilir")
                     else f"YETERSIZ (<{idy.MIN_GUN} gun)")
            print(f"  {i:<8}{o.get('istenen',''):<9}{o['bar']:>8}{o['gun']:>6}"
                  f"{o.get('bar_gun',0):>9.1f}{o.get('ilk',''):>13}"
                  f"{o.get('son',''):>13}  {durum}")
        print()
        oneri = idy.onerilen_aralik(k)
        if oneri.get("ok"):
            print(f"  ONERILEN BIRINCIL ARALIK: {oneri['birincil']}")
            print(f"  Uygun adaylar: {', '.join(oneri['adaylar'])}")
        else:
            print(f"  ONERI YOK: {oneri.get('reason')}")
        print()
        print(f"  Olcut FARKLI GUN sayisi (en az {idy.MIN_GUN}). Bar sayisi")
        print("  yaniltir: 1 dakikalik veride binlerce bar olur ama hepsi")
        print("  birkac gunden gelir; bagimsiz gozlem sayisi gun sayisidir.")
        print(f"\n  Kaydedildi: {idy.KAPSAM}")
        return 0

    # --- havuzu cek
    hd = hv.yukle()
    if not hd:
        print("HATA: havuz yok. Once: python run.py havuz", file=sys.stderr)
        return 1
    semboller = hv.semboller(hd, havuz_id=args.havuz)
    if not semboller:
        print("HATA: havuzda sembol yok.", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"GUN ICI CEKIM — {args.interval}")
    print("=" * 78)
    print(f"  {len(semboller)} sembol (havuz), istekler arasi "
          f"{args.bekleme}s bekleme")

    def ilerleme(i, ok):
        print(f"      {i}/{len(semboller)} denendi, {ok} basarili", flush=True)

    sonuc = idy.havuz_cek(semboller, args.interval, args.period,
                          bekleme=args.bekleme, ilerleme=ilerleme)
    print()
    print(f"  durum   : {sonuc['durum']}")
    print(f"  cekilen : {sonuc['cekilen']}")
    print(f"  hatali  : {sonuc['hatali']}")
    if sonuc["kalan"]:
        print(f"  kalan   : {sonuc['kalan']}  (hiz siniri, sonra devam et)")
    if sonuc["bundles"]:
        ilk = next(iter(sonuc["bundles"].values()))
        o = idy.olcum(ilk)
        print(f"  ornek   : {o['bar']} bar, {o['gun']} gun, "
              f"{o['ilk']} -> {o['son']}")
    return 0 if sonuc["durum"] == "tamam" else 2


def main() -> int:
    p = argparse.ArgumentParser(
        description="Cok faktorlu hisse yatirim skorlama sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="cmd")

    # --- varsayilan komut: scan
    p.add_argument("--config", default=str(ROOT / "config" / "weights.yaml"))
    p.add_argument("--universe", default="emerging,wsb",
                   help="virgullu. Piyasa degeri on ayarlari: micro, smallcap, midcap, "
                        "emerging (200M-10Mr, varsayilan), largecap, us. "
                        "Endeksler: sp500, nasdaq100. Ayrica: wsb, file")
    p.add_argument("--min-mcap", type=float, default=None, dest="min_mcap",
                   help="piyasa degeri alt siniri (USD); on ayari ezer")
    p.add_argument("--max-mcap", type=float, default=None, dest="max_mcap",
                   help="piyasa degeri ust siniri (USD); on ayari ezer")
    p.add_argument("--symbols-file", default=None, help="--universe file icin sembol listesi")
    p.add_argument("--limit", type=int, default=None, help="evreni ilk N sembolle sinirla")
    # Hiz siniri yuzunden tek seferde ~800 hisse cekilebiliyor. Donusumlu tarama
    # sayesinde birkac gunde tum evren dolasilir (bulgu K3).
    p.add_argument("--no-backfill", action="store_true",
                   help="onbellekteki eski hisseleri skorlamaya katma")
    # 5 gundu; 12'ye cikarildi (02.09.2026). Sebep: donusumlu tarama turda 800
    # sembol deniyor, evren ~2790 -> tam tur EN IYI ihtimalle 3.5 gun. Hiz siniri
    # yuzunden gercek basari %50-78 oldugundan tam tur pratikte 7-10 gun suruyor.
    # 5 gunluk pencere bu turu YAPISAL OLARAK kapsayamiyordu: onbellekteki 3350
    # hissenin ancak ~450'si pencereye giriyor, skorlanan sayi cokuyor, cikti
    # guvenlik kapisi panoyu reddediyor, gun isaretlenmiyor, onbellek daha da
    # yasleniyor -- kendini besleyen bir sarmal (bir haftaligina buna girildi).
    # Pencere artik gercek tur suresiyle ayni buyuklukte. Bedeli veri bayatligi,
    # o yuzden diag["data_age"] ile sayilip panoda gosteriliyor.
    p.add_argument("--backfill-days", type=int, default=12,
                   help="onbellekten geri doldurmada kabul edilen azami veri yasi (gun)")
    p.add_argument("--batch", type=int, default=800,
                   help="bir turda taranacak azami sembol (0 = sinirsiz). "
                        "Evren buyukse gunlere bolunerek dolasilir")
    p.add_argument("--wsb-top", type=int, default=60, help="WSB'den kac sembol alinsin")
    p.add_argument("--top", type=int, default=40, help="panoda gosterilecek hisse sayisi")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--period", default="2y", help="fiyat gecmisi uzunlugu")
    # Yahoo yuksek eszamanlilikta 401 'Invalid Crumb' donuyor; 4 guvenli nokta.
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--disable", default="", help="virgullu faktor id listesi")
    p.add_argument("--weight", action="append", help="faktor_id=deger (tekrarlanabilir)")
    p.add_argument("--use-learned", action="store_true", help="ogrenilmis agirliklari kullan")
    p.add_argument("--no-filters", action="store_true", help="on elemeyi tamamen kapat")
    p.add_argument("--no-fallback", action="store_true",
                   help="Yahoo basarisiz olan sembollerde yedek fiyat kaynagini "
                        "(Nasdaq) deneme")
    p.add_argument("--no-cache", action="store_true")

    lp = sub.add_parser("learn", help="agirliklari gecmis verilerden ogren")
    lp.add_argument("--config", default=str(ROOT / "config" / "weights.yaml"))
    lp.add_argument("--horizon", type=int, default=21, help="ileri getiri ufku (islem gunu)")
    lp.add_argument("--method", default="ic", choices=["ic", "ridge"])
    lp.add_argument("--pretrain", action="store_true",
                    help="canli feature store yerine gecmise donuk panelden olc "
                         "(bugun sonuc almak icin; yanlilik tasir)")
    lp.add_argument("--no-cache", action="store_true")

    wp = sub.add_parser("watch", help="izleme listesi / pozisyon takibi")
    wp.add_argument("action", choices=["add", "remove", "list", "update", "import"],
                    help="add: ekle · remove: sil · list: listele · update: gunluk analiz "
                         "· import: panodan indirilen JSON'u ice aktar")
    wp.add_argument("ticker", nargs="?", default=None,
                    help="sembol (virgullu birden fazla olabilir); import icin dosya yolu")
    wp.add_argument("--price", type=float, default=None,
                    help="alis fiyati; verilmezse sadece izlenir")
    wp.add_argument("--qty", type=float, default=None, help="adet")
    wp.add_argument("--note", default="", help="serbest not")
    wp.add_argument("--use-cache", action="store_true",
                    help="onbellek kullan (gunluk takipte onerilmez)")

    mp = sub.add_parser("ml", help="ogrenme sistemi: durum / egitim / terfi")
    mp.add_argument("ml_action", choices=["status", "train", "evaluate"],
                    help="status: hazirlik durumu · train: egit+degerlendir · "
                         "evaluate: yalnizca degerlendir")
    mp.add_argument("--models", default=None,
                    help="virgullu model listesi (ridge, mlp, seq). "
                         "Bos birakilirsa hepsi denenir; ridge her zaman dahil")
    mp.add_argument("--horizon", type=int, default=21, help="ileri getiri ufku (islem gunu)")
    mp.add_argument("--splits", type=int, default=5, help="ileri yuruyus katman sayisi")
    mp.add_argument("--embargo", type=int, default=5,
                    help="arindirmaya ek tampon gun (seri korelasyon icin)")
    mp.add_argument("--window", type=int, default=10, help="dizi modeli pencere uzunlugu")
    mp.add_argument("--promote", action="store_true",
                    help="esikleri gecen en iyi modeli sampiyon yap")
    mp.add_argument("--force", action="store_true",
                    help="yetersiz veriye ragmen egit (sonuclar guvenilmez)")
    mp.add_argument("--pretrain", action="store_true",
                    help="canli feature store yerine gecmise donuk panelden egit "
                         "(once 'run.py backfill'). Sampiyon URETMEZ")
    mp.add_argument("--min-rows", type=int, default=30,
                    help="bir gunun panele girmesi icin gereken en az hisse sayisi")
    mp.add_argument("--horizons", default=None,
                    help="virgullu ufuk listesi (orn. 5,21,63). Verilirse her ufuk "
                         "ayri egitilir ve karsilastirma tablosu basilir")
    mp.add_argument("--no-ensemble", action="store_true",
                    help="modellerin yuzdelik siralarini harmanlayan toplulugu kurma")
    mp.add_argument("--no-cache", action="store_true")

    # Ad "history": tarama tarafindaki --no-backfill/--backfill-days bayraklari
    # ONBELLEK doldurmayi anlatiyor, bu komut ise GECMIS uretiyor. Ayni kelime
    # iki farkli sey icin kullanilmasin. Eski ad takma ad olarak duruyor.
    bf = sub.add_parser("history", aliases=["backfill"],
                        help="onbellekteki fiyat gecmisinden gecmise donuk anlik "
                             "goruntu uret (ogrenmenin baslangic sermayesi)")
    bf.add_argument("--step", type=int, default=3,
                    help="kac islem gununde bir anlik goruntu (1 = her gun)")
    bf.add_argument("--snapshots", type=int, default=90,
                    help="en fazla kac anlik goruntu uretilsin")
    bf.add_argument("--horizon", type=int, default=21,
                    help="etiket ufku; son bu kadar gun izgaraya alinmaz")
    bf.add_argument("--workers", type=int, default=4)
    bf.add_argument("--limit", type=int, default=None,
                    help="yalnizca ilk N sembol (deneme icin)")
    bf.add_argument("--restart", action="store_true",
                    help="kaldigi yerden devam etme, bastan uret")
    bf.add_argument("--merge-only", action="store_true",
                    help="uretim yapma; birikmis yiginlari panele cevir "
                         "(uretim devam ederken elde olani kullanmak icin)")

    kp = sub.add_parser("paper", aliases=["defter"],
                        help="kagit uzerinde portfoy defteri: ilk N'in gercek "
                             "getirisi (sistemin karnesi)")
    kp.add_argument("paper_action", nargs="?", default="show",
                    choices=["show", "build", "mark"],
                    help="show: karne · build: defteri doldur · mark: degerle")
    kp.add_argument("--config", default=str(ROOT / "config" / "weights.yaml"))
    kp.add_argument("--top", type=int, default=20,
                    help="her gun kac hisse deftere yazilsin")
    kp.add_argument("--horizon", type=int, default=21,
                    help="tutma suresi (islem gunu)")
    kp.add_argument("--panel", action="store_true",
                    help="geriye donuk panelden de uret (11 ay, ama yanlilik tasir)")

    bkp = sub.add_parser("backup", aliases=["yedek"],
                         help="yeniden uretilemeyen veriyi sifreli arsivle")
    bkp.add_argument("backup_action", nargs="?", default="create",
                     choices=["create", "list", "restore"],
                     help="create: yedek al · list: yedekleri listele · "
                          "restore: ayri bir dizine ac")
    bkp.add_argument("--dir", default=None, help="yedek dizini")
    bkp.add_argument("--file", default=None, help="restore icin yedek dosyasi")
    bkp.add_argument("--target", default=None,
                     help="restore hedef dizini (varsayilan: yedek_acilan/)")
    bkp.add_argument("--overwrite", action="store_true",
                     help="restore sirasinda mevcut dosyalarin uzerine yaz")
    bkp.add_argument("--label", default="", help="dosya adina eklenecek etiket")
    bkp.add_argument("--keep", type=int, default=8,
                     help="tutulacak en yeni yedek sayisi (0 = hepsi)")

    pp = sub.add_parser("publish", help="panolari sifreleyip tek dosya yap")
    pp.add_argument("--set-password", action="store_true",
                    help="parolayi iki kez sorarak dogrula")
    pp.add_argument("--force", action="store_true",
                    help="zayif parolaya ragmen devam et")

    sp = sub.add_parser("serve", help="panolari yerel agda yayinla")
    sp.add_argument("--port", type=int, default=8800)
    sp.add_argument("--lan", action="store_true",
                    help="ayni Wi-Fi agindaki cihazlara ac (yalnizca sifreli surumle onerilir)")

    dpl = sub.add_parser("deploy", help="sifreli panolari GitHub Pages'e gonder")
    dpl.add_argument("--repo", default=None, help="kullanici/depo (orn. yigaykut/hisse-pano)")
    dpl.add_argument("--branch", default="main")
    dpl.add_argument("--build-only", action="store_true",
                     help="yalnizca yayin dizinini hazirla, gonderme")

    kv_p = sub.add_parser("kisa", aliases=["short"],
                          help="kisa vadeli kurulum taramasi ve kalibrasyonu "
                               "(uzun vadeli siralamadan AYRI)")
    kv_p.add_argument("kisa_action", nargs="?", default="tara",
                      choices=["tara", "kalibre", "panel"],
                      help="tara: bugunku kurulumlar - kalibre: gecmisten "
                           "guven degerlerini olc - panel: ileride egitilecek "
                           "model icin satir satir ozellik+sonuc tablosu")
    kv_p.add_argument("--frekans", default="1d",
                      choices=["1d", "1h", "30m", "15m"],
                      help="bar frekansi. 1d tum evreni, gun ici olanlar "
                           "YALNIZCA havuzu kullanir. Ufuklar BAR cinsindendir "
                           "ve frekansa gore degisir.")
    kv_p.add_argument("--period", default="2y", help="onbellek gecmis araligi")
    kv_p.add_argument("--cache-days", type=int, default=30,
                      help="onbellekte bu kadar gunden eski kayit kullanilmaz")
    kv_p.add_argument("--limit", type=int, default=None,
                      help="yalnizca ilk N sembol (deneme icin)")
    kv_p.add_argument("--benchmark", default="SPY",
                      help="kazanc 'endeksten iyi' diye olculur")

    hp = sub.add_parser("havuz", aliases=["pool"],
                        help="benzer sirketlerden test havuzu kur "
                             "(kisa vade olcumu icin)")
    hp.add_argument("--boyut", type=int, default=25,
                    help="havuz basina hisse sayisi")
    hp.add_argument("--sektor-sayisi", type=int, default=6,
                    help="en fazla kac sektorde havuz kurulsun")
    hp.add_argument("--min-fiyat", type=float, default=5.0)
    hp.add_argument("--min-hacim", type=float, default=1e6,
                    help="gunluk dolar hacim alt siniri")
    hp.add_argument("--period", default="2y")
    hp.add_argument("--cache-days", type=int, default=30)
    hp.add_argument("--limit", type=int, default=None)

    ip = sub.add_parser("intraday", aliases=["gunici"],
                        help="gun ici veri: saglayici sinirlarini olc, "
                             "havuzu cek")
    ip.add_argument("intraday_action", nargs="?", default="kapsam",
                    choices=["kapsam", "cek"],
                    help="kapsam: hangi aralikta ne kadar gecmis var - "
                         "cek: havuzun gun ici barlarini indir")
    ip.add_argument("--symbol", default="SPY", help="kapsam olcumu sembolu")
    ip.add_argument("--interval", default="1h")
    ip.add_argument("--period", default=None)
    ip.add_argument("--havuz", default=None, help="yalnizca bu havuz")
    ip.add_argument("--bekleme", type=float, default=1.2,
                    help="istekler arasi saniye (hiz siniri icin)")
    ip.add_argument("--no-cache", action="store_true")

    cp = sub.add_parser("clear-cache", help="veri onbellegini temizle")
    cp.add_argument("--namespace", default=None)
    cp.add_argument("--invalid-only", action="store_true",
                    help="sadece bos/bozuk kayitlari sil (saglam veriyi koru)")

    dp = sub.add_parser("daily",
                        help="gunluk tam dongu: tarama + izleme listesi + ogrenme")
    dp.add_argument("--no-train", action="store_true",
                    help="gunluk dongude yeniden egitimi atla")
    dp.add_argument("--retrain-every", type=int, default=5,
                    help="kac taramada bir yeniden egitilsin")
    dp.add_argument("--horizon", type=int, default=21,
                    help="ogrenme icin ileri getiri ufku (islem gunu)")
    for a in p._actions:                       # tarama secenekleri aynen gecerli
        if a.dest in ("help", "cmd"):
            continue
        dp._add_action(a)

    args = p.parse_args()
    if args.cmd == "ml":
        return cmd_ml(args)
    if args.cmd in ("paper", "defter"):
        return cmd_paper(args)
    if args.cmd in ("backup", "yedek"):
        return cmd_backup(args)
    if args.cmd in ("history", "backfill"):
        return cmd_backfill(args)
    if args.cmd == "publish":
        return cmd_publish(args)
    if args.cmd == "deploy":
        return cmd_deploy(args)
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "daily":
        return cmd_daily(args)
    if args.cmd == "watch":
        return cmd_watch(args)
    if args.cmd == "learn":
        return cmd_learn(args)
    if args.cmd in ("intraday", "gunici"):
        return cmd_intraday(args)
    if args.cmd in ("havuz", "pool"):
        return cmd_havuz(args)
    if args.cmd in ("kisa", "short"):
        return cmd_kisa(args)
    if args.cmd == "clear-cache":
        return cmd_clear_cache(args)
    return cmd_scan(args)


if __name__ == "__main__":
    raise SystemExit(main())
