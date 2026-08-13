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

from src import factors, ml, report, scoring, universe          # noqa: E402
from src.providers import cache, reddit_wsb, yahoo              # noqa: E402

OUT = ROOT / "output"


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
def cmd_scan(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    t0 = time.time()

    # --- 1) Evren
    sources = [s.strip() for s in args.universe.split(",") if s.strip()]
    print(f"[1/6] Evren olusturuluyor: {', '.join(sources)}")
    tickers, breakdown = universe.build(sources, wsb_top=args.wsb_top,
                                        symbols_file=args.symbols_file, limit=args.limit,
                                        min_mcap=args.min_mcap, max_mcap=args.max_mcap)

    # --- Izleme listesi HER ZAMAN evrene dahil ---------------------------------
    # Liste gunluk yenilenir ve siralama degisir; ama kullanicinin sectigi
    # hisseler evrenden dusse bile taranmaya ve gosterilmeye devam eder.
    from src import scanlog
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

    # --- Kote disi tespiti icin gunluk evren kaydi (bulgu Y3) -----------------
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
    tickers = scanlog.order_by_staleness(tickers, pinned)
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
            elif bundle.get("history") is None:
                # Cekim istisna atmadi ama fiyat serisi gelmedi. Bunu "yetersiz
                # gecmis" saymak yaniltici olur — cogunlukla hiz siniridir.
                no_data.append(tk)
            else:
                bundles[tk] = bundle
            if done % 25 == 0 or done == len(tickers):
                print(f"      {done}/{len(tickers)}", end="\r", flush=True)

    # Basarili cekimleri kaydet -> bir sonraki tur bunlari sona atar
    if bundles:
        scanlog.record(list(bundles.keys()))

    # Cekim basari orani, GERI DOLDURMADAN ONCE hesaplanir; aksi halde
    # onbellekten gelenler oranı sisirir ve %100'u asar.
    fetched_live = len(bundles)
    ok_rate = 100.0 * fetched_live / max(1, len(tickers))

    # --- Onbellekten geri doldurma -------------------------------------------
    # Donusumlu tarama bu turda evrenin bir dilimini cekti. Daha once cekilmis
    # hisseleri de skorlamaya katiyoruz: ag maliyeti YOK (yalnizca onbellek
    # okumasi) ama siralama her turda evrenin daha buyuk bir kismini kapsar.
    backfilled = 0
    if not args.no_backfill:
        for tk in universe_all:
            if tk in bundles:
                continue
            b = yahoo.fetch_cached(tk, period=args.period,
                                   max_age_seconds=args.backfill_days * 24 * 3600)
            if b is not None and b.get("history") is not None:
                bundles[tk] = b
                backfilled += 1
        if backfilled:
            print(f"      onbellekten eklendi: {backfilled} hisse "
                  f"(ag istegi yok) -> toplam {len(bundles)} hisse skorlanacak")

    print(f"      {fetched_live}/{len(tickers)} basarili (%{ok_rate:.0f})"
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
    diag["fetch_success_rate"] = round(ok_rate / 100, 3)
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

    # --- Ciktilar
    OUT.mkdir(parents=True, exist_ok=True)
    html_path = report.write_html(result, diag, OUT / "dashboard.html", top_n=args.top)
    csv_path = report.write_csv(result, OUT / "ranking.csv")

    factor_ids = [f["id"] for f in cfg["factors"]]
    feat = ml.to_feature_matrix(result, factor_ids)
    snap_path = ml.save_snapshot(feat)
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


def cmd_learn(args: argparse.Namespace) -> int:
    """Biriken anlik goruntuleri ileri getiriyle etiketle ve agirliklari ogren."""
    cfg = load_config(Path(args.config))
    factor_ids = [f["id"] for f in cfg["factors"]]

    snaps = ml.load_all_snapshots()
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

    ic = ml.information_coefficients(labeled, factor_ids, label_col)
    if not ic.empty:
        print("\n" + "=" * 66)
        print("BILGI KATSAYILARI (IC) — faktorlerin gercek ongoru gucu")
        print("=" * 66)
        print(f"  {'FAKTOR':<32} {'IC':>8} {'ICIR':>8} {'DONEM':>7}")
        for _, r in ic.iterrows():
            print(f"  {r['factor'][:32]:<32} {r['ic_mean']:>8.4f} "
                  f"{(r['icir'] if r['icir'] is not None else float('nan')):>8.2f} {r['periods']:>7}")
        print("\n  Yorum: |IC|>0.03 zayif-kullanilabilir, >0.05 iyi, >0.10 cok iyi")

    weights = ml.learn_weights(labeled, factor_ids, label_col, method=args.method)
    if not weights:
        print("\nAgirlik ogrenilemedi (yetersiz veri veya pozitif IC yok).", file=sys.stderr)
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

    print("\n" + "=" * 78)
    print(f"  Pano:   {html}")
    print(f"  Gecmis: {hist_path}")
    return 0


def cmd_daily(args: argparse.Namespace) -> int:
    """Gunluk tam dongu: taze tarama + izleme listesi analizi.

    Gorev Zamanlayici'ya baglanacak tek komut budur.
    """
    from src import watchlist

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print("=" * 78)
    print(f"GUNLUK CALISMA — {stamp}")
    print("=" * 78)

    # Bayat fiyatla gunluk karar verilmez: once bos/bozuk kayitlari at.
    purged = cache.purge_invalid("yahoo")
    if purged:
        print(f"[on hazirlik] {purged} bos/bozuk onbellek kaydi temizlendi\n")

    print(">>> 1/2  TARAMA\n")
    rc = cmd_scan(args)
    if rc != 0:
        print("\nUYARI: tarama basarisiz; izleme listesi yine de guncellenecek.",
              file=sys.stderr)

    if not watchlist.load():
        print("\n>>> 2/2  IZLEME LISTESI — bos, atlandi")
        print("     Panodan '+ EKLE' ile hisse sec, sonra:")
        print("     python run.py watch add <SEMBOL> --price <FIYAT>")
        return rc

    print("\n" + ">>> 2/2  IZLEME LISTESI\n")
    watch_args = argparse.Namespace(action="update", ticker=None, price=None,
                                    qty=None, note="", use_cache=False)
    rc2 = cmd_watch(watch_args)

    # --- 3/3  OGRENME DONGUSU ----------------------------------------------
    # Kendi kendini besleyen kisim: veri yeterliyse periyodik olarak yeniden
    # egitir ve yalnizca esikleri gecen modeli terfi ettirir. Yetersizse
    # sessizce ilerlemeyi bildirir — her gun bosuna egitmez.
    from src import dataset as _ds
    from src import training as _tr

    ready = _ds.readiness(getattr(args, "horizon", 21))
    print("\n" + ">>> 3/3  OGRENME DONGUSU\n")
    if not ready["ready_to_train"]:
        print(f"     Veri birikiyor: {ready['snapshots']}/{ready['need_snapshots']} "
              f"anlik goruntu, {ready['span_days']}/{ready['need_span_days']} gun "
              f"(%{ready['progress_pct']})")
        print("     Egitim, esik asilinca kendiliginden baslayacak.")
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
            print(f"     Yeniden egitim basliyor (ufuk {getattr(args,'horizon',21)})...")
            train_args = argparse.Namespace(
                ml_action="train", models=None, horizon=getattr(args, "horizon", 21),
                splits=5, embargo=5, window=10, promote=True, force=False,
                no_cache=False)
            cmd_ml(train_args)

    print("\n" + "=" * 78)
    print("GUNLUK CALISMA TAMAMLANDI")
    print(f"  Tarama panosu : {OUT / 'dashboard.html'}")
    print(f"  Izleme panosu : {OUT / 'watchlist.html'}")
    return rc or rc2


def cmd_ml(args: argparse.Namespace) -> int:
    """Derin ogrenme / geri besleme dongusu komutlari."""
    from src import dataset as ds
    from src import models as mz
    from src import training as tr

    action = args.ml_action

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
        ready = ds.readiness(args.horizon)
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

        results: dict[str, dict] = {}
        for name in names:
            if name not in mz.AVAILABLE:
                print(f"  ! bilinmeyen model atlandi: {name}")
                continue
            print(f"\n>>> {name} egitiliyor (ufuk {args.horizon}, "
                  f"{args.splits} katman, embargo {args.embargo})...")
            res = tr.walk_forward(name, horizon=args.horizon, n_splits=args.splits,
                                  embargo=args.embargo, window=args.window,
                                  use_cache=not args.no_cache)
            results[name] = res
            if not res.get("ok"):
                print(f"    BASARISIZ: {res.get('reason')}")
                continue
            print(f"    IC {res['ic_mean']}  ICIR {res['icir']}  "
                  f"katman {res['folds']} ({res['positive_folds']} pozitif)")
            print(f"    ilk-dilim getiri farki: {res['top_decile_spread']}")

        base = results.get("ridge")
        print("\n" + "=" * 74)
        print("DEGERLENDIRME")
        print("=" * 74)
        print(f"  {'MODEL':<8} {'IC':>9} {'ICIR':>8} {'KATMAN':>7} {'DILIM FARKI':>13}  TERFI")
        best_name, best_dec = None, None
        for name, res in results.items():
            if not res.get("ok"):
                print(f"  {name:<8} {'—':>9} {'—':>8} {'—':>7} {'—':>13}  hayir")
                continue
            dec = tr.promotion_check(res, baseline=base if name != "ridge" else None)
            tr.record_candidate(res, dec)
            print(f"  {name:<8} {res['ic_mean']:>9.4f} {(res['icir'] or 0):>8.2f} "
                  f"{res['folds']:>7} {(res['top_decile_spread'] or 0):>13.4f}  "
                  f"{'EVET' if dec['promote'] else 'hayir'}")
            if dec["promote"] and (best_dec is None or
                                   (res["ic_mean"] or 0) > (results[best_name]["ic_mean"] or 0)):
                best_name, best_dec = name, dec

        if best_name is None:
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
    p.add_argument("--backfill-days", type=int, default=5,
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
    p.add_argument("--no-cache", action="store_true")

    lp = sub.add_parser("learn", help="agirliklari gecmis verilerden ogren")
    lp.add_argument("--config", default=str(ROOT / "config" / "weights.yaml"))
    lp.add_argument("--horizon", type=int, default=21, help="ileri getiri ufku (islem gunu)")
    lp.add_argument("--method", default="ic", choices=["ic", "ridge"])
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
    mp.add_argument("--no-cache", action="store_true")

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
    if args.cmd == "clear-cache":
        return cmd_clear_cache(args)
    return cmd_scan(args)


if __name__ == "__main__":
    raise SystemExit(main())
