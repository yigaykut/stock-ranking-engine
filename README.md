# Stock Ranking Engine

Screens roughly 2,800 US small and mid-cap stocks every day, scores each one on
28 parameters, and ranks them. Built to find companies that still have room to
grow, not the ones that already did.

The output is a self-contained HTML dashboard you open by double-clicking. No
server, no internet needed once it's generated.

> **This is not investment advice.** Every score here is a statistical
> indicator computed from public data. None of it predicts the future, and the
> weights have not been validated against actual returns yet. See
> [Where this stands](#where-this-stands).

---

## Quick start

```bash
pip install -r requirements.txt
cp config/weights.example.yaml config/weights.yaml
python run.py daily --universe smallcap,midcap,wsb --workers 4
```

The example config ships with placeholder weights. Set your own before you
trust the output — mine are tuned and not in this repo.

Then open `output/dashboard.html`.

That's the whole thing. It takes 10-15 minutes on a cold cache, a few minutes
after that.

If you just want to try it without waiting:

```bash
python run.py --universe smallcap --limit 40
```

---

## What you get

Three files land in `output/`:

| File | What it is |
|------|-----------|
| `dashboard.html` | The ranking, with a breakdown of why each stock scored what it did |
| `watchlist.html` | Price targets, stops and sell signals for stocks you're tracking |
| `ranking.csv` | Everything, flat, for Excel |

The dashboard has four sections: the ranked chart, the parameter weights, a
sortable table, and a question-answering assistant that works offline.

---

## Picking the universe

This matters more than the parameters do. The S&P 500 is full of companies that
already finished growing, so screening it for "promising small companies" gets
you nowhere.

The system pulls every US-listed stock from the Nasdaq listing service and
filters by market cap:

| Preset | Range | Stocks | Use it for |
|--------|-------|-------:|-----------|
| `micro` | 50M – 300M | ~3,300 | High risk, high potential |
| `smallcap` | 300M – 2B | ~1,500 | The classic small-cap band |
| `midcap` | 2B – 10B | ~1,240 | Proven but still growing |
| `emerging` | 200M – 10B | ~2,750 | Default |
| `largecap` | 10B – 200B | ~660 | Mature companies |
| `sp500` | — | 503 | Index constituents |
| `wsb` | — | ~60 | Most-mentioned on r/wallstreetbets |

```bash
python run.py --universe smallcap,midcap,wsb
python run.py --min-mcap 5e8 --max-mcap 4e9      # your own band
```

`--limit` samples evenly across the band rather than taking the first N, so you
don't accidentally narrow it to just the biggest names.

---

## How scoring works

Each parameter produces a raw number in its own units. Those get converted to a
0-100 score based on where the stock sits **relative to everything else in the
scan** — a P/E of 12 means nothing on its own, but "cheaper than 85% of the
universe" does.

Scores are then z-scored within each sector, so "all tech is expensive right
now" doesn't distort the comparison.

**Missing data doesn't hurt a stock.** If a parameter can't be computed, its
weight is redistributed across the parameters that could. But the score is also
pulled toward neutral (50) in proportion to how much data is missing:

```
final = 50 + (raw − 50) × coverage
```

This isn't a penalty. It's symmetric — a bad score with thin data moves *up*
toward 50 too. The logic is that if you don't have the data, you can't make a
strong claim in either direction.

A parameter that's measurable for less than 15% of the universe gets switched
off entirely and its weight redistributed. Reddit mentions usually end up here
when you scan small caps.

---

## The parameters

28 of them. The weights come from factor literature and established practitioner
systems — Zacks Rank, IBD's CAN SLIM, the Piotroski F-Score, Fama-French,
Novy-Marx — not from guessing.

The weights themselves are tuned and kept private, but the structure is all
here. `config/weights.example.yaml` lists every parameter with placeholder
weights — copy it to `config/weights.yaml` and set your own. Only the ratios
matter; the engine normalizes the total to 100.

Grouped by family, the parameters are:

| Family | Parameters |
|--------|-----------|
| Technical | Trend structure, Stage 2 breakout, breakout setup, 30-signal summary, trend smoothness, volume confirmation |
| Emerging | Revenue scaling, size opportunity, Rule of 40, undiscovered |
| Potential | Analyst target upside, chart position |
| Quality | Financial health, cash runway, profitability |
| Analyst | Estimate revisions, earnings surprise, rating level |
| Value | Valuation composite |
| Momentum | Price momentum (12-1), relative strength |
| Risk | Volatility/beta/drawdown, liquidity |
| Sentiment | Reddit WSB attention, short squeeze |
| Preference | Nominal price fit |
| Model | Learned prediction (starts at zero weight) |

Each one is documented in detail in `docs/PARAMETRELER.md`, which stays local
because it lists the tuned weights.

### Correlated parameters share a budget

Six of the technical parameters correlate with each other at r = 0.64 to 0.79.
They're all asking the same question: is this stock trending up? Left alone,
their weights added up to a quarter of the total score, which meant a quarter of
the score was one bet wearing six hats.

They now share a single weight budget. Their proportions to each other stay
intact, but the total is capped at roughly half what it was. You still see each
one broken out in the dashboard.

### Penalties scale with the distribution

Five penalty rules subtract from the total. They're defined in units of standard
deviation rather than fixed points, so they stay proportional as the score
distribution widens or narrows. In severity order:

1. Negative free cash flow combined with high debt — the most common way to lose
   money in small caps
2. Hype plus overbought (RSI above 75 while trending on WSB)
3. Parabolic extension, more than 40% above the 50-day average — good stock, bad
   entry
4. Distribution: price rising while on-balance volume falls
5. Earnings within 7 days

The exact magnitudes are in your config. An earlier version used fixed point
values, and an audit found a single penalty was averaging five times the gap
between the top ten stocks and the next ten. Scaling them to the distribution
fixed that.

---

## Three parameters you asked for, and what I did with them

**"Show a Buy or Strong Buy on Investing.com"** — included, but weighted low on
purpose.
The evidence says the *level* of an analyst rating barely predicts anything;
most analysts say Buy most of the time. What does predict is the *direction of
revisions*. So there are two parameters from the same family, and the one that
measures change carries almost twice the weight.

**"Should be on the WSB chart"** — included, weighted low. Studies show mention
spikes
produce positive returns over 1-5 days and significantly *negative* alpha over
longer horizons. So the system measures acceleration rather than raw mentions
(going from 20 to 200 tells you something; sitting at 200 doesn't), applies a
penalty when hype coincides with overbought conditions, and switches the
parameter off when coverage drops below 15%.

**"Price shouldn't be too high"** — included at the lowest weight in the system. A stock's nominal
price has no statistical relationship with its future return. A $500 stock isn't
riskier than a $5 one; splits prove this. What you actually meant is covered by
two other parameters that carry real weight: `valuation_composite` for "is it
expensive" and `risk_drawdown` for "how likely is it to fall". Your preference
is still in there. Turn it up if you disagree:

```bash
python run.py --weight nominal_price_fit=6
```

---

## Technical analysis

30 signals, computed locally rather than scraped:

- **12 moving averages** — SMA and EMA at 5, 10, 20, 50, 100, 200
- **9 classic oscillators** — RSI, Stochastic, StochRSI, MACD, ADX, Williams %R,
  CCI, ROC, Ultimate
- **9 extended** — Bollinger %B, Aroon, MFI, Donchian position, Supertrend,
  Ichimoku cloud, MA cross, trend R², volume confirmation

The vote count reproduces Investing.com's technical summary. Doing it locally
means the site's HTML can change without breaking anything, and the intermediate
values stay available as model features.

---

## Watchlist

Add a stock and the system tracks it daily.

```bash
python run.py watch add LQDT --price 41.50
python run.py watch add PRG,OMDA           # no position, just watching
python run.py watch update
```

Or click **+ EKLE** in the dashboard, set the entry price in the panel that
appears, and copy the command.

You get a price axis per stock:

```
stop ──────── entry ── now ──────── short target ──── long target
```

Short-term targets (1-3 months) come from the median of three technical methods:
the next resistance cluster, an ATR projection, and a measured move. Long-term
targets (12 months) blend the analyst consensus with a valuation estimate. The
method used is always shown, so you can disagree with a specific one.

Stops use ATR rather than a fixed percentage, because 8% is a rounding error on
a volatile biotech and a catastrophe on a utility. A chandelier trailing stop
ratchets up as the price rises and never moves down.

15 sell rules, each with a severity from 1 to 5. Severity determines the risk
level:

| Level | Triggers | What to do |
|-------|----------|-----------|
| **SAT** | Stop hit, Stage 4 downtrend | Exit |
| **YÜKSEK RİSK** | Below MA150, target reached, distribution, score dropped 12+ | Take partial, tighten stop |
| **DİKKAT** | Below MA50, parabolic, stop within 4% | Watch closely, don't add |
| **İZLE** | RSI > 78, volatility spike, MACD cross, earnings soon | Stick to the plan |

Stocks in your watchlist never leave the ranking, even if they drop to last
place. They're marked with a star. This matters: in one scan the bottom-ranked
stock out of 1,094 was in the watchlist, stayed visible, and threw a SAT signal
the same day. Drop it from the list and you'd never have seen that.

---

## The assistant

Section IV of the dashboard answers questions about the scan. It runs entirely
offline and costs nothing, because it isn't a language model — it reads the data
already embedded in the page.

The tradeoff is real: it can't hallucinate, but it only understands patterns it
knows.

| Ask | Get |
|-----|-----|
| `LQDT nasil?` | Rank, score, strongest and weakest parameters, returns |
| `en iyi 10` | Top 10 by total score |
| `en ucuz 5` | Cheapest by valuation |
| `kirilim kurulumu en iyi` | Best on any single parameter |
| `teknoloji hisseleri` | Sector filter |
| `50 dolar alti` | Price filter |
| `LQDT vs PRG` | Side-by-side by category |
| `40 kurali nedir` | Parameter explanation plus top 5 on it |

Parameter explanations come from `rationale_tr` in the config, so changing a
weight changes what the assistant tells you.

---

## Running it daily

A Windows scheduled task called `HisseSiralama_Gunluk` runs `scripts/gunluk.bat`
at 07:00 every day. US markets close at 23:00 local time, so the data has
settled by then.

The task is set to run when the machine next wakes if it was off, and it won't
start a second copy on top of a running one. Logs go to `logs/`, 30 days
retained.

```powershell
Get-ScheduledTaskInfo -TaskName "HisseSiralama_Gunluk"    # status
Start-ScheduledTask   -TaskName "HisseSiralama_Gunluk"    # run now
Disable-ScheduledTask -TaskName "HisseSiralama_Gunluk"    # pause
Unregister-ScheduledTask -TaskName "HisseSiralama_Gunluk" -Confirm:$false
```

The one requirement is that the computer is on.

### Rate limits and rotation

Yahoo's free endpoint won't hand over 2,800 stocks × 12 requests in one go. It
starts refusing partway through.

Each run fetches the 800 least-recently-scanned symbols. Everything fetched in
earlier runs gets scored too, straight from cache, with no network calls. So one
run costs ~800 requests and ranks ~2,400 stocks, and coverage grows every time.
It reached 99.7% after a handful of runs.

If Yahoo starts refusing, a circuit breaker stops the scan cleanly after 25
consecutive rejections rather than burning an hour failing. Failed fetches are
never cached, so a temporary rate limit doesn't mark a stock as "no data" for
the next six hours.

---

## Publishing

The dashboard can be encrypted into a single file you can host anywhere.

```bash
python run.py publish        # prompts for a password
```

AES-256-GCM with PBKDF2-SHA256 at 600,000 iterations. Decryption happens in the
browser through WebCrypto. The password never leaves your machine — there's no
server involved.

That means you can put the file on GitHub Pages, a web host, cloud storage, or
in an email, and you don't have to trust any of them.

For GitHub Pages, see **[docs/YAYIN.md](docs/YAYIN.md)**. Short version:

```bash
gh auth login
gh repo create hisse-pano --public
setx DASHBOARD_PASSWORD "five-random-words-here"
setx HISSE_REPO "user/hisse-pano"
python run.py publish && python run.py deploy --repo user/hisse-pano
```

Once those two environment variables are set, the daily task publishes on its
own.

### About the password

Security here rests entirely on the password. The ciphertext sits in the file,
so an attacker can try as many guesses as they like offline. 600,000 PBKDF2
iterations makes each guess slow, but that won't save a short password.

Five random words is fine. `sifre123` is not. The tool refuses anything under
~45 bits of entropy unless you pass `--force`.

### Nothing sensitive gets published

Publishing copies files into a separate `publish/` directory rather than turning
the project into a repo. Only encrypted files go there.

Before pushing, every file is scanned. The deploy aborts if it finds a missing
encryption marker, any plaintext trace (`const DATA`, ticker names, section
headings), or PBKDF2 iterations below 100,000.

Tested: plaintext dashboards get rejected, tampered files get caught, weakened
encryption gets refused. Your positions, cache, logs and source never enter the
repo.

### Viewing on your phone

```bash
python run.py serve --lan
```

Anything on the same Wi-Fi can reach it. Serves the encrypted version if one
exists.

---

## Learning system

The system trains on its own accumulated snapshots and feeds what it learns back
into scoring — but only when there's evidence it works.

```bash
python run.py ml status
python run.py ml train --promote
```

The daily loop does this automatically as step 3.

### Collection is daily, training is periodic

Three different rhythms, each tied to something different:

| Step | Rhythm | Why |
|------|--------|-----|
| Collection | Once a day, after close | The parameters resolve daily. Intraday sampling adds API load and half-bar noise, not signal |
| Labeling | Continuous, automatic | A day's label matures 21 trading days later |
| Retraining | Every 5th scan | One day of new data won't move the weights, but it will add overfitting risk |

Streaming updates would be wrong here. The target is a 21-day forward return;
adjusting the model intraday means fitting to outcomes that haven't happened.

### The model can't run away with your money

Its weight in the score is proportional to measured out-of-sample skill. No
evidence, no weight.

| Measured ICIR | Score weight |
|---------------|-------------:|
| No skill | **0.0** |
| 0.35 (weak) | 0.9 |
| 1.20 (strong) | 12.0 (ceiling) |

The ceiling is 12 against a total of ~137, so the model can never account for
more than about 8% of a score. It doesn't replace the readable parameters. It
adds an opinion alongside them.

To get promoted a model needs IC above 0.02, ICIR above 0.30, at least 3 folds,
positive results in 60% of them, and it has to beat a plain ridge regression by
0.005 IC. That last one matters: a neural net that only matches ridge is more
overfitting risk and less transparency for nothing.

### Three models

- **RidgeRanker** — closed-form, numpy only. The baseline. Nothing gets used
  unless it beats this.
- **MLPRanker** — catches non-linear interactions, like "cheapness only helps
  when the trend is intact".
- **SeqRanker** — a GRU over each stock's last 10 days of parameter history.

The sequence model is where deep learning actually earns its place here.
Cross-sectional models see what a stock looks like today. This one sees how it
got there — whether the score rose into this level or fell into it.

That claim is tested. On synthetic data where the signal lives *only* in the
change, the cross-sectional model scores IC 0.308 and the sequence model 0.757.

### Leakage prevention

21-day labels on consecutive daily snapshots overlap by about 95%. Naive
cross-validation makes any model look good because training labels contain
prices from the test period.

Splits purge `horizon + embargo` days backward from each test window. The tests
verify that training and test dates never intersect and that the gap is at least
that wide.

### What state it's in

```
snapshots : 2 / 60      span : 1 / 120 days      progress : 0.8%
ready to train : no
```

Training is blocked until there's enough data, and that's deliberate. A model
trained on a few days produces confident-looking numbers fitted entirely to
noise, and you find out it's wrong after losing money.

The pipeline's correctness was verified today with synthetic data instead:

```bash
python tests/test_ml.py     # 22 tests
```

The one that matters most: on data with no signal at all, the model finds
IC 0.001. If it were overfitting, it would find something there too.

More detail: **[docs/OGRENME.md](docs/OGRENME.md)** (Turkish)

---

## Where this stands

I audited this system and wrote up what's wrong with it — 20 findings, each
backed by measurements from production data. 11 have been fixed. The full report
(`docs/EKSIKLIKLER.md`) stays local because it quotes the tuned weights, but
here's what the fixes changed:

| Finding | Before | After |
|---------|-------:|------:|
| Stocks scored | 1,094 | 2,396 |
| Universe coverage | 46% fetched | 92% scanned |
| Average penalty | −6.60 pts | −1.95 pts |
| Penalty vs top-10 gap | 5.0× | 1.7× |
| Trend cluster share of score | 25.1% | 12.2% |
| size ↔ liquidity correlation | +0.65 | +0.00 |

**The big one is still open, and it can't be fixed with code.** Nobody has
measured whether these weights actually predict anything. That takes 90 days of
data. The dashboard says so at the top rather than hiding it.

Two other things worth knowing. The universe comes from today's listings, so
companies that went bankrupt or got delisted never appear — any backtest built
from this data will be biased upward. And the scores at the top are closer
together than one day of noise, which is why the dashboard shows percentile
bands and a ± range instead of leaning on exact rank.

---

## Tests

```bash
python tests/test_scoring.py      # 21 — scoring engine
python tests/test_ml.py           # 22 — learning pipeline, synthetic
node   tests/test_dashboard.js    # 28 — dashboard UI, real DOM
```

The UI tests need jsdom:

```bash
npm install jsdom --no-save
```

They run against a real DOM for a reason. An earlier stub version returned an
empty list from `querySelectorAll`, so the code inside button loops never ran,
and a `ReferenceError` thrown at load time went unnoticed. In a browser that
error killed the script, the chat form never got its handler, and submitting it
reloaded the page. A real DOM catches that class of bug.

---

## Layout

```
config/weights.example.yaml   parameter template (real weights are private)
run.py                  CLI
src/
  universe.py           universe construction and market-cap bands
  factors.py            28 raw parameters and penalty flags
  scoring.py            normalize, weight, redistribute, rank
  indicators.py         RSI, MACD, ADX, OBV, Bollinger, Ichimoku...
  investing_summary.py  30-signal technical vote
  targets.py            price targets, stops, sell signals
  watchlist.py          position tracking
  dataset.py            training panel, leak-free splits, readiness gate
  models.py             RidgeRanker, MLPRanker, SeqRanker
  training.py           walk-forward training, evaluation, promotion
  publish.py            AES-256-GCM encryption
  deploy.py             GitHub Pages with leak protection
  report.py             dashboard HTML
  theme.py              visual identity, parametric curves
```

The dashboard interface is Turkish. Some Turkish docs stay local because
they quote the tuned weights.

---

## Data sources

| Source | Provides | Key needed |
|--------|----------|-----------|
| Yahoo Finance (`yfinance`) | Prices, fundamentals, analyst estimates | No |
| Nasdaq screener | Full US listings with market caps | No |
| ApeWisdom | WSB mention counts and 24h change | No |
| Tradestie | WSB sentiment | No |

**Why not scrape Investing.com?** Both parts are already covered. The technical
summary is a deterministic vote count that `investing_summary.py` reproduces
locally, which means site changes can't break it. The analyst consensus is the
same sell-side data Yahoo exposes through an API.

---

## Known limits

- Yahoo's endpoint is unofficial and rate-limited. Wide scans spread across
  several runs.
- Fundamentals arrive with a quarterly lag. Restatements aren't tracked.
- WSB data is a snapshot with no history, so acceleration is limited to a 24-hour
  window.
- The system ranks cross-sectionally — which stock looks better than which
  today. It says nothing about whether today is a good day to buy.
- Commissions, spreads, slippage and taxes aren't modeled.
- US equities only.
