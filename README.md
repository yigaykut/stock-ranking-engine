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

The first version of this had one trigger at 07:00 and `StartWhenAvailable` set,
which sounds like enough. It isn't: two runs were missed on days the machine
happened to be off at 07:00 and the catch-up never fired. Now there are eight
triggers a day — 07:00 repeating every two hours until 21:00 — plus one three
minutes after you log in.

That would be eight full scans a day, so the batch file makes itself idempotent.
A successful run writes the date into `logs/son_basari.txt`, and any later run
that same day exits immediately. Seven of the eight triggers cost nothing; the
one that matters is whichever fires first after the machine comes on. Force a
rerun with `gunluk.bat force`.

The task won't start a second copy on top of a running one, and it retries three
times at 15-minute intervals if the script fails. Logs go to `logs/`, 30 days
retained.

```bash
python scripts/durum.py      # is it running, where's the counter, is there a model
```

```powershell
Get-ScheduledTaskInfo -TaskName "HisseSiralama_Gunluk"    # status
Start-ScheduledTask   -TaskName "HisseSiralama_Gunluk"    # run now
Disable-ScheduledTask -TaskName "HisseSiralama_Gunluk"    # pause
Unregister-ScheduledTask -TaskName "HisseSiralama_Gunluk" -Confirm:$false
```

The dashboard used to show one fewer snapshot than it had, every single day —
the page was written before the day's snapshot was saved, so it always reported
yesterday's count. Watching a counter that never seems to move is a good way to
conclude the automation is broken when it isn't. The snapshot is now saved
first, and the banner also gives the date the counter is expected to fill.

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

### Four models

- **RidgeRanker** — closed-form, numpy only. The baseline. Nothing gets used
  unless it beats this.
- **MLPRanker** — catches non-linear interactions, like "cheapness only helps
  when the trend is intact".
- **SeqRanker** — a GRU over each stock's last 10 days of parameter history.
- **AttnRanker** — attention across the whole cross-section of one day.

The sequence model is where deep learning actually earns its place here.
Cross-sectional models see what a stock looks like today. This one sees how it
got there — whether the score rose into this level or fell into it.

That claim is tested. On synthetic data where the signal lives *only* in the
change, the cross-sectional model scores IC 0.308 and the sequence model 0.757.

Until recently the GRU could be measured but never used: live prediction bailed
out on any sequence model, because a single day's row is not a sequence. It now
builds the window out of the feature store — the previous snapshots for that
ticker, followed by today's row — with the same ordering as training, percentile
first and windowing second. Get that order wrong and the model quietly sees
something other than what it learned on.

The attention model attacks a different gap. The other three score each stock
on its own; the only thing making them cross-sectional is the loss function.
But ranking is relative by nature. Momentum of 8% means one thing on a day when
everything else sits at 2% and something else entirely when the field is at
20%. AttnRanker takes a day's stocks as a set and updates each stock's
representation by attending over the others in the same day, so the comparison
lives in the architecture rather than being applied afterwards by
normalization.

There is no positional encoding, and that is the interesting part. A day's
stocks are an unordered set, so permuting the input has to permute the output
and change nothing else. The test checks it directly and the deviation comes
out at 4.8e-07. Adding positional encoding would break that test, which is
exactly why the test is there — it pins down a design decision that would
otherwise be invisible.

Attention costs O(n²) in the size of the day. Training splits each day into
chunks of at most 256 stocks, which bounds the cost and doubles as
regularization, since every chunk is a random subset of that day's
cross-section. Inference runs the day whole.

A stock with too little history gets no prediction rather than a window padded
out with repeats of one row. It drops out of that parameter and the coverage
machinery handles the rest, which is the honest outcome: no opinion is better
than a fabricated one.

### Leakage prevention

21-day labels on consecutive daily snapshots overlap by about 95%. Naive
cross-validation makes any model look good because training labels contain
prices from the test period.

Splits purge `horizon + embargo` days backward from each test window. The tests
verify that training and test dates never intersect and that the gap is at least
that wide.

### Getting off the ground without waiting four months

The feature store grows one snapshot a day and the gate wants 60 of them over
120 days. Starting from zero that's four months before you can even find out
whether the GRU is worth having.

But the cache already holds two years of daily bars for every stock. Price-based
factors can be recomputed for any past day — slice the series at that date and
call the same function. `python run.py history` does that across the cached
universe and produces roughly a year of historical snapshots in about an hour.

It calls `factors.f_*` on a truncated DataFrame rather than reimplementing the
formulas in vectorised form. Vectorised would be maybe fifty times faster, but
it would also mean training features and live features could drift apart without
anyone noticing.

An hour-long job that loses everything when interrupted is a bad job. Results
are written to disk every 150 symbols along with the list of what's been
processed, so a killed run resumes where it stopped. `--restart` starts over,
and `--merge-only` turns whatever batches exist into a usable panel without
waiting for the rest — you can train on 300 stocks while the other 3,000 are
still being computed.

Labelling reads from cache first. It used to call the normal fetch path, whose
six-hour TTL meant a few hundred symbols turned into a few hundred network
requests; one training run sat waiting on the network for minutes before doing
any arithmetic. Forward returns come from price history that is already on disk,
so the network is now a fallback for symbols with no cache entry at all.

Only the eleven price and volume factors are backfilled. Fundamentals, analyst
ratings, EPS revisions, short interest and ownership come from Yahoo as they
stand today, with no history. Writing today's known profitability onto a day
twelve months ago is textbook look-ahead, so those factors are simply absent.

Two biases remain and neither is fixable here. The cached universe is today's
listings, so companies that collapsed and got delisted are missing — measured
skill on this panel is higher than what you'd have earned. And rotation means the
cache is the most recently scanned slice, not a random sample.

So this panel cannot promote a champion. `promotion_check` refuses any result
carrying the `pretrain` flag no matter how good the numbers look, the pretrained
model is saved to a separate file, and the two stores never merge. It's for
choosing an architecture. The decision to let a model touch the score still waits
for real forward snapshots.

### What state it's in

```
live snapshots : 14 / 60     span : 24 / 120 days     progress : 20%
ready to train : no
```

Training on live data is blocked until there's enough of it, and that's
deliberate. A model trained on a few days produces confident-looking numbers
fitted entirely to noise, and you find out it's wrong after losing money.

### All four measured on the reconstructed panel

188,465 labelled rows, 73 dates, 21-day horizon, purged walk-forward, with
training properly seeded:

| Model | IC | ICIR | folds | decile spread |
|---|---:|---:|---:|---:|
| ridge (baseline) | −0.0226 | −0.85 | 3 (1 positive) | +0.031 |
| mlp | **+0.0272** | **+0.90** | 3 (**3 positive**) | +0.024 |
| seq | +0.0239 | +0.74 | 2 (1 positive) | +0.019 |
| attn | +0.0042 | +0.09 | 3 (2 positive) | +0.003 |
| ensemble (all four) | −0.0056 | −0.46 | 2 (1 positive) | +0.006 |

An earlier version of this table had attn at +0.0307 with an ICIR of 1.24 and
called it the most consistent of the four. That run wasn't seeded — see below.
With the seeding fixed, attn lands at +0.0042, which is not distinguishable
from nothing. Its whole apparent advantage was which random initialisation it
happened to get.

What survives repetition: the linear baseline is the only negative model, and
mlp is the only one positive on every fold. The gap between mlp and seq (0.003)
is far smaller than the run-to-run spread I measured before seeding was fixed
(0.014), so those two are not separated by this data.

The ensemble is worse than every member it contains, which is the same thing
the August run found. Averaging helps when the members make independent
mistakes; here they don't, so averaging reinforces the shared error rather than
cancelling it. The ensemble is a candidate like any other and it got rejected
like any other.

Nothing promotes. The panel carries survivorship bias, fold counts are small,
and the gate refuses pretrain results by construction.

### The seeding bug that produced the first table

Running the same command twice on the identical panel gave mlp +0.0321 once and
+0.0181 the other time. ridge matched exactly both times, which is what pointed
at the cause — ridge has no torch in it.

`torch.manual_seed` was called inside the training loop, but the network is
built before that in `fit()`, so weight initialisation read whatever state the
global generator happened to be in. The per-epoch batch shuffle used
`np.random.shuffle` on the unseeded global numpy generator, so SGD walked a
different path each run. The `seed` argument was threaded through the whole
stack and quietly ignored.

Both fixed, and locked down by `tests/test_tekrarlanabilirlik.py`: two fits
with the same seed are bit-identical for all three torch models, a different
seed changes the result, and corrupting both global generators beforehand
changes nothing.

`attn` is expensive regardless: attention is O(n²) in the number of stocks in a
day, and a day here is ~2,580 stocks even chunked at 256. It takes roughly
three hours where the other three take minutes.

```bash
python tests/test_ml.py         # 29 — pipeline correctness, synthetic data
python tests/test_backfill.py   # 34 — historical panel, leakage, bias brake, resume
```

The two that matter most: on data with no signal at all the model finds IC 0.001,
so it isn't overfitting; and tripling every price bar after a snapshot's date
changes nothing about that snapshot, so the historical panel isn't reading the
future.

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
python tests/test_ml.py           # 29 — learning pipeline, synthetic
python tests/test_backfill.py     # 34 — historical panel, leakage, bias brake, resume
python tests/test_evren.py        # 17 — universe collapse, stale-cache recovery
python tests/test_topluluk.py     # 18 — ensemble blending, fold alignment
python tests/test_otomasyon.py    #  9 — daily stage isolation
python tests/test_kodlama.py      #  5 — source encoding guard
python tests/test_kaynak_uyum.py  # 15 — provider shape contract, timezone join
python tests/test_attn.py         # 14 — set behaviour, permutation equivariance
python tests/test_kayip.py        # 18 — ranking loss, outlier insensitivity
python tests/test_faktor_zaman.py # 30 — overlap-corrected t, decay, regime split
python tests/test_tekrarlanabilirlik.py  # 11 — same seed, same result
node   tests/test_dashboard.js    # 28 — dashboard UI, real DOM
```

They don't run under pytest — each file calls `sys.exit()`, which pytest treats
as a collection error. Run them as scripts.

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
  backfill.py           historical snapshots rebuilt from cached price bars
  models.py             RidgeRanker, MLPRanker, SeqRanker, AttnRanker
  training.py           walk-forward training, evaluation, ensemble, promotion
  paper.py              cohort ledger — what the top 20 actually did
  fundamentals.py       daily point-in-time fundamentals archive
  delisting.py          survivorship-bias ledger
  regime.py             market regime: trend, breadth, volatility;
                        back-labels past dates with the same rule
  faktor_zaman.py       factor strength over time: overlap-corrected t,
                        first-half vs second-half, regime split
  notify.py             alerts — file, desktop, optional Telegram
  backup.py             encrypted archive of the irreplaceable data
  publish.py            AES-256-GCM encryption
  deploy.py             GitHub Pages with leak protection
  report.py             dashboard HTML
  theme.py              visual identity, parametric curves
  providers/
    yahoo.py            primary source
    nasdaq.py           fallback price source, listings
    cache.py            disk cache with stale-read support
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
  several runs. There's a fallback price source, but it only covers prices.
- Fundamentals arrive with a quarterly lag. Restatements aren't tracked.
- Fundamental history only starts from the day the archive was switched on.
  Anything before that is gone for good — Yahoo won't serve it.
- WSB data is a snapshot with no history, so acceleration is limited to a 24-hour
  window.
- The system ranks cross-sectionally — which stock looks better than which
  today. The regime banner gives context, but nothing here times the market.
- Spreads and slippage are *estimated* from liquidity, not measured. Taxes
  aren't modeled at all.
- US equities only.

---

## The scorecard

Ranking quality was invisible for a long time. IC measurement needs 60
snapshots, which won't arrive until December. But "what happened to the top 20"
can be answered from price data that's already on disk.

Every scan writes its top 20 into a ledger as a **cohort**. Each cohort is held
21 trading days and marked against SPY. It isn't a portfolio simulation —
capital, position sizing and cash management are deliberately absent, because
none of those measure the thing being tested. 73 cohorts are 73 independent
measurements of the ranking itself.

```bash
python run.py paper build --panel   # fill the ledger, including history
python run.py paper                 # show the scorecard
```

The first run over 11 months of reconstructed history:

| | Top 20, 21-day hold |
|---|---:|
| Cohorts | 73 |
| Mean return | +1.81% |
| SPY over the same windows | +1.53% |
| **Excess** | **+0.29%** |
| Hit rate | 55.6% |
| t-statistic (per cohort) | 0.67 |

t = 0.67 means the excess is **indistinguishable from noise**. And this is the
optimistic reading: the historical panel carries survivorship bias, uses only
the 11 price-derived parameters, and applies no penalties. It's an upper bound,
not a result.

The dashboard shows both tracks — real scans and reconstructed history — with
that warning attached to the second one.

---

## Do the parameters work?

Same question, one level down. Information Coefficient is the rank correlation
between a parameter's score and the stock's forward return. One average per
parameter over 73 dates was the whole answer for a while, and it turned out to
be three questions wearing one number.

**Is the average different from zero?** The labels are 21-day forward returns
sampled every ~3 days, so about nine consecutive readings share the same future
window. They are not independent, and the usual `t = mean/(std/√n)` assumes
they are. Newey-West with a Bartlett kernel fixes it, and the lag comes from
the actual snapshot spacing:

| Parameter | naive t | corrected t |
| --- | ---: | ---: |
| momentum_persistence | **2.20** | **1.26** |
| breakout_setup | −1.96 | −1.10 |
| stage2_breakout | 1.43 | 0.83 |
| trend_structure | 1.43 | 0.81 |

None of the 11 clear |t| ≥ 2. Without the correction `momentum_persistence`
would have read as significant.

**Is it fading?** Split the window in half and a pattern shows up that no
single parameter shows on its own:

| Parameter | first half | second half |
| --- | ---: | ---: |
| momentum_persistence | +0.0564 | −0.0025 |
| trend_structure | +0.0476 | −0.0146 |
| chart_position | +0.0417 | −0.0150 |
| relative_strength | +0.0386 | −0.0448 |
| stage2_breakout | +0.0340 | +0.0016 |
| technical_oscillators | +0.0221 | −0.0323 |
| price_momentum_12_1 | +0.0168 | −0.0214 |

The whole trend/momentum family earns in the first half and gives it back in
the second. Six of the seven are already in the `trend` correlation cluster,
whose budget gets scaled down to about 40% of its configured total, so their
applied share of the score is well under half what the config asks for. The
correlation budget was already damping exactly this family, without knowing any
of this — it only knew they move together.

(Per-parameter weights stay out of this file, same as everywhere else. Where
weight matters to a finding it's described in relative terms.)

**Does it only work in one kind of market?** Regime labels turn out to be
reconstructable for past dates: the rule reads only the index's own price
history, and breadth never enters the label, only the warning text. So the 73
panel dates get the same label the live scan would have given them.

| Parameter | rising (57d) | transitional (16d) |
| --- | ---: | ---: |
| momentum_persistence | +0.0280 | +0.0212 |
| stage2_breakout | +0.0263 | −0.0134 |
| risk_drawdown | +0.0175 | −0.0474 |
| technical_oscillators | +0.0065 | −0.0482 |
| breakout_setup | −0.0107 | **−0.0707** |
| price_momentum_12_1 | −0.0167 | +0.0479 |

Breakout and trend parameters earn while the trend holds and give it back when
it doesn't. `breakout_setup` is one of the four heaviest parameters in the
config and it posts the largest single effect in the table, in the wrong
direction. `momentum_persistence` is the only one that behaves about the same
in both.

And the limit that matters most: **the window contains no falling market at
all** — 57 rising days, 16 transitional, zero down. The index never went below
its 200-day average between 2025-09 and 2026-07. For a system this weighted
toward trend, that's the case you'd most want measured, and it isn't here.

Nothing is auto-adjusted from any of this. The measurement is a suggestion;
changing weights is the user's call. Table on the dashboard, summary in
`scripts/durum.py`, raw output in `data/faktor_zaman.json`.

```bash
python run.py learn --pretrain      # measure against reconstructed history
python run.py learn                 # measure against real snapshots
```

---

## Everything else that got added

**Fundamentals archive.** Yahoo only serves *today's* fundamentals. Every scan
was computing them, using them, and throwing them away — which is why the
reconstructed panel could only carry 11 price parameters. They're now stored
daily (`data/fundamentals/`, gzipped, ~90 KB/day). In six months there'll be
enough history to train on all 28.

**Delisting ledger.** Companies that disappear are tracked against the *full*
listings feed, not the market-cap-filtered universe — a company crossing the
$20B ceiling also leaves the universe, and counting that as a delisting would
invert the bias instead of fixing it. Confirmed after 5 missing days; the last
known price is captured on the first missing day, while it's still in cache.
Positions in delisted names close at that price rather than silently vanishing.

**Fallback price source.** Yahoo was returning 55% of requested symbols. Failed
symbols are now retried against Nasdaq's historical endpoint, keeping cached
Yahoo fundamentals and swapping in the fresh price series. The series is
unadjusted, so anything with a split-sized jump in it gets rejected rather than
silently mis-scored. (Stooq was the first candidate — it now sits behind a
JavaScript bot check, so it's out.)

First production run with it, after the circuit breaker had already fired:

```
DURDURULDU: Yahoo hiz siniri uyguluyor (25 ardisik ret).
yedek kaynak deneniyor: 272 sembol
yedek kaynaktan kurtarilan: 193 hisse
721/800 basarili (%90; Yahoo %66 + yedek 193)
```

55% → 90%. The two rates are reported separately on purpose: blending them into
one number would hide how the primary source is actually doing. The retry list
is capped at 300 symbols — burning the second source to save the first is a bad
trade.

**Fetch prioritisation.** The daily budget of ~800 symbols used to go strictly
to the stalest names. It now goes to the watchlist first, then the previous
scan's top decile, then staleness. A stale price on the 1,900th-ranked stock
changes nobody's decision; a stale price in the top 20 makes the list wrong.

**Market regime.** A one-line banner: index versus its 50 and 200-day averages,
breadth (share of scanned stocks above their own 50-day), and realised
volatility against its own year. It changes no score. It exists because a
momentum-weighted ranking behaves very differently in a downtrend, and because
IC can only be measured per regime if the regime was recorded on the day.

**Targets as ranges.** The methods behind each target were already computed and
then collapsed into one number. The spread between them *is* the uncertainty, so
it's now shown: low/high, spread percentage, and a confidence label. Anything
above 25% spread is labelled "read the range, not the target".

**Transaction costs.** Spread estimated from dollar volume and tick size,
market impact from assumed participation. A $3 stock trading $1M/day comes out
around 4.8% round trip — which turns an "8% target" into 3%. Targets now carry a
net figure alongside the gross one, and flag when cost eats a third of the move.

**Earnings countdown.** Already existed as a −0.20σ penalty; the date itself was
invisible. It's the single biggest driver of a 21-day return, so it's on the
card now.

**Model ensemble.** Champion-take-all threw away the other two models. The
ensemble averages the members' *within-day percentile ranks* — raw predictions
can't be blended when one model outputs on a different scale. Members are
aligned by (date, ticker) because the sequence model drops rows that lack
enough history. On synthetic data with two independent weak predictors, the
blend beats both members.

**Multiple horizons.** `--horizons 5,21,63` trains each separately. Three times
the evidence from the same data, and the shape across horizons says what kind of
signal it is: strong short and weak long is a momentum effect; the reverse is a
valuation effect.

**Raw series features.** The models could only see the 28 human-designed scores,
which caps them at human hypotheses. 20 raw quantities now go into the feature
store — returns at seven lags, realised volatility, volume ratios, distance from
moving averages in ATR units, drawdown, skew. They are not parameters: no
weight, no dashboard presence, training only.

**Notifications.** A broken stop can't wait to be noticed. Alerts go to a file
and a local desktop notification; Telegram too, if `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID` happen to be set in the environment. Repeats are suppressed
for 5 days.

**Encrypted backup.** `data/feature_store` cannot be regenerated — Yahoo won't
serve yesterday's fundamentals. OneDrive syncs it, but sync isn't backup: a
silent corruption propagates to every copy instantly. `python run.py backup`
freezes it into an AES-256-GCM archive, same crypto as the dashboard. Restore
writes to a separate directory by default, because an untested backup isn't a
backup. The daily job takes one automatically if a week has passed.

**Health panel.** What `scripts/durum.py` prints, on the site itself: last run,
fetch rate, fallback usage, universe coverage, archive size, delisting counts.

---

## Two failures worth documenting

**One character stopped the automation for three days.** A U+FFFD left behind by
an encoding repair sat inside a cosmetic `print` in the learning step. The scan
completed fine — 2,390 stocks scored, dashboard written — then that line raised
`UnicodeEncodeError` on a cp1254 console, the process exited 1, the day never
got marked, and all eight triggers re-ran the same scan. Two fixes: console
output is pinned to UTF-8, and daily stages are isolated so a non-essential
step can't void a successful one. `tests/test_kodlama.py` fails the build if
that character ever reappears.

**A network error emptied the site.** `api.nasdaq.com` was unreachable one
evening. The listings fetch was uncached and returned an empty list on failure,
so the universe collapsed to the 4 watchlist symbols — and the scan reported
success, wrote a 4-stock ranking over the dashboard, and published it. Three
separate holes, all closed: listings are cached and fall back to a stale copy;
a collapsed universe recovers from yesterday's recorded universe or aborts; and
a scan that scores far fewer stocks than the previous one refuses to overwrite
the dashboard at all.
