# ASX 200 Daily Picks + Paper Portfolio + Site

A Python project that:

1. **Screens the S&P/ASX 200 every day** using a six-pillar fundamental model and produces a top-5 list of long-term candidates (≥10% projected annualised return).
2. **Tracks a $10,000 paper portfolio** that holds the picks equal-weighted and rebalances monthly so you can see real performance vs. expectations.
3. **Backtests the price-based half of the model** over 10 years to show how the timing/selection signal would have done historically.
4. **Generates a private static website** (HTML pages you open locally) showing today's picks, the live portfolio, the archive of every prior day's picks, and the backtest summary — all auto-refreshed by the daily run.

> Personal research only. **Not financial advice.** Past performance doesn't predict future returns.

## What the website looks like

The `site/` folder contains a 4-page static site (Tailwind + Plotly via CDN):

- **`index.html`** — Today's Picks. Big cards per pick with rationale + pillar bars; full ASX 200 sortable table below.
- **`portfolio.html`** — Live $10K paper portfolio with KPI cards, equity-curve chart, holdings table, rebalance history.
- **`archive.html`** — Every historical day's picks, newest first.
- **`backtest.html`** — Most recent backtest with growth-of-$10K chart and metric cards.

Open `site/index.html` directly in your browser, or serve the folder:
```
cd site
python -m http.server 8000
# then visit http://localhost:8000
```

## Project layout

```
asx_portfolio/
├── README.md
├── requirements.txt
├── config.yaml          # tunable parameters & pillar weights
├── .env.example         # SMTP / Alpha Vantage template
├── main.py              # CLI entry point
├── portfolio_state.json # paper-portfolio state (created on first run)
├── reports/             # daily CSV + Excel outputs (created on first run)
├── site/                # static HTML site (regenerated each run)
└── asx_portfolio/
    ├── universe.py      # ASX 200 ticker list (static + Wikipedia refresh)
    ├── fetcher.py       # Yahoo / ASX / Alpha Vantage data fetchers
    ├── scorer.py        # 6-pillar fundamental scoring + projected return
    ├── reporter.py      # CSV + Excel report generation
    ├── notifier.py      # SMTP email
    ├── portfolio.py     # paper portfolio state + rebalance + report
    ├── backtest.py      # historical price-based backtester
    ├── site_builder.py  # static site generator (Jinja2)
    ├── templates/       # base.html + page templates
    └── data/asx200_tickers.csv
```

## Setup

```bash
cd "Investment Portfolio/asx_portfolio"
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

(Optional) configure email:
```bash
cp .env.example .env
# then edit .env to add SMTP_* values (Gmail App Passwords work)
```

## Usage

```bash
# Quick smoke run on 10 random tickers, no email, no site
python main.py --dry-run 10 --no-email --no-site

# Full daily run: screen + portfolio update + site rebuild + email
python main.py

# Initialize the $10K paper portfolio for the first time
python main.py portfolio init

# Update the portfolio without re-screening (rare)
python main.py portfolio update

# See current portfolio status
python main.py portfolio status

# Run the historical backtest
python main.py backtest --years 10 --rebalance-days 30

# Just rebuild the static site from existing state/reports
python main.py site
```

After any run that produced reports, the site is regenerated to reflect them. Bookmark `site/index.html` and just reload it.

## How to schedule daily runs

### macOS / Linux (cron)
`crontab -e` then add (runs weekdays at 18:30 local):
```
30 18 * * 1-5  cd /path/to/asx_portfolio && /path/to/python main.py >> reports/run.log 2>&1
```

### Windows (Task Scheduler)
Trigger: Daily 18:30 → Action: `python.exe main.py` → Start in: project folder.

## Scoring methodology (in plain English)

| Pillar    | Inputs                                                       | Default weight |
|-----------|--------------------------------------------------------------|----------------|
| Value     | Earnings yield, FCF yield, P/B (lower better)                | 22% |
| Growth    | Revenue growth, earnings growth, forward EPS growth          | 22% |
| Quality   | ROE, ROA, gross margin, debt/equity (lower better)           | 22% |
| Income    | Dividend yield (with payout-ratio sanity penalty)            | 10% |
| Momentum  | 6m + 12m return; 1m extreme penalty                          | 14% |
| Risk      | Volatility, max drawdown, distance of beta from 1.0          | 10% |

Every metric is cross-sectionally percentile-ranked across the universe (0–100), then combined with pillar weights into a composite score.

**Projected long-term annualised return** is a transparent Gordon-style decomposition:
```
projected_return ≈ dividend_yield + sustainable_growth + valuation_reversion
```
This is a diagnostic, not a forecast. Tweak weights in `config.yaml`.

## Backtest caveat (important)

The backtester only tests the **price-based pillars** (momentum + risk). Yahoo Finance only exposes today's snapshot fundamentals, so testing the value/growth/quality/income pillars historically would inject look-ahead bias. To rigorously test the full model you'd need a paid point-in-time fundamentals provider (Sharadar, Capital IQ, Refinitiv).

## Disclaimer

Personal research tool only. Not financial product advice. The author is not responsible for any investment decisions made on the basis of this output.
