# ASX Research Tool — Project Context

> Load this at the start of any new session to get up to speed immediately.

---

## What this project is

A **personal ASX investment research tool** built and deployed to Azure. It is **not financial advice** — personal use only. It has two codebases that work together:

1. **Original Python CLI** (`main.py` / `asx_portfolio/`) — daily screener, paper portfolio tracker, backtester, static site generator. Runs locally.
2. **FastAPI web app** (deployed to Azure) — interactive 3-page SPA built on top of the same scoring engine.

---

## Live site

**URL:** `https://etfpicker-ckddffard0fqa9dz.australiaeast-01.azurewebsites.net`

**GitHub repo:** `https://github.com/NC-droid/EFTPicker` (branch: `main`)

**Deploy:** GitHub Actions → Azure App Service (East Australia). Every push to `main` auto-deploys via `.github/workflows/main_etfpicker.yml`.

---

## Architecture

### Backend — FastAPI (`asx_portfolio/web_app.py`)

| Route | Description |
|---|---|
| `GET /` | Landing page |
| `GET /etfs` | ETF Screener SPA |
| `GET /stocks` | ASX 200 Stock Screener SPA |
| `GET /combined` | Combined Portfolio Simulator SPA |
| `POST /api/screen` | ETF scoring (84 ETFs, Yahoo Finance) |
| `POST /api/screen-stocks` | Stock scoring (172 ASX 200 stocks) |
| `POST /api/simulate` | Monte Carlo simulation (500 paths) |
| `GET /api/categories` | ETF category list |
| `GET /api/sectors` | Stock sector list |
| `GET /api/health` | Cache status |
| `GET /api/refresh` | Force cache refresh |

### Key Python modules

| File | Purpose |
|---|---|
| `web_app.py` | FastAPI app, all routes, security middleware |
| `etf_fetcher.py` | Fetches 84 ETFs from Yahoo Finance, 1-hour cache |
| `etf_scorer.py` | 6-pillar scoring for ETFs |
| `stock_fetcher.py` | Fetches ASX 200 stocks from Yahoo Finance, 4-hour cache |
| `stock_scorer.py` | 6-pillar scoring for stocks |
| `requirements.txt` | Includes: fastapi, uvicorn, yfinance, pandas, numpy, pydantic, slowapi, jinja2 |

### Frontend — 4 HTML pages (no framework, pure CSS + Vanilla JS)

All templates in `asx_portfolio/templates/`:

- `landing.html` — Hero page, 3-step journey cards, stats
- `etf_app.html` — ETF screener SPA (~1200 lines)
- `stocks_app.html` — ASX 200 stocks screener SPA (~1050 lines)
- `combined_app.html` — Combined portfolio + Monte Carlo simulator SPA (~1010 lines)

**Design system:** Dark premium theme (`#0c0f1a` bg), CSS variables (`--gold`, `--teal`, `--violet`), DM Serif Display headings, Plus Jakarta Sans body, Geist Mono numbers. No Tailwind — deliberately removed.

**User flow:** ETF Screener → save picks to `localStorage` → ASX Stocks → save picks → Combined Simulator loads both sets, blends allocations, runs Monte Carlo.

---

## 6-Pillar Scoring Model

Used identically for ETFs and stocks. Weights shift by risk tolerance and horizon.

| Pillar | Default Weight | Metrics |
|---|---|---|
| Value | 22% | Earnings yield, FCF yield, P/B |
| Growth | 22% | Revenue growth, earnings growth, forward EPS |
| Quality | 22% | ROE, ROA, gross margin, debt/equity |
| Income | 10% | Dividend yield + payout ratio check |
| Momentum | 14% | 6M + 12M return, 1M extreme penalty |
| Risk | 10% | Volatility, max drawdown, beta |

Scoring: cross-sectional percentile ranking (0–100) per pillar, then weighted sum = composite score.

---

## Monte Carlo Simulator

- 500 log-normal paths using Box-Muller transform via NumPy
- Monthly returns: `weighted_return/12 + weighted_vol/√12 × Z`
- Returns P10, P50, P90 equity curves
- **Known limitation:** assets modelled independently (no correlation) — noted in UI
- Input bounds validated: return -200%–500%, volatility 0–300%, amount $100–$100M, years 1–50

---

## Security posture (last audited April 2026)

**Confirmed active:**
- HSTS: `max-age=31536000; includeSubDomains`
- X-Content-Type-Options: `nosniff`
- X-Frame-Options: `DENY`
- Content-Security-Policy (includes `frame-ancestors none`)
- Referrer-Policy: `no-referrer`
- Permissions-Policy set
- CORS locked to Azure domain + localhost only
- `/docs`, `/redoc`, `/openapi.json` all disabled
- Rate limiting via slowapi: 30 req/min on screen endpoints, 10 req/min on simulate
- XSS: all `innerHTML` interpolations sanitised with `sanitise()` helper
- Input validation: Pydantic + explicit bounds checks on simulate
- Error messages: generic to client, exceptions logged server-side
- HTTP → HTTPS enforced (301)

**Known remaining item (LOW risk):**
- `server: uvicorn` header — Azure's load balancer re-injects it after our middleware. Cannot be removed in Python. Mitigate with Azure Front Door WAF if needed.

---

## UX features (as of April 2026)

- **Process journey banner** on all 3 app pages: shows 3-step flow (ETF → Stocks → Combined), highlights active step, ✓ on completed steps
- **Full-screen loading overlay** with spinner, rotating stage text, progress bar
- **Scoring transparency:** expandable "How is this score calculated?" panel on every pick card showing per-pillar bars
- **Data freshness badge** in nav: "Live data" → "8m ago" → "2h ago" with amber indicator when stale
- **Smart empty states** with quick-fix suggestion chips when filters return 0 results
- **Input microcopy** on all sliders explaining what each metric means
- **Reset all filters** button on both screener pages
- **Toast notifications** for all save/load/simulate actions
- **Save to Combined** strip on ETF and Stocks pages — writes picks to `localStorage`
- **Formula disclosure** on dividend planner and Combined simulator
- **Correlation disclaimer** on Monte Carlo output

---

## Code stats (April 2026)

- **Total:** ~7,200 lines across 30 source files, ~310 KB
- **Python:** ~3,700 lines (16 files)
- **HTML/JS:** ~3,300 lines (9 files)
- **Config/CI:** ~200 lines (5 files)

---

## How to push changes

All file pushes go via GitHub REST API using a PAT. **The PAT used in this project has been rotated — ask the user for a new one before making any pushes.**

```python
import json, base64, urllib.request

TOKEN = "ghp_..."  # ask user
REPO = "NC-droid/EFTPicker"

# Get SHA of file
url = f"https://api.github.com/repos/{REPO}/contents/{path}?ref=main"
req = urllib.request.Request(url, headers={"Authorization": f"token {TOKEN}"})
with urllib.request.urlopen(req) as r:
    d = json.load(r)
sha = d["sha"]

# Push updated file
content_b64 = base64.b64encode(new_content.encode()).decode("ascii")
payload = {"message": "commit message", "content": content_b64, "sha": sha, "branch": "main"}
req2 = urllib.request.Request(
    f"https://api.github.com/repos/{REPO}/contents/{path}",
    data=json.dumps(payload).encode(),
    headers={"Authorization": f"token {TOKEN}", "Content-Type": "application/json"},
    method="PUT"
)
with urllib.request.urlopen(req2) as r:
    result = json.load(r)
```

Deploy triggers automatically on push. Monitor at `https://github.com/NC-droid/EFTPicker/actions`.

---

## Known gotchas

1. **Starlette `MutableHeaders` has no `.pop()`** — use `del response.headers[key]` or override with empty string
2. **slowapi requires `app.state.limiter = limiter`** — without this line it silently does nothing
3. **Yahoo Finance `dividendYield`** can return as decimal (0.075) OR percent (7.5) — normalise: if > 1.0, divide by 100
4. **Azure cold starts** cause intermittent 503s on the first request after idle — self-resolving, not a bug
5. **Tailwind CDN was intentionally removed** — all styles are pure CSS variables. Do not re-introduce Tailwind
6. **`data.candidate_count` is not in scope inside `renderPicks()`** — use module-level `_etfCandidateCount` / `_stockCandidateCount` variables set in `renderResults()`
7. **`force_refresh=true`** bypasses the Yahoo Finance cache — can be expensive. Rate limit is 30/min but consider a per-user cooldown if traffic grows
8. **Azure deploy sometimes fails silently** — always check Actions tab. If stocks/combined serve old code while ETF is updated, a "no-op" commit to stocks_app.html forces a fresh deploy

---

## Next potential improvements (backlog)

1. ETF comparison feature — side-by-side table for 2–3 ETFs (blueprint suggestion, genuinely additive)
2. Export picks to CSV — simple download button on screener results
3. SRI hashes on Plotly CDN `<script>` tag (cdn.plot.ly was returning 503 when last attempted)
4. Asset correlation in Monte Carlo — currently models holdings independently
5. Keyboard/accessibility pass on filter chips and toggle buttons
