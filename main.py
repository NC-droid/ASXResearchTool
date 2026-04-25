#!/usr/bin/env python3
"""Entry point for the ASX 200 daily fundamental screener + paper portfolio + site.

Commands:
    python main.py                    # daily run (screen, update portfolio, rebuild site, email)
    python main.py --no-email         # skip email
    python main.py --dry-run 10       # quick smoke test on N random tickers
    python main.py --no-site          # skip site rebuild
    python main.py --refresh-universe # refresh ASX 200 list from Wikipedia

    python main.py portfolio init     # create a new $10K paper portfolio
    python main.py portfolio update   # mark-to-market + rebalance if due
    python main.py portfolio status   # print holdings and current value

    python main.py backtest           # run the historical backtest

    python main.py site               # just rebuild the static site

Outputs in ./reports/ and ./site/.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from asx_portfolio import fetcher, notifier, reporter, scorer, universe  # noqa: E402
from asx_portfolio import portfolio as portfolio_mod  # noqa: E402

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _load_dotenv_if_available() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    defaults = {
        "top_n": 5,
        "min_projected_return": 0.10,
        "diversify_by_sector": True,
        "history_period": "5y",
        "fetch_workers": 8,
        "use_alpha_vantage": False,
        "fetch_announcements": True,
        "weights": dict(scorer.DEFAULT_WEIGHTS),
        "reports_dir": str(Path(__file__).parent / "reports"),
        "site_dir": str(Path(__file__).parent / "site"),
        "portfolio_file": str(Path(__file__).parent / "portfolio_state.json"),
        "portfolio_initial_capital": 10_000.0,
        "portfolio_rebalance_days": 30,
        "portfolio_transaction_cost": 0.0,
    }
    if not cfg_path.exists():
        return defaults
    try:
        import yaml
        with cfg_path.open() as fh:
            user_cfg = yaml.safe_load(fh) or {}
        defaults.update({k: v for k, v in user_cfg.items() if v is not None})
        if "weights" in user_cfg and user_cfg["weights"]:
            defaults["weights"] = {**scorer.DEFAULT_WEIGHTS, **user_cfg["weights"]}
        return defaults
    except ImportError:
        return defaults


def _common_log_setup(verbose: bool) -> logging.Logger:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=LOG_FORMAT)
    return logging.getLogger("asx_portfolio")


def _run_screen(cfg: dict, log: logging.Logger, dry_run: int = 0, refresh_universe: bool = False):
    uni_df = universe.load_asx200(refresh=refresh_universe)
    log.info("Universe size: %d", len(uni_df))
    if dry_run:
        uni_df = uni_df.sample(n=min(dry_run, len(uni_df)), random_state=42).reset_index(drop=True)
        log.info("Dry run: limited to %d tickers", len(uni_df))
    name_lookup = dict(zip(uni_df["ticker"], uni_df.get("name", uni_df["ticker"])))
    sector_lookup = dict(zip(uni_df["ticker"], uni_df.get("sector", ["" for _ in range(len(uni_df))])))
    fetched = fetcher.fetch_universe(uni_df["ticker"].tolist(), period=cfg["history_period"], workers=cfg["fetch_workers"])
    if not fetched:
        raise RuntimeError("No usable data fetched")
    metrics = [scorer.metrics_from_stockdata(sd, name=name_lookup.get(t, ""), sector=sector_lookup.get(t, "")) for t, sd in fetched.items()]
    scored = scorer.score_universe(metrics, weights=cfg["weights"])
    picks = scorer.select_top_picks(scored, n=cfg["top_n"], min_projected_return=cfg["min_projected_return"], diversify_by_sector=cfg["diversify_by_sector"])
    rationales = [scorer.build_rationale(row) for _, row in picks.iterrows()]
    for r in rationales:
        log.info("PICK: %s", r)
    return picks, scored, rationales, fetched, uni_df


def _write_screen_reports(picks, scored, rationales, cfg):
    csv_path, xlsx_path = reporter.make_default_paths(Path(cfg["reports_dir"]))
    reporter.write_csv(picks, csv_path)
    reporter.write_excel(picks=picks, full_screen=scored, rationales=rationales, out_path=xlsx_path)
    return csv_path, xlsx_path


def _maybe_build_site(cfg: dict, log: logging.Logger, skip: bool):
    if skip:
        return None
    try:
        from asx_portfolio import site_builder
        index = site_builder.build_site(
            site_dir=Path(cfg["site_dir"]),
            reports_dir=Path(cfg["reports_dir"]),
            state_path=Path(cfg["portfolio_file"]),
            min_projected_return=cfg["min_projected_return"],
        )
        log.info("Site rebuilt: open %s in your browser", index)
        return index
    except Exception as exc:  # noqa: BLE001
        log.warning("Site rebuild failed: %s", exc)
        return None


# Subcommand handlers ---------------------------------------------------------


def _cmd_portfolio_init(args, cfg, log):
    state_path = Path(args.state_file or cfg["portfolio_file"])
    if state_path.exists() and not args.force:
        log.error("Portfolio state already exists at %s. Use --force to overwrite.", state_path)
        return 2
    picks, scored, rationales, fetched, _ = _run_screen(cfg, log, dry_run=args.dry_run)
    picks_in_fetched = picks[picks["ticker"].isin(fetched.keys())].reset_index(drop=True)
    state = portfolio_mod.initialize(
        picks_df=picks_in_fetched, fetched=fetched, state_path=state_path,
        initial_capital=args.capital or cfg["portfolio_initial_capital"],
        rebalance_interval_days=cfg["portfolio_rebalance_days"],
        transaction_cost=cfg["portfolio_transaction_cost"],
    )
    log.info("Portfolio initialized at %s with $%.2f", state_path, state.initial_capital)
    csv_path, xlsx_path = _write_screen_reports(picks, scored, rationales, cfg)
    mtm = portfolio_mod.mark_to_market(state, fetched)
    state.save(state_path)
    port_xlsx = Path(cfg["reports_dir"]) / f"portfolio_{datetime.now():%Y-%m-%d}.xlsx"
    portfolio_mod.write_portfolio_report(state, mtm["per_position"], port_xlsx)
    _maybe_build_site(cfg, log, skip=False)
    return 0


def _cmd_portfolio_update(args, cfg, log):
    state_path = Path(args.state_file or cfg["portfolio_file"])
    state = portfolio_mod.PortfolioState.load(state_path)
    if state is None:
        log.error("No portfolio at %s. Run 'main.py portfolio init' first.", state_path)
        return 2
    picks, scored, rationales, fetched, _ = _run_screen(cfg, log, dry_run=args.dry_run)
    for t in [p.ticker for p in state.positions if p.ticker not in fetched]:
        sd = fetcher.fetch_yahoo(t, period="3mo")
        if sd.ok:
            fetched[t] = sd
    if portfolio_mod.should_rebalance(state):
        log.info("Rebalancing...")
        picks_in_fetched = picks[picks["ticker"].isin(fetched.keys())].reset_index(drop=True)
        portfolio_mod.rebalance(state, picks_in_fetched, fetched)
    mtm = portfolio_mod.mark_to_market(state, fetched)
    state.save(state_path)
    csv_path, xlsx_path = _write_screen_reports(picks, scored, rationales, cfg)
    port_xlsx = Path(cfg["reports_dir"]) / f"portfolio_{datetime.now():%Y-%m-%d}.xlsx"
    portfolio_mod.write_portfolio_report(state, mtm["per_position"], port_xlsx)
    _maybe_build_site(cfg, log, skip=False)
    if not args.no_email:
        subject = f"ASX 200 Picks — {datetime.now().strftime('%a %d %b %Y')}"
        notifier.send_daily_email(subject=subject, picks=picks, rationales=rationales, attachments=[csv_path, xlsx_path, port_xlsx])
    return 0


def _cmd_portfolio_status(args, cfg, log):
    state_path = Path(args.state_file or cfg["portfolio_file"])
    state = portfolio_mod.PortfolioState.load(state_path)
    if state is None:
        log.error("No portfolio at %s", state_path)
        return 2
    latest = state.value_history[-1] if state.value_history else {}
    print(f"\nPortfolio status — {state_path}")
    print(f"  Created:         {state.created_at}")
    print(f"  Last rebalance:  {state.last_rebalance}")
    print(f"  Initial capital: ${state.initial_capital:,.2f}")
    print(f"  Current value:   ${latest.get('total_value', 0):,.2f}")
    print(f"  Total return:    {latest.get('total_return_pct', 0)*100:.2f}%")
    print(f"  Positions ({len(state.positions)}):")
    for p in state.positions:
        print(f"    {p.ticker:8s} {p.shares:>10.4f} sh  avg ${p.avg_cost:7.2f}  ({p.sector or 'n/a'})")
    return 0


def _cmd_backtest(args, cfg, log):
    import pandas as pd
    from asx_portfolio import backtest
    uni_df = universe.load_asx200()
    start = (pd.Timestamp.today() - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    closes, benchmark = backtest.download_price_history(uni_df["ticker"].tolist(), start=start)
    if closes.empty or benchmark.empty:
        log.error("Could not download historical prices.")
        return 2
    result = backtest.run_backtest(closes, benchmark, top_n=cfg["top_n"], rebalance_interval_days=args.rebalance_days, initial_capital=args.capital)
    out = Path(cfg["reports_dir"]) / f"backtest_{datetime.now():%Y-%m-%d}.xlsx"
    backtest.write_backtest_report(result, out)
    print("\nBacktest results:")
    for k, v in result.metrics.items():
        if isinstance(v, float) and any(s in k for s in ("cagr", "return", "vol", "dd", "rate")):
            print(f"  {k:30s} {v*100:>7.2f}%")
        else:
            print(f"  {k:30s} {v:>7.2f}")
    print(f"\nWrote: {out}")
    _maybe_build_site(cfg, log, skip=False)
    return 0


def _cmd_site(args, cfg, log):
    index = _maybe_build_site(cfg, log, skip=False)
    if index is None:
        return 2
    print(f"\nSite rebuilt. Open: file://{index}")
    return 0


def _cmd_daily(args, cfg, log):
    picks, scored, rationales, fetched, _ = _run_screen(cfg, log, dry_run=args.dry_run, refresh_universe=args.refresh_universe)
    if cfg.get("fetch_announcements"):
        sel = {row["ticker"]: fetched[row["ticker"]] for _, row in picks.iterrows() if row["ticker"] in fetched}
        fetcher.enrich_with_announcements(sel)
    csv_path, xlsx_path = _write_screen_reports(picks, scored, rationales, cfg)
    state_path = Path(cfg["portfolio_file"])
    port_xlsx = None
    if state_path.exists():
        state = portfolio_mod.PortfolioState.load(state_path)
        for t in [p.ticker for p in state.positions if p.ticker not in fetched]:
            sd = fetcher.fetch_yahoo(t, period="3mo")
            if sd.ok:
                fetched[t] = sd
        if portfolio_mod.should_rebalance(state):
            picks_in_fetched = picks[picks["ticker"].isin(fetched.keys())].reset_index(drop=True)
            portfolio_mod.rebalance(state, picks_in_fetched, fetched)
        mtm = portfolio_mod.mark_to_market(state, fetched)
        state.save(state_path)
        port_xlsx = Path(cfg["reports_dir"]) / f"portfolio_{datetime.now():%Y-%m-%d}.xlsx"
        portfolio_mod.write_portfolio_report(state, mtm["per_position"], port_xlsx)
    _maybe_build_site(cfg, log, skip=args.no_site)
    if not args.no_email:
        subject = f"ASX 200 Picks — {datetime.now().strftime('%a %d %b %Y')}"
        atts = [csv_path, xlsx_path] + ([port_xlsx] if port_xlsx else [])
        notifier.send_daily_email(subject=subject, picks=picks, rationales=rationales, attachments=atts)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASX 200 daily screener + paper portfolio + site")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--no-email", action="store_true")
    p.add_argument("--no-site", action="store_true")
    p.add_argument("--refresh-universe", action="store_true")
    p.add_argument("--dry-run", type=int, default=0, metavar="N")
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--min-return", type=float, default=None)
    sub = p.add_subparsers(dest="cmd")
    portfolio = sub.add_parser("portfolio")
    psub = portfolio.add_subparsers(dest="portfolio_cmd")
    pi = psub.add_parser("init")
    pi.add_argument("--capital", type=float, default=None)
    pi.add_argument("--state-file", type=str, default=None)
    pi.add_argument("--force", action="store_true")
    pi.add_argument("--dry-run", type=int, default=0)
    pu = psub.add_parser("update")
    pu.add_argument("--state-file", type=str, default=None)
    pu.add_argument("--no-email", action="store_true")
    pu.add_argument("--dry-run", type=int, default=0)
    ps = psub.add_parser("status")
    ps.add_argument("--state-file", type=str, default=None)
    bt = sub.add_parser("backtest")
    bt.add_argument("--years", type=int, default=10)
    bt.add_argument("--rebalance-days", type=int, default=30)
    bt.add_argument("--capital", type=float, default=10_000.0)
    sub.add_parser("site")
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    log = _common_log_setup(args.verbose)
    _load_dotenv_if_available()
    cfg = _load_config()
    if args.top_n is not None:
        cfg["top_n"] = args.top_n
    if args.min_return is not None:
        cfg["min_projected_return"] = args.min_return
    if args.cmd == "portfolio":
        if args.portfolio_cmd == "init":
            return _cmd_portfolio_init(args, cfg, log)
        if args.portfolio_cmd == "update":
            return _cmd_portfolio_update(args, cfg, log)
        if args.portfolio_cmd == "status":
            return _cmd_portfolio_status(args, cfg, log)
        parser.parse_args(["portfolio", "--help"])
        return 1
    if args.cmd == "backtest":
        return _cmd_backtest(args, cfg, log)
    if args.cmd == "site":
        return _cmd_site(args, cfg, log)
    return _cmd_daily(args, cfg, log)


if __name__ == "__main__":
    raise SystemExit(main())
