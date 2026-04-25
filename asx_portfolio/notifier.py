"""Email notifier — sends the daily picks via SMTP."""

from __future__ import annotations

import logging
import mimetypes
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


def _smtp_configured() -> bool:
    needed = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "SMTP_TO"]
    return all(os.environ.get(k) for k in needed)


def _build_html_body(picks: pd.DataFrame, rationales: list[str]) -> str:
    rows = []
    for i, (_, p) in enumerate(picks.iterrows()):
        proj = p.get("projected_return", float("nan"))
        proj_pct = f"{proj * 100:.1f}%" if not pd.isna(proj) else "n/a"
        score = p.get("composite_score", float("nan"))
        rows.append(
            "<tr>"
            f"<td>{i + 1}</td><td><b>{p.get('ticker','')}</b></td>"
            f"<td>{p.get('name','')}</td><td>{p.get('sector','')}</td>"
            f"<td>{score:.1f if not pd.isna(score) else 'n/a'}</td>"
            f"<td>{proj_pct}</td>"
            f"<td>{rationales[i] if i < len(rationales) else ''}</td></tr>"
        )
    table = (
        "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;font-family:Arial;font-size:13px'>"
        "<thead style='background:#f4f4f4'><tr><th>#</th><th>Ticker</th><th>Name</th><th>Sector</th><th>Score</th><th>Proj. Return</th><th>Rationale</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )
    return (
        "<html><body><h2 style='font-family:Arial'>ASX 200 — Today's Top Picks</h2>"
        "<p style='font-family:Arial;font-size:13px'>Daily fundamental screen. Picks filtered for ≥10% projected long-term return. <i>Not financial advice.</i></p>"
        + table + "</body></html>"
    )


def _build_text_body(picks: pd.DataFrame, rationales: list[str]) -> str:
    lines = ["ASX 200 — Today's Top Picks", ""]
    for i, (_, p) in enumerate(picks.iterrows()):
        proj = p.get("projected_return", float("nan"))
        proj_pct = f"{proj*100:.1f}%" if not pd.isna(proj) else "n/a"
        lines.append(f"{i+1}. {p.get('ticker','')} — {p.get('name','')} ({p.get('sector','')})")
        lines.append(f"   Projected return: {proj_pct}")
        if i < len(rationales):
            lines.append(f"   Rationale: {rationales[i]}")
        lines.append("")
    lines.append("Not financial advice.")
    return "\n".join(lines)


def send_daily_email(subject: str, picks: pd.DataFrame, rationales: list[str], attachments: Iterable[Path] = ()) -> bool:
    if not _smtp_configured():
        log.info("SMTP not configured — skipping email.")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = os.environ["SMTP_TO"]
    msg.set_content(_build_text_body(picks, rationales))
    msg.add_alternative(_build_html_body(picks, rationales), subtype="html")
    for path in attachments:
        path = Path(path)
        if not path.exists():
            continue
        ctype, encoding = mimetypes.guess_type(path.name)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with path.open("rb") as fh:
            msg.add_attachment(fh.read(), maintype=maintype, subtype=subtype, filename=path.name)
    host = os.environ["SMTP_HOST"]
    port = int(os.environ["SMTP_PORT"])
    user = os.environ["SMTP_USER"]
    pw = os.environ["SMTP_PASSWORD"]
    use_ssl = os.environ.get("SMTP_USE_SSL", "0") == "1" or port == 465
    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as s:
                s.login(user, pw)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(user, pw)
                s.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Email send failed: %s", exc)
        return False
