"""
web_app_v2_routes.py — DEAD CODE — safe to delete.

All routes from this file have been merged into web_app.py.
This file contained duplicate Pydantic models (StockScreenRequest,
SimulateRequest) and @app.get / @app.post decorators that never
registered because `app` was never imported here — so none of the
routes were ever active.

The duplicate code has been removed; the canonical implementations
live in web_app.py.
"""
