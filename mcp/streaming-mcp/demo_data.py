"""Deterministic data and policies used by the child MCP demo tools."""

from __future__ import annotations

BLOCKED_EMAIL_DOMAINS = frozenset({"invalid.example", "blocked.example"})
DEMO_PROFILES = {
    "tuhin@gmail.com": {
        "display_name": "Tuhin",
        "plan": "developer",
        "preferences": {"product_updates": True, "weekly_digest": False},
    }
}
