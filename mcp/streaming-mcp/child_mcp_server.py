"""Child FastMCP server exposing streaming account tools."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, TypedDict

from fastmcp import Context, FastMCP

from demo_data import BLOCKED_EMAIL_DOMAINS, DEMO_PROFILES

mcp = FastMCP("child-mcp")
HTTP_HOST = os.getenv("CHILD_MCP_SERVER_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("CHILD_MCP_SERVER_PORT", "8001"))
HTTP_PATH = os.getenv("CHILD_MCP_SERVER_PATH", "/mcp")
STEP_DELAY_SECONDS = float(os.getenv("CHILD_MCP_STEP_DELAY_SECONDS", "5"))
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")


class ChildToolTraceEvent(TypedDict):
    """One progress update, retained in the final child-tool result."""

    sequence: int
    event: str
    message: str


class ChildToolResult(TypedDict):
    """JSON result returned by each child MCP tool."""

    status: str
    tool: str
    execution_trace: list[ChildToolTraceEvent]
    result: dict[str, Any]


def _build_result(
    *,
    tool: str,
    execution_trace: list[ChildToolTraceEvent],
    result: dict[str, Any],
) -> ChildToolResult:
    return {
        "status": "completed",
        "tool": tool,
        "execution_trace": execution_trace,
        "result": result,
    }


async def _report_step(
    ctx: Context,
    execution_trace: list[ChildToolTraceEvent],
    *,
    event: str,
    message: str,
) -> None:
    """Simulate asynchronous work, stream progress, and retain the step for Cursor."""
    await asyncio.sleep(STEP_DELAY_SECONDS)
    sequence = len(execution_trace) + 1
    execution_trace.append({"sequence": sequence, "event": event, "message": message})
    await ctx.report_progress(progress=sequence, total=3, message=message)


@mcp.tool()
async def validate_email_address(ctx: Context, email_address: str) -> ChildToolResult:
    """Validate an email address, streaming stages and returning a structured JSON result."""
    normalized_email = email_address.strip().lower()
    execution_trace: list[ChildToolTraceEvent] = []
    has_valid_format = bool(EMAIL_PATTERN.fullmatch(normalized_email))
    await _report_step(
        ctx,
        execution_trace,
        event="format_checked",
        message=(
            f"Email validation for {normalized_email}: format is "
            f"{'valid' if has_valid_format else 'invalid'} (1 of 3)"
        ),
    )
    if not has_valid_format:
        await _report_step(
            ctx,
            execution_trace,
            event="domain_check_skipped",
            message="Email validation: domain check skipped because the format is invalid (2 of 3)",
        )
        await _report_step(
            ctx,
            execution_trace,
            event="delivery_policy_check_skipped",
            message="Email validation: delivery-policy check skipped (3 of 3)",
        )
        return _build_result(
            tool="validate_email_address",
            execution_trace=execution_trace,
            result={
                "email_address": normalized_email,
                "is_valid": False,
                "format_valid": False,
                "domain_allowed": None,
                "delivery_policy_passed": None,
            },
        )

    domain = normalized_email.rsplit("@", maxsplit=1)[1]
    domain_is_allowed = domain not in BLOCKED_EMAIL_DOMAINS
    await _report_step(
        ctx,
        execution_trace,
        event="domain_checked",
        message=(
            f"Email validation for {normalized_email}: domain {domain} is "
            f"{'allowed' if domain_is_allowed else 'blocked'} (2 of 3)"
        ),
    )

    local_part = normalized_email.split("@", maxsplit=1)[0]
    passes_delivery_policy = domain_is_allowed and not local_part.startswith(("noreply", "no-reply"))
    await _report_step(
        ctx,
        execution_trace,
        event="delivery_policy_checked",
        message=(
            f"Email validation for {normalized_email}: delivery policy "
            f"{'passed' if passes_delivery_policy else 'failed'} (3 of 3)"
        ),
    )
    return _build_result(
        tool="validate_email_address",
        execution_trace=execution_trace,
        result={
            "email_address": normalized_email,
            "is_valid": passes_delivery_policy,
            "format_valid": True,
            "domain_allowed": domain_is_allowed,
            "delivery_policy_passed": passes_delivery_policy,
        },
    )


@mcp.tool()
async def give_profile_information(ctx: Context, email_address: str) -> ChildToolResult:
    """Retrieve a demo profile, streaming stages and returning a structured JSON result."""
    normalized_email = email_address.strip().lower()
    execution_trace: list[ChildToolTraceEvent] = []
    profile = DEMO_PROFILES.get(normalized_email)
    await _report_step(
        ctx,
        execution_trace,
        event="account_record_checked",
        message=(
            f"Profile lookup for {normalized_email}: account record "
            f"{'found' if profile else 'not found'} (1 of 3)"
        ),
    )
    if profile is None:
        await _report_step(
            ctx,
            execution_trace,
            event="preferences_lookup_skipped",
            message="Profile lookup: preference lookup skipped because no account exists (2 of 3)",
        )
        await _report_step(
            ctx,
            execution_trace,
            event="not_found_result_prepared",
            message="Profile lookup: response prepared with not-found result (3 of 3)",
        )
        return _build_result(
            tool="give_profile_information",
            execution_trace=execution_trace,
            result={"email_address": normalized_email, "profile_found": False, "profile": None},
        )

    preferences = profile["preferences"]
    enabled_preferences = [name for name, enabled in preferences.items() if enabled]
    await _report_step(
        ctx,
        execution_trace,
        event="preferences_loaded",
        message=f"Profile lookup for {normalized_email}: loaded {len(enabled_preferences)} enabled preference(s) (2 of 3)",
    )

    profile_response = {
        "email_address": normalized_email,
        "display_name": profile["display_name"],
        "plan": profile["plan"],
        "enabled_preferences": enabled_preferences,
    }
    await _report_step(
        ctx,
        execution_trace,
        event="profile_response_prepared",
        message=f"Profile lookup for {normalized_email}: safe profile response prepared (3 of 3)",
    )
    return _build_result(
        tool="give_profile_information",
        execution_trace=execution_trace,
        result={"email_address": normalized_email, "profile_found": True, "profile": profile_response},
    )


def main() -> None:
    mcp.run(transport="streamable-http", host=HTTP_HOST, port=HTTP_PORT, path=HTTP_PATH)


if __name__ == "__main__":
    main()
