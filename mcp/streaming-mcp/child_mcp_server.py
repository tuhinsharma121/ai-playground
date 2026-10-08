"""Child FastMCP server exposing streaming account tools."""
from __future__ import annotations
import asyncio
import os
import re
from fastmcp import Context, FastMCP

from demo_data import BLOCKED_EMAIL_DOMAINS, DEMO_PROFILES

mcp = FastMCP("child-mcp")
HTTP_HOST = os.getenv("CHILD_MCP_SERVER_HOST", "127.0.0.1")
HTTP_PORT = int(os.getenv("CHILD_MCP_SERVER_PORT", "8001"))
HTTP_PATH = os.getenv("CHILD_MCP_SERVER_PATH", "/mcp")
STEP_DELAY_SECONDS = float(os.getenv("CHILD_MCP_STEP_DELAY_SECONDS", "5"))
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")
async def _report_step(ctx: Context, step: int, message: str) -> None:
    """Simulate asynchronous work and emit a streamed result for one step."""
    await asyncio.sleep(STEP_DELAY_SECONDS)
    await ctx.report_progress(progress=step, total=3, message=message)

@mcp.tool()
async def validate_email_address(ctx: Context, email_address: str) -> str:
    """Validate an email address and stream each validation stage."""
    normalized_email = email_address.strip().lower()
    has_valid_format = bool(EMAIL_PATTERN.fullmatch(normalized_email))
    await _report_step(
        ctx,
        1,
        f"Email validation for {normalized_email}: format is {'valid' if has_valid_format else 'invalid'} (1 of 3)",
    )
    if not has_valid_format:
        await _report_step(ctx, 2, "Email validation: domain check skipped because the format is invalid (2 of 3)")
        await _report_step(ctx, 3, "Email validation: delivery-policy check skipped (3 of 3)")
        return f"Email address {normalized_email} is invalid."

    domain = normalized_email.rsplit("@", maxsplit=1)[1]
    domain_is_allowed = domain not in BLOCKED_EMAIL_DOMAINS
    await _report_step(
        ctx,
        2,
        f"Email validation for {normalized_email}: domain {domain} is {'allowed' if domain_is_allowed else 'blocked'} (2 of 3)",
    )

    local_part = normalized_email.split("@", maxsplit=1)[0]
    passes_delivery_policy = domain_is_allowed and not local_part.startswith(("noreply", "no-reply"))
    await _report_step(
        ctx,
        3,
        f"Email validation for {normalized_email}: delivery policy {'passed' if passes_delivery_policy else 'failed'} (3 of 3)",
    )
    return (
        f"Email address {normalized_email} passed validation."
        if passes_delivery_policy
        else f"Email address {normalized_email} did not pass validation."
    )


@mcp.tool()
async def give_profile_information(ctx: Context, email_address: str) -> str:
    """Retrieve a demo user profile and stream each lookup stage."""
    normalized_email = email_address.strip().lower()
    profile = DEMO_PROFILES.get(normalized_email)
    await _report_step(
        ctx,
        1,
        f"Profile lookup for {normalized_email}: account record {'found' if profile else 'not found'} (1 of 3)",
    )
    if profile is None:
        await _report_step(ctx, 2, "Profile lookup: preference lookup skipped because no account exists (2 of 3)")
        await _report_step(ctx, 3, "Profile lookup: response prepared with not-found result (3 of 3)")
        return f"No demo profile was found for {normalized_email}."

    preferences = profile["preferences"]
    enabled_preferences = [name for name, enabled in preferences.items() if enabled]
    await _report_step(
        ctx,
        2,
        f"Profile lookup for {normalized_email}: loaded {len(enabled_preferences)} enabled preference(s) (2 of 3)",
    )

    profile_response = {
        "email": normalized_email,
        "display_name": profile["display_name"],
        "plan": profile["plan"],
        "enabled_preferences": enabled_preferences,
    }
    await _report_step(ctx, 3, f"Profile lookup for {normalized_email}: safe profile response prepared (3 of 3)")
    return (
        "Demo profile: "
        f"email={profile_response['email']}, "
        f"display_name={profile_response['display_name']}, "
        f"plan={profile_response['plan']}, "
        f"enabled_preferences={', '.join(profile_response['enabled_preferences']) or 'none'}."
    )

def main() -> None:
    mcp.run(transport="streamable-http", host=HTTP_HOST, port=HTTP_PORT, path=HTTP_PATH)

if __name__ == "__main__":
    main()
