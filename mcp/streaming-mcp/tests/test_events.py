"""Unit tests for the parent-to-child relay envelope."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

import child_mcp_server
from events import decode_relay_event, encode_relay_event
from parent_mcp_server import format_tool_result, run_downstream_react_agent


class RelayEventTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        encoded = encode_relay_event(
            kind="child_mcp_progress",
            text="format is valid",
            tool="validate_email_address",
            call_id="call-123",
        )
        self.assertEqual(
            decode_relay_event(encoded),
            {
                "kind": "child_mcp_progress",
                "text": "format is valid",
                "tool": "validate_email_address",
                "call_id": "call-123",
            },
        )

    def test_unstructured_message_is_not_a_relay_event(self) -> None:
        self.assertIsNone(decode_relay_event("ordinary MCP progress"))

    def test_parent_tool_result_replays_execution_trace(self) -> None:
        result = format_tool_result(
            [
                {
                    "sequence": 1,
                    "event": "child_agent_started",
                    "message": "Server-side child agent started.",
                },
                {
                    "sequence": 2,
                    "event": "child_tool_started",
                    "message": "Child agent called validate_email_address.",
                    "tool": "validate_email_address",
                },
            ],
            "tuhin@gmail.com passed validation.",
        )
        json.dumps(result)
        self.assertEqual(
            result,
            {
                "status": "completed",
                "execution_trace": [
                    {
                        "sequence": 1,
                        "event": "child_agent_started",
                        "message": "Server-side child agent started.",
                    },
                    {
                        "sequence": 2,
                        "event": "child_tool_started",
                        "message": "Child agent called validate_email_address.",
                        "tool": "validate_email_address",
                    },
                ],
                "final_result": "tuhin@gmail.com passed validation.",
            },
        )

    def test_parent_tool_replays_relayed_progress_in_final_result(self) -> None:
        class FakeContext:
            def __init__(self) -> None:
                self.progress_messages: list[str] = []

            async def report_progress(self, *, progress: int, total: None, message: str) -> None:
                self.progress_messages.append(message)

        async def fake_child_agent(query: str):
            self.assertEqual(query, "Validate tuhin@gmail.com")
            yield {
                "type": "on_tool_start",
                "tool": "validate_email_address",
                "call_id": "call-1",
            }
            yield {
                "type": "on_tool_stream",
                "tool": "validate_email_address",
                "call_id": "call-1",
                "chunk": "Email validation for tuhin@gmail.com: format is valid (1 of 3)",
            }
            yield {
                "type": "on_agent_thinking",
                "text": "I should retrieve the profile next.",
            }
            yield {
                "type": "on_tool_end",
                "tool": "validate_email_address",
                "call_id": "call-1",
            }
            yield {"type": "on_chat_model_stream", "text": "tuhin@gmail.com passed validation."}

        context = FakeContext()
        with patch("parent_mcp_server.run_child_agent", fake_child_agent):
            result = asyncio.run(run_downstream_react_agent(context, "Validate tuhin@gmail.com"))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["final_result"], "tuhin@gmail.com passed validation.")
        self.assertEqual(
            result["execution_trace"],
            [
                {
                    "sequence": 1,
                    "event": "child_agent_started",
                    "message": "Server-side child agent started.",
                },
                {
                    "sequence": 2,
                    "event": "child_tool_started",
                    "message": "Child agent called validate_email_address.",
                    "tool": "validate_email_address",
                    "call_id": "call-1",
                },
                {
                    "sequence": 3,
                    "event": "child_tool_progress",
                    "message": "Email validation for tuhin@gmail.com: format is valid (1 of 3)",
                    "tool": "validate_email_address",
                    "call_id": "call-1",
                },
                {
                    "sequence": 4,
                    "event": "child_agent_reasoning",
                    "message": "Child agent evaluated the available information and selected its next action.",
                },
                {
                    "sequence": 5,
                    "event": "child_tool_finished",
                    "message": "Child agent received the validate_email_address result.",
                    "tool": "validate_email_address",
                    "call_id": "call-1",
                },
                {
                    "sequence": 6,
                    "event": "child_agent_completed",
                    "message": "Server-side child agent completed.",
                },
            ],
        )
        self.assertEqual(len(context.progress_messages), 7)
        self.assertEqual(
            decode_relay_event(context.progress_messages[0]),
            {"kind": "child_started", "text": "Server-side child ReAct agent started"},
        )

    def test_child_tools_return_json_with_their_progress_trace(self) -> None:
        class FakeContext:
            def __init__(self) -> None:
                self.progress_messages: list[str] = []

            async def report_progress(self, *, progress: int, total: int, message: str) -> None:
                self.progress_messages.append(message)

        async def run_tools() -> tuple[dict[str, object], dict[str, object], FakeContext, FakeContext]:
            validation_context = FakeContext()
            profile_context = FakeContext()
            with patch.object(child_mcp_server, "STEP_DELAY_SECONDS", 0):
                validation = await child_mcp_server.validate_email_address(validation_context, "tuhin@gmail.com")
                profile = await child_mcp_server.give_profile_information(profile_context, "tuhin@gmail.com")
            return validation, profile, validation_context, profile_context

        validation, profile, validation_context, profile_context = asyncio.run(run_tools())

        json.dumps(validation)
        json.dumps(profile)
        self.assertEqual(validation["status"], "completed")
        self.assertEqual(validation["tool"], "validate_email_address")
        self.assertEqual(validation["result"]["is_valid"], True)
        self.assertEqual(
            [step["event"] for step in validation["execution_trace"]],
            ["format_checked", "domain_checked", "delivery_policy_checked"],
        )
        self.assertEqual(profile["status"], "completed")
        self.assertEqual(profile["tool"], "give_profile_information")
        self.assertEqual(profile["result"]["profile"]["display_name"], "Tuhin")
        self.assertEqual(len(validation_context.progress_messages), 3)
        self.assertEqual(len(profile_context.progress_messages), 3)


if __name__ == "__main__":
    unittest.main()
