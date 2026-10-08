"""Unit tests for the parent-to-child relay envelope."""

from __future__ import annotations

import unittest

from events import decode_relay_event, encode_relay_event


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


if __name__ == "__main__":
    unittest.main()
