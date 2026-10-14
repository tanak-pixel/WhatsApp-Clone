#!/usr/bin/env python3
"""Automated test suite for the WhatsApp Style-Cloning Bot repository.

Run with either:
    python3 test_whatsapp_clone_bot.py
    pytest
"""

from __future__ import annotations

import io
import os
import pathlib
import tempfile
import unittest
import zipfile

import whatsapp_clone_bot as bot


class WhatsAppCloneBotTests(unittest.TestCase):
  """Tests iOS parsing, zip handling, style extraction, and API key security."""

  def test_ios_parser_and_unicode_markers(self) -> None:
    raw = (
        "\u200e[10/13/26, 7:42:19\u202fPM] Alex: hey are you free tonight?\n"
        "[10/13/26, 7:42:50\u202fPM] Jordan: yesss im down\n"
        "wait where are we meeting <This message was edited>\n"
        "\u200e[10/13/26, 7:43:10\u202fPM] Alex: \u200eimage omitted\n"
    )
    msgs = bot.parse_whatsapp_ios_export(raw)
    self.assertEqual(len(msgs), 2)
    self.assertEqual(msgs[0].sender, "Alex")
    self.assertEqual(msgs[1].sender, "Jordan")
    self.assertIn("wait where are we meeting", msgs[1].text)
    self.assertNotIn("image omitted", [m.text for m in msgs])
    self.assertNotIn("<This message was edited>", msgs[1].text)

  def test_ios_zip_archive_extraction(self) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
      zf.writestr("_chat.txt", bot.SAMPLE_IOS_WHATSAPP_EXPORT)
    extracted = bot.extract_text_from_upload(
        buf.getvalue(), "WhatsApp Chat - Maya.zip"
    )
    msgs = bot.parse_whatsapp_ios_export(extracted)
    self.assertGreaterEqual(len(msgs), 20)

  def test_style_fingerprint_and_prompt_generation(self) -> None:
    msgs = bot.parse_whatsapp_ios_export(bot.SAMPLE_IOS_WHATSAPP_EXPORT)
    turns = bot.group_into_turns(msgs, max_gap_seconds=180)
    pairs = bot.build_turn_pairs(turns, target_sender="Jordan")
    profile = bot.extract_style_profile(msgs, turns, target_sender="Jordan")

    self.assertEqual(profile.sender_name, "Jordan")
    self.assertGreater(profile.lowercase_start_pct, 50.0)
    self.assertGreater(profile.no_period_end_pct, 80.0)
    self.assertGreater(profile.multi_bubble_turn_pct, 40.0)

    selected = bot.select_few_shot_examples(
        pairs, incoming_query="wya right now??", k=6
    )
    self.assertEqual(len(selected), 6)
    prompt = bot.build_system_instruction(profile, selected)
    self.assertIn("QUANTITATIVE STYLE FINGERPRINT FOR JORDAN", prompt)

  def test_dotenv_loading_and_secret_masking(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      env_file = pathlib.Path(tmpdir) / ".env"
      env_file.write_text(
          'GEMINI_API_KEY="AIzaSyTestSecretKey987654321"\n', encoding="utf-8"
      )
      old_val = os.environ.pop("GEMINI_API_KEY", None)
      try:
        key, source = bot.resolve_api_key(dotenv_path=env_file)
        self.assertEqual(key, "AIzaSyTestSecretKey987654321")
        self.assertEqual(source, "Environment / .env")
        masked = bot.mask_api_key(key)
        self.assertEqual(masked, "AIza••••••4321")
        self.assertNotIn("TestSecretKey", masked)
      finally:
        if old_val is not None:
          os.environ["GEMINI_API_KEY"] = old_val
        else:
          os.environ.pop("GEMINI_API_KEY", None)

  def test_no_hardcoded_api_key_in_source(self) -> None:
    source_path = pathlib.Path(bot.__file__).resolve()
    code = source_path.read_text(encoding="utf-8")
    # Ensure no real AIza... Google API keys exist in the application code
    import re

    real_key_matches = re.findall(r"AIza[0-9A-Za-z\-_]{35}", code)
    self.assertEqual(
        real_key_matches,
        [],
        f"Found hardcoded API key pattern in source code: {real_key_matches}",
    )


if __name__ == "__main__":
  unittest.main(verbosity=2)
