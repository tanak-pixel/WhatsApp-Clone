#!/usr/bin/env python3
"""WhatsApp Style-Cloning Bot (iOS Export + Gemini API + Streamlit Local Web UI).

Parses an exported iOS WhatsApp chat (.txt or .zip containing _chat.txt),
analyzes your personal texting fingerprint (casing, punctuation, multi-bubble
habits, emojis, and slang), dynamically selects relevant few-shot conversation
turns, and runs a local Streamlit web chat interface powered by the official
Google GenAI SDK (`google-genai`).

Security Note:
    This app never hardcodes or logs your API key. It loads `GEMINI_API_KEY`
    automatically from a local `.env` file (ignored by Git), environment
    variables, or `.streamlit/secrets.toml`.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import io
import json
import os
import pathlib
import re
from typing import Any
import zipfile

# ---------------------------------------------------------------------------
# 0. Secure Environment & API Key Loader (Zero Key Exposure)
# ---------------------------------------------------------------------------

PLACEHOLDER_KEYS = {
    "",
    "your-api-key",
    "your-api-key-here",
    "your_gemini_api_key_here",
    "paste-your-gemini-api-key-here",
    "insert_api_key_here",
}


def load_dotenv_safe(dotenv_path: pathlib.Path | None = None) -> bool:
  """Loads key-value pairs from `.env` into `os.environ` without exposing secrets.

  Uses `python-dotenv` if installed, and falls back to a built-in parser so the
  repository works out of the box in any Python environment.
  """
  target_path = dotenv_path or (pathlib.Path(__file__).resolve().parent / ".env")
  if not target_path.is_file():
    return False

  try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=target_path, override=False)
    return True
  except ImportError:
    pass

  # Built-in zero-dependency .env parser fallback
  for raw_line in target_path.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    if line.startswith("export "):
      line = line[len("export ") :].strip()
    key, _, val = line.partition("=")
    key = key.strip()
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
      val = val[1:-1]
    if key and key not in os.environ:
      os.environ[key] = val
  return True


def resolve_api_key(dotenv_path: pathlib.Path | None = None) -> tuple[str, str]:
  """Resolves `GEMINI_API_KEY` from `.env`, environment, or Streamlit secrets.

  Returns:
      (api_key, source_description)
  """
  load_dotenv_safe(dotenv_path=dotenv_path)

  env_val = os.environ.get("GEMINI_API_KEY", "").strip()
  if env_val and env_val.lower() not in PLACEHOLDER_KEYS:
    return env_val, "Environment / .env"

  # Check Streamlit secrets safely if running inside Streamlit
  try:
    import streamlit as st

    secret_val = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    if secret_val and secret_val.lower() not in PLACEHOLDER_KEYS:
      return secret_val, ".streamlit/secrets.toml"
  except Exception:  # pylint: disable=broad-except
    pass

  return "", "Not configured"


def mask_api_key(api_key: str) -> str:
  """Masks an API key for safe status display (never reveals the full secret)."""
  cleaned = api_key.strip()
  if not cleaned or len(cleaned) < 8:
    return "••••••••"
  return f"{cleaned[:4]}••••••{cleaned[-4:]}"


# ---------------------------------------------------------------------------
# 1. iOS WhatsApp Export Parser & Turn Grouping
# ---------------------------------------------------------------------------

# iOS WhatsApp exports wrap timestamps in brackets, e.g.:
# [10/13/26, 7:42:19 PM] Alex: hey are you free tonight?
# [13/10/2026, 19:42:19] Alex: hey are you free tonight?
# Note: iOS frequently inserts U+200E (LRM) at the start of lines or before
# media notices, and U+202F (narrow no-break space) before AM/PM.
IOS_LINE_REGEX = re.compile(
    r"^[\u200e\u200f]*\["
    r"(?P<date>\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}),\s*"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202f\xa0]*[APap][Mm])?)"
    r"\]\s*"
    r"(?:(?P<sender>[^:]+?):\s*)?"
    r"(?P<text>.*)$"
)

# Fallback for Android exports just in case a mixed export is loaded:
# 10/13/26, 7:42 PM - Alex: hey are you free tonight?
ANDROID_LINE_REGEX = re.compile(
    r"^[\u200e\u200f]*"
    r"(?P<date>\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}),\s*"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202f\xa0]*[APap][Mm])?)\s*-\s*"
    r"(?:(?P<sender>[^:]+?):\s*)?"
    r"(?P<text>.*)$"
)

# System notices & media placeholders emitted by WhatsApp iOS exports
SYSTEM_OR_MEDIA_PATTERNS = [
    r"^image omitted$",
    r"^video omitted$",
    r"^audio omitted$",
    r"^sticker omitted$",
    r"^gif omitted$",
    r"^document omitted$",
    r"^contact card omitted$",
    r"^<media omitted>$",
    r"^<attached:.*>$",
    r"^messages and calls are end-to-end encrypted.*",
    r"^this message was deleted\.?$",
    r"^you deleted this message\.?$",
    r"^missed voice call\.?$",
    r"^missed video call\.?$",
    r"^voice call\b.*",
    r"^video call\b.*",
    r"^location:\s*https?://.*",
]
SYSTEM_REGEX = re.compile("|".join(SYSTEM_OR_MEDIA_PATTERNS), re.IGNORECASE)

# Inline edit marker in newer iOS exports
EDITED_TAG_REGEX = re.compile(
    r"[\u200e\u200f]*<This message was edited>", re.IGNORECASE
)

# Unicode Emoji extraction pattern
EMOJI_REGEX = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # Emoticons
    "\U0001f300-\U0001f5ff"  # Misc Symbols and Pictographs
    "\U0001f680-\U0001f6ff"  # Transport and Map
    "\U0001f1e0-\U0001f1ff"  # Flags
    "\U00002600-\U000026ff"  # Misc symbols
    "\U00002700-\U000027bf"  # Dingbats
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
    "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
    "]"
)

# Common English stopwords to exclude when identifying personal signature words
STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "in",
    "is",
    "it",
    "for",
    "that",
    "you",
    "on",
    "with",
    "as",
    "are",
    "this",
    "by",
    "be",
    "at",
    "or",
    "from",
    "an",
    "was",
    "we",
    "can",
    "have",
    "has",
    "had",
    "but",
    "not",
    "what",
    "all",
    "were",
    "when",
    "there",
    "been",
    "if",
    "more",
    "will",
    "would",
    "about",
    "which",
    "their",
    "out",
    "up",
    "one",
    "who",
    "get",
    "just",
    "me",
    "my",
    "so",
    "do",
    "no",
}

# Known colloquial slang & texting shorthand tokens to track even if short
COMMON_SLANG_TOKENS = {
    "lol",
    "lmao",
    "lmfao",
    "rofl",
    "haha",
    "hahaha",
    "hehe",
    "ikr",
    "idk",
    "idc",
    "tbh",
    "ngl",
    "fr",
    "frfr",
    "bet",
    "lowkey",
    "highkey",
    "deadass",
    "bro",
    "bruh",
    "dude",
    "man",
    "girl",
    "bestie",
    "fam",
    "yo",
    "rn",
    "wya",
    "omw",
    "wyd",
    "hmu",
    "imo",
    "imho",
    "nvm",
    "brb",
    "gtg",
    "ttyl",
    "omg",
    "omfg",
    "wtf",
    "wth",
    "smh",
    "fyi",
    "afaik",
    "irl",
    "bc",
    "cuz",
    "tho",
    "thru",
    "prolly",
    "gonna",
    "wanna",
    "gotta",
    "kinda",
    "sorta",
    "lemme",
    "gimme",
    "yeah",
    "yea",
    "yep",
    "yup",
    "nah",
    "nope",
    "aight",
    "fs",
    "ong",
    "finna",
    "wild",
    "valid",
    "fire",
    "lit",
    "cooked",
    "down",
    "keen",
    "kk",
    "ok",
    "okay",
    "ty",
    "tysm",
    "thx",
    "np",
    "pls",
    "plz",
    "u",
    "ur",
    "r",
}


@dataclass
class ChatMessage:
  """Represents a single parsed WhatsApp message."""

  timestamp: datetime | None
  sender: str
  text: str


@dataclass
class ConversationTurn:
  """Represents one speaker's contiguous turn (one or more rapid bubbles)."""

  sender: str
  bubbles: list[str] = field(default_factory=list)
  start_time: datetime | None = None
  end_time: datetime | None = None

  @property
  def combined_text(self) -> str:
    """Returns bubbles joined by newline to preserve multi-bubble rhythm."""
    return "\n".join(self.bubbles)


@dataclass
class TurnPair:
  """A [Context -> Target Reply] pair used for few-shot style prompting."""

  incoming_sender: str
  incoming_text: str
  reply_bubbles: list[str]
  timestamp: datetime | None = None

  @property
  def reply_text(self) -> str:
    return "\n".join(self.reply_bubbles)


@dataclass
class StyleProfile:
  """Quantitative and qualitative summary of the target user's texting style."""

  sender_name: str
  total_messages: int
  total_turns: int
  avg_words_per_bubble: float
  avg_bubbles_per_turn: float
  multi_bubble_turn_pct: float
  lowercase_start_pct: float
  all_caps_msg_pct: float
  no_period_end_pct: float
  exclamation_pct: float
  question_pct: float
  ellipsis_pct: float
  emoji_per_msg: float
  top_emojis: list[tuple[str, int]]
  top_slang: list[tuple[str, int]]
  top_words: list[tuple[str, int]]
  top_openers: list[tuple[str, int]]


def _parse_datetime(date_str: str, time_str: str) -> datetime | None:
  """Best-effort datetime parser across US/EU/ISO iOS WhatsApp formats."""
  cleaned_time = re.sub(r"[\u202f\xa0]+", " ", time_str).strip().upper()
  cleaned_date = date_str.strip().replace(".", "/").replace("-", "/")
  combined = f"{cleaned_date} {cleaned_time}"

  formats = [
      "%m/%d/%y %I:%M:%S %p",
      "%m/%d/%Y %I:%M:%S %p",
      "%d/%m/%y %I:%M:%S %p",
      "%d/%m/%Y %I:%M:%S %p",
      "%m/%d/%y %I:%M %p",
      "%m/%d/%Y %I:%M %p",
      "%d/%m/%y %I:%M %p",
      "%d/%m/%Y %I:%M %p",
      "%m/%d/%y %H:%M:%S",
      "%m/%d/%Y %H:%M:%S",
      "%d/%m/%y %H:%M:%S",
      "%d/%m/%Y %H:%M:%S",
      "%m/%d/%y %H:%M",
      "%m/%d/%Y %H:%M",
      "%d/%m/%y %H:%M",
      "%d/%m/%Y %H:%M",
      "%Y/%m/%d %H:%M:%S",
      "%Y/%m/%d %I:%M:%S %p",
  ]
  for fmt in formats:
    try:
      return datetime.strptime(combined, fmt)
    except ValueError:
      continue
  return None


def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
  """Extracts UTF-8 chat text from either a raw .txt file or an iOS .zip export."""
  if filename.lower().endswith(".zip") or zipfile.is_zipfile(
      io.BytesIO(file_bytes)
  ):
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
      txt_candidates = [
          name
          for name in zf.namelist()
          if name.lower().endswith(".txt") and not name.startswith("__MACOSX")
      ]
      if not txt_candidates:
        raise ValueError(
            "No .txt chat file found inside the uploaded .zip archive."
        )
      preferred = next(
          (n for n in txt_candidates if "_chat.txt" in n.lower()),
          txt_candidates[0],
      )
      raw_bytes = zf.read(preferred)
  else:
    raw_bytes = file_bytes

  for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
    try:
      return raw_bytes.decode(encoding)
    except UnicodeDecodeError:
      continue
  return raw_bytes.decode("utf-8", errors="replace")


def parse_whatsapp_ios_export(raw_text: str) -> list[ChatMessage]:
  """Parses iOS (and Android fallback) WhatsApp chat exports into ChatMessages."""
  messages: list[ChatMessage] = []
  current_msg: ChatMessage | None = None

  for raw_line in raw_text.splitlines():
    line = raw_line.rstrip("\r\n")
    match = IOS_LINE_REGEX.match(line) or ANDROID_LINE_REGEX.match(line)

    if match:
      if current_msg is not None:
        _append_if_valid(messages, current_msg)

      sender = match.group("sender")
      text = match.group("text") or ""
      ts = _parse_datetime(match.group("date"), match.group("time"))

      if sender is None:
        current_msg = None
        continue

      sender_clean = sender.strip("\u200e\u200f ").strip()
      text_clean = EDITED_TAG_REGEX.sub("", text).strip("\u200e\u200f ")
      current_msg = ChatMessage(
          timestamp=ts, sender=sender_clean, text=text_clean
      )
    else:
      if current_msg is not None:
        cleaned_cont = EDITED_TAG_REGEX.sub("", line).strip("\u200e\u200f")
        if cleaned_cont:
          current_msg.text += f"\n{cleaned_cont}"

  if current_msg is not None:
    _append_if_valid(messages, current_msg)

  return messages


def _append_if_valid(
    messages: list[ChatMessage], candidate: ChatMessage
) -> None:
  """Filters out system notices, media omissions, and empty bubbles."""
  cleaned = candidate.text.strip("\u200e\u200f ").strip()
  if not cleaned:
    return
  if SYSTEM_REGEX.match(cleaned):
    return
  candidate.text = cleaned
  messages.append(candidate)


def group_into_turns(
    messages: list[ChatMessage],
    max_gap_seconds: int = 300,
    max_bubbles_per_turn: int = 5,
) -> list[ConversationTurn]:
  """Groups consecutive messages from the same speaker into a single turn."""
  turns: list[ConversationTurn] = []
  for msg in messages:
    if not turns:
      turns.append(
          ConversationTurn(
              sender=msg.sender,
              bubbles=[msg.text],
              start_time=msg.timestamp,
              end_time=msg.timestamp,
          )
      )
      continue

    prev = turns[-1]
    same_sender = prev.sender == msg.sender
    within_gap = True
    if prev.end_time and msg.timestamp:
      gap = abs((msg.timestamp - prev.end_time).total_seconds())
      within_gap = gap <= max_gap_seconds

    if (
        same_sender
        and within_gap
        and len(prev.bubbles) < max_bubbles_per_turn
    ):
      prev.bubbles.append(msg.text)
      prev.end_time = msg.timestamp or prev.end_time
    else:
      turns.append(
          ConversationTurn(
              sender=msg.sender,
              bubbles=[msg.text],
              start_time=msg.timestamp,
              end_time=msg.timestamp,
          )
      )
  return turns


def build_turn_pairs(
    turns: list[ConversationTurn], target_sender: str
) -> list[TurnPair]:
  """Extracts (Incoming Friend Turn -> Target User Turn) pairs for few-shot learning."""
  pairs: list[TurnPair] = []
  for idx in range(1, len(turns)):
    curr_turn = turns[idx]
    prev_turn = turns[idx - 1]
    if curr_turn.sender == target_sender and prev_turn.sender != target_sender:
      if prev_turn.end_time and curr_turn.start_time:
        if (curr_turn.start_time - prev_turn.end_time) > timedelta(hours=12):
          continue
      pairs.append(
          TurnPair(
              incoming_sender=prev_turn.sender,
              incoming_text=prev_turn.combined_text,
              reply_bubbles=list(curr_turn.bubbles),
              timestamp=curr_turn.start_time,
          )
      )
  return pairs


# ---------------------------------------------------------------------------
# 2. Style Fingerprint Analyzer
# ---------------------------------------------------------------------------


def extract_style_profile(
    messages: list[ChatMessage],
    turns: list[ConversationTurn],
    target_sender: str,
) -> StyleProfile:
  """Computes a detailed linguistic and formatting profile for `target_sender`."""
  user_msgs = [m.text for m in messages if m.sender == target_sender and m.text]
  user_turns = [t for t in turns if t.sender == target_sender and t.bubbles]

  if not user_msgs:
    raise ValueError(f"No messages found for sender '{target_sender}'.")

  total_msgs = len(user_msgs)
  total_turns = max(len(user_turns), 1)

  word_counts = [len(m.split()) for m in user_msgs]
  avg_words = sum(word_counts) / total_msgs
  avg_bubbles = sum(len(t.bubbles) for t in user_turns) / total_turns
  multi_bubble_turns = sum(1 for t in user_turns if len(t.bubbles) > 1)
  multi_bubble_pct = (multi_bubble_turns / total_turns) * 100.0

  alpha_starts = [m[0] for m in user_msgs if m and m[0].isalpha()]
  lowercase_start_pct = (
      (sum(1 for ch in alpha_starts if ch.islower()) / len(alpha_starts) * 100.0)
      if alpha_starts
      else 0.0
  )

  alpha_msgs = [
      m for m in user_msgs if any(ch.isalpha() for ch in m) and len(m) >= 3
  ]
  all_caps_pct = (
      (sum(1 for m in alpha_msgs if m.isupper()) / len(alpha_msgs) * 100.0)
      if alpha_msgs
      else 0.0
  )

  no_period_end_pct = (
      sum(1 for m in user_msgs if not m.rstrip().endswith("."))
      / total_msgs
      * 100.0
  )
  exclamation_pct = sum(1 for m in user_msgs if "!" in m) / total_msgs * 100.0
  question_pct = sum(1 for m in user_msgs if "?" in m) / total_msgs * 100.0
  ellipsis_pct = (
      sum(1 for m in user_msgs if "..." in m or "…" in m) / total_msgs * 100.0
  )

  all_emojis: list[str] = []
  for m in user_msgs:
    all_emojis.extend(EMOJI_REGEX.findall(m))
  emoji_counter = Counter(all_emojis)
  emoji_per_msg = len(all_emojis) / total_msgs

  word_counter: Counter[str] = Counter()
  slang_counter: Counter[str] = Counter()
  opener_counter: Counter[str] = Counter()

  for m in user_msgs:
    tokens = re.findall(r"[a-zA-Z']+|[0-9]+", m.lower())
    if tokens:
      first_tok = tokens[0]
      if len(first_tok) >= 2:
        opener_counter[first_tok] += 1
    for tok in tokens:
      clean_tok = tok.strip("'")
      if not clean_tok:
        continue
      if clean_tok in COMMON_SLANG_TOKENS:
        slang_counter[clean_tok] += 1
      if len(clean_tok) >= 3 and clean_tok not in STOPWORDS:
        word_counter[clean_tok] += 1

  return StyleProfile(
      sender_name=target_sender,
      total_messages=total_msgs,
      total_turns=total_turns,
      avg_words_per_bubble=round(avg_words, 1),
      avg_bubbles_per_turn=round(avg_bubbles, 2),
      multi_bubble_turn_pct=round(multi_bubble_pct, 1),
      lowercase_start_pct=round(lowercase_start_pct, 1),
      all_caps_msg_pct=round(all_caps_pct, 1),
      no_period_end_pct=round(no_period_end_pct, 1),
      exclamation_pct=round(exclamation_pct, 1),
      question_pct=round(question_pct, 1),
      ellipsis_pct=round(ellipsis_pct, 1),
      emoji_per_msg=round(emoji_per_msg, 2),
      top_emojis=emoji_counter.most_common(10),
      top_slang=slang_counter.most_common(15),
      top_words=word_counter.most_common(15),
      top_openers=opener_counter.most_common(10),
  )


# ---------------------------------------------------------------------------
# 3. Dynamic Few-Shot Retriever & Prompt Engineering
# ---------------------------------------------------------------------------


def select_few_shot_examples(
    turn_pairs: list[TurnPair],
    incoming_query: str = "",
    k: int = 18,
) -> list[TurnPair]:
  """Selects a mix of contextually similar and representative few-shot turns."""
  if not turn_pairs:
    return []

  substantive_pairs = [
      p
      for p in turn_pairs
      if len(p.reply_text.strip()) >= 2 and len(p.incoming_text.strip()) >= 2
  ]
  pool = substantive_pairs if len(substantive_pairs) >= k else turn_pairs
  if len(pool) <= k:
    return pool

  query_tokens = (
      set(re.findall(r"[a-zA-Z']+", incoming_query.lower())) - STOPWORDS
  )
  query_is_question = "?" in incoming_query

  scored: list[tuple[float, int, TurnPair]] = []
  total_pool = len(pool)
  for idx, pair in enumerate(pool):
    inc_lower = pair.incoming_text.lower()
    inc_tokens = set(re.findall(r"[a-zA-Z']+", inc_lower)) - STOPWORDS

    overlap = len(query_tokens & inc_tokens)
    score = overlap * 3.0

    if query_is_question and "?" in pair.incoming_text:
      score += 1.5
    elif not query_is_question and "?" not in pair.incoming_text:
      score += 0.5

    reply_words = len(pair.reply_text.split())
    if 3 <= reply_words <= 35:
      score += 1.0

    recency_bonus = (idx / max(total_pool, 1)) * 0.8
    score += recency_bonus

    scored.append((score, idx, pair))

  context_quota = max(1, int(k * 0.6))
  top_context = [
      item[2]
      for item in sorted(scored, key=lambda x: x[0], reverse=True)[
          :context_quota
      ]
  ]
  selected_ids = {id(p) for p in top_context}

  remaining_quota = k - len(top_context)
  anchors: list[TurnPair] = []
  if remaining_quota > 0:
    candidates = [p for p in pool if id(p) not in selected_ids]
    step = max(1, len(candidates) // remaining_quota)
    for i in range(0, len(candidates), step):
      anchors.append(candidates[i])
      if len(anchors) >= remaining_quota:
        break

  combined = anchors + top_context
  return combined[:k]


def build_system_instruction(
    profile: StyleProfile,
    few_shot_examples: list[TurnPair],
    extra_persona_notes: str = "",
) -> str:
  """Constructs the system instruction encoding style metrics + few-shot turns."""
  casing_rule = (
      f"Start messages with a lowercase letter ~{profile.lowercase_start_pct:.0f}% of the time."
      if profile.lowercase_start_pct >= 45
      else (
          f"Capitalize sentence starts (~{100 - profile.lowercase_start_pct:.0f}% capitalized, "
          f"{profile.lowercase_start_pct:.0f}% lowercase)."
      )
  )

  period_rule = (
      f"Omit trailing periods at the end of messages ({profile.no_period_end_pct:.0f}% of your real texts do NOT end with a period)."
      if profile.no_period_end_pct >= 65
      else f"Leave off trailing periods about {profile.no_period_end_pct:.0f}% of the time."
  )

  bubble_rule = (
      f"You average {profile.avg_words_per_bubble:.1f} words per text bubble and {profile.avg_bubbles_per_turn:.2f} bubbles per turn "
      f"({profile.multi_bubble_turn_pct:.0f}% of your replies are multi-bubble double/triple texts). "
      "When sending multiple short bubbles in one reply, put each bubble on its own line separated by a newline (`\\n`)."
  )

  emojis_str = (
      ", ".join(f"{e} ({c}x)" for e, c in profile.top_emojis)
      if profile.top_emojis
      else "rarely uses emojis"
  )
  slang_str = (
      ", ".join(f"'{w}' ({c}x)" for w, c in profile.top_slang)
      if profile.top_slang
      else "natural casual phrasing"
  )
  openers_str = (
      ", ".join(f"'{w}'" for w, _ in profile.top_openers[:8])
      if profile.top_openers
      else "hey, yo, yeah"
  )
  vocab_str = (
      ", ".join(f"'{w}'" for w, _ in profile.top_words[:12])
      if profile.top_words
      else "casual everyday words"
  )

  examples_block_lines = []
  for idx, ex in enumerate(few_shot_examples, start=1):
    friend_formatted = ex.incoming_text.replace("\n", " | ")
    reply_formatted = "\n".join(ex.reply_bubbles)
    examples_block_lines.append(
        f"Example {idx}:\n"
        f"[Friend ({ex.incoming_sender})]: {friend_formatted}\n"
        f"[You ({profile.sender_name})]:\n{reply_formatted}"
    )
  examples_block = "\n\n".join(examples_block_lines)

  custom_notes_section = (
      f"\nADDITIONAL PERSONAL CONTEXT / RULES:\n{extra_persona_notes.strip()}\n"
      if extra_persona_notes.strip()
      else ""
  )

  return f"""You are roleplaying as {profile.sender_name} replying to a friend on WhatsApp.
Your ONLY goal is to reply in {profile.sender_name}'s exact texting voice, rhythm, slang, casing, punctuation, and emoji habits so a close friend cannot tell the difference.

STRICT ANTI-AI RULES:
1. NEVER sound like an AI assistant. Never be overly helpful, formal, or polite unless {profile.sender_name} texts that way.
2. NEVER use robotic transitions ("Certainly!", "Sounds like a plan!", "I'd love to help with that", "Let me know if you need anything else") or em-dashes (`—`).
3. Keep your reply length realistic for WhatsApp: aim for roughly {max(2, int(profile.avg_words_per_bubble * 0.6))} to {max(8, int(profile.avg_words_per_bubble * 1.8))} words per bubble.

QUANTITATIVE STYLE FINGERPRINT FOR {profile.sender_name.upper()}:
- Capitalization: {casing_rule}
- Trailing Periods: {period_rule}
- Bubble Rhythm & Double-Texting: {bubble_rule}
- Punctuation Flavor: Exclamation marks in {profile.exclamation_pct:.0f}% of texts, question marks in {profile.question_pct:.0f}%, ellipses (`...`) in {profile.ellipsis_pct:.0f}%.
- Emoji Habits ({profile.emoji_per_msg:.2f} emojis/msg): Only use emojis from or consistent with this favorite set: {emojis_str}. Do NOT invent random unrelated emojis.
- Signature Slang & Abbreviations: {slang_str}
- Common Message Openers: {openers_str}
- Frequent Personal Words: {vocab_str}
{custom_notes_section}
REAL WHATSAPP FEW-SHOT EXAMPLES FROM {profile.sender_name.upper()}'S CHAT HISTORY:
Study how {profile.sender_name} responds to different tones, questions, and casual banter below:

{examples_block}

Now reply to the friend's next message strictly as {profile.sender_name}. Output ONLY {profile.sender_name}'s message bubble(s) (separated by a newline if sending multiple short bubbles)—no sender prefix, no timestamps, and no quotation marks."""


# ---------------------------------------------------------------------------
# 4. Gemini API Client Wrapper (`google-genai` SDK)
# ---------------------------------------------------------------------------


def generate_clone_reply(
    api_key: str,
    model_name: str,
    profile: StyleProfile,
    all_turn_pairs: list[TurnPair],
    chat_history: list[dict[str, Any]],
    incoming_message: str,
    num_few_shot: int = 18,
    temperature: float = 0.85,
    extra_persona_notes: str = "",
) -> tuple[str, str, list[TurnPair]]:
  """Calls the Gemini API using the official `google-genai` SDK."""
  from google import genai
  from google.genai import types

  selected_examples = select_few_shot_examples(
      all_turn_pairs, incoming_query=incoming_message, k=num_few_shot
  )
  system_instruction = build_system_instruction(
      profile=profile,
      few_shot_examples=selected_examples,
      extra_persona_notes=extra_persona_notes,
  )

  client = genai.Client(api_key=api_key)

  contents: list[types.Content] = []
  for ex in selected_examples[-6:]:
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=ex.incoming_text)],
        )
    )
    contents.append(
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=ex.reply_text)],
        )
    )

  for turn in chat_history:
    role = "user" if turn["role"] == "user" else "model"
    contents.append(
        types.Content(
            role=role,
            parts=[types.Part.from_text(text=str(turn["content"]))],
        )
    )

  contents.append(
      types.Content(
          role="user",
          parts=[types.Part.from_text(text=incoming_message)],
      )
  )

  response = client.models.generate_content(
      model=model_name,
      contents=contents,
      config=types.GenerateContentConfig(
          system_instruction=system_instruction,
          temperature=temperature,
          top_p=0.95,
          max_output_tokens=300,
      ),
  )

  raw_reply = (response.text or "").strip()
  prefix_pattern = re.compile(
      rf"^(?:\[?(?:You|{re.escape(profile.sender_name)})\]?:\s*)+",
      re.IGNORECASE,
  )
  cleaned_lines = [
      prefix_pattern.sub("", line).strip()
      for line in raw_reply.splitlines()
      if line.strip()
  ]
  final_reply = "\n".join(cleaned_lines) if cleaned_lines else raw_reply
  return final_reply, system_instruction, selected_examples


# ---------------------------------------------------------------------------
# 5. Built-in Sample iOS WhatsApp Export (for instant testing)
# ---------------------------------------------------------------------------

SAMPLE_IOS_WHATSAPP_EXPORT = """\
[10/01/26, 6:14:02 PM] Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.
[10/01/26, 6:14:22 PM] Maya: yooo are you still down for tacos tonight??
[10/01/26, 6:15:05 PM] Jordan: yesss im starving ngl 😭
[10/01/26, 6:15:12 PM] Jordan: wait what time are we heading over
[10/01/26, 6:16:00 PM] Maya: thinking like 7:30? Leo said he needs to finish gym first
[10/01/26, 6:16:31 PM] Jordan: bet that works
[10/01/26, 6:16:39 PM] Jordan: tell him not to take 2 hours like last time lol
[10/01/26, 6:17:10 PM] Maya: Hahahaha I literally just texted him that
[10/01/26, 6:17:45 PM] Maya: image omitted
[10/01/26, 6:18:02 PM] Jordan: lmaooo he is so cooked 💀
[10/02/26, 11:20:15 AM] Maya: hey did you ever send those slides to profchen?
[10/02/26, 11:22:01 AM] Jordan: oh shoot
[10/02/26, 11:22:08 AM] Jordan: wait was that due at noon or midnight??
[10/02/26, 11:22:40 AM] Maya: midnight thank god, I panicked for a sec too
[10/02/26, 11:23:05 AM] Jordan: bro you scared me so bad 😭🙏
[10/02/26, 11:23:18 AM] Jordan: yeah im finishing the last 2 slides rn, ill drop the link in 10 mins
[10/03/26, 4:05:11 PM] Maya: wya!! we grabbed a table outside by the patio
[10/03/26, 4:05:44 PM] Jordan: omw!! literally parking rn
[10/03/26, 4:05:52 PM] Jordan: order me an iced matcha pls 🙏
[10/04/26, 9:12:30 PM] Maya: wait what did you think of the new episode??
[10/04/26, 9:13:10 PM] Jordan: ngl the ending was actually insane
[10/04/26, 9:13:22 PM] Jordan: like i did not see that twist coming at all 💀🔥
[10/04/26, 9:14:00 PM] Maya: right?? I was yelling at my tv
[10/04/26, 9:14:35 PM] Jordan: haha lowkey best season so far tho
[10/05/26, 1:40:12 PM] Maya: are you going to Sarah's birthday thing on saturday?
[10/05/26, 1:41:00 PM] Jordan: yeah im down!
[10/05/26, 1:41:09 PM] Jordan: wait are we getting a group gift or doing separate stuff
[10/05/26, 1:42:15 PM] Maya: we're all pitching in for a polaroid camera, it's like $20 each
[10/05/26, 1:42:40 PM] Jordan: bet Say less, ill venmo you rn 💸
[10/06/26, 8:10:04 PM] Maya: bro my laptop just blue screened right before saving my code
[10/06/26, 8:10:33 PM] Jordan: nooo way 💀😭
[10/06/26, 8:10:42 PM] Jordan: did git autosave any of it or is it completely gone
[10/06/26, 8:11:19 PM] Maya: vs code local history saved me thank god
[10/06/26, 8:11:45 PM] Jordan: clutchhh honestly vs code local history is a lifesaver fr
"""


# ---------------------------------------------------------------------------
# 6. Local Streamlit Web UI
# ---------------------------------------------------------------------------


def run_streamlit_ui() -> None:
  """Renders the local Streamlit Web UI for the WhatsApp style-cloning bot."""
  import streamlit as st

  st.set_page_config(
      page_title="WhatsApp Style-Cloning Bot",
      page_icon="💬",
      layout="wide",
  )

  st.title("💬 WhatsApp Style-Cloning Bot (iOS + Gemini API)")
  st.caption(
      "Upload your iOS WhatsApp chat export (`.txt` or `.zip`), inspect your "
      "extracted texting fingerprint, and chat with an AI clone that mirrors your "
      "casing, punctuation, double-texting rhythm, slang, and emojis."
  )

  if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
  if "raw_chat_text" not in st.session_state:
    st.session_state.raw_chat_text = ""
  if "source_label" not in st.session_state:
    st.session_state.source_label = ""

  # Resolve API key from .env / environment / secrets.toml without exposing it
  detected_key, key_source = resolve_api_key()

  with st.sidebar:
    st.header("1. Gemini API Security")
    if detected_key:
      st.success(
          f"🔒 API Key loaded from **{key_source}** (`{mask_api_key(detected_key)}`)"
      )
      override_key = st.text_input(
          "Override API Key (optional, session-only)",
          value="",
          type="password",
          help="Leave blank to use the key loaded from your local .env file.",
      )
      api_key = override_key.strip() if override_key.strip() else detected_key
    else:
      st.info(
          "Create a local `.env` file (`cp .env.example .env`) so your key is "
          "never hardcoded, or enter it below for this browser session."
      )
      api_key = st.text_input(
          "Gemini API Key (session-only)",
          value="",
          type="password",
          help="Never committed to Git. Stored only in browser session memory.",
      ).strip()

    model_name = st.selectbox(
        "Gemini Model",
        options=[
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
        ],
        index=0,
        help="gemini-2.5-flash is fast and great at following few-shot stylistic nuances.",
    )

    st.divider()
    st.header("2. Load iOS WhatsApp Export")
    st.markdown(
        "**On iPhone:** Open a WhatsApp chat → Tap Contact Name → **Export Chat** → "
        "**Without Media** → Save the `.zip` or `_chat.txt` file."
    )
    uploaded_file = st.file_uploader(
        "Upload `_chat.txt` or `WhatsApp Chat.zip`",
        type=["txt", "zip"],
    )
    if uploaded_file is not None:
      try:
        st.session_state.raw_chat_text = extract_text_from_upload(
            uploaded_file.getvalue(), uploaded_file.name
        )
        st.session_state.source_label = uploaded_file.name
      except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Could not read upload: {exc}")

    if st.button("Load Built-in Sample iOS Chat", use_container_width=True):
      st.session_state.raw_chat_text = SAMPLE_IOS_WHATSAPP_EXPORT
      st.session_state.source_label = "Sample iOS Chat (Jordan & Maya)"
      st.session_state.chat_history = []

  if not st.session_state.raw_chat_text:
    st.info(
        "👈 Upload your iOS WhatsApp `.txt` or `.zip` export in the sidebar (or click "
        "**Load Built-in Sample iOS Chat**) to get started."
    )
    return

  messages = parse_whatsapp_ios_export(st.session_state.raw_chat_text)
  if not messages:
    st.error(
        "No valid WhatsApp messages were detected. Ensure the file is an "
        "unmodified WhatsApp text export."
    )
    return

  sender_counts = Counter(m.sender for m in messages)
  senders_sorted = [s for s, _ in sender_counts.most_common()]

  with st.sidebar:
    st.divider()
    st.header("3. Persona & Few-Shot Tuning")
    target_sender = st.selectbox(
        "Which sender are YOU? (the style to clone)",
        options=senders_sorted,
        format_func=lambda s: f"{s} ({sender_counts[s]} msgs)",
    )
    num_few_shot = st.slider(
        "Few-shot turn pairs in prompt",
        min_value=5,
        max_value=40,
        value=18,
        step=1,
        help="Number of real [Friend -> You] exchanges injected dynamically per message.",
    )
    max_gap_sec = st.slider(
        "Multi-bubble grouping window (seconds)",
        min_value=30,
        max_value=600,
        value=180,
        step=30,
        help="Consecutive texts sent within this window are grouped as a multi-bubble reply.",
    )
    temperature = st.slider(
        "Creativity / Temperature",
        min_value=0.2,
        max_value=1.2,
        value=0.85,
        step=0.05,
    )
    extra_notes = st.text_area(
        "Optional Context (inside jokes, current city, job)",
        placeholder="e.g., Currently studying CS, loves iced matcha, hates morning classes",
    )
    if st.button("Clear Live Chat History", use_container_width=True):
      st.session_state.chat_history = []
      st.rerun()

  turns = group_into_turns(messages, max_gap_seconds=max_gap_sec)
  turn_pairs = build_turn_pairs(turns, target_sender=target_sender)
  profile = extract_style_profile(messages, turns, target_sender=target_sender)

  tab_chat, tab_style, tab_prompt = st.tabs([
      "💬 Live Chat Playground",
      "🧬 Your Style Fingerprint",
      "🔍 Prompt & Few-Shot Inspector",
  ])

  # --- TAB 1: Live Chat Playground ---
  with tab_chat:
    st.subheader(f"Texting with AI **{target_sender}**")
    st.caption(
        f"Loaded **{len(messages)}** messages and **{len(turn_pairs)}** "
        f"`[Friend → {target_sender}]` turn pairs from `{st.session_state.source_label}`."
    )

    for msg in st.session_state.chat_history:
      avatar = "🧑" if msg["role"] == "user" else "🤖"
      with st.chat_message(msg["role"], avatar=avatar):
        bubbles = [b.strip() for b in msg["content"].split("\n") if b.strip()]
        for b in bubbles:
          st.markdown(b)
        if msg["role"] == "assistant" and msg.get("few_shot_used"):
          with st.expander("View few-shot examples retrieved for this reply"):
            for ex in msg["few_shot_used"][-5:]:
              st.markdown(
                  f"- **Friend:** `{ex.incoming_text}` → **You:** `{ex.reply_text}`"
              )

    incoming = st.chat_input(f"Send a WhatsApp message to {target_sender}...")
    if incoming:
      st.session_state.chat_history.append(
          {"role": "user", "content": incoming}
      )
      with st.chat_message("user", avatar="🧑"):
        st.markdown(incoming)

      if not api_key:
        st.warning(
            "Please configure `GEMINI_API_KEY` in `.env` or enter it in the left sidebar to generate a live reply."
        )
      else:
        with st.chat_message("assistant", avatar="🤖"):
          with st.spinner(f"{target_sender} is typing..."):
            try:
              reply_text, _, used_examples = generate_clone_reply(
                  api_key=api_key,
                  model_name=model_name,
                  profile=profile,
                  all_turn_pairs=turn_pairs,
                  chat_history=st.session_state.chat_history[:-1],
                  incoming_message=incoming,
                  num_few_shot=num_few_shot,
                  temperature=temperature,
                  extra_persona_notes=extra_notes,
              )
              for bubble in [
                  b.strip() for b in reply_text.split("\n") if b.strip()
              ]:
                st.markdown(bubble)
              st.session_state.chat_history.append({
                  "role": "assistant",
                  "content": reply_text,
                  "few_shot_used": used_examples,
              })
            except Exception as exc:  # pylint: disable=broad-except
              st.error(f"Gemini API error: {exc}")

  # --- TAB 2: Style Fingerprint ---
  with tab_style:
    st.subheader(f"Extracted Texting DNA for **{profile.sender_name}**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lowercase Starts", f"{profile.lowercase_start_pct:.0f}%")
    col2.metric("No Trailing Period", f"{profile.no_period_end_pct:.0f}%")
    col3.metric("Multi-Bubble Replies", f"{profile.multi_bubble_turn_pct:.0f}%")
    col4.metric("Avg Words / Bubble", f"{profile.avg_words_per_bubble:.1f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Bubbles / Turn", f"{profile.avg_bubbles_per_turn:.2f}")
    col6.metric("Emoji / Message", f"{profile.emoji_per_msg:.2f}")
    col7.metric("Exclamation Rate", f"{profile.exclamation_pct:.0f}%")
    col8.metric("Question Rate", f"{profile.question_pct:.0f}%")

    st.divider()
    left_col, right_col = st.columns(2)
    with left_col:
      st.markdown("#### Signature Slang & Shorthand")
      if profile.top_slang:
        st.write(", ".join(f"`{w}` ({c}×)" for w, c in profile.top_slang))
      else:
        st.caption("No common slang tokens detected yet.")

      st.markdown("#### Favorite Emojis")
      if profile.top_emojis:
        st.write("  ".join(f"{e} (`{c}×`)" for e, c in profile.top_emojis))
      else:
        st.caption("No emojis found in this sender's messages.")

    with right_col:
      st.markdown("#### Go-To Message Openers")
      if profile.top_openers:
        st.write(", ".join(f"`{w}` ({c}×)" for w, c in profile.top_openers))
      st.markdown("#### Frequent Personal Vocabulary")
      if profile.top_words:
        st.write(", ".join(f"`{w}` ({c}×)" for w, c in profile.top_words))

  # --- TAB 3: Prompt & Few-Shot Inspector ---
  with tab_prompt:
    st.subheader("System Prompt & Dynamic Few-Shot Prompt Engineering")
    preview_query = st.text_input(
        "Test query to preview dynamic few-shot retrieval:",
        value="hey what are you up to tonight??",
    )
    preview_examples = select_few_shot_examples(
        turn_pairs, incoming_query=preview_query, k=num_few_shot
    )
    preview_prompt = build_system_instruction(
        profile=profile,
        few_shot_examples=preview_examples,
        extra_persona_notes=extra_notes,
    )
    st.code(preview_prompt, language="markdown")

    jsonl_lines = []
    for pair in turn_pairs:
      record = {
          "systemInstruction": {
              "role": "system",
              "parts": [{
                  "text": f"Reply in the exact WhatsApp style of {target_sender}."
              }],
          },
          "contents": [
              {"role": "user", "parts": [{"text": pair.incoming_text}]},
              {"role": "model", "parts": [{"text": pair.reply_text}]},
          ],
      }
      jsonl_lines.append(json.dumps(record, ensure_ascii=False))

    st.download_button(
        label="⬇️ Download Parsed Turn Pairs as Gemini Fine-Tuning JSONL",
        data="\n".join(jsonl_lines),
        file_name=f"{target_sender.lower().replace(' ', '_')}_whatsapp_finetune.jsonl",
        mime="application/jsonl",
    )


# ---------------------------------------------------------------------------
# 7. CLI & Self-Test Entrypoint
# ---------------------------------------------------------------------------


def run_self_test() -> None:
  """Verifies iOS parsing, turn grouping, style profiling, and secret masking."""
  msgs = parse_whatsapp_ios_export(SAMPLE_IOS_WHATSAPP_EXPORT)
  assert len(msgs) >= 20, f"Expected >=20 messages, got {len(msgs)}"

  turns = group_into_turns(msgs, max_gap_seconds=180)
  pairs = build_turn_pairs(turns, target_sender="Jordan")
  assert len(pairs) >= 6, f"Expected >=6 turn pairs, got {len(pairs)}"

  profile = extract_style_profile(msgs, turns, target_sender="Jordan")
  assert profile.sender_name == "Jordan"
  assert profile.lowercase_start_pct > 50.0
  assert profile.no_period_end_pct > 80.0
  assert profile.multi_bubble_turn_pct > 40.0

  selected = select_few_shot_examples(
      pairs, incoming_query="are you free tonight?", k=5
  )
  prompt = build_system_instruction(profile, selected)
  assert "QUANTITATIVE STYLE FINGERPRINT FOR JORDAN" in prompt
  assert "Example 1:" in prompt

  masked = mask_api_key("AIzaSyD1234567890abcdefXYZ")
  assert "1234567890" not in masked
  assert masked.startswith("AIza") and masked.endswith("fXYZ")

  print("=== Self-Test Passed ===")
  print(f"Parsed messages: {len(msgs)}")
  print(f"Grouped turns:   {len(turns)}")
  print(f"Turn pairs:      {len(pairs)}")
  print(f"Lowercase start: {profile.lowercase_start_pct}%")
  print(f"No period end:   {profile.no_period_end_pct}%")
  print(f"Multi-bubble %:  {profile.multi_bubble_turn_pct}%")
  print(f"Top slang:       {profile.top_slang[:6]}")
  print(f"Top emojis:      {profile.top_emojis[:6]}")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="WhatsApp Style-Cloning Bot (iOS Export + Gemini API + Streamlit UI)"
  )
  parser.add_argument(
      "--self-test",
      action="store_true",
      help="Run offline parser, style analyzer, and security unit checks.",
  )
  args, _ = parser.parse_known_args()

  if args.self_test:
    run_self_test()
    return

  try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
      run_streamlit_ui()
      return
  except ImportError:
    pass

  print(
      "To launch the local Web UI, run:\n"
      "    streamlit run whatsapp_clone_bot.py\n\n"
      "Or run offline verification tests with:\n"
      "    python3 whatsapp_clone_bot.py --self-test"
  )


if __name__ == "__main__":
  main()
