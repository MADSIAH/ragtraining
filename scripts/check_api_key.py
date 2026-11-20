"""Check that the Anthropic API key in .env works, without ever printing the key.

Run:  python scripts/check_api_key.py
Step 1 lists models (free) -> proves the key is valid.
Step 2 sends a one-word message (costs < $0.01) -> proves billing/credits work.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

key = os.environ.get("ANTHROPIC_API_KEY", "")
if not key or key == "paste-your-key-here":
    sys.exit("✗ No key yet: open .env and paste your key after ANTHROPIC_API_KEY=")
if not key.startswith("sk-ant-"):
    sys.exit("✗ The value in .env doesn't look like an Anthropic key (it should start with 'sk-ant-').")

try:
    import anthropic
except ImportError:
    sys.exit("✗ The 'anthropic' package isn't installed yet. Run Phase 0 setup first.")

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# Step 1: free call, validates the key
try:
    ids = [m.id for m in client.models.list(limit=5).data]
except anthropic.AuthenticationError:
    sys.exit("✗ Key rejected (401). Re-copy it from the Console or create a new one.")
except anthropic.PermissionDeniedError:
    sys.exit("✗ Key valid but not allowed (403). Check the key's workspace in the Console.")
except anthropic.APIConnectionError:
    sys.exit("✗ Could not reach the Anthropic API. Check your internet connection.")
print(f"✓ Step 1: key is valid (…{key[-4:]}). Models visible: {', '.join(ids)}")

# Step 2: tiny paid call, validates billing
try:
    msg = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
    )
except anthropic.BadRequestError as e:
    sys.exit(f"✗ Step 2 failed (400), usually no credits yet → Console › Billing. Details: {e.message}")
except anthropic.RateLimitError:
    sys.exit("✗ Step 2 hit a rate limit (429). Wait a minute and retry.")
except anthropic.APIStatusError as e:
    sys.exit(f"✗ Step 2 failed with HTTP {e.status_code}: {e.message}")

text = "".join(b.text for b in msg.content if b.type == "text").strip()
print(f"✓ Step 2: billing works. Claude replied: {text!r} "
      f"({msg.usage.input_tokens} in / {msg.usage.output_tokens} out tokens)")
