"""Central paths and settings. Import from here instead of hard-coding paths."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"  # read-only course material
DATA_DIR = ROOT / "data"        # generated artifacts (git-ignored)
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")      # ANTHROPIC_API_KEY, OLLAMA_HOST

CLAUDE_MODEL = "claude-opus-5-5"
LOCAL_MODEL = "ollama:qwen3:14b"   # free local mode (D22): tutor answers on this Mac, no API key
