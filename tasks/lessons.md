# Lessons Learned

## L001 — Python 3.9 Union Type Syntax
**What happened:** Used `str | None` union syntax in a type hint (`src/llm.py`), which requires Python 3.10+. Caused a `TypeError` on the user's Python 3.9 environment.

**Rule:** Always use `Optional[str]` from `typing` for nullable type hints. Never use `X | Y` union syntax unless the project explicitly requires Python 3.10+.

**Fix applied:** Added `from typing import Optional` and replaced `str | None` with `Optional[str]`.

**How to apply:** Before writing any type hints, check the Python version in the environment (`python3 --version`). Default to `typing` module imports for all union/optional types.
