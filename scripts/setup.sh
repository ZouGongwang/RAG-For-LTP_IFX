#!/usr/bin/env bash
# macOS / Linux setup. Usage: ./scripts/setup.sh [--lite]   (--lite skips Docling)
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [[ "${1:-}" == "--lite" ]]; then
  .venv/bin/pip install -e ".[cloud,dev]"
  echo "Lite install: set 'parsing.backend: simple' in config.yaml"
else
  .venv/bin/pip install -e ".[all]"
fi
[[ -f .env ]] || cp .env.example .env

echo
echo "Next steps:"
echo "  ollama pull deepseek-r1:8b && ollama pull llava:7b"
echo "  source .venv/bin/activate"
echo "  python -m rag doctor"
echo "  python -m rag ingest"
echo "  python -m rag ask"
