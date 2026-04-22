#!/usr/bin/env bash
set -euo pipefail

# One-command deploy helper.
# Prereqs: pip install modal && modal token new

echo ">>> Deploying useknockout to Modal..."
modal deploy main.py

echo ""
echo ">>> Deployed. Test with:"
echo ""
echo "URL=https://useknockout--api-api.modal.run"
echo "curl \$URL/health"
