#!/usr/bin/env bash
# Headless Claude Code worker billed to the Anthropic API key (Default workspace), not the subscription.
# Usage: claude_api.sh <prompt-file> [model]   (runs in the RSNA repo; full tool access)
set -euo pipefail
REPO="C:/Users/medro/Documents/_PERSONAL_PROJECTS/RSNA_Challenge"
cd "$REPO"; set -a; . ./.secrets.env; set +a
export ANTHROPIC_CUSTOM_HEADERS="anthropic-workspace-id: $ANTHROPIC_WORKSPACE_ID"
claude -p "$(cat "$1")" --model "${2:-claude-opus-5-5}" --dangerously-skip-permissions --output-format text
