#!/bin/bash

# Find the main repository root (first entry in git worktree list)
MAIN_ROOT=$(git worktree list | head -n 1 | awk '{print $1}')
# Find the current worktree root
WORKTREE=$(git rev-parse --show-toplevel)

echo "Main repository: $MAIN_ROOT"
echo "Current worktree: $WORKTREE"

# 1. Symlink the backend/.env file (where your project's env is stored)
if [ -f "$MAIN_ROOT/backend/.env" ] && [ ! -e "$WORKTREE/backend/.env" ]; then
    echo "Symlinking backend/.env..."
    ln -s "$MAIN_ROOT/backend/.env" "$WORKTREE/backend/.env"
fi

# Just in case you also add a root .env in the future
if [ -f "$MAIN_ROOT/.env" ] && [ ! -e "$WORKTREE/.env" ]; then
    echo "Symlinking root .env..."
    ln -s "$MAIN_ROOT/.env" "$WORKTREE/.env"
fi

# 2. Create Python environment & install dependencies
echo "Setting up Python virtual environment..."
cd "$WORKTREE"
if [ ! -d ".venv" ]; then
    uv venv
fi

source .venv/bin/activate
uv pip install -r backend/requirements.txt

# 3. Optional: Install frontend dependencies
if [ -d "frontend" ] && [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    (cd frontend && npm install)
fi

echo "Worktree setup complete! 🎉"
