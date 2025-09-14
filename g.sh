#!/bin/bash

# --- Git Push Automation (Ubuntu/macOS/Linux) ---

# Check if a commit message was provided
if [ -z "$1" ]; then
    echo "Usage: $0 \"Your commit message here\"" >&2
    echo "Example: $0 \"Feat: Added new login screen\"" >&2
    echo "" >&2
    echo "Please provide a commit message as the first argument." >&2
    exit 1
fi

# Store the commit message from the first argument
COMMIT_MESSAGE="$1"

echo ""
echo "--- Git Push Automation (Linux/macOS) ---"
echo "Commit Message: \"$COMMIT_MESSAGE\""
echo ""

# Step 1: git add .
echo "Running: git add ."
git add .
if [ $? -ne 0 ]; then
    echo "Error during git add. Aborting." >&2
    exit 1
fi

# Step 2: git commit -m "commit_message"
echo "Running: git commit -m \"$COMMIT_MESSAGE\""
git commit -m "$COMMIT_MESSAGE"
if [ $? -ne 0 ]; then
    echo "Error during git commit. This might be because there are no changes to commit, or a merge conflict." >&2
    exit 1
fi

# Step 3: git push
echo "Running: git push"
git push
if [ $? -ne 0 ]; then
    echo "Error during git push. Check your network, credentials, or remote status." >&2
    exit 1
fi

echo ""
echo "--- Git Push Successful! ---"
