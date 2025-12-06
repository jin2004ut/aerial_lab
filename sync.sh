#!/bin/bash

BIG_REPO="/home/wentao/rllab/copy"
SMALL_REPO="/home/wentao/rllab/aerial_lab"
SRC_DIR="source/aerial_lab/data"
DST_DIR="source/aerial_lab/data"

echo "🔄 Syncing content..."

rsync -acv --delete \
    "$BIG_REPO/$SRC_DIR/" \
    "$SMALL_REPO/$DST_DIR/"

cd "$SMALL_REPO" || exit

if [ -n "$(git status --porcelain)" ]; then
    echo "🔄 Changes detected, need to commit"
else
    echo "✨ No changes, no need to commit"
fi
