#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:$PATH"

SESSION="01221c"

echo "=== Uploading parts to $SESSION ==="
for part in /mnt/c/Users/Unnat/pusht_lance_part_*; do
    name=$(basename "$part")
    echo "  Uploading $name ($(du -sh "$part" | cut -f1))..."
    colab upload --session "$SESSION" "$part" "/content/$name"
done

echo "=== All parts uploaded. Starting training ==="
colab exec --session "$SESSION" --timeout 86400 -f /mnt/c/Users/Unnat/le-wm/colab_upload_train.py
