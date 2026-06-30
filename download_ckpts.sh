#!/bin/bash
# Usage: bash download_ckpts.sh <session-name> <num-epochs>
# Test:  bash download_ckpts.sh abc123 1
# Full:  bash download_ckpts.sh abc123 10

SESSION=$1
NUM_EPOCHS=${2:-10}
OUT="/mnt/c/Users/Unnat/le-wm/outputs/checkpoints"
COLAB="$HOME/.local/bin/colab"

if [ -z "$SESSION" ]; then
    echo "Usage: $0 <session-name> [num-epochs]"
    exit 1
fi

mkdir -p "$OUT"

echo "=== Waiting for session setup... ==="
while true; do
    timeout 20 "$COLAB" download --session "$SESSION" /content/test_ready.txt "$OUT/test_ready.txt" >/dev/null 2>&1
    [ -f "$OUT/test_ready.txt" ] && break
    echo "  not ready yet, retrying in 15s..."
    sleep 15
done
echo "✓ Session up — download confirmed working"

echo "=== Watching for checkpoints (epochs 0-$((NUM_EPOCHS-1))) ==="
for i in $(seq 0 $((NUM_EPOCHS - 1))); do
    FNAME="weights_epoch_${i}.ckpt"
    echo "Waiting for $FNAME..."
    while true; do
        timeout 60 "$COLAB" download --session "$SESSION" \
            "/content/stable-wm/checkpoints/$FNAME" \
            "$OUT/$FNAME" >/dev/null 2>&1
        if [ -f "$OUT/$FNAME" ] && [ "$(stat -c%s "$OUT/$FNAME" 2>/dev/null || stat -f%z "$OUT/$FNAME")" -gt 100000 ]; then
            echo "✓ $FNAME saved ($(du -sh "$OUT/$FNAME" | cut -f1))"
            break
        fi
        echo "  checkpoint not ready yet, retrying in 30s..."
        sleep 30
    done
done

echo ""
echo "=== All $NUM_EPOCHS checkpoints downloaded to $OUT ==="
ls -lh "$OUT"/*.ckpt
