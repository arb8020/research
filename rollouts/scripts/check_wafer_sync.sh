#!/bin/bash
# Check if wafer's vendored rollouts is in sync with upstream (this repo)
#
# Usage: ./scripts/check_wafer_sync.sh
#
# Returns exit code 0 if in sync, 1 if different

set -e

UPSTREAM="$HOME/research/rollouts/rollouts"
VENDORED="$HOME/wafer/packages/wafer-core/wafer_core/rollouts"

if [ ! -d "$VENDORED" ]; then
    echo "❌ Vendored rollouts not found at: $VENDORED"
    exit 1
fi

echo "Comparing:"
echo "  Upstream: $UPSTREAM"
echo "  Vendored: $VENDORED"
echo ""

# Get diff stats
DIFF_OUTPUT=$(diff -rq "$UPSTREAM" "$VENDORED" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='UPSTREAM.txt' \
    2>/dev/null || true)

if [ -z "$DIFF_OUTPUT" ]; then
    echo "✅ In sync!"
    exit 0
else
    # Count differences
    ONLY_UPSTREAM=$(echo "$DIFF_OUTPUT" | grep "^Only in $UPSTREAM" | wc -l | tr -d ' ')
    ONLY_VENDORED=$(echo "$DIFF_OUTPUT" | grep "^Only in $VENDORED" | wc -l | tr -d ' ')
    DIFFER=$(echo "$DIFF_OUTPUT" | grep "^Files .* differ$" | wc -l | tr -d ' ')
    
    echo "❌ Out of sync!"
    echo ""
    echo "Summary:"
    echo "  Files only in upstream: $ONLY_UPSTREAM"
    echo "  Files only in vendored: $ONLY_VENDORED"
    echo "  Files that differ: $DIFFER"
    echo ""
    
    # Show details
    if [ "$ONLY_UPSTREAM" -gt 0 ]; then
        echo "Files only in upstream (need to add to wafer):"
        echo "$DIFF_OUTPUT" | grep "^Only in $UPSTREAM" | sed 's/Only in /  /'
        echo ""
    fi
    
    if [ "$ONLY_VENDORED" -gt 0 ]; then
        echo "Files only in vendored (may need to remove or upstream):"
        echo "$DIFF_OUTPUT" | grep "^Only in $VENDORED" | sed 's/Only in /  /'
        echo ""
    fi
    
    if [ "$DIFFER" -gt 0 ]; then
        echo "Files that differ:"
        echo "$DIFF_OUTPUT" | grep "^Files .* differ$" | sed 's/Files /  /' | sed 's/ differ$//' | \
            sed "s|$UPSTREAM/||" | sed "s| and .*||"
        echo ""
    fi
    
    echo "To see full diff for a file:"
    echo "  diff $UPSTREAM/<file> $VENDORED/<file>"
    echo ""
    echo "To sync (copy upstream to vendored):"
    echo "  rsync -av --delete --exclude='__pycache__' $UPSTREAM/ $VENDORED/"
    
    exit 1
fi
