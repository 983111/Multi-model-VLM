#!/bin/bash
# setup.sh — run this ONCE in Codespaces to fix all dependency issues
# Usage: bash setup.sh

set -e
echo "=== Step 1: Downgrade numpy to <2.0 (torch 2.2 needs it) ==="
pip install "numpy<2.0" --force-reinstall -q

echo "=== Step 2: Install pinned requirements ==="
pip install -r requirements.txt -q

echo "=== Step 3: Verify torch imports cleanly ==="
python -c "import torch; print('torch', torch.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "from transformers import AutoProcessor; print('transformers OK')"
python -c "from fastapi import FastAPI; print('fastapi OK')"

echo ""
echo "All done! Now run the pipeline:"
echo "  python data/prepare_dataset.py --mode synthetic --max_samples 300"
echo "  python models/finetune.py --dry_run"
echo "  python eval/run_eval.py --dry_run"
echo "  python api/main.py"
