#!/bin/bash
source venv/bin/activate || source new_venv/bin/activate

rm -f results.txt

echo "1: HOLDOUT SET (35°C Unseen)"
echo "--- Holdout - Baseline (No Scaling) ---" >> results.txt
python train.py --no-scaling --holdout_temp 35.0 --plot_dir plots_holdout_unscaled

echo "--- Holdout - Proposed (Scaling) ---" >> results.txt
python train.py --holdout_temp 35.0 --plot_dir plots_holdout_scaled

echo ""
echo "2: RANDOM SPLIT (Mixed Temperatures)"
echo "--- Random Split - Baseline (No Scaling) ---" >> results.txt
python train.py --no-scaling --plot_dir plots_random_unscaled

echo "--- Random Split - Proposed (Scaling) ---" >> results.txt
python train.py --plot_dir plots_random_scaled


echo "DONE 100"