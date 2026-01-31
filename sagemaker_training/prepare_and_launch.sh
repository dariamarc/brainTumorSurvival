#!/bin/bash
# Prepare and launch SageMaker Training Job
#
# This script:
# 1. Copies necessary project files to the training package
# 2. Uploads to S3
# 3. Launches the training job
#
# Usage:
#   ./prepare_and_launch.sh your-bucket-name [additional-args]
#
# Examples:
#   ./prepare_and_launch.sh my-brats-bucket
#   ./prepare_and_launch.sh my-brats-bucket --phase1-epochs 30 --instance ml.g5.xlarge

set -e

if [ -z "$1" ]; then
    echo "Usage: ./prepare_and_launch.sh <bucket-name> [additional-args]"
    echo ""
    echo "Examples:"
    echo "  ./prepare_and_launch.sh my-brats-bucket"
    echo "  ./prepare_and_launch.sh my-brats-bucket --phase1-epochs 30"
    exit 1
fi

BUCKET=$1
shift  # Remove bucket from args, rest are passed to launch script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PACKAGE_DIR="$SCRIPT_DIR/package"

echo "=================================================="
echo "Preparing SageMaker Training Package"
echo "=================================================="

# Clean and create package directory
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# Copy training script
cp "$SCRIPT_DIR/train.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$PACKAGE_DIR/"

# Copy project modules
echo "Copying project modules..."
cp -r "$PROJECT_ROOT/ResNet_architecture" "$PACKAGE_DIR/"
cp -r "$PROJECT_ROOT/data_processing" "$PACKAGE_DIR/"

# Remove unnecessary files to reduce package size
find "$PACKAGE_DIR" -name "*.pyc" -delete
find "$PACKAGE_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$PACKAGE_DIR" -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Package contents:"
find "$PACKAGE_DIR" -type f -name "*.py" | head -20
echo "..."

# Get package size
PACKAGE_SIZE=$(du -sh "$PACKAGE_DIR" | cut -f1)
echo "Package size: $PACKAGE_SIZE"

echo ""
echo "=================================================="
echo "Launching Training Job"
echo "=================================================="

# Launch the training job
cd "$PACKAGE_DIR"
python train.py --help > /dev/null 2>&1 || pip install -r requirements.txt > /dev/null

cd "$SCRIPT_DIR"
python launch_training.py --bucket "$BUCKET" "$@"
