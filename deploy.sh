#!/bin/bash
# Simple deployment script for RunPod Serverless

set -e

echo "============================================"
echo "Chandra OCR - RunPod Serverless Deployment"
echo "============================================"
echo ""

# Check if .env.runpod exists
if [ ! -f .env.runpod ]; then
    echo "❌ Error: .env.runpod file not found"
    echo ""
    echo "Please create .env.runpod file:"
    echo "  cp .env.runpod.example .env.runpod"
    echo "  # Edit .env.runpod with your configuration"
    echo ""
    exit 1
fi

# Load environment variables
echo "Loading configuration from .env.runpod..."
export $(cat .env.runpod | grep -v '^#' | xargs)

# Verify RUNPOD_API_KEY is set
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "❌ Error: RUNPOD_API_KEY is not set in .env.runpod"
    exit 1
fi

echo "✓ Configuration loaded"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed"
    exit 1
fi

# Install required Python packages
echo "Installing Python dependencies..."
pip install -q requests
echo "✓ Dependencies installed"
echo ""

# Run deployment script
echo "Starting deployment..."
python3 deploy_runpod.py

echo ""
echo "Deployment complete!"
