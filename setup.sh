#!/bin/bash
# Setup script

# uv handles venv creation automatically

# Install dependencies using uv
uv sync

# Create directories
mkdir -p models vectorstore

# Download model
python code/download_model.py

# Run ingestion
python code/ingest.py
