#!/bin/bash

# Install dependencies (Render will usually handle this in the build step)
#pip install -r requirements.txt

# Create output directory
mkdir -p output

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 10000
