#!/bin/bash
# Build Sphinx documentation in background with logging

echo "Starting Sphinx documentation build..."
echo "Build started at: $(date)" > docs_build.log

# Create build directory if it doesn't exist
mkdir -p docs/_build

# Run sphinx-build in background, redirecting output to log file
cd docs && sphinx-build -b html -v . _build/html >> ../docs_build.log 2>&1 &

# Get the process ID
PID=$!
echo "Sphinx build started with PID: $PID"
echo "Build PID: $PID" >> ../docs_build.log

# Save PID to file so we can check it later
echo $PID > ../docs_build.pid

echo "Documentation build running in background..."
echo "Check progress with: tail -f docs_build.log"
echo "Check if complete with: ps -p \$(cat docs_build.pid)"
echo "View results when done at: docs/_build/html/index.html"