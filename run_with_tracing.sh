#!/bin/bash
# Script to run the RAG system with Phoenix tracing enabled

echo "============================================================"
echo "Starting Phoenix Server for Tracing"
echo "============================================================"
echo ""

# Start Phoenix in background using python module
echo "Starting Phoenix server on port 6006..."
python3 -m phoenix serve > /dev/null 2>&1 &
PHOENIX_PID=$!

# Wait a moment for Phoenix to start
sleep 3

echo "✓ Phoenix should be running at http://localhost:6006"
echo "  (PID: $PHOENIX_PID)"
echo ""

# Set tracing environment variables
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=http://localhost:6006
export LANGCHAIN_API_KEY=phoenix

echo "Tracing enabled:"
echo "  LANGCHAIN_TRACING_V2=true"
echo "  LANGCHAIN_ENDPOINT=http://localhost:6006"
echo ""

# Run the Python test
echo "Running test with tracing..."
echo ""
python3 test_with_phoenix.py

# Cleanup: kill Phoenix when script exits
trap "kill $PHOENIX_PID 2>/dev/null" EXIT

echo ""
echo "Press Ctrl+C to stop Phoenix server"
wait $PHOENIX_PID

