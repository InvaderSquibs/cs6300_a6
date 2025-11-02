# Phoenix Observability Setup

## Quick Start

### Option 1: Start Phoenix Manually (Recommended)

1. **Start Phoenix server in a separate terminal:**
   ```bash
   python3 -m phoenix serve
   ```
   This will start Phoenix at `http://localhost:6006`
   
   **Or let the script start it automatically:**
   The `test_with_phoenix.py` script will automatically start Phoenix for you!

2. **In another terminal, run your test with tracing:**
   ```bash
   python3 test_with_phoenix.py
   ```

### Option 2: Using the Helper Script

```bash
# Make script executable (first time only)
chmod +x run_with_tracing.sh

# Run with automatic Phoenix startup
./run_with_tracing.sh
```

## How Tracing Works

Tracing is enabled via environment variables that LangChain/LangGraph reads:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=http://localhost:6006
export LANGCHAIN_API_KEY=phoenix  # Can be any string for Phoenix
```

The `test_with_phoenix.py` script sets these automatically.

## Verifying Phoenix is Running

Open your browser to: http://localhost:6006

You should see the Phoenix UI. When you run queries, traces will appear there in real-time.

## Troubleshooting

### No traces appearing?

1. **Check Phoenix is running:**
   ```bash
   curl http://localhost:6006/health
   ```

2. **Check environment variables are set:**
   ```bash
   echo $LANGCHAIN_TRACING_V2
   echo $LANGCHAIN_ENDPOINT
   ```

3. **Verify LM Studio is running (default local LLM):**
   ```bash
   curl http://localhost:1234/v1/models
   ```
   The system defaults to LM Studio at `http://localhost:1234/v1`

### Phoenix won't start?

If you see import errors with Phoenix, try:
```bash
pip install --upgrade arize-phoenix fastapi
```

Or use the manual method - Phoenix can run as a separate process.

## What You'll See in Phoenix

- **Traces**: Each workflow execution as a trace
- **Spans**: Individual nodes (check_needs_context, pull_from_chroma, etc.)
- **LLM Calls**: All LLM invocations with inputs/outputs
- **Timings**: How long each step takes
- **Token Usage**: Token counts for local LLMs (if available)

## Local LLM Tracing

Local LLMs (LM Studio, Ollama) should work with tracing if:
- Phoenix server is running
- Environment variables are set BEFORE importing LangChain
- The LLM is using LangChain's ChatOpenAI or ChatOllama classes

The `test_with_phoenix.py` script handles all of this automatically.

