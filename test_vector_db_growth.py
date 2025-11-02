"""
Test to verify vector DB growth and provide DB reset functionality.

This test demonstrates that the vector database grows when new papers are found.
It uses a baseline DB that can be saved and reset for consistent testing.

Usage:
    python3 test_vector_db_growth.py        # Run test (resets to baseline first)
    python3 test_vector_db_growth.py save   # Save current DB as baseline
    python3 test_vector_db_growth.py reset  # Reset DB to baseline only

The test will:
1. Reset DB to baseline state (empty by default, or saved state if exists)
2. Show initial document count
3. Run a query that triggers arxiv search if DB is empty
4. Verify DB count growth
5. Display proof of growth with before/after counts
"""
import os
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from src.workflow import GameTheoryRAG
from src.vector_db import VectorDBManager


# Directory paths
CHROMA_DB_DIR = "./chroma_db"
CHROMA_DB_BASELINE = "./chroma_db_baseline"
CHROMA_DB_TEST = "./chroma_db_test"


def reset_db_to_baseline(baseline_dir: str = CHROMA_DB_BASELINE, target_dir: str = CHROMA_DB_DIR):
    """Reset the vector DB by copying from baseline."""
    print(f"\n{'='*60}")
    print("Resetting Vector DB")
    print(f"{'='*60}")
    
    # Remove existing DB if it exists
    if os.path.exists(target_dir):
        print(f"Removing existing DB: {target_dir}")
        shutil.rmtree(target_dir)
    
    # Create baseline if it doesn't exist (empty)
    if not os.path.exists(baseline_dir):
        print(f"Creating empty baseline DB: {baseline_dir}")
        os.makedirs(baseline_dir, exist_ok=True)
        # Create empty ChromaDB instance to initialize structure
        db = VectorDBManager(persist_directory=baseline_dir)
        print(f"  Baseline created with {db.count()} documents")
    
    # Check if baseline has content
    baseline_exists = os.path.exists(baseline_dir) and os.path.isdir(baseline_dir)
    baseline_has_content = baseline_exists and any(os.listdir(baseline_dir))
    
    if baseline_has_content:
        print(f"Copying baseline from {baseline_dir} to {target_dir}")
        shutil.copytree(baseline_dir, target_dir)
        db = VectorDBManager(persist_directory=target_dir)
        print(f"✓ DB reset complete. Starting count: {db.count()} documents")
    else:
        print(f"Using fresh empty DB (baseline is empty)")
        os.makedirs(target_dir, exist_ok=True)
        db = VectorDBManager(persist_directory=target_dir)
        print(f"✓ DB reset complete. Starting count: {db.count()} documents")
    
    return VectorDBManager(persist_directory=target_dir)


def save_db_as_baseline(source_dir: str = CHROMA_DB_DIR, baseline_dir: str = CHROMA_DB_BASELINE):
    """Save current DB state as baseline for future tests."""
    print(f"\n{'='*60}")
    print("Saving Current DB as Baseline")
    print(f"{'='*60}")
    
    if not os.path.exists(source_dir):
        print(f"Source DB does not exist: {source_dir}")
        return
    
    # Remove existing baseline
    if os.path.exists(baseline_dir):
        print(f"Removing existing baseline: {baseline_dir}")
        shutil.rmtree(baseline_dir)
    
    # Copy current DB to baseline
    print(f"Copying {source_dir} to {baseline_dir}")
    shutil.copytree(source_dir, baseline_dir)
    
    db = VectorDBManager(persist_directory=baseline_dir)
    print(f"✓ Baseline saved with {db.count()} documents")


def test_vector_db_growth():
    """Test that vector DB grows when new papers are found."""
    load_dotenv()
    
    print("="*60)
    print("Vector DB Growth Verification Test")
    print("="*60)
    print()
    
    # Reset DB to baseline (empty or predefined state)
    print("Step 1: Resetting DB to baseline state...")
    test_db = reset_db_to_baseline()
    initial_count = test_db.count()
    
    print(f"\n✓ Initial DB count: {initial_count} documents")
    
    # Initialize RAG system (will use the reset DB)
    print("\n" + "-"*60)
    print("Step 2: Initializing RAG System")
    print("-"*60)
    
    # Use test DB directory
    api_key = os.getenv("OPENAI_API_KEY")
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    local_base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
    
    if not api_key and not use_local:
        print(f"Using LM Studio at {local_base_url} (default)")
        rag = GameTheoryRAG(
            local_llm_base_url=local_base_url,
            local_llm_model=os.getenv("LOCAL_LLM_MODEL", "local-model"),
            openai_api_key="lm-studio"
        )
        # Override the vector DB to use test directory
        rag.vector_db = VectorDBManager(persist_directory=CHROMA_DB_DIR)
    elif use_local:
        rag = GameTheoryRAG(use_local_llm=True)
        rag.vector_db = VectorDBManager(persist_directory=CHROMA_DB_DIR)
    elif api_key:
        rag = GameTheoryRAG(openai_api_key=api_key)
        rag.vector_db = VectorDBManager(persist_directory=CHROMA_DB_DIR)
    
    # Check count before query
    count_before = rag.vector_db.count()
    print(f"✓ DB count before query: {count_before} documents")
    
    print("\n" + "="*60)
    print("Step 3: Running Query (will trigger arxiv search if needed)")
    print("="*60)
    print()
    
    # Query that should trigger arxiv search if DB is empty
    test_query = "What is the Nash equilibrium in game theory?"
    print(f"Query: {test_query}")
    print()
    
    try:
        response = rag.query(test_query)
        
        # Check count after query
        count_after = rag.vector_db.count()
        
        print("\n" + "="*60)
        print("Step 4: Verifying DB Growth")
        print("="*60)
        print()
        
        print(f"Documents before query: {count_before}")
        print(f"Documents after query:  {count_after}")
        print(f"Growth:                  {count_after - count_before} new documents")
        print()
        
        if count_after > count_before:
            print("✓ SUCCESS: Vector DB grew when new papers were found!")
            print(f"  Added {count_after - count_before} new document chunks")
        elif count_before == 0 and count_after == 0:
            print("⚠ No documents added. Query may not have triggered arxiv search.")
            print("  This could happen if:")
            print("  - Query was answered without context")
            print("  - Arxiv search failed")
            print("  - Documents couldn't be processed")
        else:
            print("✓ DB state unchanged (papers may have already been in DB)")
        
        print("\n" + "-"*60)
        print("RESPONSE:")
        print("-"*60)
        print(response[:500] + "..." if len(response) > 500 else response)
        print()
        
        # Show save options
        baseline_count = 0
        if os.path.exists(CHROMA_DB_BASELINE):
            try:
                baseline_db = VectorDBManager(persist_directory=CHROMA_DB_BASELINE)
                baseline_count = baseline_db.count()
            except:
                pass
        
        print("\n" + "="*60)
        print("Step 5: Baseline Management")
        print("="*60)
        print()
        print(f"Current DB:    {count_after} documents")
        print(f"Baseline:      {baseline_count} documents")
        print()
        print("Commands:")
        print("  Save current state as baseline:")
        print("    python3 test_vector_db_growth.py save")
        print()
        print("  Reset to baseline (empty):")
        print("    python3 test_vector_db_growth.py reset")
        print()
        print("  Run test again:")
        print("    python3 test_vector_db_growth.py")
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test Complete - Summary")
    print("="*60)
    print()
    
    final_count = rag.vector_db.count()
    baseline_count = 0
    if os.path.exists(CHROMA_DB_BASELINE):
        try:
            baseline_db = VectorDBManager(persist_directory=CHROMA_DB_BASELINE)
            baseline_count = baseline_db.count()
        except:
            pass
    
    print("📊 DB State:")
    print(f"  Test DB:    {CHROMA_DB_DIR}")
    print(f"    Count:    {final_count} documents")
    print(f"  Baseline:   {CHROMA_DB_BASELINE}")
    print(f"    Count:    {baseline_count} documents")
    print()
    print("✅ Proof of Growth:")
    print(f"  Started with: {initial_count} documents")
    print(f"  Ended with:   {final_count} documents")
    print(f"  Growth:       {final_count - initial_count} new documents")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "save":
            save_db_as_baseline()
        elif sys.argv[1] == "reset":
            reset_db_to_baseline()
            print("\n✓ DB reset complete")
        else:
            print("Usage:")
            print("  python3 test_vector_db_growth.py        # Run test")
            print("  python3 test_vector_db_growth.py save   # Save current DB as baseline")
            print("  python3 test_vector_db_growth.py reset  # Reset DB to baseline")
    else:
        test_vector_db_growth()

