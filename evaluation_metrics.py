"""
Evaluation metrics collection script for Game Theory RAG agent.

This script runs queries through the agent and collects quantitative metrics
to evaluate agent performance, including:
- Query success rate
- Vector database growth
- Response quality metrics
- Tool usage statistics
- Loop efficiency metrics
"""
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

from src.workflow import GameTheoryRAG
from src.vector_db import VectorDBManager

load_dotenv()


class MetricsCollector:
    """Collects and aggregates metrics from agent execution."""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "vector_db_growth": [],
            "tool_usage": {
                "vector_db_queries": 0,
                "arxiv_searches": 0,
                "papers_added": 0,
                "papers_filtered": 0,
            },
            "response_quality": {
                "total_length": 0,
                "avg_length": 0,
                "responses_with_context": 0,
                "responses_without_context": 0,
            },
            "loop_efficiency": {
                "total_iterations": 0,
                "avg_iterations_per_query": 0,
                "queries_with_loop": 0,
            },
            "query_times": [],
        }
    
    def collect_query_metrics(
        self,
        query: str,
        response: str,
        db_count_before: int,
        db_count_after: int,
        execution_info: Dict[str, Any]
    ):
        """
        Collect metrics from a single query execution.
        
        Args:
            query: The user query string
            response: The agent's response
            db_count_before: Vector DB document count before query
            db_count_after: Vector DB document count after query
            execution_info: Dictionary with execution details:
                - vector_db_called: bool
                - arxiv_searched: bool
                - papers_found: int
                - papers_added: int
                - iterations: int
                - response_has_context: bool
        """
        self.metrics["total_queries"] += 1
        
        # Success metrics
        if response and len(response.strip()) > 0:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["failed_queries"] += 1
        
        # Vector DB growth
        growth = db_count_after - db_count_before
        if growth > 0:
            self.metrics["vector_db_growth"].append({
                "query": query[:50],
                "growth": growth,
                "before": db_count_before,
                "after": db_count_after,
            })
        
        # Tool usage
        if execution_info.get("vector_db_called"):
            self.metrics["tool_usage"]["vector_db_queries"] += 1
        if execution_info.get("arxiv_searched"):
            self.metrics["tool_usage"]["arxiv_searches"] += 1
        if execution_info.get("papers_added", 0) > 0:
            self.metrics["tool_usage"]["papers_added"] += execution_info["papers_added"]
        if execution_info.get("papers_filtered", 0) > 0:
            self.metrics["tool_usage"]["papers_filtered"] += execution_info["papers_filtered"]
        
        # Response quality
        response_length = len(response)
        self.metrics["response_quality"]["total_length"] += response_length
        if execution_info.get("response_has_context"):
            self.metrics["response_quality"]["responses_with_context"] += 1
        else:
            self.metrics["response_quality"]["responses_without_context"] += 1
        
        # Loop efficiency
        iterations = execution_info.get("iterations", 1)
        self.metrics["loop_efficiency"]["total_iterations"] += iterations
        if iterations > 1:
            self.metrics["loop_efficiency"]["queries_with_loop"] += 1
    
    def calculate_averages(self):
        """Calculate average metrics."""
        if self.metrics["total_queries"] > 0:
            self.metrics["response_quality"]["avg_length"] = (
                self.metrics["response_quality"]["total_length"] /
                self.metrics["total_queries"]
            )
            self.metrics["loop_efficiency"]["avg_iterations_per_query"] = (
                self.metrics["loop_efficiency"]["total_iterations"] /
                self.metrics["total_queries"]
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        self.calculate_averages()
        
        success_rate = 0.0
        if self.metrics["total_queries"] > 0:
            success_rate = (
                self.metrics["successful_queries"] /
                self.metrics["total_queries"]
            ) * 100
        
        total_growth = sum(g["growth"] for g in self.metrics["vector_db_growth"])
        
        return {
            "summary": {
                "total_queries": self.metrics["total_queries"],
                "successful_queries": self.metrics["successful_queries"],
                "failed_queries": self.metrics["failed_queries"],
                "success_rate_percent": round(success_rate, 2),
            },
            "vector_db": {
                "total_growth": total_growth,
                "growth_events": len(self.metrics["vector_db_growth"]),
                "growth_per_event": round(total_growth / len(self.metrics["vector_db_growth"]), 2) if self.metrics["vector_db_growth"] else 0,
            },
            "tool_usage": self.metrics["tool_usage"],
            "response_quality": {
                "avg_response_length": round(self.metrics["response_quality"]["avg_length"], 2),
                "responses_with_context": self.metrics["response_quality"]["responses_with_context"],
                "responses_without_context": self.metrics["response_quality"]["responses_without_context"],
                "context_usage_rate": round(
                    (self.metrics["response_quality"]["responses_with_context"] /
                     self.metrics["total_queries"] * 100) if self.metrics["total_queries"] > 0 else 0,
                    2
                ),
            },
            "loop_efficiency": {
                "avg_iterations": round(self.metrics["loop_efficiency"]["avg_iterations_per_query"], 2),
                "queries_with_loop": self.metrics["loop_efficiency"]["queries_with_loop"],
                "loop_rate": round(
                    (self.metrics["loop_efficiency"]["queries_with_loop"] /
                     self.metrics["total_queries"] * 100) if self.metrics["total_queries"] > 0 else 0,
                    2
                ),
            },
        }
    
    def print_summary(self):
        """Print formatted summary of metrics."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("EVALUATION METRICS SUMMARY")
        print("="*70)
        
        print("\n📊 Query Performance:")
        print(f"  Total Queries: {summary['summary']['total_queries']}")
        print(f"  Successful: {summary['summary']['successful_queries']}")
        print(f"  Failed: {summary['summary']['failed_queries']}")
        print(f"  Success Rate: {summary['summary']['success_rate_percent']}%")
        
        print("\n📈 Vector Database Growth:")
        print(f"  Total Documents Added: {summary['vector_db']['total_growth']}")
        print(f"  Growth Events: {summary['vector_db']['growth_events']}")
        if summary['vector_db']['growth_events'] > 0:
            print(f"  Avg Growth per Event: {summary['vector_db']['growth_per_event']} documents")
        
        print("\n🔧 Tool Usage:")
        print(f"  Vector DB Queries: {summary['tool_usage']['vector_db_queries']}")
        print(f"  Arxiv Searches: {summary['tool_usage']['arxiv_searches']}")
        print(f"  Papers Added: {summary['tool_usage']['papers_added']}")
        print(f"  Papers Filtered: {summary['tool_usage']['papers_filtered']}")
        
        print("\n✨ Response Quality:")
        print(f"  Avg Response Length: {summary['response_quality']['avg_response_length']} chars")
        print(f"  Responses with Context: {summary['response_quality']['responses_with_context']}")
        print(f"  Responses without Context: {summary['response_quality']['responses_without_context']}")
        print(f"  Context Usage Rate: {summary['response_quality']['context_usage_rate']}%")
        
        print("\n🔄 Loop Efficiency:")
        print(f"  Avg Iterations per Query: {summary['loop_efficiency']['avg_iterations']}")
        print(f"  Queries with Loop: {summary['loop_efficiency']['queries_with_loop']}")
        print(f"  Loop Rate: {summary['loop_efficiency']['loop_rate']}%")
        
        print("\n" + "="*70)


def run_evaluation_queries(rag: GameTheoryRAG, queries: List[str]) -> MetricsCollector:
    """
    Run a set of queries and collect metrics.
    
    Args:
        rag: Initialized GameTheoryRAG instance
        queries: List of query strings to test
    
    Returns:
        MetricsCollector with collected metrics
    """
    collector = MetricsCollector()
    vector_db = rag.vector_db
    
    print("="*70)
    print("RUNNING EVALUATION QUERIES")
    print("="*70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}/{len(queries)}: {query[:60]}...")
        
        # Get initial state
        db_count_before = vector_db.count()
        
        # Track execution info (simplified - in real implementation would need to hook into workflow)
        execution_info = {
            "vector_db_called": True,  # Always called as entry point
            "arxiv_searched": False,
            "papers_found": 0,
            "papers_added": 0,
            "papers_filtered": 0,
            "iterations": 1,  # Simplified - would need workflow tracing
            "response_has_context": False,
        }
        
        try:
            # Run query
            response = rag.query(query)
            
            # Get final state
            db_count_after = vector_db.count()
            
            # Determine if response has context (simplified heuristic)
            has_context = (
                len(response) > 100 and  # Non-trivial response
                "don't have enough information" not in response.lower() and
                "not game theory" not in response.lower()
            )
            execution_info["response_has_context"] = has_context
            
            # Check if papers were added (simplified)
            if db_count_after > db_count_before:
                execution_info["papers_added"] = db_count_after - db_count_before
                execution_info["arxiv_searched"] = True  # Likely searched if papers added
                execution_info["iterations"] = 2  # At least 2 iterations (query + add)
            
            # Collect metrics
            collector.collect_query_metrics(
                query=query,
                response=response,
                db_count_before=db_count_before,
                db_count_after=db_count_after,
                execution_info=execution_info
            )
            
            print(f"  ✓ Response length: {len(response)} chars")
            print(f"  ✓ DB growth: {db_count_after - db_count_before} documents")
            
        except Exception as e:
            print(f"  ✗ Query failed: {e}")
            collector.metrics["failed_queries"] += 1
            collector.metrics["total_queries"] += 1
    
    return collector


def main():
    """Main evaluation function."""
    print("="*70)
    print("Game Theory RAG Agent - Evaluation Metrics")
    print("="*70)
    print("\nThis script collects quantitative metrics on agent performance.")
    print("Make sure LM Studio is running or set OPENAI_API_KEY in .env")
    
    # Test queries
    test_queries = [
        "What is Nash equilibrium?",
        "Explain the prisoner's dilemma",
        "What are mixed strategies in game theory?",
        "How does game theory apply to economics?",
        "What is a dominant strategy?",
    ]
    
    print(f"\nTest queries ({len(test_queries)}):")
    for i, q in enumerate(test_queries, 1):
        print(f"  {i}. {q}")
    
    try:
        response = input("\nPress Enter to start evaluation (or Ctrl+C to cancel)... ")
    except KeyboardInterrupt:
        print("\nEvaluation cancelled.")
        sys.exit(0)
    
    # Initialize agent
    print("\nInitializing agent...")
    rag = GameTheoryRAG()
    
    # Run evaluation
    collector = run_evaluation_queries(rag, test_queries)
    
    # Print summary
    collector.print_summary()
    
    # Save results
    results_file = "evaluation_results.json"
    import json
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "metrics": collector.get_summary(),
            "raw_metrics": collector.metrics,
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

