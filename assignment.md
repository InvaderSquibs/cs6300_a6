# ReAct, Tools, and Retrieval: Pair Project

**Due:** Monday by 5:30pm  
**Points:** 200  
**Submitting:** File upload (PDF)  
**Weight:** 200 points per student  
**Duration:** 2 weeks  
**Teams:** Pairs (2 students per team)

## Overview

This two-week project asks each pair to design and implement a multi-component agent that combines the three central ideas of the course so far:

1. **ReAct reasoning**: an agent that reasons and acts iteratively
2. **Tool use**: purposeful calls to external or custom functions
3. **Retrieval**: incorporating external knowledge or context into reasoning

Your agent should tackle a larger-scope task than previous individual assignments. Ideally something requiring multiple coordinated steps, memory or retrieval, and non-trivial tool use. Examples include domain-specific research assistants, multi-stage data explorers, scheduling or planning systems, or multi-source Q&A pipelines.

## Learning Goals

- **CLO 1**: Apply the Agent–Environment model to a complex multi-tool setting.
- **CLO 2**: Implement a complete AI pipeline integrating reasoning, tool execution, and retrieval.
- **CLO 3**: Create an agentic AI solution demonstrating collaboration, modularity, and evaluation.

## Requirements

1. Implement a ReAct-style agent (using smolagents, LangChain, or similar) that can reason, call tools, and incorporate retrieved context.

2. Provide at least two tools, one of which performs retrieval (e.g., from a local vector store, API, or document set).

3. Demonstrate a clear environment–agent–environment loop, showing percepts, actions, and outcomes.

4. Include evaluation of agent success (quantitative or qualitative).

5. Document design and results in a short report (~ 3 pages).

## Deliverables

1. **Code Repository** — runnable agent, requirements, and examples/tests.
2. **Design & Evaluation Report (PDF)** — PEAS analysis, architecture diagram, tool contracts, evaluation results, and reflection on team roles and collaboration.

## Evaluation Rubric (200 pts per student)

| Component | Points | Description |
|-----------|--------|-------------|
| PEAS Analysis & Design | 40 | Clear modeling of environment, agent architecture, and tool contracts |
| Implementation & Functionality | 70 | Working ReAct agent using tools + retrieval; robust error handling |
| Evaluation & Results | 40 | Meaningful tests, metrics, or demonstrations of agent performance |
| Report & Presentation | 30 | Well-organized writeup; figures / logs / discussion; clarity of reasoning |
| Collaboration & Code Quality | 20 | Evidence of balanced team contribution and clean, reproducible code |
