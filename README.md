# Patent Intelligence MCP Server

An AI-powered Patent Intelligence Platform built using **Python**, **FastAPI**, **PostgreSQL**, **MCP (Model Context Protocol)**, and **LLMs** to analyze, search, summarize, and compare patent documents.

---

## Overview

Patent documents are often lengthy, highly technical, and written in legal language. Engineers, researchers, startups, and patent professionals spend significant time reading patents to understand existing inventions, identify prior art, and evaluate innovation opportunities.

This project aims to simplify that process by creating an AI-powered MCP server that can:

- Search patents
- Extract key claims
- Summarize technical inventions
- Find similar patents
- Perform prior-art analysis
- Compare patents
- Generate technology insights

The goal is to transform raw patent documents into structured, searchable, and actionable information.

---

## Problem Statement

Suppose an engineer develops a new invention:

> "A deep learning-based lunar landslide detection system using satellite imagery."

Before investing time and resources, they need answers to questions such as:

- Has a similar invention already been patented?
- Which organizations own related patents?
- What technologies are already protected?
- What claims exist in those patents?
- How different is the new invention from existing work?

Finding these answers manually requires reading dozens of patent documents, which is both time-consuming and difficult.

This platform automates much of that workflow using AI and semantic search.

---

## What is a Patent?

A patent is a legal document that grants exclusive rights to an inventor for a specific invention.

A patent typically contains:

- Title
- Abstract
- Background
- Detailed Description
- Drawings
- Claims

Among these sections, **Claims** are the most important because they legally define what is protected.

### Example Claim

Instead of:

> "I invented a drone."

A patent claim may state:

> "A drone comprising a camera, GPS module, and autonomous navigation system configured to monitor agricultural land."

This precise wording defines the scope of protection.

---

## Project Goal

Build an AI Patent Analyst capable of understanding patent documents and assisting users with patent research tasks.

### Example Interaction

#### Search Patents

User:

```text
Find patents related to lunar surface monitoring.
```

Tool:

```python
search_patents()
```

---

#### Summarize Patent

User:

```text
Summarize this patent.
```

Tool:

```python
summarize_patent()
```

Output:

```text
Problem:
Detect surface changes on planetary bodies.

Method:
Machine learning applied to orbital imagery.

Advantages:
Automated large-scale monitoring.
```

---

#### Extract Claims

User:

```text
What are the key claims?
```

Tool:

```python
extract_claims()
```

Output:

```text
Claim 1:
Method for detecting terrain changes using orbital imagery.

Claim 2:
Use of neural networks for classification.
```

---

#### Find Similar Patents

User:

```text
Find patents similar to this invention.
```

Tool:

```python
find_similar_patents()
```

The system uses semantic search and embeddings to locate related inventions.

---

## Core Features

### Patent Search

Search patents using keywords, topics, technologies, or invention descriptions.

```python
search_patents(query)
```

---

### Patent Summary Generation

Generate concise summaries from lengthy patent documents.

```python
summarize_patent(patent_id)
```

---

### Claim Extraction

Automatically identify and extract independent and dependent claims.

```python
extract_claims(patent_id)
```

---

### Similar Patent Discovery

Find semantically similar patents using vector embeddings.

```python
find_similar_patents(patent_id)
```

---

### Patent Comparison

Compare multiple patents and highlight similarities and differences.

```python
compare_patents(patent_a, patent_b)
```

---

### Prior Art Search

Search for existing patents related to a new invention.

```python
find_prior_art(invention_description)
```

---

### Technology Trend Analysis

Analyze innovation trends within a specific domain.

```python
technology_trends(domain)
```

Example:

```text
Technology: Wireless Charging

Output:
- Patent filings per year
- Top inventors
- Top companies
- Emerging technologies
```

---

## System Workflow

```text
Patent PDF
        │
        ▼
PDF Parsing
        │
        ▼
Text Cleaning
        │
        ▼
Chunking
        │
        ▼
LLM Processing
        │
 ┌──────┼──────┐
 ▼      ▼      ▼
Summary Claims Embeddings
        │
        ▼
PostgreSQL + pgvector
        │
        ▼
MCP Tools
        │
        ▼
User Queries
```

---

## MCP Tools

The MCP server will expose the following tools:

```python
search_patents()

get_patent()

summarize_patent()

extract_claims()

compare_patents()

find_prior_art()

find_similar_patents()

technology_trends()

company_portfolio()
```

---

## Technology Stack

### Backend

- Python
- FastAPI
- MCP SDK

### Database

- PostgreSQL
- SQLAlchemy
- Alembic

### Vector Search

- pgvector

### AI & NLP

- OpenAI API / Gemini API
- Embeddings
- Semantic Search

### Document Processing

- PyMuPDF
- pdfplumber

### Testing

- Pytest
- Pytest-Asyncio

### Logging

- Python Logging
- Structured JSON Logs

### Deployment

- Docker
- Docker Compose

---

## Database Design

### Patents Table

```sql
CREATE TABLE patents (
    id SERIAL PRIMARY KEY,
    patent_number TEXT,
    title TEXT,
    abstract TEXT,
    filing_date DATE,
    publication_date DATE
);
```

### Claims Table

```sql
CREATE TABLE claims (
    id SERIAL PRIMARY KEY,
    patent_id INTEGER,
    claim_number INTEGER,
    claim_text TEXT
);
```

### Inventors Table

```sql
CREATE TABLE inventors (
    id SERIAL PRIMARY KEY,
    name TEXT
);
```

### Embeddings Table

```sql
CREATE TABLE patent_embeddings (
    patent_id INTEGER,
    embedding VECTOR(1536)
);
```

---

## Future Enhancements

### Patent Family Analysis

Track parent-child patent relationships.

### Citation Network Visualization

Visualize references between patents.

### Multi-Language Patent Support

Support patents from multiple jurisdictions.

### Novelty Scoring

Estimate how unique a proposed invention is compared to existing patents.

### Company Patent Intelligence Dashboard

Analyze patent portfolios of organizations.

---

## Use Cases

### Researchers

- Literature and patent review
- Technology exploration

### Startups

- Prior-art analysis
- Innovation validation

### Patent Attorneys

- Claim review
- Patent comparison

### Engineering Teams

- Competitive intelligence
- Technology trend analysis

---

## Learning Outcomes

This project demonstrates:

- Backend Engineering
- API Design
- Database Design
- Vector Search
- Information Retrieval
- AI Integration
- PDF Processing
- Logging & Monitoring
- Testing
- MCP Development
- Software Architecture

---

## Vision

The long-term vision is to create an AI-powered Patent Intelligence Assistant capable of understanding patent documents, extracting valuable insights, identifying innovation opportunities, and accelerating patent research workflows.