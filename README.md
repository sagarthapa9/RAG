# Business RAG System

A learning project to build a production-ready Retrieval-Augmented Generation (RAG) system for business document analysis.

## 🎯 Project Goals

- **Learn RAG fundamentals**: Embeddings, vector databases, retrieval, LLM integration
- **Build step-by-step**: Start simple, add complexity incrementally  
- **Production-ready**: Docker containers, proper testing, scalable architecture
- **Real business value**: Document Q&A, contract analysis, policy queries

## 🏗️ Current Status

**✅ Phase 1: Document Processing**
- Basic document reader (PDF, DOCX, TXT)
- Docker development environment
- Project structure and tooling

**🔄 Next: Text Chunking & Preprocessing**  
**📋 Later: Embeddings → Vector Storage → LLM Integration → UI**

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- uv (Python package manager)
- Business documents for testing

### Setup
```bash
# Clone/create project
mkdir rag-system && cd uc rag-system

# Install uv (Windows)
irm https://astral.sh/uv/install.ps1 | iex

# Start development environment
docker-compose up -d rag-dev

# Test document reader
docker-compose exec rag-dev python test_reader.py
```

## 📁 Project Structure

```
business-rag-system/
├── src/rag/          # Main source code
├── tests/                     # Unit tests
├── documents/                 # 📋 Add your business docs here
├── data/                      # Processed data
├── docker-compose.yml         # Development environment
└── test_reader.py            # Main test script
```

## 🧪 Development Workflow

```bash
# Enter development container
docker-compose exec rag-dev bash

# Run tests
python test_reader.py

# Add new documents to documents/ folder and test
```

## 📚 Learning Path

1. **Document Processing** ← We are here
2. **Text Chunking** - Break documents into searchable pieces
3. **Embeddings** - Convert text to vector representations
4. **Vector Storage** - Store and search embeddings efficiently
5. **Retrieval** - Find relevant document chunks
6. **LLM Integration** - Generate answers from retrieved context
7. **UI Development** - User interface for document Q&A
8. **Production Deployment** - Scale and monitor the system

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **Package Management**: uv
- **Containerization**: Docker + Docker Compose
- **Document Processing**: PyPDF2, python-docx
- **Vector DB**: ChromaDB (development) → Pinecone (production)
- **Embeddings**: sentence-transformers → OpenAI
- **LLM**: OpenAI GPT-4 / Local models via Ollama
- **Framework**: LangChain
- **UI**: Streamlit
- **Testing**: pytest

## 📋 Next Steps

1. Add business documents to `documents/` folder
2. Test document reader with your files
3. Move to text chunking phase
4. Add embeddings generation

## 🤝 Learning Approach

This project follows a **step-by-step learning methodology**:
- Build one component at a time
- Test each component thoroughly  
- Understand the theory behind each step
- Focus on production-ready patterns
- Real business use cases throughout

Each phase adds new capabilities while maintaining what we've already built.

---

**Status**: 🏗️ In Development - Phase 1 (Document Processing)  
**Next**: Text Chunking and Preprocessing