# 🤖 AI Customer Support Agent Platform

> **Enterprise-grade multi-agent customer support system** — RAG + Agentic AI + MLOps + Production Infra

[![CI/CD](https://github.com/your-org/ai-support-platform/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-org/ai-support-platform/actions/workflows/ci-cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A full-stack AI platform that handles customer support end-to-end: a **LangGraph multi-agent pipeline** routes, answers, and escalates queries using **RAG over Pinecone**, grounded by **Groq (Llama 3)**, tracked with **MLflow**, and deployed to **Kubernetes** via a complete **GitHub Actions CI/CD** pipeline.

---

## Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │            FastAPI Application               │
                     │  POST /api/v1/chat · JWT Auth · /metrics     │
                     └─────────────────────┬───────────────────────┘
                                           │
                     ┌─────────────────────▼───────────────────────┐
                     │          LangGraph State Machine             │
                     │                                              │
                     │   ┌──────────────────────────────────────┐  │
                     │   │           Router Agent               │  │
                     │   │  (LLM classifier → JSON route)       │  │
                     │   └────┬──────────┬───────────┬──────────┘  │
                     │        │          │           │             │
                     │   ┌────▼──┐  ┌───▼───┐  ┌───▼────┐        │
                     │   │Knowl- │  │Support│  │Billing │        │
                     │   │edge   │  │Agent  │  │Agent   │        │
                     │   │Agent  │  │RAG +  │  │RAG +   │        │
                     │   │(RAG)  │  │Tools  │  │Tools   │        │
                     │   └────┬──┘  └───┬───┘  └───┬────┘        │
                     │        └──────────┴──────────┘             │
                     │                  │                          │
                     │        ┌─────────▼──────────┐              │
                     │        │    Quality Gate     │              │
                     │        │  confidence < 0.6?  │              │
                     │        └─────────┬───────────┘              │
                     │           ok ▼       ▼ low conf             │
                     │          [END]  ┌────▼────────┐             │
                     │                 │ Escalation  │             │
                     │                 │   Agent     │             │
                     │                 │(human hdoff)│             │
                     │                 └─────────────┘             │
                     └─────────────────────────────────────────────┘
                                           │
     ┌─────────────────────────────────────┼──────────────────────────────────┐
     │                                     │                                  │
┌────▼───────────────┐         ┌───────────▼──────────┐          ┌───────────▼──────┐
│   PostgreSQL 16    │         │  Pinecone (Vector DB) │          │  MLflow Tracking │
│  Users · Tickets  │         │  RAG Retrieval        │          │  Run logging     │
│  Messages · Runs  │         │  all-MiniLM-L6-v2     │          │  Offline eval    │
└────────────────────┘         └──────────────────────┘          └──────────────────┘
```

### Request lifecycle

```
User → POST /api/v1/chat
  │
  ├─ JWT auth (deps.py)
  ├─ Fetch ticket + conversation history (PostgreSQL)
  ├─ run_agent_pipeline(ticket_id, message, history)
  │    ├─ Router classifies intent → route ∈ {knowledge, support, billing, escalation}
  │    ├─ Specialist agent: embed query → Pinecone top-k → Groq completion
  │    ├─ Quality gate: confidence check → auto-escalate if low
  │    └─ Returns AgentState {response, agent_used, confidence, trace, latency_ms}
  ├─ Persist agent message to conversation (PostgreSQL)
  ├─ Background task: log AgentRun to MLflow
  └─ Return ChatResponse JSON
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI 0.115, Pydantic v2, Uvicorn |
| **Agentic AI** | LangGraph 0.2 (state machine), LangChain 0.3 |
| **LLM** | Groq (`llama-3.3-70b-versatile`, fallback `llama-3.1-8b-instant`) |
| **Vector DB** | Pinecone (serverless) |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` (384-dim) |
| **Database** | PostgreSQL 16 + SQLAlchemy 2.0 |
| **MLOps** | MLflow 2.17 — run tracking, offline eval, prompt versioning |
| **Observability** | structlog (structured JSON logs), Prometheus `/metrics`, Grafana |
| **Auth** | JWT (python-jose) + bcrypt password hashing |
| **Containerisation** | Docker (multi-stage build), docker-compose |
| **Orchestration** | Kubernetes — Deployment, Service, ConfigMap, Secret, HPA, Ingress |
| **CI/CD** | GitHub Actions — lint → test → eval gate → build → push → deploy |

---

## Project Structure

```
ai-support-platform/
├── app/
│   ├── agents/              # LangGraph multi-agent system
│   │   ├── state.py         #   shared TypedDict graph state
│   │   ├── orchestrator.py  #   graph wiring + entry point
│   │   ├── router_agent.py  #   LLM intent classifier
│   │   ├── knowledge_agent.py  # RAG Q&A
│   │   ├── support_agent.py    # technical troubleshooting + tools
│   │   ├── billing_agent.py    # billing/refunds + tools
│   │   ├── escalation_agent.py # human handoff
│   │   ├── quality_gate.py     # auto-escalation on low confidence
│   │   └── tools.py            # simulated business tool calls
│   ├── api/                 # FastAPI routers
│   │   ├── auth_routes.py   #   register / login
│   │   ├── chat_routes.py   #   POST /chat — main agent endpoint
│   │   ├── ticket_routes.py #   CRUD tickets
│   │   ├── kb_routes.py     #   knowledge-base ingestion (admin)
│   │   ├── health_routes.py #   /health  /ready  (K8s probes)
│   │   └── deps.py          #   JWT auth dependency
│   ├── core/                # config, logging, security
│   ├── db/                  # SQLAlchemy models + session
│   ├── rag/                 # embeddings, chunking, Pinecone, retriever
│   ├── schemas/             # Pydantic request/response models
│   └── services/            # LLM client, MLflow logger, eval, prompt registry
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml         # ⚠ template only — do not commit real values
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── ingress.yaml
├── .github/workflows/
│   └── ci-cd.yml            # lint → test → eval → build → push → deploy
├── scripts/
│   └── run_offline_eval.py  # golden-dataset regression eval + MLflow logging
├── tests/
│   └── golden_eval_dataset.json
├── infra/
│   └── prometheus.yml
├── Dockerfile               # multi-stage: builder + runtime, non-root user
├── docker-compose.yml       # full local stack: API + PG + MLflow + Prometheus + Grafana
├── requirements.txt
└── .env.example
```

---

## Quick Start (Local)

### Option A — docker-compose (recommended)

```bash
git clone https://github.com/your-org/ai-support-platform.git
cd ai-support-platform

# 1. Configure secrets
cp .env .env
# Edit .env: set GROQ_API_KEY and PINECONE_API_KEY

# 2. Start the full stack
docker compose up --build

# 3. Open API docs
open http://localhost:8000/docs

# Other services:
# MLflow:     http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

### Option B — bare Python

```bash
pip install -r requirements.txt
cp .env .env   # fill in keys

uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

---

## Key API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create user account |
| `POST` | `/api/v1/auth/login` | Get JWT access token |
| `POST` | `/api/v1/tickets` | Open a support ticket |
| `GET` | `/api/v1/tickets` | List your tickets |
| `POST` | `/api/v1/chat` | Send message → full agent pipeline |
| `POST` | `/api/v1/knowledge-base/ingest` | Ingest document into Pinecone (admin) |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (checks DB) |
| `GET` | `/metrics` | Prometheus metrics |

---

## MLOps — Offline Evaluation

```bash
python scripts/run_offline_eval.py
```

Runs 8 golden queries through the full agent pipeline and scores:

| Metric | Threshold |
|---|---|
| Routing accuracy | ≥ 70% (CI gate — exits non-zero if below) |
| Keyword coverage | logged per-query |
| Average confidence | logged |
| Average latency | logged |

All results are logged as an MLflow run. This script is wired into the CI/CD pipeline as a regression gate before any deployment.

---

## CI/CD Pipeline

```
Push to main
    │
    ├── lint          ruff check + format, mypy on core/schemas
    ├── test          pytest (unit + integration) against live PG service
    ├── eval          offline golden-dataset eval — routing accuracy gate
    ├── build         Docker multi-stage build → push to GHCR (SHA tag + latest)
    └── deploy        kubectl set image → rollout → smoke test /health + /ready
```

All jobs run in parallel where possible. The deploy job targets the `production` environment (add GitHub environment protection rules for manual approval if needed).

---

## Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml      # fill real values first!
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Watch rollout
kubectl rollout status deployment/ai-support-api -n ai-support

# Check pods
kubectl get pods -n ai-support
```

The HPA scales between **2–8 replicas** based on CPU (≥65%) and memory (≥75%) utilisation, with a 5-minute scale-down stabilisation window to avoid churn.

---

## Required Secrets (GitHub → Settings → Secrets)

| Secret | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (used in eval job) |
| `PINECONE_API_KEY` | Pinecone API key (used in eval job) |
| `KUBECONFIG` | base64-encoded kubeconfig for the target cluster |

---

## Roadmap

- [ ] Alembic migrations (replace `create_all`)
- [ ] Streaming responses (`StreamingResponse` + SSE)
- [ ] Webhook endpoint for real-time human-agent handoff
- [ ] Helm chart for multi-environment deploys
- [ ] External Secrets Operator / Vault Sidecar integration

---

## License

MIT — see [LICENSE](LICENSE).
