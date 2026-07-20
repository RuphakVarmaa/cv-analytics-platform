# CV Analytics Platform

A real-time computer vision analytics platform that streams video frames through a YOLOv8 ONNX inference pipeline and surfaces live object-detection telemetry — FPS, latency percentiles, class counts — in a Next.js dashboard over WebSocket.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT BROWSER                             │
│                     Next.js (React + WebSocket)                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTPS / WSS
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Vercel CDN / Edge                            │
│                    (Static assets + SSR pages)                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ WSS (WebSocket upgrade)
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                   Fly.io  ─  Backend Service                      │
│                     FastAPI  +  Uvicorn                           │
│                                                                   │
│   ┌─────────────────┐        ┌──────────────────────────────┐    │
│   │  WebSocket Hub  │        │  REST  /sessions  /telemetry │    │
│   │  (frame relay)  │        │  (history, auth, user API)   │    │
│   └────────┬────────┘        └──────────────┬───────────────┘    │
│            │ HTTP POST (frames)              │                    │
└────────────┼────────────────────────────────┼────────────────────┘
             │                                │
             ▼                                ▼
┌────────────────────────┐      ┌─────────────────────────────────┐
│  Fly.io  ─  ML Service │      │  Fly.io  ─  PostgreSQL          │
│  YOLOv8n ONNX Runtime  │      │  users, inference_sessions,     │
│  (FastAPI + onnxruntime│      │  telemetry_snapshots            │
│   port 8001)           │      └─────────────────────────────────┘
└────────────────────────┘
                                ┌─────────────────────────────────┐
                                │  Fly.io  ─  Redis               │
                                │  Session pub/sub, rate-limit     │
                                │  counters, ephemeral queue       │
                                └─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         CI / CD                                     │
│                                                                     │
│   GitHub ──► GitHub Actions ──► test-backend  (pytest + ruff)      │
│                             ├── test-frontend (tsc + vitest)        │
│                             └── deploy                             │
│                                  ├── flyctl deploy  (backend)      │
│                                  ├── flyctl deploy  (ml-service)   │
│                                  └── vercel deploy  (frontend)     │
└─────────────────────────────────────────────────────────────────────┘
```

## Live Demo

| Service | URL | Status |
|---------|-----|--------|
| **Frontend (Vercel)** | https://frontend-nu-three-92.vercel.app | ✅ Live |
| **GitHub Repository** | https://github.com/RuphakVarmaa/cv-analytics-platform | ✅ Public |
| **Backend API (Fly.io)** | https://cv-analytics-backend.fly.dev | ⏳ Deploy pending |
| **ML Service (Fly.io)** | https://cv-ml-service.fly.dev/health | ⏳ Deploy pending |

**Demo mode works without login** — open https://frontend-nu-three-92.vercel.app/dashboard, select **Demo** source, and click **Connect**. Live bounding box overlay and all 4 charts are functional once the Fly.io backend is deployed.

## Performance Benchmarks

| Metric | CPU (YOLOv8n ONNX) | GPU (YOLOv8m) |
|--------|---------------------|----------------|
| p50 latency | 18 ms | 6 ms |
| p95 latency | 38 ms | 12 ms |
| p99 latency | 52 ms | 18 ms |
| Throughput | 28 FPS | 60+ FPS |

*Measured on Fly.io shared-4x CPU (4 vCPU, 4 GB RAM)*

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend framework | Next.js 14 (App Router) |
| UI components | Tailwind CSS + shadcn/ui |
| Auth | NextAuth.js (GitHub OAuth) |
| Real-time transport | Native WebSocket API |
| Backend framework | FastAPI 0.111 |
| ASGI server | Uvicorn + Gunicorn |
| ML inference | YOLOv8n ONNX via onnxruntime-cpu |
| Task queue | Redis pub/sub + asyncio queues |
| Database | PostgreSQL 16 (Fly.io Postgres) |
| Cache / ephemeral store | Redis 7 (Fly.io Redis) |
| Container runtime | Docker (multi-stage builds) |
| Frontend hosting | Vercel (Edge Network) |
| Backend hosting | Fly.io (shared-cpu-4x) |
| CI/CD | GitHub Actions |
| Linting (Python) | Ruff + mypy |
| Testing (Python) | pytest + pytest-asyncio + httpx |
| Testing (JS) | Vitest + Testing Library |

## Local Development

### Prerequisites

- Docker Desktop 4.x+
- Node.js 20+ and npm
- Python 3.11+ (optional — Docker handles this)
- A GitHub OAuth App (for auth; demo mode works without it)

### Start all services with Docker Compose

```bash
git clone https://github.com/RuphakVarmaa/cv-analytics-platform.git
cd cv-analytics-platform

# Copy and edit backend env
cp backend/.env.example backend/.env

# Pull images and build containers (first run takes ~3 min)
docker compose up --build
```

Services will be available at:

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API + WS | http://localhost:8000 |
| ML Service | http://localhost:8001 |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

The schema is applied automatically by Docker Compose via `infra/schema.sql` on first startup.

### Run backend tests locally

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest pytest-asyncio pytest-cov httpx ruff mypy

export DATABASE_URL=postgresql://cv_user:cv_password@localhost:5432/cv_analytics
export REDIS_URL=redis://localhost:6379
export JWT_SECRET=dev-secret
export ML_SERVICE_URL=http://localhost:8001

ruff check app/
mypy app/ --ignore-missing-imports
pytest tests/ --cov=app --cov-report=term-missing
```

### Run frontend tests locally

```bash
cd frontend
npm ci
npx tsc --noEmit
npm test -- --run
npm run build
```

### Environment variables

Copy `backend/.env.example` to `backend/.env` and fill in values:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `ML_SERVICE_URL` | Internal URL of the ML inference service |
| `JWT_SECRET` | Secret for signing JWT tokens |
| `LOG_LEVEL` | Logging verbosity (`INFO`, `DEBUG`, `WARNING`) |

Frontend env vars (set in Vercel dashboard or `.env.local`):

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_BACKEND_WS_URL` | WebSocket URL of the backend |
| `NEXT_PUBLIC_BACKEND_HTTP_URL` | HTTP URL of the backend |
| `BACKEND_URL` | Server-side backend URL (SSR calls) |
| `NEXTAUTH_URL` | Canonical URL of the frontend |
| `NEXTAUTH_SECRET` | NextAuth signing secret |
| `GITHUB_CLIENT_ID` | GitHub OAuth App client ID |
| `GITHUB_CLIENT_SECRET` | GitHub OAuth App client secret |

## Deployment

### 1. Install Fly CLI and log in

```bash
brew install flyctl
fly auth login
```

### 2. Create Fly apps

```bash
fly apps create cv-analytics-backend
fly apps create cv-ml-service
```

### 3. Provision Fly Postgres and Redis

```bash
fly postgres create --name cv-analytics-db --region iad
fly postgres attach cv-analytics-db --app cv-analytics-backend

fly redis create --name cv-analytics-redis --region iad
# Note the connection string and set it as a secret below
```

Apply the schema after provisioning:

```bash
fly postgres connect -a cv-analytics-db < infra/schema.sql
```

### 4. Set backend secrets

```bash
fly secrets set \
  JWT_SECRET="<strong-random-secret>" \
  REDIS_URL="redis://<fly-redis-host>:6379" \
  ML_SERVICE_URL="https://cv-ml-service.fly.dev" \
  --app cv-analytics-backend
```

### 5. Deploy backend and ML service

```bash
cd backend  && fly deploy --remote-only --app cv-analytics-backend
cd ../ml    && fly deploy --remote-only --app cv-ml-service
```

### 6. Deploy frontend to Vercel

```bash
npm i -g vercel
cd frontend
vercel --prod
```

Set the following environment variables in the Vercel dashboard under **Settings → Environment Variables**:

- `NEXT_PUBLIC_BACKEND_WS_URL` = `wss://cv-analytics-backend.fly.dev`
- `NEXT_PUBLIC_BACKEND_HTTP_URL` = `https://cv-analytics-backend.fly.dev`
- `BACKEND_URL` = `https://cv-analytics-backend.fly.dev`
- `NEXTAUTH_SECRET` = `<strong-random-secret>`
- `NEXTAUTH_URL` = `https://cv-analytics-platform.vercel.app`
- `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` from your GitHub OAuth App

### 7. Add GitHub Actions secrets

In your repo under **Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|--------|-------|
| `FLY_API_TOKEN` | From `fly tokens create deploy` |
| `VERCEL_TOKEN` | From Vercel → Account Settings → Tokens |
| `VERCEL_ORG_ID` | From `.vercel/project.json` after `vercel link` |
| `VERCEL_PROJECT_ID` | From `.vercel/project.json` after `vercel link` |

## CI/CD

The GitHub Actions pipeline at `.github/workflows/ci-cd.yml` runs on every push and pull request to `main`:

1. **test-backend** — spins up Postgres 16 and Redis 7 as service containers, installs dependencies, runs `ruff` lint, `mypy` type check, and `pytest` with 75% coverage gate.
2. **test-frontend** — installs Node 20 deps, runs `tsc --noEmit`, Vitest, and a production `next build`.
3. **deploy** (main branch only, after both test jobs pass) — deploys backend and ML service to Fly.io via `flyctl`, then deploys frontend to Vercel via the Vercel CLI.

## Project Structure

```
cv-analytics-platform/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions CI/CD pipeline
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py          # Settings via pydantic-settings
│   │   ├── models.py          # SQLAlchemy / raw SQL models
│   │   ├── middleware/
│   │   │   ├── auth.py        # JWT verification middleware
│   │   │   └── rate_limit.py  # Redis-backed rate limiting
│   │   ├── routers/           # FastAPI route handlers
│   │   └── services/
│   │       ├── frame_capture.py   # WebSocket frame ingestion
│   │       └── queue_manager.py   # Async frame queue
│   ├── tests/
│   ├── .env.example
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/                   # Next.js App Router pages
│   ├── components/            # React components
│   ├── lib/                   # Utilities, API clients
│   ├── types/                 # TypeScript type definitions
│   ├── Dockerfile
│   └── package.json
├── ml/
│   ├── app/
│   │   ├── main.py            # FastAPI inference service
│   │   └── model.py           # YOLOv8n ONNX wrapper
│   ├── tests/
│   ├── Dockerfile
│   ├── fly.toml
│   └── requirements.txt
├── infra/
│   ├── schema.sql             # PostgreSQL DDL (tables + indexes)
│   └── redis.conf             # Redis memory / persistence config
├── docker-compose.yml         # Local development stack
├── .gitignore
└── README.md
```
