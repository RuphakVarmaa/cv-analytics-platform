-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE,
    github_id TEXT UNIQUE,
    github_username TEXT,
    avatar_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

-- Inference sessions
CREATE TABLE IF NOT EXISTS inference_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('webcam', 'rtsp', 'upload', 'demo')),
    source_url TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    total_frames INTEGER DEFAULT 0,
    dropped_frames INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0,
    avg_fps FLOAT,
    avg_latency_ms FLOAT,
    is_deleted BOOLEAN DEFAULT FALSE
);

-- Telemetry snapshots (1 per second per session)
CREATE TABLE IF NOT EXISTS telemetry_snapshots (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES inference_sessions(id) ON DELETE CASCADE,
    captured_at TIMESTAMPTZ DEFAULT NOW(),
    fps FLOAT,
    p50_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    queue_depth INTEGER,
    drop_rate FLOAT,
    class_counts JSONB DEFAULT '{}',
    total_detections_snapshot INTEGER DEFAULT 0
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_telemetry_session_time
    ON telemetry_snapshots(session_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON inference_sessions(user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_active
    ON inference_sessions(ended_at) WHERE ended_at IS NULL;
