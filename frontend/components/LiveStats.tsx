"use client"

import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"

interface SSEMetrics {
  total_frames?: number
  uptime_seconds?: number
  fps?: number
}

export function LiveStats() {
  const [metrics, setMetrics] = useState<SSEMetrics | null>(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const es = new EventSource("/api/metrics/sse")

    es.onopen = () => setConnected(true)

    es.onmessage = (event: MessageEvent<string>) => {
      try {
        const data = JSON.parse(event.data) as SSEMetrics
        setMetrics(data)
        setConnected(true)
      } catch {
        // ignore parse errors
      }
    }

    es.onerror = () => {
      setConnected(false)
    }

    return () => {
      es.close()
    }
  }, [])

  const formatUptime = (seconds?: number): string => {
    if (!seconds) return "0s"
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    if (h > 0) return `${h}h ${m}m ${s}s`
    if (m > 0) return `${m}m ${s}s`
    return `${s}s`
  }

  return (
    <Card className="border-border/50 w-full max-w-md">
      <CardContent className="pt-6">
        <div className="flex items-center gap-2 mb-4">
          <span
            className={`h-2 w-2 rounded-full ${
              connected ? "bg-green-500 animate-pulse" : "bg-muted-foreground"
            }`}
          />
          <span className="text-xs text-muted-foreground font-mono">
            {connected ? "Live" : "Connecting..."}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-4 font-mono text-sm">
          <div>
            <p className="text-muted-foreground text-xs">Frames Processed</p>
            <p className="text-foreground font-semibold text-lg">
              {metrics?.total_frames?.toLocaleString() ?? "—"}
            </p>
          </div>
          <div>
            <p className="text-muted-foreground text-xs">Uptime</p>
            <p className="text-foreground font-semibold text-lg">
              {metrics?.uptime_seconds !== undefined
                ? formatUptime(metrics.uptime_seconds)
                : "—"}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
