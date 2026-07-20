"use client"

import React, { useState, useEffect, useCallback } from "react"
import { format } from "date-fns"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import type { Session, TelemetrySnapshot } from "@/lib/types"

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_HTTP_URL ?? "https://cv-analytics-backend.fly.dev"

function formatDuration(startedAt: string, endedAt?: string): string {
  const start = new Date(startedAt).getTime()
  const end = endedAt ? new Date(endedAt).getTime() : Date.now()
  const diffMs = end - start
  const totalSeconds = Math.floor(diffMs / 1000)
  const h = Math.floor(totalSeconds / 3600)
  const m = Math.floor((totalSeconds % 3600) / 60)
  const s = totalSeconds % 60
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

export default function HistoryPage() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedSession, setSelectedSession] = useState<string | null>(null)
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot[]>([])
  const [page, setPage] = useState(0)

  const fetchSessions = useCallback(async (p: number) => {
    setLoading(true)
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/sessions?limit=20&offset=${p * 20}`
      )
      if (res.ok) {
        const data = (await res.json()) as Session[]
        setSessions(data)
      }
    } catch {
      // silently fail
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchSessions(page)
  }, [fetchSessions, page])

  const handleRowClick = async (sessionId: string) => {
    if (selectedSession === sessionId) {
      setSelectedSession(null)
      setTelemetry([])
      return
    }
    setSelectedSession(sessionId)
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/sessions/${sessionId}/telemetry`
      )
      if (res.ok) {
        const data = (await res.json()) as TelemetrySnapshot[]
        setTelemetry(data)
      }
    } catch {
      setTelemetry([])
    }
  }

  const handleExport = (sessionId: string) => {
    window.open(
      `${BACKEND_URL}/api/sessions/${sessionId}/export?format=csv`,
      "_blank"
    )
  }

  return (
    <div className="min-h-screen bg-background text-foreground p-6">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">Session History</h1>
          <Badge variant="secondary">Page {page + 1}</Badge>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Sessions</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8 text-muted-foreground text-sm">
                Loading sessions...
              </div>
            ) : sessions.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground text-sm">
                No sessions found.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-muted-foreground text-xs uppercase tracking-wider">
                      <th className="text-left py-2 pr-4">Start Time</th>
                      <th className="text-left py-2 pr-4">Duration</th>
                      <th className="text-left py-2 pr-4">Source</th>
                      <th className="text-right py-2 pr-4">Frames</th>
                      <th className="text-right py-2 pr-4">Detections</th>
                      <th className="text-left py-2">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sessions.map((session) => (
                      <React.Fragment key={session.id}>
                        <tr
                          className="border-b border-border/50 hover:bg-accent/30 cursor-pointer transition-colors"
                          onClick={() => void handleRowClick(session.id)}
                        >
                          <td className="py-2 pr-4 font-mono text-xs">
                            {format(
                              new Date(session.started_at),
                              "yyyy-MM-dd HH:mm:ss"
                            )}
                          </td>
                          <td className="py-2 pr-4 text-xs text-muted-foreground">
                            {formatDuration(session.started_at, session.ended_at)}
                          </td>
                          <td className="py-2 pr-4">
                            <Badge variant="outline" className="text-xs">
                              {session.source_type}
                            </Badge>
                          </td>
                          <td className="py-2 pr-4 text-right font-mono text-xs">
                            {session.total_frames.toLocaleString()}
                          </td>
                          <td className="py-2 pr-4 text-right font-mono text-xs">
                            {session.total_detections.toLocaleString()}
                          </td>
                          <td className="py-2">
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-6 text-xs"
                              onClick={(e) => {
                                e.stopPropagation()
                                handleExport(session.id)
                              }}
                            >
                              Export
                            </Button>
                          </td>
                        </tr>
                        {selectedSession === session.id && (
                          <tr className="bg-accent/20">
                            <td colSpan={6} className="py-4 px-4">
                              <p className="text-xs text-muted-foreground mb-2">
                                FPS over time
                              </p>
                              {telemetry.length > 0 ? (
                                <ResponsiveContainer width="100%" height={120}>
                                  <LineChart data={telemetry}>
                                    <XAxis
                                      dataKey="captured_at"
                                      tickFormatter={(v: string) =>
                                        format(new Date(v), "HH:mm:ss")
                                      }
                                      tick={{ fontSize: 9 }}
                                    />
                                    <YAxis tick={{ fontSize: 9 }} />
                                    <Tooltip
                                      contentStyle={{ fontSize: 11 }}
                                    />
                                    <Line
                                      type="monotone"
                                      dataKey="fps"
                                      stroke="#22c55e"
                                      dot={false}
                                      strokeWidth={1.5}
                                    />
                                  </LineChart>
                                </ResponsiveContainer>
                              ) : (
                                <p className="text-xs text-muted-foreground">
                                  No telemetry data available.
                                </p>
                              )}
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Pagination */}
            <div className="flex items-center justify-end gap-2 mt-4">
              <Button
                size="sm"
                variant="outline"
                className="h-7 text-xs"
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
              >
                Previous
              </Button>
              <span className="text-xs text-muted-foreground">Page {page + 1}</span>
              <Button
                size="sm"
                variant="outline"
                className="h-7 text-xs"
                disabled={sessions.length < 20}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
