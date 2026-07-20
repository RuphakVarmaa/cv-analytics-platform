"use client"

import { useRef, useEffect, useCallback } from "react"
import { useSession } from "next-auth/react"
import { toast } from "sonner"
import { useStore } from "@/lib/store"
import { WebSocketManager, buildWsUrl } from "@/lib/websocket"
import { TopBar } from "@/components/dashboard/TopBar"
import { VideoPanel } from "@/components/dashboard/VideoPanel"
import { ChartsPanel } from "@/components/dashboard/ChartsPanel"
import type { FrameResult, MetricsUpdate, QueueStatus } from "@/lib/types"
import type { ConnectionStatus } from "@/lib/store"

export default function DashboardPage() {
  const { data: session } = useSession()
  const wsManagerRef = useRef<WebSocketManager | null>(null)
  const {
    connectionStatus,
    source,
    sourceUrl,
    setConnectionStatus,
    addDetectionResult,
    updateMetrics,
  } = useStore()

  const handleConnect = useCallback(() => {
    if (connectionStatus === "connected" || connectionStatus === "connecting") {
      wsManagerRef.current?.disconnect()
      setConnectionStatus("disconnected")
      wsManagerRef.current = null
      return
    }

    const token = (session as { accessToken?: string } | null)?.accessToken ?? ""

    setConnectionStatus("connecting")

    const onFrameResult = (result: FrameResult) => {
      addDetectionResult(result)
    }

    const onMetricsUpdate = (metrics: MetricsUpdate) => {
      updateMetrics(metrics)
    }

    const onQueueStatus = (status: QueueStatus) => {
      if (status.warning) {
        toast.warning(`Queue depth warning: ${status.depth} items pending`)
      }
    }

    const onStatusChange = (status: ConnectionStatus) => {
      setConnectionStatus(status)
      if (status === "error") {
        toast.error("WebSocket connection error")
      }
    }

    const url = buildWsUrl(source, sourceUrl, token)
    const manager = new WebSocketManager(
      url,
      onFrameResult,
      onMetricsUpdate,
      onQueueStatus,
      onStatusChange
    )
    wsManagerRef.current = manager
    manager.connect()
  }, [
    connectionStatus,
    session,
    source,
    sourceUrl,
    setConnectionStatus,
    addDetectionResult,
    updateMetrics,
  ])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsManagerRef.current?.disconnect()
    }
  }, [])

  return (
    <div className="flex flex-col h-screen bg-background overflow-hidden">
      <TopBar onConnect={handleConnect} />
      <main className="flex flex-1 overflow-hidden">
        <div className="w-2/5 flex flex-col p-2 border-r border-border">
          <VideoPanel />
        </div>
        <div className="w-3/5 overflow-y-auto">
          <ChartsPanel />
        </div>
      </main>
    </div>
  )
}
