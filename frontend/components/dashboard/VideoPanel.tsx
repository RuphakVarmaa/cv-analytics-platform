"use client"

import { useRef, useEffect } from "react"
import { useStore } from "@/lib/store"
import { getClassColor } from "@/lib/colors"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

export function VideoPanel() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const {
    currentDetections,
    lastFrameTimestamp,
    connectionStatus,
    source,
    isPaused,
    setIsPaused,
  } = useStore()

  // Draw detections on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Sync canvas size to display size
    const container = containerRef.current
    if (container) {
      canvas.width = container.offsetWidth
      canvas.height = container.offsetHeight
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (source === "demo" && connectionStatus !== "connected") {
      ctx.fillStyle = "#111827"
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = "#6b7280"
      ctx.font = "bold 20px Inter, sans-serif"
      ctx.textAlign = "center"
      ctx.textBaseline = "middle"
      ctx.fillText("DEMO MODE", canvas.width / 2, canvas.height / 2)
      return
    }

    if (connectionStatus !== "connected") return

    // Bboxes are normalized 0-1; scale directly to canvas dimensions
    for (const det of currentDetections) {
      const color = getClassColor(det.class_name)
      const x = det.bbox.x1 * canvas.width
      const y = det.bbox.y1 * canvas.height
      const w = (det.bbox.x2 - det.bbox.x1) * canvas.width
      const h = (det.bbox.y2 - det.bbox.y1) * canvas.height

      ctx.strokeStyle = color
      ctx.lineWidth = Math.max(1, det.confidence * 3)
      ctx.strokeRect(x, y, w, h)

      // Label background
      const label = `${det.class_name} ${Math.round(det.confidence * 100)}%`
      ctx.font = "12px Inter, sans-serif"
      const textWidth = ctx.measureText(label).width
      const labelHeight = 16
      const labelY = y > labelHeight + 2 ? y - labelHeight - 2 : y + 2

      ctx.fillStyle = color.replace("hsl", "hsla").replace(")", ", 0.75)")
      ctx.fillRect(x, labelY, textWidth + 6, labelHeight)

      ctx.fillStyle = "#ffffff"
      ctx.textBaseline = "top"
      ctx.fillText(label, x + 3, labelY + 2)
    }

    // Bottom-left: timestamp
    if (lastFrameTimestamp) {
      const d = new Date(lastFrameTimestamp)
      const ts = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}.${String(d.getMilliseconds()).padStart(3, "0")}`
      ctx.font = "11px Inter, sans-serif"
      ctx.fillStyle = "rgba(156, 163, 175, 0.9)"
      ctx.textBaseline = "bottom"
      ctx.textAlign = "left"
      ctx.fillText(ts, 8, canvas.height - 8)
    }

    // Top-right: resolution
    ctx.font = "11px Inter, sans-serif"
    ctx.fillStyle = "rgba(156, 163, 175, 0.9)"
    ctx.textBaseline = "top"
    ctx.textAlign = "right"
    ctx.fillText("1280×720", canvas.width - 8, 8)
  }, [currentDetections, lastFrameTimestamp, connectionStatus, source])

  // Handle resize
  useEffect(() => {
    const observer = new ResizeObserver(() => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (canvas && container) {
        canvas.width = container.offsetWidth
        canvas.height = container.offsetHeight
      }
    })
    if (containerRef.current) observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  return (
    <div className="flex flex-col h-full">
      <div
        ref={containerRef}
        className="relative flex-1 bg-gray-950 rounded-t border border-border overflow-hidden"
      >
        <canvas ref={canvasRef} className="w-full h-full block" />

        {/* Disconnected overlay */}
        {connectionStatus !== "connected" && source !== "demo" && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80">
            <p className="text-gray-400 text-sm text-center px-4">
              Not Connected — Select a source and click Connect
            </p>
          </div>
        )}

        {/* PAUSED badge */}
        {isPaused && (
          <Badge
            variant="secondary"
            className="absolute top-2 left-2 text-xs bg-black/60 text-amber-400 border-amber-500/30"
          >
            PAUSED
          </Badge>
        )}

        {/* Pause/Resume button */}
        {connectionStatus === "connected" && (
          <Button
            size="sm"
            variant="secondary"
            className="absolute bottom-8 right-2 h-7 text-xs opacity-80 hover:opacity-100"
            onClick={() => setIsPaused(!isPaused)}
          >
            {isPaused ? "Resume" : "Pause"}
          </Button>
        )}
      </div>

      {/* Bottom stats bar */}
      <div className="flex items-center gap-4 px-3 py-1.5 bg-card border border-t-0 border-border rounded-b text-xs font-mono text-muted-foreground">
        <span>
          Frame:{" "}
          {lastFrameTimestamp
            ? String(lastFrameTimestamp).slice(-8)
            : "--------"}
        </span>
        <span>Detections: {currentDetections.length}</span>
      </div>
    </div>
  )
}
