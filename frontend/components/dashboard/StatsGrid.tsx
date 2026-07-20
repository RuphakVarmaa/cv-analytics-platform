"use client"

import { useStore } from "@/lib/store"
import { Card, CardContent } from "@/components/ui/card"

interface StatCellProps {
  label: string
  value: string
}

function StatCell({ label, value }: StatCellProps) {
  return (
    <Card>
      <CardContent className="pt-4 pb-4 px-4">
        <p className="text-xl font-bold font-mono tabular-nums">{value}</p>
        <p className="text-xs text-muted-foreground mt-1">{label}</p>
      </CardContent>
    </Card>
  )
}

export function StatsGrid() {
  const {
    totalFrames,
    totalDetections,
    avgConfidence,
    peakFps,
    missedFrames,
  } = useStore()

  return (
    <div className="grid grid-cols-5 gap-2">
      <StatCell label="Total Frames" value={totalFrames.toLocaleString()} />
      <StatCell label="Total Detections" value={totalDetections.toLocaleString()} />
      <StatCell
        label="Avg Confidence"
        value={`${(avgConfidence * 100).toFixed(1)}%`}
      />
      <StatCell label="Peak FPS" value={peakFps.toFixed(1)} />
      <StatCell label="Missed Frames" value={missedFrames.toLocaleString()} />
    </div>
  )
}
