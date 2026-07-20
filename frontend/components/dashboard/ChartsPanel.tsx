"use client"

import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts"
import { format } from "date-fns"
import { useStore } from "@/lib/store"
import { getClassColor } from "@/lib/colors"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { StatsGrid } from "./StatsGrid"

function getUniqueClassKeys(data: Record<string, number>[]): string[] {
  const keys = new Set<string>()
  for (const point of data) {
    for (const key of Object.keys(point)) {
      if (key !== "timestamp") keys.add(key)
    }
  }
  return Array.from(keys)
}

export function ChartsPanel() {
  const {
    detectionRateHistory,
    classCounts,
    latencyHistory,
    queueHistory,
    setSelectedClass,
  } = useStore()

  const classKeys = getUniqueClassKeys(
    detectionRateHistory as Record<string, number>[]
  )

  const topClasses = Object.entries(classCounts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([name, count]) => ({ name, count }))

  const formatTs = (ts: number) => {
    try {
      return format(new Date(ts), "HH:mm:ss")
    } catch {
      return ""
    }
  }

  return (
    <div className="flex flex-col gap-4 p-4">
      <StatsGrid />

      {/* Detection Rate */}
      <Card>
        <CardHeader className="pb-2 pt-4 px-4">
          <CardTitle className="text-sm">Detection Rate</CardTitle>
        </CardHeader>
        <CardContent className="px-2 pb-4">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={detectionRateHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatTs}
                tick={{ fontSize: 10 }}
                stroke="hsl(var(--muted-foreground))"
              />
              <YAxis
                label={{
                  value: "Detections/sec",
                  angle: -90,
                  position: "insideLeft",
                  style: { fontSize: 10, fill: "hsl(var(--muted-foreground))" },
                }}
                tick={{ fontSize: 10 }}
                stroke="hsl(var(--muted-foreground))"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  fontSize: 11,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              {classKeys.map((key) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={getClassColor(key)}
                  dot={false}
                  strokeWidth={1.5}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Top Classes */}
      <Card>
        <CardHeader className="pb-2 pt-4 px-4">
          <CardTitle className="text-sm">Top Classes</CardTitle>
        </CardHeader>
        <CardContent className="px-2 pb-4">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={topClasses} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis type="number" tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ fontSize: 10 }}
                width={80}
                stroke="hsl(var(--muted-foreground))"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  fontSize: 11,
                }}
              />
              <Bar
                dataKey="count"
                fill="hsl(var(--primary))"
                onClick={(data: { name: string }) => setSelectedClass(data.name)}
                radius={[0, 3, 3, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Latency */}
      <Card>
        <CardHeader className="pb-2 pt-4 px-4">
          <CardTitle className="text-sm">Latency (ms)</CardTitle>
        </CardHeader>
        <CardContent className="px-2 pb-4">
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={latencyHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatTs}
                tick={{ fontSize: 10 }}
                stroke="hsl(var(--muted-foreground))"
              />
              <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  fontSize: 11,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <ReferenceLine
                y={45}
                stroke="#ef4444"
                strokeDasharray="3 3"
                label={{ value: "SLO 45ms", fontSize: 10, fill: "#ef4444" }}
              />
              <Area
                type="monotone"
                dataKey="p50"
                stroke="#22c55e"
                fill="#22c55e"
                fillOpacity={0.3}
                strokeWidth={1.5}
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="p95"
                stroke="#f59e0b"
                fill="#f59e0b"
                fillOpacity={0.2}
                strokeWidth={1.5}
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="p99"
                stroke="#ef4444"
                fill="#ef4444"
                fillOpacity={0.1}
                strokeWidth={1.5}
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Queue Depth */}
      <Card>
        <CardHeader className="pb-2 pt-4 px-4">
          <CardTitle className="text-sm">Queue Depth</CardTitle>
        </CardHeader>
        <CardContent className="px-2 pb-4">
          <ResponsiveContainer width="100%" height={100}>
            <AreaChart data={queueHistory}>
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatTs}
                tick={{ fontSize: 10 }}
                stroke="hsl(var(--muted-foreground))"
              />
              <YAxis tick={{ fontSize: 10 }} stroke="hsl(var(--muted-foreground))" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  fontSize: 11,
                }}
              />
              <Area
                type="monotone"
                dataKey="depth"
                stroke="#f59e0b"
                fill="#f59e0b"
                fillOpacity={0.4}
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
