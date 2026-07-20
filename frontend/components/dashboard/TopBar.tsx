"use client"

import { useSession, signOut } from "next-auth/react"
import Link from "next/link"
import Image from "next/image"
import { useStore } from "@/lib/store"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import type { SourceType } from "@/lib/store"

interface TopBarProps {
  onConnect?: () => void
}

export function TopBar({ onConnect }: TopBarProps) {
  const { data: session } = useSession()
  const {
    source,
    sourceUrl,
    setSource,
    setSourceUrl,
    connectionStatus,
    fps,
    p95Latency,
  } = useStore()

  const isConnected = connectionStatus === "connected"
  const isConnecting = connectionStatus === "connecting"

  const latencyVariant =
    p95Latency < 30
      ? "default"
      : p95Latency < 60
      ? "secondary"
      : "destructive"

  const latencyLabel = `p95: ${p95Latency.toFixed(0)}ms`

  return (
    <header className="flex items-center justify-between h-14 px-4 border-b border-border bg-card shrink-0">
      {/* Left: source controls */}
      <div className="flex items-center gap-2">
        <Select
          value={source}
          onValueChange={(val) => setSource(val as SourceType)}
        >
          <SelectTrigger className="w-36 h-8 text-xs">
            <SelectValue placeholder="Source" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="demo">Demo</SelectItem>
            <SelectItem value="webcam">Webcam</SelectItem>
            <SelectItem value="rtsp">RTSP</SelectItem>
            <SelectItem value="upload">Upload</SelectItem>
          </SelectContent>
        </Select>

        {source === "rtsp" && (
          <Input
            value={sourceUrl}
            onChange={(e) => setSourceUrl(e.target.value)}
            placeholder="rtsp://..."
            className="h-8 text-xs w-56"
          />
        )}

        {source === "upload" && (
          <label className="cursor-pointer">
            <span className="inline-flex items-center justify-center h-8 px-3 rounded-md border border-input bg-background text-xs hover:bg-accent hover:text-accent-foreground transition-colors">
              Choose File
            </span>
            <input type="file" accept="video/*" className="sr-only" />
          </label>
        )}

        <Button
          size="sm"
          variant={isConnected ? "destructive" : "default"}
          className="h-8 text-xs"
          onClick={onConnect}
          disabled={isConnecting}
        >
          {isConnecting ? "Connecting..." : isConnected ? "Disconnect" : "Connect"}
        </Button>
      </div>

      {/* Right: metrics + user */}
      <div className="flex items-center gap-3">
        <span className="font-mono text-sm tabular-nums text-foreground">
          {fps.toFixed(1)} fps
        </span>

        <Badge
          variant={latencyVariant}
          className={cn(
            "text-xs font-mono",
            p95Latency < 30 && "bg-green-500/20 text-green-400 border-green-500/30",
            p95Latency >= 30 && p95Latency < 60 && "bg-amber-500/20 text-amber-400 border-amber-500/30"
          )}
        >
          {latencyLabel}
        </Badge>

        {/* WS status dot */}
        <span
          className={cn(
            "h-2.5 w-2.5 rounded-full",
            connectionStatus === "connected" && "bg-green-500 animate-pulse",
            connectionStatus === "connecting" && "bg-amber-500 animate-pulse",
            connectionStatus === "error" && "bg-red-500",
            connectionStatus === "disconnected" && "bg-muted-foreground"
          )}
          title={`WebSocket: ${connectionStatus}`}
        />

        {/* User section */}
        {session?.user ? (
          <div className="flex items-center gap-2">
            {session.user.image && (
              <Image
                src={session.user.image}
                alt={session.user.name ?? "User"}
                width={24}
                height={24}
                className="rounded-full"
              />
            )}
            <span className="text-xs text-muted-foreground hidden sm:block">
              {session.user.name}
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={() => void signOut()}
            >
              Sign Out
            </Button>
          </div>
        ) : (
          <Link href="/auth/signin">
            <Button variant="ghost" size="sm" className="h-7 text-xs">
              Sign In
            </Button>
          </Link>
        )}
      </div>
    </header>
  )
}
