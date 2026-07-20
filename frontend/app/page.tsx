import Link from "next/link"
import { Zap, Activity, Eye, Github } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { LiveStats } from "@/components/LiveStats"

const TECH_BADGES = [
  "Next.js 14",
  "FastAPI",
  "YOLOv8n",
  "WebSocket",
  "PostgreSQL",
  "Redis",
  "Fly.io",
  "Vercel",
]

const FEATURES = [
  {
    icon: Zap,
    title: "Sub-30ms Inference",
    description:
      "YOLOv8n ONNX optimized for real-time CPU inference with batch processing and SIMD acceleration",
  },
  {
    icon: Activity,
    title: "Live Telemetry",
    description:
      "p50/p95/p99 latency, FPS, queue depth streamed live via WebSocket at 1Hz",
  },
  {
    icon: Eye,
    title: "Multi-class Detection",
    description:
      "80 COCO classes with per-class color coding, confidence thresholds, and object tracking",
  },
]

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Navbar */}
      <nav className="border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="container mx-auto flex h-14 items-center justify-between px-4">
          <span className="text-lg font-bold tracking-tight">CV Analytics</span>
          <div className="flex items-center gap-4">
            <Link
              href="/dashboard"
              className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              Dashboard
            </Link>
            <Link href="/auth/signin">
              <Button variant="outline" size="sm">
                Sign In
              </Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative overflow-hidden py-24 md:py-32">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-chart-1/10 pointer-events-none" />
        <div className="container mx-auto px-4 text-center relative">
          <Badge variant="secondary" className="mb-6 text-xs">
            Real-Time · Sub-30ms · 80 Classes
          </Badge>
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight mb-6 bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text text-transparent">
            Real-Time Computer Vision Analytics
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10">
            Sub-30ms inference. Live telemetry. Multi-class detection.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link href="/dashboard">
              <Button size="lg" className="w-full sm:w-auto">
                Open Dashboard
              </Button>
            </Link>
            <Link href="/dashboard">
              <Button variant="outline" size="lg" className="w-full sm:w-auto">
                View Demo
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="container mx-auto px-4 pb-20">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
          {FEATURES.map((feature) => {
            const Icon = feature.icon
            return (
              <Card key={feature.title} className="border-border/50">
                <CardHeader>
                  <Icon className="h-8 w-8 mb-2 text-primary" />
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                  <CardDescription>{feature.description}</CardDescription>
                </CardHeader>
              </Card>
            )
          })}
        </div>

        {/* Live Stats */}
        <div className="flex justify-center mb-16">
          <LiveStats />
        </div>

        {/* Tech Stack */}
        <div className="text-center">
          <p className="text-sm text-muted-foreground mb-4 font-medium uppercase tracking-widest">
            Built with
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            {TECH_BADGES.map((tech) => (
              <Badge key={tech} variant="outline" className="text-xs">
                {tech}
              </Badge>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/40 py-6">
        <div className="container mx-auto px-4 flex items-center justify-between text-sm text-muted-foreground">
          <span>© 2025 CV Analytics Platform</span>
          <a href="#" aria-label="GitHub" className="hover:text-foreground transition-colors">
            <Github className="h-5 w-5" />
          </a>
        </div>
      </footer>
    </div>
  )
}
