import { NextResponse } from "next/server"

export const runtime = "nodejs"
export const dynamic = "force-dynamic"

export async function GET(): Promise<Response> {
  const backendUrl = process.env.BACKEND_URL ?? "https://cv-analytics-backend.fly.dev"

  try {
    const upstream = await fetch(`${backendUrl}/api/metrics/stream`, {
      headers: {
        Accept: "text/event-stream",
        "Cache-Control": "no-cache",
      },
    })

    if (!upstream.ok || !upstream.body) {
      return NextResponse.json({ error: "Backend unavailable" }, { status: 502 })
    }

    return new Response(upstream.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    })
  } catch {
    // Return a minimal SSE stream that just keeps the connection alive
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("data: {}\n\n"))
      },
    })
    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    })
  }
}
