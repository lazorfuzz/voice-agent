import { useEffect, useRef } from 'react'

// The agent's living presence: an organic ember orb that breathes at idle, blooms
// amber when the agent speaks, and shimmers jade when you speak. One rAF loop, no
// React re-renders. Two sizes: "stage" (welcome screen hero) and "dock" (composer chip).
export default function Presence({ analysersRef, connected, muted, size = 'dock' }) {
  const canvasRef = useRef(null)
  const liveRef = useRef({ connected, muted })
  useEffect(() => { liveRef.current = { connected, muted } }, [connected, muted])

  useEffect(() => {
    const cvs = canvasRef.current
    const ctx = cvs.getContext('2d')
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const resize = () => {
      cvs.width = cvs.clientWidth * dpr
      cvs.height = cvs.clientHeight * dpr
    }
    resize()
    window.addEventListener('resize', resize)

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const buf = new Uint8Array(1024)
    const rms = (an) => {
      an.getByteTimeDomainData(buf)
      let s = 0
      for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; s += v * v }
      return Math.sqrt(s / buf.length)
    }

    // smoothed levels + color mix (0 = ember/agent, 1 = jade/you)
    let lvl = 0, mix = 0, raf = 0, t0 = performance.now()

    const draw = (now) => {
      raf = requestAnimationFrame(draw)
      const { you, agent } = analysersRef.current
      const { connected, muted } = liveRef.current
      const W = cvs.width, H = cvs.height
      const cx = W / 2, cy = H / 2
      const R = Math.min(W, H) / 2
      const t = (now - t0) / 1000

      const youLvl = you ? rms(you) : 0
      const agentLvl = agent ? rms(agent) : 0
      const active = Math.max(youLvl, agentLvl)
      const targetMix = youLvl > agentLvl && youLvl > 0.02 ? 1 : 0
      lvl += ((active > 0.02 ? Math.min(active * 3, 1) : 0) - lvl) * 0.12
      mix += (targetMix - mix) * 0.1

      // breath: slow at idle, still gently alive while speaking
      const breath = reduced ? 0 : Math.sin(t * (2 * Math.PI / 4.6)) * 0.035
      const dim = connected ? 1 : 0.62
      const mutedDim = muted && connected ? 0.5 : 1
      const r = R * (0.52 + breath + lvl * 0.30)

      // color blend ember -> jade
      const c = (a, b) => Math.round(a + (b - a) * mix)
      const core = `rgba(${c(255, 140)},${c(197, 240)},${c(126, 200)},`
      const mid = `rgba(${c(240, 92)},${c(162, 214)},${c(76, 172)},`
      const rim = `rgba(${c(240, 60)},${c(134, 180)},${c(74, 150)},`

      ctx.clearRect(0, 0, W, H)
      ctx.globalCompositeOperation = 'lighter'

      // outer halo
      let g = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, R * 0.98)
      g.addColorStop(0, mid + (0.34 * dim * mutedDim) + ')')
      g.addColorStop(1, rim + '0)')
      ctx.fillStyle = g
      ctx.beginPath(); ctx.arc(cx, cy, R * 0.98, 0, Math.PI * 2); ctx.fill()

      // two drifting lobes give it organic asymmetry
      for (let i = 0; i < 2; i++) {
        const a = t * (reduced ? 0 : (0.5 + i * 0.33)) + i * 2.4
        const ox = Math.cos(a) * r * 0.22, oy = Math.sin(a * 0.8) * r * 0.2
        g = ctx.createRadialGradient(cx + ox, cy + oy, 0, cx + ox, cy + oy, r * 0.85)
        g.addColorStop(0, mid + (0.28 * dim * mutedDim) + ')')
        g.addColorStop(1, mid + '0)')
        ctx.fillStyle = g
        ctx.beginPath(); ctx.arc(cx + ox, cy + oy, r * 0.85, 0, Math.PI * 2); ctx.fill()
      }

      // hot core
      g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 0.62)
      g.addColorStop(0, core + (0.95 * dim * mutedDim) + ')')
      g.addColorStop(0.55, core + (0.35 * dim * mutedDim) + ')')
      g.addColorStop(1, core + '0)')
      ctx.fillStyle = g
      ctx.beginPath(); ctx.arc(cx, cy, r * 0.62, 0, Math.PI * 2); ctx.fill()

      ctx.globalCompositeOperation = 'source-over'
    }
    raf = requestAnimationFrame(draw)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [analysersRef])

  return <canvas ref={canvasRef} className={'presence ' + size} aria-hidden="true" />
}
