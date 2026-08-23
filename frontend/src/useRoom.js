import { useCallback, useEffect, useRef, useState } from 'react'
import { Room, RoomEvent, Track } from 'livekit-client'

// same-origin when served behind the HTTPS proxy (iPhone/LAN); direct on localhost dev
const sameOrigin = (location.protocol === 'https:' || location.port === '8443')
const TOKEN_URL = sameOrigin ? '/token' : 'http://127.0.0.1:8790/token'
// Text chat goes over plain HTTP/SSE (not the WebRTC data channel), so it works
// off-network with no TURN. Independent of the LiveKit voice session.
const CHAT_URL = sameOrigin ? '/chat' : 'http://127.0.0.1:8790/chat'
// Speaker enrollment for ambient mode (record a voice signature server-side).
const ENROLL_URL = sameOrigin ? '/enroll' : 'http://127.0.0.1:8790/enroll'
const ENROLLED_URL = sameOrigin ? '/enrolled' : 'http://127.0.0.1:8790/enrolled'
// Persona list/create/edit/delete + the TTS voice catalog (for building custom personas).
const PERSONAS_URL = sameOrigin ? '/personas' : 'http://127.0.0.1:8790/personas'
const VOICES_URL = sameOrigin ? '/voices' : 'http://127.0.0.1:8790/voices'
const VOICE_PREVIEW_URL = sameOrigin ? '/voice_preview' : 'http://127.0.0.1:8790/voice_preview'
// Frequently-used chat commands (server-tracked) for the home-screen pills.
const COMMANDS_URL = sameOrigin ? '/commands' : 'http://127.0.0.1:8790/commands'
const _NGROK = { 'ngrok-skip-browser-warning': 'true' }
// Ring live-view WebRTC broker (WebSocket signalling + ICE servers).
const LIVE_ICE_URL = sameOrigin ? '/live/ice' : 'http://127.0.0.1:8791/live/ice'
const _wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:'
const LIVE_WS_URL = sameOrigin ? `${_wsProto}//${location.host}/live/ws` : 'ws://127.0.0.1:8791/live/ws'
// One short take is enough: the ECAPA speaker model is robust to far-field, and the gate
// ADAPTS online (each accepted utterance folds into your profile), so accuracy climbs with
// use instead of demanding lots of up-front recording. (Multi-take stays supported — bump
// ENROLL_TAKES if you want a stronger initial profile.)
const ENROLL_TAKES = 1
const ENROLL_TAKE_SECONDS = 6

export function useRoom() {
  const [connected, setConnected] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [status, setStatus] = useState({ state: '', text: 'Disconnected' })
  const [messages, setMessages] = useState([])   // {id, isAgent, text, final}
  const [mics, setMics] = useState([])            // {id, label}
  const [selectedMic, setSelectedMic] = useState('')
  const [muted, setMuted] = useState(false)
  const [persona, setPersona] = useState(() => localStorage.getItem('persona') || 'kronik')
  const personaRef = useRef('kronik')
  useEffect(() => { personaRef.current = persona }, [persona])
  const [personaList, setPersonaList] = useState([])  // [{key,label,voice,personality,greeting,builtin}]
  const [voices, setVoices] = useState([])            // catalog voice names for the create form
  const [cmdPills, setCmdPills] = useState([])        // frequent commands for the home screen
  const [previewingVoice, setPreviewingVoice] = useState('')  // voice currently being previewed
  const previewAudioRef = useRef(null)
  const [ambient, setAmbient] = useState(() =>       // always-listen, respond only when addressed; persisted
    localStorage.getItem('ambient') === '1')
  const ambientRef = useRef(false)
  useEffect(() => { ambientRef.current = ambient }, [ambient])
  const [enrolled, setEnrolled] = useState([])       // enrolled speaker names (ambient voice gate)
  const [enrolling, setEnrolling] = useState(false)  // true while the multi-take enroll flow runs
  const [enrollLabel, setEnrollLabel] = useState('') // live status shown on the enroll button

  const roomRef = useRef(null)
  const wakeLockRef = useRef(null)
  const audioCtxRef = useRef(null)
  const localIdentityRef = useRef(null)
  const analysersRef = useRef({ you: null, agent: null })   // read imperatively by the waveform
  const selectedMicRef = useRef('')
  useEffect(() => { selectedMicRef.current = selectedMic }, [selectedMic])
  const messagesRef = useRef([])
  useEffect(() => { messagesRef.current = messages }, [messages])

  /* ---------- screen wake lock ----------
     iOS auto-lock (~30s) suspends Safari's mic capture while the WebRTC track stays
     "published" — the agent hears silence and the call looks dead (observed live
     2026-07-10: sessions died ~30-60s in, revived on reload). Hold a screen wake lock
     for the duration of the call; re-acquire on tab return (locks auto-release when
     the page is hidden). */
  const acquireWakeLock = useCallback(async () => {
    try {
      if ('wakeLock' in navigator) {
        wakeLockRef.current = await navigator.wakeLock.request('screen')
      }
    } catch (e) { /* unsupported / low battery — degrade gracefully */ }
  }, [])
  const releaseWakeLock = useCallback(() => {
    try { wakeLockRef.current?.release() } catch (e) {}
    wakeLockRef.current = null
  }, [])

  const ensureAudioCtx = () => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)()
    }
    return audioCtxRef.current
  }
  const makeAnalyser = (mediaStreamTrack) => {
    const ctx = ensureAudioCtx()
    const src = ctx.createMediaStreamSource(new MediaStream([mediaStreamTrack]))
    const an = ctx.createAnalyser()
    an.fftSize = 1024
    an.smoothingTimeConstant = 0.8
    src.connect(an)
    return an
  }

  /* ---------- microphones ---------- */
  const populateMics = useCallback(async () => {
    let devices = []
    try { devices = await navigator.mediaDevices.enumerateDevices() } catch (e) { return }
    const found = devices
      .filter(d => d.kind === 'audioinput')
      .map((d, i) => ({ id: d.deviceId, label: d.label || `Microphone ${i + 1}` }))
    setMics(found)
    setSelectedMic(prev => prev || (found[0]?.id || ''))
  }, [])

  const changeMic = useCallback(async (deviceId) => {
    setSelectedMic(deviceId)
    const room = roomRef.current
    if (room) {
      try { await room.switchActiveDevice('audioinput', deviceId) }
      catch (e) { console.warn('switch mic failed', e) }
    }
  }, [])

  /* ---------- transcripts ---------- */
  // Gemma Ears defers the transcript: bubbles buffer here during the call and flush
  // into the chat when the user hits End (ChatGPT-voice style).
  const deferredBufRef = useRef([])
  const onGemmaTranscription = useCallback(async (reader, info) => {
    try {
      const attrs = reader.info?.attributes || {}
      const segId = attrs['lk.segment_id'] || reader.info?.id || Math.random().toString(36).slice(2)
      const identity = typeof info === 'string' ? info : (info?.identity || '')
      // gemma-ears can't impersonate the sender, so it tags bubbles with kronik.role
      const role = attrs['kronik.role']
      const isAgent = role
        ? role === 'agent'
        : !!(identity && localIdentityRef.current && identity !== localIdentityRef.current)
      const buf = deferredBufRef.current
      const msg = { id: segId, isAgent, text: '', final: false }
      if (!buf.some(m => m.id === segId)) buf.push(msg)
      let text = ''
      for await (const chunk of reader) {
        text += chunk
        msg.text = text
      }
      msg.text = text
      msg.final = true
    } catch (e) { console.warn('transcription error', e) }
  }, [])

  // LiveKit Agents publishes normal voice transcripts through this event as well as
  // lk.transcription text streams. Only Gemma needs the stream's custom role attributes.
  const onTranscription = useCallback((segments, participant) => {
    const identity = participant?.identity || ''
    const isAgent = !!(identity && localIdentityRef.current && identity !== localIdentityRef.current)
    setMessages(prev => {
      const next = [...prev]
      for (const segment of segments) {
        const index = next.findIndex(message => message.id === segment.id)
        const message = {
          id: segment.id,
          isAgent,
          text: segment.text || '',
          final: segment.final,
        }
        if (index === -1) next.push(message)
        else next[index] = message
      }
      return next
    })
  }, [])

  /* ---------- connect / disconnect ---------- */
  const teardown = useCallback(() => {
    releaseWakeLock()
    setConnected(false)
    setConnecting(false)
    setMuted(false)
    analysersRef.current = { you: null, agent: null }
    if (roomRef.current) { try { roomRef.current.disconnect() } catch (e) {} roomRef.current = null }
    if (deferredBufRef.current.length) {
      const transcript = deferredBufRef.current
        .filter(m => m.text.trim())
        .map(m => ({ ...m, final: true }))
      deferredBufRef.current = []
      setMessages(prev => [...prev, ...transcript])
    }
    setStatus({ state: '', text: 'Disconnected' })
  }, [releaseWakeLock])

  const connect = useCallback(async () => {
    setConnecting(true)
    setStatus({ state: 'connecting', text: 'Connecting…' })
    try {
      // Ask for mic permission the instant Connect is tapped — do it FIRST, while the
      // click's user-gesture is still active, so the OS prompt shows immediately (not after
      // the room negotiation). Drop this probe track; LiveKit acquires the real one below.
      try {
        const probe = await navigator.mediaDevices.getUserMedia({ audio: true })
        probe.getTracks().forEach((t) => t.stop())
      } catch (e) {
        setStatus({ state: 'error', text: 'Microphone access is needed for voice — enable it and try again.' })
        setConnecting(false)
        return
      }

      const ctx = ensureAudioCtx()
      if (ctx.state === 'suspended') await ctx.resume()

      const identity = 'user-' + Math.random().toString(36).slice(2, 9)
      localIdentityRef.current = identity
      // ?room= URL override kept as a hidden dev hook (e.g. ?room=moshi to talk to an
      // experimental agent in another room); default is the normal Kronik room.
      const roomName = new URLSearchParams(location.search).get('room') ||
        (personaRef.current === '__gemma__' ? 'gemma' : 'voice')
      const res = await fetch(
        `${TOKEN_URL}?room=${encodeURIComponent(roomName)}&identity=${identity}&persona=${personaRef.current}` +
        `&ambient=${ambientRef.current ? 'on' : 'off'}`, {
        headers: { 'ngrok-skip-browser-warning': 'true' },   // skip ngrok's free-tier interstitial
      })
      const { token, url, iceServers } = await res.json()
      // Behind a TLS proxy (Caddy/ngrok) reach LiveKit same-origin; direct ws on localhost dev.
      const lkUrl = (location.protocol === 'https:') ? `wss://${location.host}` : url
      const connectOpts = (iceServers && iceServers.length) ? { rtcConfig: { iceServers } } : undefined

      const room = new Room({ adaptiveStream: true, dynacast: true })
      roomRef.current = room
      if (personaRef.current === '__gemma__') {
        room.registerTextStreamHandler('lk.transcription', onGemmaTranscription)
      } else {
        room.on(RoomEvent.TranscriptionReceived, onTranscription)
      }
      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) {
          const el = track.attach(); el.style.display = 'none'; document.body.appendChild(el)
          try { analysersRef.current.agent = makeAnalyser(track.mediaStreamTrack) } catch (e) {}
        }
      })
      room.on(RoomEvent.Disconnected, () => teardown())

      await room.connect(lkUrl, token, connectOpts)

      const deviceId = selectedMicRef.current && selectedMicRef.current !== 'default'
        ? selectedMicRef.current : undefined
      // WebRTC-layer processing at the source (browser built-ins: AEC/AGC). In ambient
      // mode browser noise suppression is OFF: it gates faint far-field speech to
      // near-zero, and the server-side DTLN denoiser already covers noise — double
      // suppression is why across-the-room requests never transcribed.
      await room.localParticipant.setMicrophoneEnabled(true, {
        ...(deviceId ? { deviceId } : {}),
        echoCancellation: true, noiseSuppression: !ambientRef.current, autoGainControl: true,
      })
      const pub = room.localParticipant.getTrackPublication(Track.Source.Microphone)
      if (pub?.audioTrack) {
        try { analysersRef.current.you = makeAnalyser(pub.audioTrack.mediaStreamTrack) } catch (e) {}
      }

      await populateMics()
      setMuted(false)
      setConnected(true)
      setConnecting(false)
      setStatus({ state: 'connected', text: 'Connected' })
      acquireWakeLock()   // keep the phone screen on so iOS can't suspend the mic
    } catch (e) {
      console.error(e)
      setStatus({ state: 'error', text: 'Error: ' + (e.message || e) })
      setConnecting(false)
      teardown()
    }
  }, [onGemmaTranscription, onTranscription, populateMics, teardown, acquireWakeLock])

  // Text chat over HTTP/SSE — no WebRTC/TURN, works standalone (no voice connection).
  const sendChatMessage = useCallback(async (text) => {
    const msg = (text || '').trim()
    if (!msg) return
    const uid = 'u-' + Date.now() + '-' + Math.random().toString(36).slice(2)
    const aid = 'a-' + Date.now() + '-' + Math.random().toString(36).slice(2)
    // history = everything shown so far (voice + text) plus this new message
    const history = [...messagesRef.current, { isAgent: false, text: msg }]
      .filter(m => m.text && m.text.trim())
      .map(m => ({ role: m.isAgent ? 'assistant' : 'user', content: m.text }))
    setMessages(prev => [...prev,
      { id: uid, isAgent: false, text: msg, final: true },
      { id: aid, isAgent: true, text: '', final: false }])
    try {
      const res = await fetch(CHAT_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
        body: JSON.stringify({ messages: history, persona: personaRef.current }),
      })
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop()
        for (const part of parts) {
          const line = part.split('\n').find(l => l.startsWith('data:'))
          if (!line) continue
          let ev
          try { ev = JSON.parse(line.slice(5).trim()) } catch { continue }
          if (ev.type === 'delta') {
            setMessages(prev => prev.map(m => (m.id === aid ? { ...m, text: m.text + ev.text } : m)))
          } else if (ev.type === 'error') {
            setMessages(prev => prev.map(m => (m.id === aid
              ? { ...m, text: m.text || 'Something went wrong.', final: true } : m)))
          }
        }
      }
      setMessages(prev => prev.map(m => (m.id === aid ? { ...m, final: true } : m)))
    } catch (e) {
      console.warn('chat failed', e)
      setMessages(prev => prev.map(m => (m.id === aid
        ? { ...m, text: m.text || "Couldn't reach the chat server.", final: true } : m)))
    }
  }, [])

  const toggleMute = useCallback(async () => {
    const room = roomRef.current
    if (!room) return
    const pub = room.localParticipant.getTrackPublication(Track.Source.Microphone)
    if (!pub) return
    const next = !muted
    try {
      if (next) await pub.mute(); else await pub.unmute()   // stops/resumes sending mic audio
      setMuted(next)
    } catch (e) { console.warn('mute toggle failed', e) }
  }, [muted])

  /* ---------- speaker enrollment (ambient voice gate) ---------- */
  const fetchEnrolled = useCallback(async () => {
    try {
      const r = await fetch(ENROLLED_URL, { headers: { 'ngrok-skip-browser-warning': 'true' } })
      const d = await r.json()
      setEnrolled(d.speakers || [])
    } catch (e) { /* server may be offline */ }
  }, [])

  const enrollVoice = useCallback(async (name) => {
    const nm = (name || '').trim()
    if (!nm || enrolling) return
    let stream
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }) }
    catch (e) { alert('Microphone permission is needed to enroll a voice.'); return }
    setEnrolling(true)
    let speakers = null
    try {
      for (let take = 1; take <= ENROLL_TAKES; take++) {
        const rec = new MediaRecorder(stream)
        const chunks = []
        rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data) }
        const stopped = new Promise((res) => { rec.onstop = res })
        rec.start()
        const takeTag = ENROLL_TAKES > 1 ? `Take ${take}/${ENROLL_TAKES} · ` : 'Recording · '
        for (let s = ENROLL_TAKE_SECONDS; s > 0; s--) {
          setEnrollLabel(`${takeTag}${s}s — keep talking`)
          await new Promise((r) => setTimeout(r, 1000))
        }
        rec.stop(); await stopped
        setEnrollLabel(ENROLL_TAKES > 1 ? `Saving take ${take}/${ENROLL_TAKES}…` : 'Saving…')
        const r = await fetch(`${ENROLL_URL}?name=${encodeURIComponent(nm)}`, {
          method: 'POST', body: new Blob(chunks, { type: 'audio/webm' }),
          headers: { 'Content-Type': 'application/octet-stream', 'ngrok-skip-browser-warning': 'true' },
        })
        const d = await r.json()
        if (d.error) { alert('Enroll failed: ' + d.error); break }
        speakers = d.speakers || speakers
        if (take < ENROLL_TAKES) {
          setEnrollLabel('Nice — pause a beat, then keep talking')
          await new Promise((r) => setTimeout(r, 900))   // brief gap => natural variation across takes
        }
      }
      if (speakers) setEnrolled(speakers)
    } catch (e) {
      alert('Enroll failed — is the server reachable?')
    } finally {
      stream.getTracks().forEach((t) => t.stop())
      setEnrolling(false)
      setEnrollLabel('')
    }
  }, [enrolling])

  const deleteSpeaker = useCallback(async (name) => {
    try {
      const r = await fetch(`${ENROLL_URL}?name=${encodeURIComponent(name)}`, {
        method: 'DELETE', headers: { 'ngrok-skip-browser-warning': 'true' } })
      const d = await r.json()
      setEnrolled(d.speakers || [])
    } catch (e) { /* ignore */ }
  }, [])

  /* persist persona + ambient selection across sessions */
  useEffect(() => { localStorage.setItem('persona', persona) }, [persona])
  useEffect(() => { localStorage.setItem('ambient', ambient ? '1' : '0') }, [ambient])

  /* ---------- personas (built-in + user-created) ---------- */
  const fetchPersonas = useCallback(async () => {
    try {
      const d = await fetch(PERSONAS_URL, { headers: _NGROK }).then((r) => r.json())
      const list = d.personas || []
      setPersonaList(list)
      // if our selected persona was deleted elsewhere, fall back to the default
      if (list.length && !list.some((p) => p.key === personaRef.current)) setPersona('kronik')
    } catch (e) { /* ignore */ }
  }, [])

  const fetchVoices = useCallback(async () => {
    try {
      const d = await fetch(VOICES_URL, { headers: _NGROK }).then((r) => r.json())
      setVoices(d.voices || [])
    } catch (e) { /* ignore */ }
  }, [])

  const fetchCommands = useCallback(async () => {
    try {
      const d = await fetch(COMMANDS_URL, { headers: _NGROK }).then((r) => r.json())
      setCmdPills(Array.isArray(d.commands) ? d.commands : [])
    } catch (e) { /* ignore — pills are a nice-to-have */ }
  }, [])

  const savePersona = useCallback(async ({ key, label, personality, voice, greeting }) => {
    const res = await fetch(PERSONAS_URL, {
      method: 'POST', headers: { 'Content-Type': 'application/json', ..._NGROK },
      body: JSON.stringify({ key, label, personality, voice, greeting }),
    })
    const d = await res.json()
    if (!res.ok) throw new Error(d.error || 'Could not save persona')
    await fetchPersonas()
    return d.persona   // {key, label, ...}
  }, [fetchPersonas])

  const deletePersona = useCallback(async (key) => {
    try {
      await fetch(`${PERSONAS_URL}?key=${encodeURIComponent(key)}`, { method: 'DELETE', headers: _NGROK })
    } catch (e) { /* ignore */ }
    if (personaRef.current === key) setPersona('kronik')
    await fetchPersonas()
  }, [fetchPersonas])

  const previewVoice = useCallback(async (voice) => {
    if (!voice) return
    try { previewAudioRef.current?.pause() } catch (e) { /* */ }
    setPreviewingVoice(voice)
    try {
      // fetch as a blob (so the ngrok-skip header applies) then play it
      const res = await fetch(`${VOICE_PREVIEW_URL}?voice=${encodeURIComponent(voice)}`, { headers: _NGROK })
      if (!res.ok) throw new Error('preview failed')
      const url = URL.createObjectURL(await res.blob())
      const audio = new Audio(url)
      previewAudioRef.current = audio
      const done = () => { setPreviewingVoice(''); URL.revokeObjectURL(url) }
      audio.onended = done
      audio.onerror = done
      await audio.play()
    } catch (e) { setPreviewingVoice('') }
  }, [])

  /* ---------- Ring live view (WebRTC) ---------- */
  const liveVideoRef = useRef(null)
  const livePcRef = useRef(null)
  const liveWsRef = useRef(null)
  const [liveActive, setLiveActive] = useState(false)
  const [liveStatus, setLiveStatus] = useState('')   // '', 'connecting', 'live', 'failed', 'error…'

  const stopLiveView = useCallback(() => {
    try { liveWsRef.current?.send(JSON.stringify({ type: 'stop' })) } catch (e) { /* */ }
    try { liveWsRef.current?.close() } catch (e) { /* */ }
    try { livePcRef.current?.close() } catch (e) { /* */ }
    if (liveVideoRef.current) liveVideoRef.current.srcObject = null
    liveWsRef.current = null; livePcRef.current = null
    setLiveActive(false); setLiveStatus('')
  }, [])

  const startLiveView = useCallback(async (camera = 'bedroom') => {
    if (livePcRef.current) return
    setLiveActive(true); setLiveStatus('connecting')
    try {
      const ice = await fetch(LIVE_ICE_URL, { headers: { 'ngrok-skip-browser-warning': 'true' } })
        .then((r) => r.json()).catch(() => ({ iceServers: [] }))
      const pc = new RTCPeerConnection({ iceServers: ice.iceServers || [] })
      livePcRef.current = pc
      const stream = new MediaStream()
      pc.addTransceiver('video', { direction: 'recvonly' })
      pc.addTransceiver('audio', { direction: 'recvonly' })
      pc.ontrack = (e) => {
        stream.addTrack(e.track)
        if (liveVideoRef.current) liveVideoRef.current.srcObject = stream
      }
      pc.onconnectionstatechange = () => {
        const s = pc.connectionState
        if (s === 'connected') setLiveStatus('live')
        else if (s === 'failed' || s === 'disconnected' || s === 'closed') setLiveStatus(s)
      }
      const ws = new WebSocket(LIVE_WS_URL + '?camera=' + encodeURIComponent(camera))
      liveWsRef.current = ws
      pc.onicecandidate = (e) => {
        if (e.candidate && ws.readyState === 1) {
          ws.send(JSON.stringify({ type: 'ice', candidate: e.candidate.candidate,
                                   sdpMLineIndex: e.candidate.sdpMLineIndex }))
        }
      }
      ws.onmessage = async (ev) => {
        const m = JSON.parse(ev.data)
        if (m.type === 'answer') await pc.setRemoteDescription({ type: 'answer', sdp: m.sdp })
        else if (m.type === 'ice') {
          try { await pc.addIceCandidate({ candidate: m.candidate, sdpMLineIndex: m.sdpMLineIndex }) }
          catch (e) { /* */ }
        } else if (m.type === 'error') { setLiveStatus('error: ' + m.message); stopLiveView() }
      }
      ws.onopen = async () => {
        const offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp }))
      }
    } catch (e) { setLiveStatus('error'); stopLiveView() }
  }, [stopLiveView])

  /* ---------- init ---------- */
  useEffect(() => {
    // NOTE: do NOT request mic permission here — that would prompt on page load. We only
    // enumerate devices (no prompt; labels stay blank until permission is granted at connect).
    populateMics()
    fetchEnrolled()
    fetchPersonas()
    fetchVoices()
    fetchCommands()
    const onDeviceChange = () => populateMics()
    navigator.mediaDevices?.addEventListener('devicechange', onDeviceChange)
    // wake locks auto-release when the tab is hidden; re-acquire when the user
    // comes back if the call is still up
    const onVis = () => {
      if (document.visibilityState === 'visible' && roomRef.current) acquireWakeLock()
    }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      navigator.mediaDevices?.removeEventListener('devicechange', onDeviceChange)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [populateMics, acquireWakeLock, fetchPersonas, fetchVoices, fetchCommands])

  return {
    connected, connecting, status, messages, mics, selectedMic, muted, analysersRef,
    persona, setPersona, personaList, voices, savePersona, deletePersona,
    previewVoice, previewingVoice,
    ambient, setAmbient,
    enrolled, enrolling, enrollLabel, enrollVoice, deleteSpeaker,
    liveActive, liveStatus, liveVideoRef, startLiveView, stopLiveView,
    connect, disconnect: teardown, toggleMute, changeMic, sendChatMessage,
    cmdPills,
  }
}
