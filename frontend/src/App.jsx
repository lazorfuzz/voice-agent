import { useEffect, useRef, useState } from 'react'
import { useRoom } from './useRoom.js'
import Presence from './Presence.jsx'

const SEND = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="19" x2="12" y2="5" />
    <polyline points="5 12 12 5 19 12" />
  </svg>
)

const MIC_ON = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="9" y="2" width="6" height="12" rx="3" />
    <path d="M5 10a7 7 0 0 0 14 0" />
    <line x1="12" y1="19" x2="12" y2="23" />
  </svg>
)
const MIC_OFF = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 4a3 3 0 0 1 6 0v6" />
    <path d="M15 12.5V14" />
    <path d="M5 10a7 7 0 0 0 10.9 5.8M19 10a6.97 6.97 0 0 1-.9 3.4" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="3" y1="3" x2="21" y2="21" />
  </svg>
)
const GEAR = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
)
const CAM = (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M23 7l-7 5 7 5V7z" />
    <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
  </svg>
)

export default function App() {
  const r = useRoom()
  const mainRef = useRef(null)
  const [draft, setDraft] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [personaForm, setPersonaForm] = useState(null)   // null | {key?, label, personality, voice}
  const [personaErr, setPersonaErr] = useState('')
  const [savingPersona, setSavingPersona] = useState(false)
  const cur = r.personaList.find((p) => p.key === r.persona)
  const who = { name: cur?.label || 'Kronik', initial: (cur?.label || 'K').trim()[0].toUpperCase() }
  const builtins = r.personaList.filter((p) => p.builtin)
  const customs = r.personaList.filter((p) => !p.builtin)

  const startNew = () => { setPersonaErr(''); setPersonaForm({ label: '', personality: '', voice: r.voices[0] || 'alba' }) }
  const startEdit = (p) => { setPersonaErr(''); setPersonaForm({ key: p.key, label: p.label, personality: p.personality, voice: p.voice }) }
  const savePersonaForm = async () => {
    setSavingPersona(true); setPersonaErr('')
    try {
      const saved = await r.savePersona(personaForm)
      r.setPersona(saved.key)
      setPersonaForm(null)
    } catch (err) { setPersonaErr(err.message || 'Could not save') }
    finally { setSavingPersona(false) }
  }

  useEffect(() => {
    const m = mainRef.current
    if (m) m.scrollTop = m.scrollHeight
  }, [r.messages])

  const muteLabel = r.muted ? 'Unmute microphone' : 'Mute microphone'

  const submitText = (e) => {
    e.preventDefault()
    const t = draft.trim()
    if (!t) return
    r.sendChatMessage(t)
    setDraft('')
  }

  const enroll = () => {
    const nm = prompt('Name for this voice (e.g. your name):')
    if (nm) r.enrollVoice(nm)
  }

  return (
    <>
      <header>
        <div className="brand">
          <div className="logo">{who.initial}</div>
          <div className="titles">
            <h1>{who.name}</h1>
            <div className="statusRow">
              <span className={'dot' + (r.status.state ? ' ' + r.status.state : '')}></span>
              <span className="statusText">{r.status.text}</span>
            </div>
          </div>
        </div>

        <div className="actions">
          {r.connected && (
            <button
              className={'iconBtn' + (r.muted ? ' danger' : '')}
              onClick={r.toggleMute} title={muteLabel} aria-label={muteLabel}
            >
              {r.muted ? MIC_OFF : MIC_ON}
            </button>
          )}
          <button
            className={'iconBtn' + (settingsOpen ? ' on' : '')}
            onClick={() => setSettingsOpen((v) => !v)}
            title="Settings" aria-label="Settings"
          >
            {GEAR}
          </button>
          <button
            className={'connectBtn' + (r.connected ? ' connected' : '')}
            disabled={r.connecting}
            onClick={() => (r.connected ? r.disconnect() : r.connect())}
          >
            {r.connected ? 'End' : (r.connecting ? '…' : 'Connect')}
          </button>
        </div>
      </header>

      {r.liveActive && (
        <div className="liveView">
          <div className="liveBar">
            <span className="liveTitle">
              <span className={'liveDot' + (r.liveStatus === 'live' ? ' on' : '')} />
              Bedroom · {r.liveStatus === 'live' ? 'live' : (r.liveStatus || 'connecting…')}
            </span>
            <button className="liveClose" onClick={r.stopLiveView} title="Close live view">✕</button>
          </div>
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <video ref={r.liveVideoRef} autoPlay playsInline className="liveVideo" />
        </div>
      )}

      {settingsOpen && (
        <>
          <div className="sheetBackdrop" onClick={() => setSettingsOpen(false)} />
          <div className="sheet" role="dialog" aria-label="Settings">
            <div className="sheetGrab" />

            <div className="field">
              <div className="fieldTop">
                <span className="fieldLabel">Persona</span>
                {!personaForm && (
                  <button className="linkBtn" disabled={r.connected} onClick={startNew}>＋ New persona</button>
                )}
              </div>

              {!personaForm && (
                <>
                  <select
                    className="fieldSelect"
                    value={r.persona}
                    disabled={r.connected}
                    onChange={(e) => r.setPersona(e.target.value)}
                  >
                    {builtins.length > 0 && (
                      <optgroup label="Built-in">
                        {builtins.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
                      </optgroup>
                    )}
                    {customs.length > 0 && (
                      <optgroup label="Custom">
                        {customs.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
                      </optgroup>
                    )}
                    <optgroup label="Experimental">
                      <option value="__gemma__">🧪 Gemma Ears (audio-native)</option>
                    </optgroup>
                  </select>
                  {cur && !cur.builtin && !r.connected && (
                    <div className="btnRow">
                      <button className="linkBtn" onClick={() => startEdit(cur)}>Edit</button>
                      <button
                        className="linkBtn danger"
                        onClick={() => { if (confirm(`Delete “${cur.label}”?`)) r.deletePersona(cur.key) }}
                      >Delete</button>
                    </div>
                  )}
                  {r.connected && <span className="fieldHint">Disconnect to switch or edit personas.</span>}
                </>
              )}

              {personaForm && (
                <div className="personaForm">
                  <input
                    className="formInput"
                    placeholder="Name — e.g. Jarvis"
                    value={personaForm.label}
                    onChange={(e) => setPersonaForm({ ...personaForm, label: e.target.value })}
                  />
                  <textarea
                    className="formArea"
                    rows={6}
                    placeholder="Personality — how they talk and behave. e.g. You are Jarvis, a dry, hyper-competent British butler-AI. Warm but efficient. Keep replies short (1–2 sentences) since they're spoken aloud."
                    value={personaForm.personality}
                    onChange={(e) => setPersonaForm({ ...personaForm, personality: e.target.value })}
                  />
                  <div className="formVoiceRow">
                    <span className="fieldLabel">Voice</span>
                    <select
                      className="fieldSelect"
                      value={personaForm.voice}
                      onChange={(e) => setPersonaForm({ ...personaForm, voice: e.target.value })}
                    >
                      {r.voices.map((v) => <option key={v} value={v}>{v}</option>)}
                    </select>
                    <button
                      type="button"
                      className="previewBtn"
                      disabled={r.previewingVoice === personaForm.voice}
                      onClick={() => r.previewVoice(personaForm.voice)}
                      title="Hear this voice"
                    >
                      {r.previewingVoice === personaForm.voice ? '♪ …' : '► Preview'}
                    </button>
                  </div>
                  <span className="fieldHint">First preview loads the voice model (a few seconds); instant after.</span>
                  {personaErr && <span className="formErr">{personaErr}</span>}
                  <div className="btnRow">
                    <button className="sheetDone alt" disabled={savingPersona} onClick={savePersonaForm}>
                      {savingPersona ? 'Saving…' : (personaForm.key ? 'Save changes' : 'Create persona')}
                    </button>
                    <button className="linkBtn" onClick={() => setPersonaForm(null)}>Cancel</button>
                  </div>
                </div>
              )}
            </div>

            <div className="rowField">
              <div className="rowText">
                <span className="fieldLabel">Ambient mode</span>
                <span className="fieldHint">Always listening — only replies to “Hey {who.name}”.</span>
              </div>
              <button
                className={'toggle' + (r.ambient ? ' on' : '')}
                disabled={r.connected}
                aria-pressed={r.ambient}
                onClick={() => r.setAmbient(!r.ambient)}
              ><span /></button>
            </div>

            <div className="field">
              <span className="fieldLabel">Microphone</span>
              <select className="fieldSelect" value={r.selectedMic} onChange={(e) => r.changeMic(e.target.value)}>
                {r.mics.length === 0
                  ? <option value="">Default microphone</option>
                  : r.mics.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
            </div>

            {r.ambient && (
              <div className="field">
                <div className="fieldTop">
                  <span className="fieldLabel">Allowed speakers{r.enrolled.length ? ` · ${r.enrolled.length}` : ''}</span>
                  <button className="linkBtn" disabled={r.enrolling} onClick={enroll}>
                    {r.enrolling ? (r.enrollLabel || 'Recording…') : '＋ Enroll'}
                  </button>
                </div>
                <span className="fieldHint">
                  {r.enrolled.length
                    ? `Only these voices can wake and talk to ${who.name}.`
                    : `No voices enrolled — ${who.name} responds to anyone.`}
                </span>
                {r.enrolled.length > 0 && (
                  <div className="chips">
                    {r.enrolled.map((n) => (
                      <span key={n} className="chip">
                        {n}
                        <button title="Remove" onClick={() => r.deleteSpeaker(n)}>×</button>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}

            <button
              className="sheetAction"
              onClick={() => { r.liveActive ? r.stopLiveView() : r.startLiveView('bedroom'); setSettingsOpen(false) }}
            >
              {CAM}<span>{r.liveActive ? 'Close bedroom camera' : 'View bedroom camera'}</span>
            </button>

            <button className="sheetDone" onClick={() => setSettingsOpen(false)}>Done</button>
          </div>
        </>
      )}

      <main ref={mainRef}>
        <div className="chat">
          {r.messages.length === 0 ? (
            <div className="empty">
              <div className="stage">
                <Presence analysersRef={r.analysersRef} connected={r.connected} muted={r.muted} size="stage" />
              </div>
              <div className="emptyTitle">{who.name}</div>
              <div className="emptyHint">
                {r.connected ? (r.muted ? 'Mic muted — unmute or type below.' : 'Listening — say something, or type below.') : 'Talk or type to start.'}
              </div>
              {!r.connected && (
                <button className="big-connect" disabled={r.connecting} onClick={() => r.connect()}>
                  {r.connecting ? 'Connecting…' : 'Connect'}
                </button>
              )}
            </div>
          ) : (
            r.messages.map((m, i) => {
              const prev = r.messages[i - 1]
              const grouped = prev && prev.isAgent === m.isAgent
              return (
                <div key={m.id} className={'row ' + (m.isAgent ? 'agent' : 'you') + (grouped ? ' grouped' : '')}>
                  <div className="avatar">{grouped ? '' : (m.isAgent ? who.initial : 'You')}</div>
                  <div className={'bubble' + (m.final ? '' : ' partial')}>{m.text}</div>
                </div>
              )
            })
          )}
        </div>
      </main>

      {/* text chat works standalone over HTTP — no voice connection / TURN needed */}
      <form className="composer" onSubmit={submitText}>
        {r.connected && r.messages.length > 0 && (
          <div className="dockOrb" title={r.muted ? 'Mic muted' : 'Live'}>
            <Presence analysersRef={r.analysersRef} connected={r.connected} muted={r.muted} size="dock" />
          </div>
        )}
        <div className="composerPill">
          <input
            className="composerInput"
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={`Message ${who.name}…`}
            autoComplete="off"
            enterKeyHint="send"
          />
          <button className="composerSend" type="submit" disabled={!draft.trim()} aria-label="Send">
            {SEND}
          </button>
        </div>
      </form>
    </>
  )
}
