# Integrations & onboarding

Out of the box the assistant has **no tools** — it just converses. **Every** capability,
including OpenCode background sessions and each home integration, is **opt-in**: a tool only
appears to the voice + chat agent once you onboard it, which links the integration, stores its
secrets in `agent/.env.local`, and sets its `<NAME>_ENABLED` flag.

Start by copying the template:

```bash
cp agent/.env.local.example agent/.env.local   # fill in the core LiveKit/LLM values
```

Then onboard any integrations you want with the unified CLI:

```bash
cd ~/voice-agent && .venv/bin/python agent/onboard.py <name>
```

Restart the agent afterward so it picks up the newly enabled tools.

| `<name>` | What it controls | Onboarding flow | External prerequisites |
|---|---|---|---|
| `opencode` | Background coding/research sessions (`run_agent`, etc.) | Verifies the opencode binary + a local-LLM provider, installs the lean `local` agent | The [opencode](https://opencode.ai) CLI + a local OpenAI-compatible LLM |
| `wiz`      | WiZ smart lights            | UDP-discovers bulbs on your LAN, you name each one | Bulbs set up in the WiZ app on the same LAN |
| `nest`     | Nest thermostat             | Guided: paste the SDM project/OAuth values | A Google **Device Access** (SDM) project + OAuth client (a one-time Google Cloud setup) |
| `tesla`    | Tesla vehicle               | Registers your domain, OAuth login (browser), then you tap a virtual-key link | A Tesla **developer app** + a public HTTPS domain hosting the command-signing key |
| `ring`     | Ring cameras                | Amazon email/password/2FA login | A Ring account |
| `tv`       | Android / Google TV         | Enter the TV's IP, type the on-screen PIN | TV on the LAN with the Android TV Remote service |
| `petlibro` | PetLibro pet feeder         | PetLibro email/password login | A PetLibro account |
| `roomba`   | iRobot Roomba               | iRobot cloud login to fetch the local password, then enter the Roomba's LAN IP | A Roomba on the LAN + iRobot account |

## Identity & names (optional)

These env vars (in `agent/.env.local`) make prompts/responses generic — all optional:

- `ASSISTANT_NAME` (default `Kronik`), `WAKE_WORD` (default `kronik`)
- `VACUUM_NAME` (default `the vacuum`), `CAR_NAME` (default `the car`), `PET_NAME` (default `the cat`)

## Disabling an integration

Set its flag to `false` (or delete the line) in `agent/.env.local`, e.g. `WIZ_ENABLED=false`,
and restart. Its secrets stay saved, so re-enabling is just flipping the flag back.

## Robinhood (advanced, not agent-exposed)

The repo includes a standalone Robinhood CLI/MCP (`agent/robinhood_agent.py`,
`agent/robinhood_mcp.py`, `setup_robinhood.sh`). It is **intentionally not wired into the voice
or chat agent** (voice-driven trades are risky to ship by default). Use it standalone if you want.
