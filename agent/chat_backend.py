"""HTTP text-chat brain for Kronik — the same persona + tools as the voice agent, but
driven over plain HTTP/SSE instead of LiveKit's WebRTC data channel. This lets the UI's
text chat work from anywhere (through ngrok) with no TURN server / NAT traversal.

`run_chat(client_messages)` is a generator of SSE event dicts:
  {"type":"tool","name":...}   a tool is running
  {"type":"delta","text":...}  a chunk of the reply
  {"type":"done"}              turn finished
  {"type":"error","message":}  something failed
"""
import os
import re
import json
import time
import threading
import urllib.request

import asyncio

import agent_tools
import tv_tools
import nest_tools
import roomba_tools
import wiz_tools
import ring_tools
import tesla_tools
import petlibro_tools
import doordash_tools
import personas
import config

# The device cloud tools (ring/tesla/petlibro) are async and cache an aiohttp session bound to a
# single event loop, so run them on ONE persistent background loop — not asyncio.run() per call,
# which would strand the cached session on a closed loop and break the next request.
_aioloop = None
_aiolock = threading.Lock()


def _await(coro, timeout=90):
    global _aioloop
    with _aiolock:
        if _aioloop is None:
            _aioloop = asyncio.new_event_loop()
            threading.Thread(target=_aioloop.run_forever, daemon=True).start()
    return asyncio.run_coroutine_threadsafe(coro, _aioloop).result(timeout=timeout)

LLM_BASE = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLM_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "mlx-community/Qwen3.6-35B-A3B-4bit")

MAX_TOOL_ITERS = 6


def _tool(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}

# Plain param types only (no ["type","null"] lists — those crash the mlx qwen3_coder parser).
# The full catalog; _llm() sends only the tools whose integration is enabled (config-gated).
_ALL_TOOLS = [
    _tool("run_agent",
          "Dispatch a background local AI agent (opencode) to do a coding, research, or analysis "
          "task, or to figure out something you don't know. Runs async, saved under `name`.",
          {"task": {"type": "string", "description": "what the agent should do, in plain language."},
           "name": {"type": "string", "description": "a short memorable name for the session."}},
          ["task", "name"]),
    _tool("list_sessions", "List recent saved agent sessions with status and how long ago each was active.",
          {}, []),
    _tool("get_session_status", "Get the latest state/result of a saved agent session by name.",
          {"name": {"type": "string"}}, ["name"]),
    _tool("message_session", "Send a new instruction into an existing saved agent session.",
          {"name": {"type": "string"}, "message": {"type": "string"}}, ["name", "message"]),
    _tool("delete_session", "Delete a saved agent session by name.",
          {"name": {"type": "string"}}, ["name"]),
    _tool("control_tv",
          "Control the living-room TV. action is one of: volume_up, volume_down, set_volume, mute, "
          "unmute, play, pause, stop, status. level (0-100) only for set_volume.",
          {"action": {"type": "string"}, "level": {"type": "integer"}}, ["action"]),
    _tool("play_on_tv",
          "Play a show/movie/video on the TV. Pass the corrected title. service='youtube' for "
          "music/videos; a streaming app name to force that app; else empty.",
          {"title": {"type": "string"}, "service": {"type": "string"}}, ["title"]),
    _tool("thermostat_status",
          "Get the Nest thermostat state: room temp, humidity, target setpoint(s), mode, "
          "heating/cooling.", {}, []),
    _tool("set_thermostat_temperature",
          "Set the Nest thermostat target temperature in Fahrenheit. `target` is 'heat' or "
          "'cool' only when it's in auto mode, to say which setpoint to change.",
          {"temperature": {"type": "integer"}, "target": {"type": "string"}}, ["temperature"]),
    _tool("set_thermostat_mode",
          "Switch the Nest thermostat mode: 'heat', 'cool', 'auto', or 'off'.",
          {"mode": {"type": "string"}}, ["mode"]),
    _tool("roomba_status",
          "Get the Roomba vacuum's status: what it's doing (cleaning/charging/docked/paused), "
          "battery, bin full.", {}, []),
    _tool("control_roomba",
          "Control the Roomba vacuum. action: 'clean' (start), 'stop', 'pause', 'resume', "
          "'dock' (send home), or 'find'.",
          {"action": {"type": "string"}}, ["action"]),
    _tool("lights_status",
          "Check the smart lights (WiZ): on/off, brightness, color/scene. which = a light or group "
          "name ('bedside', 'living room') or empty for all.",
          {"which": {"type": "string"}}, []),
    _tool("control_lights",
          "Control the smart lights. action: on, off, brightness (needs brightness), color (needs "
          "color), identify, pulse_on, pulse_off. brightness 1-100. color = a color name, white tone "
          "(warm/neutral/cool/daylight), or scene (nightlight, cozy, fireplace, romance...). which = "
          "a light/group name ('bedside', 'living room') or empty for all.",
          {"action": {"type": "string"}, "brightness": {"type": "integer"},
           "color": {"type": "string"}, "which": {"type": "string"}}, ["action"]),
    _tool("check_camera",
          "Look at a Ring camera now: capture a fresh snapshot and describe what's visible. camera = "
          "a name ('bedroom') or empty for the default. Takes a few seconds to wake the camera.",
          {"camera": {"type": "string"}}, []),
    _tool("camera_recent_activity",
          "A Ring camera's recent motion/activity history. camera = name or empty; hours = only "
          "events within the last N hours.",
          {"camera": {"type": "string"}, "hours": {"type": "integer"}}, []),
    _tool("tesla_status",
          "Check the car: battery %, range, charging, locked, climate. Waking the "
          "car takes a few seconds.", {}, []),
    _tool("tesla_climate",
          "Start/stop the Tesla's climate pre-conditioning. on=true to start, false to stop. "
          "temperature = optional target cabin temp in Fahrenheit.",
          {"on": {"type": "boolean"}, "temperature": {"type": "integer"}}, ["on"]),
    _tool("tesla_lock", "Lock or unlock the Tesla. locked=true to lock, false to unlock.",
          {"locked": {"type": "boolean"}}, ["locked"]),
    _tool("tesla_charging",
          "Control Tesla charging. action: 'start', 'stop', or 'limit'. limit = charge limit percent "
          "(with action 'limit').",
          {"action": {"type": "string"}, "limit": {"type": "integer"}}, ["action"]),
    _tool("tesla_find", "Honk or flash the Tesla to locate it. mode: 'honk' or 'flash'.",
          {"mode": {"type": "string"}}, []),
    _tool("tesla_locate", "Where the Tesla is currently parked.", {}, []),
    _tool("feed_cat",
          "Dispense food from the automatic pet feeder. 'feed the cat' / 'feed the pet' = a "
          "normal feeding of 2 portions (leave portions empty). Only set portions if the user asks "
          "for a specific different amount.", {"portions": {"type": "integer"}}, []),
    _tool("cat_feeder_status",
          "Check the PetLibro cat feeder: online, HOPPER food store, how many times it's fed today, "
          "battery, desiccant. This feeder has NO bowl scale — it can't measure food in the bowl; "
          "today's feeding count is the proxy for that.", {}, []),
    _tool("doordash_find_items",
          "Prefer this for requested food or drink. Preserve the user's requested food in item_query; "
          "never substitute an example or different item. Leave store_query empty to infer plausible "
          "restaurant types and scan their menus. Supply it when the user names a store; named-store "
          "searches are strict and must never be replaced with a different merchant.",
          {"store_query": {"type": "string"}, "item_query": {"type": "string"},
           "intent": {"type": "string"}, "limit": {"type": "integer"}},
          ["item_query", "intent"]),
    _tool("doordash_search",
          "Find nearby restaurants by specialty, cuisine, or store name using the saved default "
          "address. Preserve distinctive food terms that identify a restaurant concept.",
          {"store_query": {"type": "string"}, "intent": {"type": "string"},
           "limit": {"type": "integer"}}, ["store_query", "intent"]),
    _tool("doordash_default_address",
          "Get the canonical saved DoorDash delivery address. Use whenever the user asks for their "
          "delivery/default address. Never guess an address or ask for it unless this reports none.",
          {"intent": {"type": "string"}}, ["intent"]),
    _tool("doordash_menu", "Find matching items inside one restaurant's menu.",
          {"store_id": {"type": "string"}, "query": {"type": "string"},
           "intent": {"type": "string"}}, ["store_id", "query", "intent"]),
    _tool("doordash_item_details",
          "Get a restaurant item's required and optional customization choices before cart add.",
          {"store_id": {"type": "string"}, "menu_id": {"type": "string"},
           "item_id": {"type": "string"}, "intent": {"type": "string"}},
          ["store_id", "menu_id", "item_id", "intent"]),
    _tool("doordash_cart",
          "Show one active cart by cart_uuid, or list active carts when cart_uuid is empty; store_id "
          "optionally filters the list.",
          {"intent": {"type": "string"}, "cart_uuid": {"type": "string"},
           "store_id": {"type": "string"}}, ["intent"]),
    _tool("doordash_add_to_cart",
          "Add one identified restaurant item, automatically reusing a same-store cart. When the "
          "user stated modifier preferences, call this directly with configuration_request containing "
          "their natural words; it infers quantities and distinct configurations, then resolves "
          "modifier IDs internally. Never construct modifier JSON or invent option IDs. Never "
          "narrate that you will check modifiers without making this tool call.",
          {"store_id": {"type": "string"}, "menu_id": {"type": "string"},
           "item_id": {"type": "string"}, "item_name": {"type": "string"},
           "quantity": {"type": "integer"}, "intent": {"type": "string"},
           "cart_uuid": {"type": "string"}, "fulfillment": {"type": "string"},
           "configuration_request": {"type": "string"}},
          ["store_id", "menu_id", "item_id", "item_name", "quantity", "intent"]),
    _tool("doordash_remove_from_cart", "Remove one cart line using its cart-line ID.",
          {"cart_uuid": {"type": "string"}, "cart_item_id": {"type": "string"},
           "intent": {"type": "string"}}, ["cart_uuid", "cart_item_id", "intent"]),
    _tool("doordash_delete_cart", "Delete an active cart after the user asks to replace/delete it.",
          {"cart_uuid": {"type": "string"}, "intent": {"type": "string"}},
          ["cart_uuid", "intent"]),
    _tool("doordash_preview_order",
          "Required after the cart is ready and before stating checkout facts. Get canonical pricing, "
          "address, suggested tip, ETA, and masked payment. Use only its exact values; never guess. "
          "State the money details, then ask only whether the address is correct.",
          {"cart_uuid": {"type": "string"}, "intent": {"type": "string"}},
          ["cart_uuid", "intent"]),
    _tool("doordash_submit_order",
          "Charge and place an order after one yes to the disclosed delivery-address question. "
          "Use the suggested tip from doordash_preview_order unless the user requested another "
          "amount. This polls internally. Report success only when order_successful=true and "
          "status=successful. Never retry the same cart. For payment trouble, use the returned "
          "checkout_url and never ask for card number, expiration, security code, or billing "
          "details. tip_cents is cents; 500 is $5. confirmed must be true.",
          {"cart_uuid": {"type": "string"}, "tip_cents": {"type": "integer"},
           "confirmed": {"type": "boolean"}, "intent": {"type": "string"}},
          ["cart_uuid", "tip_cents", "confirmed", "intent"]),
    _tool("doordash_order_status",
          "Check a prior order when the user later asks for an update. Top-level success is true only "
          "when the order itself is successful. Do not call immediately after submit because submit "
          "already polls internally.",
          {"order_uuid": {"type": "string"}, "intent": {"type": "string"}},
          ["order_uuid", "intent"]),
    _tool("doordash_checkout_url",
          "Get browser checkout for payment changes or when the user asks to finish there. This is "
          "the only supported payment-change path; never collect payment credentials in chat.",
          {"cart_uuid": {"type": "string"}, "intent": {"type": "string"}},
          ["cart_uuid", "intent"]),
    _tool("doordash_roulette_prepare",
          "For 'surprise me': randomly prepare a recent favorite under an all-in cap. Read its ONE "
          "address question without revealing the food, then wait. Default cap is 8000 cents.",
          {"intent": {"type": "string"}, "max_total_cents": {"type": "integer"}}, ["intent"]),
    _tool("doordash_roulette_submit",
          "Charge the prepared roulette order only after a clear yes to its one confirmation.",
          {"confirmed": {"type": "boolean"}, "intent": {"type": "string"}},
          ["confirmed", "intent"]),
    _tool("doordash_roulette_cancel", "Delete the prepared surprise cart when the user says no.",
          {"intent": {"type": "string"}}, ["intent"]),
]


def _fmt_age(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 60:
        return "just now"
    m = s // 60
    if m < 60:
        return f"{m}m ago"
    h, rm = divmod(m, 60)
    if h < 24:
        return f"{h}h {rm}m ago" if rm else f"{h}h ago"
    return f"{h // 24}d ago"


def run_tool(name: str, args: dict) -> str:
    try:
        if name == "run_agent":
            return agent_tools.dispatch(args["name"], args["task"])
        if name == "list_sessions":
            rows = agent_tools.list_recent()
            if not rows:
                return "No agent sessions yet."
            now = time.time()
            return "Recent sessions: " + "; ".join(
                f"{r['name']} — {r['status']}, last active {_fmt_age(now - (r.get('updated_at') or now))}"
                for r in rows)
        if name == "get_session_status":
            s = agent_tools.get_state(args["name"])
            if not s:
                return f"No session called {args['name']}."
            age = _fmt_age(time.time() - (s.get("updated_at") or time.time()))
            if s["status"] == "running":
                return f"{args['name']} is still working on it (last active {age})."
            out = (s.get("last_output") or "").strip()
            return f"{args['name']} is {s['status']}, last active {age}. Result: {out[-3000:]}"
        if name == "message_session":
            return agent_tools.send_message(args["name"], args["message"])
        if name == "delete_session":
            return agent_tools.delete_session(args["name"])
        if name == "control_tv":
            return tv_tools.control(args["action"], args.get("level"))
        if name == "play_on_tv":
            # slow (search + navigate ~7-17s); background it and confirm immediately, like voice.
            title, service = args["title"], args.get("service")
            threading.Thread(target=tv_tools.play, args=(title, service), daemon=True).start()
            where = f" on {service}" if service else ""
            return f"Starting '{title}'{where} on the TV — it'll come up in a few seconds."
        if name == "thermostat_status":
            return nest_tools.status()
        if name == "set_thermostat_temperature":
            return nest_tools.set_temperature(args["temperature"], args.get("target"))
        if name == "set_thermostat_mode":
            return nest_tools.set_mode(args["mode"])
        if name == "roomba_status":
            return roomba_tools.status()
        if name == "control_roomba":
            # commands can take a few seconds to connect; background + confirm immediately
            action = args["action"]
            threading.Thread(target=roomba_tools.control, args=(action,), daemon=True).start()
            return f"On it — {action} the Roomba."
        # --- smart lights (sync; text chat has no voice-pulse to coordinate with) ---
        if name == "lights_status":
            return wiz_tools.status(args.get("which"))
        if name == "control_lights":
            return wiz_tools.control(args["action"], args.get("brightness"),
                                     args.get("color"), args.get("which"))
        # --- async cloud tools (Ring / Tesla / PetLibro), run on the shared loop ---
        if name == "check_camera":
            desc, cam = _await(ring_tools.look(args.get("camera") or ""))
            return (f"The {cam} camera shows: {desc}" if desc
                    else f"Couldn't get an image from the {args.get('camera') or 'camera'} just now.")
        if name == "camera_recent_activity":
            evs, cam = _await(ring_tools.recent_events(args.get("camera") or "", limit=10))
            if not evs:
                return f"No recent activity on the {cam or args.get('camera') or 'camera'}."
            now = time.time()
            hours = args.get("hours")
            cutoff = now - hours * 3600 if hours else 0
            lines = [f"{e.get('kind', 'event')} {_fmt_age(now - e['time'])}"
                     for e in evs if e.get("time") and (not cutoff or e["time"] >= cutoff)]
            if not lines:
                return f"No activity on the {cam} camera in {('the last %d hour(s)' % hours) if hours else 'that window'}."
            return f"{cam} camera activity: " + "; ".join(lines) + "."
        if name == "tesla_status":
            return _await(tesla_tools.status())
        if name == "tesla_climate":
            return _await(tesla_tools.climate(args["on"], args.get("temperature")))
        if name == "tesla_lock":
            return _await(tesla_tools.lock(args["locked"]))
        if name == "tesla_charging":
            return _await(tesla_tools.charge(args["action"], args.get("limit")))
        if name == "tesla_find":
            return _await(tesla_tools.find(args.get("mode") or "flash"))
        if name == "tesla_locate":
            return _await(tesla_tools.locate())
        if name == "feed_cat":
            return _await(petlibro_tools.feed(args.get("portions") or 2))
        if name == "cat_feeder_status":
            return _await(petlibro_tools.status())
        if name == "doordash_find_items":
            return doordash_tools.find_items(
                args["item_query"], args["intent"], args.get("store_query") or "",
                args.get("limit") or 5)
        if name == "doordash_search":
            return doordash_tools.search(
                args["store_query"], args["intent"], args.get("limit") or 5)
        if name == "doordash_default_address":
            return doordash_tools.default_address(args["intent"])
        if name == "doordash_menu":
            return doordash_tools.menu(args["store_id"], args["query"], args["intent"])
        if name == "doordash_item_details":
            return doordash_tools.item_details(
                args["store_id"], args["menu_id"], args["item_id"], args["intent"])
        if name == "doordash_cart":
            return doordash_tools.cart(
                args["intent"], args.get("cart_uuid") or "", args.get("store_id") or "")
        if name == "doordash_add_to_cart":
            return doordash_tools.add_to_cart(
                args["store_id"], args["menu_id"], args["item_id"], args["item_name"],
                args["quantity"], args["intent"], args.get("cart_uuid") or "",
                args.get("fulfillment") or "delivery", "",
                "", args.get("configuration_request") or "")
        if name == "doordash_remove_from_cart":
            return doordash_tools.remove_from_cart(
                args["cart_uuid"], args["cart_item_id"], args["intent"])
        if name == "doordash_delete_cart":
            return doordash_tools.delete_cart(args["cart_uuid"], args["intent"])
        if name == "doordash_preview_order":
            return doordash_tools.preview_order(args["cart_uuid"], args["intent"])
        if name == "doordash_submit_order":
            return doordash_tools.submit_order(
                args["cart_uuid"], args["tip_cents"], bool(args["confirmed"]), args["intent"])
        if name == "doordash_order_status":
            return doordash_tools.order_status(args["order_uuid"], args["intent"])
        if name == "doordash_checkout_url":
            return doordash_tools.checkout_url(args["cart_uuid"], args["intent"])
        if name == "doordash_roulette_prepare":
            return doordash_tools.roulette_prepare(
                args["intent"], args.get("max_total_cents") or 0)
        if name == "doordash_roulette_submit":
            return doordash_tools.roulette_submit(bool(args["confirmed"]), args["intent"])
        if name == "doordash_roulette_cancel":
            return doordash_tools.roulette_cancel(args["intent"])
        return f"Unknown tool: {name}."
    except Exception as e:
        return f"That tool failed ({type(e).__name__})."


def _llm(messages):
    tools = [t for t in _ALL_TOOLS if t["function"]["name"] in config.enabled_tool_names()]
    body = {"model": LLM_MODEL, "messages": messages, "tools": tools, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False}}
    if os.environ.get("LLM_REASONING_EFFORT"):   # thinking models (qwen3.5 on Ollama)
        body["reasoning_effort"] = os.environ["LLM_REASONING_EFFORT"]
    req = urllib.request.Request(
        LLM_BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY}"})
    resp = json.load(urllib.request.urlopen(req, timeout=120))
    return resp["choices"][0]["message"]


def run_chat(client_messages, persona=None):
    """Generator of SSE event dicts for one text-chat turn. `persona` selects the
    personality (kronik/consuela) — same tools, different voice/style."""
    try:
        system = personas.build_text_instructions(
            personas.resolve(persona), config.enabled_integrations())
        messages = [{"role": "system", "content": system}]
        for m in (client_messages or [])[-24:]:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        if len(messages) < 2:
            yield {"type": "error", "message": "empty message"}
            return

        for _ in range(MAX_TOOL_ITERS):
            msg = _llm(messages)
            tcs = msg.get("tool_calls")
            if tcs:
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
                for tc in tcs:
                    fn = tc.get("function", {})
                    name = fn.get("name") or ""
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    yield {"type": "tool", "name": name}
                    result = run_tool(name, args)
                    messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": result})
                continue
            final = (msg.get("content") or "").strip()
            if not final:
                final = "…"
            for i, word in enumerate(final.split(" ")):
                yield {"type": "delta", "text": (word if i == 0 else " " + word)}
                time.sleep(0.012)   # gentle typewriter pacing
            yield {"type": "done"}
            return

        yield {"type": "delta", "text": "Hmm, I got tangled up on that one — try rephrasing?"}
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
