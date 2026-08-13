"""Typed, shell-free bridge to DoorDash's official ``dd-cli`` binary."""

import json
import logging
import os
import random
import subprocess
import threading
import time
import urllib.request
import fcntl
from concurrent.futures import ThreadPoolExecutor


_DEFAULT_BINARY = os.path.expanduser("~/.local/bin/dd-cli")
_MAX_RESULT_CHARS = 30_000
_ROULETTE_TTL_SECONDS = 15 * 60
_ORDER_STATUS_POLL_ATTEMPTS = 6
_ORDER_STATUS_POLL_SECONDS = 1.0
_DEFAULT_ADDRESS_TTL_SECONDS = 60
_ORDER_HISTORY_TTL_SECONDS = 2 * 60
_MENU_TTL_SECONDS = 3 * 60
_MAX_MENU_WORKERS = 5
_SUBMITTED_CARTS_PATH = os.path.expanduser(
    os.environ.get("DOORDASH_SUBMITTED_CARTS_PATH")
    or "~/.local/state/kronik/doordash-submitted-carts"
)
_roulette_lock = threading.Lock()
_roulette_pending = None
_cache_lock = threading.Lock()
_address_cache = {}
_history_cache = {}
_menu_cache = {}
_log = logging.getLogger("kronik.doordash")


def _cache_get(cache: dict, key):
    now = time.monotonic()
    with _cache_lock:
        entry = cache.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at <= now:
            cache.pop(key, None)
            return None
        return value


def _cache_put(cache: dict, key, value, ttl_seconds: float):
    with _cache_lock:
        cache[key] = (time.monotonic() + ttl_seconds, value)


def _invoke_operation(args: list[str]) -> str:
    if not args:
        return "unknown"
    if args[0] in {"address", "cart", "order", "payment-method"} and len(args) > 1:
        return f"{args[0]}_{args[1]}"
    return args[0]


def _binary() -> str:
    return os.path.expanduser(os.environ.get("DD_CLI_PATH") or _DEFAULT_BINARY)


def _intent_value(intent: str) -> str:
    request = (intent or "").strip()
    if not request:
        raise ValueError("DoorDash needs the user's original request in the intent field.")
    return (
        "Summary: Help the user complete their DoorDash request\n"
        f"user prompt/purpose: {json.dumps(request[:1000])}"
    )


def _invoke(args: list[str], intent: str, *, text_output: bool = False):
    started = time.perf_counter()
    operation = _invoke_operation(args)

    def finish(data, error):
        _log.info(
            "doordash_cli operation=%s duration_ms=%d success=%s",
            operation,
            round((time.perf_counter() - started) * 1000),
            not bool(error),
        )
        return data, error

    binary = _binary()
    if not os.path.isfile(binary):
        return finish(None, f"DoorDash CLI is not installed at {binary}.")
    try:
        intent_value = _intent_value(intent)
    except ValueError as exc:
        return finish(None, str(exc))

    argv = [binary] + ([] if text_output else ["--json-output"]) + args
    argv.extend(["--intent", intent_value])
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return finish(None, "DoorDash timed out after 90 seconds.")
    except OSError as exc:
        return finish(None, f"DoorDash could not start: {exc}.")

    output = (proc.stdout or "").strip()
    error = (proc.stderr or "").strip()
    if proc.returncode:
        detail = error or output or f"exit code {proc.returncode}"
        return finish(None, f"DoorDash failed: {detail[-4000:]}")
    if text_output:
        return finish(output or "DoorDash completed successfully.", None)
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return finish(None, "DoorDash returned an unreadable response.")
    return finish(payload.get("structuredContent", payload), None)


def _result(data) -> str:
    output = json.dumps(data, separators=(",", ":"))
    if len(output) > _MAX_RESULT_CHARS:
        output = output[:_MAX_RESULT_CHARS] + "\n[Result truncated; narrow the DoorDash query.]"
    return output


def _failure(data, error: str | None, action: str) -> str | None:
    if error:
        return error
    if isinstance(data, dict) and data.get("success") is False:
        return f"DoorDash could not {action}: {data.get('message') or 'unknown error'}"
    return None


def _compact_text(value: str) -> str:
    return "".join(char.casefold() for char in str(value) if char.isalnum())


def _default_address(intent: str):
    cached = _cache_get(_address_cache, "default")
    if cached is not None:
        _log.info("doordash_cache resource=default_address hit=true")
        return cached, None

    data, error = _invoke(["address", "list"], intent)
    failure = _failure(data, error, "read the saved addresses")
    if failure:
        return None, failure
    addresses = data.get("addresses") or []
    address = next((item for item in addresses if item.get("is_default") is True), None)
    if not address:
        return None, "DoorDash has no identifiable default delivery address. Choose one in DoorDash first."
    if address.get("lat") is None or address.get("lng") is None:
        return None, "The default DoorDash address has no usable location coordinates."
    _cache_put(_address_cache, "default", address, _DEFAULT_ADDRESS_TTL_SECONDS)
    return address, None


def default_address(intent: str) -> str:
    """Return the canonical saved delivery address without exposing location coordinates."""
    address, error = _default_address(intent)
    if error:
        return error
    return _result({
        "success": True,
        "default_address": address.get("printable_address"),
    })


def search(store_query: str, intent: str, limit: int = 5) -> str:
    """Search restaurant/store names near the saved default address."""
    store_query = (store_query or "").strip()
    if not store_query:
        return "Tell me what kind of restaurant or store to search for."
    address, error = _default_address(intent)
    if error:
        return error
    limit = min(10, max(1, int(limit or 5)))
    data, error = _invoke([
        "search", "--query", store_query,
        "--lat", str(address["lat"]), "--lng", str(address["lng"]),
        "--limit", str(limit),
    ], intent)
    failure = _failure(data, error, "search nearby restaurants")
    if failure:
        return failure
    stores = [{
        key: store.get(key)
        for key in ("store_id", "name", "distance", "delivery_time", "rating", "review_count")
        if store.get(key) is not None
    } for store in (data.get("stores") or [])]
    return _result({"success": True, "store_query": store_query, "stores": stores})


def _menu_data(store_id: str, intent: str):
    store_id = str(store_id)
    cached = _cache_get(_menu_cache, store_id)
    if cached is not None:
        _log.info("doordash_cache resource=menu hit=true")
        return cached, None

    data, error = _invoke(["menu", "--store-id", str(store_id)], intent)
    failure = _failure(data, error, "retrieve that menu")
    if failure:
        return None, failure
    _cache_put(_menu_cache, store_id, data, _MENU_TTL_SECONDS)
    return data, None


def menu(store_id: str, query: str, intent: str) -> str:
    """Fetch a restaurant menu and return only items matching the requested words."""
    data, error = _menu_data(store_id, intent)
    if error:
        return error

    words = [word for word in (query or "").casefold().split() if word]
    matches = []
    for item in data.get("items") or []:
        searchable = " ".join(str(item.get(key) or "") for key in (
            "name", "description", "category_name",
        )).casefold()
        if words and not all(word in searchable for word in words):
            continue
        matches.append({
            key: item.get(key)
            for key in (
                "item_id", "name", "price", "description", "category_name",
                "has_modifiers", "has_required_modifiers", "is_orderable",
            )
            if item.get(key) is not None
        })
    total = len(matches)
    requested_name = _compact_text(query)
    matches.sort(key=lambda item: (
        _compact_text(item.get("name") or "") != requested_name,
        abs(len(_compact_text(item.get("name") or "")) - len(requested_name)),
    ))
    matches = matches[:40]
    return _result({
        "success": True,
        "store_id": str(store_id),
        "menu_id": data.get("menu_id"),
        "query": query,
        "match_count": total,
        "items": matches,
        "more_matches": total > len(matches),
    })


def _local_json(system_prompt: str, payload: dict, *, max_tokens: int = 120,
                operation: str = "local_json"):
    """Run one small structured inference request against the configured local model."""
    started = time.perf_counter()

    def finish(data, error):
        _log.info(
            "doordash_inference operation=%s duration_ms=%d success=%s",
            operation,
            round((time.perf_counter() - started) * 1000),
            not bool(error),
        )
        return data, error

    base_url = (
        os.environ.get("VOICE_LLM_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://127.0.0.1:8080/v1"
    ).rstrip("/")
    model = os.environ.get("VOICE_LLM_MODEL") or os.environ.get("LLM_MODEL")
    if not model:
        return finish(None, "No local inference model is configured.")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_payload = json.load(response)
        content = str(response_payload["choices"][0]["message"]["content"] or "").strip()
        try:
            return finish(json.loads(content), None)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                raise
            return finish(json.loads(content[start:end + 1]), None)
    except Exception as exc:
        return finish(None, f"{type(exc).__name__}: {exc}")


def _infer_item_search(item_query: str):
    """Infer a canonical menu phrase and restaurant searches likely to sell an item."""
    inferred, error = _local_json(
        (
            "Infer DoorDash restaurant search queries for a requested prepared food or "
            "drink. Return JSON only as "
            "{\"menu_query\":\"...\",\"queries\":[\"...\"]}. menu_query is the concise, "
            "canonical catalog name for the same requested item. Omit quantity, articles, size, "
            "temperature, and modifier wording such as iced, hot, or milk type because those are "
            "handled during item configuration; never substitute a different item. Give three "
            "distinct, concise restaurant types, cuisines, or merchant categories likely to "
            "sell it. The first query must be the broad seller or merchant type most likely "
            "to carry it, never the exact item name. Rank the rest by likelihood. Use "
            "categories that restaurant search can find. A later query may use the item name "
            "when it is itself a cuisine, restaurant concept, or brand."
        ),
        {"item_query": item_query[:300]},
        operation="item_search",
    )
    if error:
        return item_query, [], error

    menu_query = str(inferred.get("menu_query") or "").strip() or item_query

    queries = []
    seen = set()
    for value in inferred.get("queries") or []:
        query = str(value or "").strip()
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) == 4:
            break
    if not queries:
        return menu_query, [], "The local model returned no usable restaurant searches."
    return menu_query, queries, None


def _merchant_name_matches(query: str, name: str) -> bool:
    requested = _compact_text(query)
    candidate = _compact_text(name)
    return bool(requested and candidate and (requested in candidate or candidate in requested))


def _best_item_matches(items, query: str):
    requested = _compact_text(query)
    exact = [item for item in items if _compact_text(item.get("name") or "") == requested]
    return exact or items


def _order_history(intent: str):
    cached = _cache_get(_history_cache, "merchant_lookup")
    if cached is not None:
        _log.info("doordash_cache resource=order_history hit=true")
        return cached, None
    data, error = _invoke([
        "order", "history", "--max", "100", "--days", "365",
    ], intent)
    if not error and isinstance(data, dict):
        _cache_put(_history_cache, "merchant_lookup", data, _ORDER_HISTORY_TTL_SECONDS)
    return data, error


def find_items(item_query: str, intent: str, store_query: str = "", limit: int = 5) -> str:
    """Find a restaurant item, inferring likely seller searches when none is supplied."""
    item_query = (item_query or "").strip()
    if not item_query:
        return "Tell me which food or drink to find."
    results = []
    menu_errors = []
    stores_checked = 0
    seen_stores = set()
    named_store_seen = False
    store_query = (store_query or "").strip()
    inference_error = None
    menu_query = item_query
    history = None
    history_error = None
    if store_query:
        # Merchant history is the reliable fallback for stores that DoorDash search omits. Run it
        # alongside the small item-normalization inference, then use an exact history match before
        # paying for a slower merchant search.
        with ThreadPoolExecutor(max_workers=2) as pool:
            inference_future = pool.submit(_infer_item_search, item_query)
            history_future = pool.submit(_order_history, intent)
            menu_query, _, inference_error = inference_future.result()
            history, history_error = history_future.result()
        store_queries = [store_query]
    else:
        menu_query, store_queries, inference_error = _infer_item_search(item_query)

    seen_queries = {query.casefold() for query in store_queries}
    if not store_query and item_query.casefold() not in seen_queries:
        store_queries.append(item_query)
    attempts = [(query, limit if index == 0 else 10)
                for index, query in enumerate(store_queries)]

    def check_stores(stores):
        nonlocal stores_checked, named_store_seen
        candidates = []
        for store in stores:
            if store_query and not _merchant_name_matches(store_query, store.get("name") or ""):
                continue
            if store_query:
                named_store_seen = True
            store_id = str(store.get("store_id") or "")
            if not store_id or store_id in seen_stores:
                continue
            seen_stores.add(store_id)
            candidates.append(store)

        stores_checked += len(candidates)
        if not candidates:
            return

        def inspect(store):
            store_id = str(store["store_id"])
            menu_result = menu(store_id, menu_query, intent)
            try:
                menu_data = json.loads(menu_result)
            except json.JSONDecodeError:
                return None, store.get("name") or store_id
            items = _best_item_matches(menu_data.get("items") or [], menu_query)
            if not items:
                return None, None
            return {
                "store": store,
                "menu_id": menu_data.get("menu_id"),
                "match_count": menu_data.get("match_count"),
                "items": items,
            }, None

        # Menu downloads are independent. Preserve DoorDash's store ordering in the response,
        # while allowing the slow network calls themselves to overlap.
        with ThreadPoolExecutor(max_workers=min(_MAX_MENU_WORKERS, len(candidates))) as pool:
            inspected = [(store, pool.submit(inspect, store)) for store in candidates]
            for _, future in inspected:
                match, menu_error = future.result()
                if menu_error:
                    menu_errors.append(menu_error)
                elif match is not None and len(results) < 3:
                    results.append(match)

    queries_tried = []
    history_fallback_used = False
    if store_query and not history_error and isinstance(history, dict):
        history_stores = []
        for order in history.get("orders") or []:
            store_id = str(order.get("store_id") or "")
            store_name = str(order.get("store_name") or "")
            if store_id and _merchant_name_matches(store_query, store_name):
                history_stores.append({
                    "store_id": store_id,
                    "name": store_name,
                    "source": "order_history",
                })
        if history_stores:
            history_fallback_used = True
            check_stores(history_stores)

    if not results:
        for query, query_limit in attempts:
            queries_tried.append(query)
            stores_result = search(query, intent, query_limit)
            try:
                stores_data = json.loads(stores_result)
            except json.JSONDecodeError:
                return stores_result
            check_stores(stores_data.get("stores") or [])
            if results:
                break
    named_store_found = named_store_seen if store_query else None
    if store_query and not named_store_found:
        next_action = (
            f"DoorDash did not return {store_query!r} near the saved delivery address. "
            "Do not substitute another merchant and do not claim the store is closed or nonexistent."
        )
    elif store_query and not results:
        next_action = (
            f"DoorDash found {store_query!r}, but its menu did not match {menu_query!r}. "
            "Do not substitute another merchant."
        )
    else:
        next_action = (
            "Choose an exact result and call doordash_add_to_cart. No item has been added yet."
        )
    return _result({
        "success": True,
        "cart_modified": False,
        "store_query": store_query or None,
        "named_store_found": named_store_found,
        "item_query": item_query,
        "menu_query": menu_query,
        "inferred_store_queries": store_queries if not store_query else [],
        "inference_error": inference_error,
        "store_queries_tried": queries_tried,
        "stores_checked": stores_checked,
        "history_fallback_used": history_fallback_used,
        "results": results,
        "menu_errors": menu_errors,
        "next_action": next_action,
    })


def item_details(store_id: str, menu_id: str, item_id: str, intent: str) -> str:
    """Get an item's required and optional customization choices."""
    data, error = _invoke([
        "restaurant-item-details",
        "--store-id", str(store_id),
        "--menu-id", str(menu_id),
        "--item-id", str(item_id).removeprefix("i_"),
    ], intent)
    failure = _failure(data, error, "retrieve that item's options")
    if failure:
        return failure
    item = data.get("item") or {}
    item.pop("image_url", None)
    item.pop("is_popular", None)
    item.pop("popularity_rank", None)
    item.pop("popular_modifications", None)
    return _result({"success": True, "item": item})


def cart(intent: str, cart_uuid: str = "", store_id: str = "") -> str:
    """Show one active cart, or list active carts."""
    if cart_uuid:
        args = ["cart", "show", "--cart-uuid", str(cart_uuid)]
        action = "retrieve that cart"
    else:
        args = ["cart", "list"]
        if store_id:
            args.extend(["--store-id", str(store_id)])
        action = "list active carts"
    data, error = _invoke(args, intent)
    failure = _failure(data, error, action)
    return failure or _result(data)


def _compact_modifier_groups(groups):
    compact = []
    for group in groups or []:
        options = []
        for option in group.get("options") or []:
            entry = {
                "id": str(option.get("option_id") or option.get("id") or ""),
                "name": str(option.get("name") or ""),
            }
            children = _compact_modifier_groups(option.get("extras") or [])
            if children:
                entry["groups"] = children
            options.append(entry)
        compact.append({
            "title": str(group.get("title") or ""),
            "min": int(group.get("min_num_options") or 0),
            "max": int(group.get("max_num_options") or 0),
            "options": options,
        })
    return compact


def _canonical_modifier_options(selected, groups):
    if not isinstance(selected, list):
        raise ValueError("nested_options must be a JSON array")

    selected_by_id = {}
    for entry in selected:
        if not isinstance(entry, dict):
            raise ValueError("each selected modifier must be an object")
        option_id = str(entry.get("id") or "")
        if not option_id or option_id in selected_by_id:
            raise ValueError("selected modifier IDs must be present and unique")
        selected_by_id[option_id] = entry

    known_ids = {
        str(option.get("id") or "")
        for group in groups
        for option in (group.get("options") or [])
    }
    unknown = set(selected_by_id) - known_ids
    if unknown:
        raise ValueError("the model selected a modifier ID outside the item catalog")

    canonical = []
    for group in groups:
        group_options = group.get("options") or []
        chosen = [option for option in group_options if option.get("id") in selected_by_id]
        minimum = int(group.get("min") or 0)
        maximum = int(group.get("max") or 0)
        if len(chosen) < minimum or (maximum and len(chosen) > maximum):
            raise ValueError(f"modifier group {group.get('title')!r} has an invalid selection count")
        for option in chosen:
            requested = selected_by_id[option["id"]]
            output = {"id": option["id"], "name": option["name"], "quantity": 1}
            child_groups = option.get("groups") or []
            child_selected = requested.get("options") or []
            if child_groups:
                output["options"] = _canonical_modifier_options(child_selected, child_groups)
            elif child_selected:
                raise ValueError("a leaf modifier cannot contain nested selections")
            canonical.append(output)
    return canonical


def _configured_cart_items(store_id: str, menu_id: str, item_id: str, item_name: str,
                           quantity: int, intent: str, configurations_json: str):
    try:
        configurations = json.loads(configurations_json)
    except json.JSONDecodeError:
        return None, "DoorDash configurations_json must be a JSON array."
    if not isinstance(configurations, list) or not configurations:
        return None, "DoorDash configurations_json must be a non-empty JSON array."

    normalized = []
    total_quantity = 0
    for configuration in configurations:
        if not isinstance(configuration, dict):
            return None, "Each DoorDash item configuration must be an object."
        configured_quantity = int(configuration.get("quantity") or 1)
        preferences = configuration.get("preferences") or []
        if configured_quantity < 1 or not isinstance(preferences, list):
            return None, "Each configuration needs a positive quantity and a preferences array."
        preferences = [str(value).strip() for value in preferences if str(value).strip()]
        normalized.append({"quantity": configured_quantity, "preferences": preferences})
        total_quantity += configured_quantity
    if total_quantity != quantity:
        return None, "Configuration quantities must add up to the requested item quantity."

    details, error = _invoke([
        "restaurant-item-details",
        "--store-id", str(store_id),
        "--menu-id", str(menu_id),
        "--item-id", str(item_id).removeprefix("i_"),
    ], intent)
    failure = _failure(details, error, "retrieve that item's options")
    if failure:
        return None, failure
    item = (details or {}).get("item") or {}
    modifier_groups = _compact_modifier_groups(item.get("extras") or [])

    resolved, inference_error = _local_json(
        (
            "Resolve natural-language preferences to DoorDash modifier options. Return JSON only "
            "as {\"items\":[{\"quantity\":1,\"nested_options\":[...]}]}. Return one item for "
            "each input configuration, in the same order and with the same quantity. Every selected "
            "option must use an exact id and name from the supplied modifier catalog and quantity 1. "
            "Nested child choices go in the parent option's options array. Satisfy every group whose "
            "min is greater than zero. Honor every stated preference. For an unspecified required "
            "group, choose the most ordinary neutral option. Do not add optional extras unless a "
            "preference requests them or they are the required parent of a requested child choice."
        ),
        {
            "item_name": item.get("name") or item_name,
            "configurations": normalized,
            "modifier_catalog": modifier_groups,
        },
        max_tokens=1600,
        operation="modifier_resolution",
    )
    if inference_error:
        return None, f"DoorDash could not resolve the item preferences: {inference_error}"
    resolved_items = resolved.get("items") or []
    if len(resolved_items) != len(normalized):
        return None, "DoorDash preference inference returned the wrong number of configured items."

    cart_items = []
    for requested, inferred in zip(normalized, resolved_items):
        if int(inferred.get("quantity") or 0) != requested["quantity"]:
            return None, "DoorDash preference inference changed an item quantity."
        try:
            nested_options = _canonical_modifier_options(
                inferred.get("nested_options") or [], modifier_groups)
        except ValueError as exc:
            return None, f"DoorDash could not validate the inferred item preferences: {exc}."
        cart_items.append({
            "item_id": str(item_id).removeprefix("i_"),
            "item_name": str(item_name),
            "quantity": requested["quantity"],
            "nested_options": nested_options,
        })
    return cart_items, None


def _configuration_groups_from_request(item_name: str, intent: str, quantity_hint: int,
                                       configuration_request: str):
    inferred, error = _local_json(
        (
            "Convert a user's restaurant-item request into configuration groups. Return JSON only "
            "as {\"configurations\":[{\"quantity\":1,\"preferences\":[\"...\"]}]}. Make one "
            "group per distinct set of preferences and combine identical units. The original_intent "
            "and configuration_request are authoritative; quantity_hint is only a fallback and may "
            "be wrong. Preserve the requested total quantity and every stated modifier. Do not "
            "invent preferences the user did not state."
        ),
        {
            "item_name": item_name,
            "original_intent": intent[:1000],
            "quantity_hint": quantity_hint,
            "configuration_request": configuration_request[:1500],
        },
        max_tokens=500,
        operation="configuration_parsing",
    )
    if error:
        return None, 0, f"DoorDash could not understand the item configuration: {error}"
    configurations = inferred.get("configurations") or []
    if not isinstance(configurations, list) or not configurations:
        return None, 0, "DoorDash could not identify any configured items from the request."
    total = 0
    normalized = []
    for configuration in configurations:
        if not isinstance(configuration, dict):
            return None, 0, "DoorDash inferred an invalid item configuration."
        configured_quantity = int(configuration.get("quantity") or 0)
        preferences = configuration.get("preferences") or []
        if configured_quantity < 1 or not isinstance(preferences, list):
            return None, 0, "DoorDash inferred an invalid item quantity or preference list."
        preferences = [str(value).strip() for value in preferences if str(value).strip()]
        normalized.append({"quantity": configured_quantity, "preferences": preferences})
        total += configured_quantity
    if total < 1 or total > 20:
        return None, 0, "DoorDash inferred a total item quantity outside the allowed range."
    return normalized, total, None


def add_to_cart(store_id: str, menu_id: str, item_id: str, item_name: str,
                quantity: int, intent: str, cart_uuid: str = "",
                fulfillment: str = "delivery", customizations_json: str = "",
                configurations_json: str = "", configuration_request: str = "") -> str:
    """Add one fully identified restaurant item to a cart."""
    quantity = int(quantity or 1)
    if quantity < 1 or quantity > 20:
        return "DoorDash item quantity must be between 1 and 20."
    if fulfillment not in ("delivery", "pickup"):
        return "DoorDash fulfillment must be delivery or pickup."

    if cart_uuid and _cart_was_submitted(cart_uuid):
        return _submitted_cart_result(cart_uuid, intent)

    if not cart_uuid:
        existing, error = _invoke(["cart", "list", "--store-id", str(store_id)], intent)
        failure = _failure(existing, error, "check for an existing cart")
        if failure:
            return failure
        carts = existing.get("carts") or []
        if carts:
            cart_uuid = str(carts[0].get("cart_uuid") or "")
            if not cart_uuid:
                return "DoorDash found an active cart but did not return its cart ID."
            if _cart_was_submitted(cart_uuid):
                return _submitted_cart_result(cart_uuid, intent)

    item = {
        "item_id": str(item_id).removeprefix("i_"),
        "item_name": str(item_name),
        "quantity": quantity,
    }
    supplied_configuration_paths = sum(bool(value) for value in (
        customizations_json, configurations_json, configuration_request,
    ))
    if supplied_configuration_paths > 1:
        return "Use only one DoorDash item-configuration input."
    if not supplied_configuration_paths:
        configuration_request = intent
    if customizations_json:
        try:
            options = json.loads(customizations_json)
        except json.JSONDecodeError:
            return "DoorDash customizations_json must be a JSON array from item option choices."
        if not isinstance(options, list):
            return "DoorDash customizations_json must be a JSON array."
        item["nested_options"] = options

    items = [item]
    if configuration_request:
        configurations, inferred_quantity, configuration_error = _configuration_groups_from_request(
            item_name, intent, quantity, configuration_request)
        if configuration_error:
            return _result({
                "success": False,
                "item_added": False,
                "error_message": configuration_error,
            })
        quantity = inferred_quantity
        configurations_json = json.dumps(configurations, separators=(",", ":"))
    if configurations_json:
        items, configuration_error = _configured_cart_items(
            store_id, menu_id, item_id, item_name, quantity, intent, configurations_json)
        if configuration_error:
            return _result({
                "success": False,
                "item_added": False,
                "error_message": configuration_error,
            })

    args = [
        "cart", "add-items",
        "--store-id", str(store_id),
        "--menu-id", str(menu_id),
        "--items-json", json.dumps(items, separators=(",", ":")),
    ]
    if cart_uuid:
        args.extend(["--cart-uuid", str(cart_uuid)])
    else:
        args.extend(["--fulfillment", fulfillment])
    data, error = _invoke(args, intent)
    if error:
        return error
    if isinstance(data, dict) and data.get("success") is not False and data.get("cart_uuid"):
        data = dict(data)
        data["next_action"] = (
            "Call doordash_preview_order with this cart_uuid before stating checkout details."
        )
    return _result(data)


def remove_from_cart(cart_uuid: str, cart_item_id: str, intent: str) -> str:
    data, error = _invoke([
        "cart", "remove-item",
        "--cart-uuid", str(cart_uuid),
        "--cart-item-id", str(cart_item_id),
    ], intent)
    failure = _failure(data, error, "remove that item")
    return failure or _result(data)


def delete_cart(cart_uuid: str, intent: str) -> str:
    data, error = _invoke([
        "cart", "delete", "--cart-uuid", str(cart_uuid),
    ], intent)
    failure = _failure(data, error, "delete that cart")
    return failure or _result(data)


def _default_payment_details(intent: str):
    data, error = _invoke(["payment-method", "list"], intent)
    if error or not isinstance(data, dict):
        return None
    default_id = data.get("default_payment_method_id")
    card = next((item for item in (data.get("cards") or [])
                 if item.get("payment_method_id") == default_id), None)
    if not card:
        return None
    try:
        expiration = (int(card.get("exp_year")), int(card.get("exp_month")))
        current_month = (time.localtime().tm_year, time.localtime().tm_mon)
        expired = expiration < current_month
    except (TypeError, ValueError):
        expired = False
    return {
        "brand": card.get("brand") or "card",
        "last4": card.get("last4") or "unknown",
        "exp_month": card.get("exp_month"),
        "exp_year": card.get("exp_year"),
        "expired": expired,
    }


def _default_payment_text(card) -> str:
    if not card:
        return "Default payment could not be identified; it may be a wallet. Offer browser checkout."
    if card["expired"]:
        expiration = f"{int(card['exp_month']):02d}/{card['exp_year']}"
        return (
            f"Default payment: {card['brand']} ending {card['last4']} expired {expiration}. "
            "Do not submit or ask for card details. The user must update payment in browser checkout."
        )
    return f"Default payment: {card['brand']} ending {card['last4']}."


def _preview_data(cart_uuid: str, intent: str):
    return _invoke(["order", "preview", "--cart-uuid", str(cart_uuid)], intent)


def _suggested_tip_cents(quote: dict):
    groups = quote.get("tips_suggestion_details") or []
    if not groups:
        return None
    group = groups[0]
    values = group.get("percentage_to_amount_monetary_values") or []
    try:
        index = int(group.get("default_index"))
        return int(values[index]["unit_amount"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _cleanup_cart(cart_uuid: str, intent: str):
    if cart_uuid:
        _invoke(["cart", "delete", "--cart-uuid", str(cart_uuid)], intent)


def _claim_cart_submit(cart_uuid: str) -> bool:
    """Atomically remember a submit attempt across agent processes and restarts."""
    directory = os.path.dirname(_SUBMITTED_CARTS_PATH)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    with open(_SUBMITTED_CARTS_PATH, "a+", encoding="utf-8") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX)
        ledger.seek(0)
        if cart_uuid in {line.strip() for line in ledger if line.strip()}:
            return False
        ledger.write(cart_uuid + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())
        return True


def _cart_was_submitted(cart_uuid: str) -> bool:
    if not cart_uuid or not os.path.isfile(_SUBMITTED_CARTS_PATH):
        return False
    with open(_SUBMITTED_CARTS_PATH, encoding="utf-8") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_SH)
        return cart_uuid in {line.strip() for line in ledger if line.strip()}


def _submitted_cart_result(cart_uuid: str, intent: str) -> str:
    return _result({
        "success": False,
        "item_added": False,
        "status": "previously_submitted_cart",
        "action_required": True,
        "error_message": (
            "This existing cart was already submitted once, so it cannot be changed or submitted "
            "again from the agent."
        ),
        "next_action": "Check this cart in DoorDash. Do not claim the item was added.",
    })


def preview_order(cart_uuid: str, intent: str, include_checkout_url: bool = True) -> str:
    preview, error = _invoke([
        "order", "preview", "--cart-uuid", str(cart_uuid), "--beautify",
    ], intent, text_output=True)
    if error:
        return error
    payment = _default_payment_details(intent)
    if payment and payment["expired"]:
        expiration = f"{int(payment['exp_month']):02d}/{payment['exp_year']}"
        browser = " Open DoorDash to update the payment method."
        if include_checkout_url:
            checkout, checkout_error = _checkout_url_value(cart_uuid, intent)
            browser = f" Browser checkout: {checkout}" if checkout else ""
            if checkout_error:
                browser = f" Browser checkout could not be created: {checkout_error}"
        return (
            preview + "\n" +
            f"Payment update required: {payment['brand']} ending {payment['last4']} expired "
            f"{expiration}. Do not submit, retry, or ask for card details.{browser}"
        )
    return preview + "\n" + _default_payment_text(payment)


def submit_order(cart_uuid: str, tip_cents: int, confirmed: bool, intent: str,
                 include_checkout_url: bool = True) -> str:
    if not confirmed:
        return (
            "Purchase blocked. State the final total, Dasher tip, and named payment method, then "
            "ask only whether the delivery address is correct and wait for a clear yes."
        )
    tip_cents = int(tip_cents)
    if tip_cents < 0:
        return "DoorDash tip_cents cannot be negative."

    payment = _default_payment_details(intent)
    if payment and payment["expired"]:
        expiration = f"{int(payment['exp_month']):02d}/{payment['exp_year']}"
        result = {
            "success": False,
            "order_successful": False,
            "status": "payment_method_update_required",
            "action_required": True,
            "error_message": (
                f"The default {payment['brand']} ending {payment['last4']} expired {expiration}."
            ),
            "next_action": (
                "Tell the user to open DoorDash and update the payment method. Never ask for card "
                "number, expiration, security code, or billing details. Do not retry this cart."
            ),
        }
        if include_checkout_url:
            checkout, checkout_error = _checkout_url_value(cart_uuid, intent)
            result["checkout_url"] = checkout
            result["checkout_error"] = checkout_error
        return _result(result)

    if not _claim_cart_submit(cart_uuid):
        result = {
            "success": False,
            "order_successful": False,
            "status": "duplicate_submit_blocked",
            "action_required": True,
            "error_message": "This cart was already submitted once and must not be retried.",
            "next_action": "Check the order in DoorDash; do not submit this cart again.",
        }
        if include_checkout_url:
            checkout, checkout_error = _checkout_url_value(cart_uuid, intent)
            result["checkout_url"] = checkout
            result["checkout_error"] = checkout_error
        return _result(result)

    data, error = _invoke([
        "order", "submit",
        "--cart-uuid", str(cart_uuid),
        "--tip-cents", str(tip_cents),
        "--yes",
    ], intent)
    if error:
        return error
    failure = _failure(data, None, "submit that order")
    if failure:
        return _result({
            "success": False,
            "order_successful": False,
            "status": "submit_failed",
            "error_message": failure,
            "next_action": (
                "Briefly say the order was not placed and why. If the merchant is unavailable, "
                "offer to find the same items at another restaurant. Do not give a URL and do not "
                "retry this cart."
            ),
        })

    order_uuid = data.get("order_uuid") if isinstance(data, dict) else None
    if not order_uuid:
        return _result({
            "success": False,
            "order_successful": False,
            "status": "unknown",
            "error_message": "DoorDash accepted the submit call but returned no order ID.",
        })

    last_status = None
    last_error = None
    for attempt in range(_ORDER_STATUS_POLL_ATTEMPTS):
        status_data, status_error = _invoke([
            "order", "status", "--order-uuid", str(order_uuid),
        ], intent)
        if status_error:
            last_error = status_error
        elif isinstance(status_data, dict):
            last_status = _normalized_order_status(status_data, str(order_uuid))
            if last_status["status"] != "pending":
                if not last_status["order_successful"]:
                    last_status["next_action"] = (
                        "Briefly say the order was not placed and why. Offer a useful recovery "
                        "choice; do not give a URL and do not retry this cart."
                    )
                return _result(last_status)
        if attempt + 1 < _ORDER_STATUS_POLL_ATTEMPTS:
            time.sleep(_ORDER_STATUS_POLL_SECONDS)

    if last_status:
        last_status["message"] = (
            "DoorDash still reports this order as pending. Do not say it was placed successfully."
        )
        return _result(last_status)
    return _result({
        "success": False,
        "order_successful": False,
        "status": "status_unavailable",
        "order_uuid": str(order_uuid),
        "error_message": last_error or "DoorDash order status could not be verified.",
    })


def _normalized_order_status(data: dict, order_uuid: str = "") -> dict:
    """Make top-level success describe the order, not merely the status lookup request."""
    status = str(data.get("status") or "unknown").strip().casefold()
    order_successful = status == "successful"
    result = {
        "success": order_successful,
        "order_successful": order_successful,
        "status": status,
        "action_required": bool(data.get("action_required")),
    }
    if order_uuid:
        result["order_uuid"] = order_uuid
    if data.get("error_message"):
        result["error_message"] = data["error_message"]
    if data.get("message"):
        result["message"] = data["message"]
    if not order_successful and status == "failed" and not result.get("error_message"):
        result["error_message"] = "DoorDash reports that the order failed."
    return result


def order_status(order_uuid: str, intent: str) -> str:
    data, error = _invoke([
        "order", "status", "--order-uuid", str(order_uuid),
    ], intent)
    if error:
        return error
    failure = _failure(data, None, "check that order")
    if failure:
        return _result({
            "success": False,
            "order_successful": False,
            "status": "status_unavailable",
            "order_uuid": str(order_uuid),
            "error_message": failure,
        })
    return _result(_normalized_order_status(data, str(order_uuid)))


def _checkout_url_value(cart_uuid: str, intent: str):
    data, error = _invoke([
        "order", "checkout-url", "--cart-uuid", str(cart_uuid),
    ], intent)
    failure = _failure(data, error, "create a browser checkout link")
    if failure:
        return None, failure
    return data.get("checkout_url"), None


def checkout_url(cart_uuid: str, intent: str) -> str:
    checkout, error = _checkout_url_value(cart_uuid, intent)
    if error:
        return error
    return _result({"success": True, "checkout_url": checkout})


def roulette_prepare(intent: str, max_total_cents: int = 0) -> str:
    """Prepare a random recent favorite and return one compliant confirmation prompt."""
    global _roulette_pending
    max_total_cents = int(max_total_cents or os.environ.get("DOORDASH_ROULETTE_MAX_CENTS", "8000"))
    if max_total_cents < 1000 or max_total_cents > 20_000:
        return "DoorDash roulette's all-in limit must be between $10 and $200."

    with _roulette_lock:
        old_plan = _roulette_pending
        _roulette_pending = None
    if old_plan:
        _cleanup_cart(old_plan.get("cart_uuid"), intent)

    history, error = _invoke(["order", "history", "--max", "20", "--days", "90"], intent)
    failure = _failure(history, error, "read recent orders for roulette")
    if failure:
        return failure
    candidates = [
        order for order in (history.get("orders") or [])
        if order.get("is_reorderable") is True
        and order.get("order_target") == "ORDER_TARGET_RESTAURANT"
        and order.get("fulfillment_type") == "FULFILLMENT_TYPE_DX_DELIVERY"
        and order.get("items")
    ][:12]
    random.SystemRandom().shuffle(candidates)
    if not candidates:
        return "DoorDash roulette needs at least one recent reorderable restaurant delivery."

    payment = _default_payment_details(intent)
    if not payment:
        return "DoorDash roulette could not identify the default card; use browser checkout instead."

    for order in candidates:
        existing, existing_error = _invoke(
            ["cart", "list", "--store-id", str(order.get("store_id"))], intent)
        if existing_error or (existing.get("carts") or []):
            continue

        reordered, reorder_error = _invoke(
            ["order", "reorder", "--order-uuid", str(order.get("order_uuid"))], intent)
        if reorder_error or not reordered.get("success"):
            continue
        cart_uuid = reordered.get("cart_uuid")
        if not cart_uuid:
            continue

        preview, preview_error = _preview_data(cart_uuid, intent)
        if preview_error or not preview.get("success"):
            _cleanup_cart(cart_uuid, intent)
            continue
        quote = preview.get("quote") or {}
        if (quote.get("store_order_cart") or {}).get("fulfillment_type") != "DELIVERY":
            _cleanup_cart(cart_uuid, intent)
            continue
        total = (quote.get("net_total_before_tip") or {}).get("unit_amount")
        tip_cents = _suggested_tip_cents(quote)
        address = (quote.get("delivery_address") or {}).get("printable_address")
        try:
            total_cents = int(total)
        except (TypeError, ValueError):
            _cleanup_cart(cart_uuid, intent)
            continue
        if tip_cents is None or not address or total_cents + tip_cents > max_total_cents:
            _cleanup_cart(cart_uuid, intent)
            continue

        plan = {
            "cart_uuid": cart_uuid,
            "tip_cents": tip_cents,
            "total_cents": total_cents + tip_cents,
            "address": address,
            "payment": payment,
            "created_at": time.monotonic(),
        }
        with _roulette_lock:
            _roulette_pending = plan
        return (
            "Surprise ready. Do not reveal the food. Say this is "
            f"${plan['total_cents'] / 100:.2f} including a ${tip_cents / 100:.2f} Dasher tip, "
            f"charged to {payment['brand']} ending {payment['last4']}. Then ask the only question: "
            f"Is {address} the right delivery address? Wait for a clear yes before calling "
            "doordash_roulette_submit."
        )

    return (
        f"DoorDash roulette couldn't prepare a recent favorite under ${max_total_cents / 100:.2f} "
        "without touching an existing cart. No order was placed."
    )


def roulette_submit(confirmed: bool, intent: str, include_checkout_url: bool = True) -> str:
    """Submit the pending roulette order once, after its single confirmation."""
    global _roulette_pending
    if not confirmed:
        return "Roulette purchase blocked until the user clearly confirms the prepared surprise."
    with _roulette_lock:
        plan = _roulette_pending
        _roulette_pending = None
    if not plan:
        return "No current DoorDash roulette order is waiting for confirmation. Prepare a new one."
    if time.monotonic() - plan["created_at"] > _ROULETTE_TTL_SECONDS:
        _cleanup_cart(plan.get("cart_uuid"), intent)
        return "That DoorDash roulette choice expired, so I cleared its cart. Prepare a new one."
    # Remove the pending plan before submit because DoorDash submit is not idempotent.
    return submit_order(
        plan["cart_uuid"], plan["tip_cents"], True, intent, include_checkout_url)


def roulette_cancel(intent: str) -> str:
    """Cancel the pending roulette cart without charging anything."""
    global _roulette_pending
    with _roulette_lock:
        plan = _roulette_pending
        _roulette_pending = None
    if not plan:
        return "No DoorDash roulette order is waiting."
    _cleanup_cart(plan.get("cart_uuid"), intent)
    return "Canceled the DoorDash roulette cart. Nothing was ordered or charged."
