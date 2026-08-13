import json
import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import doordash_tools as doordash


class DoorDashLatencyTests(unittest.TestCase):
    def setUp(self):
        with doordash._cache_lock:
            doordash._address_cache.clear()
            doordash._history_cache.clear()
            doordash._menu_cache.clear()

    def test_default_address_is_cached(self):
        calls = []

        def invoke(args, intent, **kwargs):
            calls.append(args)
            return {
                "addresses": [{
                    "is_default": True,
                    "lat": 37.6,
                    "lng": -121.8,
                    "printable_address": "Saved address",
                }]
            }, None

        with patch.object(doordash, "_invoke", side_effect=invoke):
            first, first_error = doordash._default_address("find lunch")
            second, second_error = doordash._default_address("find dinner")

        self.assertIsNone(first_error)
        self.assertIsNone(second_error)
        self.assertEqual(first, second)
        self.assertEqual(calls, [["address", "list"]])

    def test_preview_reads_payment_methods_once(self):
        calls = []

        def invoke(args, intent, **kwargs):
            calls.append(args[:2])
            if args[:2] == ["order", "preview"]:
                return "Order preview", None
            if args[:2] == ["payment-method", "list"]:
                return {
                    "default_payment_method_id": "card-1",
                    "cards": [{
                        "payment_method_id": "card-1",
                        "brand": "Visa",
                        "last4": "1234",
                        "exp_month": 12,
                        "exp_year": 2099,
                    }],
                }, None
            self.fail(f"unexpected command: {args}")

        with patch.object(doordash, "_invoke", side_effect=invoke):
            result = doordash.preview_order("cart-1", "preview dinner")

        self.assertIn("Visa ending 1234", result)
        self.assertEqual(calls.count(["payment-method", "list"]), 1)

    def test_voice_preview_does_not_return_checkout_url(self):
        expired_card = {
            "brand": "Visa",
            "last4": "1234",
            "exp_month": 1,
            "exp_year": 2020,
            "expired": True,
        }
        with (
            patch.object(doordash, "_invoke", return_value=("Order preview", None)),
            patch.object(doordash, "_default_payment_details", return_value=expired_card),
            patch.object(doordash, "_checkout_url_value") as checkout,
        ):
            result = doordash.preview_order(
                "cart-1", "preview dinner", include_checkout_url=False)

        checkout.assert_not_called()
        self.assertNotIn("http", result)
        self.assertIn("Open DoorDash", result)

    def test_closed_store_does_not_offer_browser_checkout(self):
        def invoke(args, intent, **kwargs):
            if args[:2] == ["order", "submit"]:
                return {"success": False, "message": "Store is closed"}, None
            self.fail(f"unexpected command: {args}")

        with (
            patch.object(doordash, "_default_payment_details", return_value=None),
            patch.object(doordash, "_claim_cart_submit", return_value=True),
            patch.object(doordash, "_invoke", side_effect=invoke),
            patch.object(doordash, "_checkout_url_value") as checkout,
        ):
            result = json.loads(doordash.submit_order(
                "cart-1", 200, True, "order matcha", include_checkout_url=False))

        checkout.assert_not_called()
        self.assertEqual(result["status"], "submit_failed")
        self.assertIn("Store is closed", result["error_message"])
        self.assertNotIn("checkout_url", result)
        self.assertIn("another restaurant", result["next_action"])

    def test_item_discovery_fetches_menus_concurrently(self):
        activity_lock = threading.Lock()
        active = 0
        maximum_active = 0

        stores = [{"store_id": str(index), "name": f"Cafe {index}"} for index in range(5)]

        def menu(store_id, query, intent):
            nonlocal active, maximum_active
            with activity_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.04)
            with activity_lock:
                active -= 1
            return json.dumps({
                "success": True,
                "menu_id": f"menu-{store_id}",
                "match_count": 1,
                "items": [{"item_id": f"item-{store_id}", "name": "Matcha Latte"}],
            })

        with (
            patch.object(doordash, "_infer_item_search",
                         return_value=("matcha latte", ["coffee shop"], None)),
            patch.object(doordash, "search",
                         return_value=json.dumps({"success": True, "stores": stores})),
            patch.object(doordash, "menu", side_effect=menu),
        ):
            result = json.loads(doordash.find_items(
                "matcha latte", "find a matcha latte", limit=5))

        self.assertGreater(maximum_active, 1)
        self.assertEqual(result["stores_checked"], 5)
        self.assertEqual(len(result["results"]), 3)

    def test_named_merchant_uses_history_before_search(self):
        history = {
            "orders": [{"store_id": "23312003", "store_name": "Matsu Matcha"}],
        }
        menu = json.dumps({
            "success": True,
            "menu_id": "12854334",
            "match_count": 1,
            "items": [{"item_id": "item-1", "name": "Matcha Latte"}],
        })

        with (
            patch.object(doordash, "_infer_item_search",
                         return_value=("matcha latte", [], None)),
            patch.object(doordash, "_order_history", return_value=(history, None)),
            patch.object(doordash, "search",
                         side_effect=AssertionError("merchant search should be skipped")),
            patch.object(doordash, "menu", return_value=menu),
        ):
            result = json.loads(doordash.find_items(
                "iced matcha latte", "find Matsu Matcha", "Matsu Matcha", 5))

        self.assertTrue(result["history_fallback_used"])
        self.assertEqual(result["store_queries_tried"], [])
        self.assertEqual(result["results"][0]["store"]["name"], "Matsu Matcha")


if __name__ == "__main__":
    unittest.main()
