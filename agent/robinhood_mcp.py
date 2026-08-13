"""
Robinhood MCP Client — connects to a Robinhood MCP server and exposes
stock-lookup and trading tools via the MCP (Model Context Protocol).

Usage as a library:
    from robinhood_mcp import RobinhoodMCPClient
    client = RobinhoodMCPClient(server_url="http://localhost:8081")
    await client.connect()
    prices = await client.get_quote("AAPL")
    await client.buy_stock("AAPL", 10)
    await client.sell_stock("TSLA", 5)
    await client.get_portfolio()
    await client.disconnect()

Usage as a CLI MCP server:
    python robinhood_mcp.py --server --port 8081

Usage as a CLI tool (single command):
    python robinhood_mcp.py quote AAPL
    python robinhood_mcp.py buy AAPL 10
    python robinhood_mcp.py sell TSLA 5
    python robinhood_mcp.py portfolio
"""

import asyncio
import json
import sys
import argparse
import time
import uuid
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Minimal MCP JSON-RPC client (no external dependency)
# ---------------------------------------------------------------------------

MCP_VERSION = "2024-11-05"


class MCPError(Exception):
    def __init__(self, code, message, data=None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"[{code}] {message}")


class RobinhoodMCPClient:
    """Lightweight MCP client that talks to a Robinhood MCP server over HTTP."""

    def __init__(self, server_url: str = "http://localhost:8081"):
        self.server_url = server_url.rstrip("/")
        self._session_id: Optional[str] = None
        self._initialized = False
        self._tools: list = []

    # -- low-level JSON-RPC ------------------------------------------------

    async def _request(self, method: str, params: Optional[dict] = None) -> dict:
        import http.client
        from urllib.parse import urlparse

        parsed = urlparse(self.server_url)
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params or {},
        })

        conn = http.client.HTTPConnection(
            parsed.hostname, parsed.port or 80, timeout=30
        )
        conn.request("POST", parsed.path or "/rpc", body, {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        })
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        conn.close()

        if "error" in data:
            err = data["error"]
            raise MCPError(err.get("code", -32600), err.get("message", "Unknown error"), err.get("data"))
        return data.get("result", {})

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Initialize the MCP session and discover available tools."""
        await self._request("initialize", {
            "protocolVersion": MCP_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "robinhood-mcp-agent", "version": "1.0.0"},
        })
        await self._request("notifications/initialized", {})
        self._initialized = True
        await self._discover_tools()

    async def _discover_tools(self) -> None:
        result = await self._request("tools/list", {})
        self._tools = result.get("tools", [])

    async def disconnect(self) -> None:
        self._initialized = False
        self._tools.clear()

    # -- tool helpers ------------------------------------------------------

    def _find_tool(self, name: str) -> dict:
        for t in self._tools:
            if t["name"] == name:
                return t
        raise MCPError(-32601, f"Tool '{name}' not found. Available: {[t['name'] for t in self._tools]}")

    async def _call_tool(self, name: str, **kwargs) -> dict:
        tool = self._find_tool(name)
        arguments = {}
        for param in tool.get("inputSchema", {}).get("properties", {}):
            if param in kwargs:
                arguments[param] = kwargs[param]
        result = await self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        return result

    # -- high-level trading API --------------------------------------------

    async def get_quote(self, symbol: str) -> dict:
        """Get real-time quote for a stock symbol."""
        result = await self._call_tool("get_quote", symbol=symbol.upper())
        return self._parse_result(result)

    async def get_portfolio(self) -> dict:
        """Get current portfolio holdings."""
        result = await self._call_tool("get_portfolio")
        return self._parse_result(result)

    async def get_positions(self) -> dict:
        """Get current open positions."""
        result = await self._call_tool("get_positions")
        return self._parse_result(result)

    async def buy_stock(self, symbol: str, quantity: int) -> dict:
        """Place a market buy order for a stock.

        Args:
            symbol: Stock ticker symbol (e.g. 'AAPL')
            quantity: Number of shares to buy
        """
        result = await self._call_tool("place_order", symbol=symbol.upper(), quantity=quantity, side="buy")
        return self._parse_result(result)

    async def sell_stock(self, symbol: str, quantity: int) -> dict:
        """Place a market sell order for a stock.

        Args:
            symbol: Stock ticker symbol (e.g. 'TSLA')
            quantity: Number of shares to sell
        """
        result = await self._call_tool("place_order", symbol=symbol.upper(), quantity=quantity, side="sell")
        return self._parse_result(result)

    async def get_order_history(self, limit: int = 10) -> dict:
        """Get recent order history."""
        result = await self._call_tool("get_order_history", limit=limit)
        return self._parse_result(result)

    async def get_account(self) -> dict:
        """Get account information (cash, buying power, etc.)."""
        result = await self._call_tool("get_account")
        return self._parse_result(result)

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _parse_result(result: dict) -> dict:
        """Extract content from MCP tool result (handles text + blob formats)."""
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    texts.append(item.get("text", str(item)))
                else:
                    texts.append(str(item))
            text = "\n".join(texts)
        elif isinstance(content, dict):
            text = content.get("text", str(content))
        else:
            text = str(content)

        # Try to parse as JSON for structured data
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"raw": text}


# ---------------------------------------------------------------------------
# CLI: MCP server mode (runs as an MCP server exposing Robinhood tools)
# ---------------------------------------------------------------------------

def _create_mcp_server_tools():
    """
    Create a set of MCP-compatible tools that wrap the Robinhood API.

    This function is used when running as an MCP server. It returns a list
    of tool definitions compatible with the MCP specification.
    """
    return [
        {
            "name": "get_quote",
            "description": "Get the current price and market data for a stock symbol. Returns bid, ask, last price, change, and market status.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL', 'TSLA', 'MSFT')"
                    }
                },
                "required": ["symbol"]
            }
        },
        {
            "name": "get_portfolio",
            "description": "Get the current portfolio with all holdings, quantities, and market values.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "get_positions",
            "description": "Get current open positions with unrealized P&L.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "place_order",
            "description": "Place a market order to buy or sell a stock. For buying, you need sufficient buying power. For selling, you need sufficient shares in your portfolio.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of shares"
                    },
                    "side": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Whether to buy or sell"
                    }
                },
                "required": ["symbol", "quantity", "side"]
            }
        },
        {
            "name": "get_order_history",
            "description": "Get recent order history with status and fill information.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent orders to return (default 10)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "get_account",
            "description": "Get account details including cash balance, buying power, and portfolio value.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
    ]


# ---------------------------------------------------------------------------
# CLI: Direct tool execution (no server needed — calls Robinhood API directly)
# ---------------------------------------------------------------------------

class RobinhoodDirectAPI:
    """
    Direct Robinhood API client (bypasses MCP server).
    Uses the public Robinhood REST API endpoints.

    NOTE: This requires a Robinhood username/password. For production use,
    always go through the official Robinhood SDK or MCP server.
    """

    BASE_URL = "https://api.robinhood.com"

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.session = None
        self.token = None
        self.mfa_token = None

    def login(self) -> bool:
        import http.client

        if self.session:
            return True

        self.session = http.client.HTTPSConnection("api.robinhood.com", timeout=15)

        body = json.dumps({
            "username": self.username,
            "password": self.password,
        })
        self.session.request("POST", "/oauth2/token/", body, {
            "Content-Type": "application/json",
            "User-Agent": "robinhood-mcp-agent/1.0",
        })
        resp = self.session.getresponse()
        data = json.loads(resp.read().decode())
        self.session.close()

        if "access_token" in data:
            self.token = data["access_token"]
            self.mfa_token = data.get("mfa_token")
            return True
        return False

    def _authenticated_request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        if not self.token:
            if not self.login():
                raise MCPError(-32000, "Login failed. Check credentials.")

        self.session = http.client.HTTPSConnection("api.robinhood.com", timeout=15)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "User-Agent": "robinhood-mcp-agent/1.0",
        }
        if self.mfa_token:
            headers["X-Robinhood-MFA"] = self.mfa_token

        request_body = json.dumps(body) if body else None
        self.session.request(method, path, request_body, headers)
        resp = self.session.getresponse()
        raw = resp.read().decode()
        self.session.close()

        if resp.status >= 400:
            raise MCPError(resp.status, f"API error {resp.status}: {raw}")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def get_quote(self, symbol: str) -> dict:
        data = self._authenticated_request("GET", f"/quotes/{symbol.upper()}/")
        return {
            "symbol": data.get("symbol", symbol.upper()),
            "ask_price": data.get("ask_price"),
            "bid_price": data.get("bid_price"),
            "last_price": data.get("last_price"),
            "previous_close": data.get("previous_close"),
            "change": data.get("last_trade_price", 0) - (data.get("previous_close", 0)),
            "percent_change": data.get("percent_change"),
            "market_status": data.get("market_status"),
            "high": data.get("high"),
            "low": data.get("low"),
            "volume": data.get("volume"),
        }

    def get_portfolio(self) -> dict:
        accounts = self._authenticated_request("GET", "/accounts/")
        account_id = accounts.get("results", [{}])[0].get("account_number")

        portfolio = self._authenticated_request("GET", f"/margin/portfolios/{account_id}/")
        positions = portfolio.get("equity_details", [])

        holdings = []
        total_value = 0.0
        for pos in positions:
            symbol = pos.get("instrument", {}).get("symbol", "?")
            market_value = pos.get("market_value", 0)
            total_value += market_value
            holdings.append({
                "symbol": symbol,
                "quantity": pos.get("quantity"),
                "market_value": market_value,
                "percentage": pos.get("percentage"),
            })

        return {
            "total_value": portfolio.get("equity"),
            "cash": portfolio.get("cash"),
            "withdrawable_cash": portfolio.get("withdrawable_cash"),
            "extended_hours_equity": portfolio.get("extended_hours_equity"),
            "holdings": holdings,
        }

    def place_order(self, symbol: str, quantity: int, side: str) -> dict:
        # Get the instrument ID
        quote_data = self._authenticated_request("GET", f"/quotes/{symbol.upper()}/")
        instrument_url = quote_data.get("instrument")
        instrument = self._authenticated_request("GET", instrument_url)

        # Get account
        accounts = self._authenticated_request("GET", "/accounts/")
        account_id = accounts.get("results", [{}])[0].get("url")

        order_data = {
            "account": account_id,
            "symbol": symbol.upper(),
            "quantity": quantity,
            "side": side,
            "type": "market",
            "time_in_force": "gfd",
        }

        result = self._authenticated_request("POST", "/orders/", order_data)
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Robinhood MCP Agent — trade stocks and look up prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run as an MCP server (for other agents to connect to):
  python robinhood_mcp.py --server --port 8081

  # Direct API mode (no server needed):
  python robinhood_mcp.py quote AAPL
  python robinhood_mcp.py buy AAPL 10
  python robinhood_mcp.py sell TSLA 5
  python robinhood_mcp.py portfolio
  python robinhood_mcp.py account
  python robinhood_mcp.py orders

  # As an MCP client connecting to a server:
  python robinhood_mcp.py --client --server-url http://localhost:8081 quote AAPL
        """,
    )

    parser.add_argument("--server", action="store_true", help="Run as an MCP server")
    parser.add_argument("--client", action="store_true", help="Run as an MCP client")
    parser.add_argument("--port", type=int, default=8081, help="Port for MCP server (default: 8081)")
    parser.add_argument("--server-url", type=str, default="http://localhost:8081", help="MCP server URL (client mode)")
    parser.add_argument("--username", type=str, help="Robinhood username (direct API mode)")
    parser.add_argument("--password", type=str, help="Robinhood password (direct API mode)")

    # Subcommands for direct API mode
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.add_parser("quote", help="Get stock quote")
    subparsers.add_parser("portfolio", help="Get portfolio")
    subparsers.add_parser("account", help="Get account info")
    subparsers.add_parser("orders", help="Get order history")

    buy_parser = subparsers.add_parser("buy", help="Buy stock")
    buy_parser.add_argument("symbol", type=str)
    buy_parser.add_argument("quantity", type=int)

    sell_parser = subparsers.add_parser("sell", help="Sell stock")
    sell_parser.add_argument("symbol", type=str)
    sell_parser.add_argument("quantity", type=int)

    args = parser.parse_args()

    # --- MCP Server mode ---
    if args.server:
        print(f"Starting Robinhood MCP server on port {args.port}...")
        print("Tools available: get_quote, get_portfolio, get_positions, place_order, get_order_history, get_account")
        print("Connect with an MCP client. This server exposes Robinhood trading tools.")
        # In production, this would start an actual MCP server (e.g., using FastMCP)
        print("NOTE: Full MCP server implementation requires a framework like FastMCP.")
        print("For now, use direct API mode or connect a client to a Robinhood MCP server.")
        return

    # --- MCP Client mode ---
    if args.client:
        async def run_client():
            client = RobinhoodMCPClient(server_url=args.server_url)
            try:
                await client.connect()
                if args.command == "quote" and hasattr(args, "symbol"):
                    result = await client.get_quote(args.symbol)
                    print(json.dumps(result, indent=2))
                elif args.command == "portfolio":
                    result = await client.get_portfolio()
                    print(json.dumps(result, indent=2))
                elif args.command == "buy" and hasattr(args, "symbol"):
                    result = await client.buy_stock(args.symbol, args.quantity)
                    print(json.dumps(result, indent=2))
                elif args.command == "sell" and hasattr(args, "symbol"):
                    result = await client.sell_stock(args.symbol, args.quantity)
                    print(json.dumps(result, indent=2))
                elif args.command == "account":
                    result = await client.get_account()
                    print(json.dumps(result, indent=2))
                elif args.command == "orders":
                    result = await client.get_order_history()
                    print(json.dumps(result, indent=2))
                else:
                    print("Available commands: quote, portfolio, buy, sell, account, orders")
            finally:
                await client.disconnect()

        asyncio.run(run_client())
        return

    # --- Direct API mode (no server) ---
    if args.command:
        if not args.username or not args.password:
            print("Error: Direct API mode requires --username and --password")
            print("Set these in ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD environment variables, or pass --username and --password")
            sys.exit(1)

        api = RobinhoodDirectAPI(args.username, args.password)

        try:
            if args.command == "quote" and hasattr(args, "symbol"):
                result = api.get_quote(args.symbol)
                print(json.dumps(result, indent=2))
            elif args.command == "portfolio":
                result = api.get_portfolio()
                print(json.dumps(result, indent=2))
            elif args.command == "buy" and hasattr(args, "symbol"):
                result = api.place_order(args.symbol, args.quantity, "buy")
                print(json.dumps(result, indent=2))
            elif args.command == "sell" and hasattr(args, "symbol"):
                result = api.place_order(args.symbol, args.quantity, "sell")
                print(json.dumps(result, indent=2))
            elif args.command == "account":
                accounts = api._authenticated_request("GET", "/accounts/")
                print(json.dumps(accounts, indent=2))
            elif args.command == "orders":
                orders = api._authenticated_request("GET", "/orders/?non_zero=true&limit=10")
                print(json.dumps(orders, indent=2))
            else:
                parser.print_help()
        except MCPError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
