"""
Robinhood Trading Agent — integrates stock lookup and trading into the Kronik voice agent.

This module adds Robinhood trading tools to the existing Agent class.
It can be used:
  1. Standalone: python robinhood_agent.py
  2. Integrated: import from agent.py (see integration example below)

Integration with existing agent.py:
    # In agent.py, add these imports:
    from robinhood_agent import RobinhoodTools

    # In the Assistant class, add these methods:
    @function_tool
    async def get_stock_price(self, ctx: RunContext, symbol: str) -> str:
        return await RobinhoodTools.get_stock_price(symbol)

    @function_tool
    async def buy_stock(self, ctx: RunContext, symbol: str, quantity: int) -> str:
        return await RobinhoodTools.buy_stock(symbol, quantity)

    @function_tool
    async def sell_stock(self, ctx: RunContext, symbol: str, quantity: int) -> str:
        return await RobinhoodTools.sell_stock(symbol, quantity)

    @function_tool
    async def get_portfolio(self, ctx: RunContext) -> str:
        return await RobinhoodTools.get_portfolio()

    @function_tool
    async def get_account(self, ctx: RunContext) -> str:
        return await RobinhoodTools.get_account()
"""

import os
import json
import asyncio
from typing import Optional
from robinhood_mcp import RobinhoodMCPClient, RobinhoodDirectAPI, MCPError


class RobinhoodTools:
    """
    Static utility class providing Robinhood trading tools.
    Can be called from any agent (voice or CLI).
    """

    _client: Optional[RobinhoodMCPClient] = None
    _direct_api: Optional[RobinhoodDirectAPI] = None
    _mode: str = "direct"  # "direct" or "mcp"

    @classmethod
    def configure(cls, mode: str = "direct", **kwargs) -> None:
        """
        Configure the Robinhood tools.

        Args:
            mode: "direct" to call Robinhood API directly,
                  "mcp" to connect to an MCP server.
            **kwargs:
                - username: Robinhood username (direct mode)
                - password: Robinhood password (direct mode)
                - server_url: MCP server URL (MCP mode, default: http://localhost:8081)
        """
        cls._mode = mode

        if mode == "direct":
            username = kwargs.get("username") or os.environ.get("ROBINHOOD_USERNAME")
            password = kwargs.get("password") or os.environ.get("ROBINHOOD_PASSWORD")
            if not username or not password:
                raise ValueError(
                    "Direct mode requires ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD "
                    "environment variables or --username/--password arguments."
                )
            cls._direct_api = RobinhoodDirectAPI(username, password)

        elif mode == "mcp":
            server_url = kwargs.get("server_url") or os.environ.get(
                "ROBINHOOD_MCP_SERVER_URL", "http://localhost:8081"
            )
            cls._client = RobinhoodMCPClient(server_url=server_url)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'direct' or 'mcp'.")

    @classmethod
    async def _ensure_connected(cls):
        """Ensure the appropriate client is connected."""
        if cls._mode == "direct" and not cls._direct_api.login():
            raise MCPError(-32000, "Robinhood login failed. Check credentials.")
        if cls._mode == "mcp" and not cls._client._initialized:
            await cls._client.connect()

    # -- Stock Lookup --------------------------------------------------------

    @classmethod
    async def get_stock_price(cls, symbol: str) -> str:
        """Get real-time stock price and market data."""
        await cls._ensure_connected()

        if cls._mode == "direct":
            result = cls._direct_api.get_quote(symbol)
        else:
            result = await cls._client.get_quote(symbol)

        # Format a human-readable response
        symbol = result.get("symbol", symbol.upper())
        last_price = result.get("last_price", "N/A")
        change = result.get("change", 0)
        percent_change = result.get("percent_change", 0)
        market_status = result.get("market_status", "unknown")

        change_str = f"+{change:.2f}" if change >= 0 else f"{change:.2f}"
        percent_str = f"+{percent_change:.2f}%" if percent_change >= 0 else f"{percent_change:.2f}%"

        if market_status == "open":
            status_text = "currently trading"
        elif market_status == "pre":
            status_text = "in pre-market"
        elif market_status == "after":
            status_text = "in after-hours"
        else:
            status_text = "market is closed"

        return (
            f"{symbol} is {status_text} at ${last_price}. "
            f"Changed {change_str} ({percent_str}). "
            f"Bid: ${result.get('bid_price', 'N/A')}, "
            f"Ask: ${result.get('ask_price', 'N/A')}."
        )

    @classmethod
    async def search_stock(cls, query: str) -> str:
        """Search for a stock by name or partial symbol."""
        await cls._ensure_connected()

        if cls._mode == "direct":
            # Robinhood doesn't have a search endpoint in the public API,
            # so we try common symbols
            query = query.upper()
            result = cls._direct_api.get_quote(query)
            return json.dumps(result, indent=2)
        else:
            result = await cls._client.get_quote(query)
            return json.dumps(result, indent=2)

    # -- Trading -------------------------------------------------------------

    @classmethod
    async def buy_stock(cls, symbol: str, quantity: int) -> str:
        """Place a market buy order."""
        await cls._ensure_connected()

        if cls._mode == "direct":
            result = cls._direct_api.place_order(symbol, quantity, "buy")
        else:
            result = await cls._client.buy_stock(symbol, quantity)

        status = result.get('status', 'unknown')
        symbol = result.get('symbol', symbol.upper())

        if status == "accepted":
            return (
                f"Buy order for {quantity} shares of {symbol} has been accepted. "
                f"Order ID: {result.get('reference_id', 'N/A')}. "
                f"Will execute at market price."
            )
        elif status == "filled":
            fill_price = result.get('average_price', 'N/A')
            return (
                f"Buy order for {quantity} shares of {symbol} filled at ${fill_price}. "
                f"Total: ${float(result.get('total', 0)):.2f}."
            )
        else:
            return f"Buy order for {quantity} shares of {symbol} has status: {status}."

    @classmethod
    async def sell_stock(cls, symbol: str, quantity: int) -> str:
        """Place a market sell order."""
        await cls._ensure_connected()

        if cls._mode == "direct":
            result = cls._direct_api.place_order(symbol, quantity, "sell")
        else:
            result = await cls._client.sell_stock(symbol, quantity)

        status = result.get('status', 'unknown')
        symbol = result.get('symbol', symbol.upper())

        if status == "accepted":
            return (
                f"Sell order for {quantity} shares of {symbol} has been accepted. "
                f"Order ID: {result.get('reference_id', 'N/A')}. "
                f"Will execute at market price."
            )
        elif status == "filled":
            fill_price = result.get('average_price', 'N/A')
            return (
                f"Sell order for {quantity} shares of {symbol} filled at ${fill_price}. "
                f"Total: ${float(result.get('total', 0)):.2f}."
            )
        else:
            return f"Sell order for {quantity} shares of {symbol} has status: {status}."

    # -- Portfolio & Account -------------------------------------------------

    @classmethod
    async def get_portfolio(cls) -> str:
        """Get current portfolio holdings."""
        await cls._ensure_connected()

        if cls._mode == "direct":
            result = cls._direct_api.get_portfolio()
        else:
            result = await cls._client.get_portfolio()

        total_value = result.get('total_value', 0)
        cash = result.get('cash', 0)
        holdings = result.get('holdings', [])

        lines = [f"Portfolio value: ${total_value:.2f}"]
        lines.append(f"Cash: ${cash:.2f}")
        lines.append(f"Holdings ({len(holdings)} positions):")

        for h in holdings:
            qty = h.get('quantity', 0)
            mv = h.get('market_value', 0)
            pct = h.get('percentage', 0)
            lines.append(f"  {h['symbol']}: {qty} shares (${mv:.2f}, {pct}%)")

        return "\n".join(lines)

    @classmethod
    async def get_account(cls) -> str:
        """Get account details."""
        await cls._ensure_connected()

        if cls._mode == "direct":
            accounts = cls._direct_api._authenticated_request("GET", "/accounts/")
            result = accounts.get('results', [{}])[0]
        else:
            result = await cls._client.get_account()

        cash = result.get('cash', 0)
        buying_power = result.get('buying_power', 0)
        portfolio_value = result.get('portfolio_value', 0)

        return (
            f"Cash: ${float(cash):.2f} | "
            f"Buying power: ${float(buying_power):.2f} | "
            f"Portfolio value: ${float(portfolio_value):.2f}"
        )

    @classmethod
    async def get_order_history(cls, limit: int = 10) -> str:
        """Get recent order history."""
        await cls._ensure_connected()

        if cls._mode == "direct":
            orders = cls._direct_api._authenticated_request(
                "GET", f"/orders/?non_zero=true&limit={limit}"
            )
            order_list = orders.get('results', [])
        else:
            result = await cls._client.get_order_history(limit)
            order_list = result.get('results', [])

        if not order_list:
            return "No recent orders."

        lines = ["Recent orders:"]
        for order in order_list[:limit]:
            symbol = order.get('symbol', '?')
            side = order.get('side', '?')
            qty = order.get('quantity', '?')
            status = order.get('status', '?')
            created = order.get('created', '')

            lines.append(
                f"  {side.upper()} {qty} {symbol} — {status} "
                f"(created: {created})"
            )

        return "\n".join(lines)

    @classmethod
    async def disconnect(cls):
        """Cleanly disconnect from Robinhood."""
        if cls._client and cls._client._initialized:
            await cls._client.disconnect()
        cls._direct_api = None
        cls._client = None


# ---------------------------------------------------------------------------
# Standalone CLI agent
# ---------------------------------------------------------------------------

async def cli_main():
    """Command-line interface for the Robinhood agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Robinhood Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robinhood_agent.py price AAPL
  python robinhood_agent.py buy AAPL 10
  python robinhood_agent.py sell TSLA 5
  python robinhood_agent.py portfolio
  python robinhood_agent.py account
  python robinhood_agent.py orders
  python robinhood_agent.py interactive    # interactive mode
        """,
    )

    parser.add_argument("command", nargs="?", help="Command: price, buy, sell, portfolio, account, orders")
    parser.add_argument("symbol", nargs="?", help="Stock symbol")
    parser.add_argument("quantity", type=int, nargs="?", help="Number of shares")
    parser.add_argument("--username", help="Robinhood username")
    parser.add_argument("--password", help="Robinhood password")
    parser.add_argument("--mode", choices=["direct", "mcp"], default="direct", help="API mode")
    parser.add_argument("--server-url", help="MCP server URL")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # Configure
    kwargs = {}
    if args.username:
        kwargs["username"] = args.username
    if args.password:
        kwargs["password"] = args.password
    if args.server_url:
        kwargs["server_url"] = args.server_url

    RobinhoodTools.configure(mode=args.mode, **kwargs)

    if args.interactive:
        print("Robinhood Trading Agent — Interactive Mode")
        print("Commands: price <SYMBOL>, buy <SYMBOL> <QTY>, sell <SYMBOL> <QTY>")
        print("          portfolio, account, orders, help, quit")
        print()
        while True:
            try:
                user_input = input("robinhood> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            parts = user_input.split()
            cmd = parts[0].lower()

            try:
                if cmd == "help":
                    print("Commands: price <SYMBOL>, buy <SYMBOL> <QTY>, sell <SYMBOL> <QTY>")
                    print("          portfolio, account, orders, help, quit")
                elif cmd == "price" and len(parts) >= 2:
                    result = await RobinhoodTools.get_stock_price(parts[1])
                    print(result)
                elif cmd == "buy" and len(parts) >= 3:
                    result = await RobinhoodTools.buy_stock(parts[1], int(parts[2]))
                    print(result)
                elif cmd == "sell" and len(parts) >= 3:
                    result = await RobinhoodTools.sell_stock(parts[1], int(parts[2]))
                    print(result)
                elif cmd == "portfolio":
                    result = await RobinhoodTools.get_portfolio()
                    print(result)
                elif cmd == "account":
                    result = await RobinhoodTools.get_account()
                    print(result)
                elif cmd == "orders":
                    result = await RobinhoodTools.get_order_history()
                    print(result)
                else:
                    print("Unknown command. Type 'help' for options.")
            except MCPError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Error: {e}")

    else:
        # Non-interactive mode
        commands = {
            "price": lambda: RobinhoodTools.get_stock_price(args.symbol),
            "buy": lambda: RobinhoodTools.buy_stock(args.symbol, args.quantity),
            "sell": lambda: RobinhoodTools.sell_stock(args.symbol, args.quantity),
            "portfolio": lambda: RobinhoodTools.get_portfolio(),
            "account": lambda: RobinhoodTools.get_account(),
            "orders": lambda: RobinhoodTools.get_order_history(),
        }

        if args.command and args.command in commands:
            result = await commands[args.command]()
            print(result)
        else:
            parser.print_help()

    await RobinhoodTools.disconnect()


if __name__ == "__main__":
    asyncio.run(cli_main())
