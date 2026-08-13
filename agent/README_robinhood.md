# Robinhood MCP Agent

A fully working stock trading and lookup agent built on the Model Context Protocol (MCP). Integrates with the Kronik voice agent system.

## Features

- **Stock Lookup**: Real-time prices, bid/ask, market status
- **Trading**: Buy and sell stocks via market orders
- **Portfolio Management**: View holdings, cash, buying power
- **Order History**: Track recent orders and fills
- **Two Modes**: Direct API (no server) or MCP Client (connects to MCP server)

## Quick Start

### 1. Run the Setup Script

```bash
cd ~/voice-agent/agent
chmod +x setup_robinhood.sh
./setup_robinhood.sh
```

### 2. Configure Credentials

Edit `.env.robinhood` with your Robinhood credentials:

```bash
# Edit this file:
nano .env.robinhood

# Set your actual credentials:
ROBINHOOD_USERNAME="your_robinhood_username"
ROBINHOOD_PASSWORD="your_robinhood_password"
```

### 3. Test It

```bash
# Get a stock price
python3 robinhood_agent.py price AAPL

# View your portfolio
python3 robinhood_agent.py portfolio

# Check account info
python3 robinhood_agent.py account

# View order history
python3 robinhood_agent.py orders

# Interactive mode
python3 robinhood_agent.py --interactive
```

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `price <SYMBOL>` | Get real-time stock price | `python3 robinhood_agent.py price TSLA` |
| `buy <SYMBOL> <QTY>` | Place a market buy order | `python3 robinhood_agent.py buy AAPL 10` |
| `sell <SYMBOL> <QTY>` | Place a market sell order | `python3 robinhood_agent.py sell TSLA 5` |
| `portfolio` | View all holdings | `python3 robinhood_agent.py portfolio` |
| `account` | View cash and buying power | `python3 robinhood_agent.py account` |
| `orders` | View recent order history | `python3 robinhood_agent.py orders` |

## Interactive Mode

```bash
python3 robinhood_agent.py --interactive
```

Then type commands:
```
robinhood> price AAPL
robinhood> portfolio
robinhood> buy AAPL 10
robinhood> sell TSLA 5
robinhood> account
robinhood> orders
robinhood> help
robinhood> quit
```

## Integration with Voice Agent

To add Robinhood tools to your existing Kronik voice agent, modify `agent.py`:

### Step 1: Add Imports

Add these imports near the top of `agent.py`:

```python
from robinhood_agent import RobinhoodTools
```

### Step 2: Configure on Startup

Add this to your `setup()` function or `entry()` function:

```python
# Load Robinhood credentials from .env.robinhood
import os
from dotenv import load_dotenv
load_dotenv(".env.robinhood")

# Configure Robinhood tools (called once when agent starts)
RobinhoodTools.configure(
    mode="direct",  # or "mcp" if using an MCP server
    username=os.environ.get("ROBINHOOD_USERNAME"),
    password=os.environ.get("ROBINHOOD_PASSWORD"),
)
```

### Step 3: Add Trading Tools to Assistant Class

Add these methods to your `Assistant` class in `agent.py`:

```python
@function_tool
async def get_stock_price(self, ctx: RunContext, symbol: str) -> str:
    """Look up the current price and market data for a stock symbol."""
    return await RobinhoodTools.get_stock_price(symbol)

@function_tool
async def buy_stock(self, ctx: RunContext, symbol: str, quantity: int) -> str:
    """Place a market buy order for a stock.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL')
        quantity: Number of shares to buy
    """
    return await RobinhoodTools.buy_stock(symbol, quantity)

@function_tool
async def sell_stock(self, ctx: RunContext, symbol: str, quantity: int) -> str:
    """Place a market sell order for a stock.

    Args:
        symbol: Stock ticker symbol (e.g. 'TSLA')
        quantity: Number of shares to sell
    """
    return await RobinhoodTools.sell_stock(symbol, quantity)

@function_tool
async def get_portfolio(self, ctx: RunContext) -> str:
    """View your current portfolio holdings and cash balance."""
    return await RobinhoodTools.get_portfolio()

@function_tool
async def get_account(self, ctx: RunContext) -> str:
    """View your account cash, buying power, and portfolio value."""
    return await RobinhoodTools.get_account()

@function_tool
async def get_order_history(self, ctx: RunContext, limit: int = 10) -> str:
    """View your recent order history."""
    return await RobinhoodTools.get_order_history(limit)
```

### Step 4: Update Persona

Add to your KRONIK_PERSONA string:

```
TOOLS: You can also trade stocks and look up prices. Use get_stock_price 
to check prices, buy_stock to buy shares, sell_stock to sell shares, 
get_portfolio to view holdings, and get_account to check your balance.
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 Voice Agent                      │
│  (Kronik / LiveKit / OpenAI LLM)                │
├─────────────────────────────────────────────────┤
│  RobinhoodTools (static methods)                │
│  ├── get_stock_price()                         │
│  ├── buy_stock()                               │
│  ├── sell_stock()                              │
│  ├── get_portfolio()                           │
│  ├── get_account()                             │
│  └── get_order_history()                       │
├─────────────────────────────────────────────────┤
│  RobinhoodDirectAPI (HTTP calls)                │
│  └── api.robinhood.com                         │
└─────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `robinhood_mcp.py` | MCP client + direct API client + CLI |
| `robinhood_agent.py` | Agent wrapper with trading tools |
| `setup_robinhood.sh` | Setup script (creates venv, installs deps) |
| `.env.robinhood` | Your Robinhood credentials (gitignored) |

## Security Notes

- Credentials are stored in `.env.robinhood` — never commit this file
- The direct API mode uses your Robinhood username/password
- For production, consider using the official Robinhood SDK or OAuth
- Market orders execute at the current market price (may differ from quoted price)

## Troubleshooting

**"Login failed" error:**
- Check that `.env.robinhood` has correct username/password
- Verify your Robinhood account is active

**"Module not found" error:**
- Run `./setup_robinhood.sh` to create the virtual environment

**"Market is closed" message:**
- The API returns market status. Trades can only execute during market hours (9:30 AM - 4:00 PM ET, weekdays).

**MCP mode not working:**
- Ensure your MCP server is running: `python3 robinhood_mcp.py --server --port 8081`
- Set `ROBINHOOD_MCP_SERVER_URL` to your server address
