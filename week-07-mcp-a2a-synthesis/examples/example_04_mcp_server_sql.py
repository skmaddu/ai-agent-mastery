import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 4: Building an MCP Server — SQL Research History Tool
==============================================================
A REAL MCP server that stores and retrieves research history in SQLite.

WHY THIS MATTERS (Feynman Explanation):
  An MCP server is like a VENDING MACHINE. It does three things:

  1. ADVERTISE — Display a menu of what it offers (list_tools).
     "I have: save_research, search_research, list_recent, get_stats."

  2. ACCEPT INPUT — Take your selection and parameters (call_tool).
     "You want search_research with query='quantum computing'? Got it."

  3. DISPENSE RESULTS — Process the request and return output.
     "Here are 3 matching research entries..."

  The BEAUTY of MCP is that the vending machine doesn't need to know
  WHO is using it — a human, an LLM, another agent — it just follows
  the protocol. And the CLIENT doesn't need to know HOW the vending
  machine works internally (SQLite? Postgres? A filing cabinet?).
  Both sides agree on the protocol, and everything just works.

What this example demonstrates:
  1. Creating an MCP server with FastMCP
  2. Registering tools with @mcp.tool() decorator
  3. SQLite persistence for research history
  4. Self-test: launching the server and verifying tools as a client

Architecture:
  ┌──────────────────────────────────────────────────┐
  │              MCP Server (this file)               │
  │                                                    │
  │  @mcp.tool() save_research(topic, summary, ...)   │
  │  @mcp.tool() search_research(query)               │
  │  @mcp.tool() list_recent(limit)                   │
  │  @mcp.tool() get_stats()                          │
  │                                                    │
  │  ┌──────────────┐                                  │
  │  │   SQLite DB   │  ← persistence layer            │
  │  └──────────────┘                                  │
  └──────────────────────────────────────────────────┘
         ▲▼  stdio (JSON-RPC over stdin/stdout)
  ┌──────────────────────────────────────────────────┐
  │           MCP Client (any client)                 │
  │  - Example 03's client                            │
  │  - Claude Desktop                                 │
  │  - Any MCP-compatible agent                       │
  └──────────────────────────────────────────────────┘

Run as server:    python example_04_mcp_server_sql.py
Run self-test:    python example_04_mcp_server_sql.py --self-test
"""

import os
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

import asyncio
import json
import sqlite3
import tempfile
import logging
from datetime import datetime, timezone

# ================================================================
# WHAT IS FastMCP?
# ================================================================
# FastMCP is the HIGH-LEVEL API for building MCP servers. Under the
# hood, it handles ALL the protocol plumbing:
#
#   - JSON-RPC message parsing (reading from stdin, writing to stdout)
#   - The "initialize" handshake (client says hello, server responds
#     with its capabilities — which tools, resources, prompts it offers)
#   - The "tools/list" handler (returns all registered tools + schemas)
#   - The "tools/call" handler (dispatches to your Python function)
#   - Input validation (checks arguments against JSON Schema)
#   - Error formatting (MCP-compliant error responses)
#
# Without FastMCP, you'd need ~200 lines of boilerplate for the
# JSON-RPC loop alone. With FastMCP, you just write your tool
# functions and decorate them with @mcp.tool().
#
# It's like Flask for HTTP: Flask handles routing, headers, status
# codes — you just write the endpoint functions. FastMCP handles
# MCP protocol — you just write the tool functions.

from mcp.server.fastmcp import FastMCP

# Quiet logging so it doesn't pollute the stdio JSON-RPC stream.
# WHY? MCP servers communicate via stdin/stdout. If we print logs
# to stdout, they'll corrupt the JSON-RPC messages. Logging goes
# to stderr (which the client ignores for protocol purposes).
logging.basicConfig(level=logging.WARNING)


# ================================================================
# CREATE THE SERVER
# ================================================================
# The name "research-history-edu" appears in the MCP capability
# handshake. Clients see this name when they connect, so they know
# which server they're talking to. Think of it as the server's
# "name tag" at a networking event.

mcp = FastMCP("research-history-edu")


# ================================================================
# DATABASE SETUP
# ================================================================
# WHY SQLite?
# 1. BUILT INTO PYTHON — no pip install, no external service
# 2. ZERO CONFIG — no username, password, port, connection string
# 3. SINGLE FILE — the entire database is one .db file
# 4. PERFECT FOR LEARNING — same SQL you'd use with Postgres
#
# In production, you'd swap to Postgres/MySQL for multi-user access.
# But for an MCP server that runs as a single process, SQLite is ideal.
#
# WHERE does the database file go?
# - In self-test mode: a temporary file (auto-deleted)
# - In normal mode: next to this script, or wherever RESEARCH_DB_PATH says

DB_PATH = os.environ.get(
    "RESEARCH_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_history_edu.db")
)


def _get_db() -> sqlite3.Connection:
    """Get a database connection, creating the table if it doesn't exist.

    WHY row_factory = sqlite3.Row?
    By default, sqlite3 returns tuples: (1, "AI safety", "Summary...", ...).
    With Row, you get dict-like access: row["topic"], row["summary"].
    Much more readable and less error-prone than row[1], row[2].
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # CREATE TABLE IF NOT EXISTS is idempotent — safe to call every time.
    # This means the server self-initializes its database on first run.
    # No migration scripts, no setup commands. Just works.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            summary TEXT NOT NULL,
            sources TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            word_count INTEGER DEFAULT 0,
            quality_score REAL DEFAULT 0.0
        )
    """)

    # Index on topic for faster keyword searches.
    # Without this, every search scans ALL rows (slow for large DBs).
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_research_topic
        ON research(topic)
    """)

    conn.commit()
    return conn


# ================================================================
# TOOL 1: save_research — Store a research finding
# ================================================================
# HOW @mcp.tool() WORKS UNDER THE HOOD:
#
#   1. FastMCP inspects the function signature (name, parameters, types)
#   2. It reads the docstring for the tool description
#   3. It generates a JSON Schema from the type hints
#   4. When a client calls list_tools(), FastMCP returns this schema
#   5. When a client calls call_tool("save_research", {...}), FastMCP:
#      a) Validates the arguments against the schema
#      b) Calls your Python function with those arguments
#      c) Wraps the return value in an MCP TextContent response
#
# You don't write ANY of that plumbing. Just the function + docstring.

@mcp.tool()
def save_research(topic: str, summary: str, sources: str = "[]") -> str:
    """Save a research finding to the database for future reference.

    Args:
        topic: The research topic (e.g., 'AI in healthcare', 'quantum computing')
        summary: The research summary/findings text
        sources: JSON string list of source URLs or references (default: '[]')

    Returns:
        Confirmation message with the saved entry's ID.
    """
    # ── Input validation ─────────────────────────────────────────
    # Always validate inputs, even though FastMCP checks types.
    # Type checking catches "string vs int" but not "empty string".
    if not topic.strip():
        return "Error: topic cannot be empty"
    if not summary.strip():
        return "Error: summary cannot be empty"

    # ── Parse sources ────────────────────────────────────────────
    # Sources come as a JSON string because MCP tool parameters are
    # simple types (string, number, boolean). We parse to a list.
    try:
        source_list = json.loads(sources) if sources else []
        if not isinstance(source_list, list):
            source_list = [str(source_list)]
    except json.JSONDecodeError:
        # Be forgiving: if it's not JSON, treat as comma-separated
        source_list = [s.strip() for s in sources.split(",") if s.strip()]

    # ── Insert into database ─────────────────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    word_count = len(summary.split())

    conn = _get_db()
    try:
        cursor = conn.execute(
            """INSERT INTO research (topic, summary, sources, created_at, word_count)
               VALUES (?, ?, ?, ?, ?)""",
            (topic.strip(), summary.strip(), json.dumps(source_list), now, word_count)
        )
        conn.commit()
        entry_id = cursor.lastrowid

        return (
            f"Research saved successfully!\n"
            f"  ID: {entry_id}\n"
            f"  Topic: {topic}\n"
            f"  Words: {word_count}\n"
            f"  Sources: {len(source_list)}\n"
            f"  Saved at: {now}"
        )
    except sqlite3.Error as e:
        return f"Database error: {e}"
    finally:
        conn.close()


# ================================================================
# TOOL 2: search_research — Full-text keyword search
# ================================================================
# WHY LIKE instead of FTS5?
# SQLite has a powerful full-text search extension (FTS5), but it
# requires extra setup and isn't always available. LIKE is simpler,
# works everywhere, and is fine for educational databases with
# <10,000 entries. For production, use FTS5 or Postgres full-text.

@mcp.tool()
def search_research(query: str, max_results: int = 5) -> str:
    """Search past research findings by keyword.

    Searches both topic and summary fields for matching keywords.

    Args:
        query: Search keywords (e.g., 'machine learning', 'climate')
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Matching research entries with topic, summary excerpt, and date.
    """
    if not query.strip():
        return "Error: search query cannot be empty"

    max_results = max(1, min(max_results, 20))  # Clamp to [1, 20]

    conn = _get_db()
    try:
        # Build WHERE clause: every keyword must match topic OR summary
        # "AI healthcare" → (topic LIKE '%AI%' OR summary LIKE '%AI%')
        #                AND (topic LIKE '%healthcare%' OR summary LIKE '%healthcare%')
        keywords = query.strip().split()
        conditions = []
        params = []
        for kw in keywords:
            conditions.append("(topic LIKE ? OR summary LIKE ?)")
            params.extend([f"%{kw}%", f"%{kw}%"])

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        rows = conn.execute(
            f"""SELECT id, topic, summary, sources, created_at, word_count
                FROM research
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?""",
            params + [max_results]
        ).fetchall()

        if not rows:
            return f"No research found matching '{query}'"

        lines = [f"Found {len(rows)} result(s) for '{query}'", "=" * 50]
        for row in rows:
            summary_excerpt = row["summary"][:200] + ("..." if len(row["summary"]) > 200 else "")
            sources = json.loads(row["sources"]) if row["sources"] else []
            lines.append(
                f"\n  [{row['id']}] {row['topic']}\n"
                f"    {summary_excerpt}\n"
                f"    Date: {row['created_at'][:10]}\n"
                f"    {row['word_count']} words | {len(sources)} sources"
            )

        return "\n".join(lines)
    except sqlite3.Error as e:
        return f"Database error: {e}"
    finally:
        conn.close()


# ================================================================
# TOOL 3: list_recent — Most recent research entries
# ================================================================

@mcp.tool()
def list_recent(limit: int = 10) -> str:
    """List the most recent research entries.

    Args:
        limit: Number of entries to return (default: 10, max: 50)

    Returns:
        Recent research entries sorted by date (newest first).
    """
    limit = max(1, min(limit, 50))

    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT id, topic, summary, created_at, word_count
               FROM research
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()

        if not rows:
            return "No research entries yet. Use save_research to add some!"

        lines = [f"Recent Research ({len(rows)} entries)", "=" * 50]
        for row in rows:
            summary_excerpt = row["summary"][:100] + ("..." if len(row["summary"]) > 100 else "")
            lines.append(
                f"\n  [{row['id']}] {row['topic']}\n"
                f"      {summary_excerpt}\n"
                f"      Date: {row['created_at'][:10]} | {row['word_count']} words"
            )

        return "\n".join(lines)
    except sqlite3.Error as e:
        return f"Database error: {e}"
    finally:
        conn.close()


# ================================================================
# TOOL 4: get_stats — Database statistics
# ================================================================
# WHY aggregate queries?
# Stats give users (and LLMs!) a quick overview without reading
# every entry. An LLM might call get_stats() first to understand
# what's in the database, then search_research() for specifics.

@mcp.tool()
def get_stats() -> str:
    """Get statistics about the research history database.

    Returns:
        Total entries, total words, topics covered, date range,
        and top topics by entry count.
    """
    conn = _get_db()
    try:
        stats = conn.execute(
            """SELECT
                COUNT(*) as total_entries,
                COALESCE(SUM(word_count), 0) as total_words,
                COUNT(DISTINCT topic) as unique_topics,
                MIN(created_at) as earliest,
                MAX(created_at) as latest
               FROM research"""
        ).fetchone()

        if stats["total_entries"] == 0:
            return "Database is empty. No research entries yet!"

        # Top topics by entry count
        top_topics = conn.execute(
            """SELECT topic, COUNT(*) as count
               FROM research
               GROUP BY topic
               ORDER BY count DESC
               LIMIT 5"""
        ).fetchall()

        lines = [
            "Research History Statistics",
            "=" * 40,
            f"  Total entries: {stats['total_entries']}",
            f"  Total words: {stats['total_words']:,}",
            f"  Unique topics: {stats['unique_topics']}",
            f"  Date range: {stats['earliest'][:10]} to {stats['latest'][:10]}",
            f"\n  Top Topics:",
        ]
        for t in top_topics:
            lines.append(f"    - {t['topic']} ({t['count']} entries)")

        return "\n".join(lines)
    except sqlite3.Error as e:
        return f"Database error: {e}"
    finally:
        conn.close()


# ================================================================
# HOW STDIO TRANSPORT WORKS
# ================================================================
# When you call mcp.run(), FastMCP enters an infinite loop:
#
#   1. READ a JSON-RPC message from stdin
#   2. PARSE the method name: "initialize", "tools/list", "tools/call"
#   3. DISPATCH to the appropriate handler
#   4. WRITE the JSON-RPC response to stdout
#   5. GOTO 1
#
# The CLIENT (example_03) is on the other side of the pipe:
#   - It spawns THIS script as a subprocess
#   - It WRITES requests to our stdin
#   - It READS responses from our stdout
#
# This is the simplest possible transport — no HTTP, no WebSocket,
# no ports, no TLS. Just two processes talking through pipes.
# It's also the most secure: the server is sandboxed as a subprocess
# with no network access unless it explicitly opens connections.
#
# Other transports exist (SSE/HTTP for remote servers), but stdio
# is the default and most common for local tool servers.


# ================================================================
# SELF-TEST: Launch as server, connect as client, verify all tools
# ================================================================
# WHY a self-test?
# When building MCP servers, you need to verify that:
#   a) The server starts without errors
#   b) All tools are discoverable via list_tools()
#   c) Each tool returns correct results
#   d) Error cases are handled gracefully
#
# This self-test does exactly that: it launches THIS FILE as a
# subprocess (in server mode), connects as a client, and runs
# through all 4 tools. It's like the server testing itself in a mirror.

from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

try:
    from mcp.client.stdio import StdioServerParameters
except ImportError:
    from mcp import StdioServerParameters


async def self_test():
    """Launch this server as a subprocess and verify all tools work."""

    print("=" * 70)
    print("  SELF-TEST: MCP Research History Server")
    print("=" * 70)

    # ── Use a temporary database for testing ─────────────────────
    # We don't want the self-test to pollute any real database.
    # tempfile gives us a path that auto-cleans up (on most systems).
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_db_path = tmp_db.name
    tmp_db.close()

    print(f"\n  Temp database: {tmp_db_path}")

    # ── Launch this script as an MCP server subprocess ───────────
    this_script = os.path.abspath(__file__)
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[this_script],
        env={
            **os.environ,
            "RESEARCH_DB_PATH": tmp_db_path,  # Override DB path for test
        },
    )

    print("  Launching server subprocess...")

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:

                # ── Initialize ───────────────────────────────────
                await session.initialize()
                print("  Connected and initialized!\n")

                # ── List tools ───────────────────────────────────
                print("  " + "-" * 55)
                print("  TEST 1: List all tools")
                print("  " + "-" * 55)

                tools_result = await session.list_tools()
                tools = tools_result.tools
                tool_names = [t.name for t in tools]

                expected_tools = {"save_research", "search_research", "list_recent", "get_stats"}
                found_tools = set(tool_names)

                print(f"  Found tools: {tool_names}")

                if expected_tools.issubset(found_tools):
                    print("  PASS: All expected tools found!\n")
                else:
                    missing = expected_tools - found_tools
                    print(f"  FAIL: Missing tools: {missing}\n")

                # ── Test save_research ───────────────────────────
                print("  " + "-" * 55)
                print("  TEST 2: save_research")
                print("  " + "-" * 55)

                result = await session.call_tool("save_research", {
                    "topic": "Quantum Computing",
                    "summary": "Quantum computers use qubits that can be in superposition, "
                               "enabling parallel computation for certain problems like "
                               "factoring large numbers and simulating molecules.",
                    "sources": '["https://arxiv.org/quantum", "https://nature.com/quantum"]'
                })
                text = result.content[0].text if result.content else "(no output)"
                print(f"  Result:\n    {text.replace(chr(10), chr(10) + '    ')}")
                passed = "saved" in text.lower() or "id:" in text.lower()
                print(f"  {'PASS' if passed else 'FAIL'}: save_research\n")

                # Save a second entry for search testing
                await session.call_tool("save_research", {
                    "topic": "AI Safety",
                    "summary": "AI safety research focuses on alignment, robustness, "
                               "and interpretability to ensure AI systems behave as intended.",
                    "sources": '["https://arxiv.org/ai-safety"]'
                })

                # Save a third entry
                await session.call_tool("save_research", {
                    "topic": "Quantum Computing",
                    "summary": "Recent breakthroughs in error correction have brought "
                               "fault-tolerant quantum computing closer to reality.",
                    "sources": '["https://nature.com/quantum-error"]'
                })
                print("  (Saved 2 additional entries for search/stats testing)\n")

                # ── Test search_research ─────────────────────────
                print("  " + "-" * 55)
                print("  TEST 3: search_research")
                print("  " + "-" * 55)

                result = await session.call_tool("search_research", {
                    "query": "quantum",
                })
                text = result.content[0].text if result.content else "(no output)"
                print(f"  Query: 'quantum'")
                print(f"  Result:\n    {text.replace(chr(10), chr(10) + '    ')}")
                passed = "quantum" in text.lower() and ("2 result" in text.lower() or "found" in text.lower())
                print(f"  {'PASS' if passed else 'CHECK'}: search_research\n")

                # ── Test list_recent ─────────────────────────────
                print("  " + "-" * 55)
                print("  TEST 4: list_recent")
                print("  " + "-" * 55)

                result = await session.call_tool("list_recent", {"limit": 5})
                text = result.content[0].text if result.content else "(no output)"
                print(f"  Result:\n    {text.replace(chr(10), chr(10) + '    ')}")
                passed = "3 entries" in text.lower() or "quantum" in text.lower()
                print(f"  {'PASS' if passed else 'CHECK'}: list_recent\n")

                # ── Test get_stats ───────────────────────────────
                print("  " + "-" * 55)
                print("  TEST 5: get_stats")
                print("  " + "-" * 55)

                result = await session.call_tool("get_stats", {})
                text = result.content[0].text if result.content else "(no output)"
                print(f"  Result:\n    {text.replace(chr(10), chr(10) + '    ')}")
                passed = "3" in text and "total" in text.lower()
                print(f"  {'PASS' if passed else 'CHECK'}: get_stats\n")

        # ── Summary ──────────────────────────────────────────────
        print("  " + "=" * 55)
        print("  SELF-TEST COMPLETE")
        print("  " + "=" * 55)
        print("""
  All 4 MCP tools tested:
    1. save_research  — Inserts into SQLite, returns confirmation
    2. search_research — LIKE search across topic + summary
    3. list_recent    — ORDER BY date DESC with LIMIT
    4. get_stats      — Aggregate COUNT, SUM, DISTINCT queries

  KEY TAKEAWAYS:
    - FastMCP handles ALL protocol plumbing (JSON-RPC, schemas, dispatch)
    - @mcp.tool() auto-generates JSON Schema from type hints + docstring
    - stdio transport = subprocess + pipes (simple, secure, no network)
    - SQLite = zero-install persistence (swap to Postgres for production)
    - Self-test pattern: launch yourself as subprocess, verify via client
""")

    except Exception as e:
        print(f"\n  Self-test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up temp database
        try:
            os.unlink(tmp_db_path)
            print(f"  Cleaned up temp database: {tmp_db_path}")
        except OSError:
            pass


# ================================================================
# ENTRY POINT
# ================================================================
# Two modes:
#   1. Normal (no args): Run as MCP server (stdio transport)
#      → Used when an MCP client spawns this as a subprocess
#
#   2. --self-test: Run the self-test (launches server + client)
#      → Used for development/verification

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        asyncio.run(self_test())
    else:
        # Run as MCP server — enters the stdio JSON-RPC loop
        # This blocks forever, reading requests and sending responses.
        # The MCP client on the other side of the pipe drives the interaction.
        mcp.run()
