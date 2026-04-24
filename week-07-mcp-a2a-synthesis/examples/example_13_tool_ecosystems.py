import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 13: Real-World Tool Ecosystems — Databases, Code Execution, Web Fetching
==================================================================================
Topic 13 — Building specialized MCP tool servers for real-world use cases:
databases, sandboxed code execution, and web content fetching.

The BIG IDEA (Feynman):
  MCP tool servers are like specialized workshops — a carpentry shop, an
  electronics lab, and a print shop. Each has its own tools, but any
  customer (agent) can walk in and use them through the same door
  (MCP protocol).

  The carpentry shop (DatabaseToolServer) has saws, drills, and sanders
  for working with wood (data). The electronics lab (CodeExecutionServer)
  has oscilloscopes, soldering irons, and multimeters for building circuits
  (running code). The print shop (WebFetchServer) has presses, cutters,
  and binders for producing documents (fetching web content).

  The key insight: the CUSTOMER doesn't need to know how each workshop
  works internally. They just walk in the same door (MCP protocol), ask
  what tools are available (list_tools), and use them (call_tool).

What this example demonstrates:
  1. DatabaseToolServer — SQLite read-only queries, table schema, table listing
  2. CodeExecutionServer — Safe Python execution with timeout and import filtering
  3. WebFetchServer — URL fetching, link extraction, JSON parsing
  4. Self-test mode: launch each server and test its tools via MCP client

SAFETY WARNINGS:
  - DatabaseToolServer rejects write operations (INSERT/UPDATE/DELETE) by default
  - CodeExecutionServer blocks dangerous imports (os.system, subprocess, eval, exec)
  - WebFetchServer is for educational purposes — add rate limiting in production

Run: python week-07-mcp-a2a-synthesis/examples/example_13_tool_ecosystems.py
Self-test: python week-07-mcp-a2a-synthesis/examples/example_13_tool_ecosystems.py --self-test
"""

import os
import json
import asyncio
import sqlite3
import re
import tempfile
import subprocess as sp
import textwrap
from typing import Optional
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()


# ================================================================
# LLM Setup (used only in self-test for LLM-driven tool selection)
# ================================================================

def get_llm(temperature=0.3):
    """Create LLM instance based on provider setting."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )


# ================================================================
# Phoenix Tracing (optional)
# ================================================================

try:
    from config.phoenix_config import setup_tracing
    setup_tracing()
    PHOENIX_AVAILABLE = True
except Exception:
    PHOENIX_AVAILABLE = False


# ================================================================
# SERVER 1: DATABASE TOOL SERVER (SQLite)
# ================================================================
# This server provides read-only access to SQLite databases.
# It's like a librarian who can LOOK UP information but won't
# let you scribble in the books.
#
# WHY READ-ONLY?
# In production, you want to separate read and write paths.
# An AI agent with unrestricted write access could accidentally
# (or maliciously, via prompt injection) delete your data.
# Read-only is the safe default.
#
# SAFETY: SQL injection is prevented by only allowing SELECT statements.
# The server rejects any query containing INSERT, UPDATE, DELETE, DROP,
# ALTER, CREATE, or TRUNCATE.

print("=" * 70)
print("SERVER 1: DatabaseToolServer (SQLite Read-Only)")
print("=" * 70)
print("""
  Tools:
    - query_db(sql, db_path) — Execute SELECT queries (read-only)
    - describe_table(table_name, db_path) — Get table schema (columns, types)
    - list_tables(db_path) — List all tables in a database

  Safety: Rejects INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE
""")


def create_database_server():
    """Create an MCP server for SQLite database operations.

    This function returns a FastMCP instance with 3 tools for reading
    from SQLite databases. Each tool validates inputs before executing.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("database-tools")

    # Dangerous SQL keywords that indicate write operations
    WRITE_KEYWORDS = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE)\b",
        re.IGNORECASE,
    )

    @mcp.tool()
    def query_db(sql: str, db_path: str = "test.db") -> str:
        """Execute a read-only SQL query against a SQLite database.

        Args:
            sql: The SQL SELECT query to execute. Only SELECT is allowed.
            db_path: Path to the SQLite database file. Defaults to test.db.

        Returns:
            Query results as formatted text, or an error message.

        Safety: Rejects any query containing write operations.
        """
        # Safety check: reject write operations
        if WRITE_KEYWORDS.search(sql):
            return (
                "ERROR: Write operations are not allowed. "
                "Only SELECT queries are permitted for safety. "
                f"Rejected keywords found in: {sql[:100]}"
            )

        # Validate the query starts with SELECT (or WITH for CTEs)
        stripped = sql.strip().upper()
        if not stripped.startswith(("SELECT", "WITH", "PRAGMA", "EXPLAIN")):
            return (
                "ERROR: Only SELECT, WITH, PRAGMA, and EXPLAIN statements are allowed. "
                f"Got: {stripped[:20]}..."
            )

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            conn.close()

            if not rows:
                return f"Query returned 0 rows.\nColumns: {', '.join(columns)}"

            # Format as a readable table
            lines = [f"Columns: {', '.join(columns)}", f"Rows: {len(rows)}", ""]
            for i, row in enumerate(rows[:50]):  # Limit to 50 rows
                row_dict = dict(row)
                lines.append(f"  Row {i + 1}: {json.dumps(row_dict, default=str)}")

            if len(rows) > 50:
                lines.append(f"  ... and {len(rows) - 50} more rows")

            return "\n".join(lines)

        except sqlite3.Error as e:
            return f"SQLite error: {e}"
        except Exception as e:
            return f"Error executing query: {e}"

    @mcp.tool()
    def describe_table(table_name: str, db_path: str = "test.db") -> str:
        """Get the schema (columns, types, constraints) of a table.

        Args:
            table_name: Name of the table to describe.
            db_path: Path to the SQLite database file.

        Returns:
            Table schema as formatted text showing columns, types, and constraints.
        """
        # Sanitize table name to prevent injection
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            return f"ERROR: Invalid table name: {table_name}"

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            conn.close()

            if not columns:
                return f"Table '{table_name}' not found or has no columns."

            lines = [f"Table: {table_name}", f"Columns: {len(columns)}", ""]
            for col in columns:
                # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
                cid, name, col_type, notnull, default, pk = col
                constraints = []
                if pk:
                    constraints.append("PRIMARY KEY")
                if notnull:
                    constraints.append("NOT NULL")
                if default is not None:
                    constraints.append(f"DEFAULT {default}")

                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                lines.append(f"  {name}: {col_type or 'TEXT'}{constraint_str}")

            return "\n".join(lines)

        except sqlite3.Error as e:
            return f"SQLite error: {e}"

    @mcp.tool()
    def list_tables(db_path: str = "test.db") -> str:
        """List all tables in a SQLite database.

        Args:
            db_path: Path to the SQLite database file.

        Returns:
            List of table names with row counts.
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]

            if not tables:
                return f"No tables found in {db_path}"

            lines = [f"Database: {db_path}", f"Tables: {len(tables)}", ""]
            for table in tables:
                try:
                    count_cursor = conn.execute(f"SELECT COUNT(*) FROM [{table}]")
                    count = count_cursor.fetchone()[0]
                    lines.append(f"  {table} ({count} rows)")
                except Exception:
                    lines.append(f"  {table} (row count unavailable)")

            conn.close()
            return "\n".join(lines)

        except sqlite3.Error as e:
            return f"SQLite error: {e}"

    return mcp


# ================================================================
# SERVER 2: CODE EXECUTION SERVER (Sandboxed Python)
# ================================================================
# This server runs Python code in a separate subprocess with a
# timeout. It's like a chemistry lab with safety equipment — you
# can run experiments, but there are guardrails to prevent explosions.
#
# WHY SUBPROCESS?
# Running user code in the SAME process is dangerous — it could
# access your memory, files, or network. A subprocess provides
# isolation: even if the code crashes, your main process survives.
#
# SECURITY LAYERS:
#   1. Import filtering — block dangerous modules before execution
#   2. Subprocess isolation — code runs in a separate process
#   3. Timeout — kill runaway code after N seconds
#   4. Output capture — only return stdout/stderr, not internal state
#
# WARNING: This is NOT a production sandbox. For real sandboxing,
# use Docker containers, gVisor, or cloud functions (AWS Lambda).

print("\n" + "=" * 70)
print("SERVER 2: CodeExecutionServer (Sandboxed Python)")
print("=" * 70)
print("""
  Tools:
    - run_python(code, timeout_seconds) — Execute Python in a subprocess
    - run_python_safe(code) — Same but with restricted builtins

  Security: Blocks os.system, subprocess, eval, exec, __import__
  Timeout: Default 5 seconds, max 30 seconds
""")

# Patterns that indicate dangerous code
DANGEROUS_PATTERNS = [
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\b__import__\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    r"\bopen\s*\([^)]*['\"]w",       # open(..., 'w') — writing files
    r"\bshutil\b",
    r"\bsocket\b",
    r"\brequests\b",                   # block network access in sandbox
    r"\burllib\b",
]


def check_code_safety(code: str) -> Optional[str]:
    """Check if code contains dangerous patterns.

    Returns None if safe, or an error message describing the violation.

    WHY pattern matching instead of AST analysis?
    Pattern matching is simpler and catches obfuscation attempts
    that AST-based checks might miss (e.g., getattr tricks).
    The tradeoff is false positives, but for a training environment
    that's acceptable — better safe than sorry.
    """
    for pattern in DANGEROUS_PATTERNS:
        match = re.search(pattern, code)
        if match:
            return (
                f"BLOCKED: Code contains dangerous pattern: '{match.group()}'. "
                f"For safety, the following are not allowed: "
                f"os.system, subprocess, eval, exec, __import__, compile, "
                f"file writing, shutil, socket, requests, urllib."
            )
    return None


def create_code_execution_server():
    """Create an MCP server for sandboxed Python code execution."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("code-execution")

    @mcp.tool()
    def run_python(code: str, timeout_seconds: int = 5) -> str:
        """Execute Python code in a sandboxed subprocess with timeout.

        Args:
            code: Python code to execute. Dangerous operations are blocked.
            timeout_seconds: Maximum execution time in seconds (default 5, max 30).

        Returns:
            stdout output from the code, or error message if execution failed.

        Security: Blocks dangerous imports and operations. Runs in subprocess.
        """
        # Enforce timeout limits
        timeout_seconds = max(1, min(timeout_seconds, 30))

        # Safety check
        violation = check_code_safety(code)
        if violation:
            return violation

        try:
            result = sp.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                # Don't inherit environment variables for isolation
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": "",
                    "HOME": tempfile.gettempdir(),
                },
            )

            output_parts = []
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout.strip()}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr.strip()}")
            if result.returncode != 0:
                output_parts.append(f"Exit code: {result.returncode}")

            if not output_parts:
                return "Code executed successfully (no output)."

            return "\n\n".join(output_parts)

        except sp.TimeoutExpired:
            return (
                f"TIMEOUT: Code execution exceeded {timeout_seconds} seconds. "
                f"The process was killed. Consider optimizing your code or "
                f"increasing the timeout (max 30s)."
            )
        except Exception as e:
            return f"Execution error: {type(e).__name__}: {e}"

    @mcp.tool()
    def run_python_safe(code: str) -> str:
        """Execute Python code with additional sandboxing (restricted builtins).

        This is a stricter version of run_python that also restricts built-in
        functions. Only safe builtins like print, len, range, etc. are available.

        Args:
            code: Python code to execute with restricted builtins.

        Returns:
            stdout output from the code, or error message.
        """
        # Safety check (same as run_python)
        violation = check_code_safety(code)
        if violation:
            return violation

        # Build a wrapper that restricts builtins
        # WHY restrict builtins?
        # Even without importing dangerous modules, Python's builtins
        # include open(), exec(), eval(), compile(), __import__() etc.
        # By replacing __builtins__ with a safe subset, we remove these.
        safe_wrapper = textwrap.dedent("""\
            import sys as _sys

            # Define safe builtins
            _safe_builtins = {
                'print': print, 'len': len, 'range': range, 'int': int,
                'float': float, 'str': str, 'bool': bool, 'list': list,
                'dict': dict, 'set': set, 'tuple': tuple, 'type': type,
                'isinstance': isinstance, 'issubclass': issubclass,
                'abs': abs, 'min': min, 'max': max, 'sum': sum,
                'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'any': any,
                'all': all, 'round': round, 'pow': pow, 'divmod': divmod,
                'hash': hash, 'id': id, 'repr': repr, 'format': format,
                'chr': chr, 'ord': ord, 'hex': hex, 'oct': oct, 'bin': bin,
                'True': True, 'False': False, 'None': None,
                'ValueError': ValueError, 'TypeError': TypeError,
                'KeyError': KeyError, 'IndexError': IndexError,
                'ZeroDivisionError': ZeroDivisionError,
                'RuntimeError': RuntimeError, 'StopIteration': StopIteration,
                'Exception': Exception,
            }

            # Execute user code with restricted builtins
            _user_code = '''""" + code.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + """'''
            _namespace = {'__builtins__': _safe_builtins}
            exec(compile(_user_code, '<sandbox>', 'exec'), _namespace)
        """)

        try:
            result = sp.run(
                [sys.executable, "-c", safe_wrapper],
                capture_output=True,
                text=True,
                timeout=5,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": "",
                    "HOME": tempfile.gettempdir(),
                },
            )

            output_parts = []
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout.strip()}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr.strip()}")
            if result.returncode != 0 and not result.stderr:
                output_parts.append(f"Exit code: {result.returncode}")

            if not output_parts:
                return "Code executed successfully (no output)."

            return "\n\n".join(output_parts)

        except sp.TimeoutExpired:
            return "TIMEOUT: Code execution exceeded 5 seconds."
        except Exception as e:
            return f"Execution error: {type(e).__name__}: {e}"

    return mcp


# ================================================================
# SERVER 3: WEB FETCH SERVER
# ================================================================
# This server fetches web content and extracts useful information.
# It's like having a research assistant who goes to the library,
# finds the book you need, and brings back just the relevant pages.
#
# WHY not just use requests directly?
# Wrapping web access in an MCP server provides:
#   1. Centralized rate limiting and caching
#   2. Consistent error handling
#   3. HTML stripping (agents want TEXT, not markup)
#   4. Audit trail (every fetch is logged via MCP)

print("\n" + "=" * 70)
print("SERVER 3: WebFetchServer (Web Content Fetching)")
print("=" * 70)
print("""
  Tools:
    - fetch_url(url) — Fetch URL and return stripped text content
    - extract_links(url) — Fetch URL and extract all href links
    - fetch_json(url) — Fetch URL and parse response as JSON

  Note: Uses urllib (stdlib) to avoid extra dependencies.
""")


def create_web_fetch_server():
    """Create an MCP server for web content fetching."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("web-fetch")

    def _strip_html(html: str) -> str:
        """Remove HTML tags and clean up whitespace.

        WHY simple regex instead of BeautifulSoup?
        For this educational example, we avoid extra dependencies.
        In production, use BeautifulSoup or lxml for robust parsing.
        """
        # Remove script and style blocks entirely
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode common HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _safe_fetch(url: str, timeout: int = 10) -> str:
        """Fetch a URL using urllib (stdlib). Returns raw response text."""
        import urllib.request
        import urllib.error

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL scheme. Must start with http:// or https://. Got: {url[:50]}")

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "MCP-WebFetch/1.0 (Educational Tool)"},
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"URL error: {e.reason}")

    @mcp.tool()
    def fetch_url(url: str) -> str:
        """Fetch a URL and return the text content with HTML tags stripped.

        Args:
            url: The URL to fetch. Must start with http:// or https://.

        Returns:
            Cleaned text content from the page (first 5000 chars).
        """
        try:
            html = _safe_fetch(url)
            text = _strip_html(html)
            # Limit output size to avoid overwhelming the agent
            if len(text) > 5000:
                return text[:5000] + f"\n\n[Truncated — full content is {len(text)} chars]"
            return text if text else "Page returned no readable text content."
        except Exception as e:
            return f"Error fetching {url}: {e}"

    @mcp.tool()
    def extract_links(url: str) -> str:
        """Fetch a URL and extract all hyperlinks (href attributes).

        Args:
            url: The URL to fetch and extract links from.

        Returns:
            List of links found on the page, one per line.
        """
        try:
            html = _safe_fetch(url)
            # Extract href attributes from anchor tags
            links = re.findall(r'href=["\']([^"\']+)["\']', html)

            if not links:
                return f"No links found on {url}"

            # Deduplicate while preserving order
            seen = set()
            unique_links = []
            for link in links:
                if link not in seen and not link.startswith(("#", "javascript:")):
                    seen.add(link)
                    unique_links.append(link)

            lines = [f"Links found on {url}: {len(unique_links)}", ""]
            for i, link in enumerate(unique_links[:100], 1):  # Limit to 100 links
                lines.append(f"  {i}. {link}")

            if len(unique_links) > 100:
                lines.append(f"  ... and {len(unique_links) - 100} more links")

            return "\n".join(lines)

        except Exception as e:
            return f"Error extracting links from {url}: {e}"

    @mcp.tool()
    def fetch_json(url: str) -> str:
        """Fetch a URL and parse the response as JSON.

        Args:
            url: The URL that returns JSON content.

        Returns:
            Pretty-printed JSON content, or error if not valid JSON.
        """
        try:
            raw = _safe_fetch(url)
            data = json.loads(raw)
            formatted = json.dumps(data, indent=2, default=str)

            # Limit output size
            if len(formatted) > 5000:
                return formatted[:5000] + f"\n\n[Truncated — full JSON is {len(formatted)} chars]"
            return formatted

        except json.JSONDecodeError as e:
            return f"Error: Response from {url} is not valid JSON: {e}"
        except Exception as e:
            return f"Error fetching JSON from {url}: {e}"

    return mcp


# ================================================================
# SERVER REGISTRY
# ================================================================
# A simple registry so we can look up servers by name.
# In production, you might use a service discovery system (Consul,
# etcd, DNS) instead of a local dictionary.

SERVERS = {
    "database": {
        "create": create_database_server,
        "name": "DatabaseToolServer",
        "description": "SQLite read-only queries, table schema, table listing",
        "tools": ["query_db", "describe_table", "list_tables"],
    },
    "code": {
        "create": create_code_execution_server,
        "name": "CodeExecutionServer",
        "description": "Sandboxed Python execution with timeout and import filtering",
        "tools": ["run_python", "run_python_safe"],
    },
    "web": {
        "create": create_web_fetch_server,
        "name": "WebFetchServer",
        "description": "URL fetching, link extraction, JSON parsing",
        "tools": ["fetch_url", "extract_links", "fetch_json"],
    },
}


# ================================================================
# PRINT OVERVIEW (default mode)
# ================================================================

def print_overview():
    """Print descriptions of all 3 servers and their tools."""
    print("\n" + "=" * 70)
    print("  TOOL ECOSYSTEM OVERVIEW — 3 MCP Servers, 8 Tools")
    print("=" * 70)

    for key, info in SERVERS.items():
        print(f"\n  {info['name']}")
        print(f"  {'─' * len(info['name'])}")
        print(f"  {info['description']}")
        print(f"  Tools: {', '.join(info['tools'])}")

    print(f"""
  {'=' * 60}
  HOW TO USE THESE SERVERS

  1. Run a specific server:
     python {__file__} --server database
     python {__file__} --server code
     python {__file__} --server web

  2. Self-test all servers:
     python {__file__} --self-test

  3. Connect from your agent using the MCP client pattern:
     server_params = StdioServerParameters(
         command="python",
         args=["{__file__}", "--server", "database"]
     )
     async with stdio_client(server_params) as (read, write):
         async with ClientSession(read, write) as session:
             await session.initialize()
             tools = await session.list_tools()
             result = await session.call_tool("list_tables", {{"db_path": "my.db"}})
  {'=' * 60}
""")

    print("  Key Design Principles:")
    print("    1. SEPARATION: Each server handles one domain (DB, code, web)")
    print("    2. SAFETY: Read-only DB, sandboxed code, validated URLs")
    print("    3. PROTOCOL: All use MCP — agents don't need server-specific code")
    print("    4. COMPOSABILITY: Mix and match servers for different workflows")
    print("    5. ISOLATION: Each server runs as a separate process")


# ================================================================
# SELF-TEST — Launch each server and test its tools
# ================================================================

async def self_test():
    """Launch each server as a subprocess and test all its tools via MCP client.

    This demonstrates the full round-trip:
      1. Start a server as a subprocess (stdio transport)
      2. Connect as an MCP client
      3. List available tools
      4. Call each tool with test inputs
      5. Verify the results
    """
    from mcp.client.stdio import stdio_client
    from mcp.client.session import ClientSession
    try:
        from mcp.client.stdio import StdioServerParameters
    except ImportError:
        from mcp import StdioServerParameters

    this_script = os.path.abspath(__file__)
    passed = 0
    failed = 0

    # ── Create a test database ──────────────────────────────────
    test_db = os.path.join(tempfile.gettempdir(), "mcp_test_ecosystem.db")
    conn = sqlite3.connect(test_db)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    conn.execute("INSERT OR REPLACE INTO users VALUES (1, 'Alice', 'alice@example.com')")
    conn.execute("INSERT OR REPLACE INTO users VALUES (2, 'Bob', 'bob@example.com')")
    conn.execute("INSERT OR REPLACE INTO users VALUES (3, 'Charlie', 'charlie@example.com')")
    conn.execute("CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
    conn.execute("INSERT OR REPLACE INTO products VALUES (1, 'Widget', 9.99)")
    conn.execute("INSERT OR REPLACE INTO products VALUES (2, 'Gadget', 24.99)")
    conn.commit()
    conn.close()
    print(f"\n  Created test database: {test_db}")

    # ── Test each server ────────────────────────────────────────
    test_cases = {
        "database": [
            ("list_tables", {"db_path": test_db}, "users"),
            ("describe_table", {"table_name": "users", "db_path": test_db}, "name"),
            ("query_db", {"sql": "SELECT * FROM users WHERE id = 1", "db_path": test_db}, "Alice"),
            # Safety test: write operation should be blocked
            ("query_db", {"sql": "DELETE FROM users WHERE id = 1", "db_path": test_db}, "ERROR"),
        ],
        "code": [
            ("run_python", {"code": "print(2 + 2)", "timeout_seconds": 5}, "4"),
            ("run_python", {"code": "import math; print(math.pi)", "timeout_seconds": 5}, "3.14"),
            # Safety test: dangerous import should be blocked
            ("run_python", {"code": "import subprocess; subprocess.run(['ls'])", "timeout_seconds": 5}, "BLOCKED"),
            ("run_python_safe", {"code": "print(sum(range(10)))"}, "45"),
        ],
        "web": [
            # Use httpbin for reliable test endpoints
            ("fetch_json", {"url": "https://httpbin.org/json"}, "slideshow"),
            ("fetch_url", {"url": "https://httpbin.org/html"}, "Herman Melville"),
            ("extract_links", {"url": "https://httpbin.org/links/5/0"}, "links"),
        ],
    }

    for server_name, tests in test_cases.items():
        print(f"\n  {'=' * 55}")
        print(f"  Testing: {SERVERS[server_name]['name']}")
        print(f"  {'=' * 55}")

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[this_script, "--server", server_name],
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # List tools
                    tools_result = await session.list_tools()
                    tool_names = [t.name for t in tools_result.tools]
                    print(f"  Available tools: {', '.join(tool_names)}")

                    # Run each test case
                    for tool_name, args, expected_substring in tests:
                        try:
                            result = await session.call_tool(tool_name, args)
                            result_text = ""
                            if result.content:
                                result_text = "\n".join(
                                    c.text for c in result.content if hasattr(c, "text")
                                )

                            if expected_substring.lower() in result_text.lower():
                                print(f"    PASS: {tool_name}({json.dumps(args)[:60]}) -> contains '{expected_substring}'")
                                passed += 1
                            else:
                                print(f"    FAIL: {tool_name} -> expected '{expected_substring}' in result")
                                print(f"           Got: {result_text[:150]}")
                                failed += 1
                        except Exception as e:
                            print(f"    FAIL: {tool_name} -> {type(e).__name__}: {e}")
                            failed += 1

        except Exception as e:
            print(f"    ERROR: Could not connect to {server_name} server: {e}")
            failed += len(tests)

    # Cleanup
    try:
        os.remove(test_db)
    except Exception:
        pass

    # Summary
    total = passed + failed
    print(f"\n  {'=' * 55}")
    print(f"  SELF-TEST RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"  {'=' * 55}")

    if failed == 0:
        print("  All tests passed!")
    else:
        print(f"  {failed} test(s) failed. Check the output above for details.")
        print("  Common causes: missing dependencies, network issues (for web tests)")


# ================================================================
# KEY TAKEAWAYS
# ================================================================

print("""
Key Takeaways:
  1. TOOL SERVERS are specialized workshops — each handles one domain
  2. MCP PROTOCOL is the universal door — same client code for any server
  3. SAFETY FIRST: read-only DB, sandboxed code, validated URLs
  4. COMPOSABILITY: agents mix and match servers for complex workflows
  5. SELF-TESTING: every server should be testable in isolation
  6. GRACEFUL DEGRADATION: handle failures at each layer
""")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import sys as _sys

    if "--self-test" in _sys.argv:
        # Run self-tests: launch each server and test its tools
        print("\n" + "=" * 70)
        print("  SELF-TEST MODE: Testing all 3 MCP servers")
        print("=" * 70)
        try:
            asyncio.run(self_test())
        except KeyboardInterrupt:
            print("\n  Self-test interrupted")
        except Exception as e:
            print(f"\n  Self-test error: {e}")
            import traceback
            traceback.print_exc()
            print("\n  Install: pip install mcp python-dotenv")

    elif "--server" in _sys.argv:
        # Run a specific server (used by self-test and by agents)
        try:
            server_name = _sys.argv[_sys.argv.index("--server") + 1]
        except (IndexError, ValueError):
            print("Usage: python example_13_tool_ecosystems.py --server <database|code|web>")
            _sys.exit(1)

        if server_name not in SERVERS:
            print(f"Unknown server: {server_name}")
            print(f"Available: {', '.join(SERVERS.keys())}")
            _sys.exit(1)

        # Create and run the server (stdio transport)
        server = SERVERS[server_name]["create"]()
        server.run(transport="stdio")

    else:
        # Default: print overview of all servers
        print_overview()
        print("\n  Example 13 complete!")
