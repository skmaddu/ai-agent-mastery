import sys; import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
Example 14: Agent Versioning and Prompt Registries
====================================================
Topic 14 — How to manage agent versions and prompt templates at scale.

The BIG IDEA (Feynman):
  Think of a recipe book.  Each recipe (prompt) has a version number.
  When you improve a recipe, you don't erase the old one — you add
  a new version.  If the new version tastes bad, you can roll back.
  And you can even do a taste test (A/B test) with two versions to
  see which one customers prefer.

  Agent versioning is the same: track changes to your agent's behavior
  so you can improve confidently and roll back safely.

First Principles:
  1. VERSIONING — Semantic versions (major.minor.patch) for agent configs
  2. REGISTRY — Centralized store for prompt templates with history
  3. ROLLBACK — Revert to any previous version instantly
  4. A/B TESTING — Compare two prompt versions with real queries
  5. AGENT CARDS — Extend with version metadata for A2A discovery

Previously covered:
  - Agent Cards (example_06)
  - A2A protocol (example_08)
  - Production patterns (example_10)

Run: python week-07-mcp-a2a-synthesis/examples/example_14_versioning_prompt_registry.py
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv("config/.env")
load_dotenv()

from pydantic import BaseModel, Field


# ================================================================
# LLM Setup
# ================================================================

def get_llm(temperature=0.3):
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
# PART 1: SEMANTIC VERSIONING FOR AGENTS
# ================================================================
# When do you bump each number?
#   MAJOR (3.0.0) — Breaking changes: new prompt format, removed tools,
#                    changed output schema. Clients MUST update.
#   MINOR (2.1.0) — New features: added a tool, new skill, better prompts.
#                    Backwards compatible.
#   PATCH (2.0.1) — Bug fixes: typo in prompt, wrong default value.
#                    No behavior change intended.

print("=" * 70)
print("PART 1: Agent Versioning (Semantic Versioning)")
print("=" * 70)


class AgentVersion(BaseModel):
    """A specific version of an agent's configuration.

    Like a snapshot of the agent at a point in time — you can always
    go back to this exact configuration.
    """
    major: int = 1
    minor: int = 0
    patch: int = 0
    description: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # What changed in this version
    changelog: str = ""
    # The actual configuration
    config: dict = Field(default_factory=dict)

    @property
    def version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def bump_major(self, changelog: str = "") -> "AgentVersion":
        return AgentVersion(
            major=self.major + 1, minor=0, patch=0,
            changelog=changelog, config=self.config.copy(),
        )

    def bump_minor(self, changelog: str = "") -> "AgentVersion":
        return AgentVersion(
            major=self.major, minor=self.minor + 1, patch=0,
            changelog=changelog, config=self.config.copy(),
        )

    def bump_patch(self, changelog: str = "") -> "AgentVersion":
        return AgentVersion(
            major=self.major, minor=self.minor, patch=self.patch + 1,
            changelog=changelog, config=self.config.copy(),
        )


class AgentVersionManager:
    """Manages the version history of an agent.

    Like Git for your agent's configuration — tracks every change
    and lets you go back to any point.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.versions: list[AgentVersion] = []
        self.current_index: int = -1

    def release(self, version: AgentVersion) -> str:
        """Release a new version."""
        self.versions.append(version)
        self.current_index = len(self.versions) - 1
        return f"Released {self.agent_name} v{version.version_string}: {version.changelog}"

    @property
    def current(self) -> Optional[AgentVersion]:
        if self.current_index >= 0:
            return self.versions[self.current_index]
        return None

    def rollback(self, version_string: str) -> str:
        """Roll back to a specific version."""
        for i, v in enumerate(self.versions):
            if v.version_string == version_string:
                self.current_index = i
                return f"Rolled back to v{version_string}"
        return f"Version {version_string} not found"

    def history(self) -> str:
        """Show version history."""
        lines = [f"📋 Version History for {self.agent_name}:"]
        for i, v in enumerate(self.versions):
            marker = " ← CURRENT" if i == self.current_index else ""
            lines.append(
                f"  v{v.version_string}{marker}\n"
                f"    {v.changelog or 'No changelog'}\n"
                f"    Released: {v.created_at[:10]}"
            )
        return "\n".join(lines)


# Demo versioning
print("\nDemo: Agent Version Manager")
print("-" * 40)

vm = AgentVersionManager("Research Agent")

v1 = AgentVersion(
    major=1, minor=0, patch=0,
    changelog="Initial release — basic research with web search",
    config={"model": "llama-3.3-70b", "temperature": 0.3, "max_tokens": 1000},
)
print(f"  {vm.release(v1)}")

v1_1 = v1.bump_minor("Added database search tool via MCP")
v1_1.config["tools"] = ["web_search", "db_search"]
print(f"  {vm.release(v1_1)}")

v1_1_1 = v1_1.bump_patch("Fixed prompt typo causing hallucinations")
print(f"  {vm.release(v1_1_1)}")

v2 = v1_1_1.bump_major("Breaking: new output schema (Pydantic v2)")
v2.config["output_schema"] = "ResearchSummaryV2"
print(f"  {vm.release(v2)}")

print(f"\n  Current: v{vm.current.version_string}")
print(f"  {vm.rollback('1.1.0')}")
print(f"  After rollback: v{vm.current.version_string}")
print(f"  {vm.rollback('2.0.0')}")  # Roll forward

print(f"\n{vm.history()}")


# ================================================================
# PART 2: PROMPT REGISTRY (SQLite-backed)
# ================================================================

print("\n" + "=" * 70)
print("PART 2: Prompt Registry")
print("=" * 70)
print("""
A Prompt Registry is a centralized store for all your prompt templates.
Instead of hardcoding prompts in your code, you store them in the registry
and reference them by name.  This lets you:
  - Version prompts independently from code
  - A/B test different prompt versions
  - Roll back a bad prompt without redeploying
  - Track which prompts are used and how they perform
""")


class PromptRegistry:
    """SQLite-backed prompt template registry with version history.

    Like a library catalog — every prompt has a name, and you can
    check out any version of it.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self):
        """Create the database tables."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                template TEXT NOT NULL,
                description TEXT DEFAULT '',
                hash TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_prompts_name
            ON prompts(name, version);

            CREATE TABLE IF NOT EXISTS prompt_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_name TEXT NOT NULL,
                prompt_version INTEGER NOT NULL,
                used_at TEXT NOT NULL,
                quality_score REAL DEFAULT NULL,
                latency_ms REAL DEFAULT NULL
            );
        """)
        self.conn.commit()

    def register(self, name: str, template: str, description: str = "", metadata: dict = None) -> dict:
        """Register a new prompt or a new version of an existing prompt."""
        # Calculate hash to detect duplicates
        content_hash = hashlib.sha256(template.encode()).hexdigest()[:16]

        # Check if this exact template already exists
        existing = self.conn.execute(
            "SELECT version FROM prompts WHERE name = ? AND hash = ?",
            (name, content_hash)
        ).fetchone()

        if existing:
            return {
                "status": "unchanged",
                "name": name,
                "version": existing["version"],
                "message": "Template identical to existing version",
            }

        # Get the next version number
        max_version = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) as max_v FROM prompts WHERE name = ?",
            (name,)
        ).fetchone()["max_v"]

        new_version = max_version + 1
        now = datetime.now(timezone.utc).isoformat()

        # Deactivate previous versions
        self.conn.execute(
            "UPDATE prompts SET is_active = 0 WHERE name = ?", (name,)
        )

        # Insert new version
        self.conn.execute(
            """INSERT INTO prompts (name, version, template, description, hash, is_active, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
            (name, new_version, template, description, content_hash, now,
             json.dumps(metadata or {}))
        )
        self.conn.commit()

        return {
            "status": "registered",
            "name": name,
            "version": new_version,
            "hash": content_hash,
        }

    def get(self, name: str, version: int = None) -> Optional[str]:
        """Get a prompt template by name (latest active version by default)."""
        if version:
            row = self.conn.execute(
                "SELECT template FROM prompts WHERE name = ? AND version = ?",
                (name, version)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT template FROM prompts WHERE name = ? AND is_active = 1",
                (name,)
            ).fetchone()

        return row["template"] if row else None

    def rollback(self, name: str, version: int) -> str:
        """Roll back a prompt to a specific version."""
        exists = self.conn.execute(
            "SELECT 1 FROM prompts WHERE name = ? AND version = ?",
            (name, version)
        ).fetchone()

        if not exists:
            return f"Version {version} not found for '{name}'"

        self.conn.execute(
            "UPDATE prompts SET is_active = 0 WHERE name = ?", (name,)
        )
        self.conn.execute(
            "UPDATE prompts SET is_active = 1 WHERE name = ? AND version = ?",
            (name, version)
        )
        self.conn.commit()
        return f"Rolled back '{name}' to version {version}"

    def log_usage(self, name: str, version: int, quality_score: float = None, latency_ms: float = None):
        """Log when a prompt is used (for analytics)."""
        self.conn.execute(
            """INSERT INTO prompt_usage (prompt_name, prompt_version, used_at, quality_score, latency_ms)
               VALUES (?, ?, ?, ?, ?)""",
            (name, version, datetime.now(timezone.utc).isoformat(), quality_score, latency_ms)
        )
        self.conn.commit()

    def get_history(self, name: str) -> list[dict]:
        """Get all versions of a prompt."""
        rows = self.conn.execute(
            """SELECT version, description, hash, is_active, created_at
               FROM prompts WHERE name = ? ORDER BY version""",
            (name,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self, name: str) -> dict:
        """Get usage statistics for a prompt."""
        stats = self.conn.execute(
            """SELECT
                prompt_version,
                COUNT(*) as uses,
                AVG(quality_score) as avg_quality,
                AVG(latency_ms) as avg_latency
               FROM prompt_usage
               WHERE prompt_name = ?
               GROUP BY prompt_version""",
            (name,)
        ).fetchall()
        return [dict(s) for s in stats]

    def list_all(self) -> list[dict]:
        """List all registered prompts (active versions only)."""
        rows = self.conn.execute(
            """SELECT name, version, description, created_at
               FROM prompts WHERE is_active = 1 ORDER BY name"""
        ).fetchall()
        return [dict(r) for r in rows]


# Demo the registry
print("\nDemo: Prompt Registry")
print("-" * 40)

registry = PromptRegistry()  # In-memory for demo

# Register prompts
result = registry.register(
    "research_agent_system",
    "You are a research assistant. Search for information and provide structured summaries with citations.",
    "Initial research agent prompt",
)
print(f"  📝 {result['status']}: {result['name']} v{result['version']}")

result = registry.register(
    "research_agent_system",
    "You are an expert research analyst. Search thoroughly, cross-reference sources, and provide structured summaries with confidence scores and citations.",
    "Improved: added cross-referencing and confidence scores",
)
print(f"  📝 {result['status']}: {result['name']} v{result['version']}")

result = registry.register(
    "writer_agent_system",
    "You are a professional writer. Transform research notes into engaging, well-structured articles.",
    "Initial writer agent prompt",
)
print(f"  📝 {result['status']}: {result['name']} v{result['version']}")

result = registry.register(
    "research_agent_system",
    "You are an expert research analyst specializing in technology topics. Search thoroughly using multiple sources, cross-reference for accuracy, rate source reliability (1-5), and provide structured summaries with confidence scores and full citations.",
    "V3: added source reliability rating",
)
print(f"  📝 {result['status']}: {result['name']} v{result['version']}")

# Show current prompts
print("\n  📚 Active prompts:")
for p in registry.list_all():
    print(f"    • {p['name']} v{p['version']}: {p['description'][:50]}...")

# Show history
print("\n  📜 History for 'research_agent_system':")
for h in registry.get_history("research_agent_system"):
    active = " ← ACTIVE" if h["is_active"] else ""
    print(f"    v{h['version']}: {h['description'][:50]}...{active}")

# Get a prompt
prompt = registry.get("research_agent_system")
print(f"\n  📄 Current 'research_agent_system' prompt:")
print(f"    {prompt[:80]}...")

# Rollback
print(f"\n  {registry.rollback('research_agent_system', 2)}")
prompt_v2 = registry.get("research_agent_system")
print(f"  After rollback: {prompt_v2[:80]}...")

# Log usage
registry.log_usage("research_agent_system", 2, quality_score=8.5, latency_ms=1200)
registry.log_usage("research_agent_system", 2, quality_score=7.0, latency_ms=950)
registry.log_usage("research_agent_system", 3, quality_score=9.2, latency_ms=1500)

stats = registry.get_stats("research_agent_system")
print("\n  📊 Usage stats:")
for s in stats:
    print(f"    v{s['prompt_version']}: {s['uses']} uses, "
          f"avg quality: {s['avg_quality']:.1f}, avg latency: {s['avg_latency']:.0f}ms")


# ================================================================
# PART 3: A/B TESTING PROMPTS
# ================================================================

print("\n" + "=" * 70)
print("PART 3: A/B Testing Prompts")
print("=" * 70)
print("""
A/B testing lets you compare two prompt versions with the same queries
to see which performs better — like a blind taste test.
""")


class ABTestRunner:
    """Run A/B tests comparing two prompt versions."""

    def __init__(self, registry: PromptRegistry):
        self.registry = registry

    def run_test(
        self,
        prompt_name: str,
        version_a: int,
        version_b: int,
        test_queries: list[str],
        judge_prompt: str = None,
    ) -> dict:
        """Compare two prompt versions on the same queries.

        Returns comparison results (without actual LLM calls for demo).
        In production, you'd call the LLM with each version and use
        the judge to score both responses.
        """
        prompt_a = self.registry.get(prompt_name, version_a)
        prompt_b = self.registry.get(prompt_name, version_b)

        if not prompt_a or not prompt_b:
            return {"error": "One or both versions not found"}

        results = {
            "prompt_name": prompt_name,
            "version_a": version_a,
            "version_b": version_b,
            "test_queries": len(test_queries),
            "comparisons": [],
        }

        print(f"\n  🧪 A/B Test: v{version_a} vs v{version_b}")
        print(f"  Prompt A (v{version_a}): {prompt_a[:60]}...")
        print(f"  Prompt B (v{version_b}): {prompt_b[:60]}...")

        for query in test_queries:
            # In production: call LLM with both prompts, then judge
            comparison = {
                "query": query,
                "note": "In production, both versions would be called with the LLM and scored by the judge",
            }
            results["comparisons"].append(comparison)
            print(f"\n  Query: '{query[:50]}...'")
            print(f"    Version A: [would generate response with v{version_a} prompt]")
            print(f"    Version B: [would generate response with v{version_b} prompt]")
            print(f"    Judge: [would score both and pick the winner]")

        results["recommendation"] = (
            f"Run with real LLM calls to determine winner between v{version_a} and v{version_b}"
        )
        return results


# Demo A/B testing
ab_tester = ABTestRunner(registry)
test_result = ab_tester.run_test(
    "research_agent_system",
    version_a=2,
    version_b=3,
    test_queries=[
        "Research the current state of quantum computing",
        "What are the latest advances in AI safety?",
        "Summarize the impact of MCP protocol on agent development",
    ],
)
print(f"\n  📊 Recommendation: {test_result['recommendation']}")


# ================================================================
# PART 4: VERSIONED AGENT CARDS
# ================================================================

print("\n" + "=" * 70)
print("PART 4: Versioned Agent Cards for A2A Discovery")
print("=" * 70)
print("""
Extend Agent Cards with version metadata so other agents can:
  - Discover which version of an agent is running
  - Check compatibility before sending tasks
  - Find agents that support specific API versions
""")


class VersionedAgentCard(BaseModel):
    """Agent Card extended with version and compatibility info."""
    name: str
    description: str
    url: str
    version: str  # Semantic version (e.g., "2.1.0")
    min_compatible_version: str = "1.0.0"  # Minimum client version
    deprecated_versions: list[str] = Field(default_factory=list)
    prompt_versions: dict = Field(
        default_factory=dict,
        description="Map of prompt name → active version number"
    )
    changelog_url: Optional[str] = None
    skills: list[dict] = Field(default_factory=list)
    capabilities: dict = Field(default_factory=dict)

    def is_compatible(self, client_version: str) -> bool:
        """Check if a client version is compatible with this agent."""
        def parse_version(v: str) -> tuple:
            return tuple(int(x) for x in v.split("."))

        client = parse_version(client_version)
        min_compat = parse_version(self.min_compatible_version)
        return client >= min_compat


# Demo versioned agent card
card = VersionedAgentCard(
    name="Research Agent",
    description="Expert research with web search, database, and citation support",
    url="http://localhost:8001",
    version="2.1.0",
    min_compatible_version="1.5.0",
    deprecated_versions=["1.0.0", "1.1.0"],
    prompt_versions={
        "research_agent_system": 3,
        "citation_formatter": 1,
    },
    changelog_url="https://github.com/ai-agent-mastery/CHANGELOG.md",
    skills=[
        {"id": "research", "name": "Topic Research", "version": "2.0"},
        {"id": "fact-check", "name": "Fact Checking", "version": "1.1"},
    ],
    capabilities={"streaming": True, "mcp": True, "a2a": True},
)

print(f"\n  📇 Versioned Agent Card:")
print(json.dumps(card.model_dump(), indent=2))

# Compatibility checks
print(f"\n  🔍 Compatibility checks:")
for client_ver in ["1.0.0", "1.5.0", "2.0.0", "3.0.0"]:
    compat = card.is_compatible(client_ver)
    status = "✅ Compatible" if compat else "❌ Incompatible"
    print(f"    Client v{client_ver}: {status}")


# ================================================================
# PART 5: CHANGELOG TRACKER
# ================================================================

print("\n" + "=" * 70)
print("PART 5: Changelog Tracker")
print("=" * 70)


class ChangelogEntry(BaseModel):
    version: str
    date: str
    category: str  # "added", "changed", "fixed", "removed", "security"
    description: str


class ChangelogTracker:
    """Track changes across agent versions in a structured format."""

    def __init__(self):
        self.entries: list[ChangelogEntry] = []

    def add(self, version: str, category: str, description: str):
        self.entries.append(ChangelogEntry(
            version=version,
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            category=category,
            description=description,
        ))

    def format_markdown(self) -> str:
        """Generate a CHANGELOG.md format string."""
        lines = ["# Changelog\n"]
        current_version = None

        for entry in sorted(self.entries, key=lambda e: e.version, reverse=True):
            if entry.version != current_version:
                lines.append(f"\n## [{entry.version}] - {entry.date}\n")
                current_version = entry.version

            category_emoji = {
                "added": "✨", "changed": "🔄", "fixed": "🐛",
                "removed": "🗑️", "security": "🔒",
            }
            emoji = category_emoji.get(entry.category, "📝")
            lines.append(f"- {emoji} **{entry.category.title()}**: {entry.description}")

        return "\n".join(lines)


# Demo changelog
changelog = ChangelogTracker()
changelog.add("2.1.0", "added", "MCP integration for database tool access")
changelog.add("2.1.0", "added", "A2A Agent Card with version metadata")
changelog.add("2.1.0", "changed", "Improved research prompt with source reliability scoring")
changelog.add("2.0.0", "changed", "Breaking: new output schema using Pydantic v2")
changelog.add("2.0.0", "removed", "Legacy JSON output format")
changelog.add("1.1.1", "fixed", "Prompt typo causing hallucinations in medical topics")
changelog.add("1.1.0", "added", "Database search tool via MCP")
changelog.add("1.1.0", "security", "Added input sanitization for prompt injection defense")
changelog.add("1.0.0", "added", "Initial release with web search and basic summarization")

print(changelog.format_markdown())


# ================================================================
# KEY TAKEAWAYS
# ================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("""
1. Version your agents with semantic versioning (major.minor.patch)
2. Store prompts in a registry, not hardcoded in source code
3. A/B test prompt changes before rolling them out to all users
4. Extend Agent Cards with version metadata for compatibility checks
5. Keep a structured changelog for every agent version
6. Rollback is your safety net — always keep old versions available

Best practices:
  • MAJOR bump: breaking changes to output format or removed tools
  • MINOR bump: new tools, new skills, improved prompts
  • PATCH bump: bug fixes, typo corrections
  • Always test before promoting a new version
  • Use the registry's usage stats to compare version performance
""")

print("✅ Example 14 complete!")
