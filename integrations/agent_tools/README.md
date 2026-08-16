# Shared Agent Tools

Shared MCP servers installed once, used by all agent harnesses.

## Servers

| Server | Binary | Tools | Description |
|--------|--------|-------|-------------|
| **SearXNG** | `node_modules/.bin/mcp-searxng` | `web_search`, `browse_url` | Web search + URL browsing via local SearXNG |
| **Firecrawl** | `node_modules/.bin/firecrawl-mcp` | `firecrawl_search`, `firecrawl_scrape`, crawling and extraction | Self-hosted web search, scraping, crawling, and extraction on `localhost:3002` |
| **Filesystem** | `node_modules/.bin/mcp-server-filesystem` | filesystem read/write | File system access (scoped to `/home/addison-gambhir`) |
| **GitHub** | `node_modules/.bin/mcp-server-github` | GitHub API tools | Repo/file/PR access via PAT |
| **Context7** | `node_modules/.bin/context7-mcp` | context7 tools | Documentation lookup |
| **Local LLM** | `node local-llm-mcp.js` | `local_llm_chat`, `local_llm_context`, `local_llm_status` | Read-only local-LLM delegation over llama.cpp |
| **Codebase Memory** | `node_modules/.bin/codebase-memory-mcp` | `index_repository`, `search_graph`, `query_graph`, `trace_path`, `get_code_snippet`, `get_graph_schema`, `get_architecture`, `search_code` | Code knowledge graph — indexes repos into a persistent graph (SQLite in `~/.cache/codebase-memory-mcp/`) for structural queries, call-path tracing, and code search |

## Local LLM MCP Server

The `local-llm-mcp.js` server lets any agent harness delegate bounded analysis
tasks to the local llama.cpp model. It is deliberately read-only: include the
needed context in the request, then let the calling harness verify and act on
the result with its own tools.

### Tools

- **`local_llm_chat`** — One prompt plus optional system instructions. Best for
  summaries, alternatives, log analysis, and second opinions.
- **`local_llm_context`** — Continue a supplied multi-turn conversation. The
  server does not retain state between calls.
- **`local_llm_status`** — Confirm that the configured llama.cpp endpoint and
  model are reachable without generating text.

The response-token limit defaults to 2,048 and is capped at 4,096. Model
reasoning is hidden by default to avoid needlessly consuming the cloud
harness's context; callers can request it with `include_reasoning: true`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | Loopback-only LLM API base URL |
| `LOCAL_LLM_MODEL` | `local` | Model name |
| `LOCAL_LLM_TIMEOUT` | `300000` | Request timeout in ms (5 min) |

### Use Cases

- **Context gathering**: paste a file, diff, or error log and ask for a summary
- **Reasoning**: "Analyze the tradeoffs of approach A vs B"
- **Drafting**: "Write a commit message for these changes"

## Codebase Memory MCP Server

`codebase-memory-mcp` (DeusData) indexes a codebase into a persistent knowledge
graph so agents can answer structural questions (callers/callees, paths between
symbols, architecture) in one query instead of dozens of grep/read round-trips.
Single static binary shipped via npm; fully local, no API keys.

- **8 MCP tools, ~3k tokens of schemas total** — safe for local llama.cpp models
  (unlike the 102-tool setups that overload grammar-constrained tool calling)
- Graphs persist as one SQLite db per project in `~/.cache/codebase-memory-mcp/`
- `auto_watch` is on by default: indexed projects re-sync on file changes
- CLI mode for one-off queries (note: pass args as flags, not raw JSON):
  ```bash
  node_modules/.bin/codebase-memory-mcp cli index_repository --repo_path /path/to/repo
  node_modules/.bin/codebase-memory-mcp cli search_graph --project <project> --query "..."
  ```
- Optional graph visualization UI: run with `--ui=true` (default port 9749)
- Do NOT use its built-in `install` command — it auto-edits harness configs;
  we manage MCP configs manually per harness below.

MCP config (Qwen Code, Cline, Claude Code, or another MCP client):

```json
{
  "codebase-memory": {
    "command": "/home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/node_modules/.bin/codebase-memory-mcp"
  }
}
```

OpenCode / Kilo shape:

```json
{
  "codebase-memory": {
    "type": "local",
    "command": ["/home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/node_modules/.bin/codebase-memory-mcp"]
  }
}
```

OpenHands:

```bash
openhands mcp add codebase-memory --transport stdio \
  /home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/node_modules/.bin/codebase-memory-mcp
```

No environment variables required.

## Adding to an Agent Harness

### Hermes (`.hermes/config.yaml`)

```yaml
mcp_servers:
  local_llm:
    command: node
    args:
    - /home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/local-llm-mcp.js
    env:
      LOCAL_LLM_BASE_URL: http://localhost:1234/v1
      LOCAL_LLM_MODEL: local
      LOCAL_LLM_TIMEOUT: "300000"
    trust: true
```

Hermes uses its native Firecrawl integration rather than this MCP server. Install
the `firecrawl` extra, set `FIRECRAWL_API_URL=http://localhost:3002` (and optionally
`FIRECRAWL_API_KEY=fc-local`) in `~/.hermes/.env`, then set
`web.extract_backend: firecrawl` in `~/.hermes/config.yaml`.

### Firecrawl (Qwen Code, Cline, OpenCode, Claude Code, or another MCP client)

Add the following entry alongside the existing MCP servers. The absolute binary
path keeps every client on the shared installation.

```json
{
  "firecrawl": {
    "command": "/home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/node_modules/.bin/firecrawl-mcp",
    "env": {
      "FIRECRAWL_API_URL": "http://localhost:3002",
      "FIRECRAWL_API_KEY": "fc-local"
    }
  }
}
```

For OpenCode, use the equivalent local-MCP shape:

```json
{
  "firecrawl": {
    "type": "local",
    "command": ["/home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/node_modules/.bin/firecrawl-mcp"],
    "environment": {
      "FIRECRAWL_API_URL": "http://localhost:3002",
      "FIRECRAWL_API_KEY": "fc-local"
    }
  }
}
```

Kilo CLI uses the same OpenCode-compatible MCP format in
`~/.config/kilo/kilo.jsonc`.

### Claude Code (`.claude/CLAUDE.md` or MCP config)

```json
{
  "mcpServers": {
    "local_llm": {
      "command": "node",
      "args": ["/home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/local-llm-mcp.js"],
      "env": {
        "LOCAL_LLM_BASE_URL": "http://localhost:1234/v1",
        "LOCAL_LLM_MODEL": "local",
        "LOCAL_LLM_TIMEOUT": "300000"
      }
    }
  }
}
```

### Cline (VS Code extension)

Create or edit `~/.cline/cline_mcp_settings.json`:
```json
{
  "mcpServers": {
    "local_llm": {
      "command": "node",
      "args": ["/home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/local-llm-mcp.js"],
      "env": {
        "LOCAL_LLM_BASE_URL": "http://localhost:1234/v1",
        "LOCAL_LLM_MODEL": "local",
        "LOCAL_LLM_TIMEOUT": "300000"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

Add other shared servers (context7, searxng, filesystem, github, screenshot) as additional entries under `mcpServers` with the same command/args/env pattern as the Qwen Code config above.

### Codex

Register the shared server globally:

```bash
codex mcp add local_llm \
  --env LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1 \
  --env LOCAL_LLM_MODEL=local \
  --env LOCAL_LLM_TIMEOUT=300000 \
  -- /usr/bin/node /home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/local-llm-mcp.js
```

Start a new Codex session after adding it.

### OpenHands

OpenHands stores its global MCP registry in `~/.openhands/mcp.json`. Register
the shared Firecrawl scraper and the self-hosted SearXNG search server with:

```bash
openhands mcp add firecrawl --transport stdio \
  --env FIRECRAWL_API_URL=http://localhost:3002 \
  --env FIRECRAWL_API_KEY=fc-local \
  /usr/bin/node -- /home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/node_modules/firecrawl-mcp/dist/index.js

openhands mcp add searxng --transport stdio \
  --env SEARXNG_URL=http://localhost:8080 \
  /usr/bin/node -- /home/addison-gambhir/Desktop/local_llm/integrations/agent_tools/node_modules/mcp-searxng/dist/cli.js
```

Start a new OpenHands conversation after adding them. It will receive
`firecrawl_scrape`/`firecrawl_search` plus SearXNG search and URL-reading
tools through MCP.

## Adding New Shared Tools

1. Add the package to `package.json` dependencies
2. Run `npm install`
3. Document the new server in this README
4. Add the MCP config to each agent harness that should use it
