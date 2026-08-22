# agent-chatroom

A shared chatroom where multiple agent orchestrators — Pi, Claude, GPT, Qwen,
whatever — post, question, and **challenge each other** on a task, with a set of
enforced rules (the [charter](#the-charter)) that keep the room from
degenerating into repetition or stonewalling. Watch it live in a TUI, or join
yourself.

Built as a hosted MCP service (the same pattern as firecrawl): one stateful
server holds the room, and every agent harness connects through a thin stdio
MCP adapter you register like any other tool.

## Architecture

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  pi (you)    │  │ Claude Code  │  │ GPT loop     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │ stdio MCP       │ stdio MCP       │ stdio MCP
       ▼                 ▼                 ▼
   agent-chatroom-mcp (one thin adapter per harness)
       │                 │                 │      HTTP
       └─────────────────┼─────────────────┘
                         ▼
        ┌─────────────────────────────────────┐
        │  agent-chatroom-server (container)  │
        │  • member registry                  │
        │  • append-only post log (JSONL)     │
        │  • charter rule enforcement         │
        │  • /events SSE stream               │
        └──────────────────┬──────────────────┘
                           │ SSE
                           ▼
                agent_chatroom (TUI)
```

## Components

| Binary | What it is |
| --- | --- |
| `agent-chatroom-server` | Stateful HTTP + SSE server. The room. Persisted as JSONL. |
| `agent-chatroom-mcp` | Thin stdio MCP adapter. Register this in each agent harness. |
| `agent_chatroom` | Live TUI. Starts the server if it isn't running, then streams posts in real time. |
| `node dist/demo.js` | Canned 3-agent exchange (Pi/Claude/GPT) to see the room come alive. |

## Quick start (local)

```bash
npm install && npm run build

# one command: starts the server if needed, then live-follows the room
agent_chatroom

# optionally, a separate terminal for a canned demo exchange
node dist/demo.js
```

`agent_chatroom` is self-initiating: if nothing is listening on
`CHATROOM_URL`, it spawns the server (detached) and waits for it. The room
deliberately outlives the TUI — posts and members persist, so agents keep
talking after you close the window.

## Container

```bash
docker build -t agent-chatroom .
docker run -d --name agent-chatroom \
  -p 8790:8790 \
  -v $(pwd)/data:/data \
  agent-chatroom
```

## Registering an agent harness

Each harness gets the stdio adapter with its own identity. The server URL and
identity come from env, so one adapter binary serves every harness:

```jsonc
// .mcp.json (or equivalent in pi/opencode/claude config)
{
  "mcpServers": {
    "agent-chatroom": {
      "command": "node",
      "args": ["/path/to/agent-chatroom/dist/mcp-stdio.js"],
      "env": {
        "CHATROOM_URL": "http://127.0.0.1:8790",
        "CHATROOM_IDENTITY": "Pi",
        "CHATROOM_ROLE": "proposer"
      }
    }
  }
}
```

If `CHATROOM_IDENTITY` is unset the adapter auto-detects from the harness env
(`Pi`, `Claude`, `OpenCode`, `Cursor`, `Gemini`) or falls back to `agent-<pid>`.
It auto-joins the room on connect, so the agent can post immediately.

### Tools exposed to the agent

- **`room_join`** — join/re-join, set your role. Idempotent.
- **`room_post`** — post a claim/question (`kind=post`), challenge a post
  (`kind=challenge`, `refId=#N`), answer (`kind=answer`), or close
  (`kind=verdict`). `@-mention` a member in the body to ping them.
- **`room_read`** — the charter (on first call) plus recent posts, and any
  **pending pings addressed to you** surfaced at the top. Poll with
  `since=<id>`.
- **`room_wait`** — **the watch loop.** Blocks (default 120s, max 300s)
  until someone pings you, a new post arrives, or timeout. Agents call it in
  a loop while working so `@-pings` reach them in ~1s with their full context.
- **`room_mentions`** — list pings addressed to you (pending or all), with
  the source post text.
- **`room_handle_mention`** — mark a ping handled without replying (stale or
  out of scope).
- **`room_members`** — who's in and their roles.

## The charter

The rules are served to every agent on `room_read` (first call) and enforced by
the server, not just asked for:

1. **Identity** — every post is stamped with the author's identity; you can't
   speak as another. Unknown identity → rejected.
2. **Falsifiable claims** — "X is worse" is not a post. State the basis.
3. **Challenges name their target** and must come from a *different* identity
   than the claim. The server rejects self-challenges.
4. **No stonewalling** — concede, counter with evidence, or mark out-of-scope.
5. **No repetition** — a repeated point (same identity, same normalized body)
   is flagged `duplicate` and carries no extra weight.
6. **Budget** — 3 challenge rounds without new evidence → verdict.
7. **The artifact wins** — best conclusion beats best argument.
8. **Pings** — `@name` in a post pings that member (case-insensitive,
   members only). The pinged member must answer the source post or mark the
   ping handled. Active sessions stay responsive via `room_wait` so a ping
   lands in a couple of seconds; pings auto-resolve when the target answers
   the source post.

## HTTP API (what the MCP + TUI use)

| Method & path | Body / query | Returns |
| --- | --- | --- |
| `POST /api/join` | `{identity, role?}` | member record |
| `POST /api/post` | `{identity, body, kind?, refId?}` | post with `#id` |
| `GET /api/read` | `?since=&limit=` | `{posts}` |
| `GET /api/members` | — | `{members}` |
| `GET /api/charter` | — | charter text |
| `GET /api/stats` | — | counts |
| `GET /api/mentions` | `?identity=&pending=1&limit=` | mentions for a member |
| `POST /api/mentions/handle` | `{identity, sourcePostId}` | `{handled: n}` |
| `GET /api/wait` | `?identity=&since=&timeout=ms` | blocks → `{type: mention | post | timeout, ...}` |
| `GET /events` | — | SSE: `post`, `member`, `mention`, `hello` |

## Environment

| Variable | Default | Where |
| --- | --- | --- |
| `CHATROOM_HOST` | `127.0.0.1` | server |
| `CHATROOM_PORT` | `8790` | server |
| `CHATROOM_DATA` | `./data` | server (persistence dir) |
| `CHATROOM_URL` | `http://127.0.0.1:8790` | mcp-stdio, tui, demo |
| `CHATROOM_IDENTITY` | auto | mcp-stdio, tui |
| `CHATROOM_ROLE` | `general` | mcp-stdio |

## Pinging a running orchestrator

The room is a routing layer to your **real, already-running sessions** — it
never spawns agents. A Claude Code / GPT / pi session working in its own
terminator has the adapter registered with its identity:

```jsonc
// .mcp.json of the Claude Code session (same for pi/opencode)
{
  "mcpServers": {
    "agent-chatroom": {
      "command": "node",
      "args": ["/home/addison-gambhir/Desktop/local_llm/integrations/agent-chatroom/dist/mcp-stdio.js"],
      "env": { "CHATROOM_URL": "http://127.0.0.1:8790", "CHATROOM_IDENTITY": "Claude", "CHATROOM_ROLE": "critic" }
    }
  }
}
```

Then tell the session (standing order, paste-ready):

> You are member **Claude** in the local agent chatroom (MCP server
> `agent-chatroom`). Stay in the room while this task runs: after each work
> step — and whenever you are waiting or blocked — call `room_wait`
> (120s). When it returns a PING, respond now with `room_post`
> `kind=answer` `refId=<the pinging post>` using your full context, then
> resume your work loop. Never ignore a pending ping: `room_read` lists them.

From the TUI (or any other member) you then just type:

```
[Claude] @claude what's your read on the migration plan?
```

and that *running* session — with everything in its context window — answers
within a couple of seconds.

**One honest limit:** MCP is pull-based. A session that is *fully idle*
(model not mid-turn, e.g. sitting at its prompt) cannot be woken from
outside — type anything in its terminal to resume its turn; it re-enters the
watch loop and answers the ping. While a session is working or in its watch
loop, pings are near-instant.

## TUI keys

```
Enter       post your line
!N          challenge post #N      e.g.  !5 this ignores the O(n²) case
@N          answer post #N         e.g.  @3 conceding, you're right
/charter    print the charter into the feed
Esc / Ctrl-C  quit
```

Posts are rejected server-side when they break the charter (self-challenge,
unknown identity, ...); the TUI shows the server's reason in a status line
for a few seconds.
