#!/usr/bin/env node
/**
 * agent-chatroom MCP adapter (stdio).
 *
 * The thin bridge you register in an agent harness the same way you register
 * firecrawl. It proxies four tools to the running chatroom server:
 *
 *   room_join, room_post, room_read, room_members
 *
 * On startup it joins the room with your identity so you can post
 * immediately without a separate join call.
 *
 * Environment:
 *   CHATROOM_URL      Server base URL (default: http://127.0.0.1:8790)
 *   CHATROOM_IDENTITY Your name in the room (default: auto from harness)
 *   CHATROOM_ROLE     Your role, e.g. proposer/critic/scout (default: general)
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const BASE = (process.env.CHATROOM_URL || "http://127.0.0.1:8790").replace(
  /\/$/,
  "",
);
const ROLE = process.env.CHATROOM_ROLE || "general";

// Derive a stable identity from the harness env if the user didn't set one.
function detectIdentity(): string {
  if (process.env.CHATROOM_IDENTITY) return process.env.CHATROOM_IDENTITY;
  const hints: [string, string][] = [
    ["PI_MODEL", "Pi"],
    ["CLAUDECODE", "Claude"],
    ["OPENCODE", "OpenCode"],
    ["CURSOR_TRACE_ID", "Cursor"],
    ["GEMINI", "Gemini"],
  ];
  for (const [env, name] of hints) if (process.env[env]) return name;
  return `agent-${process.pid}`;
}
const IDENTITY = detectIdentity();

async function api<T = any>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const body = (await res.json().catch(() => ({}))) as T & { error?: string };
  if (!res.ok) throw new Error(body.error || `${res.status} on ${path}`);
  return body;
}

function txt(s: string) {
  return { content: [{ type: "text" as const, text: s }] };
}

const server = new McpServer({ name: "agent-chatroom", version: "0.1.0" });

server.tool(
  "room_join",
  "Join (or re-join) the agent chatroom as your identity. Idempotent — call it any time to update your role. Returns your membership record.",
  {
    role: z
      .string()
      .optional()
      .describe("Role, e.g. proposer, critic, scout, judge"),
  },
  async ({ role }) => {
    const m = await api<any>("/api/join", {
      method: "POST",
      body: JSON.stringify({ identity: IDENTITY, role: role || ROLE }),
    });
    return txt(
      `You are in the room as [${m.identity}] (role: ${m.role}). Charter:\n- Every post states your identity (the room enforces it).\n- Claims must be falsifiable.\n- Challenges reference a post id (#N) and must come from a different identity than the claim.\n- No stonewalling: concede, counter with evidence, or mark out-of-scope.\n- Repeated points are flagged duplicate and carry no weight.\n- When pinged with @${m.identity}, respond to the source post (or mark it handled with a reason).\n\nStay responsive: while you are working or waiting, call room_wait in a loop. It blocks until someone pings you or posts, then returns — so @${m.identity} reaches you in ~1s with your full context. Post with room_post; read with room_read.`,
    );
  },
);

server.tool(
  "room_post",
  "Post to the agent chatroom. kind=post (default) for claims/questions; kind=challenge with refId=#N to attack a specific post; kind=answer with refId=#N to respond; kind=verdict to close a thread.",
  {
    body: z
      .string()
      .describe("The post text. Make claims falsifiable and state your basis."),
    kind: z.enum(["post", "challenge", "answer", "verdict"]).optional(),
    refId: z
      .number()
      .int()
      .optional()
      .describe(
        "Post id being challenged/answered (required for challenge/answer).",
      ),
  },
  async ({ body, kind, refId }) => {
    const p = await api<any>("/api/post", {
      method: "POST",
      body: JSON.stringify({
        identity: IDENTITY,
        body,
        kind: kind || "post",
        refId: refId ?? null,
      }),
    });
    const ref = p.refId ? ` →#${p.refId}` : "";
    const flag = p.flags?.length ? ` [flags: ${p.flags.join(",")}]` : "";
    return txt(`Posted #${p.id} ${ref} as [${p.identity}]${flag}\n"${p.body}"`);
  },
);

server.tool(
  "room_wait",
  "Watch the room. Blocks (up to timeoutSeconds) until one of: (1) someone @-pings you — returned with the source post so you can respond immediately, (2) a new post from another identity, or (3) timeout. Call this in a loop while working or idle so @-pings reach you in ~1s with your full context. On a ping, respond with room_post kind=answer refId=<source post id>.",
  {
    timeoutSeconds: z
      .number()
      .int()
      .min(5)
      .max(300)
      .optional()
      .describe("How long to block (default 120, max 300)."),
    since: z
      .number()
      .int()
      .optional()
      .describe(
        "Only wake on posts newer than this id. Default 0 = any new post.",
      ),
  },
  async ({ timeoutSeconds, since }) => {
    const timeoutMs = (timeoutSeconds ?? 120) * 1000;
    const res = await fetch(
      `${BASE}/api/wait?identity=${encodeURIComponent(IDENTITY)}&timeout=${timeoutMs}&since=${since ?? 0}`,
      { signal: AbortSignal.timeout(timeoutMs + 15000) },
    );
    const data = (await res.json()) as any;
    if (data.type === "timeout") {
      return txt(
        `Timed out after ${timeoutSeconds ?? 120}s — no pings, no new posts. Continue working or call room_wait again.`,
      );
    }
    if (data.type === "mention") {
      const m = data.mention;
      const p = data.post;
      const quote = p ? `\"${p.body.slice(0, 500)}\"` : "";
      return txt(
        `⚠ PING → you, from [${m.sourceIdentity}] in post #${m.sourcePostId}: ${quote}\nRespond now: room_post kind=answer refId=${m.sourcePostId} (or room_handle_mention if out of scope).`,
      );
    }
    if (data.type === "post") {
      const p = data.post;
      const ref = p.refId ? ` →#${p.refId}` : "";
      return txt(
        `NEW POST #${p.id} [${p.identity}] (${p.role}) ${p.kind}${ref}:\n${p.body.slice(0, 800)}`,
      );
    }
    return txt(`room_wait returned unknown event: ${JSON.stringify(data)}`);
  },
);

server.tool(
  "room_read",
  "Read the chatroom. Returns the charter on first call plus recent posts. Use since=<id> to only get posts after a given id (for polling). Each post shows its #id, identity, role, kind, and any ref it challenges/answers.",
  {
    since: z
      .number()
      .int()
      .optional()
      .describe("Only return posts with id greater than this."),
    limit: z.number().int().optional().describe("Max posts (default 30)."),
  },
  async ({ since, limit }) => {
    const [readRes, mentionRes] = await Promise.all([
      api<any>(`/api/read?since=${since || 0}&limit=${limit || 30}`),
      api<any>(
        `/api/mentions?identity=${encodeURIComponent(IDENTITY)}&pending=1`,
      ),
    ]);
    const posts = readRes.posts;
    const pings = await formatPings(mentionRes.mentions || []);
    if (!posts.length && !pings)
      return txt(
        since
          ? "No new posts."
          : pings || "The room is empty. Be the first to post.",
      );
    const lines = posts.map((p: any) => {
      const ref = p.refId ? ` →#${p.refId}` : "";
      const flag = p.flags?.length ? ` [${p.flags.join(",")}]` : "";
      const t = p.at.slice(11, 19);
      const pinged = p.body.match(/@[A-Za-z0-9][\w-]*/g);
      const pingNote = pinged ? ` ${pinged.join(" ")}` : "";
      return `#${p.id} [${p.identity}] (${p.role}) ${p.kind}${ref} ${t}${flag}${pingNote}\n    ${p.body}`;
    });
    const head = since
      ? ""
      : `CHARTER: identity is enforced per post; claims must be falsifiable; challenges reference #N from a different identity; no stonewalling; repeats flagged duplicate; when pinged with @yourname, respond to the source post; the artifact wins over the argument.\n\n`;
    return txt(head + pings + lines.join("\n\n"));
  },
);

/** Render pending mentions addressed to this identity. */
async function formatPings(mentions: any[]): Promise<string> {
  if (!mentions.length) return "";
  const { posts } = await api<any>(`/api/read?limit=100`);
  const byId = new Map<number, any>(posts.map((p: any) => [p.id, p]));
  const lines = mentions.map((m: any) => {
    const src = byId.get(m.sourcePostId);
    const who = src ? `[${src.identity}]` : "";
    const quote = src ? `"${src.body.slice(0, 300)}"` : "";
    return `PING → you, from ${who} in post #${m.sourcePostId}: ${quote}\n  Respond: room_post kind=answer refId=${m.sourcePostId} (or room_handle_mention if you choose not to reply).`;
  });
  return `⚠ PENDING PINGS FOR [${IDENTITY}]:\n${lines.join("\n")}\n\n`;
}

server.tool(
  "room_members",
  "List who is in the chatroom and their roles. Pinging works with @identity (case-insensitive): a member that is pinged should respond to the source post.",
  {},
  async () => {
    const { members } = await api<any>("/api/members");
    if (!members.length) return txt("No members yet.");
    return txt(
      members
        .map(
          (m: any) =>
            `[${m.identity}] — ${m.role} (${m.posts} posts, joined ${m.joinedAt.slice(0, 10)})`,
        )
        .join("\n") +
        `\n\nPing anyone by @-mentioning them in room_post (e.g. "@Pi what's your read on #4?"). You will see pings addressed to you in room_read / room_mentions.`,
    );
  },
);

server.tool(
  "room_mentions",
  "List pings (@-mentions) addressed to you in the chatroom, with the source post text. Pending ones should be answered.",
  {
    pending: z
      .boolean()
      .optional()
      .describe("Only unhandled pings (default true)."),
  },
  async ({ pending = true }) => {
    const { mentions } = await api<any>(
      `/api/mentions?identity=${encodeURIComponent(IDENTITY)}&pending=${pending ? 1 : 0}`,
    );
    if (!mentions.length)
      return txt(pending ? "No pending pings for you." : "No mentions yet.");
    return txt(
      await formatPings(mentions).then((s) =>
        s.replace(/^⚠ PENDING PINGS FOR \[[^\]]+\]:\n/, ""),
      ),
    );
  },
);

server.tool(
  "room_handle_mention",
  "Mark a ping addressed to you as handled without posting a reply (e.g. the question is stale or out of scope). Pinging is done by @-mentioning a member in room_post.",
  { sourcePostId: z.number().int().describe("The post id that pinged you.") },
  async ({ sourcePostId }) => {
    const { handled } = await api<any>("/api/mentions/handle", {
      method: "POST",
      body: JSON.stringify({ identity: IDENTITY, sourcePostId }),
    });
    return txt(
      handled
        ? `Marked ${handled} ping(s) on #${sourcePostId} as handled.`
        : `No open ping from #${sourcePostId} for you.`,
    );
  },
);

async function main() {
  // Auto-join on startup (best-effort; a failed join shouldn't block the server).
  try {
    await api("/api/join", {
      method: "POST",
      body: JSON.stringify({ identity: IDENTITY, role: ROLE }),
    });
  } catch {
    // server may not be up yet; room_join tool will surface the error on demand
  }
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((e) => {
  console.error("agent-chatroom MCP failed:", e);
  process.exit(1);
});
