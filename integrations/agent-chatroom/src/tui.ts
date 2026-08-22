#!/usr/bin/env node
/**
 * agent_chatroom — live TUI for the agent chatroom.
 *
 * Watches the room over SSE: every agent's posts appear as they happen,
 * color-coded by identity, with challenges visually linked to their target.
 * You can also post yourself from the input line.
 *
 * Keys:
 *   Enter   post your line
 *   !N      challenge post #N (e.g. "!5 this claim ignores X")
 *   @N      answer post #N  (e.g. "@3 conceding, you're right that...")
 *   /charter  print the charter in the feed
 *   Esc / Ctrl-C   quit
 *
 * Environment:
 *   CHATROOM_URL       Server base URL (default http://127.0.0.1:8790)
 *   CHATROOM_IDENTITY  Your identity in the room (default: You)
 */
import process from "node:process";
import * as readline from "node:readline";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import * as path from "node:path";

const BASE = (process.env.CHATROOM_URL || "http://127.0.0.1:8790").replace(
  /\/$/,
  "",
);
const ME = process.env.CHATROOM_IDENTITY || "You";

interface Post {
  id: number;
  identity: string;
  role: string;
  kind: "post" | "challenge" | "answer" | "verdict";
  body: string;
  refId: number | null;
  at: string;
  flags: string[];
}

const PALETTE = [
  39, 214, 213, 208, 212, 217, 205, 190, 220, 33, 202, 198, 219, 211, 209, 45,
];
function colorFor(identity: string): number {
  let h = 0;
  for (let i = 0; i < identity.length; i++)
    h = (h * 31 + identity.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

const posts: Post[] = [];
let members: { identity: string; role: string }[] = [];
const localNotes: { title: string; body: string }[] = [];
let input = "";
let cursor = 0; // insertion point within `input`
let blinkOn = true;
let quit = false;
let statusMsg: string | null = null;
let statusUntil = 0;

function upsertMember(member: { identity: string; role: string }) {
  members = members.filter(
    (m) => m.identity.toLowerCase() !== member.identity.toLowerCase(),
  );
  members.push(member);
  members.sort((a, b) => a.identity.localeCompare(b.identity));
}

function addPost(post: Post) {
  const existing = posts.findIndex((p) => p.id === post.id);
  if (existing >= 0) posts[existing] = post;
  else posts.push(post);
  posts.sort((a, b) => a.id - b.id);
  // The UI renders recent history only. Keep the backing cache bounded too,
  // so leaving a room open for days does not consume unbounded memory.
  if (posts.length > 1000) posts.splice(0, posts.length - 1000);
  upsertMember({ identity: post.identity, role: post.role });
}

function addLocalNote(title: string, body: string) {
  localNotes.push({ title, body });
  if (localNotes.length > 3) localNotes.shift();
}

const c = (code: number, s: string) => `\x1b[${code}m${s}\x1b[0m`;
const dim = (s: string) => c(2, s);

/**
 * The input line with a visible block cursor at the insertion point.
 * The window scrolls horizontally so the cursor stays in view.
 */
function inputLine(width: number): string {
  const prefixPlain = `[${ME}] `;
  const plain = prefixPlain + input;
  const cursorCol = prefixPlain.length + cursor;
  let start = 0;
  if (plain.length > width) {
    start = Math.max(0, Math.min(cursorCol + 1 - width, plain.length - width));
  }
  const vis = plain.slice(start, start + width);
  const ci = cursorCol - start;
  const at = vis[ci] ?? " ";
  const cur = blinkOn ? `\x1b[7m${at}\x1b[0m` : at;
  const left = vis.slice(0, ci);
  const right = vis.slice(ci + 1);
  if (start === 0) {
    return (
      c(colorFor(ME), prefixPlain) +
      left.slice(prefixPlain.length) +
      cur +
      right
    );
  }
  return left + cur + right;
}

function setStatus(msg: string) {
  statusMsg = msg;
  statusUntil = Date.now() + 6000;
}

function ts(p: { at: string }) {
  return p.at.slice(11, 19);
}

function wrap(text: string, width: number): string[] {
  const out: string[] = [];
  for (const raw of text.split("\n")) {
    let line = raw;
    while (line.length > width) {
      let cut = line.lastIndexOf(" ", width);
      if (cut < width * 0.5) cut = width;
      out.push(line.slice(0, cut));
      line = line.slice(cut).trimStart();
    }
    out.push(line);
  }
  return out;
}

function render(width: number) {
  const height = process.stdout.rows || 40;
  const L: string[] = [];

  const memberLine = members
    .map((m) => c(colorFor(m.identity), m.identity))
    .join("  ");
  L.push(
    c(1, "AGENT CHATROOM ") +
      dim(`${posts.length} posts · ${members.length} members · ${BASE}`) +
      "  " +
      memberLine,
  );
  L.push(dim("─".repeat(width)));

  // The bottom block (divider + status + input + help) is reserved and
  // ALWAYS on screen; the feed trims to fit the rest.
  const hasStatus = statusMsg != null && Date.now() < statusUntil;
  const reserved = 4 + (hasStatus ? 1 : 0); // divider + input + help (+status)
  const maxFeedLines = Math.max(4, height - 2 - reserved);

  const feed: string[] = [];
  for (const p of posts) {
    const ref = p.refId == null ? "" : dim(` →#${p.refId}`);
    const flag = p.flags.length ? c(33, ` [${p.flags.join(",")}]`) : "";
    const pinged = (p.body.match(/@([A-Za-z0-9][A-Za-z0-9_-]{0,31})/g) || [])
      .filter((h) =>
        members.some(
          (m) => m.identity.toLowerCase() === h.slice(1).toLowerCase(),
        ),
      )
      .map((h) => c(colorFor(h.slice(1)), h))
      .join(" ");
    const pingNote = pinged ? `  ${c(1, pinged)}` : "";
    const head =
      dim(`#${p.id}`) +
      " " +
      c(colorFor(p.identity), `[${p.identity}]`) +
      dim(` (${p.role})`) +
      ` ${c(1, p.kind)}` +
      ref +
      " " +
      dim(ts(p)) +
      flag +
      pingNote;
    feed.push(head);
    const bodyWidth = Math.max(10, width - 6);
    for (const line of wrap(p.body, bodyWidth)) feed.push("    " + line);
    feed.push("");
  }
  for (const note of localNotes) {
    feed.push(c(36, `[${note.title}]`));
    for (const line of wrap(note.body, Math.max(10, width - 4))) {
      feed.push("  " + line);
    }
    feed.push("");
  }

  if (feed.length > maxFeedLines) {
    const dropped = feed.length - maxFeedLines;
    L.push(dim(`… ${dropped} earlier feed lines trimmed …`));
    L.push(...feed.slice(-maxFeedLines));
  } else {
    L.push(...feed);
  }

  L.push(dim("─".repeat(width)));
  if (hasStatus) {
    L.push(c(31, `⚠ ${statusMsg}`));
  }
  L.push(inputLine(width));
  L.push(
    dim(
      "Enter post · ← → move · !N challenge #N · @N answer #N · /charter · Esc/Ctrl-C quit",
    ),
  );

  process.stdout.write(
    "\x1b[2J\x1b[H" + L.slice(0, height).join("\n") + "\x1b[?25l",
  );
}

/**
 * "Like pi": one command brings the room to life. If the server isn't
 * reachable, start the sibling server.js detached and wait for it to listen.
 * The room deliberately outlives the TUI — history and other agents persist.
 */
async function ensureServer() {
  const ping = async (): Promise<boolean> => {
    try {
      const res = await fetch(`${BASE}/api/stats`, {
        signal: AbortSignal.timeout(700),
      });
      return res.ok;
    } catch {
      return false;
    }
  };
  if (await ping()) return;
  const serverJs = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "server.js",
  );
  const child = spawn(process.execPath, [serverJs], {
    detached: true,
    stdio: "ignore",
  });
  child.unref();
  process.stdout.write(`starting chatroom server…\n`);
  const deadline = Date.now() + 8000;
  while (Date.now() < deadline) {
    if (await ping()) return;
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`could not start the chatroom server at ${BASE}`);
}

async function initialLoad() {
  try {
    // Join before reading membership. A new TUI used to subscribe after this
    // point and could miss its own member event until it authored a post.
    const joined = await fetch(`${BASE}/api/join`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identity: ME, role: "human" }),
    });
    if (!joined.ok) throw new Error(`join failed (${joined.status})`);
    const [res, mres] = await Promise.all([
      fetch(`${BASE}/api/read?limit=100`),
      fetch(`${BASE}/api/members`),
    ]);
    if (!res.ok || !mres.ok) throw new Error("initial room read failed");
    const data = (await res.json()) as { posts: Post[] };
    for (const post of data.posts) addPost(post);
    const mdata = (await mres.json()) as {
      members: { identity: string; role: string }[];
    };
    for (const member of mdata.members) upsertMember(member);
  } catch (e) {
    console.error(`Cannot reach chatroom at ${BASE}:`, e);
    process.exit(1);
  }
  // surface any pings already waiting for you
  try {
    const mres = await fetch(
      `${BASE}/api/mentions?identity=${encodeURIComponent(ME)}&pending=1`,
    );
    const { mentions } = (await mres.json()) as {
      mentions: { sourcePostId: number; sourceIdentity: string }[];
    };
    if (mentions.length) {
      const last = mentions[mentions.length - 1];
      setStatus(
        `${mentions.length} pending ping(s) — latest from [${last.sourceIdentity}] in #${last.sourcePostId}. Reply: @${last.sourcePostId} <text>`,
      );
    }
  } catch {
    // mentions endpoint optional at boot
  }
}

async function myPost(
  body: string,
  kind: Post["kind"] = "post",
  refId: number | null = null,
) {
  setStatus("Posting…");
  try {
    const res = await fetch(`${BASE}/api/post`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identity: ME, body, kind, refId }),
    });
    if (!res.ok) {
      const error = (await res.json().catch(() => ({}))) as { error?: string };
      setStatus(error.error || `Post failed (${res.status})`);
      return;
    }
    const post = (await res.json()) as Post;
    addPost(post);
    setStatus(`Posted #${post.id}`);
  } catch (e) {
    setStatus(`Post failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}

async function streamEvents() {
  const res = await fetch(`${BASE}/events`);
  if (!res.ok || !res.body) throw new Error(`SSE ${res.status}`);
  // The server holds this connection while we catch up, so events written
  // between the initial history read and subscription remain buffered here.
  const known = posts.at(-1)?.id || 0;
  const catchup = await fetch(`${BASE}/api/read?since=${known}&limit=1000`);
  if (catchup.ok) {
    const { posts: missed } = (await catchup.json()) as { posts: Post[] };
    for (const post of missed) addPost(post);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
      }
      if (!dataLines.length) continue;
      try {
        const data = JSON.parse(dataLines.join("\n"));
        if (event === "post") {
          addPost(data as Post);
        } else if (event === "member") {
          upsertMember(data);
        } else if (event === "mention") {
          if (data.target.toLowerCase() === ME.toLowerCase()) {
            setStatus(
              `Pinged by [${data.sourceIdentity}] in #${data.sourcePostId} — reply: @${data.sourcePostId} <text>`,
            );
          }
        }
        render(process.stdout.columns || 100);
      } catch {
        // ignore malformed frame
      }
    }
  }
}

async function followEvents() {
  while (!quit) {
    try {
      await streamEvents();
      if (!quit) setStatus("Event stream closed — reconnecting…");
    } catch (e) {
      if (!quit) {
        const detail = e instanceof Error ? e.message : String(e);
        setStatus(`Event stream unavailable: ${detail} — retrying…`);
      }
    }
    if (!quit) await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

async function main() {
  await ensureServer();
  await initialLoad();
  if (!process.stdout.isTTY) {
    // non-TTY: just follow the stream, one block per post
    process.stdout.write(
      `following ${BASE} (identity: ${ME}), Ctrl-C to stop\n`,
    );
    const res = await fetch(`${BASE}/events`);
    const reader = res.body!.getReader();
    const dec = new TextDecoder();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = dec.decode(value, { stream: true });
      for (const line of chunk.split("\n")) {
        if (line.startsWith("data: ")) {
          try {
            const d = JSON.parse(line.slice(6));
            if (d.id)
              console.log(
                `\n#${d.id} [${d.identity}] ${d.kind}${d.refId ? " →#" + d.refId : ""}\n${d.body}`,
              );
          } catch {
            // not a post frame (hello/member/ping) — nothing to print
          }
        }
      }
    }
    return;
  }

  process.stdout.write("\x1b[?25l");
  render(process.stdout.columns || 100);
  process.stdin.setRawMode(true);
  readline.emitKeypressEvents(process.stdin);

  const onKey = (str: string, key: { name?: string; ctrl?: boolean } = {}) => {
    if (quit) return;
    if (key.name === "escape" || (key.ctrl && key.name === "c")) {
      quit = true;
      return;
    }
    if (key.name === "return") {
      const line = input.trim();
      input = "";
      cursor = 0;
      if (!line) return;
      if (line === "/charter") {
        fetch(`${BASE}/api/charter`)
          .then((r) => r.text())
          .then((t) => {
            addLocalNote("CHARTER", t);
            setStatus("Charter shown locally");
            render(process.stdout.columns || 100);
          })
          .catch((e) => {
            const detail = e instanceof Error ? e.message : String(e);
            setStatus(`Could not load charter: ${detail}`);
          });
      } else {
        let m = line.match(/^!(\d+)\s*(.*)$/);
        if (m) {
          void myPost(
            m[2] || `challenge to #${m[1]}`,
            "challenge",
            Number(m[1]),
          );
        } else {
          m = line.match(/^@(\d+)\s*(.*)$/);
          if (m)
            void myPost(m[2] || `answer to #${m[1]}`, "answer", Number(m[1]));
          else void myPost(line);
        }
      }
    } else if (key.name === "backspace") {
      if (cursor > 0) {
        input = input.slice(0, cursor - 1) + input.slice(cursor);
        cursor--;
      }
    } else if (key.name === "delete") {
      if (cursor < input.length) {
        input = input.slice(0, cursor) + input.slice(cursor + 1);
      }
    } else if (key.name === "left") {
      if (cursor > 0) cursor--;
    } else if (key.name === "right") {
      if (cursor < input.length) cursor++;
    } else if (key.name === "home") {
      cursor = 0;
    } else if (key.name === "end") {
      cursor = input.length;
    } else if (str && !key.ctrl) {
      // printable text, including multi-char bracketed paste
      const clean = str.replace(/\r/g, "");
      if (clean) {
        input = input.slice(0, cursor) + clean + input.slice(cursor);
        cursor += clean.length;
      }
    }
    render(process.stdout.columns || 100);
  };
  process.stdin.on("keypress", onKey);
  process.on("SIGWINCH", () => render(process.stdout.columns || 100));
  // blink the block cursor; also retires expired status messages
  const ticker = setInterval(() => {
    if (statusMsg && Date.now() >= statusUntil) statusMsg = null;
    blinkOn = !blinkOn;
    render(process.stdout.columns || 100);
  }, 530);

  void followEvents();

  await new Promise<void>((resolve) => {
    const check = setInterval(() => {
      if (quit) {
        clearInterval(check);
        resolve();
      }
    }, 100);
  });
  clearInterval(ticker);
  process.stdin.removeListener("keypress", onKey);
  process.stdout.write("\x1b[2J\x1b[H\x1b[?25h");
  process.exit(0);
}

main().catch((e) => {
  console.error("agent_chatroom failed:", e);
  process.exit(1);
});
