#!/usr/bin/env node
/**
 * agent-chatroom-wake — wakes a parked Pi session when it's pinged in the room.
 *
 * Long-polls /api/wait (mentionOnly) for WAKE_IDENTITY. When a ping arrives it
 * spawns a detached `pi -c -p "<prompt>"` in WAKE_CWD: pi resumes the newest
 * session for that project — full context, MCP chatroom tools included — runs
 * one turn (checks room_mentions, answers via room_post), and exits. The room
 * reply lands the same way as if the human had typed into that terminal.
 *
 * Guarantees:
 *   - single-flight: one wake at a time. The running wake reads ALL pending
 *     pings via room_mentions, so suppressing spawns while one is active
 *     loses nothing.
 *   - no double-wake: mention ids are recorded in a state file; a daemon
 *     restart won't re-wake pings it already triggered (or a live session
 *     already answered — pending-only catch-up at startup).
 *
 * Environment:
 *   CHATROOM_URL    default http://127.0.0.1:8790
 *   WAKE_IDENTITY   default Pi
 *   WAKE_CWD        project dir whose newest session is woken (default $HOME)
 *   PI_BIN          default "pi"
 *   WAKE_TIMEOUT_MS per-wake kill cap, default 900000 (15 min)
 */
import * as os from "node:os";
import * as path from "node:path";
import * as fs from "node:fs";
import { spawn } from "node:child_process";
const BASE = (process.env.CHATROOM_URL || "http://127.0.0.1:8790").replace(/\/$/, "");
const IDENTITY = process.env.WAKE_IDENTITY || "Pi";
const WAKE_CWD = process.env.WAKE_CWD || os.homedir();
const PI_BIN = process.env.PI_BIN || "pi";
const WAKE_TIMEOUT_MS = Number(process.env.WAKE_TIMEOUT_MS || 900000);
const WAIT_TIMEOUT_MS = 300000; // server cap
const STATE_DIR = path.join(os.homedir(), ".local", "share", "agent-chatroom-wake");
const STATE_FILE = path.join(STATE_DIR, "notified.json");
const LOG_FILE = path.join(STATE_DIR, "wake.log");
let notified = new Set();
try {
    fs.mkdirSync(STATE_DIR, { recursive: true });
    const raw = JSON.parse(fs.readFileSync(STATE_FILE, "utf8"));
    notified = new Set(raw.notified.slice(-500));
}
catch {
    // first run
}
function saveState() {
    try {
        fs.mkdirSync(STATE_DIR, { recursive: true });
        fs.writeFileSync(STATE_FILE, JSON.stringify({ notified: [...notified] }, null, 2));
    }
    catch (e) {
        log(`state save failed: ${String(e)}`);
    }
}
function log(msg) {
    const line = `${new Date().toISOString()} ${msg}\n`;
    try {
        fs.mkdirSync(STATE_DIR, { recursive: true });
        fs.appendFileSync(LOG_FILE, line);
    }
    catch {
        // logging is best-effort
    }
    process.stderr.write(line);
}
async function getJson(url) {
    const res = await fetch(url, { signal: AbortSignal.timeout(320000) });
    if (!res.ok)
        throw new Error(`${url} -> ${res.status}`);
    return (await res.json());
}
async function pendingMentions() {
    const data = await getJson(`${BASE}/api/mentions?identity=${encodeURIComponent(IDENTITY)}&pending=1`);
    return data.mentions;
}
let wakeRunning = false;
let wakeKillTimer = null;
let rewakeNeeded = false;
function wakeFor(mentions) {
    if (wakeRunning) {
        // The running wake may already be past its room_mentions read, so a
        // ping that lands now could be missed — re-check when it finishes.
        rewakeNeeded = true;
        log(`wake in progress — ${mentions.length} new ping(s) queued for re-wake when it finishes`);
        return;
    }
    wakeRunning = true;
    const first = mentions[0];
    const prompt = [
        `⚠ CHATROOM PING — you were pinged in the agent chatroom by [${first.sourceIdentity}] in post #${first.sourcePostId}:`,
        `"${first.body ?? ""}"`,
        ...(mentions.length > 1
            ? [
                `(+${mentions.length - 1} more pending ping — room_mentions lists them all)`,
            ]
            : []),
        "",
        "Process it now: call room_mentions (agent-chatroom MCP) to see every pending ping with its source post, then answer each with room_post kind=answer refId=<source post id>, using your full context. If a ping is out of scope, room_handle_mention it with a reason. Do not start any other work — process the pings, then stop.",
    ].join("\n");
    const logStream = fs.openSync(LOG_FILE, "a");
    log(`waking pi (cwd=${WAKE_CWD}, pings: ${mentions.map((m) => m.id).join(",")})`);
    const child = spawn(PI_BIN, ["-c", "-p", prompt], {
        cwd: WAKE_CWD,
        detached: true,
        stdio: ["ignore", logStream, logStream],
    });
    child.unref();
    wakeKillTimer = setTimeout(() => {
        log(`wake timeout (${WAKE_TIMEOUT_MS}ms) — killing pi pid ${child.pid}`);
        try {
            process.kill(child.pid, "SIGKILL");
        }
        catch {
            // already gone
        }
    }, WAKE_TIMEOUT_MS);
    const done = () => {
        wakeRunning = false;
        if (wakeKillTimer)
            clearTimeout(wakeKillTimer);
        wakeKillTimer = null;
        log(`wake finished (pid ${child.pid})`);
        if (rewakeNeeded) {
            rewakeNeeded = false;
            pendingMentions()
                .then((pending) => {
                if (pending.length && !wakeRunning) {
                    log(`re-wake: ${pending.length} ping(s) still pending after wake`);
                    wakeFor(pending);
                }
            })
                .catch((e) => log(`re-wake check failed: ${String(e)}`));
        }
    };
    child.on("exit", (code) => {
        log(`wake exit code=${code}`);
        done();
    });
    child.on("error", (e) => {
        log(`wake spawn error: ${String(e)}`);
        done();
    });
}
async function catchUp() {
    try {
        const pending = await pendingMentions();
        const fresh = pending.filter((m) => !notified.has(m.id));
        if (fresh.length) {
            log(`startup catch-up: ${fresh.length} un-notified pending ping(s)`);
            for (const m of fresh)
                notified.add(m.id);
            saveState();
            wakeFor(fresh);
        }
        else if (pending.length) {
            log(`startup: ${pending.length} pending ping(s) already notified/answered path — no wake`);
        }
    }
    catch (e) {
        log(`startup catch-up failed: ${String(e)}`);
    }
}
async function waitOnce() {
    try {
        const result = await getJson(`${BASE}/api/wait?identity=${encodeURIComponent(IDENTITY)}&mentionOnly=1&since=0&timeout=${WAIT_TIMEOUT_MS}`);
        if (result.type === "mention" && result.mention) {
            const m = result.mention;
            if (!notified.has(m.id)) {
                notified.add(m.id);
                if (notified.size > 500) {
                    notified = new Set([...notified].slice(-400));
                }
                saveState();
                wakeFor([{ ...m, body: result.post?.body ?? "" }]);
            }
            else {
                log(`mention ${m.id} already notified — skipping`);
            }
        }
        // timeout -> next pass
    }
    catch (e) {
        log(`wait loop error: ${String(e)} — retrying in 2s`);
        await new Promise((r) => setTimeout(r, 2000));
    }
    return waitOnce();
}
async function main() {
    log(`wake daemon up — identity=${IDENTITY} cwd=${WAKE_CWD} pi=${PI_BIN} room=${BASE}`);
    await catchUp();
    await waitOnce();
}
main().catch((e) => {
    log(`fatal: ${String(e)}`);
    process.exit(1);
});
