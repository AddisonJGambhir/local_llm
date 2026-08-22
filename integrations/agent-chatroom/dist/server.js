#!/usr/bin/env node
/**
 * Agent Chatroom server.
 *
 * A small stateful HTTP service that holds the room (member registry +
 * append-only post log) and streams new posts over Server-Sent Events.
 * Every client — MCP adapters, the TUI, the demo — talks to it over HTTP,
 * so the room is one shared source of truth across all agent harnesses.
 *
 * Endpoints:
 *   POST /api/join     { identity, role? }
 *   POST /api/post     { identity, body, kind?, refId? }
 *   GET  /api/read     ?since=&limit=
 *   GET  /api/members
 *   GET  /api/charter
 *   GET  /api/stats
 *   GET  /events       (SSE stream: "post" and "member" events)
 *
 * Environment:
 *   CHATROOM_HOST   Bind host (default 127.0.0.1)
 *   CHATROOM_PORT   Port (default 8790)
 *   CHATROOM_DATA   Data dir for persistence (default ~/.local/share/agent-chatroom)
 */
import * as http from "node:http";
import * as path from "node:path";
import * as os from "node:os";
import * as fs from "node:fs";
import { Room, RoomError } from "./room.js";
import { CHARTER } from "./charter.js";
const HOST = process.env.CHATROOM_HOST || "127.0.0.1";
const PORT = Number.parseInt(process.env.CHATROOM_PORT || "8790", 10);
const dataDir = path.resolve(process.env.CHATROOM_DATA ||
    path.join(os.homedir(), ".local", "share", "agent-chatroom"));
const room = new Room(dataDir);
const sseClients = new Set();
let sseSeq = 0;
function sendJson(res, status, body) {
    const text = JSON.stringify(body);
    res.writeHead(status, {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(text),
    });
    res.end(text);
}
function readBody(req) {
    return new Promise((resolve, reject) => {
        let data = "";
        req.on("data", (c) => {
            data += c;
            if (data.length > 1e6)
                reject(new RoomError("Body too large", 413));
        });
        req.on("end", () => {
            try {
                resolve(data ? JSON.parse(data) : {});
            }
            catch {
                reject(new RoomError("Invalid JSON body", 400));
            }
        });
        req.on("error", reject);
    });
}
function broadcast(event, data) {
    const frame = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
    for (const c of sseClients)
        c.res.write(frame);
}
const waiters = [];
function settleWaiters(match, payload) {
    for (const w of [...waiters]) {
        if (match(w)) {
            w.finish(payload);
        }
    }
}
room.on("post", (post) => {
    broadcast("post", post);
    // wake everyone else's watchers (never the author — it already knows);
    // mentionOnly waiters (wake daemons) ignore plain posts
    settleWaiters((w) => !w.mentionOnly && post.id > w.since && post.identity !== w.identity, { type: "post", post });
});
room.on("member", (m) => broadcast("member", m));
room.on("mention", (m) => {
    broadcast("mention", m);
    // the pinged identity wakes first, with the source post attached
    settleWaiters((w) => w.identity === m.target, {
        type: "mention",
        mention: m,
        post: room.getPost(m.sourcePostId) ?? null,
    });
});
const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || "/", `http://${req.headers.host}`);
    try {
        if (req.method === "GET" && url.pathname === "/events") {
            res.writeHead(200, {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                Connection: "keep-alive",
            });
            res.write(`event: hello\ndata: ${JSON.stringify(room.stats())}\n\n`);
            const client = { id: ++sseSeq, res };
            sseClients.add(client);
            const ping = setInterval(() => res.write(": ping\n\n"), 25000);
            req.on("close", () => {
                clearInterval(ping);
                sseClients.delete(client);
            });
            return;
        }
        if (req.method === "POST" && url.pathname === "/api/join") {
            const b = await readBody(req);
            const member = room.join(String(b.identity || ""), b.role == null ? undefined : String(b.role));
            sendJson(res, 200, member);
            return;
        }
        if (req.method === "POST" && url.pathname === "/api/post") {
            const b = await readBody(req);
            const post = room.post(String(b.identity || ""), String(b.body || ""), b.kind || "post", b.refId == null ? null : Number(b.refId));
            sendJson(res, 201, post);
            return;
        }
        if (req.method === "GET" && url.pathname === "/api/read") {
            const since = Number(url.searchParams.get("since") || 0);
            const limit = Number(url.searchParams.get("limit") || 30);
            sendJson(res, 200, { posts: room.read(since, limit) });
            return;
        }
        if (req.method === "GET" && url.pathname === "/api/members") {
            sendJson(res, 200, { members: [...room.members.values()] });
            return;
        }
        if (req.method === "GET" && url.pathname === "/api/mentions") {
            const identity = String(url.searchParams.get("identity") || "");
            const pendingOnly = url.searchParams.get("pending") === "1";
            const limit = Number(url.searchParams.get("limit") || 20);
            const mentions = pendingOnly
                ? room.pendingMentions(identity, limit)
                : room.mentionsFor(identity, limit);
            sendJson(res, 200, { mentions });
            return;
        }
        if (req.method === "POST" && url.pathname === "/api/mentions/handle") {
            const b = await readBody(req);
            const n = room.handleMention(String(b.identity || ""), Number(b.sourcePostId || 0));
            sendJson(res, 200, { handled: n });
            return;
        }
        if (req.method === "GET" && url.pathname === "/api/charter") {
            res.writeHead(200, { "Content-Type": "text/plain" });
            res.end(CHARTER + "\n");
            return;
        }
        if (req.method === "GET" && url.pathname === "/api/stats") {
            sendJson(res, 200, room.stats());
            return;
        }
        if (req.method === "GET" && url.pathname === "/api/wait") {
            const identity = String(url.searchParams.get("identity") || "");
            const timeoutMs = Math.min(Number(url.searchParams.get("timeout") || 120000), 300000);
            const since = Number(url.searchParams.get("since") || 0);
            const mentionOnly = url.searchParams.get("mentionOnly") === "1" ||
                url.searchParams.get("mentionOnly") === "true";
            const pendingMention = room.pendingMentions(identity, 1).at(-1);
            if (pendingMention) {
                sendJson(res, 200, {
                    type: "mention",
                    mention: pendingMention,
                    post: room.getPost(pendingMention.sourcePostId) ?? null,
                });
                return;
            }
            const result = await new Promise((resolve) => {
                let settled = false;
                let waiter;
                const finish = (payload) => {
                    if (settled)
                        return;
                    settled = true;
                    clearTimeout(waiter.timer);
                    const i = waiters.indexOf(waiter);
                    if (i >= 0)
                        waiters.splice(i, 1);
                    resolve(payload);
                };
                waiter = {
                    identity,
                    since,
                    mentionOnly,
                    finish,
                    timer: setTimeout(() => finish({ type: "timeout" }), timeoutMs),
                };
                waiters.push(waiter);
                res.on("close", () => {
                    if (!res.writableEnded)
                        finish({ type: "cancelled" });
                });
            });
            if (res.destroyed)
                return;
            sendJson(res, 200, result);
            return;
        }
        sendJson(res, 404, { error: `Not found: ${url.pathname}` });
    }
    catch (e) {
        if (e instanceof RoomError)
            sendJson(res, e.status, { error: e.message });
        else
            sendJson(res, 500, { error: e instanceof Error ? e.message : String(e) });
    }
});
fs.mkdirSync(dataDir, { recursive: true });
server.listen(PORT, HOST, () => {
    const s = room.stats();
    console.log(`agent-chatroom listening on http://${HOST}:${PORT} ` +
        `(data: ${dataDir}, ${s.posts} posts, ${s.members} members)`);
});
