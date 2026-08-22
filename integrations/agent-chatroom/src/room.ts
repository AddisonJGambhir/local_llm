import { EventEmitter } from "node:events";
import * as fs from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";

export type PostKind = "post" | "challenge" | "answer" | "verdict";

export interface Member {
  identity: string;
  role: string;
  joinedAt: string;
  posts: number;
}

export interface Post {
  id: number;
  identity: string;
  role: string;
  kind: PostKind;
  body: string;
  refId: number | null;
  at: string;
  flags: string[];
}

/** A ping: someone @-mentioned a registered member. */
export interface Mention {
  id: number;
  target: string; // canonical member identity
  sourcePostId: number; // post containing the @mention
  sourceIdentity: string; // who did the pinging
  at: string;
  handled: boolean; // resolved once the target replies to the source post
}

export class RoomError extends Error {
  constructor(
    message: string,
    public status: number = 400,
  ) {
    super(message);
  }
}

const IDENTITY_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$/;
const MAX_BODY = 8000;
const KINDS: PostKind[] = ["post", "challenge", "answer", "verdict"];
const MENTION_RE = /@([A-Za-z0-9][A-Za-z0-9_-]{0,31})/g;

/** Case-insensitive identity lookup against the member registry. */
function matchMember(
  members: Map<string, unknown>,
  handle: string,
): string | null {
  const lower = handle.toLowerCase();
  for (const identity of members.keys()) {
    if (identity.toLowerCase() === lower) return identity;
  }
  return null;
}

function hashBody(body: string): string {
  return crypto
    .createHash("sha256")
    .update(body.trim().toLowerCase().replace(/\s+/g, " "))
    .digest("hex");
}

/**
 * The chatroom: a single append-only log of posts plus a member registry.
 * All protocol rules live here so every client (MCP adapters, TUI, HTTP)
 * gets the same enforcement. State persists as JSONL so a restart resumes
 * the same room.
 */
export class Room extends EventEmitter {
  readonly members = new Map<string, Member>();
  private posts: Post[] = [];
  private nextId = 1;
  private bodyHashes = new Map<string, number>(); // identity -> hash -> last post id
  private mentions: Mention[] = [];
  private nextMentionId = 1;

  constructor(private dataDir: string) {
    super();
    fs.mkdirSync(dataDir, { recursive: true });
    this.load();
  }

  private postsFile() {
    return path.join(this.dataDir, "posts.jsonl");
  }
  private membersFile() {
    return path.join(this.dataDir, "members.json");
  }
  private mentionsFile() {
    return path.join(this.dataDir, "mentions.jsonl");
  }

  private load() {
    if (fs.existsSync(this.postsFile())) {
      for (const line of fs
        .readFileSync(this.postsFile(), "utf8")
        .split("\n")) {
        if (!line.trim()) continue;
        try {
          const p = JSON.parse(line) as Post;
          this.posts.push(p);
          this.nextId = Math.max(this.nextId, p.id + 1);
          const key = `${p.identity}:${hashBody(p.body)}`;
          this.bodyHashes.set(key, p.id);
        } catch {
          // skip corrupt line
        }
      }
    }
    if (fs.existsSync(this.membersFile())) {
      try {
        const raw = JSON.parse(
          fs.readFileSync(this.membersFile(), "utf8"),
        ) as Member[];
        for (const m of raw) this.members.set(m.identity, m);
      } catch {
        // start fresh
      }
    }
    if (fs.existsSync(this.mentionsFile())) {
      // mentions.jsonl is an append log of state updates — the last
      // record per id wins (creation, then handled=true flips)
      const byId = new Map<number, Mention>();
      for (const line of fs
        .readFileSync(this.mentionsFile(), "utf8")
        .split("\n")) {
        if (!line.trim()) continue;
        try {
          const m = JSON.parse(line) as Mention;
          byId.set(m.id, m);
          this.nextMentionId = Math.max(this.nextMentionId, m.id + 1);
        } catch {
          // skip corrupt line
        }
      }
      this.mentions = [...byId.values()];
    }
  }

  private persistMembers() {
    fs.writeFileSync(
      this.membersFile(),
      JSON.stringify([...this.members.values()], null, 2),
    );
  }

  private persistPost(p: Post) {
    fs.appendFileSync(this.postsFile(), JSON.stringify(p) + "\n");
  }

  private persistMention(m: Mention) {
    fs.appendFileSync(this.mentionsFile(), JSON.stringify(m) + "\n");
  }

  /**
   * Extract @-mentions from a post body, resolved against the member
   * registry (case-insensitive). A mention is created per unique target,
   * never for the author pinging themselves, and re-pinging an open
   * mention on the same post doesn't stack duplicates.
   */
  private routeMentions(post: Post): Mention[] {
    const created: Mention[] = [];
    const seen = new Set<string>();
    let m: RegExpExecArray | null;
    MENTION_RE.lastIndex = 0;
    while ((m = MENTION_RE.exec(post.body)) !== null) {
      const target = matchMember(this.members, m[1]);
      if (!target || target === post.identity || seen.has(target)) continue;
      seen.add(target);
      const alreadyOpen = this.mentions.some(
        (x) => !x.handled && x.target === target && x.sourcePostId === post.id,
      );
      if (alreadyOpen) continue;
      const mention: Mention = {
        id: this.nextMentionId++,
        target,
        sourcePostId: post.id,
        sourceIdentity: post.identity,
        at: new Date().toISOString(),
        handled: false,
      };
      this.mentions.push(mention);
      this.persistMention(mention);
      created.push(mention);
      this.emit("mention", mention);
    }
    return created;
  }

  /** A target replying to its mention's source post resolves the ping. */
  private resolveMentionsForReply(identity: string, refId: number | null) {
    if (refId == null) return;
    for (const m of this.mentions) {
      if (!m.handled && m.target === identity && m.sourcePostId === refId) {
        m.handled = true;
        this.persistMention(m);
      }
    }
  }

  /** Explicitly mark a ping handled (used by responders that post plain replies). */
  handleMention(identity: string, sourcePostId: number): number {
    const canonicalIdentity = matchMember(this.members, identity) || identity;
    let n = 0;
    for (const m of this.mentions) {
      if (
        !m.handled &&
        m.target === canonicalIdentity &&
        m.sourcePostId === sourcePostId
      ) {
        m.handled = true;
        this.persistMention(m);
        n++;
      }
    }
    return n;
  }

  pendingMentions(identity: string, limit = 20): Mention[] {
    const canonicalIdentity = matchMember(this.members, identity) || identity;
    return this.mentions
      .filter((m) => m.target === canonicalIdentity && !m.handled)
      .slice(-limit);
  }

  mentionsFor(identity: string, limit = 20): Mention[] {
    const canonicalIdentity = matchMember(this.members, identity) || identity;
    return this.mentions
      .filter((m) => m.target === canonicalIdentity)
      .slice(-limit);
  }

  join(identity: string, role?: string): Member {
    if (!IDENTITY_RE.test(identity)) {
      throw new RoomError(
        `Invalid identity "${identity}". Use 2-32 chars: letters, digits, _ or -, starting alphanumeric.`,
      );
    }
    const now = new Date().toISOString();
    const canonicalIdentity = matchMember(this.members, identity);
    const existing = canonicalIdentity
      ? this.members.get(canonicalIdentity)
      : undefined;
    if (existing) {
      if (role && role !== existing.role) {
        existing.role = role;
        this.persistMembers();
        this.emit("member", existing);
      }
    } else {
      const member: Member = {
        identity,
        role: role || "general",
        joinedAt: now,
        posts: 0,
      };
      this.members.set(identity, member);
      this.persistMembers();
      this.emit("member", member);
    }
    return existing || this.members.get(identity)!;
  }

  post(
    identity: string,
    body: string,
    kind: PostKind = "post",
    refId: number | null = null,
  ): Post {
    const canonicalIdentity = matchMember(this.members, identity);
    const member = canonicalIdentity
      ? this.members.get(canonicalIdentity)
      : undefined;
    if (!member) {
      throw new RoomError(
        `Unknown identity "${identity}". Call room_join first (charter rule 1).`,
        403,
      );
    }
    if (!KINDS.includes(kind)) {
      throw new RoomError(
        `Invalid kind "${kind}". Use one of: ${KINDS.join(", ")}.`,
      );
    }
    const text = (body ?? "").trim();
    if (!text) throw new RoomError("Body is empty.");
    if (text.length > MAX_BODY)
      throw new RoomError(`Body exceeds ${MAX_BODY} chars.`);

    const flags: string[] = [];
    if (kind === "challenge" || kind === "answer") {
      if (refId == null) {
        throw new RoomError(
          `${kind} must reference a post id (refId), per charter rule 3.`,
        );
      }
      const target = this.posts.find((p) => p.id === refId);
      if (!target) throw new RoomError(`Post #${refId} does not exist.`);
      if (target.identity === member.identity) {
        throw new RoomError(
          `You cannot ${kind} your own post #${refId} (charter rule 3 — another identity must).`,
        );
      }
    }

    // duplicate detection (charter rule 5): same identity, same normalized body
    const author = member.identity;
    const key = `${author}:${hashBody(text)}`;
    if (this.bodyHashes.has(key)) flags.push("duplicate");

    const post: Post = {
      id: this.nextId++,
      identity: author,
      role: member.role,
      kind,
      body: text,
      refId: kind === "challenge" || kind === "answer" ? refId : null,
      at: new Date().toISOString(),
      flags,
    };
    this.posts.push(post);
    this.bodyHashes.set(key, post.id);
    member.posts += 1;
    this.persistMembers();
    this.persistPost(post);
    // resolve + route mentions BEFORE broadcasting, so waiters and
    // subscribers see a post whose pings are already settled
    this.resolveMentionsForReply(author, post.refId);
    this.routeMentions(post);
    this.emit("post", post);
    return post;
  }

  read(since = 0, limit = 30): Post[] {
    const after = this.posts.filter((p) => p.id > since);
    return after.slice(-Math.max(1, limit));
  }

  getPost(id: number): Post | undefined {
    return this.posts.find((p) => p.id === id);
  }

  latest(): Post | undefined {
    return this.posts[this.posts.length - 1];
  }

  stats() {
    return {
      posts: this.posts.length,
      members: this.members.size,
      mentionsOpen: this.mentions.filter((m) => !m.handled).length,
      identities: [...this.members.keys()],
    };
  }
}
