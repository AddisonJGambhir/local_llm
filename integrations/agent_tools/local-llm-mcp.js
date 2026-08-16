#!/usr/bin/env node
/**
 * Local LLM MCP Server
 *
 * A small, read-only bridge from coding harnesses to the local llama.cpp
 * OpenAI-compatible API. It intentionally returns advice/context only; the
 * calling harness retains responsibility for tool use and file changes.
 *
 * Environment variables:
 *   LOCAL_LLM_BASE_URL  Base API URL (default: http://127.0.0.1:1234/v1)
 *   LOCAL_LLM_MODEL     Model ID (default: local)
 *   LOCAL_LLM_TIMEOUT   Request timeout in ms (default: 300000)
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const BASE_URL = (process.env.LOCAL_LLM_BASE_URL || "http://127.0.0.1:1234/v1").replace(/\/$/, "");
const MODEL = process.env.LOCAL_LLM_MODEL || "local";
const TIMEOUT_MS = parsePositiveInt(process.env.LOCAL_LLM_TIMEOUT, 300000);
const DEFAULT_MAX_TOKENS = 2048;
const MAX_TOKENS = 4096;

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(value || "", 10);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function errorMessage(error) {
  if (error?.name === "AbortError") {
    return `Local LLM request timed out after ${TIMEOUT_MS} ms.`;
  }
  return error instanceof Error ? error.message : String(error);
}

function responseText(result, includeReasoning) {
  const response = result.content || "(empty response)";
  if (includeReasoning && result.reasoning) {
    return `[Reasoning]\n${result.reasoning}\n\n[Response]\n${response}`;
  }
  return response;
}

async function requestLocalLLM(messages, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        messages,
        stream: false,
        max_tokens: options.maxTokens ?? DEFAULT_MAX_TOKENS,
        temperature: options.temperature ?? 0.2,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Local LLM API returned ${response.status}${detail ? `: ${detail}` : ""}`);
    }

    const data = await response.json();
    const message = data.choices?.[0]?.message;
    if (!message) throw new Error("Local LLM API returned no completion.");

    return {
      content: message.content || "",
      reasoning: message.reasoning_content || "",
      usage: data.usage,
    };
  } finally {
    clearTimeout(timeout);
  }
}

const maxTokensSchema = z.number().int().min(64).max(MAX_TOKENS).optional()
  .describe(`Maximum completion tokens (default ${DEFAULT_MAX_TOKENS}, maximum ${MAX_TOKENS}).`);
const temperatureSchema = z.number().min(0).max(1).optional()
  .describe("Sampling temperature (default 0.2).");

const server = new McpServer({ name: "local-llm", version: "2.0.0" });

server.tool(
  "local_llm_chat",
  "Ask the local Qwen model for read-only analysis, summaries, brainstorming, or a second opinion. The local model cannot access files or run commands; include the relevant context in the prompt. Prefer this for bounded subproblems and verify its claims before acting.",
  {
    prompt: z.string().min(1).max(200000).describe("The task and any context to send to the local model."),
    system_prompt: z.string().max(20000).optional().describe("Optional behavior instructions for the local model."),
    max_tokens: maxTokensSchema,
    temperature: temperatureSchema,
    include_reasoning: z.boolean().optional().describe("Include the model's private reasoning in the result (default false)."),
  },
  async ({ prompt, system_prompt, max_tokens, temperature, include_reasoning }) => {
    const messages = [
      ...(system_prompt ? [{ role: "system", content: system_prompt }] : []),
      { role: "user", content: prompt },
    ];

    try {
      const result = await requestLocalLLM(messages, { maxTokens: max_tokens, temperature });
      return { content: [{ type: "text", text: responseText(result, include_reasoning) }] };
    } catch (error) {
      return { content: [{ type: "text", text: `[Local LLM error] ${errorMessage(error)}` }], isError: true };
    }
  },
);

server.tool(
  "local_llm_context",
  "Continue a supplied conversation with the local Qwen model. Use this when prior turns materially improve a summary or analysis. It is read-only and does not retain state between calls.",
  {
    messages: z.array(z.object({
      role: z.enum(["system", "user", "assistant"]),
      content: z.string().min(1).max(200000),
    })).min(1).max(32).describe("Conversation messages, ending with the task for the local model."),
    max_tokens: maxTokensSchema,
    temperature: temperatureSchema,
    include_reasoning: z.boolean().optional().describe("Include the model's private reasoning in the result (default false)."),
  },
  async ({ messages, max_tokens, temperature, include_reasoning }) => {
    try {
      const result = await requestLocalLLM(messages, { maxTokens: max_tokens, temperature });
      return { content: [{ type: "text", text: responseText(result, include_reasoning) }] };
    } catch (error) {
      return { content: [{ type: "text", text: `[Local LLM error] ${errorMessage(error)}` }], isError: true };
    }
  },
);

server.tool(
  "local_llm_status",
  "Check whether the configured local llama.cpp server is reachable and identify its configured model. This does not generate text.",
  {},
  async () => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), Math.min(TIMEOUT_MS, 10000));
    try {
      const response = await fetch(`${BASE_URL}/models`, { signal: controller.signal });
      if (!response.ok) throw new Error(`Local LLM API returned ${response.status}.`);
      const data = await response.json();
      const models = (data.data || data.models || []).map((model) => model.id || model.name).filter(Boolean);
      return { content: [{ type: "text", text: `Local LLM is reachable at ${BASE_URL} (configured model: ${MODEL}; available: ${models.join(", ") || "unknown"}).` }] };
    } catch (error) {
      return { content: [{ type: "text", text: `[Local LLM error] ${errorMessage(error)}` }], isError: true };
    } finally {
      clearTimeout(timeout);
    }
  },
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`local-llm MCP connected to ${BASE_URL} (model: ${MODEL})`);
}

main().catch((error) => {
  console.error("Failed to start local-llm MCP:", errorMessage(error));
  process.exit(1);
});
