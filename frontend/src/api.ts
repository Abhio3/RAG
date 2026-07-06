export const API_BASE = "http://localhost:3000";

export type ChatSummary = { id: string; title: string; created_at: string };
export type DbMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning_content: string | null;
  created_at: string;
};

export async function uploadFile(
  file: File,
  chatId: string | null = null,
  tags: string[] = []
): Promise<{ id: string; chat_id: string; filename: string; chunks: number; tags: string[] }> {
  const form = new FormData();
  form.append("file", file);
  form.append("tags", tags.join(","));
  if (chatId) {
    form.append("chat_id", chatId);
  }
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

export async function listChats(): Promise<ChatSummary[]> {
  const res = await fetch(`${API_BASE}/chats`);
  if (!res.ok) throw new Error(`Failed to load chats (${res.status})`);
  return res.json();
}

export async function getDocuments(chatId: string): Promise<{ id: string; filename: string; created_at: string; tags: string[] }[]> {
  const res = await fetch(`${API_BASE}/documents?chat_id=${chatId}`);
  if (!res.ok) throw new Error(`Failed to load documents (${res.status})`);
  return res.json();
}

export async function getMessages(chatId: string): Promise<DbMessage[]> {
  const res = await fetch(`${API_BASE}/chats/${chatId}/messages`);
  if (!res.ok) throw new Error(`Failed to load messages (${res.status})`);
  return res.json();
}

/** Every mode now streams the same NDJSON events. `status`/`plan`/`step` only appear in
 *  research modes; `reasoning` carries the <think> trace split out by the backend. */
export type ChatEvent =
  | { type: "status"; text: string }
  | { type: "plan"; questions: string[] }
  | { type: "step"; kind: string; text: string }
  | { type: "token"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "error"; text: string }
  | { type: "done" };

/** Stream a turn (chat / research / deep_research); returns the chat id it belongs to.
 *  Pre-response failures reject; mid-stream failures arrive as an `error` event (so a
 *  socket drop surfaces on the in-flight message instead of ending silently). */
export async function streamChat(
  question: string,
  chatId: string | null,
  mode: "chat" | "research" | "deep_research",
  onEvent: (e: ChatEvent) => void
): Promise<string> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, chat_id: chatId, mode }),
  });
  if (!res.ok || !res.body) throw new Error(`Chat failed (${res.status})`);
  const newChatId = res.headers.get("X-Chat-Id") || chatId || "";
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || ""; // keep the trailing partial line for the next read
      for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
    }
    if (buf.trim()) onEvent(JSON.parse(buf));
  } catch (err) {
    onEvent({ type: "error", text: (err as Error).message || "Connection lost" });
  }
  return newChatId;
}
