export const API_BASE = "http://localhost:8001";

export type ChatSummary = { id: string; title: string; created_at: string };
export type DbMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
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

/** Streams the answer; returns the chat id this turn belongs to. */
export async function streamChat(
  question: string,
  chatId: string | null,
  onChunk: (text: string) => void
): Promise<string> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, chat_id: chatId }),
  });
  if (!res.ok || !res.body) {
    throw new Error(`Chat failed (${res.status})`);
  }
  const newChatId = res.headers.get("X-Chat-Id") || chatId || "";
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    onChunk(decoder.decode(value, { stream: true }));
  }
  return newChatId;
}
