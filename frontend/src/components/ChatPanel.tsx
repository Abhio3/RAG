import { useRef, useState, useEffect, type FormEvent } from "react";
import { streamChat, getMessages } from "../api";

type Message = { role: "user" | "assistant"; content: string };

type Props = {
  chatId: string | null;
  onChatChanged: (id: string) => void;
};

export default function ChatPanel({ chatId, onChatChanged }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Load an existing thread when one is selected; clear for a new chat.
  useEffect(() => {
    if (!chatId) {
      setMessages([]);
      return;
    }
    getMessages(chatId)
      .then((rows) => setMessages(rows.map((r) => ({ role: r.role, content: r.content }))))
      .catch(() => setMessages([]));
  }, [chatId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    setMessages((m) => [...m, { role: "user", content: question }]);
    setLoading(true);

    let started = false;
    try {
      const newId = await streamChat(question, chatId, (chunk) => {
        setMessages((m) => {
          if (!started) {
            started = true;
            return [...m, { role: "assistant", content: chunk }];
          }
          const copy = [...m];
          copy[copy.length - 1] = {
            role: "assistant",
            content: copy[copy.length - 1].content + chunk,
          };
          return copy;
        });
      });
      if (newId && newId !== chatId) onChatChanged(newId);
      else onChatChanged(newId || chatId || "");
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `⚠️ ${(err as Error).message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="flex-1 space-y-4 overflow-y-auto p-6">
        {messages.length === 0 && (
          <p className="mt-20 text-center text-sm text-neutral-600">
            Upload a document, then ask a question about it.
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[75%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm ${
                m.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-neutral-800 text-neutral-100"
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}
        {loading && messages[messages.length - 1]?.role === "user" && (
          <div className="flex justify-start">
            <div className="animate-pulse rounded-2xl bg-neutral-800 px-4 py-2 text-sm text-neutral-400">
              Analyzing document…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={onSubmit}
        className="flex gap-2 border-t border-neutral-800 p-4"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question…"
          className="flex-1 rounded-md bg-neutral-900 px-4 py-2 text-sm outline-none ring-1 ring-neutral-800 focus:ring-blue-600"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="rounded-md bg-blue-600 px-5 py-2 text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </>
  );
}
