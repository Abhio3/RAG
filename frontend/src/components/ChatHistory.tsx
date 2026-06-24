import { useEffect, useState } from "react";
import { listChats, type ChatSummary } from "../api";

type Props = {
  activeChatId: string | null;
  reloadKey: number;
  onSelect: (id: string) => void;
  onNew: () => void;
};

export default function ChatHistory({ activeChatId, reloadKey, onSelect, onNew }: Props) {
  const [chats, setChats] = useState<ChatSummary[]>([]);

  useEffect(() => {
    listChats()
      .then(setChats)
      .catch(() => setChats([]));
  }, [reloadKey]);

  return (
    <div className="mt-6 flex min-h-0 flex-1 flex-col">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-[10px] font-bold uppercase tracking-wider text-neutral-500">
          Chat history
        </h2>
        <button
          onClick={onNew}
          className="rounded-lg bg-neutral-900 border border-neutral-800 px-2.5 py-1 text-xs font-medium text-neutral-300 transition-all hover:bg-neutral-800 hover:text-neutral-100"
        >
          + New Chat
        </button>
      </div>
      <ul className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
        {chats.length === 0 && (
          <li className="py-8 text-center text-xs text-neutral-600">No conversations yet.</li>
        )}
        {chats.map((c) => (
          <li key={c.id}>
            <button
              onClick={() => onSelect(c.id)}
              className={`w-full truncate rounded-xl px-3 py-2 text-left text-xs transition-all ${
                c.id === activeChatId
                  ? "bg-neutral-900 text-neutral-100 font-medium border border-neutral-800 shadow-sm"
                  : "text-neutral-400 hover:bg-neutral-900/40 hover:text-neutral-200"
              }`}
              title={c.title}
            >
              {c.title}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
