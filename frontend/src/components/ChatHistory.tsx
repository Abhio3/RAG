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
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
          Chat history
        </h2>
        <button
          onClick={onNew}
          className="rounded bg-neutral-800 px-2 py-0.5 text-xs text-neutral-300 hover:bg-neutral-700"
        >
          + New
        </button>
      </div>
      <ul className="min-h-0 flex-1 space-y-1 overflow-y-auto">
        {chats.length === 0 && (
          <li className="text-xs text-neutral-600">No conversations yet.</li>
        )}
        {chats.map((c) => (
          <li key={c.id}>
            <button
              onClick={() => onSelect(c.id)}
              className={`w-full truncate rounded px-2 py-1.5 text-left text-xs transition-colors ${
                c.id === activeChatId
                  ? "bg-blue-600/20 text-blue-300"
                  : "text-neutral-300 hover:bg-neutral-900"
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
