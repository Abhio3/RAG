import { useState } from "react";
import UploadPanel from "./components/UploadPanel";
import ChatPanel from "./components/ChatPanel";
import ChatHistory from "./components/ChatHistory";

export default function App() {
  const [chatId, setChatId] = useState<string | null>(null);
  // Bumped whenever the message set changes, to refresh the history list.
  const [historyKey, setHistoryKey] = useState(0);

  return (
    <div className="flex h-full bg-neutral-950 text-neutral-100">
      <aside className="flex w-80 shrink-0 flex-col border-r border-neutral-800 p-5">
        <h1 className="mb-1 text-lg font-semibold">RAG App</h1>
        <p className="mb-6 text-xs text-neutral-500">Local · Ollama + Qdrant + Supabase</p>
        <UploadPanel />
        <ChatHistory
          activeChatId={chatId}
          reloadKey={historyKey}
          onSelect={setChatId}
          onNew={() => setChatId(null)}
        />
      </aside>
      <main className="flex min-w-0 flex-1 flex-col">
        <ChatPanel
          chatId={chatId}
          onChatChanged={(id) => {
            setChatId(id);
            setHistoryKey((k) => k + 1);
          }}
        />
      </main>
    </div>
  );
}
