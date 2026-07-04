import { useRef, useState, useEffect, type FormEvent, type ChangeEvent } from "react";
import { streamChat, streamResearch, getMessages, uploadFile, getDocuments } from "../api";

type Mode = "chat" | "research" | "deep_research";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Message = { role: "user" | "assistant"; content: string; error?: boolean };

type Props = {
  chatId: string | null;
  onChatChanged: (id: string) => void;
};

export default function ChatPanel({ chatId, onChatChanged }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [mode, setMode] = useState<Mode>("chat");
  const [steps, setSteps] = useState<string[]>([]);
  const [sessionDocs, setSessionDocs] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setSessionDocs([]);
    if (!chatId) {
      setMessages([]);
      return;
    }
    
    Promise.all([
      getMessages(chatId),
      getDocuments(chatId)
    ]).then(([msgRows, docRows]) => {
      setMessages(msgRows.map((r) => ({ role: r.role, content: r.content })));
      setSessionDocs(docRows.map((d) => d.filename));
    }).catch(() => {
      setMessages([]);
      setSessionDocs([]);
    });
  }, [chatId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, sessionDocs]);

  async function handleFileUpload(e: ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    let currentChatId = chatId;
    for (const file of Array.from(files)) {
      try {
        const res = await uploadFile(file, currentChatId);
        if (res.chat_id && res.chat_id !== currentChatId) {
          currentChatId = res.chat_id;
          onChatChanged(res.chat_id);
        }
        setSessionDocs(prev => [...prev, res.filename]);
      } catch (err) {
        setMessages((m) => [...m, { role: "assistant", content: `Upload failed: ${(err as Error).message}`, error: true }]);
      }
    }
    setUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function onSubmit(e?: FormEvent) {
    if (e) e.preventDefault();
    const question = input.trim();
    if (!question || loading || uploading) return;

    setInput("");
    setMessages((m) => [...m, { role: "user", content: question }]);
    setLoading(true);
    setSteps([]);

    let started = false;
    const appendToken = (chunk: string) => {
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
    };

    try {
      let newId: string;
      if (mode === "chat") {
        newId = await streamChat(question, chatId, appendToken);
      } else {
        newId = await streamResearch(question, chatId, mode, (ev) => {
          if (ev.type === "token") appendToken(ev.text);
          else if (ev.type === "plan") setSteps((s) => [...s, `Plan: ${ev.questions.join(" · ")}`]);
          else if (ev.type === "status" || ev.type === "step") setSteps((s) => [...s, ev.text]);
          else if (ev.type === "done") setSteps([]);
        });
      }
      if (newId && newId !== chatId) onChatChanged(newId);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: (err as Error).message, error: true },
      ]);
    } finally {
      setLoading(false);
      setSteps([]);
    }
  }

  return (
    <div className="flex h-full flex-col bg-neutral-950">
      <div className="flex-1 overflow-y-auto p-6 scrollbar-gutter-stable">
        <div className="mx-auto max-w-3xl space-y-6">
          {messages.length === 0 && (
            <div className="mt-32 flex flex-col items-center justify-center text-center">
              <div className="mb-4 rounded-2xl bg-neutral-900 p-4 ring-1 ring-neutral-800">
                <svg className="h-8 w-8 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h2 className="mb-2 text-2xl font-semibold tracking-tight text-neutral-100">How can I help you today?</h2>
              <p className="text-sm text-neutral-500 max-w-sm">
                Ask any question or upload local documents to chat and analyze them in real-time.
              </p>
            </div>
          )}
          {messages.length > 0 && sessionDocs.length > 0 && (
            <div className="flex justify-end mb-6">
              <div className="flex flex-col items-end gap-2">
                <span className="text-[10px] text-neutral-500 font-bold uppercase tracking-wider">Context Files</span>
                <div className="flex flex-wrap gap-2 justify-end">
                  {sessionDocs.map((doc, idx) => (
                    <div key={idx} title={doc} className="flex items-center gap-1.5 rounded-xl bg-neutral-900/60 px-3 py-1.5 text-xs text-neutral-300 ring-1 ring-neutral-800 cursor-default">
                      <svg className="h-3.5 w-3.5 shrink-0 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <span className="truncate max-w-[200px] font-medium">{doc}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex w-full ${m.role === "user" ? "justify-end" : "justify-start"} animate-fade-in`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-5 py-3 shadow-md ${
                  m.role === "user"
                    ? "bg-blue-600 text-neutral-100 rounded-br-none"
                    : m.error
                    ? "bg-red-950/40 border border-red-900/60 text-red-200 rounded-bl-none"
                    : "bg-neutral-900 border border-neutral-800/80 text-neutral-100 rounded-bl-none"
                }`}
              >
                {m.error ? (
                  <div className="flex items-start gap-2 text-sm">
                    <svg className="mt-0.5 h-4 w-4 shrink-0 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                    </svg>
                    <span>{m.content}</span>
                  </div>
                ) : (
                  <div
                    className={`prose prose-sm prose-invert chat-prose max-w-none ${
                      m.role === "user"
                        ? "prose-p:text-neutral-100 prose-a:text-blue-200"
                        : "prose-pre:bg-neutral-950 prose-pre:border prose-pre:border-neutral-800 prose-pre:rounded-xl prose-a:text-blue-400"
                    }`}
                  >
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {m.content}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            </div>
          ))}
          {steps.length > 0 && (
            <div className="flex justify-start animate-fade-in">
              <div className="max-w-[80%] space-y-1 rounded-2xl rounded-bl-none border border-neutral-800/80 bg-neutral-900 px-4 py-3 text-xs text-neutral-400">
                {steps.map((s, i) => (
                  <div key={i} className="flex gap-2">
                    <span className="text-blue-400">›</span>
                    <span>{s}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {loading && steps.length === 0 && messages[messages.length - 1]?.role === "user" && (
            <div className="flex justify-start animate-fade-in">
              <div className="rounded-2xl rounded-bl-none border border-neutral-800/80 bg-neutral-900 px-5 py-4 shadow-sm">
                <div className="flex items-center gap-1.5 py-1">
                  <div className="h-2.5 w-2.5 animate-bounce rounded-full bg-neutral-500 [animation-delay:-0.3s]"></div>
                  <div className="h-2.5 w-2.5 animate-bounce rounded-full bg-neutral-500 [animation-delay:-0.15s]"></div>
                  <div className="h-2.5 w-2.5 animate-bounce rounded-full bg-neutral-500"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="p-4">
        <div className="mx-auto max-w-3xl rounded-3xl bg-neutral-900 ring-1 ring-neutral-800 focus-within:ring-neutral-700">
          <form
            onSubmit={onSubmit}
            className="flex items-end gap-2 p-3"
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              className="hidden"
              onChange={handleFileUpload}
            />
            <button
              type="button"
              disabled={uploading}
              onClick={() => fileInputRef.current?.click()}
              className="rounded-full p-2 text-neutral-400 hover:bg-neutral-800 hover:text-neutral-200 disabled:opacity-50"
              title="Upload document"
            >
              {uploading ? (
                <svg className="h-6 w-6 animate-spin text-neutral-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                </svg>
              )}
            </button>

            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as Mode)}
              disabled={loading || uploading}
              title="Answer mode"
              className="shrink-0 self-stretch rounded-full bg-neutral-800 px-3 text-xs text-neutral-300 outline-none disabled:opacity-50"
            >
              <option value="chat">Chat</option>
              <option value="research">Research</option>
              <option value="deep_research">Deep research</option>
            </select>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (input.trim() && !loading && !uploading) onSubmit();
                }
              }}
              placeholder="Ask a question…"
              className="max-h-60 min-h-[44px] flex-1 resize-none bg-transparent py-2 text-[15px] text-neutral-100 outline-none placeholder:text-neutral-500"
              rows={1}
            />
            
            <button
              type="submit"
              disabled={loading || !input.trim() || uploading}
              className={`rounded-full p-2 transition-colors ${
                input.trim() && !loading && !uploading
                  ? "bg-neutral-100 text-neutral-900 hover:bg-neutral-300"
                  : "bg-neutral-800 text-neutral-500"
              }`}
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19V6m0 0l-8 8m8-8l8 8" />
              </svg>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
