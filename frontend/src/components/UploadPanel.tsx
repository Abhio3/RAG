import { useRef, useState, useEffect, type DragEvent } from "react";
import { uploadFile } from "../api";

const STORAGE_KEY = "rag-indexed-files";

export default function UploadPanel() {
  const [files, setFiles] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("");
  const [tags, setTags] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) setFiles(JSON.parse(saved));
  }, []);

  function persist(next: string[]) {
    setFiles(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  }

  async function handleFiles(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return;
    setBusy(true);
    for (const file of Array.from(fileList)) {
      setStatus(`Uploading ${file.name}…`);
      try {
        const tagList = tags.split(",").map((t) => t.trim()).filter(Boolean);
        const res = await uploadFile(file, tagList);
        setStatus(`Indexed ${res.filename} (${res.chunks} chunks)`);
        persist([...new Set([...files, res.filename])]);
      } catch (e) {
        setStatus((e as Error).message);
      }
    }
    setBusy(false);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  }

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-lg border-2 border-dashed p-6 text-center text-sm transition-colors ${
          dragOver
            ? "border-blue-500 bg-blue-500/10"
            : "border-neutral-700 hover:border-neutral-600"
        }`}
      >
        <p className="text-neutral-300">Drag & drop a PDF or TXT</p>
        <p className="mt-1 text-xs text-neutral-500">or click to browse</p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      <input
        value={tags}
        onChange={(e) => setTags(e.target.value)}
        placeholder="tags (comma separated)"
        className="mt-3 w-full rounded-md bg-neutral-900 px-3 py-2 text-xs outline-none ring-1 ring-neutral-800 focus:ring-blue-600"
      />

      <button
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        className="mt-2 w-full rounded-md bg-blue-600 py-2 text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
      >
        {busy ? "Uploading…" : "Upload"}
      </button>

      {status && <p className="mt-3 text-xs text-neutral-400">{status}</p>}

      {files.length > 0 && (
        <div className="mt-6">
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-neutral-500">
            Indexed files
          </h2>
          <ul className="space-y-1">
            {files.map((f) => (
              <li
                key={f}
                className="truncate rounded bg-neutral-900 px-2 py-1 text-xs text-neutral-300"
              >
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
