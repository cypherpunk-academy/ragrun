"use client";

import { useEffect, useMemo, useState } from "react";
import { signIn, signOut, useSession } from "next-auth/react";
import { useLocale, useTranslations } from "next-intl";
import useSWR from "swr";

import { Link, usePathname, useRouter } from "@/i18n/routing";
import { apiGet, apiPost, toApiUrl } from "@/lib/api";
import type { CollectionStat } from "@/lib/types";

type ChatMessage = { role: "user" | "assistant"; content: string };

export function Sidebar() {
  const t = useTranslations();
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();
  const { data: session } = useSession();
  const [collapsed, setCollapsed] = useState(false);
  const [assistant, setAssistant] = useState("philo-von-freisinn");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sending, setSending] = useState(false);

  const { data: collections } = useSWR<CollectionStat[]>(
    "/api/v1/admin/collections",
    apiGet,
    { refreshInterval: 20000 }
  );

  const assistantOptions = useMemo(() => {
    if (!collections?.length) return [];
    return collections.map((entry) => entry.name).sort();
  }, [collections]);

  async function handleSendMessage() {
    const prompt = input.trim();
    if (!prompt || sending) return;
    setInput("");
    setSending(true);
    setMessages((prev) => [...prev, { role: "user", content: prompt }]);

    try {
      const response = await fetch(toApiUrl(`/api/v1/agent/${assistant}/chat/stream`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: prompt, thread_id: `dashboard-${Date.now()}` }),
      });
      const body = await response.text();
      const assistantText = body
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.replace(/^data:\s*/, ""))
        .join("\n")
        .trim();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: assistantText || "No response stream payload." },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${(error as Error).message}`,
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  useEffect(() => {
    async function syncUser() {
      if (!session?.user?.githubId || !session?.user?.githubLogin) return;
      await apiPost("/api/v1/admin/users/upsert", {
        github_id: session.user.githubId,
        github_login: session.user.githubLogin,
        email: session.user.email || null,
        name: session.user.name || null,
        avatar_url: session.user.image || null,
      });
    }
    void syncUser();
  }, [session?.user?.githubId, session?.user?.githubLogin, session?.user?.email, session?.user?.image, session?.user?.name]);

  return (
    <aside
      className={`border-r border-zinc-800 bg-zinc-900/95 p-3 transition-all ${
        collapsed ? "w-16" : "w-96"
      }`}
    >
      <div className="mb-4 flex items-center justify-between">
        {!collapsed && <h1 className="text-sm font-semibold">{t("appTitle")}</h1>}
        <button
          onClick={() => setCollapsed((v) => !v)}
          className="rounded border border-zinc-700 px-2 py-1 text-xs hover:bg-zinc-800"
        >
          {collapsed ? ">" : "<"}
        </button>
      </div>

      {!collapsed && (
        <>
          <div className="mb-4">
            <label className="mb-1 block text-xs text-zinc-400">{t("assistantPlaceholder")}</label>
            <select
              className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-2 text-sm"
              value={assistant}
              onChange={(e) => setAssistant(e.target.value)}
            >
              <option value="philo-von-freisinn">philo-von-freisinn</option>
              {assistantOptions.map((entry) => (
                <option key={entry} value={entry}>
                  {entry}
                </option>
              ))}
            </select>
          </div>

          <nav className="mb-4 space-y-2">
            <Link
              href="/stats"
              className={`block rounded px-2 py-2 text-sm ${
                pathname.endsWith("/stats") ? "bg-zinc-800" : "hover:bg-zinc-800/60"
              }`}
            >
              {t("stats")}
            </Link>
            <Link
              href="/talks"
              className={`block rounded px-2 py-2 text-sm ${
                pathname.endsWith("/talks") ? "bg-zinc-800" : "hover:bg-zinc-800/60"
              }`}
            >
              {t("talks")}
            </Link>
          </nav>

          <div className="mb-4 rounded border border-zinc-800 p-2">
            <h2 className="mb-2 text-xs uppercase tracking-wide text-zinc-400">{t("agentArena")}</h2>
            <div className="mb-2 h-40 overflow-y-auto rounded bg-zinc-950 p-2 text-xs">
              {messages.length === 0 && <p className="text-zinc-500">No messages yet.</p>}
              {messages.map((msg, idx) => (
                <p key={idx} className={msg.role === "user" ? "text-zinc-200" : "text-emerald-300"}>
                  <span className="font-semibold">{msg.role === "user" ? "You" : "AI"}:</span>{" "}
                  {msg.content}
                </p>
              ))}
            </div>
            <textarea
              className="mb-2 w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs"
              rows={2}
              placeholder={t("messagePlaceholder")}
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button
              className="w-full rounded bg-emerald-700 px-2 py-1 text-xs font-medium hover:bg-emerald-600 disabled:opacity-50"
              onClick={handleSendMessage}
              disabled={sending}
            >
              {sending ? "..." : t("send")}
            </button>
          </div>

          <div className="mt-auto space-y-2 border-t border-zinc-800 pt-3">
            <div className="flex gap-2">
              <button
                className={`rounded border px-2 py-1 text-xs ${locale === "de" ? "border-emerald-600" : "border-zinc-700"}`}
                onClick={() => router.replace(pathname, { locale: "de" })}
              >
                DE
              </button>
              <button
                className={`rounded border px-2 py-1 text-xs ${locale === "en" ? "border-emerald-600" : "border-zinc-700"}`}
                onClick={() => router.replace(pathname, { locale: "en" })}
              >
                EN
              </button>
            </div>

            {!session?.user ? (
              <button
                className="w-full rounded border border-zinc-700 px-3 py-2 text-sm hover:bg-zinc-800"
                onClick={() => signIn("github")}
              >
                {t("login")}
              </button>
            ) : (
              <div className="rounded border border-zinc-800 p-2 text-xs">
                <p className="font-semibold">{session.user.name || session.user.githubLogin}</p>
                <p className="mb-2 text-zinc-400">@{session.user.githubLogin}</p>
                <button
                  className="w-full rounded border border-zinc-700 px-2 py-1 hover:bg-zinc-800"
                  onClick={() => signOut()}
                >
                  {t("logout")}
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
