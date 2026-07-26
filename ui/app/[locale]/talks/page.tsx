"use client";

import { useMemo, useState } from "react";
import { useSession } from "next-auth/react";
import { useTranslations } from "next-intl";
import useSWR from "swr";

import { apiGet, apiPatch } from "@/lib/api";
import type { AdminTalksResponse, TalkDetails } from "@/lib/types";

const STATUS_COLORS: Record<string, string> = {
  draft: "bg-zinc-700 text-zinc-100",
  published: "bg-emerald-700 text-emerald-50",
  personal: "bg-blue-700 text-blue-50",
  bug: "bg-rose-700 text-rose-50",
};

function StatusBadge({ value }: { value: string }) {
  const cls = STATUS_COLORS[value] || "bg-zinc-700 text-zinc-100";
  return <span className={`rounded-full px-2 py-1 text-xs ${cls}`}>{value}</span>;
}

export default function TalksPage() {
  const t = useTranslations();
  const { data: session } = useSession();
  const [search, setSearch] = useState("");
  const [statuses, setStatuses] = useState<string[]>([]);
  const [selectedTalkId, setSelectedTalkId] = useState<string>("");
  const [editingTurnId, setEditingTurnId] = useState<string>("");
  const [editingText, setEditingText] = useState("");

  const statusParam = useMemo(() => statuses.join(","), [statuses]);
  const talksUrl = `/api/v1/admin/talks?q=${encodeURIComponent(search)}&statuses=${encodeURIComponent(statusParam)}&limit=200`;
  const { data: talksData, mutate: mutateTalks } = useSWR<AdminTalksResponse>(talksUrl, apiGet);
  const { data: talkDetails, mutate: mutateDetails } = useSWR<TalkDetails>(
    selectedTalkId ? `/api/v1/admin/talks/${selectedTalkId}` : null,
    apiGet
  );

  async function saveTurn(turnId: string) {
    await apiPatch(`/api/v1/admin/turns/${turnId}`, { user_message: editingText });
    setEditingTurnId("");
    await mutateDetails();
  }

  async function setStatus(status: string) {
    if (!selectedTalkId) return;
    await apiPatch(`/api/v1/admin/talks/${selectedTalkId}`, { publishing_status: status });
    await Promise.all([mutateTalks(), mutateDetails()]);
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">{t("talks")}</h1>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-4">
        <input
          className="rounded border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm lg:col-span-2"
          placeholder={t("searchTalks")}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="rounded border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm"
          value=""
          onChange={(e) => {
            const val = e.target.value;
            if (!val) return;
            setStatuses((prev) => (prev.includes(val) ? prev : [...prev, val]));
          }}
        >
          <option value="">{t("status")} +</option>
          <option value="draft">draft</option>
          <option value="published">published</option>
          <option value="personal">personal</option>
          <option value="bug">bug</option>
        </select>
        <select
          className="rounded border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm"
          value={selectedTalkId}
          onChange={(e) => setSelectedTalkId(e.target.value)}
        >
          <option value="">Talk wählen...</option>
          {talksData?.items.map((talk) => (
            <option key={talk.talk_id} value={talk.talk_id}>
              {talk.title} ({talk.publishing_status})
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-wrap gap-2">
        {statuses.map((entry) => (
          <button
            key={entry}
            className="rounded-full border border-zinc-700 px-2 py-1 text-xs"
            onClick={() => setStatuses((prev) => prev.filter((v) => v !== entry))}
          >
            {entry} x
          </button>
        ))}
      </div>

      {talkDetails && (
        <section className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold">{talkDetails.title}</h2>
              <p className="text-sm text-zinc-400">{talkDetails.slug}</p>
            </div>
            <StatusBadge value={talkDetails.publishing_status} />
          </div>

          <div className="mb-4 flex gap-2">
            {["draft", "published", "personal", "bug"].map((st) => (
              <button
                key={st}
                className="rounded border border-zinc-700 px-2 py-1 text-xs hover:bg-zinc-800 disabled:opacity-40"
                disabled={!session?.user}
                onClick={() => setStatus(st)}
              >
                set {st}
              </button>
            ))}
          </div>

          <div className="space-y-4">
            {talkDetails.turns.map((turn) => (
              <article key={turn.turn_id} className="rounded border border-zinc-800 bg-zinc-950 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <p className="text-xs text-zinc-400">Turn #{turn.turn_index}</p>
                  <p className="text-xs text-zinc-500">{turn.updated_at || ""}</p>
                </div>

                <div className="mb-3 rounded border border-zinc-700 p-2">
                  <p className="mb-1 text-xs font-semibold text-zinc-400">User</p>
                  {editingTurnId === turn.turn_id ? (
                    <div className="space-y-2">
                      <textarea
                        className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-sm"
                        rows={4}
                        value={editingText}
                        onChange={(e) => setEditingText(e.target.value)}
                      />
                      <div className="flex gap-2">
                        <button
                          className="rounded bg-emerald-700 px-3 py-1 text-xs"
                          onClick={() => saveTurn(turn.turn_id)}
                        >
                          {t("save")}
                        </button>
                        <button
                          className="rounded border border-zinc-700 px-3 py-1 text-xs"
                          onClick={() => setEditingTurnId("")}
                        >
                          {t("cancel")}
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <p className="whitespace-pre-wrap text-sm">{turn.user_message}</p>
                      <button
                        className="mt-2 rounded border border-zinc-700 px-2 py-1 text-xs disabled:opacity-40"
                        disabled={!session?.user}
                        onClick={() => {
                          setEditingTurnId(turn.turn_id);
                          setEditingText(turn.user_message);
                        }}
                      >
                        edit
                      </button>
                    </>
                  )}
                </div>

                <div className="mb-3 rounded border border-zinc-700 p-2">
                  <p className="mb-1 text-xs font-semibold text-zinc-400">Assistant</p>
                  <p className="whitespace-pre-wrap text-sm text-zinc-200">{turn.assistant_message}</p>
                </div>

                <details className="mb-2 rounded border border-zinc-700 p-2">
                  <summary className="cursor-pointer text-sm font-medium">{t("references")} ({turn.references.length})</summary>
                  <div className="mt-2 space-y-2 text-sm">
                    {turn.references.map((ref) => (
                      <div key={ref.ref_id} className="rounded border border-zinc-800 p-2">
                        <p>#{ref.ref_index} {ref.source_title || "untitled"}</p>
                        <p className="text-zinc-400">{ref.segment_title || "-"} · {ref.chunk_id || "-"}</p>
                        <p className="text-zinc-400">relevance: {ref.relevance ?? "-"}</p>
                      </div>
                    ))}
                  </div>
                </details>

                <details className="rounded border border-zinc-700 p-2">
                  <summary className="cursor-pointer text-sm font-medium">{t("usage")} ({turn.usage.length})</summary>
                  <div className="mt-2 space-y-1 text-sm">
                    {turn.usage.map((usage) => (
                      <p key={usage.id}>
                        {usage.model || "unknown"} · {usage.total_tokens || 0} tokens ({usage.prompt_tokens || 0}/
                        {usage.completion_tokens || 0})
                      </p>
                    ))}
                  </div>
                </details>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
