"use client";

import { useTranslations } from "next-intl";
import useSWR from "swr";
import { Bar, BarChart, CartesianGrid, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { apiGet } from "@/lib/api";
import type { AdminStats } from "@/lib/types";

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <p className="text-xs uppercase tracking-wide text-zinc-400">{label}</p>
      <p className="mt-2 text-2xl font-semibold">{value}</p>
    </div>
  );
}

export default function StatsPage() {
  const t = useTranslations();
  const { data, error, isLoading } = useSWR<AdminStats>("/api/v1/admin/stats", apiGet, {
    refreshInterval: 15000,
  });

  if (isLoading) return <p className="text-zinc-400">Loading stats...</p>;
  if (error || !data) return <p className="text-red-400">Failed to load stats.</p>;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">{t("stats")}</h1>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatCard label="rag_chunks" value={data.rag_chunks_total} />
        <StatCard label="vector_chunks" value={data.vector_chunks_total} />
        <StatCard label="rag_talks" value={data.rag_talks_total} />
        <StatCard label="rag_turns" value={data.rag_turns_total} />
        <StatCard label="rag_references" value={data.rag_references_total} />
        <StatCard label="usage tokens" value={data.rag_usage_total_tokens} />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <section className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <h2 className="mb-3 font-medium">Chunks by Partition</h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.rag_chunks_by_partition}>
                <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
                <XAxis dataKey="name" stroke="#a1a1aa" />
                <YAxis stroke="#a1a1aa" />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <h2 className="mb-3 font-medium">Talk Status</h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={data.rag_talks_by_status} dataKey="count" nameKey="status" outerRadius={90} fill="#3b82f6" />
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      <section className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
        <h2 className="mb-3 font-medium">Usage by Model</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-zinc-400">
              <tr>
                <th className="pb-2 text-left">Model</th>
                <th className="pb-2 text-right">Calls</th>
                <th className="pb-2 text-right">Prompt</th>
                <th className="pb-2 text-right">Completion</th>
                <th className="pb-2 text-right">Total</th>
              </tr>
            </thead>
            <tbody>
              {data.rag_usage_by_model.map((row) => (
                <tr key={row.model} className="border-t border-zinc-800">
                  <td className="py-2">{row.model}</td>
                  <td className="py-2 text-right">{row.call_count}</td>
                  <td className="py-2 text-right">{row.prompt_tokens}</td>
                  <td className="py-2 text-right">{row.completion_tokens}</td>
                  <td className="py-2 text-right">{row.total_tokens}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
