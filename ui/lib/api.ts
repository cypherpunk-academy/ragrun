const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export function toApiUrl(path: string): string {
  if (path.startsWith("http")) {
    return path;
  }
  return `${API_BASE}${path}`;
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(toApiUrl(path), { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Request failed (${res.status}): ${path}`);
  }
  return (await res.json()) as T;
}

export async function apiPatch<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(toApiUrl(path), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Patch failed (${res.status}): ${path}`);
  }
  return (await res.json()) as T;
}

export async function apiPost<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(toApiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Post failed (${res.status}): ${path}`);
  }
  return (await res.json()) as T;
}
