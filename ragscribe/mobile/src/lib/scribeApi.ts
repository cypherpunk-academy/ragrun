import Constants from 'expo-constants';
import { fetch } from 'expo/fetch';

const PROD_HOST = 'scribe.ragxxx.com';

type ExtraConfig = {
  cfAccessClientId?: string;
  cfAccessClientSecret?: string;
};

function cfAccessExtra(): ExtraConfig {
  return (
    (Constants.expoConfig?.extra as ExtraConfig | undefined) ?? {}
  );
}

function isProdHost(baseUrl: string): boolean {
  try {
    return new URL(baseUrl).hostname === PROD_HOST;
  } catch {
    return false;
  }
}

export type JobStatus = 'ok' | 'queued' | 'processing' | 'error';

export type ScribeJob = {
  id: string;
  url: string;
  source: 'youtube' | 'tiktok' | 'facebook' | 'other' | string;
  status: JobStatus | string;
  status_emoji: string;
  created_at?: string | null;
  duration_sec?: number | null;
  title?: string | null;
  text: string;
  error?: string | null;
  pct?: number;
};

function httpError(status: number, detail: string): Error {
  if (status === 403 && /cloudflare access/i.test(detail)) {
    return new Error(
      'Cloudflare Access blockiert scribe.ragxxx.com. Diese Dev-App hat kein Access-Token.',
    );
  }
  const text = /<!doctype html/i.test(detail) ? '' : detail.trim().slice(0, 180);
  return new Error(`Server HTTP ${status}${text ? `: ${text}` : ''}`);
}

function requestHeaders(
  baseUrl: string,
  extra?: Record<string, string>,
): Record<string, string> {
  const headers = { ...extra };
  if (/ngrok/i.test(baseUrl)) {
    headers['ngrok-skip-browser-warning'] = 'true';
  }
  if (isProdHost(baseUrl)) {
    const { cfAccessClientId, cfAccessClientSecret } = cfAccessExtra();
    if (cfAccessClientId && cfAccessClientSecret) {
      headers['CF-Access-Client-Id'] = cfAccessClientId;
      headers['CF-Access-Client-Secret'] = cfAccessClientSecret;
    }
  }
  return headers;
}

export async function checkHealth(baseUrl: string): Promise<{
  ready: boolean;
  model?: string;
}> {
  const res = await fetch(`${baseUrl}/health`, {
    headers: requestHeaders(baseUrl),
    signal: AbortSignal.timeout(5_000),
  });
  if (!res.ok) return { ready: false };
  const body = (await res.json()) as {
    ready?: boolean;
    model?: string;
  };
  return {
    ready: Boolean(body.ready),
    ...(body.model ? { model: body.model } : {}),
  };
}

export async function createJob(args: {
  baseUrl: string;
  url: string;
  language?: string;
  signal?: AbortSignal;
}): Promise<ScribeJob> {
  const res = await fetch(`${args.baseUrl}/api/v1/jobs`, {
    method: 'POST',
    headers: requestHeaders(args.baseUrl, {
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify({
      url: args.url,
      ...(args.language ? { language: args.language } : {}),
    }),
    signal: args.signal,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw httpError(res.status, detail);
  }
  return (await res.json()) as ScribeJob;
}

export async function listJobs(args: {
  baseUrl: string;
  source?: string | null;
  q?: string;
  limit?: number;
  signal?: AbortSignal;
}): Promise<ScribeJob[]> {
  const params = new URLSearchParams();
  if (args.source) params.set('source', args.source);
  if (args.q?.trim()) params.set('q', args.q.trim());
  if (args.limit) params.set('limit', String(args.limit));
  const qs = params.toString();
  const res = await fetch(
    `${args.baseUrl}/api/v1/jobs${qs ? `?${qs}` : ''}`,
    {
      headers: requestHeaders(args.baseUrl),
      signal: args.signal ?? AbortSignal.timeout(15_000),
    },
  );
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw httpError(res.status, detail);
  }
  const body = (await res.json()) as { jobs?: ScribeJob[] };
  return body.jobs ?? [];
}

export async function getJob(args: {
  baseUrl: string;
  id: string;
  signal?: AbortSignal;
}): Promise<ScribeJob> {
  const res = await fetch(`${args.baseUrl}/api/v1/jobs/${args.id}`, {
    headers: requestHeaders(args.baseUrl),
    signal: args.signal ?? AbortSignal.timeout(10_000),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw httpError(res.status, detail);
  }
  return (await res.json()) as ScribeJob;
}

export async function deleteJob(args: {
  baseUrl: string;
  id: string;
}): Promise<void> {
  const res = await fetch(`${args.baseUrl}/api/v1/jobs/${args.id}`, {
    method: 'DELETE',
    headers: requestHeaders(args.baseUrl),
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw httpError(res.status, detail);
  }
}

export function formatDuration(sec?: number | null): string {
  if (sec == null || !Number.isFinite(sec)) return '';
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, '0')}`;
}

export function sourceLabel(source: string): string {
  switch (source) {
    case 'youtube':
      return 'YouTube';
    case 'tiktok':
      return 'TikTok';
    case 'facebook':
      return 'Facebook';
    default:
      return 'Andere';
  }
}

export function shortUrl(url: string): string {
  try {
    const u = new URL(url);
    const path = u.pathname.length > 24 ? `${u.pathname.slice(0, 24)}…` : u.pathname;
    return `${u.hostname}${path}`;
  } catch {
    return url.length > 40 ? `${url.slice(0, 40)}…` : url;
  }
}
