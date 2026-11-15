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

export type ScribeEvent =
  | { type: 'fetch'; pct?: number }
  | {
      type: 'start';
      duration_sec?: number;
      model?: string;
      device?: string;
    }
  | {
      type: 'progress';
      processed_sec?: number;
      duration_sec?: number;
      pct?: number;
    }
  | {
      type: 'done';
      text: string;
      duration_sec?: number;
      language?: string;
    }
  | { type: 'error'; detail: string }
  | { type: 'heartbeat' };

export type ScribeProgress = {
  phase: 'idle' | 'fetch' | 'transcribe' | 'done' | 'error';
  pct: number;
  label: string;
  text?: string;
  error?: string;
};

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

function parseLine(line: string): ScribeEvent | null {
  const t = line.trim();
  if (!t) return null;
  try {
    return JSON.parse(t) as ScribeEvent;
  } catch {
    return null;
  }
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

/**
 * POST /api/v1/scribe and stream NDJSON progress.
 */
export async function scribeVideoUrl(args: {
  baseUrl: string;
  url: string;
  language?: string;
  onEvent: (event: ScribeEvent) => void;
  signal?: AbortSignal;
}): Promise<string> {
  const res = await fetch(`${args.baseUrl}/api/v1/scribe`, {
    method: 'POST',
    headers: requestHeaders(args.baseUrl, {
      'Content-Type': 'application/json',
    }),
    body: JSON.stringify({
      input: 'video_url',
      url: args.url,
      stream: true,
      ...(args.language ? { language: args.language } : {}),
    }),
    signal: args.signal,
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(
      `Server HTTP ${res.status}${detail ? `: ${detail.slice(0, 400)}` : ''}`,
    );
  }
  if (!res.body) {
    throw new Error('Empty response body');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let leftover = '';
  let text = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    leftover += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = leftover.indexOf('\n')) >= 0) {
      const line = leftover.slice(0, nl);
      leftover = leftover.slice(nl + 1);
      const event = parseLine(line);
      if (!event) continue;
      args.onEvent(event);
      if (event.type === 'error') {
        throw new Error(event.detail || 'Scribe error');
      }
      if (event.type === 'done') {
        text = event.text ?? '';
      }
    }
  }
  if (leftover.trim()) {
    const event = parseLine(leftover);
    if (event) {
      args.onEvent(event);
      if (event.type === 'error') {
        throw new Error(event.detail || 'Scribe error');
      }
      if (event.type === 'done') {
        text = event.text ?? '';
      }
    }
  }

  if (!text.trim()) {
    throw new Error('No transcript returned');
  }
  return text;
}

export function progressFromEvent(
  event: ScribeEvent,
  prev: ScribeProgress,
): ScribeProgress {
  if (event.type === 'fetch') {
    return {
      phase: 'fetch',
      pct: event.pct ?? prev.pct,
      label: `Audio laden… ${event.pct ?? 0}%`,
    };
  }
  if (event.type === 'start') {
    return {
      phase: 'transcribe',
      pct: 0,
      label: event.duration_sec
        ? `Transkription… 0:00 / ${formatClock(event.duration_sec)}`
        : 'Transkription…',
    };
  }
  if (event.type === 'progress') {
    const processed = event.processed_sec ?? 0;
    const duration = event.duration_sec;
    return {
      phase: 'transcribe',
      pct: event.pct ?? prev.pct,
      label: duration
        ? `Transkription… ${formatClock(processed)} / ${formatClock(duration)}`
        : `Transkription… ${event.pct ?? 0}%`,
    };
  }
  if (event.type === 'done') {
    return {
      phase: 'done',
      pct: 100,
      label: 'Fertig',
      text: event.text,
    };
  }
  if (event.type === 'error') {
    return {
      phase: 'error',
      pct: prev.pct,
      label: 'Fehler',
      error: event.detail,
    };
  }
  return prev;
}

function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${String(sec).padStart(2, '0')}`;
}
