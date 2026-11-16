import AsyncStorage from '@react-native-async-storage/async-storage';
import Constants from 'expo-constants';

const STORAGE_KEY = 'ragscribe.serverUrl';
const PROD_HOST = 'scribe.ragxxx.com';
/** Reserved ngrok domain for local ragscribe (see ragscribe/README.md). */
const DEV_SERVER = 'https://shana-peatier-not.ngrok-free.dev';

type ExtraConfig = {
  defaultServerUrl?: string;
  cfAccessClientId?: string;
  cfAccessClientSecret?: string;
};

function extraConfig(): ExtraConfig {
  return (Constants.expoConfig?.extra as ExtraConfig | undefined) ?? {};
}

function isProdUrl(url: string): boolean {
  try {
    return new URL(url).hostname === PROD_HOST;
  } catch {
    return false;
  }
}

function hasCfAccess(): boolean {
  const extra = extraConfig();
  return Boolean(extra.cfAccessClientId && extra.cfAccessClientSecret);
}

export async function getServerUrl(): Promise<string> {
  const stored = await AsyncStorage.getItem(STORAGE_KEY);
  const fallback = (extraConfig().defaultServerUrl ?? `https://${PROD_HOST}`).trim();
  let raw = (stored ?? fallback).trim().replace(/\/$/, '') || fallback;
  // A dev client has no Cloudflare Access token. The stored production URL
  // only ever returns the Access login page.
  if (__DEV__ && !hasCfAccess() && isProdUrl(raw)) {
    raw = DEV_SERVER;
  }
  return raw;
}

export async function setServerUrl(url: string): Promise<string> {
  const cleaned = url.trim().replace(/\/$/, '');
  if (!/^https?:\/\//i.test(cleaned)) {
    throw new Error('Server URL must start with http:// or https://');
  }
  await AsyncStorage.setItem(STORAGE_KEY, cleaned);
  return cleaned;
}

export function extractHttpUrl(text: string): string | null {
  const match = text.match(/https?:\/\/[^\s<>"']+/i);
  if (!match) return null;
  return match[0].replace(/[.,);]+$/, '');
}
