import AsyncStorage from '@react-native-async-storage/async-storage';
import Constants from 'expo-constants';

const STORAGE_KEY = 'ragscribe.serverUrl';

const defaultFromExtra =
  (Constants.expoConfig?.extra as { defaultServerUrl?: string } | undefined)
    ?.defaultServerUrl ?? 'https://scribe.ragxxx.com';

export async function getServerUrl(): Promise<string> {
  const stored = await AsyncStorage.getItem(STORAGE_KEY);
  const raw = (stored ?? defaultFromExtra).trim();
  return raw.replace(/\/$/, '') || defaultFromExtra;
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
