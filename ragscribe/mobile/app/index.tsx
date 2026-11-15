import { useLocalSearchParams, useRouter } from 'expo-router';
import { useShareIntentContext } from 'expo-share-intent';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  Share,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { extractHttpUrl, getServerUrl } from '@/lib/config';
import {
  progressFromEvent,
  scribeVideoUrl,
  type ScribeProgress,
} from '@/lib/scribeApi';

const idleProgress: ScribeProgress = {
  phase: 'idle',
  pct: 0,
  label: '',
};

export default function HomeScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ url?: string | string[] }>();
  const paramUrl = Array.isArray(params.url) ? params.url[0] : params.url;
  const { hasShareIntent, shareIntent, resetShareIntent } =
    useShareIntentContext();

  const [url, setUrl] = useState('');
  const [progress, setProgress] = useState<ScribeProgress>(idleProgress);
  const [result, setResult] = useState('');
  const [error, setError] = useState('');
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const autoStarted = useRef<string | null>(null);

  const run = useCallback(async (rawUrl: string) => {
    const videoUrl = extractHttpUrl(rawUrl) ?? rawUrl.trim();
    if (!/^https?:\/\//i.test(videoUrl)) {
      setError('Bitte eine http(s)-URL einfügen');
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setRunning(true);
    setError('');
    setResult('');
    setUrl(videoUrl);
    setProgress({
      phase: 'fetch',
      pct: 0,
      label: 'Verbinde…',
    });

    try {
      const baseUrl = await getServerUrl();
      const text = await scribeVideoUrl({
        baseUrl,
        url: videoUrl,
        signal: controller.signal,
        onEvent: (event) => {
          setProgress((prev) => progressFromEvent(event, prev));
        },
      });
      setResult(text);
      setProgress({
        phase: 'done',
        pct: 100,
        label: 'Fertig',
        text,
      });
    } catch (e) {
      if (controller.signal.aborted) return;
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setProgress({
        phase: 'error',
        pct: 0,
        label: 'Fehler',
        error: msg,
      });
    } finally {
      setRunning(false);
    }
  }, []);

  useEffect(() => {
    if (!hasShareIntent) return;
    const shared = shareIntent.webUrl || shareIntent.text || '';
    const extracted = extractHttpUrl(shared);
    resetShareIntent();
    if (!extracted || autoStarted.current === extracted) return;
    autoStarted.current = extracted;
    setUrl(extracted);
    void run(extracted);
  }, [
    hasShareIntent,
    shareIntent.webUrl,
    shareIntent.text,
    resetShareIntent,
    run,
  ]);

  useEffect(() => {
    if (!paramUrl) return;
    if (autoStarted.current === paramUrl) return;
    autoStarted.current = paramUrl;
    setUrl(paramUrl);
    void run(paramUrl);
  }, [paramUrl, run]);

  const shareResult = useCallback(async () => {
    if (!result.trim()) return;
    await Share.share({ message: result });
  }, [result]);

  return (
    <ScrollView
      contentContainerStyle={styles.container}
      keyboardShouldPersistTaps="handled"
    >
      <View style={styles.headerRow}>
        <Text style={styles.subtitle}>Video-Link → Text</Text>
        <Pressable onPress={() => router.push('/settings')}>
          <Text style={styles.settingsLink}>Einstellungen</Text>
        </Pressable>
      </View>

      <TextInput
        style={styles.input}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        placeholder="https://… YouTube / TikTok"
        value={url}
        onChangeText={setUrl}
        editable={!running}
      />

      <Pressable
        style={[styles.button, running && styles.buttonDisabled]}
        onPress={() => void run(url)}
        disabled={running || !url.trim()}
      >
        {running ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Transkription starten</Text>
        )}
      </Pressable>

      {progress.phase !== 'idle' ? (
        <View style={styles.progressBlock}>
          <Text style={styles.progressLabel}>{progress.label}</Text>
          <View style={styles.barTrack}>
            <View
              style={[
                styles.barFill,
                { width: `${Math.max(0, Math.min(100, progress.pct))}%` },
              ]}
            />
          </View>
        </View>
      ) : null}

      {error ? <Text style={styles.error}>{error}</Text> : null}

      {result ? (
        <View style={styles.resultBlock}>
          <View style={styles.resultHeader}>
            <Text style={styles.resultTitle}>Ergebnis</Text>
            <Pressable onPress={() => void shareResult()}>
              <Text style={styles.shareLink}>Teilen</Text>
            </Pressable>
          </View>
          <Text selectable style={styles.resultText}>
            {result}
          </Text>
        </View>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    gap: 14,
    paddingBottom: 40,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  subtitle: {
    fontSize: 15,
    color: '#555',
  },
  settingsLink: {
    color: '#3d2b6b',
    fontWeight: '600',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 16,
    backgroundColor: '#fff',
  },
  button: {
    backgroundColor: '#3d2b6b',
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  progressBlock: {
    gap: 8,
  },
  progressLabel: {
    fontSize: 14,
    color: '#333',
  },
  barTrack: {
    height: 10,
    borderRadius: 5,
    backgroundColor: '#e6e0f0',
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    backgroundColor: '#3d2b6b',
  },
  error: {
    color: '#a11',
    fontSize: 14,
  },
  resultBlock: {
    gap: 10,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: '#e5e5e5',
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  resultTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  shareLink: {
    color: '#3d2b6b',
    fontWeight: '600',
    fontSize: 15,
  },
  resultText: {
    fontSize: 15,
    lineHeight: 22,
    color: '#1a1a1a',
  },
});
