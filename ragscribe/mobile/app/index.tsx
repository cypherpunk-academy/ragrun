import { useFocusEffect, useLocalSearchParams, useRouter } from 'expo-router';
import { useShareIntentContext } from 'expo-share-intent';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Modal,
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
  createJob,
  deleteJob,
  formatDuration,
  getJob,
  listJobs,
  shortUrl,
  type ScribeJob,
} from '@/lib/scribeApi';

type SourceFilter = '' | 'youtube' | 'tiktok';

export default function HomeScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ url?: string | string[] }>();
  const paramUrl = Array.isArray(params.url) ? params.url[0] : params.url;
  const { hasShareIntent, shareIntent, resetShareIntent } =
    useShareIntentContext();

  const [url, setUrl] = useState('');
  const [jobs, setJobs] = useState<ScribeJob[]>([]);
  const [q, setQ] = useState('');
  const [source, setSource] = useState<SourceFilter>('');
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<ScribeJob | null>(null);
  const autoStarted = useRef<string | null>(null);
  const qRef = useRef(q);
  const sourceRef = useRef(source);
  qRef.current = q;
  sourceRef.current = source;

  const refresh = useCallback(async () => {
    try {
      const baseUrl = await getServerUrl();
      const items = await listJobs({
        baseUrl,
        source: sourceRef.current || null,
        q: qRef.current,
        limit: 100,
      });
      setJobs(items);
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const submit = useCallback(
    async (rawUrl: string) => {
      const videoUrl = extractHttpUrl(rawUrl) ?? rawUrl.trim();
      if (!/^https?:\/\//i.test(videoUrl)) {
        setError('Bitte eine http(s)-URL einfügen');
        return;
      }
      setSubmitting(true);
      setError('');
      setUrl(videoUrl);
      try {
        const baseUrl = await getServerUrl();
        const job = await createJob({ baseUrl, url: videoUrl });
        setJobs((prev) => [job, ...prev.filter((j) => j.id !== job.id)]);
        setSelected(job);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setSubmitting(false);
      }
    },
    [],
  );

  useFocusEffect(
    useCallback(() => {
      setLoading(true);
      void refresh().finally(() => setLoading(false));
    }, [refresh]),
  );

  useEffect(() => {
    void refresh();
  }, [refresh, source]);

  useEffect(() => {
    const t = setTimeout(() => {
      void refresh();
    }, 300);
    return () => clearTimeout(t);
  }, [q, refresh]);

  useEffect(() => {
    const id = setInterval(() => {
      const busy = jobs.some(
        (j) => j.status === 'queued' || j.status === 'processing',
      );
      if (busy || selected?.status === 'queued' || selected?.status === 'processing') {
        void refresh().then(async () => {
          if (!selected) return;
          try {
            const baseUrl = await getServerUrl();
            const fresh = await getJob({ baseUrl, id: selected.id });
            setSelected(fresh);
          } catch {
            /* ignore poll errors */
          }
        });
      }
    }, 2000);
    return () => clearInterval(id);
  }, [jobs, selected, refresh]);

  useEffect(() => {
    if (!hasShareIntent) return;
    const shared = shareIntent.webUrl || shareIntent.text || '';
    const extracted = extractHttpUrl(shared);
    resetShareIntent();
    if (!extracted || autoStarted.current === extracted) return;
    autoStarted.current = extracted;
    setUrl(extracted);
    void submit(extracted);
  }, [
    hasShareIntent,
    shareIntent.webUrl,
    shareIntent.text,
    resetShareIntent,
    submit,
  ]);

  useEffect(() => {
    if (!paramUrl) return;
    if (autoStarted.current === paramUrl) return;
    autoStarted.current = paramUrl;
    setUrl(paramUrl);
    void submit(paramUrl);
  }, [paramUrl, submit]);

  const onDelete = useCallback(
    (job: ScribeJob) => {
      Alert.alert('Eintrag löschen?', shortUrl(job.url), [
        { text: 'Abbrechen', style: 'cancel' },
        {
          text: 'Löschen',
          style: 'destructive',
          onPress: () => {
            void (async () => {
              try {
                const baseUrl = await getServerUrl();
                await deleteJob({ baseUrl, id: job.id });
                setSelected(null);
                await refresh();
              } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
              }
            })();
          },
        },
      ]);
    },
    [refresh],
  );

  const onShare = useCallback(async (job: ScribeJob) => {
    const parts = [
      job.url,
      job.source ? `Quelle: ${job.source}` : '',
      job.text?.trim() || '',
    ].filter(Boolean);
    await Share.share({ message: parts.join('\n\n') });
  }, []);

  const renderItem = ({ item }: { item: ScribeJob }) => {
    const dur = formatDuration(item.duration_sec);
    return (
      <Pressable style={styles.row} onPress={() => setSelected(item)}>
        <Text style={styles.emoji}>{item.status_emoji || '❓'}</Text>
        <View style={styles.rowBody}>
          <Text style={styles.rowTitle} numberOfLines={1}>
            {item.source} · {shortUrl(item.url)}
          </Text>
          <Text style={styles.rowMeta} numberOfLines={1}>
            {item.created_at?.replace('T', ' ').replace(/\+00:00$/, ' UTC') ?? ''}
            {dur ? ` · ${dur}` : ''}
            {item.status === 'processing' && item.pct != null
              ? ` · ${item.pct}%`
              : ''}
          </Text>
        </View>
      </Pressable>
    );
  };

  return (
    <View style={styles.screen}>
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
        editable={!submitting}
      />

      <Pressable
        style={[styles.button, submitting && styles.buttonDisabled]}
        onPress={() => void submit(url)}
        disabled={submitting || !url.trim()}
      >
        {submitting ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Transkription starten</Text>
        )}
      </Pressable>

      <TextInput
        style={styles.input}
        placeholder="Filter Text / URL…"
        value={q}
        onChangeText={setQ}
      />

      <View style={styles.chips}>
        {(
          [
            ['', 'Alle'],
            ['youtube', 'YouTube'],
            ['tiktok', 'TikTok'],
          ] as const
        ).map(([value, label]) => (
          <Pressable
            key={label}
            style={[styles.chip, source === value && styles.chipActive]}
            onPress={() => setSource(value)}
          >
            <Text
              style={[
                styles.chipText,
                source === value && styles.chipTextActive,
              ]}
            >
              {label}
            </Text>
          </Pressable>
        ))}
      </View>

      {error ? <Text style={styles.error}>{error}</Text> : null}

      {loading && jobs.length === 0 ? (
        <ActivityIndicator style={{ marginTop: 20 }} />
      ) : (
        <FlatList
          data={jobs}
          keyExtractor={(item) => item.id}
          renderItem={renderItem}
          contentContainerStyle={styles.list}
          ListEmptyComponent={
            <Text style={styles.empty}>Noch keine Einträge</Text>
          }
        />
      )}

      <Modal
        visible={selected != null}
        animationType="slide"
        onRequestClose={() => setSelected(null)}
      >
        <View style={styles.modal}>
          <View style={styles.modalHeader}>
            <Pressable onPress={() => setSelected(null)}>
              <Text style={styles.settingsLink}>Schließen</Text>
            </Pressable>
            <Text style={styles.modalTitle}>
              {selected?.status_emoji} {selected?.status}
            </Text>
            <View style={{ width: 70 }} />
          </View>
          <ScrollView contentContainerStyle={styles.modalBody}>
            <Text style={styles.detailLabel}>URL</Text>
            <Text selectable style={styles.detailText}>
              {selected?.url}
            </Text>
            <Text style={styles.detailLabel}>Quelle</Text>
            <Text style={styles.detailText}>{selected?.source}</Text>
            {selected?.duration_sec != null ? (
              <>
                <Text style={styles.detailLabel}>Länge</Text>
                <Text style={styles.detailText}>
                  {formatDuration(selected.duration_sec)}
                </Text>
              </>
            ) : null}
            {selected?.status === 'processing' ||
            selected?.status === 'queued' ? (
              <Text style={styles.detailText}>
                Fortschritt: {selected.pct ?? 0}%
              </Text>
            ) : null}
            {selected?.error ? (
              <>
                <Text style={styles.detailLabel}>Fehler</Text>
                <Text style={styles.error}>{selected.error}</Text>
              </>
            ) : null}
            {selected?.text ? (
              <>
                <Text style={styles.detailLabel}>Text</Text>
                <Text selectable style={styles.resultText}>
                  {selected.text}
                </Text>
              </>
            ) : null}
          </ScrollView>
          <View style={styles.modalActions}>
            <Pressable
              style={styles.secondaryBtn}
              onPress={() => selected && void onShare(selected)}
              disabled={!selected}
            >
              <Text style={styles.secondaryBtnText}>Teilen</Text>
            </Pressable>
            <Pressable
              style={[styles.secondaryBtn, styles.dangerBtn]}
              onPress={() => selected && onDelete(selected)}
            >
              <Text style={[styles.secondaryBtnText, styles.dangerText]}>
                Löschen
              </Text>
            </Pressable>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    padding: 16,
    gap: 10,
    backgroundColor: '#f7f5fb',
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
  chips: {
    flexDirection: 'row',
    gap: 8,
  },
  chip: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#fff',
  },
  chipActive: {
    backgroundColor: '#3d2b6b',
    borderColor: '#3d2b6b',
  },
  chipText: {
    color: '#333',
    fontSize: 13,
  },
  chipTextActive: {
    color: '#fff',
    fontWeight: '600',
  },
  list: {
    paddingBottom: 40,
    gap: 8,
  },
  row: {
    flexDirection: 'row',
    gap: 10,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: '#e5e5e5',
    alignItems: 'center',
    marginBottom: 8,
  },
  emoji: {
    fontSize: 22,
    width: 32,
    textAlign: 'center',
  },
  rowBody: {
    flex: 1,
    gap: 2,
  },
  rowTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1a1a1a',
  },
  rowMeta: {
    fontSize: 12,
    color: '#666',
  },
  empty: {
    textAlign: 'center',
    color: '#888',
    marginTop: 24,
  },
  error: {
    color: '#a11',
    fontSize: 14,
  },
  modal: {
    flex: 1,
    paddingTop: 56,
    backgroundColor: '#f7f5fb',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingBottom: 12,
  },
  modalTitle: {
    fontSize: 16,
    fontWeight: '700',
  },
  modalBody: {
    padding: 16,
    gap: 8,
    paddingBottom: 40,
  },
  detailLabel: {
    marginTop: 8,
    fontSize: 12,
    color: '#666',
    fontWeight: '600',
    textTransform: 'uppercase',
  },
  detailText: {
    fontSize: 15,
    color: '#222',
  },
  resultText: {
    fontSize: 15,
    lineHeight: 22,
    color: '#1a1a1a',
  },
  modalActions: {
    flexDirection: 'row',
    gap: 12,
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#e5e5e5',
    backgroundColor: '#fff',
  },
  secondaryBtn: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 14,
    borderRadius: 10,
    backgroundColor: '#eee8f7',
  },
  secondaryBtnText: {
    color: '#3d2b6b',
    fontWeight: '700',
  },
  dangerBtn: {
    backgroundColor: '#f8e8e8',
  },
  dangerText: {
    color: '#a11',
  },
});
