import { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { getServerUrl, setServerUrl } from '@/lib/config';
import { checkHealth } from '@/lib/scribeApi';

export default function SettingsScreen() {
  const [url, setUrl] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void getServerUrl().then(setUrl);
  }, []);

  const save = useCallback(async () => {
    setBusy(true);
    setStatus('');
    try {
      const saved = await setServerUrl(url);
      setUrl(saved);
      const health = await checkHealth(saved);
      setStatus(
        health.ready
          ? `Verbunden${health.model ? ` (${health.model})` : ''}`
          : 'Server erreichbar, Modell noch nicht bereit',
      );
    } catch (e) {
      setStatus(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [url]);

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Server-Adresse</Text>
      <Text style={styles.hint}>
        Produktion: https://scribe.ragxxx.com (Cloudflare Access im Release-APK).
        Dev: ngrok auf Port 8002 oder http://192.168.x.x:8002 im LAN.
      </Text>
      <TextInput
        style={styles.input}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        placeholder="https://scribe.ragxxx.com"
        value={url}
        onChangeText={setUrl}
      />
      <Pressable
        style={[styles.button, busy && styles.buttonDisabled]}
        onPress={() => void save()}
        disabled={busy}
      >
        {busy ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Speichern & prüfen</Text>
        )}
      </Pressable>
      {status ? <Text style={styles.status}>{status}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    gap: 12,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1a1a1a',
  },
  hint: {
    fontSize: 13,
    color: '#666',
    lineHeight: 18,
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
  status: {
    fontSize: 14,
    color: '#333',
  },
});
