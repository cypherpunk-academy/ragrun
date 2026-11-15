import { useRouter } from 'expo-router';
import { useShareIntentContext } from 'expo-share-intent';
import { useEffect } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';

/**
 * Landing screen for the share extension deeplink.
 * The home screen reads the shared URL from context; this route only
 * forwards there once the native module has delivered it.
 */
export default function ShareIntentScreen() {
  const router = useRouter();
  const { isReady, hasShareIntent, error } = useShareIntentContext();

  useEffect(() => {
    if (!isReady || !hasShareIntent) return;
    router.replace('/');
  }, [isReady, hasShareIntent, router]);

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#3d2b6b" />
      <Text style={styles.text}>
        {error ? error : 'Link wird übernommen…'}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
    backgroundColor: '#faf8ff',
    padding: 24,
  },
  text: {
    fontSize: 15,
    color: '#555',
    textAlign: 'center',
  },
});
