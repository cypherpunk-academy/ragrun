import { Stack } from 'expo-router';
import { ShareIntentProvider } from 'expo-share-intent';
import { StatusBar } from 'expo-status-bar';

export default function RootLayout() {
  return (
    <ShareIntentProvider
      options={{
        // Keep the payload until the home screen copies the URL.
        // The default clears it when the app passes through background
        // on the way in from the share sheet.
        resetOnBackground: false,
      }}
    >
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: '#faf8ff' },
          headerTintColor: '#1a1a1a',
          contentStyle: { backgroundColor: '#faf8ff' },
        }}
      >
        <Stack.Screen name="index" options={{ title: 'ragscribe' }} />
        <Stack.Screen name="settings" options={{ title: 'Einstellungen' }} />
        <Stack.Screen
          name="shareintent"
          options={{ title: 'Teilen', headerShown: false }}
        />
      </Stack>
    </ShareIntentProvider>
  );
}
