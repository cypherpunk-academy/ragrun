const IS_STAGING = process.env.APP_VARIANT === 'staging';

/** @type {import('expo/config').ExpoConfig} */
const config = {
  name: IS_STAGING ? 'ragscribe (Staging)' : 'ragscribe',
  slug: 'ragscribe',
  owner: 'lafisrap',
  scheme: 'ragscribe',
  version: '1.0.0',
  orientation: 'portrait',
  icon: './assets/icon.png',
  userInterfaceStyle: 'light',
  newArchEnabled: true,
  splash: {
    image: './assets/splash-icon.png',
    resizeMode: 'contain',
    backgroundColor: '#ffffff',
  },
  ios: {
    supportsTablet: true,
    bundleIdentifier: IS_STAGING
      ? 'berlin.cypherpunkacademy.ragscribe.staging'
      : 'berlin.cypherpunkacademy.ragscribe',
    infoPlist: {
      NSAppTransportSecurity: {
        NSAllowsLocalNetworking: true,
      },
    },
  },
  android: {
    package: IS_STAGING
      ? 'berlin.cypherpunkacademy.ragscribe.staging'
      : 'berlin.cypherpunkacademy.ragscribe',
    adaptiveIcon: {
      foregroundImage: './assets/adaptive-icon.png',
      backgroundColor: '#ffffff',
    },
    usesCleartextTraffic: true,
  },
  web: {
    favicon: './assets/favicon.png',
  },
  plugins: [
    'expo-router',
    [
      'expo-share-intent',
      {
        iosActivationRules: {
          NSExtensionActivationSupportsWebURLWithMaxCount: 1,
          NSExtensionActivationSupportsWebPageWithMaxCount: 1,
          NSExtensionActivationSupportsText: true,
        },
        androidIntentFilters: ['text/*', 'text/plain'],
      },
    ],
    [
      'expo-splash-screen',
      {
        image: './assets/splash-icon.png',
        imageWidth: 200,
        resizeMode: 'contain',
        backgroundColor: '#ffffff',
      },
    ],
  ],
  extra: {
    defaultServerUrl:
      process.env.EXPO_PUBLIC_RAGSCRIBE_URL ??
      'https://scribe.ragxxx.com',
    cfAccessClientId: process.env.EXPO_PUBLIC_CF_ACCESS_CLIENT_ID ?? '',
    cfAccessClientSecret:
      process.env.EXPO_PUBLIC_CF_ACCESS_CLIENT_SECRET ?? '',
    eas: {
      projectId: '72fdc55f-05c8-403a-9d91-2bf5c522e4c6',
    },
  },
};

module.exports = config;
