# ragscribe mobile

Expo app (Android & iOS) for [ragscribe](../README.md). Accept shared video links, show transcription progress, share the finished text.

## Identifiers (same pattern as ragapp)

| | Production | Staging (`APP_VARIANT=staging`) |
|--|------------|----------------------------------|
| Name | ragscribe | ragscribe (Staging) |
| Slug / scheme | `ragscribe` | `ragscribe` |
| iOS / Android ID | `berlin.cypherpunkacademy.ragscribe` | `berlin.cypherpunkacademy.ragscribe.staging` |
| Expo owner | `lafisrap` | |
| Default server | `https://scribe.ragxxx.com` | (unused; staging profile not shipped) |

Create an EAS project once: `cd mobile && eas init` (fills `extra.eas.projectId`).

## Requirements

- Node 20+
- Expo Dev Client (Share Intent does not work in Expo Go)
- Production server on Tuxai, or local `../start.sh` for LAN / ngrok

## Setup

```bash
cd mobile
yarn install   # or npm install
npx expo prebuild
yarn ios       # or yarn android
```

In **Einstellungen**, the default is `https://scribe.ragxxx.com`. For local work, use `http://<LAN-IP>:8002` or temporarily `ngrok http 8002 --domain=shana-peatier-not.ngrok-free.dev` (same domain as Philo Dev — only one at a time).

## Production APK (internal, no Play Store)

Set EAS sensitive env for the production profile to the **same** Cloudflare Access
service token as ragrun (`RAGRUN_CF_ACCESS_CLIENT_ID` / `RAGRUN_CF_ACCESS_CLIENT_SECRET`):

- `EXPO_PUBLIC_CF_ACCESS_CLIENT_ID`
- `EXPO_PUBLIC_CF_ACCESS_CLIENT_SECRET`

`eas.json` already sets `EXPO_PUBLIC_RAGSCRIBE_URL=https://scribe.ragxxx.com` and builds an APK (`distribution: internal`).

```bash
cd mobile
eas build -p android --profile production
```

Install the APK from the EAS link. If a debug build with the same package id is already on the device, uninstall it first — AsyncStorage keeps the old server URL and would override the production default.

Access headers are sent only when the configured host is `scribe.ragxxx.com`. ngrok and LAN URLs stay without those headers.

Server ops: [`deploy/tuxai/ragscribe/README.md`](../../deploy/tuxai/ragscribe/README.md).

## Flow

1. Share a YouTube/TikTok link → choose ragscribe (or paste the URL).
2. Progress bar: audio fetch, then Whisper transcription.
3. **Teilen** sends the transcript as plain text.
