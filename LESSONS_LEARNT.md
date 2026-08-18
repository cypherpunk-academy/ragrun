# Lessons Learnt

Betriebsnotizen aus Debugging und Dev-Workflows über ragrun / ragapp / ragkeep.

---

## Philo-App: Anmeldung im iOS-Simulator

**Problem:** Beim Aufforderung „Philo im iPad-Simulator anmelden“ versucht der Agent, den Login manuell in der Simulator-GUI durchzuführen. Das ist nicht möglich und führt zu Verwirrung — obwohl es einen etablierten Dev-Workflow gibt.

**Lösung:** Dev-Login über Supabase Service Role + Deep Link (nicht manuelles Klicken im Simulator).

```bash
cd ragapp
yarn auth:login lavisrap@gmail.com --ios
```

Das Script (`ragapp/scripts/dev-login.mjs`):

1. Sucht den User in Supabase (`auth.users`)
2. Erzeugt eine Session (Magic-Link-Flow serverseitig)
3. Verbindet den Dev Client mit Metro (`:8081`)
4. Öffnet `ragapp://auth-callback?access_token=…&refresh_token=…` im booted iOS Simulator via `xcrun simctl openurl`

**Voraussetzungen:**

- iPad/iOS Simulator läuft (`yarn ios` oder `scripts/run-ios-simulator.sh`)
- Metro läuft (`yarn start` → Port 8081)
- `ragapp/.env.local` mit `EXPO_PUBLIC_SUPABASE_URL`, `EXPO_PUBLIC_SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`

**Varianten:**

```bash
yarn auth:login <email> --android          # Android-Emulator
yarn auth:login <email> --print            # nur Deep Link ausgeben
```

**Merksatz für Agents:** „Simulator anmelden“ = `yarn auth:login <email> --ios` in `ragapp` ausführen.
