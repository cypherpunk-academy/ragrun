import { serve } from "https://deno.land/std@0.177.0/http/server.ts";

const RESEND_API_KEY = Deno.env.get("RESEND_API_KEY") ?? "";
const FROM_EMAIL = Deno.env.get("INVITATION_FROM_EMAIL") ?? "noreply@michaelschmidt.berlin";

serve(async (req) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  // Verify service-role key (set as Authorization header by the backend)
  const authHeader = req.headers.get("Authorization") ?? "";
  if (!authHeader.startsWith("Bearer ")) {
    return new Response("Unauthorized", { status: 401 });
  }

  const { invitee_email, code, google_play_url } = await req.json();

  if (!invitee_email || !code) {
    return new Response(
      JSON.stringify({ error: "invitee_email and code required" }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  }

  const playStoreSection = google_play_url
    ? `\nInstallieren Sie die App über den Google Play Store:\n${google_play_url}\n`
    : "";

  const textBody = `Guten Tag,

Sie wurden eingeladen, Philo von Freisinn zu nutzen.
${playStoreSection}
Ihr persönlicher Einladungscode lautet:

    ${code}

Dieser Code ist 48 Stunden gültig.

Öffnen Sie die App, geben Sie Ihre E-Mail-Adresse ein und
wählen Sie "Einladungscode eingeben".

Mit freundlichen Grüßen
Philo von Freisinn`;

  const htmlBody = `<p>Guten Tag,</p>
<p>Sie wurden eingeladen, <strong>Philo von Freisinn</strong> zu nutzen.</p>
${google_play_url ? `<p>Installieren Sie die App über den <a href="${google_play_url}">Google Play Store</a>.</p>` : ""}
<p>Ihr persönlicher Einladungscode lautet:</p>
<p style="font-size: 24px; font-weight: bold; letter-spacing: 4px; text-align: center; padding: 16px; background: #f5f5f5; border-radius: 8px;">${code}</p>
<p>Dieser Code ist <strong>48 Stunden</strong> gültig.</p>
<p>Öffnen Sie die App, geben Sie Ihre E-Mail-Adresse ein und wählen Sie <em>"Einladungscode eingeben"</em>.</p>
<p>Mit freundlichen Grüßen<br/>Philo von Freisinn</p>`;

  if (!RESEND_API_KEY) {
    console.error("RESEND_API_KEY not configured");
    return new Response(
      JSON.stringify({ error: "Email service not configured" }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }

  const resendResp = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: `Philo von Freisinn <${FROM_EMAIL}>`,
      to: [invitee_email],
      subject: "Philo von Freisinn lädt Sie ein.",
      text: textBody,
      html: htmlBody,
    }),
  });

  if (!resendResp.ok) {
    const errBody = await resendResp.text();
    console.error("Resend error:", resendResp.status, errBody);
    return new Response(
      JSON.stringify({ error: "Failed to send email", details: errBody }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }

  const result = await resendResp.json();
  return new Response(
    JSON.stringify({ sent: true, id: result.id }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
});
