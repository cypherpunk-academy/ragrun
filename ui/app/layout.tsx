import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ragrun Agent Arena",
  description: "Multilingual dashboard for ragrun",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="de" className="h-full antialiased">
      <body className="min-h-full bg-zinc-950 text-zinc-100">{children}</body>
    </html>
  );
}
