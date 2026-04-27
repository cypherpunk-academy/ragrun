"use client";

import { SessionProvider } from "next-auth/react";
import { NextIntlClientProvider } from "next-intl";

import { APP_TIME_ZONE } from "@/i18n/constants";

type Props = {
  children: React.ReactNode;
  locale: string;
  messages: Record<string, string>;
};

export function Providers({ children, locale, messages }: Props) {
  return (
    <SessionProvider basePath="/api/auth">
      <NextIntlClientProvider
        locale={locale}
        messages={messages}
        timeZone={APP_TIME_ZONE}
      >
        {children}
      </NextIntlClientProvider>
    </SessionProvider>
  );
}
