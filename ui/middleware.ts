import createMiddleware from "next-intl/middleware";
import { NextResponse, type NextRequest } from "next/server";
import { getToken } from "next-auth/jwt";

import { routing } from "./i18n/routing";

const intlMiddleware = createMiddleware(routing);

function requiresAuth(pathname: string): boolean {
  return pathname.includes("/api/admin/turns/") || pathname.includes("/api/admin/talks/");
}

export async function middleware(request: NextRequest) {
  const pathname = request.nextUrl.pathname;

  // Next.js API routes (incl. next-auth) live at /api/* — do not prefix with locale.
  if (pathname.startsWith("/api/")) {
    if (requiresAuth(pathname)) {
      const token = await getToken({
        req: request,
        secret: process.env.NEXTAUTH_SECRET,
      });
      if (!token) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
      }
    }
    return NextResponse.next();
  }

  return intlMiddleware(request);
}

export const config = {
  matcher: ["/((?!_next|.*\\..*).*)"],
};
