import type { NextAuthOptions } from "next-auth";
import GithubProvider from "next-auth/providers/github";

export const authOptions: NextAuthOptions = {
  providers: [
    GithubProvider({
      clientId: process.env.GITHUB_CLIENT_ID || "",
      clientSecret: process.env.GITHUB_CLIENT_SECRET || "",
    }),
  ],
  session: {
    strategy: "jwt",
  },
  pages: {
    signIn: "/de/stats",
  },
  callbacks: {
    async jwt({ token, profile }) {
      if (profile) {
        token.githubId = String(profile.id || "");
        token.githubLogin = (profile as { login?: string }).login || "";
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.sub || "";
        session.user.githubId = String(token.githubId || "");
        session.user.githubLogin = String(token.githubLogin || "");
      }
      return session;
    },
  },
};
