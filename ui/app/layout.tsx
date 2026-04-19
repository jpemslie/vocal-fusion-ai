import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VocalFusion",
  description: "AI vocal fusion engine",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-[#e5e5e5] font-mono">
        {children}
      </body>
    </html>
  );
}
