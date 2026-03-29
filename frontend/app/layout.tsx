import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DefectFill Industrial QC",
  description: "Real-time smart manufacturing quality control dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
