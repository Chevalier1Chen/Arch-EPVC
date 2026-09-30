import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://edutwin-shandong-school.chevalierchen.chatgpt.site"),
  title: "Arch-EPVC · Building Performance Intelligence",
  description: "Multimodal and temporal design evaluation for school buildings.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: { title: "Arch-EPVC", description: "Building Performance Intelligence", images: [{ url: "/og.png", width: 1536, height: 1024 }] },
  twitter: { card: "summary_large_image", title: "Arch-EPVC", description: "Building Performance Intelligence", images: ["/og.png"] },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
