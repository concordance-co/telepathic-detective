import type { Metadata } from "next";
import { Barlow, Michroma } from "next/font/google";
import "./globals.css";

const michroma = Michroma({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-display"
});

const barlow = Barlow({
  weight: ["400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-body"
});

export const metadata: Metadata = {
  title: "Telepathic Detective — A J-Space Interrogation Game",
  description: "Interrogate two suspects and read the memory-sensitive thoughts beneath their testimony using Jacobian Lens."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${michroma.variable} ${barlow.variable}`}>{children}</body>
    </html>
  );
}
