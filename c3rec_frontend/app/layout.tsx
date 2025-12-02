import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "C3Rec · Intelligent Learning Platform",
  description: "Course, question & mastery recommendation platform powered by C3Rec.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100">
        {children}
      </body>
    </html>
  );
}
