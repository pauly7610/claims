import React from 'react'
import './globals.css'

export const metadata = {
  title: 'Claims Admin Panel',
  description: 'Administrative dashboard for insurance claims management',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
