// components/Header.tsx
import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function Header() {
    return (
        <header className="w-full border-b shadow-sm bg-white dark:bg-zinc-900">
            <div className="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
                <Link href="/" className="text-xl font-bold text-primary">
                    🧠ADHD - Adaptive Dynamic Hazard Detection 🎯
                </Link>
                <nav className="space-x-2">
                    <Link href="/upload">
                        <Button variant="ghost">Upload</Button>
                    </Link>
                    <Link href="/dashboard">
                        <Button variant="ghost">Dashboard</Button>
                    </Link>
                </nav>
            </div>
        </header>
    )
}
