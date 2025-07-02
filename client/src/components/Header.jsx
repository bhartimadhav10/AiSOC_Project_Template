// Header.jsx
import { Sparkles, Sun, Moon } from 'lucide-react';
import { useEffect, useState } from 'react';

export default function Header() {
    const [theme, setTheme] = useState('light');

    useEffect(() => {
        const savedTheme = localStorage.getItem('theme') || 'light';
        setTheme(savedTheme);
        document.documentElement.setAttribute('data-theme', savedTheme);
    }, []);

    const toggleTheme = () => {
        const newTheme = theme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    };

    return (
        <header className="sticky top-0 z-20 backdrop-blur bg-base-100/90 border-b border-base-200 ">
            <div className="flex items-center justify-between px-6 py-3 max-w-7xl mx-auto">
                <div className="flex items-center gap-3">
                    <Sparkles />
                    <h1 className="text-xl font-semibold tracking-tight text-base-content">
                        AskPU – University AI Assistant
                    </h1>
                </div>

                <div className="flex items-center gap-4">
                    <button
                        className="btn btn-ghost btn-sm btn-square"
                        onClick={toggleTheme}
                        aria-label="Toggle Theme"
                    >
                        {theme === 'light' ? (
                            <Moon className="w-4 h-4" />
                        ) : (
                            <Sun className="w-4 h-4" />
                        )}
                    </button>
                </div>
            </div>
        </header>
    );
}