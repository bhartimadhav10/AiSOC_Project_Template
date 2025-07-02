export default function Footer() {
    return (
        <footer className="w-full border-t border-base-300 bg-base-100/90 backdrop-blur-sm py-3 mt-6">
            <div className="max-w-4xl mx-auto px-4">
                <p className="text-center text-xs text-base-content/70 leading-relaxed">
                    © {new Date().getFullYear()}{' '}
                    <span className="font-medium text-base-content/90">Madhav Bharti | Jatin Dhiman</span> •
                    <span className="hidden sm:inline"> Proton AI </span>
                    {/* <a
                        href="https://github.com/jatindhiman05"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline hover:text-primary transition-colors duration-200"
                    >
                        GitHub
                    </a> */}
                </p>
            </div>
        </footer>
    );
}