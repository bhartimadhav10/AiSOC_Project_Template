// ChatInputUI.jsx
import sendIcon from '../assets/images/send.svg';
import useAutosize from '../hooks/useAutosize';
import Spinner from './Spinner';

export default function ChatInputUI({ value, onChange, onSubmit, isLoading }) {
    const textareaRef = useAutosize(value);
    const isEmpty = !value.trim();

    function handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey && !isLoading && !isEmpty) {
            e.preventDefault();
            onSubmit();
        }
    }

    return (
        <div className="sticky bottom-0 z-10 bg-base-100/95 backdrop-blur-lg border-t border-base-200 pt-5 pb-6">
            <div className="max-w-3xl mx-auto px-6">
                <div className="relative">
                    <div className="relative bg-base-200 border border-base-300 rounded-xl shadow-sm transition-all duration-200 focus-within:ring-2 focus-within:ring-primary/50 focus-within:border-primary/30 hover:border-base-content/20">
                        <textarea
                            ref={textareaRef}
                            rows="1"
                            value={value}
                            onChange={onChange}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask me anything"
                            className="block w-full bg-transparent resize-none px-5 py-3 pr-16 text-sm placeholder:text-base-content/50 focus:outline-none rounded-xl"
                            disabled={isLoading}
                        />

                        <div className="absolute right-3 top-1/2 -translate-y-1/2">
                            <button
                                onClick={onSubmit}
                                disabled={isEmpty || isLoading}
                                className={`p-2 rounded-full transition-all ${isEmpty || isLoading ? 'opacity-40 cursor-not-allowed' : 'hover:bg-primary/10 active:bg-primary/20 active:scale-95'}`}
                                aria-label="Send message"
                            >
                                {isLoading ? (
                                    <Spinner size={20} />
                                ) : (
                                    <img src={sendIcon} alt="Send" className="w-5 h-5" />
                                )}
                            </button>
                        </div>
                    </div>
                    {/* <div className="mt-2 text-xs text-base-content/40 text-center">
                        Press <kbd className="kbd kbd-sm">Enter</kbd> to send, <kbd className="kbd kbd-sm">Shift</kbd> + <kbd className="kbd kbd-sm">Enter</kbd> for new line
                    </div> */}
                </div>
            </div>
        </div>
    );
}