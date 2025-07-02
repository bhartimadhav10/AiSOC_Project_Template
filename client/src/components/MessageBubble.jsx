// MessageBubble.jsx
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Spinner from '../components/Spinner';
import userIcon from '../assets/images/user.svg';
import botIcon from '../assets/images/bot.svg';
import errorIcon from '../assets/images/error.svg';
import { format } from 'date-fns';

export default function MessageBubble({ role, content, loading, error, timestamp }) {
    const formattedTime = timestamp ? format(new Date(timestamp), 'h:mm a') : '';

    return (
        <div className={`flex gap-4 ${role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {role === 'assistant' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                    <img className="w-5 h-5" src={botIcon} alt="AI assistant" />
                </div>
            )}

            <div className={`max-w-[85%] flex flex-col ${role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className={`px-4 py-3 rounded-2xl ${role === 'user'
                    ? 'bg-primary text-primary-content rounded-br-none'
                    : 'bg-base-200 rounded-bl-none'}`}
                >
                    {loading && !content ? (
                        <div className="flex items-center gap-2">
                            <Spinner size={18} />
                            <span>Thinking...</span>
                        </div>
                    ) : role === 'assistant' ? (
                        <div className="prose prose-sm max-w-none">
                            <Markdown remarkPlugins={[remarkGfm]}>
                                {content}
                            </Markdown>
                        </div>
                    ) : (
                        <div className="whitespace-pre-line">{content}</div>
                    )}
                </div>

                {error && (
                    <div className="flex items-center gap-1 mt-1 text-xs text-error">
                        <img className="h-3 w-3" src={errorIcon} alt="error" />
                        <span>Error generating response</span>
                    </div>
                )}

                {timestamp && (
                    <div className="text-xs text-base-content/50 mt-1">
                        {formattedTime}
                    </div>
                )}
            </div>

            {role === 'user' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                    <img className="w-4 h-4" src={userIcon} alt="user" />
                </div>
            )}
        </div>
    );
}
