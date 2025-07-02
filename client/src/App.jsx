// App.jsx
import { useState, useRef, useEffect } from 'react';
import Header from './components/Header';
import ChatInputUI from './components/ChatInputUI';
import MessageBubble from './components/MessageBubble';
import Footer from './components/Footer';
import axios from 'axios';

function ChatMessages({ messages }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4 bg-base-100">
      <div className="max-w-3xl mx-auto space-y-6">
        {messages.map((msg, i) => (
          <MessageBubble key={i} {...msg} />
        ))}
        <div ref={endRef} className="h-4" />
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your university AI assistant. How can I help you today?',
      loading: false,
      timestamp: new Date()
    }
  ]);
  const [newMessage, setNewMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  async function submitNewMessage() {
    const trimmed = newMessage.trim();
    if (!trimmed || isLoading) return;

    const userMessage = {
      role: 'user',
      content: trimmed,
      loading: false,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setNewMessage('');
    setIsLoading(true);

    try {
      // Use the correct endpoint and headers
      const res = await axios.post(
        'http://127.0.0.1:5000/chat', 
        { prompt: trimmed },
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
      
      const botReply = {
        role: 'assistant',
        content: res.data.answer,
        loading: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botReply]);
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: "I'm having trouble connecting right now. Please try again later.",
        loading: false,
        timestamp: new Date(),
        error: true
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error("API Error:", error);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex flex-col h-screen bg-base-100">
      <Header />
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatMessages messages={messages} />
        <ChatInputUI
          value={newMessage}
          onChange={e => setNewMessage(e.target.value)}
          onSubmit={submitNewMessage}
          isLoading={isLoading}
        />
      </main>
      <Footer />
    </div>
  );
}