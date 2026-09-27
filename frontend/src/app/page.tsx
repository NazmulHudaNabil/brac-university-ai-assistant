"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Bot, BookOpen, Calendar, GraduationCap, Phone, Edit3, MoreVertical, PlusCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Citation {
  document_id?: string;
  url?: string;
  title?: string;
  snippet: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Citation[];
  timestamp: string;
}

const getCurrentTime = () => {
  const now = new Date();
  return `Today ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
};

const initialGreeting: Message = {
  role: "assistant",
  content: "Hello! I am the **BRAC University AI Assistant**. \n\nWhat do you want to learn more about?",
  timestamp: getCurrentTime()
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([initialGreeting]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const sendSpecificMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessage: Message = { 
      role: "user", 
      content: text,
      timestamp: getCurrentTime()
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const apiUrl = `${baseUrl.replace(/\/$/, '')}/chat`;
      
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMessage.content, conversation_id: "default-web" }),
      });

      if (!response.ok) throw new Error("Failed to fetch response");

      const data = await response.json();
      
      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        timestamp: getCurrentTime()
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Sorry, I encountered an error. Please ensure the backend is running.", timestamp: getCurrentTime() }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = () => sendSpecificMessage(input);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") sendMessage();
  };

  const handleNewSession = () => {
    setMessages([{...initialGreeting, timestamp: getCurrentTime()}]);
    setIsMenuOpen(false);
    setInput("");
  };

  const suggestedQuestions = [
    { label: "How to apply?", icon: <Edit3 className="w-4 h-4" /> },
    { label: "Programmes", icon: <GraduationCap className="w-4 h-4" /> },
    { label: "Open days", icon: <Calendar className="w-4 h-4" /> },
    { label: "Contact us", icon: <Phone className="w-4 h-4" /> },
  ];

  return (
    <div className="flex flex-col h-[100dvh] md:p-6 lg:p-8 bg-background text-foreground font-sans items-center justify-center">
      <div className="flex flex-col w-full max-w-5xl h-full bg-card md:shadow-2xl md:rounded-3xl border border-border overflow-hidden relative">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 bg-primary shadow-sm z-20">
          <div className="flex items-center gap-3">
            <div className="relative bg-background p-2 rounded-full shadow-sm">
              <Bot className="w-5 h-5 text-primary" />
              <div className="absolute bottom-0 right-0 w-2.5 h-2.5 bg-green-500 border-2 border-background rounded-full"></div>
            </div>
            <div>
              <h1 className="text-[17px] font-semibold text-primary-foreground tracking-wide">
                BRAC University Bot
              </h1>
              <p className="text-[12px] text-primary-foreground/80 font-medium">AI Assistant</p>
            </div>
          </div>
          
          {/* 3-Dot Menu */}
          <div className="relative">
            <button 
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="p-2 text-primary-foreground hover:bg-primary-dark rounded-full transition-colors"
            >
              <MoreVertical className="w-5 h-5" />
            </button>
            
            {isMenuOpen && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setIsMenuOpen(false)}></div>
                <div className="absolute right-0 mt-2 w-48 bg-popover rounded-xl shadow-xl border border-border overflow-hidden z-20 py-1">
                  <button 
                    onClick={handleNewSession}
                    className="w-full flex items-center gap-3 px-4 py-3 text-[15px] text-popover-foreground hover:bg-secondary transition-colors"
                  >
                    <PlusCircle className="w-4 h-4 text-primary" />
                    <span className="font-medium">New Session</span>
                  </button>
                </div>
              </>
            )}
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto px-4 md:px-8 pt-6 pb-4 bg-card" ref={scrollRef}>
          <div className="flex flex-col gap-6">
            {messages.map((msg, idx) => (
              <div key={idx} className="flex flex-col">
                <span suppressHydrationWarning className={`text-[11px] font-medium text-muted-foreground mb-1.5 ${msg.role === "user" ? "text-right mr-1" : "text-left ml-11"}`}>
                  {msg.timestamp}
                </span>
                
                <div className={`flex gap-3 items-start ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                  
                  {/* Assistant Avatar */}
                  {msg.role === "assistant" && (
                    <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <Bot className="w-4 h-4 text-primary" />
                    </div>
                  )}

                  {/* Message Bubble */}
                  <div className="flex flex-col gap-2 max-w-[90%] md:max-w-[85%]">
                    <div 
                      className={`px-5 py-3.5 text-[15px] leading-relaxed shadow-sm ${
                        msg.role === "user" 
                          ? "bg-primary text-primary-foreground rounded-2xl rounded-tr-sm" 
                          : "bg-secondary text-secondary-foreground rounded-2xl rounded-tl-sm"
                      }`}
                    >
                      {msg.role === "assistant" ? (
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            p: ({node, ...props}) => <p className="mb-3 last:mb-0" {...props} />,
                            strong: ({node, ...props}) => <strong className="font-semibold text-foreground" {...props} />,
                            ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-3 space-y-1" {...props} />,
                            ol: ({node, ...props}) => <ol className="list-decimal pl-5 mb-3 space-y-1" {...props} />,
                            li: ({node, ...props}) => <li className="" {...props} />,
                            a: ({node, ...props}) => <a className="text-accent underline underline-offset-2 hover:brightness-110 transition-colors font-medium" {...props} />,
                            h1: ({node, ...props}) => <h1 className="text-lg font-bold mb-2 mt-4" {...props} />,
                            h2: ({node, ...props}) => <h2 className="text-base font-bold mb-2 mt-3" {...props} />,
                            h3: ({node, ...props}) => <h3 className="text-sm font-bold mb-1 mt-2" {...props} />,
                            hr: ({node, ...props}) => <hr className="my-4 border-border" {...props} />
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      ) : (
                        msg.content
                      )}
                    </div>

                    {/* Example Questions for the very first message */}
                    {idx === 0 && msg.role === "assistant" && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {suggestedQuestions.map((q, qIdx) => (
                          <button
                            key={qIdx}
                            onClick={() => sendSpecificMessage(q.label)}
                            className="flex items-center gap-2 px-4 py-2 bg-card border border-border rounded-full text-[14px] font-medium text-primary hover:bg-secondary transition-colors shadow-sm"
                          >
                            {q.icon}
                            {q.label}
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Citations block */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="flex flex-col gap-2 mt-1">
                        <div className="flex flex-col rounded-xl border border-border bg-card overflow-hidden shadow-sm">
                          {msg.sources.map((source, sIdx) => (
                            <a 
                              key={sIdx} 
                              href={source.url || "#"} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className={`flex items-center gap-2.5 px-4 py-3 text-sm text-muted-foreground hover:bg-secondary transition-colors ${
                                sIdx !== msg.sources!.length - 1 ? "border-b border-border" : ""
                              }`}
                              title={source.snippet}
                            >
                              <BookOpen className="w-4 h-4 flex-shrink-0 text-muted-foreground/70" />
                              <span className="font-medium truncate">{source.title || source.document_id || `Source ${sIdx + 1}`}</span>
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex flex-col">
                <span suppressHydrationWarning className="text-[11px] font-medium text-muted-foreground mb-1.5 text-left ml-11">
                  {getCurrentTime()}
                </span>
                <div className="flex gap-3 items-start flex-row">
                  <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Bot className="w-4 h-4 text-primary" />
                  </div>
                  <div className="px-5 py-3.5 bg-secondary rounded-2xl rounded-tl-sm shadow-sm flex items-center gap-3">
                    <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                    <span className="text-secondary-foreground text-[15px]">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Input Area */}
        <div className="bg-card border-t border-border pb-safe">
          <div className="flex items-center px-4 py-3 gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Send a message..."
              className="flex-1 bg-transparent border-none outline-none text-foreground placeholder:text-muted-foreground text-[15px] px-2"
              disabled={isLoading}
            />
            <button 
              onClick={sendMessage} 
              disabled={!input.trim() || isLoading}
              className="p-2.5 rounded-full text-muted-foreground hover:text-primary transition-colors disabled:opacity-50"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
