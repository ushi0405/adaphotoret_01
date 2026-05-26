import { useState, useRef, useEffect } from 'react';
import {
  Send,
  ChevronDown,
  ChevronUp,
  Brain,
  Zap,
  FileText,
  Image as ImageIcon,
  User,
  Sparkles,
} from 'lucide-react';
import type { ChatMessage, Conversation } from '../types';

const BACKEND_URL = 'http://172.25.104.83:8000';

function generateId() {
  return Math.random().toString(36).slice(2, 10);
}

function ReportBlock({ content }: { content: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="mt-3 border border-white/8 rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full px-4 py-2.5 bg-white/5 hover:bg-white/8 text-left"
      >
        <FileText size={13} className="text-[#6e8efb]" />
        <span className="text-xs font-medium text-gray-300 flex-1">检索分析报告</span>
        {expanded ? '▲' : '▼'}
      </button>
      {expanded && (
        <div className="px-4 py-3 bg-[#0f0f0f]/60 text-xs whitespace-pre-wrap">
          {content}
        </div>
      )}
    </div>
  );
}

function PhotoStrip({ photos }: { photos: { id: string; url: string }[] }) {
  return (
    <div className="flex gap-2 mt-3 flex-wrap">
      {photos.map(photo => (
        <div key={photo.id} className="w-[120px] h-[90px] rounded-lg overflow-hidden bg-[#1e1e1e]">
          <img src={photo.url} className="w-full h-full object-cover" alt="" />
        </div>
      ))}
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-lg bg-gradient-to-br from-[#6e8efb]/25 to-[#a777e3]/20 border border-[#6e8efb]/20 rounded-2xl rounded-tr-md px-4 py-2.5 text-sm text-white">
          {message.content}
        </div>
        <div className="ml-2.5 mt-0.5 w-7 h-7 rounded-full bg-gradient-to-br from-[#6e8efb] to-[#a777e3] flex items-center justify-center">
          <User size={13} className="text-white" />
        </div>
      </div>
    );
  }

  const photos = (message.photos || []).map(p => ({ id: p.id, url: p.url }));
  return (
    <div className="flex items-start gap-2.5">
      <div className="mt-0.5 w-7 h-7 rounded-full bg-[#1e1e1e] border border-[#6e8efb]/30 flex items-center justify-center">
        <Sparkles size={13} className="text-[#6e8efb]" />
      </div>
      <div className="flex-1 max-w-2xl">
        {message.thinking && (
          <div className="mb-2 bg-[#1a1a1a] border border-white/6 rounded-xl px-3 py-2.5">
            <Brain size={12} className="text-gray-600 inline mr-1" />
            <span className="text-xs text-gray-600 font-mono">{message.thinking}</span>
          </div>
        )}
        {message.action && (
          <div className="mb-2 bg-[#0f1a2e] border border-[#6e8efb]/15 rounded-xl px-3 py-2.5">
            <Zap size={12} className="text-[#6e8efb] inline mr-1" />
            <span className="text-xs text-[#7aa4f7] font-mono">{message.action}</span>
          </div>
        )}
        {message.content && (
          <div className="bg-[#1e1e1e] border border-white/8 rounded-2xl rounded-tl-md px-4 py-3">
            <p className="text-sm text-gray-200 whitespace-pre-wrap">{message.content}</p>
            {photos.length > 0 && (
              <>
                <div className="flex items-center gap-1.5 mt-3 mb-1">
                  <ImageIcon size={12} className="text-gray-500" />
                  <span className="text-[11px] text-gray-500">检索结果 · {photos.length} 张</span>
                </div>
                <PhotoStrip photos={photos} />
              </>
            )}
            {message.report && <ReportBlock content={message.report} />}
          </div>
        )}
      </div>
    </div>
  );
}

function WelcomeScreen({ onQuickQuery }: { onQuickQuery: (q: string) => void }) {
  const examples = ['水煮肉片', '青城山', '宠物猫', '海滩日落'];
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 text-center">
      <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-[#6e8efb]/20 to-[#a777e3]/20 border border-[#6e8efb]/20 flex items-center justify-center mb-4">
        <Sparkles size={24} className="text-[#6e8efb]" />
      </div>
      <h2 className="text-xl font-semibold text-white mb-2">精细检索模式</h2>
      <p className="text-sm text-gray-500 max-w-sm leading-relaxed mb-8">
        通过自然语言对话，调用 AI 模型精确定位您想要的照片。
      </p>
      <div className="w-full max-w-lg space-y-2">
        <p className="text-xs text-gray-600 mb-3">快速示例</p>
        {examples.map(q => (
          <button
            key={q}
            onClick={() => onQuickQuery(q)}
            className="w-full text-left px-4 py-3 bg-[#1e1e1e] border border-white/8 rounded-xl text-sm text-gray-400 hover:text-gray-200 hover:border-[#6e8efb]/30"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

interface ChatInterfaceProps {
  conversation: Conversation;
  onUpdateConversation: (conv: Conversation) => void;
}

export default function ChatInterface({ conversation, onUpdateConversation }: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation.messages]);

  const updateConversation = (newConv: Conversation) => {
    onUpdateConversation(newConv);
  };

  const handleSend = async () => {
    const query = inputValue.trim();
    if (!query || isProcessing) return;

    setInputValue('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    const userMsg: ChatMessage = { id: generateId(), role: 'user', content: query, timestamp: new Date() };
    const newTitle = conversation.messages.length === 0 ? query.slice(0, 20) + (query.length > 20 ? '...' : '') : conversation.title;
    let updatedConv = { ...conversation, title: newTitle, messages: [...conversation.messages, userMsg], updatedAt: new Date() };
    updateConversation(updatedConv);

    setIsProcessing(true);

    const assistantId = generateId();
    const assistantPlaceholder: ChatMessage = { id: assistantId, role: 'assistant', content: '', thinking: '', action: '', photos: [], report: '', isStreaming: true, timestamp: new Date() };
    updatedConv = { ...updatedConv, messages: [...updatedConv.messages, assistantPlaceholder] };
    updateConversation(updatedConv);

    const patchAssistant = (patch: Partial<ChatMessage>) => {
      const conv = { ...updatedConv };
      const idx = conv.messages.findIndex(m => m.id === assistantId);
      if (idx !== -1) {
        conv.messages[idx] = { ...conv.messages[idx], ...patch };
        conv.updatedAt = new Date();
        updateConversation(conv);
      }
    };

    patchAssistant({ thinking: '正在检索...', action: `检索：${query}` });

    try {
      const response = await fetch(`${BACKEND_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await response.json();
      const photoUrls = data.images || [];
      const report = data.report || '未生成报告。';
      const photos = photoUrls.map((url: string, i: number) => ({ id: `photo-${Date.now()}-${i}`, url: `${BACKEND_URL}${url}` }));
      patchAssistant({ photos, content: report, report, isStreaming: false });
    } catch (err) {
      console.error(err);
      patchAssistant({ content: `检索失败：${err instanceof Error ? err.message : '未知错误'}`, isStreaming: false });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isEmpty = conversation.messages.length === 0;

  return (
    <div className="flex flex-col h-full">
      <div className="flex-shrink-0 px-6 py-4 border-b border-white/5">
        <h2 className="text-sm font-medium text-white">{isEmpty ? '新对话' : conversation.title}</h2>
        <p className="text-[11px] text-gray-600">精细检索 · 可解释 AI</p>
      </div>

      <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6">
        {isEmpty ? (
          <WelcomeScreen onQuickQuery={(q) => { setInputValue(q); textareaRef.current?.focus(); }} />
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {conversation.messages.map(msg => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="flex-shrink-0 px-4 sm:px-6 py-4 border-t border-white/5">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-3 bg-[#1e1e1e] border border-white/10 rounded-2xl px-4 py-3">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="描述您想找的照片，按 Enter 发送"
              rows={1}
              disabled={isProcessing}
              className="flex-1 bg-transparent text-sm text-white placeholder-gray-600 outline-none resize-none"
              style={{ minHeight: '40px' }}
            />
            <button
              onClick={handleSend}
              disabled={!inputValue.trim() || isProcessing}
              className={`w-8 h-8 rounded-xl flex items-center justify-center ${
                inputValue.trim() && !isProcessing
                  ? 'bg-gradient-to-br from-[#6e8efb] to-[#a777e3] text-white'
                  : 'bg-white/5 text-gray-600 cursor-not-allowed'
              }`}
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}