import { useState, useCallback } from 'react';
import { Menu } from 'lucide-react';
import Sidebar from './components/Sidebar';
import GlobalSearch from './components/GlobalSearch';
import ChatInterface from './components/ChatInterface';
import type { Mode, Conversation } from './types';
import { INITIAL_CONVERSATIONS } from './mockData';

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}

export default function App() {
  const [mode, setMode] = useState<Mode>('global');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>(INITIAL_CONVERSATIONS);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    INITIAL_CONVERSATIONS[0]?.id ?? null
  );

  const handleNewConversation = useCallback(() => {
    const newConv: Conversation = {
      id: generateId(),
      title: '新对话',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    setConversations((prev) => [newConv, ...prev]);
    setActiveConversationId(newConv.id);
  }, []);

  const handleSelectConversation = useCallback((id: string) => {
    setActiveConversationId(id);
  }, []);

  const handleRenameConversation = useCallback((id: string, title: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title } : c))
    );
  }, []);

  const handleDeleteConversation = useCallback((id: string) => {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (activeConversationId === id) {
        setActiveConversationId(next[0]?.id ?? null);
      }
      return next;
    });
  }, [activeConversationId]);

  // 关键：必须返回新数组对象，触发 React 重新渲染
  const handleUpdateConversation = useCallback((updated: Conversation) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === updated.id ? updated : c))
    );
  }, []);

  const handleModeChange = useCallback((newMode: Mode) => {
    setMode(newMode);
    if (newMode === 'fine' && conversations.length === 0) {
      const newConv: Conversation = {
        id: generateId(),
        title: '新对话',
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date(),
      };
      setConversations([newConv]);
      setActiveConversationId(newConv.id);
    }
  }, [conversations.length]);

  const activeConversation = conversations.find((c) => c.id === activeConversationId);

  return (
    <div className="flex h-screen bg-[#0f0f0f] text-white overflow-hidden font-sans">
      {!sidebarCollapsed && (
        <div
          className="fixed inset-0 bg-black/50 z-10 md:hidden"
          onClick={() => setSidebarCollapsed(true)}
        />
      )}

      <div
        className={`
          fixed md:relative z-20 h-full
          transition-transform duration-300
          ${sidebarCollapsed ? '-translate-x-full md:translate-x-0' : 'translate-x-0'}
        `}
      >
        <Sidebar
          mode={mode}
          onModeChange={handleModeChange}
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelectConversation={(id) => {
            handleSelectConversation(id);
            if (window.innerWidth < 768) setSidebarCollapsed(true);
          }}
          onNewConversation={() => {
            handleNewConversation();
            if (window.innerWidth < 768) setSidebarCollapsed(true);
          }}
          onRenameConversation={handleRenameConversation}
          onDeleteConversation={handleDeleteConversation}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
        />
      </div>

      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <div className="flex-shrink-0 flex items-center gap-3 px-4 py-3 border-b border-white/5 md:hidden">
          <button
            onClick={() => setSidebarCollapsed((v) => !v)}
            className="p-1.5 rounded-lg hover:bg-white/8 text-gray-400 hover:text-white transition-colors"
          >
            <Menu size={18} />
          </button>
          <h1 className="text-base font-bold gradient-text">AdaphotoRet</h1>
        </div>

        <div className="flex-1 overflow-hidden">
          {mode === 'global' ? (
            <GlobalSearch />
          ) : activeConversation ? (
            <ChatInterface
              conversation={activeConversation}
              onUpdateConversation={handleUpdateConversation}
            />
          ) : (
            <EmptyFineMode onNew={handleNewConversation} />
          )}
        </div>
      </main>
    </div>
  );
}

function EmptyFineMode({ onNew }: { onNew: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-6">
      <p className="text-gray-500 text-sm mb-4">请在左侧新建一个对话</p>
      <button
        onClick={onNew}
        className="px-5 py-2.5 bg-gradient-to-r from-[#6e8efb] to-[#a777e3] text-white text-sm font-medium rounded-xl hover:opacity-90 transition-opacity"
      >
        新建对话
      </button>
    </div>
  );
}