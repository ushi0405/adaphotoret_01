import { useState, useRef, useEffect } from 'react';
import {
  Plus,
  MoreHorizontal,
  Pencil,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Images,
  MessageSquare,
} from 'lucide-react';
import type { Mode, Conversation, DateGroup } from '../types';
import { groupConversations } from '../types';

// ─── Props ────────────────────────────────────────────────────────────────────

interface SidebarProps {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onRenameConversation: (id: string, title: string) => void;
  onDeleteConversation: (id: string) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

// ─── Context menu ─────────────────────────────────────────────────────────────

interface ContextMenuProps {
  conversationId: string;
  title: string;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}

function ConversationContextMenu({ conversationId, title, onRename, onDelete }: ContextMenuProps) {
  const [open, setOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState(title);
  const menuRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function handler(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  // Focus input when renaming starts
  useEffect(() => {
    if (renaming) inputRef.current?.focus();
  }, [renaming]);

  function handleRenameSubmit() {
    if (renameValue.trim()) {
      onRename(conversationId, renameValue.trim());
    }
    setRenaming(false);
    setOpen(false);
  }

  if (renaming) {
    return (
      <input
        ref={inputRef}
        value={renameValue}
        onChange={(e) => setRenameValue(e.target.value)}
        onBlur={handleRenameSubmit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') handleRenameSubmit();
          if (e.key === 'Escape') { setRenaming(false); setOpen(false); }
        }}
        onClick={(e) => e.stopPropagation()}
        className="w-full bg-[#2a2a2a] text-white text-sm px-2 py-0.5 rounded border border-[#6e8efb]/50 outline-none focus:border-[#6e8efb]"
      />
    );
  }

  return (
    <div ref={menuRef} className="relative" onClick={(e) => e.stopPropagation()}>
      <button
        onClick={() => setOpen(!open)}
        className="p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity hover:bg-white/10 text-gray-400 hover:text-white"
        aria-label="More options"
      >
        <MoreHorizontal size={14} />
      </button>

      {open && (
        <div className="absolute right-0 top-6 z-50 w-36 bg-[#2a2a2a] border border-white/10 rounded-lg shadow-2xl overflow-hidden animate-fadeIn">
          <button
            onClick={() => { setRenaming(true); setOpen(false); }}
            className="flex items-center gap-2 w-full px-3 py-2 text-sm text-gray-300 hover:bg-white/10 hover:text-white transition-colors"
          >
            <Pencil size={13} />
            重命名
          </button>
          <button
            onClick={() => { onDelete(conversationId); setOpen(false); }}
            className="flex items-center gap-2 w-full px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 hover:text-red-300 transition-colors"
          >
            <Trash2 size={13} />
            删除
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Main Sidebar ─────────────────────────────────────────────────────────────

const DATE_GROUP_ORDER: DateGroup[] = ['今天', '昨天', '7天内', '30天内', '更早'];

export default function Sidebar({
  mode,
  onModeChange,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onRenameConversation,
  onDeleteConversation,
  collapsed,
  onToggleCollapse,
}: SidebarProps) {
  const grouped = groupConversations(conversations);

  return (
    <aside
      className={`
        relative flex flex-col h-full bg-[#1e1e1e] border-r border-white/5
        transition-all duration-300 ease-in-out flex-shrink-0
        ${collapsed ? 'w-0 overflow-hidden' : 'w-64'}
      `}
    >
      {/* Collapse toggle button */}
      <button
        onClick={onToggleCollapse}
        className="absolute -right-3 top-5 z-20 w-6 h-6 rounded-full bg-[#2a2a2a] border border-white/10 flex items-center justify-center text-gray-400 hover:text-white hover:bg-[#333] transition-all shadow-lg"
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
      </button>

      <div className="flex flex-col h-full min-w-[256px] overflow-hidden">
        {/* ── Logo ── */}
        <div className="px-5 pt-5 pb-4 border-b border-white/5">
          <h1 className="text-xl font-bold tracking-tight gradient-text select-none">
            AdaphotoRet
          </h1>
          <p className="text-xs text-gray-500 mt-0.5 font-light">AI 照片检索助手</p>
        </div>

        {/* ── Mode Switch ── */}
        <div className="px-4 py-4 border-b border-white/5">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2.5 px-1">
            检索模式
          </p>
          <div className="flex gap-1 p-1 bg-[#0f0f0f] rounded-lg">
            <ModeButton
              active={mode === 'global'}
              onClick={() => onModeChange('global')}
              icon={<Images size={13} />}
              label="全局检索"
            />
            <ModeButton
              active={mode === 'fine'}
              onClick={() => onModeChange('fine')}
              icon={<MessageSquare size={13} />}
              label="精细检索"
            />
          </div>
          <p className="text-[11px] text-gray-600 mt-2 px-1 leading-relaxed">
            {mode === 'global'
              ? '在全部照片中快速语义搜索'
              : '多轮对话驱动的可解释精细匹配'}
          </p>
        </div>

        {/* ── Content area ── */}
        <div className="flex-1 overflow-y-auto">
          {mode === 'fine' ? (
            <div className="px-3 py-3">
              {/* New conversation button */}
              <button
                onClick={onNewConversation}
                className="flex items-center gap-2 w-full px-3 py-2.5 rounded-lg text-sm font-medium
                  bg-gradient-to-r from-[#6e8efb]/20 to-[#a777e3]/20
                  border border-[#6e8efb]/25 text-[#a0b4ff]
                  hover:from-[#6e8efb]/30 hover:to-[#a777e3]/30 hover:text-white
                  transition-all duration-200 mb-3 group"
              >
                <Plus size={15} className="group-hover:rotate-90 transition-transform duration-200" />
                新建对话
              </button>

              {/* Conversation groups */}
              {DATE_GROUP_ORDER.map((group) => {
                const convs = grouped[group];
                if (!convs || convs.length === 0) return null;
                return (
                  <div key={group} className="mb-3">
                    <p className="text-[11px] font-medium text-gray-600 uppercase tracking-wider px-2 mb-1">
                      {group}
                    </p>
                    <div className="space-y-0.5">
                      {convs.map((conv) => (
                        <ConversationItem
                          key={conv.id}
                          conversation={conv}
                          isActive={conv.id === activeConversationId}
                          onSelect={onSelectConversation}
                          onRename={onRenameConversation}
                          onDelete={onDeleteConversation}
                        />
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="px-5 py-6">
              <div className="space-y-3 text-sm text-gray-500 leading-relaxed">
                <p className="flex items-start gap-2">
                  <span className="text-[#6e8efb] mt-0.5">•</span>
                  输入关键词或自然语言描述
                </p>
                <p className="flex items-start gap-2">
                  <span className="text-[#6e8efb] mt-0.5">•</span>
                  实时语义匹配全部照片库
                </p>
                <p className="flex items-start gap-2">
                  <span className="text-[#6e8efb] mt-0.5">•</span>
                  按相关性得分自动排序
                </p>
                <p className="flex items-start gap-2">
                  <span className="text-[#6e8efb] mt-0.5">•</span>
                  支持中英文混合查询
                </p>
              </div>
            </div>
          )}
        </div>

        {/* ── Footer ── */}
        <div className="px-5 py-4 border-t border-white/5">
          <p className="text-[11px] text-gray-700">AdaphotoRet v0.1 · 演示模式</p>
        </div>
      </div>
    </aside>
  );
}

// ─── Mode toggle button ───────────────────────────────────────────────────────

function ModeButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        flex-1 flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-md text-xs font-medium
        transition-all duration-200
        ${active
          ? 'bg-gradient-to-r from-[#6e8efb] to-[#a777e3] text-white shadow-lg shadow-[#6e8efb]/20'
          : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'}
      `}
    >
      {icon}
      {label}
    </button>
  );
}

// ─── Conversation list item ───────────────────────────────────────────────────

function ConversationItem({
  conversation,
  isActive,
  onSelect,
  onRename,
  onDelete,
}: {
  conversation: Conversation;
  isActive: boolean;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div
      onClick={() => onSelect(conversation.id)}
      className={`
        group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer
        transition-all duration-150
        ${isActive
          ? 'bg-gradient-to-r from-[#6e8efb]/15 to-[#a777e3]/10 border border-[#6e8efb]/20 text-white'
          : 'text-gray-400 hover:bg-white/5 hover:text-gray-200 border border-transparent'}
      `}
    >
      <MessageSquare size={13} className={isActive ? 'text-[#6e8efb]' : 'text-gray-600'} />
      <span className="flex-1 text-sm truncate leading-snug">{conversation.title}</span>
      <ConversationContextMenu
        conversationId={conversation.id}
        title={conversation.title}
        onRename={onRename}
        onDelete={onDelete}
      />
    </div>
  );
}
