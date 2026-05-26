// ─── Core Types ─────────────────────────────────────────────────────────────

export type Mode = 'global' | 'fine';

export interface Photo {
  id: string;
  url: string;
  width: number;
  height: number;
  score?: number;
  tags?: string[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  thinking?: string;
  action?: string;
  photos?: Photo[];
  report?: string;
  isStreaming?: boolean;
  timestamp: Date;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

// ─── Date grouping ───────────────────────────────────────────────────────────

export type DateGroup = '今天' | '昨天' | '7天内' | '30天内' | '更早';

export function getDateGroup(date: Date): DateGroup {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return '今天';
  if (diffDays === 1) return '昨天';
  if (diffDays <= 7) return '7天内';
  if (diffDays <= 30) return '30天内';
  return '更早';
}

export function groupConversations(conversations: Conversation[]): Record<DateGroup, Conversation[]> {
  const groups: Record<DateGroup, Conversation[]> = {
    今天: [],
    昨天: [],
    '7天内': [],
    '30天内': [],
    更早: [],
  };
  for (const conv of conversations) {
    const group = getDateGroup(conv.updatedAt);
    groups[group].push(conv);
  }
  return groups;
}
