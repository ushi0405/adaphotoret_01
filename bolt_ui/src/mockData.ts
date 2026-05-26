import type { Photo, Conversation } from './types';

// ─── Photo pool ──────────────────────────────────────────────────────────────
// Uses picsum.photos with fixed seed IDs for reproducible results

const PHOTO_IDS = [
  10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
  110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
  210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
  310, 320, 330, 340, 350, 360, 370, 380, 390, 400,
  410, 420, 430, 440, 450, 460, 470, 480, 490, 500,
  510, 520, 530, 540, 550, 560, 570, 580, 590, 600,
];

export function buildPhoto(seedId: number, index: number): Photo {
  return {
    id: `photo-${seedId}-${index}`,
    url: `https://picsum.photos/seed/${seedId}/400/300`,
    width: 400,
    height: 300,
    score: parseFloat((Math.random() * 0.4 + 0.6).toFixed(3)),
    tags: ['风景', '自然', '旅行'].slice(0, Math.floor(Math.random() * 3) + 1),
  };
}

export const GLOBAL_PHOTOS: Photo[] = PHOTO_IDS.map((id, i) => buildPhoto(id, i));

// Simulate relevance-sorted retrieval — replace with real API call
export function searchPhotos(query: string): Photo[] {
  if (!query.trim()) return GLOBAL_PHOTOS;
  // Shuffle to simulate re-ranking; real implementation calls backend
  const shuffled = [...GLOBAL_PHOTOS];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// ─── Sample conversations ────────────────────────────────────────────────────

function daysAgo(n: number): Date {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d;
}

export const INITIAL_CONVERSATIONS: Conversation[] = [
  {
    id: 'conv-1',
    title: '查找海边日落照片',
    messages: [],
    createdAt: daysAgo(0),
    updatedAt: daysAgo(0),
  },
  {
    id: 'conv-2',
    title: '城市夜景检索',
    messages: [],
    createdAt: daysAgo(1),
    updatedAt: daysAgo(1),
  },
  {
    id: 'conv-3',
    title: '山地风光组合查询',
    messages: [],
    createdAt: daysAgo(3),
    updatedAt: daysAgo(3),
  },
  {
    id: 'conv-4',
    title: '人像摄影精细匹配',
    messages: [],
    createdAt: daysAgo(12),
    updatedAt: daysAgo(12),
  },
  {
    id: 'conv-5',
    title: '秋季叶片色彩分析',
    messages: [],
    createdAt: daysAgo(45),
    updatedAt: daysAgo(45),
  },
];

// ─── Streaming simulation ────────────────────────────────────────────────────

export const THINKING_TEMPLATES = [
  '正在分析您的查询意图，提取关键视觉特征...\n识别到核心概念：场景语义 + 色调分布 + 构图特征\n准备调用多模态检索模型...',
  '理解用户需求：精细语义匹配模式\n分解查询为：主体特征 / 环境氛围 / 时间线索\n生成查询向量并与图库向量空间对比...',
  '处理自然语言查询，转化为视觉嵌入向量\n在 60,000 张照片中执行近似最近邻搜索\n按相关性得分重新排序候选集...',
];

export const ACTION_TEMPLATES = [
  '【行动】检索：`scene:outdoor lighting:golden-hour composition:rule-of-thirds`\n召回 Top-12 候选，相关性阈值 > 0.72\n正在生成分析报告...',
  '【行动】检索：`subject:portrait mood:natural depth-of-field:bokeh`\n召回 Top-8 候选，语义相似度 > 0.81\n汇总特征维度数据...',
  '【行动】检索：`season:autumn color:warm-tones texture:leaf-detail`\n召回 Top-10 候选，色彩匹配度 > 0.76\n生成可解释性报告...',
];

export const REPORT_TEMPLATES = [
  `## 检索报告

### 查询解析
- **主语义**：黄金时段户外场景
- **视觉特征**：暖色调、逆光、自然光晕
- **构图分析**：三分法则构图，水平线居中

### 检索策略
\`\`\`
CLIP 嵌入向量 → ANN 搜索 (FAISS)
阈值过滤 → 多维重排序 → Top-K 输出
\`\`\`

### 结果统计
| 指标 | 值 |
|------|-----|
| 召回数量 | 12 张 |
| 平均相关性 | 0.847 |
| 最高得分 | 0.923 |

### 可解释性
模型关注区域集中在 **光源方向**、**色温分布** 和 **主体轮廓**，符合查询语义。`,

  `## 检索报告

### 查询解析
- **主语义**：自然人像，浅景深
- **视觉特征**：柔焦背景、自然肤色、眼神接触
- **情绪基调**：温暖、亲近

### 检索策略
\`\`\`
人脸检测预过滤 → 属性向量提取
→ 语义相似度排序 → 多样性重排
\`\`\`

### 结果统计
| 指标 | 值 |
|------|-----|
| 召回数量 | 8 张 |
| 平均相关性 | 0.863 |
| 最高得分 | 0.941 |

### 可解释性
检索模型对 **背景虚化程度**、**主体清晰度** 和 **色彩柔和度** 赋予了更高权重。`,
];
