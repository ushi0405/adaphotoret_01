import { useState, useEffect } from 'react';
const BACKEND_URL = 'http://172.25.104.83:8000';

export default function GlobalSearch() {
  const [photos, setPhotos] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    async function fetchPhotos() {
      setLoading(true);
      try {
        if (searchQuery.trim()) {
          // 调用全局排序接口（返回所有照片按相似度降序）
          const res = await fetch(`${BACKEND_URL}/api/global-search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: searchQuery }),
          });
          const data = await res.json();
          const images = data.images || [];
          setPhotos(images.map((url: string) => `${BACKEND_URL}${url}`));
        } else {
          // 无查询时加载所有照片
          const res = await fetch(`${BACKEND_URL}/api/all-photos`);
          const data = await res.json();
          const urls = data.photos || [];
          setPhotos(urls.map((url: string) => `${BACKEND_URL}${url}`));
        }
      } catch (err) {
        console.error('加载失败', err);
        setPhotos([]);
      } finally {
        setLoading(false);
      }
    }
    fetchPhotos();
  }, [searchQuery]);

  return (
    <div className="h-full flex flex-col p-6">
      <div className="mb-6 flex-shrink-0">
        <input
          type="text"
          placeholder="🔍 输入关键词检索照片（留空显示所有照片）"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-4 py-2 bg-[#1e1e1e] border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-[#6e8efb]"
        />
      </div>
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex justify-center items-center h-64">加载中...</div>
        ) : photos.length === 0 ? (
          <div className="text-center text-gray-500 py-20">没有找到照片</div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
            {photos.map((url, idx) => (
              <div key={idx} className="aspect-square rounded-lg overflow-hidden bg-[#1e1e1e]">
                <img src={url} alt="" className="w-full h-full object-cover" />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}