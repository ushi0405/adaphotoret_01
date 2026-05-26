# download_model.py
import os
from sentence_transformers import SentenceTransformer

# 设置镜像源，加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("正在从 Hugging Face 下载完整模型...")
# 下载模型（会自动获取所有需要的文件，包括 modules/ 子目录）
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 保存到本地文件夹（推荐放在项目目录下，命名为 local_model）
save_path = "./local_model"
print(f"正在保存模型到本地：{save_path}")
model.save_pretrained(save_path)

print("模型下载并保存成功！")