# AdaphotoRet · 照片检索智能体

以自然语言检索照片：多语言语义召回、属性规则重排、对话式需求澄清，以及照片标签/自定义图库。主入口是 **`streamlit_ui.py`**；保留原有 Gradio、FastAPI、标注与评估代码，不使用假结果替代检索。

## 公共演示

- 打开网页即可浏览 **全部 66 张用户明确授权公开的演示照片**，不需要作者的 API 密钥。
- 全局搜索使用本地运行的 Sentence Transformers 模型和 FAISS，不调用收费 LLM API；首次搜索需要下载约 470 MB 的模型权重，可能较慢。
- 精细对话搜索需要在侧边栏输入**访问者自己的 DeepSeek API 密钥**并同意外部处理。调用费用属于访问者自己的服务账户。作者不提供公共额度。
- 密钥保存在该 Streamlit 会话的服务器内存中，提交给 DeepSeek 时用于认证；不写入文件、Git、日志或共享资源缓存。会话结束会失效；可用侧边栏清除。托管者仍运行服务端程序，若不愿向托管服务提交密钥，请使用下方本地方式。
- 同一张照片使用同一个精细搜索对话：按时间合并用户自己的线索，助手/系统/工具回复不参与实体提取。寻找另一张照片请点「＋ 新建对话」，切换对话不会混合线索。明确的「不是黑色，是白色」类更正会撤回旧线索；「不知道」不会确认助手猜测。
- 精细搜索命中现有直接检索阈值时立即展示候选；每个照片对话最多澄清 3 次，回答第 3 次后不再追问。每次最多一个简短问题；「直接给结果」或对追问回答「不知道」会立即按已有线索展示最接近的候选。分数是原有重排分数，不是命中概率。
- 对话、标签和自定义图库默认仅属于当前会话，不读取旧的共享聊天文件；关闭/重建会话后不保证保留。
- 精细搜索会向 DeepSeek 发送当前照片对话中的全部用户线索和候选照片的文字描述/路径。公共演示不会自动调用照片标注，不会把原图发送给 DeepSeek。

演示库包含这次上传的完整 66 张照片；它不是检索效果基准，不声明评测准确率。可试：`屋顶上的橘猫`、`戴草帽的狗`、`海边草地上的羊`、`花丛中的猫和狗`。模型下载或服务失败时会显示错误，不会编造搜索结果。

## 本地运行（Conda）

推荐 Python **3.12**，从项目根目录执行：

```bash
conda create -n adaphotoret python=3.12 pip -y
conda activate adaphotoret
python -m pip install -r requirements.txt
python download_model.py
python -m streamlit run streamlit_ui.py
```

默认是完整演示照片集/会话模式。无需密钥就能浏览和全局搜索。`download_model.py` 是提前下载模型；不执行时，第一次语义搜索会下载相同的固定版本。

### 使用本地照片和自己的密钥

1. 将自己的照片放入 `data/`；或把原项目的 `data/` 和 `metadata_cache.json` 复制到本地。**不要提交这些私人文件。**
2. 如果使用私有元数据而非已授权的公开演示集，开启明确的本地模式。Windows PowerShell：

```powershell
$env:ADAPHOTORET_LOCAL_MODE = "1"
# 避免将真实密钥写进命令历史：先在隐藏输入中输入，再设置进当前进程环境。
$secure = Read-Host "Your DeepSeek API key" -AsSecureString
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new("", $secure).Password
python -m streamlit run streamlit_ui.py
```

macOS / Linux Bash：

```bash
export ADAPHOTORET_LOCAL_MODE=1
read -r -s -p 'Your DeepSeek API key: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
python -m streamlit run streamlit_ui.py
```

也可以仍在网页侧边栏输入自己的密钥。本地模式把会话和图库保存到 `.local_data/`，默认读取 `metadata_cache.json`。可设置 `ADAPHOTORET_METADATA_FILE` 指向另一份本地元数据，`ADAPHOTORET_MODEL_DIR` 指向完整模型目录。

`.env.example` 只是变量模板，应用**不会自动读取 `.env`**。不要把真实密钥写进代码、提交 `.env`、或提交 `.streamlit/secrets.toml`。在公共托管环境不要设置 `ADAPHOTORET_LOCAL_MODE=1`，也不要配置作者的 API 密钥。

### 生成照片元数据（仅本地，需自己的 DashScope 密钥）

把 `DASHSCOPE_API_KEY` 按上述隐藏输入方式放入当前终端环境，然后：

```bash
python auto_label.py --consent-external-processing
```

这会将选定照片的图像内容发送给 DashScope 并可能产生费用；请先确认有权处理这些照片。HEIC/HEIF 大小写扩展名均支持，发送前转换成真正的 JPEG。默认不发送 GPS、不执行逆地理编码。

只有明确需要位置处理时才加 `--enable-geocoding`：此时会将 EXIF GPS 发给高德（设置 `AMAP_API_KEY` 时）或 Nominatim，并保存本地位置元数据。使用前检查相应服务条款、额度及照片授权。`BAILIAN_VL_MODEL` 可选；默认模型名见 `auto_label.py`。

## Streamlit Community Cloud 部署

1. 使用已有应用的管理页，核对仓库 `ushi0405/adaphotoret_01`、分支和入口 `streamlit_ui.py`；更新已关联分支可能自动重新部署。
2. 在 **Advanced settings** 中手动选择 Python **3.12**，不要直接沿用平台默认版本。当前固定的 CPU Torch 2.8.0 没有 Python 3.14 对应的安装包；若误用 3.14，需要重新部署并选择 3.12，单纯重启不会改变 Python 版本。根目录 `requirements.txt` 是唯一默认依赖入口；不放入根目录 `environment.yml` 与其争夺优先级。
3. 不配置作者的 API secrets、仅发布本次用户授权的完整照片及元数据，不上传聊天历史或密钥；保留默认公共模式。
4. 检查应用构建日志、无密钥首页、首次语义搜索、两个会话隔离及 BYOK 提示后再分享链接。

已有应用链接只有实际部署验证后才可视为有效；此 README 不声称任何特定 URL 当前可用。登录、GitHub 授权、休眠唤醒和资源限制由 Streamlit 平台管理。

官方说明：[部署](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)、[依赖](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)、[Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)。

## 其他原有入口

```bash
python AdaphotoRet_run.py                       # Gradio，本地入口
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
python -m uvicorn api:app --host 127.0.0.1 --port 8000  # 兼容旧前端的 /api/*
python convert_heic.py data                     # 可选转换，保留原文件、不覆盖已有 JPG
```

`bolt_ui/` 保留旧 React/Vite 界面供本地开发；**公共演示使用 Streamlit**。旧前端默认访问 `http://127.0.0.1:8000`，可由 `VITE_API_BASE_URL` 配置。旧前端的界面示例数据不是检索结果；不要把原型当作已部署服务。API 默认不消费环境密钥；请求级解释功能须同时提供自己的 `X-DeepSeek-API-Key` 和 `X-Allow-External-Processing: true`。该可选 API 未添加生产级身份认证/限流，不建议裸露到公网。

## 项目结构

- `streamlit_ui.py`：主界面、全局/精细搜索、会话、标签、图库。
- `AdaphotoRet_run.py` / `attributes.py`：真实语义召回与规则重排。
- `llm_explainer.py`：请求/会话级 DeepSeek 客户端与结果说明。
- `auto_label.py`：本地照片标注和显式授权的位置处理。
- `evaluate.py`：P@K/MRR。需自行提供 `test_queries.json`；不调用 LLM 解释。
- `feedback.py`：保留原有反馈辅助模块；主界面未接入该旧模块。
- `metadata.demo.json` / `data/`：完整 66 张授权演示照片及原始检索元数据，详见 [数据来源](DATA_PROVENANCE.md)。
- `tests/`：离线隐私/兼容性/界面测试；不调用真实收费 API。

## 验证与边界

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q .
```

离线测试不代表真实账户/计费/API 可用，也不是检索精度评测。模型来自 Apache-2.0 上游，固定版本与使用说明见 [MODEL.md](MODEL.md)。本项目未擅自为用户原创代码或照片添加新的许可。

安全清理不会抹除 Git 历史中的旧密钥；曾公开的密钥应先在对应服务撤销/轮换，历史重写需独立评估和授权。仅删除当前文件或加入 `.gitignore` 不足以撤销泄露。

### 精细搜索 API 的多轮线索

`POST /api/search` 兼容原有 `{"query":"猫"}`；也可发送 `{"query":"在海边", "messages":[{"role":"user","content":"猫"},{"role":"assistant","content":"是黑色的吗？"}]}`。只合并 `user` 内容和当前 `query`，不存储服务端对话。新目标传空 `messages`；勿发送其他目标的历史。全局搜索仍为单次查询。
