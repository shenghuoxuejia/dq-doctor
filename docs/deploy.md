# 部署到 Streamlit Community Cloud（免费，约 5 分钟）

招聘方点开就能用，比让他们 clone 仓库的转化率高很多。

## 前置

1. 仓库已推到 GitHub：`shenghuoxuejia/dq-doctor`（公开仓库）。
2. 有一个 GitHub 账号即可，无需信用卡。

## 步骤

1. 打开 <https://share.streamlit.io>，用 GitHub 登录。
2. 点 **New app** → 选 **Deploy a public app from GitHub**。
3. 填写：
   - Repository：`shenghuoxuejia/dq-doctor`
   - Branch：`main`
   - Main file path：`app.py`
   - App URL：自定义一个，例如 `dq-doctor`
4. 点 **Deploy**，等待 1–3 分钟构建。
5. （可选）给大模型洞察配置密钥：进入 app → **Settings → Secrets**，粘贴：

   ```toml
   DEEPSEEK_API_KEY = "sk-你的key"
   ```

   保存后 app 会自动重启。**不配也能用**：缺少 Key 时会走确定性规则引擎，
   界面结构与配置了 Key 时一致，只是洞察文字由规则生成而非大模型。

## 拿到 URL 之后

把徽章加到 README 顶部（替换占位）：

```markdown
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dq-doctor.streamlit.app)
```

## 已知注意事项

- 云端是 Linux 环境，`benchmarks/` 里的中文字体会回退到 Noto Sans CJK（脚本已做候选字体列表）；
  界面本身不依赖字体。
- 上传文件上限约 200 MB（Streamlit 默认），示例数据最大只有 57 KB，不受影响。
- 免费实例空闲会自动休眠，首次访问需要唤醒几十秒，属正常现象。
