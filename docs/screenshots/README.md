# 界面截图

- `dashboard.png` —— 已就位：上传 `data/toxic_data.csv`、目标列选 `target`、点「运行诊断」后的首屏。
- `report.png` —— 可选：把页面继续向下滚动，截「质量指标概览」两个面板（缺失 / 泄漏）。

重新截取的步骤：

```bash
streamlit run app.py
```

1. 浏览器打开 <http://localhost:8501>；
2. 上传 `data/toxic_data.csv`，目标列选 `target`，点「运行诊断」；
3. 首屏保存为 `dashboard.png`，向下滚动的两个面板块保存为 `report.png`。

建议宽度 ≥ 1440px，PNG 格式。配置了 `DEEPSEEK_API_KEY` 时面板上方是模型生成的洞察，
不配 Key 时会显示规则引擎生成的同一结构报告（截图里是配了 Key 的效果）。
