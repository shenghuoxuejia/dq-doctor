import os
os.environ["PYTHONUTF8"] = "1"
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from .state import DQState

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.1
)

# ===== 修改在这里：加入防幻觉的严格规则 =====
SYNTH_PROMPT = """你是一个资深的数据科学专家。下面是自动化工具对一份建模数据集的结构化诊断结果。

诊断结果 JSON:
{diagnoses}

请完成以下任务：
1. 按“对建模决策的威胁程度”从高到低对各个诊断进行排序（例如 ["leakage", "missingness"]）。
2. 针对最严重的问题写一段解释。 

⚠️ 极其重要的规则：
如果诊断结果中显示“数据完美，无任何缺失值”、“未发现目标相关的信息性缺失”、“0个单特征预测能力异常”、“未发现明显的数据泄漏”，这代表**数据质量极其优秀**。
此时，你**绝对不可以虚构或脑补不存在的风险**。请直接表扬该数据质量很高，并提醒用户注意过拟合或数据分布漂移等正常建模注意事项即可。不要为了凑字数而强行编造“灾难”！

请严格输出以下 JSON 格式（不要包含任何 Markdown 代码块符号）：
{{"severity_order": ["排第一的模块名", "排第二的模块名"], "top_insight": "你的专业解释..."}}
"""

def synthesize_node(state: DQState) -> dict:
    diagnoses = state.get("diagnoses", {})
    
    if not diagnoses:
        return {"messages": [("ai", "未发现任何明显的数据质量问题。")]}

    prompt = SYNTH_PROMPT.format(
        diagnoses=json.dumps(diagnoses, ensure_ascii=False, indent=2)
    )
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        parsed = json.loads(content)
        severity_order = parsed.get("severity_order", [])
        top_insight = parsed.get("top_insight", "分析完成。")
    except Exception as e:
        severity_order = list(diagnoses.keys())
        top_insight = f"LLM 分析失败，错误信息: {str(e)}"
        
    return {
        "severity_order": severity_order,
        "messages": [("ai", top_insight)] 
    }