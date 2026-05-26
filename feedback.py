# feedback_ui.py
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st

# 反馈记录文件
FEEDBACK_LOG_FILE = "feedback_log.json"

def load_feedback_log() -> List[Dict]:
    """加载已有的反馈日志"""
    if os.path.exists(FEEDBACK_LOG_FILE):
        with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_feedback_entry(entry: Dict):
    """追加一条反馈到日志"""
    log = load_feedback_log()
    log.append(entry)
    with open(FEEDBACK_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def render_feedback_section(
    top_results: List[Dict],
    user_query: str,
    system_best_index: int = 0
) -> Optional[Dict]:
    """
    在 Streamlit 页面中渲染反馈收集区域。

    Parameters
    ----------
    top_results : list of dict
        每个元素应包含:
        - 'img_path': 图片路径
        - 'score': 最终得分
        - 'trace': [(规则名, 分数变化, 解释), ...]
    user_query : str
        用户原始的查询文本
    system_best_index : int
        系统认为的最佳图片在 top_results 中的下标（通常为 0）

    Returns
    -------
    dict or None
        如果用户成功提交反馈，返回:
        {
            "query": user_query,
            "system_best": top_results[system_best_index]["img_path"],
            "user_best": top_results[user_choice_index]["img_path"],
            "mismatch_reasons": [选中的不匹配加分项名称],
            "comment": 用户手动输入的补充说明,
            "timestamp": 时间戳
        }
        否则返回 None。
    """
    st.markdown("---")
    st.subheader("📝 您的反馈帮助我们更懂你")

    # 1. 选择你心目中的最佳图片
    st.markdown("**① 请点击选择你心中最匹配的照片：**")
    cols = st.columns(len(top_results))
    choice_index = st.session_state.get("feedback_choice", system_best_index)

    for i, res in enumerate(top_results):
        with cols[i]:
            st.image(res["img_path"], use_container_width=True)
            # 单选框，同一组 key 保证互斥
            if st.button(f"🥇 选这张", key=f"sel_{i}"):
                st.session_state.feedback_choice = i
                choice_index = i
                st.rerun()

    # 2. 如果用户选择和系统最佳不同，展示可能不匹配的加分项
    mismatch_reasons = []
    if choice_index != system_best_index:
        st.warning("您选择的最佳照片与系统判定不同，我们将优化理解。")
        st.markdown("**② 是哪一项匹配规则不准确？**（可多选）")
        
        # 取出系统最佳图片的加分项（delta > 0）
        system_best_trace = top_results[system_best_index].get("trace", [])
        positive_rules = [
            (name, evidence)
            for name, delta, evidence in system_best_trace
            if delta > 0
        ]
        rule_labels = [f"{name}: {evidence}" for name, evidence in positive_rules]

        if rule_labels:
            selected_labels = st.multiselect(
                "勾选您觉得不合理的加分项：",
                options=rule_labels,
                default=[],
            )
            mismatch_reasons = [
                label.split(":")[0].strip() for label in selected_labels
            ]
        else:
            st.info("系统最佳图片没有明确加分项，您可以手动描述问题。")

        # 3. 额外的手动说明（可选）
        comment = st.text_input("③ 补充说明（可选）", placeholder="例如：这张照片的情绪不对，风景不是我要的")
    else:
        comment = ""
        # 如果用户选择的就是系统最佳，也可以给出肯定反馈
        st.success("✅ 系统最佳与您的选择一致，感谢肯定！")
        # 这里我们可以记录正向反馈（可选）
        if st.button("确认提交正向反馈"):
            entry = {
                "query": user_query,
                "system_best": top_results[system_best_index]["img_path"],
                "user_best": top_results[system_best_index]["img_path"],
                "mismatch_reasons": [],
                "comment": "用户认为系统最佳正确",
                "timestamp": datetime.now().isoformat()
            }
            save_feedback_entry(entry)
            st.toast("正向反馈已记录，谢谢！")
            return None

    # 4. 提交按钮
    if st.button("📬 提交反馈"):
        entry = {
            "query": user_query,
            "system_best": top_results[system_best_index]["img_path"],
            "user_best": top_results[choice_index]["img_path"],
            "mismatch_reasons": mismatch_reasons,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        save_feedback_entry(entry)
        st.toast("反馈已记录，感谢您的帮助！")
        return entry
    return None