"""User-owned photo clues. No assistant text, credentials, or shared session state.

One conversation is one photo-search target. Start a new conversation (or send
an empty API history) for a different target. Never infer a reset from prose.
"""
import re


# Compare complete tokens/utterances, never remove substrings (请柬/请客 stay intact).
DIALOGUE_WORDS = frozenset({
    "请", "请问", "告诉", "告诉我", "准确", "不够", "谢谢", "你好", "您好",
    "帮", "帮忙", "麻烦", "想", "想要", "需要", "找到", "寻找", "查找", "搜索", "检索",
    "找", "要", "给", "匹配", "可以", "继续", "描述", "查询",
    "回答", "提供", "调整", "记得", "记不清", "不知道", "不确定",
})
NO_CLUE_UTTERANCES = DIALOGUE_WORDS | frozenset({
    "好的", "好", "是的", "对", "对的", "不是", "不对", "不准确", "不够准确",
    "谢谢你", "谢谢您", "请告诉我", "告诉我吧", "请继续", "继续", "记不清了",
    "记不清楚", "我记不清了", "我不知道", "我不确定", "不知道了", "没有了",
    "我记错了", "更正", "纠正", "重新描述",
})


def is_no_clue(text):
    compact = re.sub(r"[\s，,。.!！?？;；:：、]+", "", text)
    return not compact or compact in NO_CLUE_UTTERANCES


def build_photo_query(messages=(), current_user_message=None):
    """Combine only explicitly user-authored turns from the supplied target.

    The current input is separate so callers never drop it when taking a history
    snapshot. Consecutive identical turns are idempotent; nonconsecutive repeats
    retain their order so a later correction can itself be corrected.
    """
    turns = [m.get("content", "") for m in messages
             if isinstance(m, dict) and m.get("role") == "user"]
    if current_user_message is not None:
        turns.append(current_user_message)
    clues = []
    for turn in turns:
        if not isinstance(turn, str) or is_no_clue(turn):
            continue
        turn = turn.strip()
        if not clues or clues[-1] != turn:
            clues.append(turn)
    return "\n".join(clues)


def positive_photo_clues(text, tokenize):
    """Resolve explicit clause-level corrections before attributes are parsed.

    Supports e.g. 黑色的猫 / 不是黑色，是白色 and 不在海边，是在公园.
    Negated clauses are not positive evidence. Retraction matches whole token
    sequences, not substrings: rejecting 猫 must not damage 猫头鹰. This is not a
    general natural-language/Boolean-query parser; ambiguous replies add no fact.
    """
    active = []
    for clause in re.split(r"[\n，,。；;！!？?]+", text):
        clause = clause.strip()
        if is_no_clue(clause):
            continue
        # Only standalone correction clauses. Attributive descriptions such as
        # 没有戴帽子的狗 / 不在海边的猫 must not be discarded as whole negatives.
        negative = re.match(r"^(?:不是|并非|不在)([^的]+?)(?:的)?$", clause)
        if negative:
            rejected, *replacement = re.split(r"(?:而是|而是在)", negative.group(1), maxsplit=1)
            rejected_tokens = [w for w, _ in tokenize(rejected) if w.strip()]
            if rejected_tokens:
                updated = []
                for previous in active:
                    tokens = list(tokenize(previous))
                    words = [w for w, _ in tokens]
                    kept, i = [], 0
                    while i < len(words):
                        if words[i:i + len(rejected_tokens)] == rejected_tokens:
                            i += len(rejected_tokens)
                        else:
                            kept.append(words[i])
                            i += 1
                    updated.append("".join(kept))
                active = updated
            if replacement:
                active.append(replacement[0])
        else:
            active.append(clause)
    return "\n".join(dict.fromkeys(part for part in active if part.strip()))
