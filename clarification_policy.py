"""Bounded dialogue control; assistant questions are never retrieval evidence."""
import re
from difflib import SequenceMatcher

from photo_query import build_photo_query, is_unclear_reply, requests_direct_results


MAX_CLARIFICATIONS = 3
MAX_QUESTION_LENGTH = 40


def _compact(text):
    return re.sub(r"[\s，,。.!！?？;；:：、]+", "", text or "")


def clarification_questions(history):
    """Count actual assistant clarification turns, including pre-policy histories.

    No global/session counter: switching targets naturally switches the budget.
    The initial greeting and result messages with feedback questions do not count.
    """
    questions, seen_user = [], False
    for message in history:
        if message.get("role") == "user":
            seen_user = True
        if message.get("role") != "assistant" or not seen_user:
            continue
        kind = message.get("kind")
        if kind == "clarification":
            questions.append(message.get("content", ""))
        elif kind is None and not (message.get("images") or message.get("report")):
            content = message.get("content", "")
            if re.search(r"[？?]|(?:请)?(?:告诉我|补充|描述一下|确认一下)|还有什么|哪(?:里|儿|个)|是否", content):
                questions.append(content)
    return questions


def must_return_results(history, user_message, confident=False):
    questions = clarification_questions(history)
    return (confident or len(questions) >= MAX_CLARIFICATIONS
            or requests_direct_results(user_message) or bool(questions and is_unclear_reply(user_message)))


# Conservative topic aliases catch paraphrased repeats, not just identical text.
# Rejecting a question always falls back to real candidates, never another retry.
QUESTION_TOPICS = {
    "color": r"颜色|什么色|哪种色|黑色|白色|红色|橘色|黄色|蓝色|绿色",
    "place": r"哪里|哪儿|地点|位置|哪个城市|哪座城市|哪座山|什么地方",
    "time": r"时间|时候|几点|哪年|哪天|白天|夜晚|晚上|季节",
    "count": r"几只|几个人|几位|数量|多少|一只|两只",
    "scene": r"室内|室外|户外|背景|场景|海边|屋顶|公园|餐桌",
    "food": r"食材|饭菜|菜品|主食|配菜",
}


def _distinguishes_candidates(question, candidates, user_clues):
    """Require an observable difference, not another request to confirm a miss."""
    descriptions = [str(c.get("scene", "")) + " " + str(c.get("description", ""))
                    for c in candidates]
    if len(set(descriptions)) < 2:
        return False
    # A quoted detail must occur in some, but not all, candidates. Ignore generic
    # question phrasing; no details are copied into the user-owned photo query.
    detail = re.sub(r"照片|图片|画面|记得|您|你|是在|还是|是否|能看到|可以看到|什么|哪个|哪种|请问", "", question)
    words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", detail)
    distinguishing = set()
    for word in words:
        for index in range(len(word) - 1):
            fragment = word[index:index + 2]
            hits = sum(fragment in description for description in descriptions)
            if 0 < hits < len(descriptions):
                distinguishing.add(fragment)
    return bool(distinguishing) and not any(fragment in user_clues for fragment in distinguishing)


def concise_question(response, history, candidates):
    """Only accept one short, nonrepeated clarification action; otherwise search.

    Freeform/回答 replies cannot smuggle an uncounted question past the cap.
    Validity is deliberately conservative, not a claim of semantic understanding.
    """
    if not isinstance(response, str) or response.count("【行动】") != 1:
        return None
    action = response.split("【行动】", 1)[1].strip()
    if not action.startswith("提问："):
        return None
    question = action[len("提问："):].strip()
    if (not question or len(question) > MAX_QUESTION_LENGTH
            or len(re.findall(r"[？?]", question)) != 1
            or not question.endswith(("？", "?"))
            or re.search(r"[\n。！!；;，,、【】]|另外|以及|同时|然后|还有", question)
            or len(re.findall(r"吗|么|呢", question)) > 1):
        return None
    # Avoid compound generic interrogations even when the model uses one '?'.
    if len(re.findall(r"哪里|哪儿|什么|多少|几只|几点|哪个|哪种", question)) > 1:
        return None
    normalized = _compact(question)
    topics = {key for key, pattern in QUESTION_TOPICS.items() if re.search(pattern, question)}
    for previous in clarification_questions(history):
        previous_topics = {key for key, pattern in QUESTION_TOPICS.items() if re.search(pattern, previous)}
        if (topics & previous_topics
                or SequenceMatcher(None, normalized, _compact(previous)).ratio() >= 0.65):
            return None
    return question if _distinguishes_candidates(question, candidates, build_photo_query(history)) else None
