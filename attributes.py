# attributes.py
from typing import Dict, List

# 人物属性 
def extract_person_attributes(text: str) -> Dict[str, List[str]]:
    attr = {"count": [], "gender": [], "ethnicity": []}
    text_lower = text.lower()

    count_map = {
        "一个": ["一个", "单个", "一人", "一名", "一位", "单人", "一只", "一头", "一条"],
        "两个": ["两个", "两人", "一对", "二位", "两名", "双人","两只","两件","两头","两条"],
        "三个": ["三个", "三人", "三位", "三名","三只","三头","三条","三件"],
        "一群": ["一群", "一堆", "多人", "人们", "一群人","多只","多头","多件","多条"],
    }
    for key, phrases in count_map.items():
        if any(p in text for p in phrases):
            attr["count"].append(key)

    if any(w in text for w in ["男", "男性", "男子", "男人", "boy", "man", "male"]):
        attr["gender"].append("男")
    if any(w in text for w in ["女", "女性", "女子", "女人", "girl", "woman", "female"]):
        attr["gender"].append("女")

    western = ["外国", "外国人", "西方", "欧美", "欧洲", "美国", "英国", "法国", "德国", "白种人", "白人", "caucasian", "western", "european", "american"]
    asian = ["亚洲", "中国", "日本", "韩国", "黄种人", "asian", "chinese", "japanese", "korean"]
    black = ["非洲", "黑人", "黑种人", "african", "black"]
    if any(w in text_lower for w in western):
        attr["ethnicity"].append("西方/白种人")
    elif any(w in text_lower for w in asian):
        attr["ethnicity"].append("亚洲/黄种人")
    elif any(w in text_lower for w in black):
        attr["ethnicity"].append("非洲/黑种人")
    elif "外国" in text:
        attr["ethnicity"].append("外国")

    return attr


def expand_person_attributes(attr: Dict[str, List[str]]) -> List[str]:
    expansions = []
    if "男" in attr["gender"]:
        expansions.extend(["男性", "男子", "man", "male"])
    if "女" in attr["gender"]:
        expansions.extend(["女性", "女子", "woman", "female"])
    if "一个" in attr["count"]:
        expansions.extend(["一个人", "单人", "single person"])
    elif "两个" in attr["count"]:
        expansions.extend(["两个人", "双人", "two people", "couple"])
    elif "一群" in attr["count"]:
        expansions.extend(["一群人", "多人", "crowd", "group of people"])
    if "西方/白种人" in attr["ethnicity"]:
        expansions.extend(["西方人", "白人", "Caucasian", "European", "American"])
    elif "亚洲/黄种人" in attr["ethnicity"]:
        expansions.extend(["亚洲人", "黄种人", "Asian", "Chinese", "Japanese"])
    elif "非洲/黑种人" in attr["ethnicity"]:
        expansions.extend(["非洲人", "黑人", "African", "Black person"])
    elif "外国" in attr["ethnicity"]:
        expansions.extend(["外国人", "foreigner"])
    return expansions


# ────────── 宠物属性 ──────────
PET_ANIMAL_TYPES = {"猫", "猫咪", "猫猫", "狗", "狗狗", "犬", "兔子", "鸟", "鱼", "鹦鹉", "仓鼠"}
PET_COLORS = {"白色", "黑色", "棕色", "黄色", "金色", "灰色", "橙色", "奶油色", "米黄色", "浅棕色"}
PET_ACTIONS = {"跑", "跳", "趴", "卧", "坐", "站立", "咬", "叼", "追逐", "挠", "叫", "看镜头", "证件照", "合照", "嗅花"}
PET_ENVS = {"草地", "草坪", "沙滩", "公园", "室内", "地毯", "沙发", "床上", "浴室", "户外", "街道", "屋顶", "溪流", "田野", "花丛"}

# 品种归一化映射（口语简称 → 归一化品种名）
BREED_NORMALIZATION = {
    "蓝白猫": "英短蓝白",
    "蓝白": "英短蓝白",
    "蓝猫": "英短蓝猫",
    "英短": "英短蓝猫",
    "美短": "美国短毛猫",
    "布偶": "布偶猫",
    "暹罗": "暹罗猫",
    "加菲": "异国短毛猫",
    "金毛": "金毛寻回犬",
    "拉布拉多": "拉布拉多犬",
    "泰迪": "泰迪犬",
    "柯基": "柯基犬",
    "柴犬": "柴犬",
    "橘猫": "橘猫",
    "比格": "比格犬",
    "秋田": "秋田犬",
    "梅花鹿": "梅花鹿",
}

# 归一化品种名 → 所有可能的同义表达（用于匹配图片品种字段）
BREED_ALIASES = {
    "英短蓝白": ["英短蓝白", "英国短毛猫", "British Shorthair", "英短"],
    "英短蓝猫": ["英短蓝猫", "英国短毛猫", "British Shorthair", "英短"],
    "美国短毛猫": ["美国短毛猫", "American Shorthair", "美短"],
    "暹罗猫": ["暹罗猫", "Siamese"],
    "布偶猫": ["布偶猫", "Ragdoll"],
    "异国短毛猫": ["异国短毛猫", "Exotic Shorthair", "加菲"],
    "金毛寻回犬": ["金毛寻回犬", "Golden Retriever", "金毛"],
    "拉布拉多犬": ["拉布拉多犬", "Labrador Retriever", "拉布拉多"],
    "泰迪犬": ["泰迪犬", "泰迪", "Teddy"],
    "柯基犬": ["柯基犬", "Corgi", "柯基"],
    "柴犬": ["柴犬", "Shiba Inu"],
    "橘猫": ["橘猫", "orange cat"],
    "比格犬": ["比格犬", "Beagle", "比格"],
    "秋田犬": ["秋田犬", "Akita", "秋田"],
    "梅花鹿": ["梅花鹿", "Sika deer"],
}


def extract_pet_attributes(text: str) -> Dict[str, List[str]]:
    attr = {
        "animal_type": [],
        "breed": [],
        "coat_color": [],
        "action": [],
        "environment": [],
        "count": [],
        "life_stage": []
    }
    # 动物类型
    for animal in PET_ANIMAL_TYPES:
        if animal in text:
            if animal in {"猫", "猫咪", "猫猫"}:
                attr["animal_type"].append("猫")
            elif animal in {"狗", "狗狗", "犬"}:
                attr["animal_type"].append("狗")
            else:
                attr["animal_type"].append(animal)

    # 品种提取（先利用归一化映射，避免分词错误）
    for breed_key, normalized in BREED_NORMALIZATION.items():
        if breed_key in text:
            attr["breed"].append(normalized)

    # 毛色
    for color in PET_COLORS:
        if color in text:
            attr["coat_color"].append(color)

    # 动作
    for act in PET_ACTIONS:
        if act in text:
            attr["action"].append(act)

    # 环境
    for env in PET_ENVS:
        if env in text:
            attr["environment"].append(env)

    # 数量
    if any(w in text for w in ["一只", "一条", "单个", "一个"]):
        attr["count"].append("一个")
    elif any(w in text for w in ["两只", "两条", "一对"]):
        attr["count"].append("两个")
    elif any(w in text for w in ["三只", "三条"]):
        attr["count"].append("三个")
    elif any(w in text for w in ["一群", "多只", "一堆"]):
        attr["count"].append("一群")

    # 年龄
    if any(w in text for w in ["幼年", "幼猫", "幼犬", "幼崽", "baby", "kitten", "puppy"]):
        attr["life_stage"].append("幼年")
    elif any(w in text for w in ["成年", "adult"]):
        attr["life_stage"].append("成年")
    elif any(w in text for w in ["老年", "老猫", "老狗", "senior"]):
        attr["life_stage"].append("老年")

    return attr


def expand_pet_attributes(attr: Dict[str, List[str]]) -> List[str]:
    expansions = []
    for vals in attr.values():
        expansions.extend(vals)
    return list(set(expansions))