import re
import random
import pandas as pd
from tqdm import tqdm


system_message_templates = (
    "你现在是一个病人，下面会描述你的病情，user是一名医生，你们会因为你的病情展开一段对话\n\n你的病情：\n{state}"
)


def extract_messages(row):
    dialogue = row["instruction"]
    output = row["output"]
    messages = []
    pattern = re.compile(
        r"<User>(.*?)(?=<MedChat>|$)|<MedChat>(.*?)(?=<User>|$)", re.DOTALL
    )

    matches = pattern.finditer(dialogue)
    for _, match in enumerate(matches):
        user_message = match.group(1)
        medchat_message = match.group(2)

        # if turn == 0:
        #     messages.append(
        #         {
        #             "role": "system",
        #             "content": system_message_templates.format_map(
        #                 {"state": user_message.strip()}
        #             ),
        #         }
        #     )
        # else:
        if user_message:
            messages.append({"role": "assistant", "content": user_message.strip()})
        elif medchat_message:
            messages.append({"role": "user", "content": medchat_message.strip()})
    messages.append({"role": "user", "content": output})
    system_messages = messages[0]
    # 要抽取带病情描述的数据
    if "MedChat" in system_messages["content"] or len(system_messages["content"]) < 10:
        return []

    system_messages["role"] = "system"
    system_messages["content"] = system_message_templates.format_map(
        {"state": system_messages["content"].strip()}
    )
    return [system_messages]


if __name__ == "__main__":
    # # 测试
    # dialogue = "<User>您好，我是一位93岁的女性患者，最近反复出现眩晕症状，已经持续了10多天了。之前也曾复诊过，但仍然感到头晕。我有高血压和安装起搏器的既往史。<MedChat>您好，很高兴见到您。您的眩晕症状是反复发作的吗？除了头晕之外，还有其他不适感吗？<User>是的，眩晕症状确实是反复发作的。除了头晕，我没有其他不适感。<MedChat>您有偏头痛的病史吗？还有晕车的情况吗？<User>我有头晕的病史，但没有偏头痛。至于晕车，我没有这个情况。<MedChat>您有家族史方面的了解吗？是否有相关疾病的遗传倾向？<User>很抱歉，我不记得有关家族史的情况了。我母亲早逝，但具体原因我不太清楚。<MedChat>好的，那您目前有在使用任何药物吗？还有睡眠情况如何？<User>我目前没有使用任何药物。至于睡眠情况，我睡得还不错。<MedChat>根据您的症状和问诊信息，我初步怀疑您可能患有良性阵发性位置性眩晕（BPPV）和右后半规管异常（RPC）。这只是初步诊断，仅供参考，请遵医嘱。<User>谢谢你的诊断，我会按照医嘱进行治疗的。"
    # messages = extract_messages(dialogue)

    # # 打印结果
    # for message in messages:
    #     print(f"{message['role']}: {message['content']}")

    test_df = pd.read_json("./data/instruct_v3.json", lines=True)
    dialogues = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting"):
        messages = extract_messages(row)
        # 只要带病人信息的部分，多余的不要，需要模拟来生成
        if not messages:
            continue
        dialogues.append({"id": idx, "num_turns": 1, "history": messages})
    dialogues = random.sample(dialogues, k=2000)  # 采样2k进行评估

    pd.DataFrame(dialogues).to_json(
        "./data/instruct_chatgpt_messages.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )
