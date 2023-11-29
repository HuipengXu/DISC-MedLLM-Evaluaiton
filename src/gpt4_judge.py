import re
import os
import json
import pandas as pd
from tqdm import tqdm
from loguru import logger
from collections import defaultdict

from openai import OpenAI

from gpt4_judge_prompt import JUDGE_PROMPT

client = OpenAI(api_key="")


def extract_ratings(input_string):
    # 定义正则表达式模式
    pattern = re.compile(
        r"Proactivity:(\d)\s+Accuracy:(\d)\s+Helpfulness:(\d)\s+Linguistic Quality:(\d)"
    )

    # 在输入字符串中搜索匹配项
    match = pattern.search(input_string)

    # 将匹配项转换为字典
    if match:
        ratings_dict = {
            "Proactivity": int(match.group(1)),
            "Accuracy": int(match.group(2)),
            "Helpfulness": int(match.group(3)),
            "Linguistic Quality": int(match.group(4)),
        }
        return ratings_dict
    else:
        return None


def main(conversation_path):
    conversation_df = pd.read_json(conversation_path)
    ratings_dict = defaultdict(list)
    for _, row in tqdm(
        conversation_df.iterrows(), desc="Judging", total=len(conversation_df)
    ):
        conversation = row.conversation
        system_message = JUDGE_PROMPT.format_map({"conversation": conversation})
        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": conversation},
        ]
        response = client.chat.completions.create(
            model="gpt-4-1106-preview", messages=messages
        )
        results = response.choices[0].message.content.strip()
        ratings = extract_ratings(results)
        if ratings is None:
            logger.info(f'对话{row["id"]}评分结果解析错误！')
        else:
            for metric, score in ratings.items():
                ratings_dict[metric].append(score)
    avg_ratings = {
        metric: sum(scores) / len(scores) for metric, scores in ratings_dict.items()
    }
    ratings_path = os.path.join(os.path.dirname(conversation_path), "ratings.json")
    with open(ratings_path, "w", encoding="utf8") as f:
        json.dump(avg_ratings, f, ensure_ascii=False)


if __name__ == "__main__":
    main("data/medchat_chatgpt_multiturn_dialogue/conversations.json")
