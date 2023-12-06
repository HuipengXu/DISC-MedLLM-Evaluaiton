import re
import os
import json
import time
import pandas as pd
from tqdm import tqdm
from loguru import logger
from collections import defaultdict

from openai import OpenAI

from gpt4_judge_prompt import JUDGE_PROMPT
from doctors import openai_api_key

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
        return input_string


def parse_gpt4_reponse(ratings_dir):
    ratings_dict = defaultdict(list)
    error_ids = []
    for file in os.listdir(ratings_dir):
        with open(os.path.join(ratings_dir, file), "r", encoding="utf8") as f:
            response = json.load(f)
            id_ = response["id"]
            rating_str = response["rating_str"]
            ratings = extract_ratings(rating_str)
            if isinstance(ratings, str):
                error_ids.append(id_)
            else:
                for metric, score in ratings.items():
                    ratings_dict[metric].append(score)
    avg_ratings = {
        metric: sum(scores) / len(scores) for metric, scores in ratings_dict.items()
    }
    ratings_path = os.path.join(ratings_dir, "ratings.json")
    with open(ratings_path, "w", encoding="utf8") as f:
        json.dump(avg_ratings, f, ensure_ascii=False)

    with open(f"{ratings_dir}/parse_error_id.txt", "w", encoding="utf8") as f:
        f.write("\n".join(map(str, error_ids)))


def judge(conversation_path):
    conversation_df = pd.read_json(conversation_path, orient="records")
    ratings_dir = os.path.join(os.path.dirname(conversation_path), "ratings")
    os.makedirs(ratings_dir, exist_ok=True)
    processed_ids = {
        file.split(".")[0].split("_")[-1] for file in os.listdir(ratings_dir)
    }
    for _, row in tqdm(
        conversation_df.iterrows(), desc="Judging", total=len(conversation_df)
    ):
        if str(row["id"]) in processed_ids:
            print(f'conversation {row["id"]} has been processed, skip it')
            continue

        conversation = str(row.conversation)
        messages = [
            {
                "role": "system",
                "content": JUDGE_PROMPT,
            },
            {"role": "user", "content": conversation},
        ]
        response = client.chat.completions.create(
            model="gpt-4", messages=messages
        )
        results = response.choices[0].message.content.strip()
        with open(
            f'{ratings_dir}/rating_str_{row["id"]}.json', "w", encoding="utf8"
        ) as f:
            json.dump({"id": row["id"], "rating_str": results}, f, ensure_ascii=False)
            
        time.sleep(30)
    return ratings_dir


def main(conversation_path):
    ratings_dir = judge(conversation_path)
    parse_gpt4_reponse(ratings_dir)


if __name__ == "__main__":
    main("data/gpt-3.5-turbo-1106_chatgpt_multiturn_dialogue/conversations.json")
