import os
import json
from tqdm import tqdm
import pandas as pd

import openai

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from doctors import *


client = openai.OpenAI(api_key="")
# client = None


def gpt3_interaction(chat_history):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106", messages=chat_history
    )
    response_text = response.choices[0].message.content.strip()
    chat_history += [{"role": "assistant", "content": response_text}]
    return chat_history


def gpt3_interaction_test(chat_history):
    response_text = "测试"
    chat_history += [{"role": "assistant", "content": response_text}]
    return chat_history


def chat_with_chatgpt(doctor):
    model_dir = doctors_path[doctor]
    print(model_dir)

    dataset = pd.read_json("./data/instruct_chatgpt_messages.jsonl", lines=True).head(3)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, legacy=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    chat_func = doctors_call[doctor]

    save_dir = f"data/{doctor}_chatgpt_multiturn_dialogue"
    for _, row in tqdm(dataset.iterrows(), desc="Chating", total=len(dataset)):
        gpt_chat_history = row.history

        # （这里只是示例，你可以根据需要调整）
        num_turns = 2
        for _ in range(num_turns):
            gpt_chat_history = chat_func(
                model,
                tokenizer,
                history=gpt_chat_history,
            )

            # 与 GPT-3.5 进行对话
            gpt_chat_history = gpt3_interaction(gpt_chat_history)
            # gpt_chat_history = gpt3_interaction_test(gpt_chat_history)

        # 对话以医生结束
        gpt_chat_history = chat_func(
            model,
            tokenizer,
            history=gpt_chat_history,
        )

        # 将对话历史保存为 JSON 文件
        output_file_name = f"{save_dir}/conversation_{row['id']}.json"
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        patient_doctor_chat_history = format_gpt_chat_history_for_eval(gpt_chat_history)
        conversation = {
            "id": row["id"],
            "model": doctor,
            "conversation": patient_doctor_chat_history,
        }
        with open(output_file_name, "w", encoding='utf8') as json_file:
            json.dump(conversation, json_file, indent=4, ensure_ascii=False)

    conversations = []
    for file in os.listdir(save_dir):
        with open(os.path.join(save_dir, file), "r", encoding='utf8') as f:
            conversations.append(json.load(f))

    with open(f"{save_dir}/conversations.json", "w", encoding='utf8') as json_file:
        json.dump(conversations, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    doctor = llama2_7b_instruct_doctor
    chat_with_chatgpt(doctor)
