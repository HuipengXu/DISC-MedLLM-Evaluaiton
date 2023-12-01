import os
import json
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from doctors import *
from process_parallel import process_row


def chat_with_chatgpt(doctor):
    model_dir = doctors_path[doctor]
    print(model_dir)

    dataset = pd.read_json("./data/instruct_chatgpt_messages.jsonl", lines=True).head(9)

    if doctor.startswith("gpt"):
        model = doctor
        tokenizer = None
    else:
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

    # Use joblib to parallelize the for loop
    Parallel(n_jobs=3, verbose=10)(
        delayed(process_row)(row, doctor, model, tokenizer, chat_func, save_dir)
        for _, row in tqdm(dataset.iterrows(), desc="Chating", total=len(dataset))
    )

    conversations = []
    for file in os.listdir(save_dir):
        with open(os.path.join(save_dir, file), "r", encoding="utf8") as f:
            conversations.append(json.load(f))

    with open(f"{save_dir}/conversations.json", "w", encoding="utf8") as json_file:
        json.dump(conversations, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    doctor = gpt3_doctor
    chat_with_chatgpt(doctor)
