import os
import json

import openai

from doctors import openai_api_key, format_gpt_chat_history_for_eval


client = openai.OpenAI(api_key=openai_api_key)


def gpt3_interaction(chat_history):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=chat_history
    )
    response_text = response.choices[0].message.content.strip()
    chat_history += [{"role": "assistant", "content": response_text}]
    return chat_history


def gpt3_interaction_test(chat_history):
    response_text = "测试"
    chat_history += [{"role": "assistant", "content": response_text}]
    return chat_history


def process_row(row, doctor, model, tokenizer, chat_func, save_dir):
    # （这里只是示例，你可以根据需要调整）
    try:
        num_turns = 2
        gpt_chat_history = row.history

        for _ in range(num_turns):
            gpt_chat_history = chat_func(
                model,
                tokenizer,
                history=gpt_chat_history,
            )

            # 与 GPT-3.5 进行对话
            # gpt_chat_history = gpt3_interaction(gpt_chat_history)
            gpt_chat_history = gpt3_interaction_test(gpt_chat_history)

            # 对话以医生结束
        gpt_chat_history = chat_func(
            model,
            tokenizer,
            history=gpt_chat_history,
        )

        # 将对话历史保存为 JSON 文件
        output_file_name = f"{save_dir}/conversation_{row['id']}.json"
        patient_doctor_chat_history = format_gpt_chat_history_for_eval(gpt_chat_history)
        conversation = {
            "id": row["id"],
            "model": doctor,
            "conversation": patient_doctor_chat_history,
        }
        with open(output_file_name, "w", encoding="utf8") as json_file:
            json.dump(conversation, json_file, indent=4, ensure_ascii=False)

    except Exception as e:
        # Print the exception for debugging
        print(f"Error processing row {row['id']}: {e}")
