import copy

import openai

from transformers import GenerationConfig

openai_api_key = "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"
client = openai.OpenAI(api_key=openai_api_key)


llama2_7b_instruct_doctor = "medchat"
baichuan2_7b_chat_doctor = "baichuan2-7b-chat"
bianque2_doctor = "bianque-2"
# gpt3_doctor = "gpt-3.5-turbo-1106"
gpt3_doctor = "gpt-3.5-turbo"
gpt4_doctor = "gpt-4-1106-preview"


doctors_path = {
    llama2_7b_instruct_doctor: "models/llama-2-7b-instruct-all_v3-e3_merged",
    baichuan2_7b_chat_doctor: "models/Baichuan2-7B-Chat",
    bianque2_doctor: "models/AI-ModelScope/BianQue-2",
    gpt3_doctor: gpt3_doctor,
    gpt4_doctor: gpt4_doctor,
}

docters_instruction_templates = {
    llama2_7b_instruct_doctor: "下面是患者和你的历史对话，请你运用你的知识来正确回答提问。\n### 问题:\n{instruction}\n### 回答:\n",
    baichuan2_7b_chat_doctor: "<reserved_106>{instruction}<reserved_107>",
}

doctors_generation_config = {
    llama2_7b_instruct_doctor: GenerationConfig(
        temperature=0.5,
        top_k=40,
        top_p=0.75,
        do_sample=True,
        max_new_tokens=256,
    ),
    baichuan2_7b_chat_doctor: GenerationConfig.from_pretrained(
        doctors_path[baichuan2_7b_chat_doctor]
    ),
}


def generate_prompt(doctor, instruction):
    return docters_instruction_templates[doctor].format_map(
        {"instruction": instruction}
    )


def preprocess_history(history):
    history_copy = copy.deepcopy(history)
    history_copy[0]["role"] = "assistant"
    history_copy[0]["content"] = history_copy[0]["content"].split("你的病情：\n")[-1]
    return history_copy


def format_gpt_chat_history_for_eval(history):
    history_copy = preprocess_history(history)
    history_copy = [
        {"role": "patient", "content": single_turn["content"]}
        if single_turn["role"] == "assistant"
        else {"role": "doctor", "content": single_turn["content"]}
        for single_turn in history_copy
    ]
    return history_copy


def llama2_chat(model, tokenizer, history=None):
    history_copy = preprocess_history(history)

    med_chat_history = "".join(
        f"<User>{single_turn['content']}"
        if single_turn["role"] == "assistant"
        else f"<MedChat>{single_turn['content']}"
        for single_turn in history_copy
    )

    input_text = generate_prompt(
        llama2_7b_instruct_doctor, instruction=med_chat_history
    )

    inputs = tokenizer(input_text, return_tensors="pt")
    generation_output = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        attention_mask=inputs["attention_mask"].cuda(),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=doctors_generation_config[llama2_7b_instruct_doctor],
    )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    response = output.split("### 回答:")[-1].strip()
    history += [{"role": "user", "content": response}]
    return history


def baichuan2_7b_chat(model, tokenizer, history=None):
    history_copy = preprocess_history(history)
    history_copy = [
        {
            "role": "user" if turn["role"] == "assistant" else "assistant",
            "content": turn["content"],
        }
        for turn in history_copy
    ]
    system_message = "你现在是一名专业的医生，需要你根据病人提供的信息和病人展开对话，进行专业的诊断以及给出治疗建议"
    history_copy = [{"role": "system", "content": system_message}] + history_copy
    model.generation_config = doctors_generation_config[baichuan2_7b_chat_doctor]
    response = model.chat(tokenizer, history_copy).strip()
    history += [{"role": "user", "content": response}]
    return history


def bianque2(model, tokenizer, history=None):
    # 多轮对话调用模型的chat函数
    # 注意：本项目使用"\n病人："和"\n医生："划分不同轮次的对话历史
    # 注意：user_history比bot_history的长度多1
    history_copy = preprocess_history(history)
    system_message = [
        {"role": "病人", "content": "你好"},
        {"role": "医生", "content": "我是利用人工智能技术，结合大数据训练得到的智能医疗问答模型扁鹊，你可以向我提问。"},
    ]
    history_messages = [
        {"role": "病人", "content": hist["content"]}
        if hist["role"] == "assistant"
        else {"role": "医生", "content": hist["content"]}
        for hist in history_copy
    ]
    messages = system_message + history_messages

    input_text = (
        "\n".join(f"{message['role']}：{message['content']}" for message in messages)
        + "\n医生："
    )
    response = model.chat(
        tokenizer,
        query=input_text,
        history=None,
        max_length=2048,
        num_beams=1,
        do_sample=True,
        top_p=0.75,
        temperature=0.95,
        logits_processor=None,
    )[0]
    history += [{"role": "user", "content": response}]
    return history


def gpt(model, tokenizer=None, history=None):
    history_copy = preprocess_history(history)
    history_copy = [
        {
            "role": "user" if turn["role"] == "assistant" else "assistant",
            "content": turn["content"],
        }
        for turn in history_copy
    ]
    system_message = "你现在是一名专业的医生，需要你根据病人提供的信息和病人展开对话，进行专业的诊断以及给出治疗建议"
    history_copy = [{"role": "system", "content": system_message}] + history_copy
    response = client.chat.completions.create(model=model, messages=history_copy)
    response_text = response.choices[0].message.content.strip()
    history += [{"role": "user", "content": response_text}]
    return history


doctors_call = {
    llama2_7b_instruct_doctor: llama2_chat,
    baichuan2_7b_chat_doctor: baichuan2_7b_chat,
    bianque2_doctor: bianque2,
    gpt3_doctor: gpt,
    gpt4_doctor: gpt,
}


if __name__ == "__main__":
    user_history = ["你好", "我最近失眠了"]
    bot_history = ["我是利用人工智能技术，结合大数据训练得到的智能医疗问答模型扁鹊，你可以向我提问。"]
    # 拼接对话历史
    context = "\n".join(
        [f"病人：{user_history[i]}\n医生：{bot_history[i]}" for i in range(len(bot_history))]
    )
    input_text = context + "\n病人：" + user_history[-1] + "\n医生："

    print(context)
    print("\n")
    print(input_text)
