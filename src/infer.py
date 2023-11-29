import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     torch_dtype=torch.float16,
        # )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    model_vocab_size = model.get_input_embeddings().weight.size(0)  # model的vocab size
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size != tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenzier_vocab_size)

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instruction,
            input=None,
            temperature=0.5, # 0.1->0.5
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    # testing code for readme
    for instruction in [
        "<User>你好，MedChat。我是一个37岁的男性，我有耳石症半年的既往史。最近半个月，我感到头晕，特别是在劳累后和起床时更加明显。我还有一个体征，就是当我将头向右转动时，眼睛会向右扭转。我没有偏头痛史和晕车史，但我母亲有晕车的家族史。我没有使用过任何药物，睡眠状况良好。请问，我应该怎么办呢？",
        "<User>您好，我是一个79岁的女性，有高血压糖尿病心脏病的既往史。反复眩晕发作了1天。起卧床出现，持续Dix左+。您看我应该怎么办？",
        "<User>你好",
        "<User>你好<MedChat>你之前有因为生病吃过什么药吗？<User>我没吃过什么药。我家没有遗传病。",
        "<User>你好<MedChat>你之前有因为生病吃过什么药吗？<User>我没吃过什么药。我家没有遗传病。<MedChat>坐车的时候晕么？<User>我从来不晕车。",
        "<User>你好<MedChat>你之前有因为生病吃过什么药吗？<User>我没吃过什么药。我家没有遗传病。<MedChat>坐车的时候晕么？<User>我从来不晕车。<MedChat>有没有偏头痛？<User>我之前没有偏头痛过。",
        "<User>你好<MedChat>你之前有因为生病吃过什么药吗？<User>我没吃过什么药。我家没有遗传病。<MedChat>坐车的时候晕么？<User>我从来不晕车。<MedChat>有没有偏头痛？<User>我之前没有偏头痛过。<MedChat>你感觉身体哪里不舒服，现在有什么感觉？<User>我感觉我打鼾2年，白天困乏半年。体重增加2年。",
        "<User>你好<MedChat>你之前有因为生病吃过什么药吗？<User>我没吃过什么药。我家没有遗传病。<MedChat>坐车的时候晕么？<User>我从来不晕车。<MedChat>有没有偏头痛？<User>我之前没有偏头痛过。<MedChat>你感觉身体哪里不舒服，现在有什么感觉？<User>我感觉我打鼾2年，白天困乏半年。体重增加2年。<MedChat>请问你的年龄是？<User>我13岁了。",
        "<User>你好<MedChat>你之前有因为生病吃过什么药吗？<User>我没吃过什么药。我家没有遗传病。<MedChat>坐车的时候晕么？<User>我从来不晕车。<MedChat>有没有偏头痛？<User>我之前没有偏头痛过。<MedChat>你感觉身体哪里不舒服，现在有什么感觉？<User>我感觉我打鼾2年，白天困乏半年。体重增加2年。<MedChat>请问你的年龄是？<User>我13岁了。<MedChat>请问您是男性还是女性？<User>我是男的。<MedChat>你晚上休息得如何？<User>我休息的挺好的，睡眠没什么问题。",

    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()

# {"instruction": "", "input": "", "output": "根据检查结果和症状，我认为你是左侧扁桃腺肥大。仅供参考，请遵医嘱。"}


if __name__ == "__main__":
    fire.Fire(main)

'''
--load_8bit
--base_model
--lora_weights
--prompt_template
'template'
'''