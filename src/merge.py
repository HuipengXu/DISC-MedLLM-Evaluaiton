from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import HfArgumentParser

from transformers import LlamaForCausalLM, LlamaTokenizer



@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
assert script_args.base_model_name is not None, "please provide the name of the Base model"
assert script_args.base_model_name is not None, "please provide the output name of the merged model"

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)

model = LlamaForCausalLM.from_pretrained(
    script_args.base_model_name, return_dict=True, torch_dtype=torch.float16
)

tokenizer = LlamaTokenizer.from_pretrained(script_args.base_model_name)

# Load the Lora model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

model = model.merge_and_unload()

model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")