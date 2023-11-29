CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --base_model models/llama-2-7b-instruct-all_v3-e3_merged \
    --with_prompt \
    --interactive

# CUDA_VISIBLE_DEVICES=0 python src/inference.py \
#     --base_model models/llama-2-7b-instruct-all_v2-copy \
#     --lora_model models/llama-2-7b-instruct-all_v3-e3-copy \
#     --with_prompt \
#     --interactive
