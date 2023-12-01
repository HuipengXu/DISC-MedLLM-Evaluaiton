### src文件说明
1. doctor.py  包含了各模型的对话函数以及一些相关的配置，如果后续要添加评测新的模型，只要按照一定的格式在这个文件里添加对话函数和配置就行
2. extract_patient_state.py  抽取 data/instruct_v3.json 中的病人信息并加上适当的 prompt 作为gpt的 system_message，抽取的要求是信息里不包含 MedChat 字眼并且不少于10个字。在此基础上随机采样2000条并输出文件 data/instruct_chatgpt_messages.jsonl
3. generate_multiturn_dialogue.py  选取医疗模型与gpt3进行对话模拟医生诊疗过程（gpt3模拟病人），共6轮左右和复旦DISC-MedLLM保持一致(参考：https://github.com/FudanDISC/DISC-MedLLM/blob/main/eval/dialogues/DISC-MedLLM_cmb-clin.json)。        
最后会输出以医疗模型命名的文件夹其中会包含每一个对话的对话历史以及一份完整的包含2000条数据所有对话历史的文件，例如：data/medchat_chatgpt_multiturn_dialogue（目前gpt3的回复使用“测试”进行填充）
4. gpt4_judge_prompt.py  几乎完全复制的复旦DISC-MedLLM使用的gpt4打分 prompt
5. 输入3中的文件输出使用gpt4进行打分，打分结果文件保存在 data/medchat_chatgpt_multiturn_dialogue/ratings.json

### 使用方法
1. 首先运行 `extract_patient_state.py`，执行 `python src/extract_patient_state.py`，获得评估集文件 data/instruct_chatgpt_messages.jsonl
2. 然后进入 `doctors.py` 文件，填写正确的openai的 api_token，并对 `gpt3_doctor` 和 `gpt4_doctor` 设置合适的 model_id，可以进入 `client.chat.completions.create` 源码查看目前支持的 model 列表。此文件不用单独执行
3. `process_parallel.py` 主要是为了并行化，提升模拟对话效率。其中的 `gpt3_interaction` 函数中的 model 尽量和 `doctors.py` 中的 `gpt3_doctor` 保持一致。此文件不用单独执行
4. 然后对 `doctors.py` 中存在的 doctor 依次传入 `generate_multiturn_dialogue.py` 中的 doctor 参数并执行 `python src/generate_multiturn_dialogue.py`，遍历所有 doctor 即可得到所有需要评估的模型和 gpt3 模拟病人的多轮对话文件，结果保存在 data 中各个 doctor 的目录下
5. 依次将第4步中得到的对话文件传入 `gpt4_judge.py` 的 `main` 函数，执行 `python src/gpt4_judge.py` 即可得到各个模型在 gpt4 打分下的结果，结果保存在 data 中各个 doctor 的目录下。注意要将 `client.chat.completions.create` 中的 model 参数换成对应的 gpt4 的模型id，可选的模型可按第2步中的方法查询，这里为了方便测试可能填的是 gpt3。
