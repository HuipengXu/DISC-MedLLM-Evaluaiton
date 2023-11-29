### src文件说明
1. doctor.py  包含了各模型的对话函数以及一些相关的配置，如果后续要添加评测新的模型，只要按照一定的格式在这个文件里添加对话函数和配置就行
2. extract_patient_state.py  抽取 data/instruct_v3.json 中的病人信息并加上适当的 prompt 作为gpt的 system_message，抽取的要求是信息里不包含 MedChat 字眼并且不少于10个字。在此基础上随机采样2000条并输出文件 data/instruct_chatgpt_messages.jsonl
3. generate_multiturn_dialogue.py  选取医疗模型与gpt3进行对话模拟医生诊疗过程（gpt3模拟病人），共6轮左右和复旦DISC-MedLLM保持一致(参考：https://github.com/FudanDISC/DISC-MedLLM/blob/main/eval/dialogues/DISC-MedLLM_cmb-clin.json)。最后会输出以医疗模型命名的文件夹其中会包含每一个对话的对话历史以及一份完整的包含2000条数据所有对话历史的文件，例如：data/medchat_chatgpt_multiturn_dialogue（目前gpt3的回复使用“测试”进行填充）
4. gpt4_judge_prompt.py  几乎完全复制的复旦DISC-MedLLM使用的gpt4打分 prompt
5. 输入3中的文件输出使用gpt4进行打分，打分结果文件保存在 data/medchat_chatgpt_multiturn_dialogue/ratings.json
