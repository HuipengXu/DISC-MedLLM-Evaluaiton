import openai

client = openai.OpenAI(api_key="")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "你现在是一个病人，下面会描述你的病情，user是一名医生，你们会因为你的病情展开一段对话\n\n你的病情：\n您好，我是一名76岁的男性患者，最近半个月出现了反复眩晕的情况，加重了一天。起床、翻身以及低抬头时会出现眩晕，持续几分钟后会有所改善。",
        },
        {"role": "user", "content": "您好，很高兴见到您。您有高血压的既往史，除了眩晕之外还有其他不适的症状吗？"},
    ],
)

print(response.choices[0].message.content.strip())