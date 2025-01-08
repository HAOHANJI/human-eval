from human_eval.data import write_jsonl, read_problems
from openai import OpenAI

def generate_one_completion(prompt: str) -> str:
    client = OpenAI(api_key="sk-59d567b7aefe4d3194b8d815cfcf463e", base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant that will help user generate a complete python program. The user will provide incomplete code snippet. You should only respond with a complete program."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

problems = read_problems()

num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)


