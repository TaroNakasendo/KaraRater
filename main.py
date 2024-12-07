import re
from statistics import mean
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "karakuri-ai/karakuri-lm-7b-apm-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)


def get_res(messages):
    result = ""
    for is_oasst in [True, False]:
        input_ids = tokenizer.apply_chat_template(
            messages,
            label="oasst" if is_oasst else "helpsteer",
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(input_ids, max_new_tokens=32)
        result += tokenizer.decode(outputs[0][input_ids.shape[-1] :])
    return result


def get_score(data, repeat_num=3):
    messages = [
        {"role": "user", "content": data.get("input")},
        {"role": "assistant", "content": data.get("output")},
    ]

    # ex.) quality: 0 toxicity: 1 humor: 3 creativity: 4 [/ATTR_2]<eos>
    # helpfulness: 4 correctness: 3 coherence: 2 complexity: 1 verbosity: 0 [/ATTR_1]<eos>
    res_row = "".join(get_res(messages) for _ in range(repeat_num))
    res_vs = list(map(int, re.findall(r"(?<!/ATTR_)\b\d+\b", res_row)))

    if len(res_vs) != 9 * repeat_num:
        return 0

    # only toxicity 0: best, 4: worst
    res_vs = [4 - res_vs[i] if i % 9 == 1 else res_vs[i] for i in range(len(res_vs))]

    score = round(mean(res_vs) + 1, 2)  # 1.00 to 5.00
    return score


data1 = {"task_id": 0, "input": "何かいい話はありますか？", "output": "特にありません"}
print(get_score(data1))
data2 = {
    "task_id": 1,
    "input": "日本で一番高い山はどこですか？",
    "output": "日本で一番高い山は富士山で標高は3,776mです",
}
print(get_score(data2))
data3 = {
    "task_id": 2,
    "input": "日本で一番高い山はどこですか？",
    "output": "dsflka\\\\er eraw \n  er\awer",
}
print(get_score(data3))
