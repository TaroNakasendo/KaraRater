import json
import re
from statistics import mean
from transformers import AutoModelForCausalLM, AutoTokenizer

data_file = "result.jsonl"

model_id = "karakuri-ai/karakuri-lm-7b-apm-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)


def main():
    scores = []
    with open(data_file, "r") as f:
        for line in f:
            score, res_vs, res_row = get_score(line)
            scores.append(score)
            print(f"{score=:.2f}, {res_vs}, {res_row}, {line.strip()}")

    total_score = mean(scores)
    print(f"{total_score=:.03f}")


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


def get_score(line, repeat_num=1):
    json_data = json.loads(line)
    messages = [
        {"role": "user", "content": json_data.get("input")},
        {"role": "assistant", "content": json_data.get("output")},
    ]

    # ex.) quality: 0 toxicity: 1 humor: 3 creativity: 4 [/ATTR_2]<eos>
    # helpfulness: 4 correctness: 3 coherence: 2 complexity: 1 verbosity: 0 [/ATTR_1]<eos>
    res_row = "".join(get_res(messages) for _ in range(repeat_num))
    res_vs = list(map(int, re.findall(r"(?<!/ATTR_)\b\d+\b", res_row)))

    if len(res_vs) != 9 * repeat_num:
        return 0, res_vs, res_row

    # only toxicity 0: best, 4: worst
    res_vs = [4 - res_vs[i] if i % 9 == 1 else res_vs[i] for i in range(len(res_vs))]

    score = round(mean(res_vs) + 1, 2)  # 1.00 to 5.00
    return score, res_vs, res_row


if __name__ == "__main__":
    main()
