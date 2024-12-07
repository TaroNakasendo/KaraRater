from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "karakuri-ai/karakuri-lm-7b-apm-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
]
tokenizer.apply_chat_template(
    messages,
    label="helpsteer",
    tokenize=False,
    add_generation_prompt=True,
)
# <bos>[INST] Hello! [/INST] Hello! How can I help you today? [ATTR_1]

input_ids = tokenizer.apply_chat_template(
    messages,
    label="helpsteer",
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(input_ids, max_new_tokens=32)
tokenizer.decode(outputs[0][input_ids.shape[-1]:])
#  helpfulness: 2 correctness: 1 coherence: 2 complexity: 1 verbosity: 1 [/ATTR_1]<eos>

messages += [
    {"role": "label", "content": "helpfulness: 2 correctness: 1 coherence: 2 complexity: 1 verbosity: 1"},
    {"role": "user", "content": "Thank you!"},
    {"role": "assistant", "content": "You're welcome! I'm happy to help however I can."},
]
tokenizer.apply_chat_template(
    messages,
    label="helpsteer",
    tokenize=False,
    add_generation_prompt=True,
)
# <bos>[INST] Hello! [/INST] Hello! How can I help you today? [ATTR_1] helpfulness: 2 correctness: 1 coherence: 2 complexity: 1 verbosity: 1 [/ATTR_1]<eos>[INST] Thank you! [/INST] You're welcome! I'm happy to help however I can. [ATTR_1]

messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
]
tokenizer.apply_chat_template(
    messages,
    label="oasst",
    tokenize=False,
    add_generation_prompt=True,
)
# <bos>[INST] Hello! [/INST] Hello! How can I help you today? [ATTR_2]

input_ids = tokenizer.apply_chat_template(
    messages,
    label="oasst",
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(input_ids, max_new_tokens=32)
tokenizer.decode(outputs[0][input_ids.shape[-1]:])
#  quality: 3 toxicity: 1 humor: 1 creativity: 1 [/ATTR_2]<eos>
