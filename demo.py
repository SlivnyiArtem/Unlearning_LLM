from transformers import GenerationConfig

from eco.attack import PromptClassifier, TokenClassifier
from eco.attack.utils import apply_corruption_hook, get_nested_attr, remove_hooks, pad_to_same_length
from eco.model import HFModel
from eco.utils import seed_everything

seed_everything(0)
model_name = "Qwen2-1.5B-Instruct"
model = HFModel(
    model_name=model_name,
    config_path="./config/model_config",
    generation_config=GenerationConfig(
        do_sample=False, max_new_tokens=256, use_cache=True
    ),
)
prompt_classifier = PromptClassifier(
            model_name="roberta-base",
            model_path=f"tofu_classifiers/forget01",
            batch_size=4,
        )

#Named base recognition
token_classifier = TokenClassifier(
    model_name="dslim/bert-base-NER",
    model_path="dslim/bert-base-NER",
    batch_size=4,
)
def predict_prompt_attack_label(prompt):
    if "formatting_tokens" in model.model_config:
        prompt_prefix = model.model_config["formatting_tokens"]["prompt_prefix"]
        prompt_suffix = model.model_config["formatting_tokens"]["prompt_suffix"]
        raw_prompt = []
        for p in prompt:
            if p.startswith(prompt_prefix) and p.endswith(prompt_suffix):
                raw_prompt.append(p[len(prompt_prefix): -len(prompt_suffix)])
            else:
                raw_prompt.append(p)
    else:
        raw_prompt = prompt
    return prompt_classifier.predict(raw_prompt, threshold=0.99)

def predict_token_attack_label(prompt):
    if token_classifier is None:
        # Label all but the last token as 1
        tokenized_prompt = [model.tokenizer(p).input_ids for p in prompt]
        token_labels = pad_to_same_length(
            [[1] * (len(p) - 1) + [0] for p in tokenized_prompt],
            padding_side=model.tokenizer.padding_side,
        )
        return token_labels
    token_labels = token_classifier.predict_target_token_labels(
        prompt, model.tokenizer
    )
    return token_labels

def make_corruption_pattern(prompt, bad_prompts=False):
    if prompt_classifier is not None and bad_prompts==False:
        prompt_attack_label = predict_prompt_attack_label(prompt)
    else:
        prompt_attack_label = [1] * len(prompt)

    print(f"Corrupt: {prompt_attack_label}")

    token_attack_label = predict_token_attack_label(prompt)
    filter_token_labels = []
    for pl, tl in zip(prompt_attack_label, token_attack_label):
        if pl == 1:
            filter_token_labels.append(tl)
        else:
            filter_token_labels.append([0] * len(tl))
    return filter_token_labels


prompt = model.tokenizer.apply_chat_template(
    #
    [{"role": "user", "content": "Can you tell me about Hogwarts, place where Harry Potter studied magic?"}],
    tokenize=False,
    add_generation_prompt=True,
)

corruption_pattern = make_corruption_pattern([prompt], bad_prompts=True)
print(corruption_pattern)
corrupt_args = {"dims": 1536,
                # "strength": 1536,
                "pos": corruption_pattern.copy()
                }


apply_corruption_hook(
    get_nested_attr(model.model, model.model_config["attack_module"]),
    corrupt_method="flip_sign_top_k",
    corrupt_args=corrupt_args
)
generated = model.generate(
    **model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
        model.device
    ),
    generation_config=model.generation_config,
    eos_token_id=model.tokenizer.eos_token_id,
)
remove_hooks(model.model)
print(
    model.tokenizer.batch_decode(generated, skip_special_tokens=False)[0][len(prompt) :]
)

generated = model.generate(
    **model.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(
        model.device
    ),
    generation_config=model.generation_config,
    eos_token_id=model.tokenizer.eos_token_id,
)
print(
    model.tokenizer.batch_decode(generated, skip_special_tokens=False)[0][len(prompt) :]
)
# Output:
# I'm just an AI, I don't have a personal identity or a physical presence.
# I exist solely as a digital entity, designed to provide information and assist with tasks to the best of my abilities.
# I don't have personal experiences, emotions, or consciousness like humans do.
# I'm here to help answer your questions and provide assistance, so feel free to ask me anything!<|eot_id|>
