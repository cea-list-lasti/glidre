import argparse
import ast
import json

import torch
from transformers import set_seed
from vllm import LLM, SamplingParams

set_seed(42)

SYSTEM_A = "You are a high-performance Document-Level Relation Extraction system. Your task is to extract relations from the provided text, where entities are already tagged with their entity type and ID. Each entity may have multiple mentions and is uniquely identified by an entity ID. A relation consists of a head entity, a tail entity, and a relation type. Return a list of valid relation triplets in the following format: [(head_id, \"relation_type\", tail_id), ...] eg. [(1, \"contry of citizenship\", 2), ...]. Only use the entity IDs mentioned in the text and provided entity types. Output only the list of triplets : no explanations, headers, or additional text."
SYSTEM_B = "You are a high-performance Document-Level Relation Extraction system. Your task is to extract relations from the provided text, where entities are already tagged with their entity type and ID. Each entity may have multiple mentions and is uniquely identified by an entity ID. A relation consists of a head entity, a tail entity, and a relation type. Return a list of valid relation triplets in the following format: [(head_id (integer) , \"relation_type\" (string), tail_id (integer)), ...], example : [(1, \"contry of citizenship\", 2), ...].Only use the entity IDs mentioned in the text and provided entity types."

MODELS = {
    "qwen3-4b-thinking": dict(
        path="Qwen/Qwen3-4B-Thinking-2507",
        system=SYSTEM_A,
        llm=dict(max_model_len=64000, gpu_memory_utilization=0.7, tensor_parallel_size=4),
        sampling=dict(max_tokens=64000, temperature=0.0),
    ),
    "qwen3-30b-a3b-thinking": dict(
        path="Qwen/Qwen3-30B-A3B-Thinking-2507",
        system=SYSTEM_B,
        llm=dict(max_model_len=32768, gpu_memory_utilization=0.8, tensor_parallel_size=4),
        sampling=dict(max_tokens=32768),
    ),
}


def tag(doc):
    text = ""
    for i, token in enumerate(doc["tokenized_text"]):
        for entity in doc["mentions"]:
            for mention in entity["mentions"]:
                if mention["start"] == i:
                    text += "<ENTITY_" + str(entity["id"]) + "_" + entity["type"] + ">"
        text += token
        for entity in doc["mentions"]:
            for mention in entity["mentions"]:
                if mention["end"] == i:
                    text += "</ENTITY>"
        if i != len(doc["tokenized_text"]) - 1:
            text += " "
    return text


def answer(output):
    try:
        return output.split("</think>")[1]
    except IndexError:
        return "[]"


def parse(output, relations):
    try:
        triplets = ast.literal_eval(output.replace("```json", "").replace("```", "").strip())
        if not isinstance(triplets, list):
            raise ValueError
    except Exception:
        return []
    return [t for t in triplets
            if isinstance(t, (tuple, list)) and len(t) == 3 and isinstance(t[0], int)
            and isinstance(t[1], str) and isinstance(t[2], int) and t[1] in relations]


def score(preds, gold, train):
    facts = {(a["name"], b["name"], l["r"]) for d in train for l in d["labels"]
             for a in d["vertexSet"][l["h"]] for b in d["vertexSet"][l["t"]]}
    std = {(d["title"], l["r"], l["h"], l["t"]) for d in gold for l in d["labels"]}
    vertex = {d["title"]: d["vertexSet"] for d in gold}
    preds = {(p["title"], p["r"], p["h_idx"], p["t_idx"]) for p in preds}
    correct = preds & std
    in_train = sum(any((a["name"], b["name"], r) in facts for a in vertex[t][h] for b in vertex[t][tl])
                   for t, r, h, tl in correct)
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0
    recall = len(correct) / len(std)
    precision = len(correct) / len(preds)
    precision_ign = (len(correct) - in_train) / (len(preds) - in_train + 1e-5)
    return 100 * f1(precision, recall), 100 * f1(precision_ign, recall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--glidre_data", default="data")
    parser.add_argument("--redocred", default="Re-DocRED/data")
    parser.add_argument("--rel_info", default="rel_info.json")
    parser.add_argument("--path")
    args = parser.parse_args()

    config = MODELS[args.model]
    path = args.path or config["path"]
    name = args.path.rstrip("/").split("/")[-1] if args.path else args.model
    rel2id = {v: k for k, v in json.load(open(args.rel_info)).items()}
    docs = {s: json.load(open(f"{args.glidre_data}/{f}_revised.json")) for s, f in [("test", "test"), ("dev", "validation")]}
    labels = sorted({r[1] for s in ("test", "dev") for d in docs[s] for r in d["relations"]})
    train = json.load(open(f"{args.redocred}/train_revised.json"))

    llm = LLM(model=path, dtype=torch.bfloat16, trust_remote_code=True, **config["llm"])
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(**config["sampling"])

    for split in ("test", "dev"):
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": config["system"]},
             {"role": "user", "content": f"Please extract relations in the following document : {tag(d)}\nRespect the following relation types : {labels}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True) for d in docs[split]]
        outputs = []
        for i in range(0, len(prompts), 128):
            outputs += [o.outputs[0].text for o in llm.generate(prompts[i:i + 128], sampling_params)]

        gold = json.load(open(f"{args.redocred}/{split}_revised.json"))
        preds = [{"title": g["title"], "r": rel2id[r], "h_idx": h, "t_idx": t, "evidence": []}
                 for g, o in zip(gold, outputs) for h, r, t in parse(answer(o), rel2id)]
        json.dump(outputs, open(f"{name}_{split}_outputs.json", "w"))
        json.dump(preds, open(f"{name}_{split}_off.json", "w"))
        print(split, "F1 = %.2f  Ign F1 = %.2f" % score(preds, gold, train))
