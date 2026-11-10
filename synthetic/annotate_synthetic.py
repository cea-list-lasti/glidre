from datasets import load_from_disk
import argparse
import torch
import tqdm
from transformers import set_seed, AutoTokenizer
set_seed(42)
import random
from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk, concatenate_datasets
import pandas as pd
import os
import json
import re
import ast
import bisect
import collections
import datetime
import hashlib
import sys
import traceback
sys.excepthook = lambda *exc: (traceback.print_exception(*exc), sys.stderr.flush(), os._exit(1))   # a crash must free the GPU

TOKEN_RE = re.compile(r"\w+(?:_\w+)*|\S")      # GLiDRE word tokenizer
SENT_END = re.compile(r"(?<=[.!?؟])\s+")
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0, max_tokens=32000)   # Qwen3 thinking defaults
SEED = 42
MAX_RETRIES = 1
stats = collections.Counter()


def split_sentences(text):
    return [s for s in SENT_END.split(text) if s]


def chunk_text(text, max_tokens):
    """Cut the source text into chunks of <= max_tokens encoder tokens, at sentence ends, before annotation."""
    chunks, cur, cur_len = [], [], 0
    for s in split_sentences(text.strip()):
        n = len(chunk_tokenizer(s, add_special_tokens=False)["input_ids"])
        if n > max_tokens:
            stats["phrase trop longue ignorée"] += 1
            continue
        if cur and cur_len + n > max_tokens:
            chunks.append(" ".join(cur)); cur, cur_len = [], 0
        cur.append(s); cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def parse_llm_json(output):
    if args.reasonning and "</think>" not in output:
        return None, "think_not_closed"
    answer = output.rsplit("</think>", 1)[-1]
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", answer, re.DOTALL)
    payload = (blocks[-1] if blocks else answer).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(payload)
            return (parsed, "ok") if isinstance(parsed, list) else (None, "bad_schema")
        except Exception:   # literal_eval can also raise TypeError ({...} set of dicts), MemoryError, RecursionError
            pass
    return None, "unparsable"


try:
    from vllm.sampling_params import RequestOutputKind
    FINAL_ONLY = {"output_kind": RequestOutputKind.FINAL_ONLY}   # step() only returns finished requests
except ImportError:
    FINAL_ONLY = {}


def use_stats(counter):
    """The helpers count into the global `stats`: point it at the counter of the chunk being processed."""
    global stats
    stats = counter


def load_done(path):
    """Chunks already saved by a previous run. A last line truncated by a kill is dropped so appends stay valid."""
    if not os.path.exists(path):
        return {}
    done, valid = {}, []
    with open(path, encoding="utf8") as f:
        for line in f:
            try:
                record = json.loads(line)
                done[record["id"]] = record
                valid.append(line if line.endswith("\n") else line + "\n")
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    with open(path + ".tmp", "w", encoding="utf8") as f:
        f.writelines(valid)
    os.replace(path + ".tmp", path)
    return done


def run_pipeline(chunks, path, batch_size):
    """Keep `batch_size` requests in flight: as soon as one finishes, the next one is submitted.
    Each chunk goes entities -> relations (failed parses are resubmitted with a new seed), and every
    finished chunk is appended to `path`, so a killed job resumes where it stopped."""
    done = load_done(path)
    todo = collections.deque(c for c in chunks if c["id"] not in done)
    print(f"{len(done)} chunks déjà annotés, {len(todo)} à faire")
    engine, inflight, next_id = llm.llm_engine, {}, 0
    pbar = tqdm.tqdm(total=len(todo), desc="chunks annotés")

    def submit(item, stage, attempt):
        nonlocal next_id
        request_id = str(next_id); next_id += 1
        engine.add_request(request_id, item[f"{stage}_prompt"], SamplingParams(**SAMPLING, seed=SEED + attempt, **FINAL_ONLY))
        inflight[request_id] = (item, stage, attempt)

    with open(path, "a", encoding="utf8") as f:
        while todo or inflight:
            while todo and len(inflight) < batch_size:
                item = dict(todo.popleft(), stats=collections.Counter())
                submit(apply_chat_template_entity(item), "entity", 0)
            for out in engine.step():
                if not out.finished:
                    continue
                item, stage, attempt = inflight.pop(out.request_id)
                raw = out.outputs[0].text
                parsed, status = parse_llm_json(raw)
                if status != "ok" and attempt < MAX_RETRIES:
                    submit(item, stage, attempt + 1)
                    continue
                use_stats(item["stats"])
                stats[f"llm {stage}: {status}"] += 1
                if stage == "entity":
                    item.update(raw_entities=raw, entity_status=status,
                                entities=json.dumps(locate_mentions(parsed, item["text"]) if parsed else []))
                    if apply_chat_template_relation(item)["relation_prompt"]:
                        submit(item, "relation", 0)
                        continue
                    item.update(raw_relations="", relation_status="skipped", relations="[]")
                else:
                    ids = {e["id"] for e in json.loads(item["entities"])}
                    item.update(raw_relations=raw, relation_status=status,
                                relations=json.dumps(clean_relations(parsed, ids, len(split_sentences(item["text"])))))
                to_glidre(item)
                item.pop("entity_prompt", None); item.pop("relation_prompt", None)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                done[item["id"]] = item
                pbar.update(1)
    pbar.close()
    return done


def locate_mentions(entities, text):
    """Mentions are anchored to their sentence; a span claimed by two entities is dropped from both."""
    sents = split_sentences(text)
    starts, pos = [], 0
    for s in sents:
        starts.append(pos); pos += len(s) + 1
    located = []
    for ent in entities:
        if not isinstance(ent, dict) or not isinstance(ent.get("id"), int):
            stats["entité: format"] += 1; continue
        spans = set()
        for m in ent.get("mentions", []) if isinstance(ent.get("mentions"), list) else []:
            k, v = (m.get("sentence"), m.get("value")) if isinstance(m, dict) else (None, None)
            if not isinstance(k, int) or not 0 <= k < len(sents) or not isinstance(v, str) or not v:
                stats["mention: phrase invalide"] += 1; continue
            occ = [o.start() for o in re.finditer(r"(?<!\w)" + re.escape(v) + r"(?!\w)", sents[k])]
            if not occ:
                stats["mention: introuvable dans sa phrase"] += 1
            spans.update((starts[k] + o, starts[k] + o + len(v)) for o in occ)
        located.append((ent, spans))
    owners = collections.Counter(sp for _, spans in located for sp in spans)
    result = []
    for ent, spans in located:
        keep = sorted(sp for sp in spans if owners[sp] == 1)
        stats["mention: span ambigu"] += len(spans) - len(keep)
        if keep:
            result.append({"id": ent["id"], "type": str(ent.get("type")), "mentions": [{"start": s, "end": e} for s, e in keep]})
    return result


def to_tokens(text, entities):
    """Char spans -> GLiDRE token spans; a mention off token boundaries is dropped, never half-converted."""
    toks = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    tstarts, tends = [s for _, s, _ in toks], [e for _, _, e in toks]
    out = []
    for ent in entities:
        ms = []
        for m in ent["mentions"]:
            i, j = bisect.bisect_left(tstarts, m["start"]), bisect.bisect_left(tends, m["end"])
            if i < len(toks) and j < len(toks) and tstarts[i] == m["start"] and tends[j] == m["end"] and i <= j:
                ms.append({"start": i, "end": j, "value": text[m["start"]:m["end"]]})
            else:
                stats["mention: hors frontière de token"] += 1
        if ms:
            out.append(dict(ent, mentions=ms))
    return [t for t, _, _ in toks], out


def norm_label(label):
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", label)).strip().lower()


def clean_relations(relations, entity_ids, n_sents):
    """Keep relations with integer ids of known entities, a label and valid evidence. h == t is allowed."""
    rels = set()
    for r in relations or []:
        if not isinstance(r, dict):
            stats["relation: format"] += 1; continue
        try:
            h, t = int(r.get("head")), int(r.get("tail"))
        except (TypeError, ValueError):
            stats["relation: id non entier"] += 1; continue
        lab, ev = r.get("relation"), r.get("evidence")
        if h not in entity_ids or t not in entity_ids:
            stats["relation: entité inconnue"] += 1
        elif not isinstance(lab, str) or not norm_label(lab):
            stats["relation: label vide"] += 1
        elif not isinstance(ev, list) or not ev or not all(isinstance(k, int) and 0 <= k < n_sents for k in ev):
            stats["relation: preuve invalide"] += 1
        else:
            rels.add((h, norm_label(lab), t))
    return [list(x) for x in sorted(rels, key=str)]


def validate_doc(tokens, mentions, relations):
    n, ids, seen = len(tokens), set(), set()
    for e in mentions:
        assert e["id"] not in ids and e["mentions"]
        ids.add(e["id"])
        for m in e["mentions"]:
            assert 0 <= m["start"] <= m["end"] < n and (m["start"], m["end"]) not in seen
            seen.add((m["start"], m["end"]))
    assert all(h in ids and t in ids and r == norm_label(r) for h, r, t in relations)


entity_system = """
You are a helpful thinking coreference information extraction system that outputs only valid JSON. Your objective is to identify ALL the entities of the text, group the text spans referring to the same entity and determine their type. Finally, output the list of entities in a code block delimited by ```json at the beginning and ``` at the end. Ensure that you do not group mentions referring to different entities.
"""

rel_system = """
You are a helpful and intelligent thinking relation extraction system that outputs only valid JSON. Pay attention to the order of each relation, with the first entity being the head and the second being the tail (e.g., head "Paris", relation "located in", tail "France" is correct, but head "France", relation "located in", tail "Paris" is incorrect). Avoid duplicate entries. Finally, output the list of relations in a code block delimited by ```json at the beginning and ``` at the end.
"""


def apply_chat_template_entity(line) :
    numbered = "\n".join(f"[S{i}] {s}" for i, s in enumerate(split_sentences(line["text"])))
    entity_prompt = """List EVERY entity mentioned in the following text: people, organizations, locations, events, works, products, dates, times, numbers, quantities, concepts... Group the mentions that refer to the same entity, including pronouns and nominal mentions (e.g. "the company"). List every occurrence separately, copying the exact text span and the index of the sentence where it appears. Use the JSON format of the following example : [{"id": 0, "type": "PERSON", "mentions": [{"value": "John Wick", "sentence": 0}, {"value": "He", "sentence": 1}]}, {"id": 1, "type": "DATE", "mentions": [{"value": "2019", "sentence": 0}]}]
# TEXT (one sentence per line, prefixed by its index) :
"""+numbered+"""
# OUTPUT :
"""
    conversations_entity = tokenizer.apply_chat_template([{'role': 'system', 'content': entity_system}, {'role': 'user', 'content': entity_prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=args.reasonning)
    line["entity_prompt"] = conversations_entity
    return line

def apply_chat_template_relation(line) :
    entities = json.loads(line["entities"])
    if not entities:
        line["relation_prompt"] = ""
        return line
    numbered = "\n".join(f"[S{i}] {s}" for i, s in enumerate(split_sentences(line["text"])))
    listed = "; ".join(f'{e["id"]}: {line["text"][e["mentions"][0]["start"]:e["mentions"][0]["end"]]} ({e["type"]})' for e in entities)
    prompt_rel = """List ALL the relations between the following entities that are stated or clearly implied by the text. Use short lowercase natural-language relation labels (e.g. "born in", "founded by", "date of death"). Head and tail are integer entity ids. For each relation, give the indices of the sentences that support it. Use the JSON format of the following example : [{"head": 0, "relation": "visited", "tail": 1, "evidence": [0]}]
# TEXT (one sentence per line, prefixed by its index) :
"""+numbered+"""
# ENTITIES :
"""+ listed +"""
# RELATIONS :
"""
    conversations_rel = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': rel_system}, {'role' : 'user', 'content' : prompt_rel}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.reasonning
    )
    line["relation_prompt"] = conversations_rel
    return line


def to_glidre(line):
    tokens, mentions = to_tokens(line["text"], json.loads(line["entities"]))
    ids = {e["id"] for e in mentions}
    relations = [r for r in json.loads(line["relations"]) if r[0] in ids and r[2] in ids]
    try:
        validate_doc(tokens, mentions, relations)
    except AssertionError:
        stats["document invalide"] += 1
        mentions, relations = [], []
    line["tokenized_text"], line["mentions"], line["glidre_relations"] = tokens, json.dumps(mentions), json.dumps(relations)
    return line


def chunk_batch(examples, indices):
    out = {"text": [], "source_id": [], "chunk_idx": [], "language": []}
    source_ids = examples["id"] if "id" in examples else indices
    languages = examples["language"] if "language" in examples else [None] * len(source_ids)
    for source_id, text, language in zip(source_ids, examples["text"], languages):
        for k, chunk in enumerate(chunk_text(text, args.max_chunk_tokens)):
            out["text"].append(chunk); out["source_id"].append(str(source_id)); out["chunk_idx"].append(k); out["language"].append(language)
    return out




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generation pretraining dataset')

    parser.add_argument('--synthetic_dataset', type=str, required=True)

    parser.add_argument('--path_model', type=str, required=True)
    
    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--start_idx', type=int, required=True)

    parser.add_argument('--end_idx', type=int, required=True)    

    parser.add_argument('--batch_size', type=int, required=True)    
    
    parser.add_argument('--reasonning', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument('--chunk_tokenizer', type=str, default="microsoft/deberta-v3-large", help="tokenizer of the GLiDRE text encoder")

    parser.add_argument('--max_chunk_tokens', type=int, default=480)

    parser.add_argument('--multilingual', action='store_true', help="source texts in any language, relation labels in the text's language")

    args = parser.parse_args()
    if not args.reasonning:   # Qwen3 non-thinking defaults, without presence_penalty (it penalizes the repeated JSON keys)
        SAMPLING.update(temperature=0.7, top_p=0.8, max_tokens=8192)
    if args.multilingual:
        entity_system += "The text can be in any language: copy every mention exactly as it is written in the text.\n"
        rel_system += "The text can be in any language: write the relation labels in the language of the text.\n"

    chunk_tokenizer = AutoTokenizer.from_pretrained(args.chunk_tokenizer)
    llm = LLM(model=args.path_model, dtype=torch.bfloat16, trust_remote_code=True, gpu_memory_utilization=0.95, max_model_len = 64000, max_num_seqs = 200)
    tokenizer = llm.get_tokenizer()

    if os.path.exists(os.path.join(args.synthetic_dataset, "state.json")):
        dataset = load_from_disk(args.synthetic_dataset)
    else:   # folder of datasets (one per language): concatenated then shuffled, so that every shard mixes all of them
        parts = [load_from_disk(os.path.join(args.synthetic_dataset, f)) for f in sorted(os.listdir(args.synthetic_dataset))]
        dataset = concatenate_datasets(parts).shuffle(seed=SEED)
    dataset = dataset.select(range(args.start_idx, args.end_idx))
    dataset = dataset.map(chunk_batch, batched=True, with_indices=True, remove_columns=dataset.column_names)
    chunks = [dict(c, id=f'{c["source_id"]}_{c["chunk_idx"]}') for c in dataset]
    run_stats = stats   # chunking counters; run_pipeline then counts per chunk

    name = f"dataset_{args.start_idx}_{args.end_idx}"
    os.makedirs(args.output_path, exist_ok=True)
    done = run_pipeline(chunks, os.path.join(args.output_path, f"{name}.jsonl"), args.batch_size)

    records = [done[c["id"]] for c in chunks if c["id"] in done]
    stats = collections.Counter(run_stats)
    for r in records:
        stats.update(r["stats"])
    docs = [{"id": r["id"], "tokenized_text": r["tokenized_text"], "mentions": json.loads(r["mentions"]),
             "relations": json.loads(r["glidre_relations"]), **({"language": r["language"]} if r.get("language") else {})} for r in records if r["glidre_relations"] != "[]"]
    with open(os.path.join(args.output_path, f"{name}_glidre.json"), "w", encoding="utf8") as f:
        json.dump(docs, f, ensure_ascii=False)
    import inspect, vllm
    prompts = [inspect.getsource(apply_chat_template_entity), inspect.getsource(apply_chat_template_relation), entity_system, rel_system]
    metadata = {"model": os.path.abspath(args.path_model), "vllm": vllm.__version__, "sampling": SAMPLING, "seed": SEED, "reasoning": args.reasonning, "multilingual": args.multilingual,
                "max_retries": MAX_RETRIES, "prompts_sha1": hashlib.sha1(repr(prompts).encode()).hexdigest(),
                "source": os.path.abspath(args.synthetic_dataset), "indices": [args.start_idx, args.end_idx],
                "chunk_tokenizer": args.chunk_tokenizer, "max_chunk_tokens": args.max_chunk_tokens,
                "date": datetime.datetime.now().isoformat(), "chunks": len(chunks), "glidre_docs": len(docs), "stats": dict(stats)}
    with open(os.path.join(args.output_path, f"{name}_metadata.json"), "w", encoding="utf8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os._exit(0)   # the vLLM engine core otherwise keeps the job alive after the end

    
