import argparse
import collections
import glob
import json
import re

STOPWORDS = {"is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "a", "an", "the"}


def norm_label(label):
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", label)).strip().lower()


def synonym_key(label):
    words = norm_label(label).split()
    return " ".join([w for w in words if w not in STOPWORDS] or words)


def cap_entities(doc, max_entities):
    """Keep at most max_entities entities, by decreasing number of relations, and renumber them from 0."""
    degree = collections.Counter(x for h, _, t in doc["relations"] for x in {h, t})
    kept = sorted(doc["mentions"], key=lambda e: -degree[e["id"]])[:max_entities]
    kept.sort(key=lambda e: e["mentions"][0]["start"])
    new_id = {e["id"]: i for i, e in enumerate(kept)}
    doc["mentions"] = [dict(e, id=new_id[e["id"]]) for e in kept]
    doc["relations"] = [[new_id[h], r, new_id[t]] for h, r, t in doc["relations"] if h in new_id and t in new_id]


def main(out_path, patterns, max_entities=None):
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    docs = [doc for path in paths for doc in json.load(open(path, encoding="utf8"))]
    surface = collections.Counter(r[1] for d in docs for r in d["relations"])
    freq = collections.Counter(norm_label(r[1]) for d in docs for r in d["relations"])
    canonical = {}
    for label, _ in freq.most_common():
        canonical.setdefault(synonym_key(label), label)
    before = sum(len(d["relations"]) for d in docs)
    for d in docs:
        rels = {(h, canonical[synonym_key(r)], t) for h, r, t in d["relations"]}
        d["relations"] = [list(x) for x in sorted(rels, key=str)]
    after = sum(len(d["relations"]) for d in docs)
    if max_entities:
        trimmed = sum(len(d["mentions"]) > max_entities for d in docs)
        ents = sum(len(d["mentions"]) for d in docs)
        for d in docs:
            if len(d["mentions"]) > max_entities:
                cap_entities(d, max_entities)
        docs = [d for d in docs if d["relations"]]
        print(f"plafond {max_entities} entités : {trimmed} documents réduits, entités {ents} -> "
              f"{sum(len(d['mentions']) for d in docs)}, triplets {after} -> {sum(len(d['relations']) for d in docs)}")
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(docs, f, ensure_ascii=False)
    print(f"{len(paths)} fichiers, {len(docs)} documents")
    print(f"labels : {len(surface)} formes -> {len(freq)} après normalisation -> {len(canonical)} après synonymes")
    print(f"triplets : {before} -> {after} (doublons créés par la fusion supprimés)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_path")
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--max_entities", type=int, default=None)
    args = parser.parse_args()
    main(args.out_path, args.patterns, args.max_entities)
