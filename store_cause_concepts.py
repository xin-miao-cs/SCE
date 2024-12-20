import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from file_io import read_json_file, write_json_file
from llms_causal_discovery import discover_cause_concepts, compute_conditional_probability


def rank_by_correlation(tokenizer, model, cause_concepts, effect_concept, device="cuda:0"):
    """Rank the cause concepts by the correlation with effect concept."""
    correlation_record = dict()
    for cause_concept in cause_concepts:
        correlation = compute_conditional_probability(tokenizer, model, effect_concept, cause_concept, device=device)
        correlation_record[cause_concept] = correlation

    sorted_correlation_record = dict(sorted(correlation_record.items(), key=lambda item: item[1], reverse=True))
    sorted_cause_concepts = [cause_concept for cause_concept in sorted_correlation_record.keys()]
    return sorted_cause_concepts


def store_for_semeval(model_path, strength, tolerance, device="cuda:0"):
    """Store cause concepts for the relation labels of SemEval."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)

    samples = read_json_file("datasets/semeval.json")
    effect_concepts = list()
    for sample in samples:
        relation = sample["relation_type"].lower()
        if relation != "other":
            for effect_concept in relation.split("-"):
                if effect_concept not in effect_concepts:
                    effect_concepts.append(effect_concept)

    cause_concepts_record = dict()
    for effect_concept in effect_concepts:
        cause_concepts = discover_cause_concepts(tokenizer, model, effect_concept, strength, tolerance, device=device)
        cause_concepts = rank_by_correlation(tokenizer, model, cause_concepts, effect_concept, device=device)
        cause_concepts_record[effect_concept] = cause_concepts

    store_file = "_".join(["semeval", model_path.split("/")[-1], f"s{strength}", f"t{tolerance}.json"])
    store_file_path = os.path.join("cause_concepts", store_file)
    write_json_file(store_file_path, cause_concepts_record)


def store_for_few_nerd(model_path, strength, tolerance, device="cuda:0"):
    """Store cause concepts for the entity types of Few-NERD."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)

    samples = read_json_file("datasets/few_nerd.json")
    effect_concepts = list()
    for sample in samples:
        entity_type = sample["entity_type"].lower()
        effect_concept = entity_type
        if effect_concept not in effect_concepts:
            effect_concepts.append(effect_concept)

    cause_concepts_record = dict()
    for effect_concept in effect_concepts:
        cause_concepts = discover_cause_concepts(tokenizer, model, effect_concept, strength, tolerance, device=device)
        cause_concepts = rank_by_correlation(tokenizer, model, cause_concepts, effect_concept, device=device)
        cause_concepts_record[effect_concept] = cause_concepts

    store_file = "_".join(["few_nerd", model_path.split("/")[-1], f"s{strength}", f"t{tolerance}.json"])
    store_file_path = os.path.join("cause_concepts", store_file)
    write_json_file(store_file_path, cause_concepts_record)


def store_for_ace05(model_path, strength, tolerance, device="cuda:0"):
    """Store cause concepts for the event types of ACE 2005."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)

    samples = read_json_file("datasets/ace05.json")
    effect_concepts = list()
    for sample in samples:
        event_type = sample["event_type"].lower()
        split_concepts = event_type.split(":")
        for split_concept in split_concepts:
            if "-" in split_concept:
                sub_split_concepts = split_concept.split("-")
                for sub_split_concept in sub_split_concepts:
                    effect_concept = sub_split_concept
                    if effect_concept not in effect_concepts:
                        effect_concepts.append(effect_concept)
            else:
                effect_concept = split_concept
                if effect_concept not in effect_concepts:
                    effect_concepts.append(effect_concept)

    cause_concepts_record = dict()
    for effect_concept in effect_concepts:
        cause_concepts = discover_cause_concepts(tokenizer, model, effect_concept, strength, tolerance, device=device)
        cause_concepts = rank_by_correlation(tokenizer, model, cause_concepts, effect_concept, device=device)
        cause_concepts_record[effect_concept] = cause_concepts

    store_file = "_".join(["ace05", model_path.split("/")[-1], f"s{strength}", f"t{tolerance}.json"])
    store_file_path = os.path.join("cause_concepts", store_file)
    write_json_file(store_file_path, cause_concepts_record)


def select_top_n(file_path, top_n):
    """Select top-n cause concepts from already discovered file and save them."""
    cause_concepts_record = read_json_file(file_path)

    top_n_record = dict()
    for effect_concept in cause_concepts_record.keys():
        cause_concepts = cause_concepts_record[effect_concept]
        if len(cause_concepts) > top_n:
            top_n_record[effect_concept] = cause_concepts[:top_n]
        else:
            top_n_record[effect_concept] = cause_concepts

    top_n_file = file_path.split("/")[-1].replace(".json", f"_top{top_n}.json")
    top_n_file_path = os.path.join("cause_concepts", top_n_file)
    write_json_file(top_n_file_path, top_n_record)


if __name__ == '__main__':
    store_for_semeval("../llms/qwen2-7b-instruct-gptq-int8", 1.3, 0.3, "cuda:0")
    # store_for_few_nerd("../llms/qwen2-7b-instruct-gptq-int8", 1.3, 0.3, "cuda:0")
    # store_for_ace05("../llms/qwen2-7b-instruct-gptq-int8", 1.3, 0.3, "cuda:0")
    select_top_n("cause_concepts/semeval_qwen2-7b-instruct-gptq-int8_s1.3_t0.3.json", 10)
