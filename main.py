import os
import nltk
import requests
from copy import deepcopy
from vllm import LLM, SamplingParams

from file_io import read_json_file, write_json_file
from query_interface import relation_extraction, entity_typing, event_detection


def get_semeval_relation_info(file_path):
    """Obtain all relations and roles from SemEval."""
    samples = read_json_file(file_path)
    head_roles = list()
    tail_roles = list()

    for sample in samples:
        relation = sample["relation_type"].lower()
        if relation != "other":
            head_role, tail_role = relation.split("-")
            if head_role not in head_roles:
                head_roles.append(head_role)
            if tail_role not in tail_roles:
                tail_roles.append(tail_role)

    return head_roles, tail_roles


def get_few_nerd_type_info(file_path):
    """Obtain all entity types from Few-NERD."""
    samples = read_json_file(file_path)
    entity_types = list()

    for sample in samples:
        entity_type = sample["entity_type"].lower()
        if entity_type not in entity_types:
            entity_types.append(entity_type)

    return entity_types


def get_ace05_type_info(file_path):
    """Obtain all event types from ACE 2005."""
    samples = read_json_file(file_path)
    event_subtypes = list()

    for sample in samples:
        event_type = sample["event_type"].lower()
        split_contents = event_type.split(":")
        for split_content in split_contents:
            if "-" in split_content:
                sub_split_contents = split_content.split("-")
                for sub_split_content in sub_split_contents:
                    if sub_split_content not in event_subtypes:
                        event_subtypes.append(sub_split_content)
            else:
                if split_content not in event_subtypes:
                    event_subtypes.append(split_content)

    return event_subtypes


def get_synonyms(word, number):
    url = f"https://api.datamuse.com/words?rel_syn={word}"
    response = requests.get(url)
    synonyms = [entry["word"] for entry in response.json()]

    selected_synonyms = list()
    for synonym in synonyms:
        if len(synonym.split(" ")) == 1:
            selected_synonyms.append(synonym)

    if len(selected_synonyms) < number:
        selected_synonyms.extend([word] * (number - len(selected_synonyms)))

    return selected_synonyms


def evaluate_on_semeval(model_path, causes_path, prompt_type="ic"):
    """Evaluate the causal stability of LLMs under SemEval dataset."""
    model = LLM(model=model_path, gpu_memory_utilization=0.9, max_model_len=512)
    sampling_params = SamplingParams(temperature=0, max_tokens=512, stop=["</Instance>"])

    rec_dir = os.path.join("evaluation_results", model_path.split("/")[-1])
    rec_eval_file = os.path.join(rec_dir, f"semeval_naive_evaluation_{prompt_type}.json")
    if os.path.exists(rec_dir) and os.path.exists(rec_eval_file):
        rec_eval = read_json_file(rec_eval_file)
    else:
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        rec_eval = dict()
        rec_eval["evaluate_ids"] = list()
        rec_eval["evaluate_instances"] = dict()
        rec_eval["right_predict_ids"] = list()
        rec_eval["right_predict_history"] = dict()
        rec_eval["wrong_predict_ids"] = list()
        rec_eval["wrong_predict_history"] = dict()

        semeval_path = "datasets/semeval.json"
        semeval_insts = read_json_file(semeval_path)
        eval_prompts = list()
        eval_essentials = list()
        for inst_id, semeval_inst in enumerate(semeval_insts):
            rel_label = semeval_inst["relation_type"].lower()
            if rel_label != "other":
                rec_eval["evaluate_ids"].append(str(inst_id))
                rec_eval["evaluate_instances"][str(inst_id)] = semeval_inst
                eval_essentials.append((rel_label, str(inst_id)))

                sentence = " ".join(semeval_inst["sentence"])
                h_entity = semeval_inst["head_entity"]["span"]
                t_entity = semeval_inst["tail_entity"]["span"]
                eval_prompt = relation_extraction(sentence, h_entity, t_entity, prompt_type)
                eval_prompts.append(eval_prompt)

        eval_outputs = model.generate(eval_prompts, sampling_params)
        eval_preds = list()
        for eval_output in eval_outputs:
            eval_pred = eval_output.outputs[0].text.strip()
            if prompt_type == "cot":
                eval_pred = eval_pred.split("Relation Between the Head Entity and Tail Entity:")[-1].strip()
            eval_preds.append(eval_pred)

        assert len(eval_preds) == len(eval_essentials)
        for idx in range(len(eval_preds)):
            eval_pred = eval_preds[idx]
            rel_label, inst_id = eval_essentials[idx]

            semeval_inst = rec_eval["evaluate_instances"][inst_id]
            sentence = " ".join(semeval_inst["sentence"])
            h_entity = semeval_inst["head_entity"]["span"]
            t_entity = semeval_inst["tail_entity"]["span"]
            marked_h_entity = " ".join(["<h>", h_entity, "</h>"])
            marked_t_entity = " ".join(["<t>", t_entity, "</t>"])
            marked_sentence = sentence.replace(h_entity, marked_h_entity)
            marked_sentence = marked_sentence.replace(t_entity, marked_t_entity)

            if eval_pred == rel_label:
                rec_eval["right_predict_ids"].append(inst_id)
                rec_eval["right_predict_history"][inst_id] = {"sentence": marked_sentence,
                                                              "relation_label": rel_label,
                                                              "predict_relation": eval_pred}
            else:
                rec_eval["wrong_predict_ids"].append(inst_id)
                rec_eval["wrong_predict_history"][inst_id] = {"sentence": marked_sentence,
                                                              "relation_label": rel_label,
                                                              "predict_relation": eval_pred}

        write_json_file(rec_eval_file, rec_eval)

    rec_attack_file = os.path.join(rec_dir, causes_path.split("/")[-1].replace(".json", f"_{prompt_type}.json"))
    if os.path.exists(rec_attack_file):
        rec_attack = read_json_file(rec_attack_file)
    else:
        rec_attack = dict()
        rec_attack["success_attack_ids"] = list()
        rec_attack["success_attack_history"] = dict()
        rec_attack["failed_attack_ids"] = list()
        rec_attack["failed_attack_history"] = dict()

    semeval_path = "datasets/semeval.json"
    causes = read_json_file(causes_path)
    head_roles, tail_roles = get_semeval_relation_info(semeval_path)

    attack_prompts = list()
    attack_essentials = list()
    for inst_id, semeval_inst in rec_eval["evaluate_instances"].items():
        if inst_id in rec_eval["right_predict_ids"]:
            if inst_id not in rec_attack["success_attack_ids"]:
                if inst_id not in rec_attack["failed_attack_ids"]:
                    rel_label = semeval_inst["relation_type"].lower()
                    h_entity = semeval_inst["head_entity"]["span"]
                    t_entity = semeval_inst["tail_entity"]["span"]
                    h_entity_role = rel_label.split("-")[0]
                    t_entity_role = rel_label.split("-")[1]
                    h_entity_idx = semeval_inst["head_entity"]["start_idx"]
                    t_entity_idx = semeval_inst["tail_entity"]["start_idx"]

                    for head_role in head_roles:
                        if head_role != h_entity_role:
                            confounders = causes.get(head_role)
                            if confounders is not None:
                                for confounder in confounders:
                                    sentence_list_copy = deepcopy(semeval_inst["sentence"])
                                    sentence_list_copy.insert(h_entity_idx, confounder)
                                    attack_sentence = " ".join(sentence_list_copy)
                                    attack_prompt = relation_extraction(attack_sentence, h_entity, t_entity, prompt_type)
                                    attack_prompts.append(attack_prompt)
                                    attack_essentials.append({
                                        "instance_id": inst_id,
                                        "confounder": confounder,
                                        "attack_type": "head_attack",
                                        "relation_label": rel_label,
                                        "expected_relation": f"{head_role}-x"
                                    })

                    for tail_role in tail_roles:
                        if tail_role != t_entity_role:
                            confounders = causes.get(tail_role)
                            if confounders is not None:
                                for confounder in confounders:
                                    sentence_list_copy = deepcopy(semeval_inst["sentence"])
                                    sentence_list_copy.insert(t_entity_idx, confounder)
                                    attack_sentence = " ".join(sentence_list_copy)
                                    attack_prompt = relation_extraction(attack_sentence, h_entity, t_entity, prompt_type)
                                    attack_prompts.append(attack_prompt)
                                    attack_essentials.append({
                                        "instance_id": inst_id,
                                        "confounder": confounder,
                                        "attack_type": "tail_attack",
                                        "relation_label": rel_label,
                                        "expected_relation": f"x-{tail_role}"
                                    })

    attack_outputs = model.generate(attack_prompts, sampling_params)
    attack_preds = list()
    for attack_output in attack_outputs:
        attack_pred = attack_output.outputs[0].text.strip()
        if prompt_type == "cot":
            attack_pred = attack_pred.split("Relation Between the Head Entity and Tail Entity:")[-1].strip()
        attack_preds.append(attack_pred)

    assert len(attack_preds) == len(attack_essentials)
    success_attack_history = list()
    failed_attack_history = list()
    curr_inst_id = None
    is_attack = False

    for idx in range(len(attack_preds)):
        attack_pred = attack_preds[idx]
        inst_id = attack_essentials[idx]["instance_id"]
        confounder = attack_essentials[idx]["confounder"]
        attack_type = attack_essentials[idx]["attack_type"]
        rel_label = attack_essentials[idx]["relation_label"]
        expected_rel = attack_essentials[idx]["expected_relation"]

        semeval_inst = rec_eval["evaluate_instances"][inst_id]
        sentence = " ".join(semeval_inst["sentence"])
        h_entity = semeval_inst["head_entity"]["span"]
        t_entity = semeval_inst["tail_entity"]["span"]
        marked_h_entity = " ".join(["<h>", h_entity, "</h>"])
        marked_t_entity = " ".join(["<t>", t_entity, "</t>"])
        marked_sentence = sentence.replace(h_entity, marked_h_entity)
        marked_sentence = marked_sentence.replace(t_entity, marked_t_entity)

        marked_confounder = " ".join(["<c>", confounder, "</c>"])
        if attack_type == "head_attack":
            marked_attack_sentence = marked_sentence.replace(marked_h_entity, " ".join([marked_confounder, marked_h_entity]))
        else:
            marked_attack_sentence = marked_sentence.replace(marked_t_entity, " ".join([marked_confounder, marked_t_entity]))

        attack_history = {"sentence": marked_attack_sentence,
                          "relation_label": rel_label,
                          "expected_relation": expected_rel,
                          "prediction_relation": attack_pred}

        if inst_id != curr_inst_id:
            if curr_inst_id is not None:
                success_attack_history_copy = deepcopy(success_attack_history)
                failed_attack_history_copy = deepcopy(failed_attack_history)
                if is_attack:
                    rec_attack["success_attack_ids"].append(curr_inst_id)
                    rec_attack["success_attack_history"][curr_inst_id] = {"success_attack": success_attack_history_copy,
                                                                          "failed_attack": failed_attack_history_copy}
                else:
                    assert len(success_attack_history) == 0
                    rec_attack["failed_attack_ids"].append(curr_inst_id)
                    rec_attack["failed_attack_history"][curr_inst_id] = failed_attack_history_copy

                success_attack_history.clear()
                failed_attack_history.clear()
                curr_inst_id = inst_id
                is_attack = False

                if attack_pred != rel_label:
                    is_attack = True
                    success_attack_history.append(attack_history)
                else:
                    failed_attack_history.append(attack_history)
            else:
                if attack_pred != rel_label:
                    is_attack = True
                    success_attack_history.append(attack_history)
                else:
                    failed_attack_history.append(attack_history)
                curr_inst_id = inst_id
        else:
            if attack_pred != rel_label:
                is_attack = True
                success_attack_history.append(attack_history)
            else:
                failed_attack_history.append(attack_history)

    if is_attack:
        rec_attack["success_attack_ids"].append(curr_inst_id)
        rec_attack["success_attack_history"][curr_inst_id] = {"success_attack": success_attack_history,
                                                              "failed_attack": failed_attack_history}
    else:
        assert len(success_attack_history) == 0
        rec_attack["failed_attack_ids"].append(curr_inst_id)
        rec_attack["failed_attack_history"][curr_inst_id] = failed_attack_history

    write_json_file(rec_attack_file, rec_attack)


def evaluate_on_few_nerd(model_path, causes_path, prompt_type="ic"):
    """Evaluate the causal stability of LLM under Few_NERD dataset."""
    model = LLM(model=model_path, gpu_memory_utilization=0.9, max_model_len=512)
    sampling_params = SamplingParams(temperature=0, max_tokens=512, stop=["</Instance>"])

    rec_dir = os.path.join("evaluation_results", model_path.split("/")[-1])
    rec_eval_file = os.path.join(rec_dir, f"few_nerd_naive_evaluation_{prompt_type}.json")
    if os.path.exists(rec_dir) and os.path.exists(rec_eval_file):
        rec_eval = read_json_file(rec_eval_file)
    else:
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        rec_eval = dict()
        rec_eval["evaluate_ids"] = list()
        rec_eval["evaluate_instances"] = dict()
        rec_eval["right_predict_ids"] = list()
        rec_eval["right_predict_history"] = dict()
        rec_eval["wrong_predict_ids"] = list()
        rec_eval["wrong_predict_history"] = dict()

        few_nerd_path = "datasets/few_nerd.json"
        few_nerd_insts = read_json_file(few_nerd_path)
        eval_prompts = list()
        eval_essentials = list()
        for inst_id, few_nerd_inst in enumerate(few_nerd_insts):
            type_label = few_nerd_inst["entity_type"].lower()
            rec_eval["evaluate_ids"].append(str(inst_id))
            rec_eval["evaluate_instances"][str(inst_id)] = few_nerd_inst
            eval_essentials.append((type_label, str(inst_id)))

            sentence = " ".join(few_nerd_inst["sentence"])
            entity = few_nerd_inst["entity"]["span"]
            eval_prompt = entity_typing(sentence, entity, prompt_type)
            eval_prompts.append(eval_prompt)

        eval_outputs = model.generate(eval_prompts, sampling_params)
        eval_preds = list()
        for eval_output in eval_outputs:
            eval_pred = eval_output.outputs[0].text.strip()
            eval_preds.append(eval_pred)

        assert len(eval_preds) == len(eval_essentials)
        for idx in range(len(eval_preds)):
            eval_pred = eval_preds[idx]
            type_label, inst_id = eval_essentials[idx]

            few_nerd_inst = rec_eval["evaluate_instances"][inst_id]
            sentence = " ".join(few_nerd_inst["sentence"])
            entity = few_nerd_inst["entity"]["span"]
            marked_entity = " ".join(["<e>", entity, "</e>"])
            marked_sentence = sentence.replace(entity, marked_entity)

            if eval_pred == type_label:
                rec_eval["right_predict_ids"].append(inst_id)
                rec_eval["right_predict_history"][inst_id] = {"sentence": marked_sentence,
                                                              "type_label": type_label,
                                                              "predict_type": eval_pred}
            else:
                rec_eval["wrong_predict_ids"].append(inst_id)
                rec_eval["wrong_predict_history"][inst_id] = {"sentence": marked_sentence,
                                                              "type_label": type_label,
                                                              "predict_type": eval_pred}

        write_json_file(rec_eval_file, rec_eval)

    rec_attack_file = os.path.join(rec_dir, causes_path.split("/")[-1].replace(".json", f"_{prompt_type}.json"))
    if os.path.exists(rec_attack_file):
        rec_attack = read_json_file(rec_attack_file)
    else:
        rec_attack = dict()
        rec_attack["success_attack_ids"] = list()
        rec_attack["success_attack_history"] = dict()
        rec_attack["failed_attack_ids"] = list()
        rec_attack["failed_attack_history"] = dict()

    few_nerd_path = "datasets/few_nerd.json"
    causes = read_json_file(causes_path)
    entity_types = get_few_nerd_type_info(few_nerd_path)

    attack_prompts = list()
    attack_essentials = list()
    for inst_id, few_nerd_inst in rec_eval["evaluate_instances"].items():
        if inst_id in rec_eval["right_predict_ids"]:
            if inst_id not in rec_attack["success_attack_ids"]:
                if inst_id not in rec_attack["failed_attack_ids"]:
                    type_label = few_nerd_inst["entity_type"].lower()
                    entity = few_nerd_inst["entity"]["span"]
                    entity_end_idx = few_nerd_inst["entity"]["end_idx"]

                    for entity_type in entity_types:
                        if entity_type != type_label:
                            confounders = causes.get(entity_type)
                            if confounders is not None:
                                for confounder in confounders:
                                    sentence_list_copy = deepcopy(few_nerd_inst["sentence"])
                                    pos_tags = nltk.pos_tag(sentence_list_copy)
                                    insert_idx = None
                                    for candidate_idx in range(entity_end_idx + 1, len(pos_tags)):
                                        _, pos_tag = pos_tags[candidate_idx]
                                        if pos_tag[0] == "N":
                                            insert_idx = candidate_idx
                                            break

                                    if insert_idx is not None:
                                        sentence_list_copy.insert(insert_idx, confounder)
                                    else:
                                        insert_idx = len(sentence_list_copy) - 1
                                        sentence_list_copy.insert(insert_idx, confounder)

                                    attack_sentence = " ".join(sentence_list_copy)
                                    attack_prompt = entity_typing(attack_sentence, entity, prompt_type)
                                    attack_prompts.append(attack_prompt)
                                    attack_essentials.append({
                                        "instance_id": inst_id,
                                        "confounder": confounder,
                                        "insert_pos": insert_idx,
                                        "type_label": type_label,
                                        "expected_type": f"{entity_type}"})

    attack_prompts = attack_prompts
    attack_essentials = attack_essentials
    attack_outputs = model.generate(attack_prompts, sampling_params)
    attack_preds = list()
    for attack_output in attack_outputs:
        attack_pred = attack_output.outputs[0].text.strip()
        attack_preds.append(attack_pred)

    assert len(attack_preds) == len(attack_essentials)
    success_attack_history = list()
    failed_attack_history = list()
    curr_inst_id = None
    is_attack = False

    for idx in range(len(attack_preds)):
        attack_pred = attack_preds[idx]
        inst_id = attack_essentials[idx]["instance_id"]
        confounder = attack_essentials[idx]["confounder"]
        insert_pos = attack_essentials[idx]["insert_pos"]
        type_label = attack_essentials[idx]["type_label"]
        expected_type = attack_essentials[idx]["expected_type"]

        few_nerd_inst_copy = deepcopy(rec_eval["evaluate_instances"][inst_id])
        sentence_list = few_nerd_inst_copy["sentence"]
        marked_confounder = " ".join(["<c>", confounder, "</c>"])
        sentence_list.insert(insert_pos, marked_confounder)
        attack_sentence = " ".join(sentence_list)

        entity = few_nerd_inst_copy["entity"]["span"]
        marked_entity = " ".join(["<e>", entity, "</e>"])
        marked_attack_sentence = attack_sentence.replace(entity, marked_entity)
        attack_history = {"sentence": marked_attack_sentence,
                          "type_label": type_label,
                          "expected_type": expected_type,
                          "prediction_type": attack_pred}

        if inst_id != curr_inst_id:
            if curr_inst_id is not None:
                success_attack_history_copy = deepcopy(success_attack_history)
                failed_attack_history_copy = deepcopy(failed_attack_history)
                if is_attack:
                    rec_attack["success_attack_ids"].append(curr_inst_id)
                    rec_attack["success_attack_history"][curr_inst_id] = {"success_attack": success_attack_history_copy,
                                                                          "failed_attack": failed_attack_history_copy}
                else:
                    assert len(success_attack_history) == 0
                    rec_attack["failed_attack_ids"].append(curr_inst_id)
                    rec_attack["failed_attack_history"][curr_inst_id] = failed_attack_history_copy

                success_attack_history.clear()
                failed_attack_history.clear()
                curr_inst_id = inst_id
                is_attack = False

                if attack_pred != type_label:
                    is_attack = True
                    success_attack_history.append(attack_history)
                else:
                    failed_attack_history.append(attack_history)
            else:
                if attack_pred != type_label:
                    is_attack = True
                    success_attack_history.append(attack_history)
                else:
                    failed_attack_history.append(attack_history)
                curr_inst_id = inst_id
        else:
            if attack_pred != type_label:
                is_attack = True
                success_attack_history.append(attack_history)
            else:
                failed_attack_history.append(attack_history)

    if is_attack:
        rec_attack["success_attack_ids"].append(curr_inst_id)
        rec_attack["success_attack_history"][curr_inst_id] = {"success_attack": success_attack_history,
                                                              "failed_attack": failed_attack_history}
    else:
        assert len(success_attack_history) == 0
        rec_attack["failed_attack_ids"].append(curr_inst_id)
        rec_attack["failed_attack_history"][curr_inst_id] = failed_attack_history

    write_json_file(rec_attack_file, rec_attack)


def evaluate_on_ace05(model_path, causes_path, prompt_type="ic"):
    """Evaluate the causal stability of LLM under ACE 2005 dataset."""
    model = LLM(model=model_path, gpu_memory_utilization=0.9, max_model_len=600)
    sampling_params = SamplingParams(temperature=0, max_tokens=600, stop=["</Instance>"])

    rec_dir = os.path.join("evaluation_results", model_path.split("/")[-1])
    rec_eval_file = os.path.join(rec_dir, f"ace05_naive_evaluation_{prompt_type}.json")
    if os.path.exists(rec_dir) and os.path.exists(rec_eval_file):
        rec_eval = read_json_file(rec_eval_file)
    else:
        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        rec_eval = dict()
        rec_eval["evaluate_ids"] = list()
        rec_eval["evaluate_instances"] = dict()
        rec_eval["right_predict_ids"] = list()
        rec_eval["right_predict_history"] = dict()
        rec_eval["wrong_predict_ids"] = list()
        rec_eval["wrong_predict_history"] = dict()

        ace05_path = "datasets/ace05.json"
        ace05_insts = read_json_file(ace05_path)
        eval_prompts = list()
        eval_essentials = list()
        for inst_id, ace05_inst in enumerate(ace05_insts):
            type_label = ace05_inst["event_type"].lower()
            rec_eval["evaluate_ids"].append(str(inst_id))
            rec_eval["evaluate_instances"][str(inst_id)] = ace05_inst
            eval_essentials.append((type_label, str(inst_id)))

            sentence = " ".join(ace05_inst["sentence"])
            trigger = ace05_inst["trigger"]["text"]
            eval_prompt = event_detection(sentence, trigger, prompt_type)
            eval_prompts.append(eval_prompt)

        eval_outputs = model.generate(eval_prompts, sampling_params)
        eval_preds = list()
        for eval_output in eval_outputs:
            eval_pred = eval_output.outputs[0].text.strip()
            eval_preds.append(eval_pred)

        assert len(eval_preds) == len(eval_essentials)
        for idx in range(len(eval_preds)):
            eval_pred = eval_preds[idx]
            type_label, inst_id = eval_essentials[idx]

            ace05_inst = rec_eval["evaluate_instances"][inst_id]
            sentence = " ".join(ace05_inst["sentence"])
            trigger = ace05_inst["trigger"]["text"]
            marked_trigger = " ".join(["<t>", trigger, "</t>"])
            marked_sentence = sentence.replace(trigger, marked_trigger)

            if eval_pred == type_label:
                rec_eval["right_predict_ids"].append(inst_id)
                rec_eval["right_predict_history"][inst_id] = {"sentence": marked_sentence,
                                                              "type_label": type_label,
                                                              "predict_type": eval_pred}
            else:
                rec_eval["wrong_predict_ids"].append(inst_id)
                rec_eval["wrong_predict_history"][inst_id] = {"sentence": marked_sentence,
                                                              "type_label": type_label,
                                                              "predict_type": eval_pred}

        write_json_file(rec_eval_file, rec_eval)

    rec_attack_file = os.path.join(rec_dir, causes_path.split("/")[-1].replace(".json", f"_{prompt_type}.json"))
    if os.path.exists(rec_attack_file):
        rec_attack = read_json_file(rec_attack_file)
    else:
        rec_attack = dict()
        rec_attack["success_attack_ids"] = list()
        rec_attack["success_attack_history"] = dict()
        rec_attack["failed_attack_ids"] = list()
        rec_attack["failed_attack_history"] = dict()

    ace05_path = "datasets/ace05.json"
    causes = read_json_file(causes_path)
    event_subtypes = get_ace05_type_info(ace05_path)

    attack_prompts = list()
    attack_essentials = list()
    for inst_id, ace05_inst in rec_eval["evaluate_instances"].items():
        if inst_id in rec_eval["right_predict_ids"]:
            if inst_id not in rec_attack["success_attack_ids"]:
                if inst_id not in rec_attack["failed_attack_ids"]:
                    type_label = ace05_inst["event_type"].lower()
                    trigger = ace05_inst["trigger"]["text"]
                    trigger_end_idx = ace05_inst["trigger"]["end"] - 1

                    for event_subtype in event_subtypes:
                        if event_subtype not in type_label:
                            confounders = causes.get(event_subtype)
                            if confounders is not None:
                                for confounder in confounders:
                                    sentence_list_copy = deepcopy(ace05_inst["sentence"])
                                    pos_tags = nltk.pos_tag(sentence_list_copy)
                                    insert_idx = None
                                    for candidate_idx in range(trigger_end_idx + 1, len(pos_tags)):
                                        _, pos_tag = pos_tags[candidate_idx]
                                        if pos_tag[0] == "N":
                                            insert_idx = candidate_idx
                                            break

                                    if insert_idx is not None:
                                        sentence_list_copy.insert(insert_idx, confounder)
                                    else:
                                        insert_idx = len(sentence_list_copy) - 1
                                        sentence_list_copy.insert(insert_idx, confounder)

                                    attack_sentence = " ".join(sentence_list_copy)
                                    attack_prompt = event_detection(attack_sentence, trigger, prompt_type)
                                    attack_prompts.append(attack_prompt)
                                    attack_essentials.append({
                                        "instance_id": inst_id,
                                        "confounder": confounder,
                                        "insert_pos": insert_idx,
                                        "type_label": type_label,
                                        "expected_type": f"{event_subtype}"})

    attack_outputs = model.generate(attack_prompts, sampling_params)
    attack_preds = list()
    for attack_output in attack_outputs:
        attack_pred = attack_output.outputs[0].text.strip()
        attack_preds.append(attack_pred)

    assert len(attack_preds) == len(attack_essentials)
    success_attack_history = list()
    failed_attack_history = list()
    curr_inst_id = None
    is_attack = False

    for idx in range(len(attack_preds)):
        attack_pred = attack_preds[idx]
        inst_id = attack_essentials[idx]["instance_id"]
        confounder = attack_essentials[idx]["confounder"]
        insert_pos = attack_essentials[idx]["insert_pos"]
        type_label = attack_essentials[idx]["type_label"]
        expected_type = attack_essentials[idx]["expected_type"]

        ace05_inst_copy = deepcopy(rec_eval["evaluate_instances"][inst_id])
        sentence_list = ace05_inst_copy["sentence"]
        marked_confounder = " ".join(["<c>", confounder, "</c>"])
        sentence_list.insert(insert_pos, marked_confounder)
        attack_sentence = " ".join(sentence_list)

        trigger = ace05_inst_copy["trigger"]["text"]
        marked_trigger = " ".join(["<t>", trigger, "</t>"])
        marked_attack_sentence = attack_sentence.replace(trigger, marked_trigger)
        attack_history = {"sentence": marked_attack_sentence,
                          "type_label": type_label,
                          "expected_type": expected_type,
                          "prediction_type": attack_pred}

        if inst_id != curr_inst_id:
            if curr_inst_id is not None:
                success_attack_history_copy = deepcopy(success_attack_history)
                failed_attack_history_copy = deepcopy(failed_attack_history)
                if is_attack:
                    rec_attack["success_attack_ids"].append(curr_inst_id)
                    rec_attack["success_attack_history"][curr_inst_id] = {"success_attack": success_attack_history_copy,
                                                                          "failed_attack": failed_attack_history_copy}
                else:
                    assert len(success_attack_history) == 0
                    rec_attack["failed_attack_ids"].append(curr_inst_id)
                    rec_attack["failed_attack_history"][curr_inst_id] = failed_attack_history_copy

                success_attack_history.clear()
                failed_attack_history.clear()
                curr_inst_id = inst_id
                is_attack = False

                if attack_pred != type_label:
                    is_attack = True
                    success_attack_history.append(attack_history)
                else:
                    failed_attack_history.append(attack_history)
            else:
                if attack_pred != type_label:
                    is_attack = True
                    success_attack_history.append(attack_history)
                else:
                    failed_attack_history.append(attack_history)
                curr_inst_id = inst_id
        else:
            if attack_pred != type_label:
                is_attack = True
                success_attack_history.append(attack_history)
            else:
                failed_attack_history.append(attack_history)

    if is_attack:
        rec_attack["success_attack_ids"].append(curr_inst_id)
        rec_attack["success_attack_history"][curr_inst_id] = {"success_attack": success_attack_history,
                                                              "failed_attack": failed_attack_history}
    else:
        assert len(success_attack_history) == 0
        rec_attack["failed_attack_ids"].append(curr_inst_id)
        rec_attack["failed_attack_history"][curr_inst_id] = failed_attack_history

    write_json_file(rec_attack_file, rec_attack)


if __name__ == '__main__':
    evaluate_on_semeval("../llms/qwen2-7b-instruct-gptq-int8", "cause_concepts/semeval_qwen2-7b-instruct-gptq-int8_s1.3_t0.3_top10.json", "ic")
    # evaluate_on_few_nerd("../llms/qwen2-7b-instruct-gptq-int8", "cause_concepts/semeval_qwen2-7b-instruct-gptq-int8_s1.3_t0.3_top10.json", "ic")
    # evaluate_on_ace05("../llms/qwen2-7b-instruct-gptq-int8", "cause_concepts/semeval_qwen2-7b-instruct-gptq-int8_s1.3_t0.3_top10.json", "ic")
