import math

from file_io import read_json_file


def compute_results(eval_path, attack_path):
    eval_file = read_json_file(eval_path)
    attack_file = read_json_file(attack_path)

    num_eval = len(eval_file["evaluate_ids"])
    num_eval_right = len(eval_file["right_predict_ids"])
    acc = (num_eval_right / num_eval) * 100

    num_at1 = 0
    num_at2 = 0
    num_at3 = 0
    for record in attack_file["success_attack_history"].values():
        num_wrong = len(record["success_attack"])
        num_correct = len(record["failed_attack"])
        wrong_ratio = math.ceil(num_wrong / (num_wrong + num_correct) * 100)
        if wrong_ratio >= 3:
            num_at1 += 1
            num_at2 += 1
            num_at3 += 1
        elif 2 <= wrong_ratio < 3:
            num_at1 += 1
            num_at2 += 1
        elif 1 <= wrong_ratio < 2:
            num_at1 += 1

    instab_at1 = num_at1 / num_eval_right
    instab_at2 = num_at2 / num_eval_right
    instab_at3 = num_at3 / num_eval_right
    avg_instab = (instab_at1 + instab_at2 + instab_at3) / 3 * 100

    acc_at1 = acc - acc * instab_at1
    acc_at2 = acc - acc * instab_at2
    acc_at3 = acc - acc * instab_at3
    avg_acc_at = (acc_at1 + acc_at2 + acc_at3) / 3
    return acc, instab_at1, instab_at2, instab_at3, avg_instab, acc_at1, acc_at2, acc_at3, avg_acc_at


def analyze_results(eval_path, attack_path):
    acc, ins_at1, ins_at2, ins_at3, avg_ins, acc_at1, acc_at2, acc_at3, avg_acc = compute_results(eval_path, attack_path)
    print("acc:", acc)
    print("ins@1:", ins_at1 * 100)
    print("ins@2:", ins_at2 * 100)
    print("ins@3:", ins_at3 * 100)
    print("avg_ins@:", avg_ins)
    print("acc@1:", acc_at1)
    print("acc@2:", acc_at2)
    print("acc@3:", acc_at3)
    print("avg_acc@:", avg_acc)


if __name__ == '__main__':
    analyze_results("evaluation_results/qwen2-7b-instruct-gptq-int8/semeval_naive_evaluation_ic.json",
                    "evaluation_results/qwen2-7b-instruct-gptq-int8/semeval_qwen2-7b-instruct-gptq-int8_s1.3_t0.3_top10_ic.json")
