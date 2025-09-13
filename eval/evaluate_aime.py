
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import contextlib
import tempfile
import multiprocessing
from sympy.parsing.latex import parse_latex
from sympy import Symbol, simplify
from tools.grader import math_equal
import pandas
import signal
import json
import os
import re
import sys
import argparse
import glob

def read_text_data(datapath):
    print("reading from %s" % datapath)
    data_list = []
    with open(datapath, "r") as f:
        for line in f:
            data_list.append(json.loads(line.strip())['output'])
    
    return data_list


def read_jsonl_data(datapath):
    print("reading from %s" % datapath)
    data_list = []
    with open(datapath, "r") as f:
        for line in f:
            data_item = json.loads(line.strip())
            data_list.append(data_item['output'])

    return data_list


def read_json_data(datapath):
    print("reading from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    return data_list


def is_completely_wrapped_by_text(input_string):
    pattern = r'^\\text{(.*)}$'
    match = re.match(pattern, input_string)
    if match:
        ## input_string is completely wrapped by \text{}
        extracted_content = match.group(1)
        extracted_content = extracted_content.replace("(", "").replace(")", "").replace(",", "")
        return extracted_content
    else:
        return None


def math_answer_cleaning(answer):
    ## remove irrelevant text and space to see whether it is exact match
    extracted_content = is_completely_wrapped_by_text(answer)
    answer = extracted_content if extracted_content else answer

    ## convert 5,\!460 into 5460; convert 14{,}916 into 14916; convert \$4 into 4; convert 50\\\% into 50
    answer = answer.replace(",\!", "").replace("{,}", "").replace("\$", "")
    ## convert \dfrac{3}{2} into frac{3}{2}
    answer = answer.replace("dfrac{", "frac{").replace("tfrac{", "frac{")
    ## convert 121^\circ into 121
    answer = answer.replace("^\circ", "")
    answer = answer.replace("^{\circ}", "")
    ## remove \quad
    answer = answer.replace("\quad", "")
    ## convert 558\,\text{nm} into 558
    answer = re.sub(r'\\,\\text\{.*?\}', '', answer)
    ## convert 558\text{nm} into 558
    answer = re.sub(r'\\text\{.*?\}', '', answer)
    ## convert 2.45e6^{-1} into 2.45e6; "15000^{-2}^{-1}" into "15000"
    answer = re.sub(r'(\s\^\{-\d+\})', '', answer)
    ## remove space
    answer = answer.replace(" ", "")
    ## remove \n
    answer = answer.replace("\n", "").replace("\\n", "")
    ## convert 3.54\times10^{10} into 3.54e10
    answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^{([+-]?\d+)}', r'\1e\2', answer)
    ## convert 3.54\times10^10 into 3.54e10
    answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^([+-]?\d+)', r'\1e\2', answer)
    ## convert 2^{10} into 2^10
    answer = re.sub(r'(\d+)\^{(\d+)}', r'\1^\2', answer)
    ## convert 10^{-5} into 1e-5; 10^{5} into 1e5
    answer = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", answer)
    ## remove comma
    answer = answer.replace(",", "")
    ## lowercase
    answer = answer.lower()

    ## convert 7.04e5\ into 7.04e5
    if answer.endswith("\\"):
        answer = answer[:-1]
    
    ## convert f(x)=ax+b into ax+b; convert z=123 into 123; convert t_r=123 into 123
    func_pattern = r'^[a-zA-Z_]\w*\([a-zA-Z_]\w*\)$'
    if "=" in answer and (re.match(func_pattern, answer.split("=")[0]) or len(answer.split("=")[0])<=3):
        answer = answer.split("=", 1)[1]

    return answer


def round_number(answer):
    def _is_float(string):
        try:
            float(string)
            return True
        except:
            return False

    if _is_float(answer) and float(answer) < 1:
        ## to consider the case like 5.56e-10 (convert 5.56e-10 into 5.6e-10)
        ## still return a string type
        return f"{float(answer):.2g}"
    
    return answer


def evaluate_math500_zeroshot(input_datapath, test_datapath):

    class _TimeoutException(Exception):
        pass

    def _timeout_handler(signum, frame):
        # raise Exception("Function took too long to complete.")
        raise _TimeoutException

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    gold_list = []
    print("reading from %s" % test_datapath)

    # suppress_prints()
    with open(test_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            answer = item['answer']
            gold_list.append(answer)

    count_output_none = 0
    count_answer_none = 0
    count_timeout = 0
    correct = 0
    print("reading from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)['output']

            # match = re.search(pattern1, line)
            matches1 = pattern1_re.findall(line)
            matches2 = pattern2_re.findall(line)
            matches3 = pattern3_re.findall(line)
            matches4 = pattern4_re.findall(line)
            matches5 = pattern5_re.findall(line)

            if len(matches1) >= 1:
                extracted_answer = matches1[-1]
            elif len(matches2) >= 1:
                extracted_answer = matches2[-1]
            elif len(matches3) >= 1:
                extracted_answer = matches3[-1]
            elif len(matches4) >= 1:
                extracted_answer = matches4[-1]
            elif len(matches5) >= 1:
                extracted_answer = matches5[-1]
            else:
                extracted_answer = None
            gold = gold_list[i]
            
            if extracted_answer is None:
                count_output_none += 1
                continue
            if gold is None:
                count_answer_none += 1
                continue

            extracted_answer = math_answer_cleaning(extracted_answer)
            gold = math_answer_cleaning(gold)

            try:
                # raise exception after 5 sections
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5)

                if math_equal(extracted_answer, gold):
                    correct += 1
                elif is_equal_after_calculation(extracted_answer, gold):
                    correct += 1
                elif check_after_fraction_mapping(extracted_answer, gold):
                    correct += 1
                
                ## Disable the alarm
                signal.alarm(0)
            
            except:
                ## Disable the alarm
                signal.alarm(0)
                count_timeout += 1

    # restore_prints()

    acc = correct / len(gold_list)
    print("len(gold_list):", len(gold_list))
    print("count_output_none:", count_output_none)
    print("count_timeout:", count_timeout)
    print("count_answer_none:", count_answer_none)
    print("accuracy:", acc)

    return acc



def calculate_numbers(input_string):
    try:
        result = eval(input_string)
        return result
    except:
        return None


def is_equal_after_calculation(extracted_answer, gold):
    ## convert \frac{3}{2} into 3/2
    gold = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', gold)
    extracted_answer = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', extracted_answer)
    gold_result = calculate_numbers(gold)
    extracted_answer_result = calculate_numbers(extracted_answer)

    if gold_result and gold_result == extracted_answer_result:
        return True
    else:
        return False


def is_math_formula_equal(extracted_answer, gold):

    try:
        extracted_answer_expr = parse_latex(extracted_answer)
        gold_expr = parse_latex(gold)

        return simplify(extracted_answer_expr - gold_expr) == 0
    
    except Exception as e:
        print("error:", e)
        return False


def check_after_fraction_mapping(extracted_answer, gold):
    return re.sub(r'\\frac{(.*?)}{(.*?)}', r'\1/\2', extracted_answer) == re.sub(r'\\frac{(.*?)}{(.*?)}', r'\1/\2', gold)


def suppress_prints():
    sys.stdout = open(os.devnull, 'w')


def restore_prints():
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def evaluate_amc23_or_aime24_zeroshot(input_datapath, test_datapath):

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    gold_list = []
    question_list = []
    print("reading from %s" % test_datapath)
    with open(test_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            answer = str(item['answer'])
            gold_list.append(answer)
            question_list.append(item['problem'])

    count_output_none = 0
    count_answer_none = 0
    correct = 0
    print("reading from %s" % input_datapath)

    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)['output']

            matches1 = pattern1_re.findall(line)
            matches2 = pattern2_re.findall(line)
            matches3 = pattern3_re.findall(line)
            matches4 = pattern4_re.findall(line)
            matches5 = pattern5_re.findall(line)

            if len(matches1) >= 1:
                extracted_answer = matches1[-1]
            elif len(matches2) >= 1:
                extracted_answer = matches2[-1]
            elif len(matches3) >= 1:
                extracted_answer = matches3[-1]
            elif len(matches4) >= 1:
                extracted_answer = matches4[-1]
            elif len(matches5) >= 1:
                extracted_answer = matches5[-1]
            else:
                extracted_answer = None
            gold = gold_list[i]
            
            if extracted_answer is None:
                count_output_none += 1
                continue
            if gold is None:
                count_answer_none += 1
                continue

            extracted_answer = math_answer_cleaning(extracted_answer)
            gold = math_answer_cleaning(gold)

            if math_equal(extracted_answer, gold):
                correct += 1
            elif round_number(extracted_answer) == round_number(gold):
                correct += 1
            elif is_equal_after_calculation(extracted_answer, gold):
                correct += 1

    acc = correct / len(gold_list)
    print("count_output_none:", count_output_none)
    print("count_answer_none:", count_answer_none)
    print("accuracy:", acc)
    
    return acc


def get_answer_by_marjority_voting(output_list):
    
    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    answer_dict = {}
    for output in output_list:
        matches1 = pattern1_re.findall(output)
        matches2 = pattern2_re.findall(output)
        matches3 = pattern3_re.findall(output)
        matches4 = pattern4_re.findall(output)
        matches5 = pattern5_re.findall(output)

        if len(matches1) >= 1:
            extracted_answer = matches1[-1]
        elif len(matches2) >= 1:
            extracted_answer = matches2[-1]
        elif len(matches3) >= 1:
            extracted_answer = matches3[-1]
        elif len(matches4) >= 1:
            extracted_answer = matches4[-1]
        elif len(matches5) >= 1:
            extracted_answer = matches5[-1]
        else:
            extracted_answer = None

        if extracted_answer is None:
            continue
        
        extracted_answer = math_answer_cleaning(extracted_answer)

        has_found = False
        for prev_ans in answer_dict:
            if extracted_answer == prev_ans:
                answer_dict[prev_ans]['count'] += 1
                has_found = True
                break
            elif math_equal(extracted_answer, prev_ans):
                answer_dict[prev_ans]['count'] += 1
                has_found = True
                break
            elif check_after_fraction_mapping(extracted_answer, prev_ans):
                answer_dict[prev_ans]['count'] += 1
                has_found = True
                break
            elif round_number(extracted_answer) == round_number(prev_ans):
                answer_dict[prev_ans]['count'] += 1
                has_found = True
                break
            elif is_equal_after_calculation(extracted_answer, prev_ans):
                answer_dict[prev_ans]['count'] += 1
                has_found = True
                break

        if not has_found:
            answer_dict[extracted_answer] = {"count": 1, "original_output": output}

    ## rank the answer based on count
    sorted_answers = sorted(answer_dict, key=lambda x: answer_dict[x]["count"], reverse=True)

    return answer_dict[sorted_answers[0]]


def evaluate_gpqa(input_datapath, test_datapath):
    class _TimeoutException(Exception):
        pass

    def _timeout_handler(signum, frame):
        # raise Exception("Function took too long to complete.")
        raise _TimeoutException

    print("reading txt...")
    output_list = read_text_data(input_datapath)
    print("reading json...")
    gold_list = read_json_data(test_datapath)
    print("reading finished...")

    num_samples = len(gold_list)
    assert len(output_list) == len(gold_list) == num_samples

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern1_re = re.compile(pattern1, re.DOTALL)

    pattern2_re = re.compile(r'\b(?:Answer|Final Answer|ANSWER)\b[:\s\*]*\(?([A-D])\)?')

    count_none = 0
    count_timeout = 0
    correct = 0
    
    for output, gold in zip(output_list, gold_list):
        choices = [gold['choice_A'], gold['choice_B'], gold['choice_C'], gold['choice_D']]
        correct_answer = gold["correct_answer"]
        correct_index = choices.index(correct_answer)
        correct_choice = "ABCD"[correct_index]
        
        matches1 = pattern1_re.findall(output)
        matches2 = pattern2_re.findall(output)
        if len(matches1) >= 1:
            extracted_answer = matches1[-1]
        elif len(matches2) >= 1:
            extracted_answer = matches2[-1]
        else:
            extracted_answer = None
        
        if extracted_answer is None:
            count_none += 1
            continue
        
        correct_answer = math_answer_cleaning(correct_answer)
        extracted_answer = math_answer_cleaning(extracted_answer)
        
        try:

            if extracted_answer.lower() == correct_choice.lower():
                correct += 1
            elif "("+correct_choice+")" in extracted_answer:
                ## (A/B/C/D) in extracted_answer
                correct += 1

        except:
            count_timeout += 1

    acc = correct / num_samples
    print("num_samples:", num_samples)
    print("count_none:", count_none)
    print("accuracy:", acc)

    return acc
def get_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--modelfolder", type=str, default=None, help="model_folder")
    parser.add_argument("--dataset", type=str, default=None, help="test data path")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    model_folder = args.modelfolder
    test_datapath = args.dataset

    # Get all prediction files from model folder
    input_datapaths = glob.glob(os.path.join(model_folder, "seed*.jsonl"))
    print(f"Found {len(input_datapaths)} prediction files")

    accs = []
    acc_tmp = 0
    for input_datapath in input_datapaths:
        acc = evaluate_amc23_or_aime24_zeroshot(input_datapath, test_datapath)
        accs.append(acc)
        acc_tmp += acc

    print(len(accs))
    avg_acc = acc_tmp / len(input_datapaths)
    std = np.std(accs) if len(accs) > 1 else 0

    # Save results to a JSON file
    results = {
        "accuracies": [float(acc) for acc in accs],
        "average": float(avg_acc),
        "std": float(std)
    }
        
    # Save to JSON file
    results_path = os.path.join(model_folder, "accuracy_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_path}")

    print("-"*80)
    print(f"Individual accuracies: {accs}")
    print(f"Average accuracy: {avg_acc:.4f} ± {std:.4f}")

    # randomly select 8 samples to calculate the accuracy
    # do it 10 times
    import random
    print("-"*80)
    for i in range(10):
        random.shuffle(accs)
        temp_accs = accs[:32]
        avg_acc = np.mean(temp_accs)
        std = np.std(temp_accs) if len(temp_accs) > 1 else 0
        print(f"Average accuracy: {avg_acc:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
