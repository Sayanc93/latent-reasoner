from vllm import LLM, SamplingParams
from tqdm import tqdm
import copy
import torch
import os
import json

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args(add_evaluation=False):
    parser = argparse.ArgumentParser(description="llm_config")

    ## model & tokenizer
    parser.add_argument('--output-folder', type=str, default=None)
    parser.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    parser.add_argument('--tokenizer-model', type=str, default=None)
    ## dataset path
    parser.add_argument('--datapath', type=str, default='')

    ## others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device-id', type=str, default=None)
    
    if add_evaluation:
        parser = _add_evaluation_argument(parser)

    args = parser.parse_args()

    return args


def _add_evaluation_argument(parser):
    group = parser.add_argument_group(title='evaluation')

    ## generation
    group.add_argument('--model-type', type=str, required=True)
    group.add_argument('--temperature', type=float, default=0)
    group.add_argument('--topk', type=int, default=1)
    group.add_argument('--topp', type=float, default=1)
    group.add_argument('--max-output-len', type=int, default=2048)
    group.add_argument('--start-idx', type=int, default=-1)
    group.add_argument('--end-idx', type=int, default=-1)
    group.add_argument('--tensor-parallel-size', type=int, default=1)

    ## inference api
    group.add_argument('--max-workers', type=int, default=16)
    group.add_argument('--eval-dataset-list', nargs='*', type=str)
    group.add_argument('--stop-token-ids', nargs='*', type=int)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--bf16', default=False, action='store_true')
    return parser
    

def get_starter_code(header_str):
    if "def " in header_str:
        starter_code = header_str.split("def")[1].split("(")[0].strip()
    else:
        starter_code = header_str

    return starter_code

def preprocess_aime(data_file, model_type):
    
    prompt_list = []
    qid_list = []
    with open(data_file, "r") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            final_question = item['problem'].strip()
            if model_type == "qwen":
                final_prompt = """<|im_start|>system\nYou are a helpful and harmless assistant. You should think step-by-step.<|im_end|>\n<|im_start|>user\n{question}\n\nPlease place your final answer inside \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n<think>\n""".format(question=final_question)
            else:
                final_prompt = """<｜begin▁of▁sentence｜><｜User｜>{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<｜Assistant｜><think>\n""".format(question=final_question)
            prompt_list.append(final_prompt)
            qid_list.append(i)
    return prompt_list, qid_list
    
def load_vllm_model(args):
    torch_dtype = torch.float16 if args.fp16 else torch.bfloat16
    tensor_parallel_size = args.tensor_parallel_size

    print("load tokenizer from %s" % args.tokenizer_model)
    print("load model from %s" % args.load)
    print("torch_dtype:", torch_dtype)
    print("tensor_parallel_size:", tensor_parallel_size)

    model_vllm = LLM(args.load, tokenizer=args.tokenizer_model, dtype=torch_dtype, tensor_parallel_size=tensor_parallel_size)

    return model_vllm


def get_prompt_list(args):
    ## get input data
    input_datapath = args.datapath
    if "aime" in args.datapath:
        prompt_list, qid_list = preprocess_aime(input_datapath, args.model_type)
    else:
        raise ValueError("Invalid dataset name")
        
    print("number of total prompt_list:", len(prompt_list))
    if args.start_idx != -1 and args.end_idx != -1:
        print("getting data from %d to %d" % (args.start_idx, args.end_idx))
        prompt_list = prompt_list[args.start_idx:args.end_idx]
        if qid_list:
            qid_list = qid_list[args.start_idx:args.end_idx]

    print("number of test samples in the dataset:", len(prompt_list))
    return prompt_list, qid_list


def main():
    args = get_args(add_evaluation=True)
    # if args.device_id:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    ## load model
    model_vllm = load_vllm_model(args)

    ## load test data
    prompt_list, qid_list = get_prompt_list(args)

    ## run inference
    print("args.max_output_len:", args.max_output_len)
   
    if args.topp < 1:
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.topp, max_tokens=args.max_output_len, seed=args.seed)
        print("args.seed:", args.seed)
        print("args.topp:", args.topp)
        print("args.temperature:", args.temperature)

    else:
        sampling_params = SamplingParams(temperature=args.temperature, top_k=args.topk, max_tokens=args.max_output_len, seed=args.seed)

    output_list = []
    for i in tqdm(range(0, len(prompt_list), args.batch_size)):
        batch_prompts = prompt_list[i:i+args.batch_size]
        if qid_list:
            batch_qids = qid_list[i:i+args.batch_size]

        outputs = model_vllm.generate(batch_prompts, sampling_params)
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            if "<|im_end|>" in generated_text:
                idx = generated_text.index("<|im_end|>")
                generated_text = generated_text[:idx]
            if "<|end_of_text|>" in generated_text:
                idx = generated_text.index("<|end_of_text|>")
                generated_text = generated_text[:idx]
            if "<|eot_id|>" in generated_text:
                idx = generated_text.index("<|eot_id|>")
                generated_text = generated_text[:idx]
            
            if qid_list:
                qid = batch_qids[j]
                output_dict = {"task_id": qid, "output": generated_text}
                output_list.append(output_dict)
            else:
                output_dict = {"output": generated_text}
                output_list.append(output_dict)

    ## write to output_datapath

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    output_name = "%dto%d_seed%d.jsonl" % (args.start_idx, args.end_idx, args.seed) \
                            if args.start_idx != -1 and args.end_idx != -1 else "seed%d.jsonl" % args.seed
    
    output_datapath = os.path.join(args.output_folder, output_name)

    print("writing to %s" % output_datapath)
    with open(output_datapath, "w", encoding='utf-8') as f:
        for output in output_list:
            if type(output) == dict:
                f.write(json.dumps(output) + "\n")
            else:
                f.write(output + "\n")

if __name__ == "__main__":
    main()

