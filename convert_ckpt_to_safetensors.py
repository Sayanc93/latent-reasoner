

import argparse
import torch
import time
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

def load_sharded_model_single_gpu(model, model_path):
    print("Loading sharded model...")
    state_dict = {"model": model.state_dict()}
    
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader(model_path),
        no_dist=True,
    )
    
    result = model.load_state_dict(state_dict["model"])
    
    print(f"Sharded state checkpoint loaded from {model_path}")
    print(result)
    return model

def convert_checkpoint(args):
    start_time = time.time()

    # Convert torch_dtype string to actual torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map.get(args.torch_dtype.lower(), torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    print("Loading pretrained model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch_dtype,
        # device_map='cpu',
        trust_remote_code=True
    )

    model = load_sharded_model_single_gpu(model, args.fsdp_model_path)
    
    print(f"Model loading time: {time.time() - start_time:.2f} seconds")

    print("saving model to %s" % args.output_path)
    model.save_pretrained(args.output_path, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.output_path)

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to HuggingFace checkpoint")
    parser.add_argument("--hf_model", type=str, default="Qwen/Qwen2.5-Math-72B-Instruct", help="Path or name of the HuggingFace model")
    parser.add_argument("--fsdp_model_path", type=str, required=True, help="Path to the FSDP checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to save the converted checkpoint")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="Torch data type for the model")
    parser.add_argument("--max_shard_size", type=str, default="10GB", help="Maximum shard size for saving the model")

    args = parser.parse_args()
    convert_checkpoint(args)

if __name__ == "__main__":
    main()



# submit_job --gpu 8  -i --email_mode never  --mounts "/lustre/" --partition batch,batch_short,interactive  --image /lustre/fsw/portfolios/llmservice/users/zihanl/inform/sqshs/vllmlatest073.sqsh  --duration 2

## command
# python3.8 convert_ckpt_to_safetensors.py --hf_model /lustre/fsw/portfolios/llmservice/users/chankyul/R1/base_checkpoint/qwen2_5_math_7b_base/ --fsdp_model_path /lustre/fs1/portfolios/llmservice/users/chankyul/R1/Post-Training-eval/sft/ckpts/sftmathr1fullv2_3_qwenbase_7b_gbsz64_seqlen16k_lr2e-5/checkpoint-496/pytorch_model_fsdp_0 --torch_dtype bfloat16 --max_shard_size 10GB --output_path /lustre/fs1/portfolios/llmservice/users/chankyul/R1/Post-Training-eval/sft/ckpts/sftmathr1fullv2_3_qwenbase_7b_gbsz64_seqlen16k_lr2e-5/checkpoint-496-hf/

# python3.8 convert_ckpt_to_safetensors.py --hf_model /lustre/fsw/portfolios/llmservice/users/chankyul/R1/base_checkpoint/qwen2_5_math_7b_base/ --fsdp_model_path /lustre/fs1/portfolios/llmservice/users/chankyul/R1/Post-Training-eval/sft/ckpts/sftmathr1fullv2_3_gr_qwenbase_7b_gbsz64_seqlen16k_lr2e-5/checkpoint-496/pytorch_model_fsdp_0 --torch_dtype bfloat16 --max_shard_size 10GB --output_path /lustre/fs1/portfolios/llmservice/users/chankyul/R1/Post-Training-eval/sft/ckpts/sftmathr1fullv2_3_gr_qwenbase_7b_gbsz64_seqlen16k_lr2e-5/checkpoint-496-hf/

# python convert_ckpt_to_safetensors.py --hf_model /lustre/fsw/portfolios/llmservice/users/chankyul/R1/base_checkpoint/qwen2_5_math_7b_base/ --fsdp_model_path /lustre/fs1/portfolios/llmservice/users/chankyul/R1/Post-Training-eval/sft/ckpts/finetuning_sft_sftmathr1fullv2_3_gr_qwenbase2_5_math_7b_2e-5_64_16k/checkpoint-3968/pytorch_model_fsdp_0 --torch_dtype bfloat16 --max_shard_size 10GB --output_path /lustre/fs1/portfolios/llmservice/users/chankyul/R1/Post-Training-eval/sft/ckpts/finetuning_sft_sftmathr1fullv2_3_gr_qwenbase2_5_math_7b_2e-5_64_16k/checkpoint-3968-hf/

# python convert_ckpt_to_safetensors.py --hf_model /lustre/fsw/portfolios/llmservice/users/chankyul/R1/base_checkpoint/qwen2_5_math_7b_base/ --fsdp_model_path /lustre/fs1/portfolios/llmservice/users/chankyul/R1/Post-Training-eval/sft/ckpts/finetuning_sft_sftmathr1fullv2_3_qwenbase2_5_math_7b_2e-5_64_16k/checkpoint-3968/pytorch_model_fsdp_0 --torch_dtype bfloat16 --max_shard_size 10GB --output_path /lustre/fs1/portfolios/llmservice/users/chankyul/R1/Post-Training-eval/sft/ckpts/finetuning_sft_sftmathr1fullv2_3_qwenbase2_5_math_7b_2e-5_64_16k/checkpoint-3968-hf/

# Total execution time: 845.12 seconds
