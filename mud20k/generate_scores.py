import json
import time
from typing import Optional

from datasets import load_dataset
import re
import threading
import queue
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import statistics
from multiprocessing import Process
from forget_eval import *
import os


import argparse
parser = argparse.ArgumentParser(description="Example of --test flag.")
parser.add_argument("--model_name_or_path", type=str, required=True, help="Model save path")
parser.add_argument("--save_path", type=str, required=True, help="Save path")
args = parser.parse_args()
model_path = args.model_name_or_path
save_path = args.save_path



def evaluate(model_path):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    #tokenizer.save_pretrained(MODEL)
    #MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    llm = LLM(model=model_path)



    gsm8k = load_dataset("openai/gsm8k", "main")["test"]

    N = len(gsm8k)
    print(f"Building pairs for N={N} items")
    run_means = []
    for test_run in range(1):
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048, seed = test_run)
            

        model_inputs = []
        for i in range(N):
            question = gsm8k[i]["question"]
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs.append(text)
            
            
        answers = llm.generate(model_inputs, sampling_params)
        items = []
        for i in range(len(answers)):
            retain_item = dict(gsm8k[i])
            ans = answers[i].outputs[0].text
            correct_answer = retain_item["answer"].split("####")[1].strip()
            matches = re.findall(r"\\*boxed\{([^}]*)\}", ans)

            if len(matches) == 1:
                answer = matches[0].strip()
            else:
                answer = None
            #print(answer, correct_answer, answer == correct_answer)
            if answer == correct_answer:
                items.append(1)
            else:
                items.append(0)
        print(answers[0])

        mean = sum(items) / len(items)
        print(mean)
        run_means.append(mean)
        
        with open(save_path.split("/")[-1], "a") as f:
            f.write(f"\n\n{model_path} util: {mean / 0.7543593631539045}")

def forget_eval(model_path):
    eps = 1e-12
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    kwargs = priv_kwargs
    device = "cuda"
    
    
    tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello2"}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
    
    

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16, device_map = device)
    print(model_path)
    model = model.to(torch.float32)
    model.eval()
    #retain_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", torch_dtype = torch.bfloat16, device_map = device)

    
    
    retain_score = extraction_strength(model, **{"data": retain_data, "collators": collator, "batch_size": 32})["agg_value"]
    print("Retain Score", retain_score)
    #data = QADataset({'name': 'forget10_perturbed', 'split': 'train', 'path': 'locuslab/TOFU'}, {'apply_chat_template': True, 'system_prompt': 'Please reason step by step, and put your final answer within \\boxed{}.', 'system_prompt_with_special_tokens': '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>', 'user_start_tag': '<|im_start|>user\n', 'user_end_tag': '<|im_end|>\n', 'asst_start_tag': '<|im_start|>assistant\n', 'asst_end_tag': '<|im_end|>\n', 'date_string': '10 Apr 2025'}, tokenizer)

    
    s_LOSS = 1 - 2 * abs(mia_loss(model, **kwargs)["agg_value"] - 0.5) + eps
    print(s_LOSS)
    s_ZLib = 1 - 2 * abs(mia_zlib(model, **kwargs)["agg_value"] - 0.5) + eps
    print(s_ZLib)
    s_Min_k = 1 - 2 * abs(mia_min_k(model, **kwargs)["agg_value"] - 0.5) + eps
    print(s_Min_k)
    s_Min_k_plus_plus = 1 - 2 * abs(mia_min_k_plus_plus(model, **kwargs)["agg_value"] - 0.5) + eps
    print(s_Min_k_plus_plus)
    priv_score = 4 / (1/s_LOSS + 1/s_ZLib + 1/s_Min_k + 1/s_Min_k_plus_plus)
    print(priv_score)
     
     
     
     
    
    
    
    #TR = 1 - truth_ratio(model, **{"data": data, "collators": collator, "batch_size": 2, 'aggregator': 'closer_to_1_better'})["agg_value"]
    ES = 1 - extraction_strength(model, **{"data": data, "collators": collator, "batch_size": 32})["agg_value"] + eps
    print(ES)
    EM = 1 - exact_memorization(model, **{"data": data, "collators": collator, "batch_size": 32})["agg_value"] + eps
    
    Prob = 1 - probability(model, **{"data": data, "collators": collator, "batch_size": 32})["agg_value"] + eps
    memorization = 3 / (1/EM + 1/ES + 1/Prob)
    print(memorization)
    
    with open(save_path.split("/")[-1], "a") as f:
        f.write(f"\n{model_path} mem: {memorization / 0.72497205567724}")# retrain scores
    with open(save_path.split("/")[-1], "a") as f:
        f.write(f"\n{model_path} priv: {priv_score / 0.8117690172876743}")



evaluate(model_path)
forget_eval(model_path)