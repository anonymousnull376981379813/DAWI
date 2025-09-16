# entrypoint.py
import argparse
import os
import subprocess
from pathlib import Path
import shutil
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, either 'TOFU' or 'MUD20k'. MUD200 currently not supported")
    parser.add_argument("--c", type=int, required=False, help="hyperparam for initial value of C")
    parser.add_argument("--mode", type=str, required=True, help="Train, eval, or figure1")
    parser.add_argument("--model_name_or_path", type=str, required=False, help="Name of the model or its path")
    parser.add_argument("--save_path", type=str, required = False, help = "Path to save model after training")
    
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "MUD200":
        raise ValueError("MUD 200 currently not supported")
    save_path = args.save_path
    if args.c is None:
        c = 5
    if args.mode == "eval":
        model_path = args.model_name_or_path
        if dataset == "TOFU":
            
            if model_path is None:
                raise ValueError("Model must have path")
            model = "Llama-3.2-1B-Instruct"
            cmd = [
                "python", "src/eval.py",
                "--config-name=eval.yaml",
                "experiment=eval/tofu/default",
                f"model={model}",
                f"model.model_args.pretrained_model_name_or_path={model_path}",
                f"retain_logs_path=saves/eval/tofu_{model}_retain90/TOFU_EVAL.json",
                f"task_name=EVAL/tempsave",
            ]
            subprocess.run(cmd, cwd="./open-unlearning", check=True)
            src = Path(f"open-unlearning/saves/eval/EVAL/tempsave/TOFU_SUMMARY.json")
            
            cmd = ["python", "score_evals.py", f"--path=../open-unlearning/saves/eval/EVAL"]
            subprocess.run(cmd, cwd="./tofu", check=True)
            dst = Path(f"evals/{model_path.split('/')[-1]}.json")
            shutil.move(str(src), str(dst))
        elif dataset == "MUD20k":
            cmd = ["python", "generate_scores.py", f"--model_name_or_path={model_path}", f"--save_path=./MUD_EVALS/{model_path}"]
            subprocess.run(cmd, cwd="./mud20k", check=True)
            
    elif args.mode == "train":
        if save_path is None:
            raise ValueError("Save path must be passed in for training")
        if dataset == "TOFU":
            path = Path("tofu/dataset.pt")
            if not path.exists():
                cmd = ["python", "generate_dataset_file.py"]
                print("Downloading dataset and generating dataset file")
                subprocess.run(cmd, cwd="./tofu", check=True)
            cmd = ["python", "DAWI.py", f"--c={c}", f"--save_path=../checkpoints/{save_path}"]
            subprocess.run(cmd, cwd="./tofu", check=True)
            
        elif dataset == "MUD20k":
            path = Path("tofu/dataset.pt")
            if not path.exists():
                cmd = ["python", "generate_dataset_file.py"]
                print("Downloading dataset and generating dataset file")
                subprocess.run(cmd, cwd="./mud20k", check=True)
                cmd = ["python", "DAWI.py", f"--c={c}", f"--save_path=../checkpoints/{save_path}"]
                subprocess.run(cmd, cwd="./tofu", check=True)
            
        
        else:
            raise ValueError("Dataset must be 'TOFU', 'MUD200', or 'MUD20k'")
    elif args.mode == "figure1":
        model_path = None
        models = [
            "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.25_ep10",
            "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff1_epoch10",
            "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr4e-05_alpha10_epoch5",
            "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta30_alpha1_epoch10",
            "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.05_alpha5_epoch10",
            "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.05_alpha2_epoch10",
            "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.05_alpha5_epoch10",
            #"open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha2_epoch5",
            "Anonymous0192830/DAWI_TOFU"]
        for model_name in models:
            model="Llama-3.2-1B-Instruct"
            cmd = [
            "python", "src/eval.py",
            "--config-name=eval.yaml",
            "experiment=eval/tofu/default",
            f"model={model}",
            f"model.model_args.pretrained_model_name_or_path={model_name}",
            f"retain_logs_path=saves/eval/tofu_{model}_retain90/TOFU_EVAL.json",
            f"task_name=FIG1/{model_name.split('/')[-1]}",
            ]
            subprocess.run(cmd, cwd="./open-unlearning", check=True)
        
            src = Path(f"open-unlearning/saves/eval/FIG1/{model_name.split('/')[-1]}/TOFU_SUMMARY.json")
            dst = Path(f"figure1/{model_name.split('/')[-1]}.json")
            shutil.copy(str(src), str(dst))
        cmd = ["python", "score_evals.py", f"--path=../open-unlearning/saves/eval/FIG1"]
        subprocess.run(cmd, cwd="./tofu", check=True)
            
            
            
            
    else:
        raise ValueError("Mode should either by 'train' or 'eval'") 
    

if __name__ == "__main__":
    main()
