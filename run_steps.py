"""Run a selected upstream stage, with a single dataset tag, without touching Student training."""
import argparse, subprocess, sys
from pathlib import Path

SCRIPTS = ['build_prior_graph.py', 'build_events.py', 'split_events.py', 'build_prompts.py', 'label_teacher.py', 'build_sft_data.py', 'build_gold_labels.py', 'build_topology.py']

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", default="brn_all", choices=("cev_all","inu_all","emd_all","brn_all","mro_all"))
    p.add_argument("--step",type=int,required=True,choices=range(8))
    p.add_argument("--split",default="val",choices=("val","test"),help="Only for step 6; gold labels")
    args=p.parse_args()
    cmd=[sys.executable,str(Path(__file__).resolve().parent/"upstream"/SCRIPTS[args.step]),"--dataset",args.dataset]
    if args.step==6:
        cmd.extend(["--split",args.split])
        if args.split=="test": cmd.append("--annotate")
    subprocess.run(cmd,check=True)

if __name__=="__main__": main()
