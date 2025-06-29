import os
import socket
import torch.distributed as dist
import torch
from pathlib import Path
import random

#  if not run using torchrun
# WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
# WORLD_RANK = int(os.environ['SLURM_PROCID'])
# LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
MASTER_ADDR = os.environ['MASTER_ADDR']
MASTER_PORT = os.environ['MASTER_PORT']

# def get_env_var(key, fallback_key=None):
#     return int(os.environ.get(key) or os.environ.get(fallback_key or "", -1))

# LOCAL_RANK = get_env_var("LOCAL_RANK", "SLURM_LOCALID")
# WORLD_RANK = get_env_var("RANK", "SLURM_PROCID")
# WORLD_SIZE = get_env_var("WORLD_SIZE", "SLURM_NTASKS")


import subprocess

def print_port_status(w_rank=0):
    print(f"\n🔍 [Rank {w_rank}] Running 'ss -tuln' to list listening TCP/UDP sockets...\n", flush=True)
    try:
        result = subprocess.check_output(["ss", "-tuln"], text=True)
        print(f"[Rank {w_rank}]: \n{result}", flush=True)
    except subprocess.CalledProcessError as e:
        print("[Rank {WORLD_RANK}]: ❌ Failed to run 'ss -tuln'", flush=True)
        print(e.output)
    except FileNotFoundError:
        print("[Rank {WORLD_RANK}]: ❌ The 'ss' command is not available on this system.", flush=True)

def main():
    random_id = random.randint(0, 10000)
    print(f"Starting distributed process... random ID: {random_id}")

    # local_rank = int(os.environ["LOCAL_RANK"])
    # global_rank = int(os.environ["RANK"])
    # group_rank = int(os.environ["GROUP_RANK"])

    # torchrun approach
    # LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_RANK = int(os.environ['RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])


    print(f"[Rank {WORLD_RANK}] Initializing process group with MASTER_ADDR: {MASTER_ADDR}, MASTER_PORT: {MASTER_PORT}, WORLD_SIZE: {WORLD_SIZE}, LOCAL_RANK: {LOCAL_RANK}")

    # dist.init_process_group(backend='gloo',init_method='env://', world_size=WORLD_SIZE, rank=WORLD_RANK)
    dist.init_process_group(backend='nccl',init_method='env://', world_size=WORLD_SIZE, rank=WORLD_RANK)

    print_port_status(WORLD_RANK)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()
    
    torch.cuda.set_device(LOCAL_RANK)
    
    group_rank = WORLD_RANK//2  # Assuming group_rank is the same as LOCAL_RANK for simplicity
    print(f"[Rank {rank}/{world_size}] Hello from {hostname}. Env vars: LOCAL_RANK: {LOCAL_RANK}, WORLD_RANK: {WORLD_RANK}, Group Rank (WORLD_RANK/2): {group_rank}")

    # print(f"[Rank {rank}/{world_size}] LOCAL_RANK: {LOCAL_RANK}, WORLD_RANK: {WORLD_RANK}, WORLD_SIZE: {WORLD_SIZE}")

    # Create output directory
    output_dir = Path("./test_file_check")
    output_dir.mkdir(parents=True, exist_ok=True)


    # Write a file per process
    file_path = output_dir / f"rank_{rank}_{hostname}.txt"
    with open(file_path, "w") as f:
        f.write(f"Rank: {rank}, Hostname: {hostname}\n")

    print(f"[Rank {rank}/{world_size}] File created at {file_path}", flush=True)

    dist.barrier()
    if rank == 0:
        print("✅ All ranks reached the barrier. DDP is working!", flush=True)

    print(f"[Rank {rank}/{world_size}] Exiting process... random ID: {random_id}", flush=True)
    dist.barrier()
    try:
        dist.destroy_process_group()
        print(f"[Rank {rank}] ✅ Process group destroyed.", flush=True)
    except Exception as e:
        print(f"[Rank {rank}] ⚠️ Error during destroy_process_group: {e}", flush=True)

    # dist.destroy_process_group()
    print(f"[Rank {rank}/{world_size}] Process group destroyed.")

if __name__ == "__main__":
    main()
