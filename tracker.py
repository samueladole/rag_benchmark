import wandb

def init():
    wandb.init(project="rag-benchmark")

def log(metrics):
    wandb.log(metrics)