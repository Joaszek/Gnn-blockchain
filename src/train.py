import argparse, os, sys, math, glob, json, time, subprocess
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from utils_data import unzip_once  # Assuming you have this helper file
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "/opt/ml/checkpoints"
MODEL_DIR = "/opt/ml/model"
DEFAULT_TRAIN_CHANNEL = "/opt/ml/input/data/train"
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", os.environ.get("SM_LOCAL_RANK", 0)))

# ---------------------------------------------------------------------------
# Custom Setup Execution
# ---------------------------------------------------------------------------
def run_setup():
    # Each host must run the script at least once (local .salientpp_built tag)
    # The setup script is idempotent, so it's safe to call on all ranks.
    print("[train] running setup_salient.sh...", flush=True)
    subprocess.run(["/opt/ml/code/setup_salient.sh"], check=True)

run_setup()

# --- FIX: The setup script installs the 'fast_sampler' module.
try:
    import fast_sampler
    print("[train] import fast_sampler OK")
except ImportError as e:
    print(f"[train] Failed to import fast_sampler: {e}")
    # Add repo to path as a fallback if needed
    if "/opt/ml/code/SALIENT_plusplus" not in sys.path:
        sys.path.append("/opt/ml/code/SALIENT_plusplus")
    import fast_sampler

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("train")

log = setup_logging()

# ---------------------------------------------------------------------------
# DDP (Distributed Data Parallel) Helpers
# ---------------------------------------------------------------------------
def init_distributed():
    if "SM_HOSTS" not in os.environ:
        log.info("Running in single-process mode (no DDP).")
        return

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    os.environ.setdefault("MASTER_ADDR", json.loads(os.environ["SM_HOSTS"])[0])
    os.environ.setdefault("MASTER_PORT", "23456")

    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    dist.barrier()
    
    if rank == 0:
        log.info(f"DDP Initialized: backend=nccl, world_size={world_size}, master={os.environ['MASTER_ADDR']}")

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_rank0():
    return not is_dist() or dist.get_rank() == 0

def get_rank_world():
    return (dist.get_rank(), dist.get_world_size()) if is_dist() else (0, 1)

# ---------------------------------------------------------------------------
# Evaluation and Checkpointing
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            y_prob.extend(probs[:, 1].cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    metrics = {"acc": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        pass
    return metrics

def latest_checkpoint():
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_epoch_*.pt")))
    return files[-1] if files else None

def save_checkpoint(epoch, model, optimizer, metrics, history, out_dir):
    if not is_rank0(): return
    os.makedirs(out_dir, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    ckpt_path = os.path.join(out_dir, f"ckpt_epoch_{epoch:04d}.pt")
    torch.save({
        "epoch": epoch, "model_state": state_dict, "optimizer_state": optimizer.state_dict(),
        "extra": {"metrics": metrics, "history": history},
    }, ckpt_path)
    log.info(f"[ckpt] Saved checkpoint: {ckpt_path}")

# ---------------------------------------------------------------------------
# Dataset and Model Definitions
# ---------------------------------------------------------------------------
class YourActualDataset(Dataset):
    """
    --- ACTION REQUIRED ---
    This is a placeholder. You MUST replace this with your actual Dataset 
    implementation that loads and processes data from the `data_dir`.
    """
    def __init__(self, data_dir, num_samples=50000):
        self.num_samples = num_samples
        self.data_dir = data_dir
        log.info(f"Dataset initialized. Reading from: {self.data_dir}")

    def __len__(self): 
        return self.num_samples

    def __getitem__(self, idx):
        # This dummy implementation returns random data.
        x = torch.randn(32)
        y = torch.randint(0, 2, (1,)).item()
        return x, y

def build_model():
    return torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    total_steps = len(loader)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if is_rank0() and (i > 0 and i % 50 == 0):
            print(f"[epoch {epoch}] step {i}/{total_steps} loss={loss.item():.4f}", flush=True)
    return running_loss / max(1, total_steps)

# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--threads_per_proc", type=int, default=1)
    parser.add_argument("--train_channel", type=str, default=DEFAULT_TRAIN_CHANNEL)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    init_distributed()
    rank, world = get_rank_world()

    if is_rank0():
        log.info(f"Starting job. Rank={rank}, World={world}, Args={args}")

    # --- FIX: Unzip data on rank 0, and make other ranks wait.
    data_destination = "/tmp/data"
    if is_rank0():
        zip_path = os.path.join(args.train_channel, "archive.zip")
        log.info(f"Rank 0 is unzipping {zip_path} to {data_destination}...")
        unzip_once(zip_path, data_destination)
    if is_dist():
        dist.barrier()
    log.info(f"Rank {rank} data is ready.")

    # --- FIX: Correct DDP data splitting and sampling.
    full_dataset = YourActualDataset(data_dir=data_destination)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    ds_train, ds_val = random_split(full_dataset, [train_size, val_size], 
                                    generator=torch.Generator().manual_seed(42))
    
    train_sampler = DistributedSampler(ds_train, num_replicas=world, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(ds_val, num_replicas=world, rank=rank, shuffle=False)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, num_workers=args.num_workers, sampler=val_sampler)
    
    if is_rank0():
        log.info(f"Data: Total={len(full_dataset)}, Train={len(ds_train)}, Val={len(ds_val)}")

    device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if is_dist():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])

    start_epoch = 1
    history = []
    if (ckpt_path := latest_checkpoint()):
        ckpt = torch.load(ckpt_path, map_location=device)
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        history = ckpt.get("extra", {}).get("history", [])
        if is_rank0(): log.info(f"Resumed from {ckpt_path}. Starting epoch {start_epoch}.")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch) # Important for shuffling
            avg_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
            
            if is_rank0():
                log.info(f"[Epoch {epoch}/{args.epochs}] Average training loss: {avg_loss:.4f}")
                val_metrics = evaluate(model, val_loader, device)
                epoch_metrics = {"train_loss": avg_loss, "val": val_metrics}
                log.info(f"[Epoch {epoch}/{args.epochs}] Val F1: {val_metrics['f1_macro']:.4f}, Val Acc: {val_metrics['acc']:.4f}")
                history.append({"epoch": epoch, **epoch_metrics})
                save_checkpoint(epoch, model, optimizer, epoch_metrics, history, CHECKPOINT_DIR)

    except Exception as e:
        if is_rank0(): log.exception(f"Training interrupted by error: {e}")
        raise

    if is_rank0():
        os.makedirs(MODEL_DIR, exist_ok=True)
        final_sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        out_path = os.path.join(MODEL_DIR, "model.pt")
        torch.save(final_sd, out_path)
        log.info(f"Saved final model to {out_path}")

    if is_dist():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()