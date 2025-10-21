import os, zipfile, torch, torch.distributed as dist

def unzip_once(zip_path, extract_dir):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1 and dist.is_initialized()

    if (not is_dist and local_rank == 0) or (is_dist and dist.get_rank() == 0):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    if is_dist:
        dist.barrier()
