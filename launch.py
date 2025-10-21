from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import sagemaker, boto3

region = "eu-central-1"
session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=session)

# --- FIX: Corrected the IAM Role ARN ---
role = "arn:aws:iam::564083281396:role/AmazonSageMakerExecutionRole"

est = PyTorch(
    entry_point="train.py",
    source_dir="./src",
    role=role,
    # --- RECOMMENDATION: Use major/minor version ---
    framework_version="2.3.0",
    py_version="py311",
    instance_type="ml.g5.12xlarge",
    instance_count=4,
    distribution={"torch_distributed": {"enabled": True}},
    use_spot_instances=True,
    max_run=24 * 3600,
    max_wait=30 * 3600,
    checkpoint_s3_uri="s3://joachim-deep-learning-blockchain-gnn/checkpoints/",
    volume_size=300,
    environment={
        "PYTHONUNBUFFERED": "1",
        "SAGEMAKER_TRAINING_LOG_LEVEL": "20",
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "NCCL_IB_DISABLE": "1",
    },
    hyperparameters={
        "epochs": 10,
        "batch_size": 128,
        "num_workers": 8,
        "threads_per_proc": 16
    },
    disable_profiler=True,
    debugger_hook_config=False,
    base_job_name="gnn-ddp-spot-salient",
    sagemaker_session=sagemaker_session,
    output_path=f"s3://{sagemaker_session.default_bucket()}/gnn-output/"
)

# This configuration for TrainingInput is correct and will work
train_input = TrainingInput(
    s3_data="s3://joachim-deep-learning-blockchain-gnn/archive.zip",
    s3_data_type="S3Prefix",
    input_mode="File"
)

est.fit({"train": train_input})