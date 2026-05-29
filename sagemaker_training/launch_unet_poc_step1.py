#!/usr/bin/env python3
"""
Launch UNet POC Step 1 — Plain UNet3D Baseline (single fold, base_channels=16)

POC experiment: no cross-validation, 80/20 split, Dice+CE loss.
Results documented in EXPERIMENT_SUMMARY.md (Mean Dice 0.704).

Usage:
    python launch_unet_poc_step1.py --bucket YOUR-BUCKET --role YOUR-ROLE-ARN
    python launch_unet_poc_step1.py --monitor unet-baseline-20240115-123456
"""

import argparse
import os
import boto3
import tarfile
import tempfile
from datetime import datetime
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket',       type=str, help='S3 bucket name')
    parser.add_argument('--role',         type=str, help='SageMaker IAM role ARN')
    parser.add_argument('--data-prefix',  type=str, default='preprocessed_data_cropped')
    parser.add_argument('--output-prefix',type=str, default='step1-outputs')
    parser.add_argument('--instance',     type=str, default='ml.g4dn.xlarge')
    parser.add_argument('--volume-size',  type=int, default=100)
    parser.add_argument('--max-runtime',  type=int, default=86400,  # 24 hours
                        help='Max runtime in seconds (default 24h)')
    parser.add_argument('--job-name',     type=str, default=None)
    parser.add_argument('--monitor',      type=str, default=None,
                        help='Monitor an existing job by name instead of launching')
    # Training hyperparameters
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--patience',     type=int,   default=10)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--base-channels',type=int,   default=16,
                        help='16 for ml.g4dn.xlarge (T4 16GB), 32 for ml.g5.xlarge (A10G 24GB)')
    parser.add_argument('--loss-alpha',   type=float, default=0.5)
    parser.add_argument('--num-volumes',  type=int,   default=369)
    return parser.parse_args()


def get_sagemaker_role():
    iam = boto3.client('iam')
    try:
        for role in iam.list_roles(MaxItems=100)['Roles']:
            if 'sagemaker' in role['RoleName'].lower():
                return role['Arn']
    except Exception as e:
        print(f"Could not auto-detect role: {e}")
    return None


def upload_code(bucket, job_name):
    package_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet_poc_step1')
    if not os.path.exists(package_dir):
        raise FileNotFoundError(f"unet_poc_step1 directory not found: {package_dir}")

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tarball_path = tmp.name

    print("Packaging code...")
    with tarfile.open(tarball_path, 'w:gz') as tar:
        for item in os.listdir(package_dir):
            tar.add(os.path.join(package_dir, item), arcname=item)

    s3_key = f'training-code/{job_name}/sourcedir.tar.gz'
    boto3.client('s3').upload_file(tarball_path, bucket, s3_key)
    os.unlink(tarball_path)
    print(f"Code uploaded: s3://{bucket}/{s3_key}")
    return f's3://{bucket}/{s3_key}'


def monitor_job(job_name):
    sm = boto3.client('sagemaker')
    print(f"\nMonitoring: {job_name}")
    last_status = None
    while True:
        resp   = sm.describe_training_job(TrainingJobName=job_name)
        status = resp['TrainingJobStatus']
        if status != last_status:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
            last_status = status
            if status in ('Completed', 'Failed', 'Stopped'):
                if status == 'Failed':
                    print(f"Failure reason: {resp.get('FailureReason', 'unknown')}")
                if 'ModelArtifacts' in resp:
                    print(f"Artifacts: {resp['ModelArtifacts']['S3ModelArtifacts']}")
                break
        time.sleep(30)


def launch(args):
    region  = boto3.Session().region_name or 'us-east-1'
    role    = args.role or get_sagemaker_role()
    if not role:
        raise ValueError("No SageMaker role found. Pass --role arn:aws:iam::ACCOUNT:role/NAME")

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name  = args.job_name or f'unet-baseline-{timestamp}'
    print(f"Job name: {job_name}")
    print(f"Role:     {role}")

    code_uri  = upload_code(args.bucket, job_name)
    s3_data   = f's3://{args.bucket}/{args.data_prefix}'
    s3_out    = f's3://{args.bucket}/{args.output_prefix}/{job_name}'

    # TF 2.13 GPU DLC — same image as existing jobs
    image_uri = (f"763104351884.dkr.ecr.{region}.amazonaws.com/"
                 f"tensorflow-training:2.13.0-gpu-py310-cu118-ubuntu20.04-sagemaker")

    hyperparameters = {
        # Volume dims — correct values confirmed by Step 0 validation
        'num-slices':    '128',
        'height':        '192',   # actual H (first spatial dim per slice)
        'width':         '160',   # actual W (second spatial dim per slice)
        'channels':      '4',
        'num-classes':   '4',
        # Dataset
        'num-volumes':   str(args.num_volumes),
        'split-ratio':   '0.2',
        # Model
        'base-channels': str(args.base_channels),
        # Training
        'epochs':        str(args.epochs),
        'patience':      str(args.patience),
        'lr':            str(args.lr),
        'loss-alpha':    str(args.loss_alpha),
        # SageMaker entry point
        'sagemaker_program':          'train.py',
        'sagemaker_submit_directory': code_uri,
    }

    print("\nHyperparameters:")
    for k, v in hyperparameters.items():
        if not k.startswith('sagemaker'):
            print(f"  {k}: {v}")

    sm = boto3.client('sagemaker')
    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=role,
        AlgorithmSpecification={
            'TrainingImage':    image_uri,
            'TrainingInputMode': 'File',
        },
        HyperParameters=hyperparameters,
        InputDataConfig=[{
            'ChannelName': 'training',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType':                'S3Prefix',
                    'S3Uri':                     s3_data,
                    'S3DataDistributionType':    'FullyReplicated',
                }
            },
        }],
        OutputDataConfig={'S3OutputPath': s3_out},
        ResourceConfig={
            'InstanceType':   args.instance,
            'InstanceCount':  1,
            'VolumeSizeInGB': args.volume_size,
        },
        StoppingCondition={'MaxRuntimeInSeconds': args.max_runtime},
        EnableManagedSpotTraining=False,
    )

    console = (f"https://{region}.console.aws.amazon.com/sagemaker/home"
               f"?region={region}#/jobs/{job_name}")
    print(f"\nJob submitted: {job_name}")
    print(f"\nMonitor:")
    print(f"  Console:  {console}")
    print(f"  Script:   python launch_unet_poc_step1.py --monitor {job_name}")
    print(f"  CLI:      aws sagemaker describe-training-job --training-job-name {job_name}")
    print(f"\nOutputs (after completion):")
    print(f"  {s3_out}/output/model.tar.gz")
    print(f"\nWhat to check in CloudWatch logs:")
    print(f"  - Loss decreasing in first 5 epochs")
    print(f"  - WT Dice > 0.65 by epoch ~30")
    print(f"  - Mean Dice > 0.40 by end of training")


def main():
    args = parse_args()
    if args.monitor:
        monitor_job(args.monitor)
    else:
        if not args.bucket:
            raise ValueError("--bucket is required. Usage: python launch_unet_poc_step1.py --bucket YOUR-BUCKET")
        launch(args)


if __name__ == '__main__':
    main()
