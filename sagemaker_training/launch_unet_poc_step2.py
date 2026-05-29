#!/usr/bin/env python3
"""
Launch UNet POC Step 2 — UNet3D + Prototype Bottleneck (single fold, base_channels=16)

POC experiment: no cross-validation, 80/20 split, pure Dice loss.
Results documented in EXPERIMENT_SUMMARY.md (Mean Dice 0.722, proto ratios ~1.0).

Usage:
    python launch_unet_poc_step2.py --bucket YOUR-BUCKET --role YOUR-ROLE-ARN
    python launch_unet_poc_step2.py --bucket YOUR-BUCKET --step1-baseline-dice 0.672
    python launch_unet_poc_step2.py --monitor unet-proto-20260429-130000
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
    parser.add_argument('--output-prefix',type=str, default='step2-outputs')
    parser.add_argument('--instance',     type=str, default='ml.g4dn.xlarge')
    parser.add_argument('--volume-size',  type=int, default=100)
    parser.add_argument('--max-runtime',  type=int, default=86400)
    parser.add_argument('--job-name',     type=str, default=None)
    parser.add_argument('--monitor',      type=str, default=None)
    # Training hyperparameters
    parser.add_argument('--epochs',            type=int,   default=50)
    parser.add_argument('--patience',          type=int,   default=10)
    parser.add_argument('--lr',                type=float, default=1e-4)
    parser.add_argument('--base-channels',     type=int,   default=16)
    parser.add_argument('--protos-per-class',  type=int,   default=3)
    parser.add_argument('--num-volumes',       type=int,   default=369)
    parser.add_argument('--step1-baseline-dice', type=float, default=0.659,
                        help='Final mean Dice from Step 1 for pass/fail comparison.')
    parser.add_argument('--step1-model-s3', type=str, default=None,
                        help='S3 URI to Step 1 model.tar.gz, e.g. '
                             's3://bucket/step1-outputs/unet-baseline-XXXX/output/model.tar.gz. '
                             'Added as a second input channel so train.py can load the weights.')
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
    package_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'unet_poc_step2')
    if not os.path.exists(package_dir):
        raise FileNotFoundError(f"unet_poc_step2 directory not found: {package_dir}")

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
    region = boto3.Session().region_name or 'us-east-1'
    role   = args.role or get_sagemaker_role()
    if not role:
        raise ValueError("No SageMaker role found. Pass --role arn:aws:iam::ACCOUNT:role/NAME")

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name  = args.job_name or f'unet-proto-{timestamp}'
    print(f"Job name: {job_name}")

    code_uri = upload_code(args.bucket, job_name)
    s3_data  = f's3://{args.bucket}/{args.data_prefix}'
    s3_out   = f's3://{args.bucket}/{args.output_prefix}/{job_name}'

    image_uri = (f"763104351884.dkr.ecr.{region}.amazonaws.com/"
                 f"tensorflow-training:2.13.0-gpu-py310-cu118-ubuntu20.04-sagemaker")

    hyperparameters = {
        'num-slices':    '128',
        'height':        '192',
        'width':         '160',
        'channels':      '4',
        'num-classes':   '4',
        'num-volumes':   str(args.num_volumes),
        'split-ratio':   '0.2',
        'base-channels': str(args.base_channels),
        'protos-per-class': str(args.protos_per_class),
        'epochs':        str(args.epochs),
        'patience':      str(args.patience),
        'lr':            str(args.lr),
        'step1-baseline-dice': str(args.step1_baseline_dice),
        **({'step1-model-s3': args.step1_model_s3} if args.step1_model_s3 else {}),
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
            'TrainingImage':     image_uri,
            'TrainingInputMode': 'File',
        },
        HyperParameters=hyperparameters,
        InputDataConfig=[{
            'ChannelName': 'training',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType':             'S3Prefix',
                    'S3Uri':                  s3_data,
                    'S3DataDistributionType': 'FullyReplicated',
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
    print(f"  Console: {console}")
    print(f"  Script:  python launch_unet_poc_step2.py --monitor {job_name}")
    if args.step1_model_s3:
        print(f"\nStep 1 weights: {args.step1_model_s3}")
        print(f"  → Encoder/decoder pre-loaded from Step 1; only prototype layers train fresh.")
    else:
        print(f"\nNo Step 1 weights provided — training from scratch.")
    print(f"\nWhat to watch:")
    print(f"  - Epoch 1 Dice should be close to Step 1 baseline ({args.step1_baseline_dice:.3f}) if weights loaded")
    print(f"  - Proto ratios > 1.0 = prototypes firing within correct class")
    print(f"  - Proto ratios ≈ 1.0 for all = prototypes have collapsed (no diversity)")
    print(f"\nOutputs: {s3_out}/output/model.tar.gz")


def main():
    args = parse_args()
    if args.monitor:
        monitor_job(args.monitor)
    else:
        if not args.bucket:
            raise ValueError("--bucket is required.")
        launch(args)


if __name__ == '__main__':
    main()
