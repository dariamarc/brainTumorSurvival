#!/usr/bin/env python3
"""
Launch UNet POC Step 3 — UNet3DProto + Prototype Learning Losses (single fold, base_channels=16)

POC experiment: no cross-validation, 80/20 split, Dice + clst/sep/div losses.
Results documented in EXPERIMENT_SUMMARY.md (Mean Dice 0.7195, proto ratios NCR=4.74 ED=3.54 ET=5.99).

Usage:
    python launch_unet_poc_step3.py --bucket YOUR-BUCKET \
        --step2-model-s3 s3://bucket/step2-outputs/.../output/model.tar.gz \
        --step2-baseline-dice 0.711
    python launch_unet_poc_step3.py --monitor unet-proto-losses-20260430-140000
"""

import argparse
import os
import boto3
import tarfile
import tempfile
from datetime import datetime
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bucket',        type=str)
    p.add_argument('--role',          type=str)
    p.add_argument('--data-prefix',   type=str, default='preprocessed_data_cropped')
    p.add_argument('--output-prefix', type=str, default='step3-outputs')
    p.add_argument('--instance',      type=str, default='ml.g4dn.xlarge')
    p.add_argument('--volume-size',   type=int, default=100)
    p.add_argument('--max-runtime',   type=int, default=86400)
    p.add_argument('--job-name',      type=str, default=None)
    p.add_argument('--monitor',       type=str, default=None)
    # Hyperparameters
    p.add_argument('--epochs',            type=int,   default=30)
    p.add_argument('--patience',          type=int,   default=10)
    p.add_argument('--lr',                type=float, default=1e-4)
    p.add_argument('--base-channels',     type=int,   default=16)
    p.add_argument('--protos-per-class',  type=int,   default=3)
    p.add_argument('--clst-weight',       type=float, default=0.2)
    p.add_argument('--sep-weight',        type=float, default=0.1)
    p.add_argument('--div-weight',        type=float, default=0.1)
    p.add_argument('--num-volumes',       type=int,   default=369)
    p.add_argument('--step2-baseline-dice', type=float, default=0.711)
    p.add_argument('--step2-model-s3',    type=str,   default=None,
                   help='S3 URI of Step 2 model.tar.gz')
    return p.parse_args()


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
                               'unet_poc_step3')
    if not os.path.exists(package_dir):
        raise FileNotFoundError(f"unet_poc_step3 not found: {package_dir}")
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
        raise ValueError("No SageMaker role found. Pass --role.")

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name  = args.job_name or f'unet-proto-losses-{timestamp}'
    print(f"Job name: {job_name}")

    code_uri = upload_code(args.bucket, job_name)
    image_uri = (f"763104351884.dkr.ecr.{region}.amazonaws.com/"
                 f"tensorflow-training:2.13.0-gpu-py310-cu118-ubuntu20.04-sagemaker")

    hyperparameters = {
        'num-slices':    '128', 'height': '192', 'width': '160',
        'channels':      '4',   'num-classes': '4',
        'num-volumes':   str(args.num_volumes),
        'split-ratio':   '0.2',
        'base-channels': str(args.base_channels),
        'protos-per-class': str(args.protos_per_class),
        'epochs':        str(args.epochs),
        'patience':      str(args.patience),
        'lr':            str(args.lr),
        'clst-weight':   str(args.clst_weight),
        'sep-weight':    str(args.sep_weight),
        'div-weight':    str(args.div_weight),
        'step2-baseline-dice': str(args.step2_baseline_dice),
        **({'step2-model-s3': args.step2_model_s3} if args.step2_model_s3 else {}),
        'sagemaker_program':          'train.py',
        'sagemaker_submit_directory': code_uri,
    }

    print("\nHyperparameters:")
    for k, v in hyperparameters.items():
        if not k.startswith('sagemaker'):
            print(f"  {k}: {v}")

    boto3.client('sagemaker').create_training_job(
        TrainingJobName=job_name,
        RoleArn=role,
        AlgorithmSpecification={'TrainingImage': image_uri, 'TrainingInputMode': 'File'},
        HyperParameters=hyperparameters,
        InputDataConfig=[{
            'ChannelName': 'training',
            'DataSource': {'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': f's3://{args.bucket}/{args.data_prefix}',
                'S3DataDistributionType': 'FullyReplicated',
            }},
        }],
        OutputDataConfig={'S3OutputPath': f's3://{args.bucket}/{args.output_prefix}/{job_name}'},
        ResourceConfig={'InstanceType': args.instance, 'InstanceCount': 1,
                        'VolumeSizeInGB': args.volume_size},
        StoppingCondition={'MaxRuntimeInSeconds': args.max_runtime},
        EnableManagedSpotTraining=False,
    )

    console = (f"https://{region}.console.aws.amazon.com/sagemaker/home"
               f"?region={region}#/jobs/{job_name}")
    print(f"\nJob submitted: {job_name}")
    print(f"Console: {console}")
    print(f"\nWhat to watch (epoch log format):")
    print(f"  loss=X [dice=X clst=X sep=X div=X] | NCR=X ED=X ET=X | WT=X Mean=X")
    print(f"  Proto ratios — NCR should cross 1.0 within 5-7 epochs")
    print(f"  If NCR ratios stay < 1.0 after epoch 7: increase --clst-weight to 0.1")


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
