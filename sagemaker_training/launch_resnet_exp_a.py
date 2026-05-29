#!/usr/bin/env python3
"""
Launch Experiment A — ResNet3D + ASPP3D Baseline (no prototypes)

Usage:
    python launch_resnet_exp_a.py --bucket your-brats2020-data
    python launch_resnet_exp_a.py --monitor resnet-exp-a-20260507-120000
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
    p.add_argument('--bucket',        type=str,   help='S3 bucket name')
    p.add_argument('--role',          type=str,   default=None)
    p.add_argument('--data-prefix',   type=str,   default='preprocessed_data_cropped')
    p.add_argument('--output-prefix', type=str,   default='resnet-exp-a-outputs')
    p.add_argument('--instance',      type=str,   default='ml.g4dn.xlarge')
    p.add_argument('--volume-size',   type=int,   default=100)
    p.add_argument('--max-runtime',   type=int,   default=259200)  # 72h for 5 folds
    p.add_argument('--job-name',      type=str,   default=None)
    p.add_argument('--monitor',       type=str,   default=None,
                   help='Monitor an existing job instead of launching a new one')
    # Hyperparameters
    p.add_argument('--n-folds',       type=int,   default=5)
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--patience',      type=int,   default=25)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--lr-min',        type=float, default=1e-6)
    p.add_argument('--base-channels', type=int,   default=64)
    p.add_argument('--aspp-channels', type=int,   default=256)
    p.add_argument('--num-volumes',   type=int,   default=369)
    return p.parse_args()


def get_sagemaker_role():
    iam = boto3.client('iam')
    try:
        for role in iam.list_roles(MaxItems=100)['Roles']:
            if 'sagemaker' in role['RoleName'].lower():
                return role['Arn']
    except Exception as e:
        print(f'Could not auto-detect role: {e}')
    return None


def upload_code(bucket, job_name):
    code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'resnet_exp_a')
    if not os.path.exists(code_dir):
        raise FileNotFoundError(f'resnet_exp_a/ not found at {code_dir}')

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tarball = tmp.name
    with tarfile.open(tarball, 'w:gz') as tar:
        for item in os.listdir(code_dir):
            tar.add(os.path.join(code_dir, item), arcname=item)

    s3_key = f'training-code/{job_name}/sourcedir.tar.gz'
    boto3.client('s3').upload_file(tarball, bucket, s3_key)
    os.unlink(tarball)
    print(f'Code uploaded: s3://{bucket}/{s3_key}')
    return f's3://{bucket}/{s3_key}'


def monitor_job(job_name):
    sm         = boto3.client('sagemaker')
    last_status = None
    print(f'\nMonitoring: {job_name}')
    while True:
        resp   = sm.describe_training_job(TrainingJobName=job_name)
        status = resp['TrainingJobStatus']
        if status != last_status:
            print(f'[{datetime.now().strftime("%H:%M:%S")}] {status}')
            last_status = status
            if status in ('Completed', 'Failed', 'Stopped'):
                if status == 'Failed':
                    print(f'Failure reason: {resp.get("FailureReason", "unknown")}')
                if 'ModelArtifacts' in resp:
                    s3_uri = resp['ModelArtifacts']['S3ModelArtifacts']
                    print(f'Artifacts : {s3_uri}')
                    # Note: SageMaker appends /{job_name}/output/ on top of
                    # S3OutputPath (which already contains job_name), so the
                    # actual weights path inside the tar is always:
                    #   {output_prefix}/{job_name}/{job_name}/output/model.tar.gz
                    print(f'Weights s3 key for Exp B1 --exp-a-model-s3 :')
                    print(f'  {s3_uri}')
                break
        time.sleep(30)


def launch(args):
    region   = boto3.Session().region_name or 'us-east-1'
    role     = args.role or get_sagemaker_role()
    if not role:
        raise ValueError('No SageMaker role found. Pass --role.')

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name  = args.job_name or f'resnet-exp-a-{timestamp}'
    print(f'Job name: {job_name}')

    code_uri  = upload_code(args.bucket, job_name)
    image_uri = (f'763104351884.dkr.ecr.{region}.amazonaws.com/'
                 f'tensorflow-training:2.13.0-gpu-py310-cu118-ubuntu20.04-sagemaker')

    hyperparameters = {
        'num-volumes':   str(args.num_volumes),
        'n-folds':       str(args.n_folds),
        'num-slices':    '128',
        'height':        '192',
        'width':         '160',
        'num-classes':   '4',
        'base-channels': str(args.base_channels),
        'aspp-channels': str(args.aspp_channels),
        'epochs':        str(args.epochs),
        'patience':      str(args.patience),
        'lr':            str(args.lr),
        'lr-min':        str(args.lr_min),
        'sagemaker_program':          'train.py',
        'sagemaker_submit_directory': code_uri,
    }

    print('\nHyperparameters:')
    for k, v in hyperparameters.items():
        if not k.startswith('sagemaker'):
            print(f'  {k}: {v}')

    boto3.client('sagemaker').create_training_job(
        TrainingJobName=job_name,
        RoleArn=role,
        AlgorithmSpecification={
            'TrainingImage': image_uri,
            'TrainingInputMode': 'File'
        },
        HyperParameters=hyperparameters,
        InputDataConfig=[{
            'ChannelName': 'training',
            'DataSource': {'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': f's3://{args.bucket}/{args.data_prefix}',
                'S3DataDistributionType': 'FullyReplicated',
            }},
        }],
        OutputDataConfig={
            'S3OutputPath': f's3://{args.bucket}/{args.output_prefix}/{job_name}'
        },
        ResourceConfig={
            'InstanceType':     args.instance,
            'InstanceCount':    1,
            'VolumeSizeInGB':   args.volume_size,
        },
        StoppingCondition={'MaxRuntimeInSeconds': args.max_runtime},
        EnableManagedSpotTraining=False,
    )

    console = (f'https://{region}.console.aws.amazon.com/sagemaker/home'
               f'?region={region}#/jobs/{job_name}')
    print(f'\nJob submitted : {job_name}')
    print(f'Console       : {console}')
    print(f'\nLog format to watch:')
    print(f'  Epoch  N/50 | loss=X | Dice NCR=X ED=X ET=X WT=X Mean=X | '
          f'HD95 NCR=Xmm ED=Xmm ET=Xmm WT=Xmm Mean=Xmm')
    print(f'\nOnce complete, pass the printed S3 artifact URI to launch_resnet_exp_b1.py '
          f'via --exp-a-model-s3')


def main():
    args = parse_args()
    if args.monitor:
        monitor_job(args.monitor)
    else:
        if not args.bucket:
            raise ValueError('--bucket is required.')
        launch(args)


if __name__ == '__main__':
    main()
