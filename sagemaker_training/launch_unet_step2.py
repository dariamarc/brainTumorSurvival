#!/usr/bin/env python3
"""
Launch UNet Step 2 — UNet3DProto, no prototype losses (5-fold CV)

Usage:
    python launch_unet_step2.py --bucket your-brats2020-data \
        --step1-weights-s3 s3://your-brats2020-data/unet-step1-outputs/.../output/model.tar.gz
    python launch_unet_step2.py --monitor unet-step2-20260510-140000
"""

import argparse, os, boto3, tarfile, tempfile, time
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bucket',             type=str)
    p.add_argument('--role',               type=str, default=None)
    p.add_argument('--data-prefix',        type=str, default='preprocessed_data_cropped')
    p.add_argument('--output-prefix',      type=str, default='unet-step2-outputs')
    p.add_argument('--instance',           type=str, default='ml.g4dn.xlarge')
    p.add_argument('--volume-size',        type=int, default=100)
    p.add_argument('--max-runtime',        type=int, default=172800)  # 48h sufficient for 5×60
    p.add_argument('--job-name',           type=str, default=None)
    p.add_argument('--monitor',            type=str, default=None)
    p.add_argument('--step1-weights-s3',   type=str, default=None,
                   help='S3 URI of Step 1 model.tar.gz (from monitor output)')
    # Hyperparameters
    p.add_argument('--n-folds',            type=int,   default=5)
    p.add_argument('--epochs',             type=int,   default=60)
    p.add_argument('--patience',           type=int,   default=20)
    p.add_argument('--lr',                 type=float, default=1e-4)
    p.add_argument('--lr-min',             type=float, default=1e-6)
    p.add_argument('--alpha-mdsc',         type=float, default=100.0)
    p.add_argument('--base-channels',      type=int,   default=32)
    p.add_argument('--protos-per-class',   type=int,   default=5)
    p.add_argument('--num-volumes',        type=int,   default=369)
    return p.parse_args()


def get_role():
    try:
        for r in boto3.client('iam').list_roles(MaxItems=100)['Roles']:
            if 'sagemaker' in r['RoleName'].lower():
                return r['Arn']
    except Exception:
        pass
    return None


def upload_code(bucket, job_name):
    code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet_step2')
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
    sm, last = boto3.client('sagemaker'), None
    print(f'\nMonitoring: {job_name}')
    while True:
        resp   = sm.describe_training_job(TrainingJobName=job_name)
        status = resp['TrainingJobStatus']
        if status != last:
            print(f'[{datetime.now().strftime("%H:%M:%S")}] {status}')
            last = status
            if status in ('Completed', 'Failed', 'Stopped'):
                if status == 'Failed':
                    print(f'Failure: {resp.get("FailureReason")}')
                if 'ModelArtifacts' in resp:
                    uri = resp['ModelArtifacts']['S3ModelArtifacts']
                    print(f'Artifacts : {uri}')
                    print(f'Pass to step3 via --step2-weights-s3 : {uri}')
                break
        time.sleep(30)


def launch(args):
    region    = boto3.Session().region_name or 'us-east-1'
    role      = args.role or get_role()
    if not role:
        raise ValueError('No SageMaker role found. Pass --role.')
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name  = args.job_name or f'unet-step2-{timestamp}'
    code_uri  = upload_code(args.bucket, job_name)
    image_uri = (f'763104351884.dkr.ecr.{region}.amazonaws.com/'
                 f'tensorflow-training:2.13.0-gpu-py310-cu118-ubuntu20.04-sagemaker')

    hp = {
        'n-folds':            str(args.n_folds),
        'num-volumes':        str(args.num_volumes),
        'num-slices':         '128', 'height': '192', 'width': '160', 'num-classes': '4',
        'base-channels':      str(args.base_channels),
        'protos-per-class':   str(args.protos_per_class),
        'epochs':             str(args.epochs),
        'patience':           str(args.patience),
        'lr':                 str(args.lr),
        'lr-min':             str(args.lr_min),
        'alpha-mdsc':         str(args.alpha_mdsc),
        'sagemaker_program':          'train.py',
        'sagemaker_submit_directory': code_uri,
    }
    # Pass Step 1 weights S3 URI so the script can download fold-specific checkpoints
    if args.step1_weights_s3:
        # SageMaker doubles the job name in the path — strip to get the actual tar URI
        hp['step1-weights-s3'] = args.step1_weights_s3

    boto3.client('sagemaker').create_training_job(
        TrainingJobName=job_name, RoleArn=role,
        AlgorithmSpecification={'TrainingImage': image_uri, 'TrainingInputMode': 'File'},
        HyperParameters=hp,
        InputDataConfig=[{'ChannelName': 'training', 'DataSource': {'S3DataSource': {
            'S3DataType': 'S3Prefix',
            'S3Uri': f's3://{args.bucket}/{args.data_prefix}',
            'S3DataDistributionType': 'FullyReplicated'}}}],
        OutputDataConfig={'S3OutputPath': f's3://{args.bucket}/{args.output_prefix}/{job_name}'},
        ResourceConfig={'InstanceType': args.instance, 'InstanceCount': 1,
                        'VolumeSizeInGB': args.volume_size},
        StoppingCondition={'MaxRuntimeInSeconds': args.max_runtime},
        EnableManagedSpotTraining=False,
    )
    console = (f'https://{region}.console.aws.amazon.com/sagemaker/home'
               f'?region={region}#/jobs/{job_name}')
    print(f'\nJob     : {job_name}')
    print(f'Console : {console}')
    if not args.step1_weights_s3:
        print('\n⚠  No --step1-weights-s3 provided — step2 will train from scratch')


def main():
    args = parse_args()
    if args.monitor:
        monitor_job(args.monitor)
    else:
        if not args.bucket:
            raise ValueError('--bucket required')
        launch(args)


if __name__ == '__main__':
    main()
