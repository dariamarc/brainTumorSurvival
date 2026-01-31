#!/usr/bin/env python3
"""
Launch SageMaker Training Job for PrototypeSegNet3D

This script submits a training job to SageMaker that runs independently
of your local machine. You can close your laptop and the training continues.

Usage:
    python launch_training.py --bucket your-bucket-name

    # With custom settings:
    python launch_training.py --bucket my-bucket --instance ml.g4dn.xlarge --phase1-epochs 30

    # Monitor existing job:
    python launch_training.py --monitor protoseg-20240115-123456
"""

import argparse
import os
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from datetime import datetime
import time
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Launch SageMaker Training Job')

    # Required
    parser.add_argument('--bucket', type=str, required=True,
                        help='S3 bucket name containing the training data')

    # S3 paths
    parser.add_argument('--data-prefix', type=str, default='preprocessed_data_cropped',
                        help='S3 prefix for training data (default: preprocessed_data_cropped)')
    parser.add_argument('--output-prefix', type=str, default='training-outputs',
                        help='S3 prefix for outputs (default: training-outputs)')

    # Instance configuration
    parser.add_argument('--instance', type=str, default='ml.g4dn.xlarge',
                        help='SageMaker instance type (default: ml.g4dn.xlarge)')
    parser.add_argument('--volume-size', type=int, default=100,
                        help='EBS volume size in GB (default: 100)')
    parser.add_argument('--max-runtime', type=int, default=432000,
                        help='Max runtime in seconds (default: 432000 = 5 days)')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--phase1-epochs', type=int, default=50)
    parser.add_argument('--phase2-epochs', type=int, default=150)
    parser.add_argument('--phase3-epochs', type=int, default=30)
    parser.add_argument('--alpha-mdsc', type=float, default=100.0)

    # Job management
    parser.add_argument('--job-name', type=str, default=None,
                        help='Custom job name (default: auto-generated)')
    parser.add_argument('--monitor', type=str, default=None,
                        help='Monitor an existing job instead of launching new one')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for job completion and stream logs')

    # IAM role
    parser.add_argument('--role', type=str, default=None,
                        help='SageMaker IAM role ARN (default: auto-detect)')

    return parser.parse_args()


def get_sagemaker_role():
    """Get SageMaker execution role."""
    try:
        # Try to get role from SageMaker session
        return sagemaker.get_execution_role()
    except ValueError:
        # If running locally, try to get from environment or IAM
        iam = boto3.client('iam')
        roles = iam.list_roles()['Roles']
        for role in roles:
            if 'SageMaker' in role['RoleName'] or 'sagemaker' in role['RoleName']:
                return role['Arn']
        raise ValueError(
            "Could not find SageMaker role. Please specify --role argument.\n"
            "Example: --role arn:aws:iam::123456789:role/SageMakerRole"
        )


def monitor_job(job_name, sm_client):
    """Monitor a training job and print status updates."""
    print(f"\nMonitoring job: {job_name}")
    print("=" * 60)

    logs_client = boto3.client('logs')
    log_group = '/aws/sagemaker/TrainingJobs'

    last_status = None
    while True:
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']

        if status != last_status:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {status}")
            last_status = status

            if status == 'InProgress':
                # Print resource config
                if 'ResourceConfig' in response:
                    rc = response['ResourceConfig']
                    print(f"  Instance: {rc.get('InstanceType')} x {rc.get('InstanceCount')}")

            elif status == 'Completed':
                print("\n" + "=" * 60)
                print("TRAINING COMPLETED SUCCESSFULLY!")
                print("=" * 60)

                # Print metrics
                if 'FinalMetricDataList' in response:
                    print("\nFinal Metrics:")
                    for metric in response['FinalMetricDataList']:
                        print(f"  {metric['MetricName']}: {metric['Value']:.4f}")

                # Print output location
                if 'ModelArtifacts' in response:
                    print(f"\nModel artifacts: {response['ModelArtifacts']['S3ModelArtifacts']}")

                return True

            elif status == 'Failed':
                print("\n" + "=" * 60)
                print("TRAINING FAILED!")
                print("=" * 60)
                if 'FailureReason' in response:
                    print(f"Reason: {response['FailureReason']}")
                return False

            elif status == 'Stopped':
                print("\nTraining job was stopped.")
                return False

        # Print billable seconds periodically
        if 'BillableTimeInSeconds' in response:
            hours = response['BillableTimeInSeconds'] / 3600
            print(f"  Billable time: {hours:.2f} hours", end='\r')

        time.sleep(30)


def launch_training_job(args):
    """Launch a SageMaker training job."""
    session = sagemaker.Session()
    sm_client = boto3.client('sagemaker')

    # Get role
    role = args.role or get_sagemaker_role()
    print(f"Using IAM role: {role}")

    # Generate job name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = args.job_name or f'protoseg-{timestamp}'
    print(f"Job name: {job_name}")

    # S3 paths
    s3_data = f's3://{args.bucket}/{args.data_prefix}'
    s3_output = f's3://{args.bucket}/{args.output_prefix}/{job_name}'
    print(f"Training data: {s3_data}")
    print(f"Output path: {s3_output}")

    # Hyperparameters
    hyperparameters = {
        'batch-size': args.batch_size,
        'phase1-epochs': args.phase1_epochs,
        'phase2-epochs': args.phase2_epochs,
        'phase3-epochs': args.phase3_epochs,
        'alpha-mdsc': args.alpha_mdsc,
        'num-volumes': 369,
        'num-slices': 128,
        'depth': 128,
        'height': 160,
        'width': 192,
        'channels': 4,
        'num-classes': 4,
        'n-prototypes': 3,
    }

    print("\nHyperparameters:")
    for k, v in hyperparameters.items():
        print(f"  {k}: {v}")

    # Determine source directory (use package/ if it exists, otherwise current dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.join(script_dir, 'package')
    source_dir = package_dir if os.path.exists(package_dir) else script_dir

    print(f"Source directory: {source_dir}")

    # Create TensorFlow estimator
    estimator = TensorFlow(
        entry_point='train.py',
        source_dir=source_dir,  # Upload packaged code
        role=role,
        instance_count=1,
        instance_type=args.instance,
        framework_version='2.15.0',
        py_version='py310',
        volume_size=args.volume_size,
        max_run=args.max_runtime,
        output_path=s3_output,
        hyperparameters=hyperparameters,
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'train_loss=([0-9\\.]+)'},
            {'Name': 'val:loss', 'Regex': 'val_loss=([0-9\\.]+)'},
            {'Name': 'dice:mean', 'Regex': 'Mean=([0-9\\.]+)'},
            {'Name': 'dice:gd_enhancing', 'Regex': 'ET=([0-9\\.]+)'},
            {'Name': 'dice:edema', 'Regex': 'ED=([0-9\\.]+)'},
            {'Name': 'dice:necrotic', 'Regex': 'NCR=([0-9\\.]+)'},
        ],
        environment={
            'TF_CPP_MIN_LOG_LEVEL': '2',  # Reduce TF verbosity
        },
        # Keep alive for debugging if needed
        keep_alive_period_in_seconds=0,
    )

    print("\n" + "=" * 60)
    print("LAUNCHING TRAINING JOB")
    print("=" * 60)

    # Launch the job
    estimator.fit(
        inputs={'training': s3_data},
        job_name=job_name,
        wait=False  # Don't wait - return immediately
    )

    print(f"\nTraining job submitted: {job_name}")
    print("\n" + "=" * 60)
    print("HOW TO MONITOR YOUR JOB")
    print("=" * 60)
    print(f"""
1. AWS Console:
   https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}

2. CloudWatch Logs:
   https://console.aws.amazon.com/cloudwatch/home#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FTrainingJobs

3. This script (run from anywhere):
   python launch_training.py --monitor {job_name}

4. AWS CLI:
   aws sagemaker describe-training-job --training-job-name {job_name}

5. Stream logs:
   aws logs tail /aws/sagemaker/TrainingJobs/{job_name}/algo-1-{timestamp} --follow
""")

    print("=" * 60)
    print("OUTPUTS (after completion)")
    print("=" * 60)
    print(f"""
Model artifacts will be saved to:
  {s3_output}/output/model.tar.gz

This tarball contains:
  - model_final.keras (final trained model)
  - model_after_phase1.keras
  - model_after_phase2.keras
  - checkpoints/

Training outputs will be in:
  {s3_output}/output/output.tar.gz

This tarball contains:
  - training_summary.json
  - training_history.json
""")

    if args.wait:
        print("\nWaiting for job completion (Ctrl+C to detach)...")
        monitor_job(job_name, sm_client)

    return job_name


def main():
    args = parse_args()

    if args.monitor:
        # Monitor existing job
        sm_client = boto3.client('sagemaker')
        monitor_job(args.monitor, sm_client)
    else:
        # Launch new job
        if not args.bucket:
            print("Error: --bucket is required to launch a new job")
            return
        launch_training_job(args)


if __name__ == '__main__':
    main()
