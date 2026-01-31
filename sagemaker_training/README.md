# SageMaker Training for PrototypeSegNet3D

Run training as a managed SageMaker job that continues even when you close your laptop.

## Prerequisites

1. **AWS CLI configured** with your credentials:
   ```bash
   aws configure
   ```

2. **Python packages** (on your local machine):
   ```bash
   pip install boto3 sagemaker
   ```

3. **Training data uploaded to S3**:
   ```bash
   aws s3 cp preprocessed_data_cropped.zip s3://your-bucket/preprocessed_data_cropped.zip
   # Or sync the extracted folder:
   aws s3 sync ./preprocessed_data_cropped s3://your-bucket/preprocessed_data_cropped/
   ```

4. **SageMaker IAM Role** with permissions to:
   - Access your S3 bucket
   - Create training jobs
   - Write to CloudWatch logs

## Quick Start

### Option 1: Using the shell script (recommended)

```bash
cd sagemaker_training
./prepare_and_launch.sh your-bucket-name
```

With custom parameters:
```bash
./prepare_and_launch.sh your-bucket-name --phase1-epochs 30 --phase2-epochs 100
```

### Option 2: Manual steps

1. **Prepare the package** (copy project files):
   ```bash
   cd sagemaker_training
   mkdir -p package
   cp train.py requirements.txt package/
   cp -r ../ResNet_architecture package/
   cp -r ../data_processing package/
   ```

2. **Launch the job**:
   ```bash
   python launch_training.py --bucket your-bucket-name
   ```

## Monitoring Your Job

### AWS Console (easiest)
After launching, you'll get a direct link to the job in the AWS Console.

### Command line
```bash
# Check status
python launch_training.py --monitor protoseg-YYYYMMDD-HHMMSS

# Or using AWS CLI
aws sagemaker describe-training-job --training-job-name protoseg-YYYYMMDD-HHMMSS
```

### CloudWatch Logs
```bash
# Stream logs in real-time
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix protoseg-YYYYMMDD
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bucket` | required | S3 bucket with training data |
| `--instance` | ml.g4dn.xlarge | EC2 instance type |
| `--phase1-epochs` | 50 | Warm-up phase epochs |
| `--phase2-epochs` | 150 | Joint fine-tuning epochs |
| `--phase3-epochs` | 30 | Refinement epochs |
| `--alpha-mdsc` | 100.0 | Weight for mDSC in hybrid loss |
| `--batch-size` | 1 | Training batch size |
| `--wait` | false | Wait and stream logs |

### Instance Recommendations

| Instance | GPU | Memory | Cost/hr* | Use Case |
|----------|-----|--------|----------|----------|
| ml.g4dn.xlarge | T4 16GB | 16GB | ~$0.75 | Development/testing |
| ml.g4dn.2xlarge | T4 16GB | 32GB | ~$1.10 | Standard training |
| ml.g5.xlarge | A10G 24GB | 16GB | ~$1.40 | Faster training |
| ml.g5.2xlarge | A10G 24GB | 32GB | ~$1.80 | Large batch sizes |
| ml.p3.2xlarge | V100 16GB | 61GB | ~$4.00 | Maximum performance |

*Prices are approximate and vary by region.

## Outputs

After training completes, outputs are saved to S3:

```
s3://your-bucket/training-outputs/protoseg-YYYYMMDD-HHMMSS/
├── output/
│   ├── model.tar.gz          # Contains trained models
│   │   ├── model_final.keras
│   │   ├── model_after_phase1.keras
│   │   ├── model_after_phase2.keras
│   │   └── checkpoints/
│   └── output.tar.gz         # Contains logs and metrics
│       ├── training_summary.json
│       └── training_history.json
```

### Download results
```bash
# Download everything
aws s3 sync s3://your-bucket/training-outputs/protoseg-YYYYMMDD-HHMMSS ./results

# Or just the final model
aws s3 cp s3://your-bucket/training-outputs/protoseg-YYYYMMDD-HHMMSS/output/model.tar.gz .
tar -xzf model.tar.gz
```

## Troubleshooting

### "ResourceLimitExceeded" error
Your AWS account may need a service limit increase for GPU instances.
Request via: AWS Console → Service Quotas → Amazon SageMaker

### "Could not find SageMaker role"
Specify the role ARN directly:
```bash
python launch_training.py --bucket your-bucket \
    --role arn:aws:iam::123456789:role/SageMakerExecutionRole
```

### Job stuck in "Starting"
GPU instances can take 5-10 minutes to provision. Check CloudWatch logs for progress.

### Out of memory
- Use a larger instance (ml.g5.2xlarge or ml.p3.2xlarge)
- Or reduce batch size: `--batch-size 1`

## Cost Estimation

For full training (~230 epochs total):
- ml.g4dn.xlarge: ~$15-25 (20-30 hours)
- ml.g5.xlarge: ~$20-35 (15-25 hours)
- ml.p3.2xlarge: ~$40-60 (10-15 hours)

*Estimates based on typical training times. Actual costs depend on epochs, early stopping, etc.*
