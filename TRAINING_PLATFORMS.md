# Alternative Training Platforms for Brain Tumor Segmentation

This guide provides options for training your 3D brain tumor segmentation model beyond Google Colab.

## Quick Comparison

| Platform | GPU Options | Time Limits | Cost | Best For |
|----------|------------|-------------|------|----------|
| **Kaggle** | T4, P100 (30h/week) | 12h per session | FREE | Best free option |
| **Google Colab Pro** | T4, V100, A100 | 24h | $10/month | Quick experiments |
| **Paperspace Gradient** | RTX 4000, A4000, A6000 | None | $0.51-$3.18/h | Medium training runs |
| **AWS SageMaker** | ml.p3, ml.g4dn, ml.g5 | None | $1.26-$4.09/h | Production training |
| **Lambda Labs** | A100, H100 | None | $1.10-$2.49/h | Cost-effective GPU cloud |

---

## 1. Kaggle Notebooks (RECOMMENDED - FREE)

**Best free alternative to Google Colab with better GPU quotas.**

### Advantages
- ✅ FREE 30 hours/week of GPU (T4 or P100)
- ✅ 12-hour session limit (same as Colab)
- ✅ More stable than free Colab
- ✅ Direct dataset integration
- ✅ Can save outputs to Kaggle Datasets

### Setup Steps

1. **Create Kaggle account**: https://www.kaggle.com/

2. **Upload your data as a Kaggle Dataset**:
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload your preprocessed H5 files (or use preprocessing script on Kaggle)
   - Make it private if needed

3. **Create a new notebook**:
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Settings → Accelerator → GPU T4 x2

4. **Add your dataset**:
   - Click "Add Data" → "Your Datasets"
   - Select your brain tumor dataset
   - It will be available at `/kaggle/input/your-dataset-name/`

5. **Upload your code**:
   - Upload `model.py`, `data_generator.py`, `losses.py`, `preprocess_data.py`
   - Or clone from GitHub: `!git clone https://github.com/dariamarc/brainTumorSurvival.git`

6. **Run training**:
   ```python
   # In Kaggle notebook
   import sys
   sys.path.append('/kaggle/working/brainTumorSurvival')

   from model import MProtoNet3D_Segmentation_Keras
   from data_generator import MRIDataGenerator
   from losses import CombinedLoss

   # Point to Kaggle dataset
   DATA_PATH = '/kaggle/input/brain-tumor-preprocessed/'

   # Train as usual...
   ```

7. **Save checkpoints**:
   ```python
   # Kaggle automatically saves /kaggle/working/ directory
   # Save models here
   model.save('/kaggle/working/best_model.keras')
   ```

### Tips for Kaggle
- Enable "Internet" in notebook settings to clone from GitHub
- Use "Save Version" frequently to checkpoint progress
- Download models from Output tab after training

---

## 2. Paperspace Gradient

**Affordable pay-as-you-go with no time limits.**

### Advantages
- ✅ No session time limits
- ✅ Various GPU options (RTX 4000, A4000, A6000, A100)
- ✅ Can pause and resume
- ✅ Persistent storage

### Pricing
- Free tier: Limited CPU (not suitable for this)
- RTX 4000 (8GB): $0.51/hour
- A4000 (16GB): $0.76/hour
- A6000 (48GB): $1.89/hour

### Setup Steps

1. **Sign up**: https://www.paperspace.com/gradient

2. **Create a notebook**:
   - New Notebook → PyTorch or TensorFlow template
   - Select GPU (RTX 4000 recommended for budget)
   - Auto-shutdown: 6 hours (to save money)

3. **Upload data**:
   ```bash
   # Option 1: Upload via UI (slow for large datasets)
   # Option 2: Download from S3
   aws s3 sync s3://your-bucket/path /storage/data

   # Option 3: Use Paperspace datasets
   # Upload preprocessed data as Paperspace dataset
   ```

4. **Clone repo and train**:
   ```bash
   git clone https://github.com/dariamarc/brainTumorSurvival.git
   cd brainTumorSurvival
   pip install -r requirements.txt
   python main.py --data_path /storage/data
   ```

5. **Monitor and save**:
   - Use `/storage` for persistent data (survives notebook restart)
   - Use Paperspace Datasets to store preprocessed data
   - Download models via UI or `paperspace-python` CLI

---

## 3. AWS SageMaker

**Enterprise-grade, most reliable, scales well.**

### Advantages
- ✅ No time limits
- ✅ Integrated with S3 (your data is already there!)
- ✅ Managed Spot training (up to 90% cost savings)
- ✅ Best for production workflows

### Pricing (On-Demand)
- ml.g4dn.xlarge (T4, 16GB): $0.526/hour
- ml.p3.2xlarge (V100, 16GB): $3.06/hour
- **Spot instances**: 50-90% cheaper

### Setup Steps

1. **Create training script** (`train.py`):
   ```python
   import os
   import argparse
   from model import MProtoNet3D_Segmentation_Keras
   from data_generator import MRIDataGenerator
   from losses import CombinedLoss
   import tensorflow as tf

   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
       parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
       parser.add_argument('--epochs', type=int, default=100)
       parser.add_argument('--batch-size', type=int, default=4)
       args = parser.parse_args()

       # Load data
       train_gen = MRIDataGenerator(args.data_dir, batch_size=args.batch_size)

       # Build model
       model = MProtoNet3D_Segmentation_Keras(...)

       # Train
       model.fit(train_gen, epochs=args.epochs)

       # Save
       model.save(f'{args.model_dir}/model.keras')
   ```

2. **Create SageMaker notebook** (for launching training):
   ```python
   import sagemaker
   from sagemaker.tensorflow import TensorFlow

   # Your S3 paths
   s3_data = 's3://your-bucket/brainTumorData'
   s3_output = 's3://your-bucket/models'

   # Create estimator
   estimator = TensorFlow(
       entry_point='train.py',
       source_dir='.',  # Directory with your code
       role=sagemaker.get_execution_role(),
       instance_count=1,
       instance_type='ml.g4dn.xlarge',  # or ml.p3.2xlarge for V100
       framework_version='2.13',
       py_version='py310',
       hyperparameters={
           'epochs': 100,
           'batch-size': 4
       },
       use_spot_instances=True,  # Save money!
       max_wait=86400,  # 24 hours
       max_run=72000    # 20 hours
   )

   # Start training
   estimator.fit({'training': s3_data})
   ```

3. **Monitor training**:
   - CloudWatch logs show training progress
   - Model saved to S3 automatically
   - Can stop/restart anytime

### Cost Optimization
- Use **Spot instances** (70-90% cheaper)
- Use **ml.g4dn.xlarge** for preprocessed data (sufficient for 160x160x96)
- Set `max_wait` and `max_run` to avoid runaway costs

---

## 4. Lambda Labs Cloud GPU

**Cheapest dedicated GPU cloud.**

### Advantages
- ✅ Lowest prices for A100 GPUs
- ✅ No time limits
- ✅ SSH access, full control
- ✅ Pre-installed deep learning frameworks

### Pricing
- RTX 6000 Ada (48GB): $0.50/hour
- A100 (40GB): $1.10/hour
- A100 (80GB): $1.29/hour
- H100 (80GB): $2.49/hour

### Setup Steps

1. **Sign up**: https://lambdalabs.com/service/gpu-cloud

2. **Launch instance**:
   - Choose GPU type (1x RTX 6000 Ada recommended)
   - Select region
   - Add SSH key

3. **Connect via SSH**:
   ```bash
   ssh ubuntu@<instance-ip>
   ```

4. **Setup environment**:
   ```bash
   # TensorFlow pre-installed
   git clone https://github.com/dariamarc/brainTumorSurvival.git
   cd brainTumorSurvival
   pip install h5py scipy tqdm

   # Upload preprocessed data via rsync or download from S3
   aws s3 sync s3://your-bucket/preprocessed /home/ubuntu/data
   ```

5. **Run training in tmux** (to survive disconnection):
   ```bash
   tmux new -s training
   python main.py --data_path /home/ubuntu/data
   # Press Ctrl+B then D to detach

   # Reattach later
   tmux attach -t training
   ```

6. **Download model**:
   ```bash
   # From your local machine
   scp ubuntu@<instance-ip>:/home/ubuntu/brainTumorSurvival/best_model.keras .
   ```

---

## 5. Google Colab Pro/Pro+

**Upgraded Colab with better GPUs and longer sessions.**

### Colab Pro ($10/month)
- V100 or A100 GPUs (not guaranteed)
- 24-hour timeout (vs 12 hours free)
- More memory

### Colab Pro+ ($50/month)
- A100 GPUs more often
- 24-hour timeout
- Background execution
- Highest priority

### Worth it?
- ✅ If you're already familiar with Colab
- ❌ Kaggle offers better value for free
- ❌ Still has disconnection issues
- ❌ Lambda Labs cheaper for long training

---

## Recommended Workflow

### For Initial Experimentation (FREE)
1. Use **Kaggle** with preprocessed data (160x160x96)
2. Train for a few epochs to validate pipeline
3. Use small subset of data first

### For Full Training Run
**Option A: Budget-conscious**
- **Lambda Labs** 1x A100 (~$30-50 for full training)
- Best price/performance

**Option B: AWS ecosystem**
- **SageMaker Spot** with ml.g4dn.xlarge (~$20-40 with spot)
- Best if data already in S3

**Option C: No credit card**
- **Kaggle** with preprocessed data
- Split training into multiple sessions

---

## Prerequisites for All Platforms

Create a `requirements.txt`:
```txt
tensorflow>=2.13.0
keras>=2.13.0
h5py>=3.8.0
numpy>=1.24.0
scipy>=1.10.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

---

## Summary & Recommendation

**For your use case (3D brain tumor segmentation, 369 volumes):**

1. **Start with Kaggle** (FREE)
   - Upload preprocessed data (160x160x96)
   - Validate full pipeline
   - Train for 10-20 epochs

2. **Full training on Lambda Labs** ($30-50 total)
   - 1x RTX 6000 Ada or A100
   - 20-30 hours training time
   - Most cost-effective

3. **Alternative: AWS SageMaker Spot**
   - If data already in S3
   - Use managed spot for cost savings
   - More complex setup but very reliable

---

## Next Steps

1. **Preprocess your data** (reduces size by 50%):
   ```bash
   python preprocess_data.py \
       --input_dir /path/to/original \
       --output_dir /path/to/preprocessed \
       --target_height 160 \
       --target_width 160 \
       --target_slices 96
   ```

2. **Choose platform** based on budget and preferences

3. **Upload preprocessed data** to chosen platform

4. **Train with smaller batch sizes first** to test

5. **Monitor training** and adjust hyperparameters

---

Need help setting up any of these platforms? Let me know which one you'd like to use!
