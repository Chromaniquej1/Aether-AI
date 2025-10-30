# 🚀 ModelForge-CV Quick Start Guide

Get your first model trained in **5 minutes**!

---

## 📋 Prerequisites

- Python 3.9+
- 4GB+ RAM (16GB recommended for GPU)
- (Optional) NVIDIA GPU with CUDA 11.8+

---

## ⚡ Quick Setup (2 minutes)

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/modelforge-cv.git
cd modelforge-cv

# Run setup script
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh

# Activate environment
source venv/bin/activate
```

### Option 2: Manual Setup

```bash
# Clone and enter directory
git clone https://github.com/yourusername/modelforge-cv.git
cd modelforge-cv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (GPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

---

## 🎯 Your First Training Run (3 minutes)

### Prepare a Sample Dataset

Create a simple image classification dataset:

```bash
mkdir -p sample_dataset/train/cats
mkdir -p sample_dataset/train/dogs

# Download some sample images or use your own
# Structure should be:
# sample_dataset/
#   train/
#     cats/
#       cat1.jpg
#       cat2.jpg
#     dogs/
#       dog1.jpg
#       dog2.jpg
```

### Compress Dataset

```bash
cd sample_dataset
zip -r ../sample_dataset.zip .
cd ..
```

### Run Training

```python
# Create file: train_example.py
from backend.core.pipeline import ModelForgePipeline

# Initialize pipeline
pipeline = ModelForgePipeline(
    project_name="my_first_classifier",
    workspace_dir="./workspaces"
)

# Run complete training pipeline
results = pipeline.run_complete_pipeline(
    dataset_zip_path="./sample_dataset.zip",
    backbone="efficientnet_b0",  # Fast model for testing
    finetuning_mode="lora",
    experiment_name="exp_001",
    epochs=5,  # Just 5 epochs for quick test
    batch_size=16,
    learning_rate=1e-3
)

# Print results
print(f"\n✅ Training Complete!")
print(f"Best Accuracy: {results['training_summary']['best_val_acc']:.2%}")
print(f"Model saved at: {results['model_path']}")
```

Run it:

```bash
python train_example.py
```

**Expected output:**
```
🚀 Starting Training
Device: cuda:0
Epochs: 5
Batch size: 16
...
✅ Training Complete!
Best Accuracy: 94.5%
Model saved at: ./workspaces/my_first_classifier/models/exp_001_final.pt
```

---

## 🔥 Using the REST API

### Start the API Server

```bash
# Terminal 1: Start backend
uvicorn backend.api.main:app --reload --port 8000
```

Visit: http://localhost:8000/docs for interactive API documentation

### Example API Usage

```python
# api_example.py
import requests

API_URL = "http://localhost:8000"

# 1. Create a project
response = requests.post(f"{API_URL}/api/projects", json={
    "name": "My Classifier",
    "description": "Testing ModelForge",
    "task_type": "classification"
})
project = response.json()
project_id = project["id"]
print(f"✅ Project created: {project_id}")

# 2. Upload dataset
with open("sample_dataset.zip", "rb") as f:
    files = {"file": ("dataset.zip", f, "application/zip")}
    response = requests.post(
        f"{API_URL}/api/projects/{project_id}/datasets",
        files=files
    )
dataset = response.json()
dataset_id = dataset["id"]
print(f"✅ Dataset uploaded: {dataset_id}")

# 3. Start training
response = requests.post(f"{API_URL}/api/training/jobs", json={
    "project_id": project_id,
    "dataset_id": dataset_id,
    "backbone": "efficientnet_b0",
    "finetuning_mode": "lora",
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 1e-4
})
job = response.json()
job_id = job["job_id"]
print(f"✅ Training started: {job_id}")

# 4. Check status
import time
while True:
    response = requests.get(f"{API_URL}/api/training/jobs/{job_id}")
    job_status = response.json()
    
    print(f"Status: {job_status['status']}")
    
    if job_status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(10)

print(f"✅ Training {job_status['status']}!")
if job_status['metrics']:
    print(f"Best Accuracy: {job_status['metrics']['best_val_acc']:.2%}")
```

---

## 🐳 Using Docker (Recommended for Production)

### Start All Services

```bash
# Start everything with docker-compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

**Services running:**
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Flower (Celery UI): http://localhost:5555

### Stop Services

```bash
docker-compose down
```

---

## 📚 Common Use Cases

### Use Case 1: Image Classification

```python
pipeline = ModelForgePipeline(project_name="image_classifier")

results = pipeline.run_complete_pipeline(
    dataset_zip_path="./datasets/flowers.zip",
    backbone="vit_base_patch16_224",  # Vision Transformer
    finetuning_mode="lora",
    epochs=30,
    batch_size=32
)
```

### Use Case 2: Quick Prototyping (Fast Training)

```python
pipeline = ModelForgePipeline(project_name="quick_test")

results = pipeline.run_complete_pipeline(
    dataset_zip_path="./datasets/test.zip",
    backbone="efficientnet_b0",  # Fastest model
    finetuning_mode="linear_probe",  # Freeze backbone
    epochs=10,
    batch_size=64
)
```

### Use Case 3: Best Accuracy (Large Dataset)

```python
pipeline = ModelForgePipeline(project_name="production_model")

results = pipeline.run_complete_pipeline(
    dataset_zip_path="./datasets/large_dataset.zip",
    backbone="vit_large_patch16_224",  # Largest model
    finetuning_mode="full",  # Full fine-tuning
    epochs=50,
    batch_size=16,
    learning_rate=1e-5
)
```

### Use Case 4: Memory-Efficient Training

```python
pipeline = ModelForgePipeline(project_name="memory_efficient")

results = pipeline.run_complete_pipeline(
    dataset_zip_path="./datasets/medium_dataset.zip",
    backbone="convnext_base",
    finetuning_mode="lora",  # Uses 90% less memory
    epochs=30,
    batch_size=8,  # Small batch size
    learning_rate=1e-4
)
```

---

## 🎓 Model Selection Guide

| Use Case | Model | Mode | Time | Accuracy |
|----------|-------|------|------|----------|
| **Quick Test** | efficientnet_b0 | linear_probe | ⚡⚡⚡ Fast | ⭐⭐ Good |
| **Balanced** | vit_base | lora | ⚡⚡ Medium | ⭐⭐⭐⭐ High |
| **Best Accuracy** | vit_large | full | ⚡ Slow | ⭐⭐⭐⭐⭐ Best |
| **Low Memory** | efficientnet_b3 | lora | ⚡⚡ Medium | ⭐⭐⭐ High |
| **Modern CNN** | convnext_base | lora | ⚡⚡ Medium | ⭐⭐⭐⭐ High |

---

## 🔍 Troubleshooting

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size or use LoRA mode

```python
# Change from:
batch_size=64

# To:
batch_size=16
finetuning_mode="lora"  # Uses 90% less memory
```

### Issue: Training Too Slow

**Solution:** Use smaller model or linear probe

```python
# Change from:
backbone="vit_large_patch16_224"

# To:
backbone="efficientnet_b0"
finetuning_mode="linear_probe"  # 3x faster
```

### Issue: Dataset Validation Failed

**Solution:** Check dataset structure

```
Correct structure:
dataset/
  class1/
    image1.jpg
    image2.jpg
  class2/
    image3.jpg
    image4.jpg

Incorrect structure:
dataset/
  image1.jpg  ❌ (needs class folders)
```

### Issue: Import Errors

**Solution:** Reinstall dependencies

```bash
pip install -r requirements.txt --force-reinstall
```

---

## 📊 Monitoring Training

### View Real-time Logs

Training progress is automatically displayed with:
- Current epoch
- Loss and accuracy
- Time per epoch
- GPU memory usage
- Estimated completion time

### Check Saved Metrics

```python
import json

# Load training metrics
with open("./workspaces/my_project/experiments/exp_001/metrics.json") as f:
    metrics = json.load(f)

# Print epoch-by-epoch results
for epoch_data in metrics:
    print(f"Epoch {epoch_data['epoch']}: "
          f"Val Acc = {epoch_data['val_acc']:.2%}")
```

---

## 🎯 Next Steps

1. **Try different models**: Experiment with various backbones
2. **Tune hyperparameters**: Adjust learning rate, batch size
3. **Deploy model**: Use model serving endpoints
4. **Build UI**: Connect to the frontend dashboard
5. **Scale up**: Use larger datasets and GPUs

---

## 💡 Pro Tips

### Tip 1: Use LoRA for 90% Less Memory

```python
finetuning_mode="lora"  # Instead of "full"
lora_r=8  # Lower = less memory, higher = better accuracy
```

### Tip 2: Early Stopping Prevents Overfitting

```python
early_stopping_patience=10  # Stop if no improvement for 10 epochs
```

### Tip 3: Adjust Learning Rate by Model Size

```python
# Small models
learning_rate=1e-3

# Medium models  
learning_rate=1e-4

# Large models
learning_rate=1e-5
```

### Tip 4: Use Augmentation for Small Datasets

Dataset < 1000 images → Heavy augmentation automatically applied

---

## 📞 Get Help

- **Documentation**: Check `docs/` folder
- **Examples**: See `examples/` folder
- **Issues**: GitHub Issues
- **Discord**: Community support

---

## ✅ Success Checklist

- [ ] Environment setup complete
- [ ] First training run successful
- [ ] Understand model selection
- [ ] API endpoints working
- [ ] Docker setup (optional)
- [ ] Read full documentation

---

**Congratulations! You're ready to train custom vision models! 🎉**

Next: Check out the [User Guide](docs/user-guide.md) for advanced features.