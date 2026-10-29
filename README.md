# LatentWeave Workflow

## 🚀 Quick Start

### Setup and Dependencies
```
conda create -n LatentWeave python=3.10
conda activate LatentWeave
pip install -r requirements.txt
```

### 1. Step 1: Run main.py (optimize & save data)

```bash
python main.py \
  --model_path <path/to/Qwen2.5-VL> \
  --dataset mmmu \
  --output_dir ./outputs \
  --start_idx 0 \
  --end_idx 300
```


### 2. Step 2: Train \<WEAVE\> tokens & WEAVEGen

```bash
python train_weave_anchors.py \
  --training_data ./outputs/vl_training_data_mmmu_new.json \
  --model_path <same Qwen2.5-VL path> \
  --output_dir ./weavegen_output \
  --epochs 3 \
  --batch_size 2
```

### 3. Step 3: Evaluate WEAVEGen

```bash
python eval_weave.py \
  --weavegen_path ./weavegen_output \
  --dataset mmmu \
  --output_dir ./eval_outputs \
  --start_idx 0 \
  --end_idx 300
```
