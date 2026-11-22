"""
AGROSMART AI - Master Training Orchestration Script
Runs end-to-end training across:
1. Crop Recommendation Classification Suite (8 algorithms)
2. Crop Yield Regression Suite (8 algorithms)
3. Deep Learning Disease Detection Suite (Custom CNN, MobileNetV2, EfficientNetB0)
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.crop_recommendation import train_and_evaluate_all_crop_models
from src.yield_prediction import train_and_evaluate_all_yield_models
from src.disease_detection import train_and_compare_deep_learning_models
from src.eda import generate_all_eda_visualizations


def run_full_training_pipeline(train_dl: bool = True, dl_epochs: int = 5):
    print("\n" + "=" * 60)
    print(">>> STARTING AGROSMART AI MASTER MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. EDA Visualizations
    print("\n[Step 1/4] Generating Complete EDA Visualizations...")
    try:
        generate_all_eda_visualizations()
    except Exception as e:
        print(f"Warning: EDA generation encountered: {e}")

    # 2. Crop Recommendation Models
    print("\n[Step 2/4] Training Crop Recommendation Classification Models...")
    crop_metrics, _, _ = train_and_evaluate_all_crop_models()
    print(f"✓ Trained {len(crop_metrics)} crop recommendation models.")

    # 3. Crop Yield Models
    print("\n[Step 3/4] Training Crop Yield Regression Models...")
    yield_metrics, _, _ = train_and_evaluate_all_yield_models()
    print(f"✓ Trained {len(yield_metrics)} yield prediction models.")

    # 4. Deep Learning Image Models
    if train_dl:
        print("\n[Step 4/4] Training Deep Learning Image Models (CNN, MobileNetV2, EfficientNetB0)...")
        dl_metrics = train_and_compare_deep_learning_models(epochs=dl_epochs)
        print(f"✓ Trained {len(dl_metrics)} deep learning disease detection models.")
    else:
        print("\n[Step 4/4] Skipping Deep Learning training (flag disabled).")

    print("\n" + "=" * 60)
    print("[SUCCESS] AGROSMART AI MASTER TRAINING PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGROSMART AI Model Training")
    parser.add_argument("--skip-dl", action="store_true", help="Skip deep learning models")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for deep learning")
    args = parser.parse_args()

    run_full_training_pipeline(train_dl=not args.skip_dl, dl_epochs=args.epochs)
