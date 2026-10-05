# AIGC-5500 Lab 3 - From Frozen Features to Fine Tuning

**Student:** Daniel Alejandro Castillo Martin  
**Student ID:** N10019284  
**Course:** Advanced Deep Learning (AIGC-5500-0NA)  
**Professor:** Hossein Pourmodheji

---

Submission for Lab 03 of the Advanced Deep Learning course in the Artificial Intelligence with Machine Learning postgraduate program at Humber Polytechnic.

This project compares feature extraction and fine-tuning of an ImageNet-pretrained ResNet-18 on a balanced CIFAR-10 subset.

## Setup

Create a virtual environment using the same Python version used for the experiment (Python 3.14.6).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate it with:

```bash
source .venv/bin/activate
```

## Run

Open `daniel_alejandro_castillo_martin_LAB3.ipynb` in VS Code or another Jupyter-compatible editor.

Select the `.venv` Python kernel, restart the kernel, and run all cells from top to bottom.

The notebook downloads CIFAR-10 into `data/` and pretrained weights into `torch_cache/`. Internet access is required for the initial downloads. Both folders are excluded from Git.

The notebook performs dataset exploration, transform verification, feature extraction, fine-tuning, and the final results comparison.

## Dataset

CIFAR-10 contains 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

A seeded selection from the official training partition provides:

- Training: 1,000 images, with 100 per class.
- Validation: 300 images, with 30 per class.
- No overlapping indices between subsets.

The official test partition is not used.

## Transformation Pipelines

Training uses:

- RandomResizedCrop(224).
- RandomHorizontalFlip().
- ColorJitter with brightness, contrast, and saturation set to 0.2.
- ToTensor().
- ImageNet normalization.

Validation uses:

- Resize(256).
- CenterCrop(224).
- ToTensor().
- The same ImageNet normalization.

Normalization uses mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

The notebook displays five passes of the same image through each pipeline. Training views vary, while evaluation views remain identical.

## Training Runs

Both runs start independently from identical pretrained weights and the same newly initialized classification head.

Shared settings:

- Random seed: 42, reset before each experiment.
- Epochs: 5.
- Batch size: 16.
- Optimizer: AdamW with default weight decay of 0.01.
- Loss: CrossEntropyLoss.
- Identical dataset split and augmentation settings.
- Fresh seeded data loaders for each run.

### Feature Extraction

All pretrained parameters are frozen. Only the new classification head trains, using a learning rate of 0.001.

The backbone remains in evaluation mode while the head is in training mode. Freezing parameters alone does not freeze BatchNorm running statistics: leaving the backbone in training mode would update its running means and variances using the target dataset, changing the features despite frozen weights.

### Fine-Tuning

The final residual stage, `layer4`, and the classification head train. Earlier layers remain frozen and in evaluation mode.

One optimizer contains two parameter groups:

- Classification head learning rate: 0.001.
- Layer4 learning rate: 0.0001.

BatchNorm statistics in layer4 update during training. Validation places the entire model in evaluation mode and disables gradient tracking.

## Results

Recorded CPU results before the final reproducibility check:

| Approach | Final validation accuracy | Training time | Trainable parameters |
|---|---:|---:|---:|
| Feature extraction | 76.00% | 254.0 seconds | 5,130 |
| Fine-tuning | 79.33% | 487.1 seconds | 8,398,858 |

Accuracy is measured after epoch five for both runs. Training time includes training passes and data loading, excluding validation and initial downloads.

## Conclusion

Fine-tuning achieved 79.33% validation accuracy versus 76.00% for feature extraction. Updating layer4 allowed the feature representation and its BatchNorm statistics to adapt to CIFAR-10, while feature extraction only trained the classification head. The improvement required approximately 1.92 times the training time.

## Known Limitations

- One seed, a small subset, and five epochs limit the scope of the comparison.
- Enlarging 32 × 32 images to 224 × 224 does not add image detail.
- Random crops can remove useful parts of the subject.
- The official test set was not evaluated.
- Augmentation benefits were not measured in a separate experiment.
- Weight updates and BatchNorm adaptation were not evaluated separately.
- Training time depends on hardware and system load; numerical results may vary across environments.

## Reproduction Check

Clone the repository into a different folder, create a fresh virtual environment, install `requirements.txt`, and run the notebook using only these instructions.

Verify that both experiments complete, the dataset checks pass, and the final metrics agree with the reported results. Training times are expected to vary.