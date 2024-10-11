# Advanced Visual Question Answering with BLIP-VQA and Custom Models

This repository contains the code and resources for advancing Visual Question Answering (VQA) through a dual-stage training approach that integrates cutting-edge techniques. The project fine-tunes the BLIP-VQA model with Low-Rank Adaptation (LoRA) layers and develops a custom VQA model leveraging the Masked Autoencoder's (MAE) Vision Transformer (ViT) for image representation and a pre-trained BERT model for text processing. Cross-attention mechanisms are applied to improve model performance in understanding and answering visual questions.

## Project Overview

### Key Contributions:
1. **BLIP-VQA Fine-tuning**: Incorporation of **Low-Rank Adaptation (LoRA)** at the decoder level to improve model adaptability while maintaining computational efficiency.
2. **Custom VQA Model**: Developed using **MAE's Vision Transformer** for image representation and **pre-trained BERT** for text processing.
3. **Cross-Attention Mechanism**: The model is trained first on image captioning tasks and then fine-tuned for VQA, allowing for deeper semantic understanding of visual and textual inputs.
4. **Dual-Stage Training**: Enhances the model’s ability to generate more accurate and contextually relevant answers by combining **LoRA-enhanced fine-tuning** with sophisticated **cross-modal attention strategies**.

## Repository Structure


```VQA-Advanced/
│
├── __pycache__/              # Cache files
├── cross_attention/          # Cross-attention mechanism implementation
├── dataloaders/              # Data loading scripts for VQA datasets
├── models/                   # Model architectures (BLIP-VQA, custom VQA, etc.)
├── question_embed/           # Embeddings for question representation
├── results/                  # Output results and evaluation metrics
├── trainings/                # Training scripts and configurations
├── visual_embed/             # Visual embeddings for image representations
├── .gitignore                # Git ignore file
├── check4_2.py               # Evaluation and testing script
├── pad_decoder.py            # Decoder module with padding and concatenation improvements
├── show_case_cross.py        # Script for showcasing cross-attention results
├── show_case_cross_attention.py  # Demonstrates cross-attention effectiveness
├── test_coco_loader.py       # Script to test COCO dataset loader
└── verify_showcase.py        # Script for verifying model showcase results
```

## Installation

### Prerequisites

Ensure you have Python 3.x installed. The following Python libraries are required:

```
torch
transformers
numpy
pandas
matplotlib
scikit-learn
opencv-python
```

You can install the required packages using the `requirements.txt` file (if provided) or manually install them:

```bash
pip install torch transformers numpy pandas matplotlib scikit-learn opencv-python
```

## Usage

### 1. Data Loading
Prepare and load the datasets using the provided scripts in the `dataloaders` directory. The dataset should include images and associated questions for VQA tasks.

### 2. Fine-Tuning BLIP-VQA
Run the `trainings` script to fine-tune the BLIP-VQA model. LoRA layers are applied at the decoder level to enhance efficiency.

```bash
python trainings/train_blip_vqa.py
```

### 3. Training the Custom VQA Model
To train the custom VQA model with MAE's Vision Transformer and BERT, use the provided training script:

```bash
python trainings/train_custom_vqa.py
```

### 4. Evaluation and Testing
Evaluate the trained models using the `check4_2.py` script:

```bash
python check4_2.py
```

### 5. Cross-Attention Showcase
Use the `show_case_cross_attention.py` script to demonstrate the results of cross-attention between image and text representations.

```bash
python show_case_cross_attention.py
```

## Model Architecture

1. **Image Representation**: Images are processed using the **MAE's Vision Transformer (ViT)** to extract visual embeddings.
2. **Text Representation**: Questions are embedded using a **pre-trained BERT** model.
3. **Cross-Attention Mechanism**: Combines visual and textual embeddings to ensure deep semantic understanding.
4. **LoRA-enhanced BLIP-VQA**: The BLIP-VQA model is fine-tuned with **Low-Rank Adaptation (LoRA)** at the decoder level to optimize performance and computational resources.

## Results

- **Performance**: The model demonstrates significant improvements in VQA performance by integrating LoRA-enhanced fine-tuning and cross-modal attention.
- **Metrics**: Detailed results and performance metrics can be found in the `results` folder.

## Future Work

- **Exploring Larger Datasets**: Implement and test the model on larger VQA datasets.
- **Advanced Cross-Attention**: Experiment with more complex cross-attention mechanisms for further performance improvement.
- **Real-Time VQA**: Investigate the potential for real-time VQA systems based on this model architecture.

## Acknowledgements

We thank the authors of BLIP-VQA, MAE's Vision Transformer, and BERT for their foundational work, which served as the basis for our custom VQA architecture. 
We thank the NYU HPC for their resources and the Dean Undergraduate Research Fund for providing us with the grant. 

## License

This project is licensed under the MIT License
