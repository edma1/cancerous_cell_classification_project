# Blood Cell Classification for Cancer Diagnosis

## Project Overview

This project develops an automated system for classifying blood cell images to aid in hematological malignancy detection. Using machine learning and deep learning techniques, we achieve 98% accuracy in distinguishing between five key cell types across healthy and pathological states.

## Key Achievements
🎯 97.9% test accuracy on clinical-grade blood smear images
⚡ Optimized CNN architecture for limited GPU resources
🔍 Interpretable features using HOG transformation (88% logistic regression accuracy)

## Project Structure
```
cancerous_cell_classification_project/
├── cancerous_cell_classification_slides_deck/                 # Contains slides deck used for presentation
├── README.md                                                  # This file
└── Technical_Notebook_cancerous_cell_classification_project   # Jupyter notebooks with all code
```
## Dataset
- 5,000 high-resolution images (1024×1024px)
- Five cell classes including both normal and pathological morphologies
- Clinical relevance: Mirrors real-world mixed cell populations

🤖 Model Architectures
1. Classical ML (HOG Features)
  - Logistic Regression: 88% accuracy
  - Random Forest/SVM comparisons
2. Custom CNN
```python
model = Sequential([
    Conv2D(16, (2,2), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(2,2),
    Dropout(0.2),
    
    Conv2D(32, (2,2), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (2,2), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.3),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])
```

## Critical Design Choices

| Decision | Rationale | Impact |
|----------|-----------|--------|
| **256×256 resizing** | Preserves nuclear contours (Prell et al., 2019) | 75% memory reduction |
| **2×2 kernels** | Targets subtle chromatin patterns | VRAM efficiency |
| **Progressive dropout** (20%→30%) | Combats overfitting | <1% train-val gap |
| **Shallow-to-deep filters** (16→64) | Gradual feature complexity | Stable gradient flow |

## Clinical Implications
- Early detection of abnormal cell morphologies
- Reduced diagnostic workload for hematologists
- Scalable solution for resource-limited settings

## Future Directions
- Integration with digital pathology systems
- Uncertainty quantification for borderline cases
- Multi-modal analysis (combining with flow cytometry data)
