# Heart Disease Prediction Using Machine Learning

**Author:** Armand Kayiranga  
**Affiliation:** African Leadership University  
**Course:** Introduction to Machine Learning - Final Summative
**Link to Video Presentation :** https://www.loom.com/share/b6865215a0a5492ab42d484b0a300453?t=80&sid=15628145-157c-4146-9340-7a544b4bdab3

## Problem Statement

Heart disease is a leading cause of death globally, and early diagnosis is critical to prevent life-threatening outcomes. This project leverages machine learning classification algorithms to predict the presence of heart disease using the Cleveland Heart Disease dataset. By implementing and comparing classical ML algorithms with neural networks using various optimization techniques, we aim to support clinicians in early screening and decision-making processes.

## Dataset

**Cleveland Heart Disease Dataset** from UCI Machine Learning Repository
- **Total Samples:** 303 patients
- **Features:** 13 clinical attributes (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- **Target:** Binary classification (0: No heart disease, 1: Heart disease present)
- **Source:** https://archive.ics.uci.edu/ml/datasets/Heart+Disease

## Models Implemented

### 1. Classical ML Algorithms (Optimized)
- **Logistic Regression** with hyperparameter tuning (C, penalty, solver)
- **XGBoost** with hyperparameter tuning (n_estimators, max_depth, learning_rate)

### 2. Neural Network Models
- **Simple Neural Network** (no optimization - baseline)
- **5 Optimized Neural Network Instances** with different combinations of:
  - Optimizers: Adam, RMSprop, SGD
  - Regularizers: L1, L2, L1+L2
  - Early stopping, dropout, learning rate variations
  - Different architectures (3-5 layers)

## Optimization Results Table

| Training Instance | Optimizer | Regularizer | Epochs | Early Stopping | Number of Layers | Learning Rate | Dropout Rate | Accuracy | Loss | F1-score | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Instance 1 (Default) | adam | None | 50 | No | 3 | 0.001 | 0.0 | 0.8525 | 0.3421 | 0.8571 | 0.8571 | 0.8571 |
| Instance 2 (Adam + L2 + Early Stopping) | adam | L2(0.01) | 100 | Yes | 4 | 0.001 | 0.3 | 0.8689 | 0.3156 | 0.8750 | 0.8750 | 0.8750 |
| Instance 3 (RMSprop + L1 + Dropout) | rmsprop | L1(0.01) | 80 | Yes | 3 | 0.01 | 0.5 | 0.8361 | 0.3789 | 0.8421 | 0.8421 | 0.8421 |
| Instance 4 (SGD + L1L2 + High LR) | sgd | L1L2(0.01,0.01) | 150 | Yes | 5 | 0.1 | 0.2 | 0.8197 | 0.4123 | 0.8261 | 0.8261 | 0.8261 |
| Instance 5 (Adam + L2 + Low LR) | adam | L2(0.001) | 200 | Yes | 4 | 0.0001 | 0.4 | 0.8525 | 0.3287 | 0.8571 | 0.8571 | 0.8571 |

## Discussion of Findings

### Which Combination Worked Better

**Best Neural Network Configuration:** Instance 2 (Adam + L2 + Early Stopping)
- **Optimizer:** Adam with learning rate 0.001
- **Regularization:** L2 regularization (0.01)
- **Architecture:** 4 layers with 30% dropout
- **Training:** Early stopping enabled
- **Performance:** Accuracy: 86.89%, F1-score: 87.50%

The combination of Adam optimizer with L2 regularization and early stopping provided the best balance between performance and generalization. The moderate dropout rate (30%) helped prevent overfitting while maintaining good predictive capability.

### Implementation Comparison: ML Algorithm vs Neural Network

**Winner: Classical ML Algorithms**

**XGBoost Performance:**
- **Accuracy:** 88.52%
- **F1-score:** 89.29%
- **Key Hyperparameters:**
  - n_estimators: 200
  - max_depth: 5
  - learning_rate: 0.1
  - subsample: 0.9
  - colsample_bytree: 0.9

**Logistic Regression Performance:**
- **Accuracy:** 83.61%
- **F1-score:** 84.21%
- **Key Hyperparameters:**
  - C: 1.0
  - penalty: 'l2'
  - solver: 'liblinear'

**Analysis:**
1. **XGBoost outperformed all neural network models**, achieving the highest accuracy (88.52%) and F1-score (89.29%)
2. **Classical ML algorithms were more efficient** in terms of training time and computational resources
3. **Neural networks showed improvement with optimization** but couldn't surpass the ensemble method
4. **The dataset size (303 samples) favored classical ML approaches** over deep learning methods
5. **Feature engineering and hyperparameter tuning were crucial** for classical ML success

## Key Insights

1. **Small Dataset Advantage:** Classical ML algorithms performed better on this relatively small dataset
2. **Optimization Impact:** Neural network optimization techniques improved performance by ~3-4%
3. **Regularization Effectiveness:** L2 regularization consistently improved neural network performance
4. **Early Stopping Benefits:** Prevented overfitting and improved generalization
5. **Ensemble Power:** XGBoost's ensemble approach proved most effective for this medical prediction task

## Instructions for Running

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow joblib
```

### Running the Notebook
1. Open `notebook.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially
3. Models will be automatically saved to `saved_models/` directory
4. Results will be displayed and exported to CSV

### Loading the Best Model
```python
import joblib
best_model = joblib.load('saved_models/xgboost_optimized.pkl')
```

## Repository Structure
```
Heart_Disease_Prediction/
├── notebook.ipynb                          # Main project notebook
├── saved_models/                           # Trained models directory
│   ├── logistic_regression_optimized.pkl  # Optimized Logistic Regression
│   ├── xgboost_optimized.pkl              # Optimized XGBoost (Best Model)
│   ├── simple_neural_network.h5           # Simple Neural Network
│   ├── neural_network_instance_1.h5       # NN Instance 1 (Default)
│   ├── neural_network_instance_2.h5       # NN Instance 2 (Best NN)
│   ├── neural_network_instance_3.h5       # NN Instance 3
│   ├── neural_network_instance_4.h5       # NN Instance 4
│   ├── neural_network_instance_5.h5       # NN Instance 5
│   └── neural_network_results.csv         # Detailed results table
└── README.md                               # This file
```

## Conclusion

This project successfully demonstrated the application of machine learning for heart disease prediction. While neural networks with optimization showed promising results, classical ML algorithms (particularly XGBoost) proved more effective for this specific dataset and problem domain. The comprehensive comparison provides valuable insights for healthcare ML applications and highlights the importance of choosing appropriate algorithms based on dataset characteristics.

## Future Work

1. Implement ensemble methods combining classical ML and neural networks
2. Explore feature engineering techniques
3. Apply the model to larger healthcare datasets
4. Investigate interpretability methods (SHAP, LIME)
5. Deploy the model as a web application for clinical use
