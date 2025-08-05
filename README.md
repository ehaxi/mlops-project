# Heart Disease Prediction ML Pipeline

🔹 **Project Overview**  
An end-to-end machine learning pipeline for predicting heart disease risk factors, featuring automated data processing, model training with hyperparameter optimization, and MLflow experiment tracking. The system processes clinical parameters to generate predictive models with explainable outputs.

## 🧠 Core Capabilities

- **Automated Data Processing**: Handles missing values, feature encoding, and normalization
- **Advanced Modeling**: Implements CatBoost classifier with multi-objective optimization
- **Visual Diagnostics**: Generates interactive performance visualizations
- **Experiment Tracking**: Full MLflow integration for parameter and metric logging

## 🏆 Acquired/Improved Skills

- **ML Pipeline Development**: Built end-to-end automated training system
- **Hyperparameter Optimization**: Mastered multi-objective tuning techniques
- **MLOps Practices**: Implemented production-grade experiment tracking
- **Clinical Data Processing**: Developed domain-specific feature engineering
- **Visual Diagnostics**: Created interactive model evaluation tools
- **Performance Optimization**: Efficient memory management for large-scale tuning

## 🛠️ Technical Stack

### Data Processing
- **Feature Engineering**: Automated label encoding, normalization (MinMaxScaler), standardization (StandardScaler)
- **Visualization**: Seaborn/Matplotlib for EDA, Plotly for interactive model diagnostics
- **Data Validation**: Comprehensive checks for duplicates, missing values, and data types

### Machine Learning
- **Algorithm**: Optimized CatBoost classifier
- **Hyperparameter Tuning**: NSGA-II multi-objective optimization (Recall, F1, PR-AUC)
- **Validation**: Stratified k-fold cross-validation
- **Metrics**: Precision-Recall curves, AUC-ROC, feature importance

### Infrastructure
- **Experiment Tracking**: MLflow with SQLite backend
- **Logging**: Structured YAML-configured logging
- **Model Serialization**: Joblib for production-ready artifacts

## 📊 Key Components

| Module | Purpose |
|--------|---------|
| **Data Processing** | Automated pipelines for feature engineering and validation |
| **Model Training** | Multi-objective hyperparameter optimization |
| **Visualization** | Interactive Pareto fronts and precision-recall curves |
| **MLOps** | Experiment tracking and model versioning |

## 🔍 Technical Highlights

- **Multi-Objective Optimization**: Simultaneously optimizes recall, F1 score and PR-AUC using genetic algorithms
- **Production-Grade Logging**: Comprehensive experiment tracking with execution metrics
- **Explainable Outputs**: Visual diagnostics for model performance interpretation
- **Modular Architecture**: Clean separation of data, training and evaluation components

## 📈 Model Performance

Optimizes three key metrics with configurable weights:
- **Recall**: Maximize true positive rate (default weight: 0.7)
- **F1 Score**: Balance precision/recall (default weight: 0.2)
- **PR-AUC**: Area under precision-recall curve (default weight: 0.1)

## 🔮 Future Enhancements

1. Containerization of the application via Docker
2. A graphical shell (site/tg-bot) that allows you to use ready-made models 
3. CI/CD Automation
