# Titanic: Machine Learning from Disaster 🚢

This repository contains my solutions and iterative improvements for the classic Kaggle competition: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic). 


## 📁 Repository Structure

```text
├── 01_Titanic_Base.ipynb                   # Exploratory Data Analysis & baseline models
├── 02_Titanic_Feature_Engineering.ipynb    # Advanced feature creation & missing value imputation
├── 03_Titanic_Grid_Search.ipynb            # Model tuning, parameter selection for best performing model
├── data/                                   # (Ignored) Place train.csv and test.csv here
└── README.md
```

## 📓 Notebook Progression

### 1. Version 1:Baseline (`01_Titanic_Base.ipynb`)
- **Focus:** Understanding the data and setting up a reliable Random Forrest classifier.
- **Key Steps:**
  - Basic Exploratory Data Analysis (survival rates by sex, class, age).
  - Baseline model implementations (Decision Trees) to establish a benchmark score.
  - One hot encoding of Features

### 2. Version 2: Feature Engineering (`02_Titanic_Feature_Engineering.ipynb`)
- **Focus:** Extracting more signal from the existing variables to improve algorithmic logic and pattern recognition.
- **Key Steps:**
  - Parsing passenger names to extract `Title` (Mr., Mrs., Miss., etc.).
  - Combining `SibSp` and `Parch` to create `FamilySize` and `IsAlone` features.
  - Filling the missin data using the medians and metrices
  - Training more robust tree-based models Random Forests.

### 3. Version 3: Advanced Modeling & Parameter selection (`03_Titanic_Grid_Search.ipynb`)
- **Focus:** Squeezing out the maximum possible accuracy for the Random Forrest Classifier.
- **Key Steps:**
  - Use grid seaech method to find the optimum model parameters for the random forrest model
  - Hyperparameter tuning.


Download the dataset from [Kaggle](https://www.kaggle.com/c/titanic/data) and place `train.csv` and `test.csv` in the `data/` folder.

## 🏆 Results
* **Notebook 1 (Baseline):** `~0.76` accuracy
* **Notebook 2 (Feature Engineering):** `~0.78` accuracy
* **Notebook 3 (Ensemble):** `~0.78+` accuracy )*
