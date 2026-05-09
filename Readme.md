# 🏠 House Price Prediction in C++ (From Scratch)

A **complete end-to-end Machine Learning project implemented purely in C++**, without using any external ML libraries. This project demonstrates how core ML concepts work under the hood, including data preprocessing, normalization, linear regression, gradient descent, evaluation, model persistence, and interactive inference.

---

## 🚀 Project Highlights

- 📊 CSV data ingestion (no pandas, no NumPy)
- 🔄 Feature & target normalization (Z-score)
- 📈 Linear Regression implemented from scratch
- 🔁 Batch Gradient Descent optimization
- 🧪 Train/Test split (80/20) with no data leakage
- 📏 Evaluation metrics: **RMSE, MAE, R² Score**
- 💾 Model persistence (Save & Load weights)
- 🧑‍💻 Interactive CLI-based prediction
- 🛡️ Input validation for realistic predictions
- ⚙️ Clean OOP design & encapsulation

> **No Python. No ML libraries. Only C++.**

---

## 🧠 Machine Learning Workflow

1. Load house price dataset from CSV
2. Normalize features and target values (Z-score)
3. Split data into 80% train / 20% test
4. Train Linear Regression model on train set using Gradient Descent
5. Evaluate model on test set using RMSE, MAE, and R²
6. Save trained model to disk
7. Load model for inference
8. Accept user input and predict house price

---

## 📂 Project Structure

```
HousePriceML-Cpp/
│── data/
│   └── house_prices.csv
│
│── include/
│   ├── dataset.h
│   ├── linear_regression.h
│   └── metrics.h
│
│── src/
│   ├── main.cpp
│   ├── dataset.cpp
│   ├── linear_regression.cpp
│   └── metrics.cpp
│
│── model.txt
│── README.md
```

---

## 🧮 Model Details

### Linear Regression Formula

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$

### Cost Function (Mean Squared Error)

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

### Optimization
- Batch Gradient Descent
- Learning Rate: `0.01`
- Epochs: `5000`

---

## 📥 Input Features

| Feature | Description |
|---|---|
| Area | House area in square feet |
| Bedrooms | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Location Score | Rating from 1 to 10 |
| Age | Age of the house in years |

---

## 🧪 Example Usage

### Compile (from project root)
```bash
g++ src/main.cpp src/dataset.cpp src/linear_regression.cpp src/metrics.cpp -Iinclude -o house_ml
```

### Run
```bash
./house_ml
```

### Sample Interaction
```
===== Model Evaluation (Test Set) =====
RMSE   : 412500.00 INR
MAE    : 310000.00 INR
R²     : 0.87
========================================

Enter house details:
Area (sqft, 200-10000): 2000
Bedrooms (1-10): 4
Bathrooms (1-10): 2
Location score (1-10): 8
Age of house (0-100 years): 15

Predicted House Price (INR): 1.28e+07
```

---

## 📊 Evaluation Metrics

The model is evaluated on a held-out **test set (20%)** that was never seen during training. Metrics are computed on **denormalized (real INR)** values for meaningful interpretation.

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error — penalizes large errors more |
| **MAE** | Mean Absolute Error — average prediction error in INR |
| **R² Score** | How well the model explains variance (1.0 = perfect) |

> Training normalization statistics are reused for the test set and user input to prevent data leakage.

---

## 🛡️ Input Validation

All user inputs are clamped to realistic ranges before inference:

| Feature | Valid Range |
|---|---|
| Area | 200 – 10,000 sqft |
| Bedrooms | 1 – 10 |
| Bathrooms | 1 – 10 |
| Location Score | 1 – 10 |
| Age | 0 – 100 years |

---

## 💡 Why C++ for Machine Learning?

- Deep understanding of ML internals
- Memory management & performance awareness
- Strong foundation before using high-level frameworks
- Interview-ready demonstration of fundamentals

---

## 🧠 What This Project Demonstrates

- Strong grasp of Machine Learning fundamentals
- Ability to implement ML algorithms without libraries
- Proper train/test evaluation pipeline (no data leakage)
- Clean C++ OOP design
- Real-world engineering practices

---

## 📌 Future Improvements

- [ ] Ridge Regression (L2 Regularization)
- [ ] Polynomial Regression
- [ ] k-Fold Cross Validation
- [ ] Menu-based CLI
- [ ] Export predictions to CSV for visualization
- [ ] GUI using Qt
- [ ] Performance comparison with Python/sklearn

---

## 👤 Author

**Anish Sharma**  
B.Tech Computer Science & Engineering

---

## ⭐ Final Note

This project is built to **show understanding, not just results**. It reflects how machine learning works internally and demonstrates strong problem-solving and system design skills.

Feel free to fork, improve, or use this project for learning and interviews.
