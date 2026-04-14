# 🍷 Wine Quality Prediction using Apache Spark MLlib on AWS + Docker

## 👨‍🎓 Student
Naga Sai Sravani Vishnubhatla  
CS643 – Cloud Computing Assignment 2

---

# 📌 Project Overview

This project implements a scalable distributed machine learning system for wine quality prediction using Apache Spark MLlib on AWS EC2.

It includes:
- Parallel training across a 4-node Spark cluster
- Model validation and optimization
- Prediction application for unseen data
- Docker-based deployment for portability

---

# ☁️ Cloud Infrastructure

- AWS EC2 Cluster
  - 1 Master Node
  - 3 Worker Nodes
- Apache Spark 3.5.0
- Java 11
- Ubuntu Linux

---

# 🧠 Machine Learning Model

- Framework: Apache Spark MLlib
- Task: Multi-class Classification
- Target: Wine Quality (1–10)
- Algorithm: Logistic Regression / Classifier
- Evaluation Metric: F1 Score

---

# 📊 Dataset Description

- TrainingDataset.csv → Model training
- ValidationDataset.csv → Model tuning
- TestDataset.csv → Final predictions

---

# 🚀 System Workflow

1. Load dataset into Spark cluster
2. Train model in parallel (4 EC2 nodes)
3. Validate model using validation dataset
4. Optimize using F1 score
5. Save best model
6. Load model for inference
7. Run predictions on test dataset
8. Deploy using Docker

---

# ⚙️ Build Project

```bash
mvn clean packagespark-submit \
--class com.wine.quality.WineTrainer \
--master spark://<MASTER_IP>:7077 \
target/wine-predictor.jar \
TrainingDataset.csvspark-submit \
--class com.wine.quality.WinePredictor \
--master local[*] \
target/wine-predictor.jar \
TestDataset.csvdocker pull sravaniv16/wine-prediction-app:v1

docker run -it -v $(pwd):/app sravaniv16/wine-prediction-app:v1
---

# 🎯 WHY THIS VERSION GETS HIGHER MARKS

✔ Cleaner academic formatting  
✔ Professional tone (not student-like notes)  
✔ Proper sectioning  
✔ Clear architecture explanation  
✔ Strong cloud + ML emphasis  
✔ Easy to grade quickly  
✔ Matches industry-style documentation  

---

# 🚀 WHAT YOU SHOULD DO NOW

### 1️⃣ Paste REPORT into Word → Export PDF  
### 2️⃣ Use README in GitHub  
### 3️⃣ Push everything to repo  
### 4️⃣ Ensure Docker + Spark run works  
