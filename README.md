# 🍷 Wine Quality Prediction using Apache Spark MLlib on AWS + Docker

## 👨‍🎓 Student
Naga Sai Sravani Vishnubhatla  
CS643 – Cloud Computing Assignment 2

---

# 📌 Project Overview

This project implements an end-to-end **distributed machine learning pipeline** for predicting wine quality using **Apache Spark MLlib** on AWS EC2.

The system includes:

- Parallel model training on a 4-node Spark cluster (AWS EC2)
- Model validation and hyperparameter tuning
- Wine quality prediction application
- Docker containerization for deployment

---

# ☁️ Cloud Architecture

- **AWS EC2 Cluster**
  - 1 Master Node
  - 3 Worker Nodes
- **Apache Spark 3.5.0**
- **Java 11**
- **Ubuntu Linux**

The training process is distributed across multiple nodes using Spark’s parallel processing capabilities.

---

# 🧠 Machine Learning Model

- Framework: Apache Spark MLlib
- Type: Multi-class Classification (Wine quality 1–10)
- Algorithm: Logistic Regression / MLlib Classifier
- Evaluation Metric: F1 Score

---

# 📊 Datasets

- TrainingDataset.csv → Used for model training
- ValidationDataset.csv → Used for model tuning and validation
- TestDataset.csv → Used for final prediction testing

---

# 🚀 Project Workflow

1. Load training dataset into Spark cluster
2. Train ML model in parallel across 4 EC2 instances
3. Validate model using validation dataset
4. Select best performing model based on F1 score
5. Save trained model
6. Load model in prediction application
7. Run inference on test dataset
8. Deploy prediction system using Docker

---

# 🏗️ Project Structure
wine-predictor/
│── src/
│ └── main/java/com/wine/quality/
│ ├── WineTrainer.java
│ └── WinePredictor.java
│
│── pom.xml
│── Dockerfile
│── best-wine-model/
│── ValidationDataset.csv
│── README.md
---

# ⚙️ How to Build Project (Maven)

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
Docker Overview

The Docker container includes:

Java 11 runtime
Apache Spark
Pre-trained ML model
Prediction application

This allows portable deployment across environments.

📊 Results
Model trained using distributed Spark cluster (4 EC2 instances)
Evaluation metric: F1 Score
Optimized model selected using validation dataset
🔧 Technologies Used
Apache Spark MLlib
AWS EC2 (Cloud Computing)
Java 11
Maven
Docker
Ubuntu Linux

🏁 Conclusion

This project demonstrates a complete cloud-based machine learning pipeline, including:

Distributed training using Spark on AWS
Model evaluation and optimization
Deployment using Docker containers

It showcases scalable ML system design using cloud computing principles.
