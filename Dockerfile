# 1. Use a modern, stable Java 11 image
FROM eclipse-temurin:11-jre-focal

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install Spark 3.5.0
RUN apt-get update && apt-get install -y curl && \
    curl -O https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz && \
    tar -xzf spark-3.5.0-bin-hadoop3.tgz && \
    mv spark-3.5.0-bin-hadoop3 /opt/spark && \
    rm spark-3.5.0-bin-hadoop3.tgz

# 4. Copy your JAR and Model from your EC2 to the container
COPY target/wine-predictor-1.0-SNAPSHOT.jar /app/
COPY best-wine-model /app/best-wine-model
COPY ValidationDataset.csv /app/

# 5. Setup Spark environment
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# 6. The command that runs when the container starts
CMD ["spark-submit", "--master", "local[*]", "--class", "com.wine.quality.WinePredictor", "wine-predictor-1.0-SNAPSHOT.jar"]
