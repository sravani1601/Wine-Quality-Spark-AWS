package com.wine.quality;

import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class WinePredictor {
    public static void main(String[] args) throws java.io.IOException {
        SparkSession spark = SparkSession.builder()
                .appName("WineQualityPredictionApp")
                .getOrCreate();

        // 1. Paths
        String modelPath = "best-wine-model";
        String testDataPath = "ValidationDataset.csv";

        // 2. Load the Model 
        // Note: Using the concrete RF model loader is more reliable in Java
        System.out.println(">>> Loading the trained model from: " + modelPath);
        RandomForestClassificationModel model = RandomForestClassificationModel.load("best-wine-model");

        // 3. Load New Data
        Dataset<Row> rawData = spark.read().option("header", "true")
                .option("inferSchema", "true").option("sep", ";").csv(testDataPath);

        // 4. Feature Engineering (Must match Trainer exactly)
        String[] featureCols = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
                                "chlorides", "free sulfur dioxide", "total sulfur dioxide", 
                                "density", "pH", "sulphates", "alcohol"};
        
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        Dataset<Row> finalData = assembler.transform(rawData);

        // 5. Run Predictions
        Dataset<Row> predictions = model.transform(finalData);

        // 6. Show Results
        System.out.println("--- PREDICTION RESULTS (First 10 Rows) ---");
        predictions.select("quality", "prediction").show(10);

        // 7. Calculate F1 Score
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality").setPredictionCol("prediction").setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);
        System.out.println("========================================");
        System.out.println("Final Test F1 Score: " + f1);
        System.out.println("========================================");

        spark.stop();
    }
}
