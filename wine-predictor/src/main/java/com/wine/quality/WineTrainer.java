package com.wine.quality;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineTrainer {
    // Note the 'throws java.io.IOException' added below:
    public static void main(String[] args) throws java.io.IOException {
        // 1. Setup Spark
        SparkSession spark = SparkSession.builder()
                .appName("WineModelTournament")
                .getOrCreate();

        // 2. Load Datasets
        String trainingPath = "/home/ubuntu/wine-predictor/TrainingDataset.csv";
        String validationPath = "/home/ubuntu/wine-predictor/ValidationDataset.csv";

        Dataset<Row> trainingRaw = spark.read().option("header", "true")
                .option("inferSchema", "true").option("sep", ";").csv(trainingPath);
        Dataset<Row> validationRaw = spark.read().option("header", "true")
                .option("inferSchema", "true").option("sep", ";").csv(validationPath);

        // 3. Feature Engineering
        String[] featureCols = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
                                "chlorides", "free sulfur dioxide", "total sulfur dioxide", 
                                "density", "pH", "sulphates", "alcohol"};
        
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        Dataset<Row> trainData = assembler.transform(trainingRaw).select("features", "quality");
        Dataset<Row> validData = assembler.transform(validationRaw).select("features", "quality");

        // 4. Setup the Evaluator
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality").setPredictionCol("prediction").setMetricName("f1");

        // --- MODEL 1: Logistic Regression ---
        System.out.println(">>> Training Model 1: Logistic Regression...");
        LogisticRegressionModel lrModel = new LogisticRegression()
                .setLabelCol("quality").setMaxIter(15).fit(trainData);
        double lrF1 = evaluator.evaluate(lrModel.transform(validData));

        // --- MODEL 2: Random Forest ---
        System.out.println(">>> Training Model 2: Random Forest...");
        RandomForestClassificationModel rfModel = new RandomForestClassifier()
                .setLabelCol("quality").setNumTrees(25).setMaxDepth(10).setSeed(42).fit(trainData);
        double rfF1 = evaluator.evaluate(rfModel.transform(validData));

        // 5. Comparison Report
        System.out.println("\n========================================");
        System.out.println("      WINE MODEL TOURNAMENT RESULTS     ");
        System.out.println("========================================");
        System.out.println("Logistic Regression F1: " + String.format("%.4f", lrF1));
        System.out.println("Random Forest F1:       " + String.format("%.4f", rfF1));
        System.out.println("========================================");

        // 6. Save the Winner
        if (rfF1 >= lrF1) {
            System.out.println("Winner: Random Forest! Saving to 'best-wine-model'...");
            rfModel.write().overwrite().save("/home/ubuntu/wine-predictor/best-wine-model");
        } else {
            System.out.println("Winner: Logistic Regression! Saving to 'best-wine-model'...");
            lrModel.write().overwrite().save("/home/ubuntu/wine-predictor/best-wine-model");
        }

        spark.stop();
    }
}
