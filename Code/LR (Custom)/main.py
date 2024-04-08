from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import time
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("train_csv_path", help="S3 path for train csv")
parser.add_argument("test_csv_path", help="S3 path for test csv")
parser.add_argument("predictions_path", help="S3 path for saving predictions")
parser.add_argument("results_path", help="S3 path for saving results")
parser.add_argument("ENP_val", help="elasticNetParam Value")
args = parser.parse_args()
# Start Spark Session
spark = SparkSession.builder.appName("MNIST").getOrCreate()

# Start time
start_time = time.time()

# Load training data
train_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(args.train_csv_path)

# Convert columns of pixels to a single vector
assembler = VectorAssembler(inputCols=train_data.columns[1:], outputCol="features")
train_data = assembler.transform(train_data)

# Create the trainer and set its parameters
trainer = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=args.ENP_val)

# trainer = LogisticRegression(maxIter=50, regParam=0.05, elasticNetParam=0.5)



# Train the model
model = trainer.fit(train_data)

# Load test data
test_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(args.test_csv_path)
test_data = assembler.transform(test_data)

result = model.transform(test_data)
predictionAndLabels = result.select("prediction", "label")

# Write the predictions to a CSV file
predictionAndLabels.write.format("csv").option("header", "true").save(args.predictions_path)

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(result)
print("Test set accuracy = " + str(accuracy))

# Calculate running time
end_time = time.time()
running_time = end_time - start_time

# Calculate training accuracy
train_result = model.transform(train_data)
train_accuracy = evaluator.evaluate(train_result)





# Write running time and test accuracy to a text file
sc = spark.sparkContext
# Write training accuracy to the same text file
train_results = sc.parallelize([(running_time, train_accuracy, accuracy)])
train_results.saveAsTextFile(args.results_path)

# Stop Spark Session
spark.stop()
