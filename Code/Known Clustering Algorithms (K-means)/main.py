from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
import time

# Start Spark Session
spark = SparkSession.builder.appName("MNIST").getOrCreate()

# Start time
start_time = time.time()

# Load training data
train_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("s3://assignment3yuentatshingbilly/mnist_train.csv")

# Convert columns of pixels to a single vector
assembler = VectorAssembler(inputCols=train_data.columns[1:], outputCol="features")
train_data = assembler.transform(train_data)

# Create the trainer and set its parameters
trainer = KMeans().setK(10).setSeed(1)

# Train the model
model = trainer.fit(train_data)

# Load test data
test_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("s3://assignment3yuentatshingbilly/mnist_test.csv")
test_data = assembler.transform(test_data)

result = model.transform(test_data)
predictionAndLabels = result.select("prediction", "label")

# Write the predictions to a CSV file
predictionAndLabels.write.format("csv").option("header", "true").save("s3://assignment3yuentatshingbilly/predictions.csv")

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(result)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Calculate running time
end_time = time.time()
running_time = end_time - start_time

# Write running time and test accuracy to a text file
sc = spark.sparkContext
results = sc.parallelize([(running_time, silhouette)])
results.saveAsTextFile("s3://labsparka/results.txt")

# Stop Spark Session
spark.stop()
