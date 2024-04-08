from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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

# Specify layers for the neural network:
# input layer of size 784 (features), two intermediate layers
# and output of size 10 (classes)
layers = [784, 128, 64, 10]

# Create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

# Train the model
model = trainer.fit(train_data)

# Load test data
test_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("s3://assignment3yuentatshingbilly/mnist_test.csv")
test_data = assembler.transform(test_data)

result = model.transform(test_data)
predictionAndLabels = result.select("prediction", "label")

# Write the predictions to a CSV file
predictionAndLabels.write.format("csv").option("header", "true").save("s3://assignment3yuentatshingbilly/predictions.csv")

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
train_results.saveAsTextFile("s3://assignment3yuentatshingbilly/results.txt")