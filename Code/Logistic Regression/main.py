from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_csv_path", help="S3 path for train csv")
parser.add_argument("test_csv_path", help="S3 path for test csv")
parser.add_argument("predictions_path", help="S3 path for saving predictions")
parser.add_argument("results_path", help="S3 path for saving results")
args = parser.parse_args()

spark = SparkSession.builder.appName("MNIST").getOrCreate()

start_time = time.time()

train_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(args.train_csv_path)

assembler = VectorAssembler(inputCols=train_data.columns[1:], outputCol="features")
train_data = assembler.transform(train_data)

trainer = LogisticRegression(maxIter=100, regParam=0.01)

model = trainer.fit(train_data)

test_data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(args.test_csv_path)
test_data = assembler.transform(test_data)

result = model.transform(test_data)
predictionAndLabels = result.select("prediction", "label")

predictionAndLabels.write.format("csv").option("header", "true").save(args.predictions_path)

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(result)
print("Test set accuracy = " + str(accuracy))

end_time = time.time()
running_time = end_time - start_time

train_result = model.transform(train_data)
train_accuracy = evaluator.evaluate(train_result)

sc = spark.sparkContext

train_results = sc.parallelize([(running_time, train_accuracy, accuracy)])
train_results.saveAsTextFile(args.results_path)

spark.stop()
