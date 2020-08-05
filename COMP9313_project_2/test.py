from pyspark.sql import *
from pyspark import SparkConf

from pyspark.sql import DataFrame
from pyspark.sql.functions import rand
from pyspark.sql.types import IntegerType, DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.classification import LogisticRegression, LinearSVC, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from submission import base_features_gen_pipeline, gen_meta_features, test_prediction

import random
rseed = 1024
random.seed(rseed)


def gen_binary_labels(df):
    df = df.withColumn('label_0', (df['label'] == 0).cast(DoubleType()))
    df = df.withColumn('label_1', (df['label'] == 1).cast(DoubleType()))
    df = df.withColumn('label_2', (df['label'] == 2).cast(DoubleType()))
    return df

# Create a Spark Session
conf = SparkConf().setMaster("local[*]").setAppName("lab3")
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Load data
train_data = spark.read.load("proj2train.csv", format="csv", sep="\t", inferSchema="true", header="true")
test_data = spark.read.load("proj2test.csv", format="csv", sep="\t", inferSchema="true", header="true")

# build the pipeline from task 1.1
base_features_pipeline = base_features_gen_pipeline()
# Fit the pipeline using train_data
base_features_pipeline_model = base_features_pipeline.fit(train_data)
# Transform the train_data using fitted pipeline
training_set = base_features_pipeline_model.transform(train_data)
# assign random groups and binarize the labels
training_set = training_set.withColumn('group', (rand(rseed)*5).cast(IntegerType()))
training_set = gen_binary_labels(training_set)

# define base models
nb_0 = NaiveBayes(featuresCol='features', labelCol='label_0', predictionCol='nb_pred_0', probabilityCol='nb_prob_0', rawPredictionCol='nb_raw_0')
nb_1 = NaiveBayes(featuresCol='features', labelCol='label_1', predictionCol='nb_pred_1', probabilityCol='nb_prob_1', rawPredictionCol='nb_raw_1')
nb_2 = NaiveBayes(featuresCol='features', labelCol='label_2', predictionCol='nb_pred_2', probabilityCol='nb_prob_2', rawPredictionCol='nb_raw_2')
svm_0 = LinearSVC(featuresCol='features', labelCol='label_0', predictionCol='svm_pred_0', rawPredictionCol='svm_raw_0')
svm_1 = LinearSVC(featuresCol='features', labelCol='label_1', predictionCol='svm_pred_1', rawPredictionCol='svm_raw_1')
svm_2 = LinearSVC(featuresCol='features', labelCol='label_2', predictionCol='svm_pred_2', rawPredictionCol='svm_raw_2')

# build pipeline to generate predictions from base classifiers, will be used in task 1.3
gen_base_pred_pipeline = Pipeline(stages=[nb_0, nb_1, nb_2, svm_0, svm_1, svm_2])
gen_base_pred_pipeline_model = gen_base_pred_pipeline.fit(training_set)

# task 1.2
meta_features = gen_meta_features(training_set, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2)

# build onehotencoder and vectorassembler pipeline
onehot_encoder = OneHotEncoderEstimator(inputCols=['nb_pred_0', 'nb_pred_1', 'nb_pred_2', 'svm_pred_0', 'svm_pred_1', 'svm_pred_2', 'joint_pred_0', 'joint_pred_1', 'joint_pred_2'], outputCols=['vec{}'.format(i) for i in range(9)])
vector_assembler = VectorAssembler(inputCols=['vec{}'.format(i) for i in range(9)], outputCol='meta_features')
gen_meta_feature_pipeline = Pipeline(stages=[onehot_encoder, vector_assembler])
gen_meta_feature_pipeline_model = gen_meta_feature_pipeline.fit(meta_features)
meta_features = gen_meta_feature_pipeline_model.transform(meta_features)

# train the meta clasifier
lr_model = LogisticRegression(featuresCol='meta_features', labelCol='label', predictionCol='final_prediction', maxIter=20, regParam=1., elasticNetParam=0)
meta_classifier = lr_model.fit(meta_features)

# task 1.3
pred_test = test_prediction(test_data, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier)

# Evaluation
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",metricName='f1')
print(evaluator.evaluate(pred_test, {evaluator.predictionCol:'final_prediction'}))
spark.stop()