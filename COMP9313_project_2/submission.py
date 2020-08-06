from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.ml import Pipeline, Transformer
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, IntegerType


def joint(a, b):
    if a == 0 and b == 0:
        return 0
    elif a == 0 and b == 1:
        return 1
    elif a == 1 and b == 0:
        return 2
    elif a == 1 and b == 1:
        return 3


def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    # white space expression tokenizer
    word_tokenizer = Tokenizer(inputCol=input_descript_col, outputCol="words")

    # bag of words count
    count_vectors = CountVectorizer(inputCol="words", outputCol=output_feature_col)

    # label indexer
    label_maker = StringIndexer(inputCol=input_category_col, outputCol=output_label_col)

    class Selector(Transformer):
        def __init__(self, outputCols):
            self.outputCols = outputCols

        def _transform(self, df: DataFrame) -> DataFrame:
            return df.select(*self.outputCols)

    selector = Selector(outputCols=['id', output_feature_col, output_label_col])
    # build the pipeline
    pipeline = Pipeline(stages=[word_tokenizer, count_vectors, label_maker, selector])
    return pipeline

def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):

    joint_udf = udf(joint, IntegerType())
    nb_result = {}
    svm_result = {}
    for i in range(5):
        condition = training_df['group'] == i
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()
        # nb

        # 0
        nb_result = save_answer(c_test, c_train, nb_0, nb_result, 'nb_pred_0', 0)
        nb_result = save_answer(c_test, c_train, nb_1, nb_result, 'nb_pred_1', 1)
        nb_result = save_answer(c_test, c_train, nb_2, nb_result, 'nb_pred_2', 2)

        svm_result = save_answer(c_test, c_train, svm_0, svm_result, 'svm_pred_0', 0)
        svm_result = save_answer(c_test, c_train, svm_1, svm_result, 'svm_pred_1', 1)
        svm_result = save_answer(c_test, c_train, svm_2, svm_result, 'svm_pred_2', 2)

        # # 1
        # lr_model_1 = nb_1.fit(c_train)
        # lr_pred_1 = lr_model_1.transform(c_test)
        # if 1 not in nb_result:
        #     nb_result[1] = lr_pred_1.select(['id', 'nb_pred_1'])
        # else:
        #     nb_result[1].union(lr_pred_1.select(['id', 'nb_pred_1']))
        #
        # # 2
        # lr_model_2 = nb_2.fit(c_train)
        # lr_pred_2 = lr_model_2.transform(c_test)
        # if 2 not in nb_result:
        #     nb_result[2] = lr_pred_2.select(['id', 'nb_pred_2'])
        # else:
        #     nb_result[2].union(lr_pred_2.select(['id', 'nb_pred_2']))
        #
        # # svm
        #
        # # 0
        # svm_model_0 = svm_0.fit(c_train)
        # svm_pred_0 = svm_model_0.transform(c_test)
        # if 0 not in svm_result:
        #     svm_result[0] = svm_pred_0.select(['id', 'svm_pred_0'])
        # else:
        #     svm_result[0].union(svm_pred_0.select(['id', 'svm_pred_0']))
        #
        # # 1
        # svm_model_1 = svm_1.fit(c_train)
        # svm_pred_1 = svm_model_1.transform(c_test)
        # if 1 not in svm_result:
        #     svm_result[1] = svm_pred_1.select(['id', 'svm_pred_1'])
        # else:
        #     svm_result[1].union(svm_pred_1.select(['id', 'svm_pred_1']))
        #
        # # 2
        # svm_model_2 = svm_2.fit(c_train)
        # svm_pred_2 = svm_model_2.transform(c_test)
        # if 2 not in svm_result:
        #     svm_result[2] = svm_pred_2.select(['id', 'svm_pred_2'])
        # else:
        #     svm_result[2].union(svm_pred_2.select(['id', 'svm_pred_2']))

    training_df = training_df.join(nb_result[0], on=['id'])
    training_df = training_df.join(nb_result[1], on=['id'])
    training_df = training_df.join(nb_result[2], on=['id'])

    training_df = training_df.join(svm_result[0], on=['id'])
    training_df = training_df.join(svm_result[1], on=['id'])
    training_df = training_df.join(svm_result[2], on=['id'])

    training_df = training_df.withColumn('joint_pred_0', (joint_udf(training_df['nb_pred_0'], training_df['svm_pred_0']).cast(DoubleType())))
    training_df = training_df.withColumn('joint_pred_1', (joint_udf(training_df['nb_pred_1'], training_df['svm_pred_1']).cast(DoubleType())))
    training_df = training_df.withColumn('joint_pred_2', (joint_udf(training_df['nb_pred_2'], training_df['svm_pred_2']).cast(DoubleType())))
    return training_df


def save_answer(c_test, c_train, model, result_dict, col, label_number):
    fitted_model = model.fit(c_train)
    pred = fitted_model.transform(c_test)
    if label_number not in result_dict:
        result_dict[label_number] = pred.select(['id', col])
    else:
        result_dict[label_number].union(pred.select(['id', col]))
    return result_dict


def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):

    # got features
    df = base_features_pipeline_model.transform(test_df)

    # get base pred
    base = gen_base_pred_pipeline_model.transform(df)

    # get meta features
    joint_udf = udf(joint, IntegerType())
    base = base.withColumn('joint_pred_0', (
        joint_udf(base['nb_pred_0'], base['svm_pred_0']).cast(DoubleType())))

    base = base.withColumn('joint_pred_1', (
        joint_udf(base['nb_pred_1'], base['svm_pred_1']).cast(DoubleType())))

    base = base.withColumn('joint_pred_2', (
        joint_udf(base['nb_pred_2'], base['svm_pred_2']).cast(DoubleType())))

    meta = gen_meta_feature_pipeline_model.transform(base)

    ans = meta_classifier.transform(meta)
    return ans
