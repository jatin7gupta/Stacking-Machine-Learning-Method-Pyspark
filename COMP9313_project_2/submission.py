from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.ml import Pipeline, Transformer
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType


def joint(a, b):
    if a == 0.0 and b == 0.0:
        return 0.0
    elif a == 0.0 and b == 1.0:
        return 1.0
    elif a == 1.0 and b == 0.0:
        return 2.0
    elif a == 1.0 and b == 1.0:
        return 3.0


def add_joint_predictions(base):
    joint_udf = udf(joint, DoubleType())
    base = base.withColumn('joint_pred_0',
                           joint_udf(base['nb_pred_0'], base['svm_pred_0']))
    base = base.withColumn('joint_pred_1',
                           joint_udf(base['nb_pred_1'], base['svm_pred_1']))
    base = base.withColumn('joint_pred_2',
                           joint_udf(base['nb_pred_2'], base['svm_pred_2']))
    return base


def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category",
                               output_feature_col="features", output_label_col="label"):
    word_tokenizer = Tokenizer(inputCol=input_descript_col, outputCol="words")
    count_vectors = CountVectorizer(inputCol="words", outputCol=output_feature_col)
    label_maker = StringIndexer(inputCol=input_category_col, outputCol=output_label_col)

    class Selector(Transformer):
        def __init__(self, output_cols):
            self.outputCols = output_cols

        def _transform(self, df: DataFrame) -> DataFrame:
            return df.select(*self.outputCols)

    selector = Selector(output_cols=['id', output_feature_col, output_label_col])

    pipeline = Pipeline(stages=[word_tokenizer, count_vectors, label_maker, selector])
    return pipeline


def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    nb_result = {}
    svm_result = {}
    for i in range(5):
        condition = training_df['group'] == i
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()

        nb_result = save_answer(c_test, c_train, nb_0, nb_result, 'nb_pred_0', 0)
        nb_result = save_answer(c_test, c_train, nb_1, nb_result, 'nb_pred_1', 1)
        nb_result = save_answer(c_test, c_train, nb_2, nb_result, 'nb_pred_2', 2)

        svm_result = save_answer(c_test, c_train, svm_0, svm_result, 'svm_pred_0', 0)
        svm_result = save_answer(c_test, c_train, svm_1, svm_result, 'svm_pred_1', 1)
        svm_result = save_answer(c_test, c_train, svm_2, svm_result, 'svm_pred_2', 2)

    training_df = training_df.join(nb_result[0], on=['id'])
    training_df = training_df.join(nb_result[1], on=['id'])
    training_df = training_df.join(nb_result[2], on=['id'])

    training_df = training_df.join(svm_result[0], on=['id'])
    training_df = training_df.join(svm_result[1], on=['id'])
    training_df = training_df.join(svm_result[2], on=['id'])

    return add_joint_predictions(training_df)


def save_answer(c_test, c_train, model, result_dict, col, label_number):
    fitted_model = model.fit(c_train)
    pred = fitted_model.transform(c_test)
    if label_number not in result_dict:
        result_dict[label_number] = pred.select(['id', col])
    else:
        result_dict[label_number] = result_dict[label_number].union(pred.select(['id', col]))
    return result_dict


def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model,
                    gen_meta_feature_pipeline_model, meta_classifier):
    df = base_features_pipeline_model.transform(test_df)
    base = gen_base_pred_pipeline_model.transform(df)
    base_joint_pred = add_joint_predictions(base)
    meta = gen_meta_feature_pipeline_model.transform(base_joint_pred)
    return meta_classifier.transform(meta)
