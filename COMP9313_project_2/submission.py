from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.ml import Pipeline, Transformer
from pyspark.sql import DataFrame


def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    # white space expression tokenizer
    word_tokenizer = Tokenizer(inputCol=input_descript_col, outputCol="words")

    # bag of words count
    count_vectors = CountVectorizer(inputCol="words", outputCol=output_feature_col)

    # label indexer
    label_maker = StringIndexer(inputCol=input_category_col, outputCol=output_label_col)

    class Selector(Transformer):
        def __init__(self, outputCols=[output_feature_col, output_label_col]):
            self.outputCols = outputCols

        def _transform(self, df: DataFrame) -> DataFrame:
            return df.select(*self.outputCols)

    selector = Selector(outputCols=[output_feature_col, output_label_col])
    # build the pipeline
    pipeline = Pipeline(stages=[word_tokenizer, count_vectors, label_maker, selector])
    return pipeline

def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    pass

def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    pass
