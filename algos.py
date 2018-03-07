import logging
import luigi

import coloredlogs
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from log_config import log_format, log_date_format, loglevel, logfile, add_handlers_and_init

logger = logging.getLogger('ALGO')
add_handlers_and_init(logger, log_format, log_date_format, loglevel, logfile)
coloredlogs.install(loglevel)

from config import get_place_of_predictor
from prepare_data import TrainHoldout


class PredictorNB(luigi.Task):
    predictor_name = 'naivestd'

    def requires(self):
        return TrainHoldout()

    def output(self):
        return luigi.LocalTarget(get_place_of_predictor(self.predictor_name), format=luigi.format.Nop)

    def run(self):
        logger.info('Reading train file: "%s"', self.input()['train'].path)
        df = pd.read_csv(self.input()['train'].path)
        logger.info('Its %s lines', df.shape[0])

        logger.info('Preraring pipeline...')
        vectorizer = CountVectorizer()
        cls = MultinomialNB()

        pipeline = Pipeline([('vectorizer', vectorizer), ('predictor', cls)])
        logger.info('Preraring targets...')

        logger.info('Fitting pipeline[%s]...', pipeline)
        pipeline.fit(df.name, df.tare)

        logger.info('Writing it to %s', self.output().path)
        with self.output().open('wb') as outcome:
            pickle.dump(pipeline, outcome)

    def apply(self, df_preprocessed):
        logger.info('Reading pipeline: %s', self.output().path)
        with self.output().open('rb') as outcome:
            pipeline = pickle.load(outcome)

        return pipeline.predict(df_preprocessed.name)
