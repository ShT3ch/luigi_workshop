import logging

import coloredlogs
import luigi
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from algos import PredictorNB
from log_config import log_format, log_date_format, loglevel, logfile, add_handlers_and_init
from prepare_data import TrainHoldout, TextsProcessed, Encoder
from utils import rebuild_path_to

logger = logging.getLogger('APPLICATOR')
add_handlers_and_init(logger, log_format, log_date_format, loglevel, logfile)
coloredlogs.install(loglevel)


class MeasureQuality(luigi.Task):
    def requires(self):
        return {
            'predictor': PredictorNB(),
            'data': TrainHoldout()
        }


    def run(self):
        logger.info('Reading holdout file: "%s"', self.input()['data']['holdout'].path)
        df = pd.read_csv(self.input()['data']['holdout'].path)

        encoder = LabelEncoder()
        targets = encoder.fit_transform(df.tare)

        results = self.requires()['predictor'].apply(df)

        logger.info('\n\n\n\n\t\t>>>>>>>>>>> ACC: %5.3f<<<<<<<<<<<<\n\n\n', accuracy_score(targets, results))


class Predict(luigi.Task):
    def requires(self):
        return {
            'predictor': PredictorNB(),
            'encoder': Encoder(),
            'data': TextsProcessed(),
            'quality': MeasureQuality()
        }

    def output(self):
        return luigi.LocalTarget(rebuild_path_to(self.input()['data']['to_predict'].path, 'predicted'))

    def run(self):
        logger.info('Reading to_predict file: "%s"', self.input()['data']['to_predict'].path)
        df = pd.read_csv(self.input()['data']['to_predict'].path)

        results = self.requires()['predictor'].apply(df)

        df['predicted_tare'] = results
        labels_invertor = self.requires()['encoder'].get_labels_invertor()
        df.predicted_tare = df.predicted_tare.map(labels_invertor)


        logger.info('Writing %s it to "%s"', 'predictions', self.output().path)
        df.to_csv(self.output().path)
