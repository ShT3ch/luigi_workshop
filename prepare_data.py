import logging
import luigi
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

import coloredlogs
from sklearn.preprocessing import LabelEncoder

from config import get_place_of_predictor
from log_config import log_format, log_date_format, loglevel, logfile, add_handlers_and_init

logger = logging.getLogger('DATA PREP')
add_handlers_and_init(logger, log_format, log_date_format, loglevel, logfile)
coloredlogs.install(loglevel)

from get_data import InputData
from text_processor import TextProcessor
from utils import rebuild_path_to


class TextsProcessed(luigi.Task):
    extension = 'processed.csv'

    def transform_target_name(self, target=None):
        if isinstance(target, str):
            to_process = target
        else:
            raise ValueError('will work only with strings')

        return rebuild_path_to(to_process, TextsProcessed.extension)

    def requires(self):
        return InputData()

    def output(self):
        return dict((k, luigi.LocalTarget(self.transform_target_name(v.path))) for k, v in self.input().items())

    def run(self):
        logger.info('Creating text processor')
        text_processor = TextProcessor()

        for file in self.input().keys():
            logger.info('Reading %s file: "%s"', file, self.input()[file].path)
            df = pd.read_csv(self.input()[file].path)

            logger.info('Its %s lines', df.shape[0])
            logger.info('Start processing %s...', file)

            df.name = df.name.map(lambda x: text_processor.process_text(x, lang='ru'))
            df.name = df.name.map(lambda x: ' '.join(x))

            logger.info('Processing of %s succeed, writing it to "%s"', file, self.output()[file].path)

            df.to_csv(self.output()[file].path)


class Encoder(luigi.Task):
    extension = 'labels_encoded.csv'

    name = 'labels_encoder.bin'

    def transform_target_name(self, target=None):
        if isinstance(target, str):
            to_process = target
        else:
            raise ValueError('will work only with strings')

        return rebuild_path_to(to_process, Encoder.extension)

    def requires(self):
        return TextsProcessed()

    def output(self):
        res = dict([('train', luigi.LocalTarget(self.transform_target_name(self.input()['train'].path)))])
        res['encoder'] = luigi.LocalTarget(get_place_of_predictor(Encoder.name), format=luigi.format.Nop)
        return res

    def run(self):
        logger.info('Encoding tare labels')

        file = 'train'
        logger.info('Reading %s file: "%s"', file, self.input()[file].path)
        df = pd.read_csv(self.input()[file].path)

        logger.info('Its %s lines', df.shape[0])
        logger.info('Start encoding %s...', file)

        encoder = LabelEncoder()
        df.tare = encoder.fit_transform(df.tare)

        logger.info('Processing of %s succeed, writing it to "%s"', file, self.output()[file].path)
        df.to_csv(self.output()[file].path)

        logger.info('Writing labels encoder to %s', self.output()['encoder'].path)
        with self.output()['encoder'].open('wb') as outcome:
            pickle.dump(encoder, outcome)

    def get_labels_invertor(self):
        logger.info('Reading pipeline: %s to extract labels invertor', self.output()['encoder'].path)
        with self.output()['encoder'].open('rb') as outcome:
            encoder = pickle.load(outcome)
        return encoder.inverse_transform


class TrainHoldout(luigi.Task):
    extension = 'part.csv'
    parts = ['train', 'holdout']

    split_params = luigi.DictParameter(dict())

    def transform_target_name(self, target=None, part=None):
        if isinstance(target, str):
            to_process = target
        else:
            raise ValueError('will work only with strings')

        return rebuild_path_to(to_process, part + TrainHoldout.extension)

    def requires(self):
        return Encoder()

    def output(self):

        res = dict()
        input_train = self.input()['train']
        for part in TrainHoldout.parts:
            res[part] = luigi.LocalTarget(self.transform_target_name(input_train.path, part=part))

        return res

    def run(self):
        logger.info('Reading train file: "%s"', self.input()['train'].path)
        df = pd.read_csv(self.input()['train'].path)

        logger.info('Splitting train with extra params: %s', self.split_params)
        train, holdout = train_test_split(df, **self.split_params)

        logger.info('Writing %s it to "%s"', 'train', self.output()['train'].path)
        train.to_csv(self.output()['train'].path)

        logger.info('Writing %s it to "%s"', 'holdout', self.output()['holdout'].path)
        holdout.to_csv(self.output()['holdout'].path)
