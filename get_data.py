import logging
import luigi
from luigi.contrib.external_program import ExternalProgramTask

from config import data_archive_link, get_place_of_data, archive_name, train_file, to_predict_file, data_dir


class DataArchive(ExternalProgramTask):
    def output(self):
        return luigi.LocalTarget(get_place_of_data(archive_name))

    def program_args(self):
        return ['wget', data_archive_link, '-O', self.output().path]


class InputData(ExternalProgramTask):
    def requires(self):
        return DataArchive()

    def output(self):
        return {
            'train': luigi.LocalTarget(get_place_of_data(train_file)),
            'to_predict': luigi.LocalTarget(get_place_of_data((to_predict_file)))
        }

    def program_args(self):
        return ['unzip', '-d', data_dir, self.input().path]
