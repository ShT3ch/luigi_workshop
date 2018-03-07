import os

data_archive_link = 'https://kontur.ru/Files/userfiles/file/edu/task.zip'

_project_dir = ''


def get_place_of(path):
    return os.path.join(_project_dir, path)


data_dir = get_place_of('data')


def get_place_of_data(file):
    return os.path.join(data_dir, file)


archive_name = 'task.zip'
train_file = 'train.csv'
to_predict_file = 'test.csv'

predictors_dir = get_place_of('predictors')

def get_place_of_predictor(name):
    return os.path.join(predictors_dir, name)
