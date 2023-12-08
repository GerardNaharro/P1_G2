import os


def sample_filepath(filename, subfolder='samples'):
    folder = os.path.dirname(__file__)
    return f'{folder}/{subfolder}/{filename}'