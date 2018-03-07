import os


def parse_filename(old_path):
    path, file = os.path.split(old_path)
    while len(file) < 1:
        path, file = os.path.split(path)
    comps = file.split('.')
    file, old_extension = '.'.join(comps[:-1]), comps[-1]
    return path, file, old_extension


def rebuild_path_to(old_path, new_extension, new_home=None):
    path, file, old_extension = parse_filename(old_path)

    if new_home is None:
        new_home = path

    res = os.path.join(new_home, '%s.%s' % (file, new_extension))

    return res
