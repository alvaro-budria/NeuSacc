import os


# Filesystem
def get_file_extension(file):
    return os.path.splitext(file)[1]


def get_parent_directory(file, levels_up=1):
    for i in range(levels_up):
        file = os.path.split(file)[0]
    return file


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_parent_directory(path):
    parent_dir = get_parent_directory(path)
    create_directory(parent_dir)
