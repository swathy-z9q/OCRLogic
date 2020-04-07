import os
import shutil


def delete_directory_contents(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def download_file_from_S3(file_path):
    dest_path = DATA_DIR + file_path.split('/')[-1]
    status = S3_Adapter().download(file_path, dest_path)
    if status:
        return (dest_path)
    else:
        raise Exception("Unable to download the file.")