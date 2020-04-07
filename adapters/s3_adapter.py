import boto3
import os
from botocore.exceptions import ClientError

from config.config import *
from log.logWrapper import customlogger, logging

clog = customlogger(logging.getLogger(__name__)).getLogger()

class S3_Adapter:
    def __init__(self):
        self.s3_client = boto3.client('s3')

    def download(self, file_path_down, dest_file_path = ""):
        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        try:
            # if dest_file_path == "":
            #     dest_file_path = os.path.join(DATA_DIR, file_path_down.split('/')[-1])
            # self.s3_client.download_file(BUCKET_NAME, file_path_down, dest_file_path)
            print("File already present")
        except ClientError as err:
            clog.exception("Unable to download file %s from s3. Reason: %s" % (file_path_down, err))
            return False
        return True

    def upload(self, file_path_down, file_path_up):
        try:
            response = self.s3_client.upload_file(file_path_down, BUCKET_NAME, file_path_up)
        except ClientError as e:
            clog.exception("Unable to upload file %s from s3. Reason: %s" % (file_path_down, err))
            return False
        return True
