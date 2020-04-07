import os
# AWS_DEFAULT_REGION = os.environ['AWS_DEFAULT_REGION']
# AWS_ENDPOINT_URL = os.environ['AWS_ENDPOINT_URL']
BUCKET_NAME = os.environ['BUCKET_NAME']
# AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
# AWS_SECURITY_TOKEN = os.environ['AWS_SECURITY_TOKEN']
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = BASE_DIR + '/data/'
S3_DIR = 'Letter/results/Fund_1/v1/filled/'
HTTP_CODE_500 = 500
HTTP_CODE_202 = 202
HTTP_CODE_401 = 401


FAILURE = "failure"
OUTPUT  = "output"