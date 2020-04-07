import os
import json

from bottle import Bottle, run, route, response, request, static_file
import bottle
import warnings

from extractor.extract import UnstructuredExtractor
from utils.utils import delete_directory_contents
from log.logWrapper import customlogger, logging
from adapters.s3_adapter import S3_Adapter
from utils.pdfHelper import pdfConverter
from config.config import *


ALLIGNMENT_DEVIATION = 25
IMAGE_DIMENSION = (0,0)


# warnings.simplefilter("ignore", ResourceWarning)
clog = customlogger(logging.getLogger(__name__)).getLogger()
app = bottle.app()

class EnableCors(object):
    name = 'enable_cors'
    api = 2

    def apply(self, fn, context):
        def _enable_cors(*args, **kwargs):

            # set CORS headers
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS, DELETE'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

            if bottle.request.method != 'OPTIONS':
                # actual request; reply with the actual response
                return fn(*args, **kwargs)
        return _enable_cors

def abort(statusCode, err):
    response.status = statusCode
    return {OUTPUT: FAILURE, "message": err}

@route('/', method='GET')
def inspect():
    return json.dumps({'message': 'IDX Extraction OCR works'})


@route('/split-pdf', method=['OPTIONS', 'POST'])
def split_pdf():
    try:
        inputJSON = json.loads(request.body.read())
        file_path = inputJSON['file_path']
        if not file_path:
            clog.error("Invalid request. File path is empty.")
            return abort(HTTP_CODE_500, "Invalid request. File path is empty.")
    except Exception as err:
        clog.exception("Unable to read request. Reason: %s" % (err))
        return abort(HTTP_CODE_500, "Unable to read request")
    
    try:
        dest_path = DATA_DIR + file_path.split('/')[-1]
        download_status = S3_Adapter().download(file_path, dest_path)

        if download_status:
            page_paths = pdfConverter().convert_pdf_to_images(dest_path)
        else:
            return abort(HTTP_CODE_500, "Unable to download the file %s from s3." % (file_path))
        
        response_json = dict()
        response_json['execution'] = 'completed'
        response_json['paths'] = page_paths
        return response_json
    except Exception as err:
        clog.exception("Unable to process the request. Reason: %s" % (err))
        return abort(HTTP_CODE_500, "Unable to process the request. Reason : %s" % (err))

@route('/find-table-structure', method=['OPTIONS', 'POST'])
def findTableStructure():   
    try:
        inputJSON = json.loads(request.body.read())
        file_path = inputJSON['file_path']
        if not file_path:
            clog.error("Invalid request. File path is empty.")
            return abort(HTTP_CODE_500, "Invalid request. File path is empty.")
    except Exception as err:
        clog.exception("Unable to read request. Reason: %s" % (err))
        return abort(HTTP_CODE_500, "Unable to read request")
    
    try:
        uns_extraction = UnstructuredExtractor(debug=True)
        dest_path = DATA_DIR + file_path #.split('/')[-2] + '_' + file_path.split('/')[-1]
        download_status = S3_Adapter().download(file_path, dest_path)
        if download_status:
            s3_json_path = uns_extraction.run_extraction_process(dest_path, file_path.split('/')[-2])
            if s3_json_path:
                response_json = dict()
                response_json['execution'] = 'completed'
                response_json['json_path'] = s3_json_path
                return response_json
            else:
                return abort(HTTP_CODE_500, "Unable to process the request.")
        else:
            return abort(HTTP_CODE_500, "Unable to download the image %s from s3" % (file_path))
    except Exception as err:
        clog.exception("Unable to process the request. Reason : %s" % (err))
        return abort(HTTP_CODE_500, "Unable to process the request. Reason : %s" % (err))

class Rest_Ext:

    @classmethod
    def start_rest_api(cls):
        app.install(EnableCors())
        app.run(server="paste", host='0.0.0.0', port=9082, debug=True)


if __name__ == '__main__':
    Rest_Ext.start_rest_api()
