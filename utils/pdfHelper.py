from pdf2image import convert_from_path
import os
import threading

from log.logWrapper import customlogger, logging
from adapters.s3_adapter import S3_Adapter
from config.config import *

clog = customlogger(logging.getLogger(__name__)).getLogger()

from config.config import *


class pdfConverter:
    def convert_pdf_to_images(self, file_path):
        pages = []
        try:
            if file_path.endswith('.pdf'):
                folder_name = file_path.split('/')[-1].split('.')[0]
                dest_folder_name = os.path.join(DATA_DIR, file_path.split('/')[-1].split('.')[0])
                if not os.path.isdir(dest_folder_name):
                    os.mkdir(dest_folder_name)

                pages = convert_from_path(file_path, 500, thread_count=10)
        except Exception as err:
            clog.exception("Unable to convert pdf to images. Reason %s" % (err))
            raise Exception("Unable to convert pdf to images")

        try:
            page_paths = []
            if len(pages) > 0:
                threads = []
                for j in range(0, len(pages)):
                    page_path = os.path.join(dest_folder_name, 'page_{0}.jpg'.format(j))
                    
                    t = threading.Thread(target = pages[j].save, args=(page_path, 'JPEG',))
                    threads.append(t)
                    t.start()
                    page_paths.append(page_path)
                
                for x in threads:
                    x.join()
                clog.info("pdf to image conversion completed")

                s3_path = os.path.join(S3_DIR, folder_name + '/')
        except Exception as err:
            clog.exception("Unable to save converted images. Reason %s" % (err))
            raise Exception("Unable to save converted images")

        # try:
        #     s3_paths = []
        #     if len(page_paths) > 0:
        #         threads = []
        #         for page_path in page_paths:
        #             s3_img_path = s3_path + page_path.split('/')[-1]
        #             t = threading.Thread(target = S3_Adapter().upload, args=(page_path,s3_img_path,))
        #             threads.append(t)
        #             t.start()
        #             s3_paths.append(s3_img_path)
        #         for x in threads:
        #             x.join()
        #         clog.info("s3 upload completed")
        #     return s3_paths
        # except Exception as err:
        #     clog.exception("Unable to upload images to s3. Reason %s" % (err))
        #     raise Exception("Unable to upload images to s3")
        # finally:
        #     for page_path in page_paths:
        #         if os.path.exists(page_path):
        #             os.unlink(page_path)
        #     if os.path.exists(dest_folder_name):
        #         os.rmdir(dest_folder_name) # Deleting the folder file

        #     if os.path.exists(file_path):
        #         os.unlink(file_path) # Deleting the pdf file
