import os
import boto3
from moto import mock_aws
import moto
from io import BytesIO
from PIL import Image
from botocore.exceptions import ClientError

def get_s3_client():
    # get s3 client
    return boto3.client('s3')

def ensure_bucket_exists(client ,bucket_name):
    # ensure bucket exists
    s3 = client
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response['Error']['Code'] == '404'or e.response['Error']['Code'] == 404:
            s3.create_bucket(Bucket=bucket_name)

def upload_folder(client, bucket_name, local_root, prefix='', max_files=None):
    # upload folder to s3
    uploaded_count = 0
    for root, dirs, files, in os.walk(local_root):
        for file in files:
            if max_files is not None and uploaded_count >= max_files:
                break
            file_path = os.path.join(root, file)
            file_name = os.path.join(prefix, file)
            client.upload_file(file_path, bucket_name, file_name)
            uploaded_count += 1

def list_images(client, bucket_name, prefix=''):
    # list images in s3
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' not in response:
        return []
    return response['Contents']

def download_image_bytes(client, bucket_name, key):
    # download image bytes from s3
    response = client.get_object(Bucket=bucket_name, Key=key)
    return response['Body'].read()

def bytes_to_pil_image(bytes):
    # convert bytes to PIL image
    image = Image.open(BytesIO(bytes))
    return image