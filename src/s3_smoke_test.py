import cloud_s3 as s3
from moto import mock_aws
if __name__ == "__main__":
    with mock_aws():
        client = s3.get_s3_client()
        s3.ensure_bucket_exists(client,bucket_name='test-bucket')
        client.upload_file('data/raw/NEU-DET/train/images/crazing/crazing_28.jpg', 'test-bucket', 'test-key')
        objects = s3.list_images(client, bucket_name='test-bucket', prefix='test-key')
        print(objects)
        image_bytes = s3.download_image_bytes(client, bucket_name='test-bucket', key='test-key')
        image = s3.bytes_to_pil_image(image_bytes)
        print(image.size)