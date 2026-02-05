import os
from moto import mock_aws
import streamlit as st
import torch
import numpy as np
from PIL import Image

from src.autoencoder import Autoencoder
from src.dataset import CleanPatchesDataset
from src.patch_extractor import get_jpg_path_from_xml
from src.detect import load_model, preprocess_image, compute_error_map, detect_anomaly, cut_image, detect_anomaly_full_image, create_defect_mask, evaluate_correlation, evaluate_by_category
import src.cloud_s3 as s3


@st.cache_resource
def get_model():
    # Charger et retourner le modèle
    model = Autoencoder(HasAttention=False)
    model.load_state_dict(torch.load("models/autoencoder.pt"))
    model.eval()  # Mode évaluation (pas de dropout, etc.)
    return model

@st.cache_resource
def s3_init():
    os.environ['AWS_ACCESS_KEY_ID'] = 'fake'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'fake'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    # Initialiser le client S3

    mock = mock_aws()
    mock.start()
    bucket_name = 'aeroscan-bucket'
    client = s3.get_s3_client()
    s3.ensure_bucket_exists(client,bucket_name=bucket_name)

    # upload quelques images
    client.upload_file('data/raw/NEU-DET/train/images/crazing/crazing_28.jpg', bucket_name, 'crazing/crazing_28.jpg')
    client.upload_file('data/raw/NEU-DET/train/images/crazing/crazing_29.jpg', bucket_name, 'crazing/crazing_29.jpg')
    client.upload_file('data/raw/NEU-DET/train/images/crazing/crazing_30.jpg', bucket_name, 'crazing/crazing_30.jpg')
    client.upload_file('data/raw/NEU-DET/train/images/crazing/crazing_31.jpg', bucket_name, 'crazing/crazing_31.jpg')
    client.upload_file('data/raw/NEU-DET/train/images/crazing/crazing_32.jpg', bucket_name, 'crazing/crazing_32.jpg')
    client.upload_file('data/raw/NEU-DET/train/images/crazing/crazing_33.jpg', bucket_name, 'crazing/crazing_33.jpg')
    client.upload_file('data/raw/NEU-DET/train/images/crazing/crazing_34.jpg', bucket_name, 'crazing/crazing_34.jpg')
    client.upload_file('data/raw/NEU-DET/train/images/crazing/crazing_35.jpg', bucket_name, 'crazing/crazing_35.jpg')

    return client, bucket_name



def main():
    # 1. Configuration de la page
    st.set_page_config(page_title="AeroScan", layout="wide")
    useAWS_mock = False
    if useAWS_mock:
        client, bucket_name = s3_init()

    if useAWS_mock:
        images = s3.list_images(client, bucket_name=bucket_name, prefix='crazing')
        selected_img = st.selectbox("Select an image", images)
        if selected_img is not None:
            img_bytes = s3.download_image_bytes(client, bucket_name, selected_img['Key'])
            imgPil = s3.bytes_to_pil_image(img_bytes)

            model = get_model()
            conteneur = st.container()
            col1, col2, col3 = conteneur.columns(3)
            # charger et afficher l'image
            img = imgPil
            ImageOriginal = col1.image(img)
            col1.markdown("Original Image")

            # Lancer l'analyse
            with st.spinner("Analyzing..."):
                is_anomaly, error_map, reconstruction = detect_anomaly_full_image(model, img)
                error_norm = np.clip(error_map / 0.1, 0, 1)
                error_norm = error_norm
                error_norm_rgb = np.repeat(error_norm[:, :, np.newaxis], 3, axis=2)
                error_norm_red_only = error_norm_rgb * np.array([[[1, 0, 0]]])
                ImageHM = col2.image(error_norm_red_only)
                col2.markdown("Error Map")
                ImageReconstruction = col3.image(reconstruction)
                col3.markdown("Reconstruction")
                if is_anomaly:
                    st.write("Defect detected !")
                else:
                    st.write("No defect detected !")


    else :

        # 2. Titre et description
        st.title("AeroScan — Defect detection")

        # 3. Charger le modèle (avec cache pour éviter de recharger à chaque interaction)
        model = get_model()


        # 4. Zone d'upload d'image
        
        uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
        # 5. Si une image est uploadée :    
        if uploaded_image is not None:
            conteneur = st.container()
            col1, col2, col3 = conteneur.columns(3)
            # charger et afficher l'image
            img = Image.open(uploaded_image)
            ImageOriginal = col1.image(img)
            col1.markdown("Original Image")

            # Lancer l'analyse
            with st.spinner("Analyzing..."):
                is_anomaly, error_map, reconstruction = detect_anomaly_full_image(model, img)
                error_norm = np.clip(error_map / 0.1, 0, 1)
                error_norm = error_norm
                error_norm_rgb = np.repeat(error_norm[:, :, np.newaxis], 3, axis=2)
                error_norm_red_only = error_norm_rgb * np.array([[[1, 0, 0]]])
                ImageHM = col2.image(error_norm_red_only)
                col2.markdown("Error Map")
                ImageReconstruction = col3.image(reconstruction)
                col3.markdown("Reconstruction")
                if is_anomaly:
                    st.write("Defect detected !")
                else:
                    st.write("No defect detected !")

if __name__ == "__main__":
    main()