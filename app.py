import os
import platform
import matplotlib.pyplot as plt
import SimpleITK as sitk
from ipywidgets import interact
from scipy import ndimage
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import streamlit as st



def resize_3D_volume(vol, target_size=(30, 256, 256)):
    """
    Given a 3D volumteric array with shape (Z,X,Y). This function will resize
    the image across z-axis.
    The purpose of this function to standardise the depth of MRI image.

    Args:
        vol: 3D array with shape (Z,X,Y) that represents the volume of a MRI image
        target_size: target size to shape into the volumetric data

    Returns:
        np.ndarray: Returns the resized MRI volume
    """
    # Set the desired depth
    desired_depth, desired_width, desired_height = target_size
    # Get current depth
    current_depth = vol.shape[0]
    current_width = vol.shape[1]
    current_height = vol.shape[2]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    resized_vol = ndimage.zoom(vol, (depth_factor, width_factor, height_factor), order=1)
    return resized_vol



def denoise_3D_volume(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume to denoise

    Returns:
        np.ndarray: Returns denoised MRI volume
    """
    vol_sitk = sitk.GetImageFromArray(vol)
    denoised_vol_sitk = sitk.CurvatureFlow(vol_sitk, timeStep=0.01, numberOfIterations=7)
    denoised_vol = sitk.GetArrayFromImage(denoised_vol_sitk)
    return denoised_vol


def efficient_bias_field_correction_volume(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume to perform efficient bias field correction

    Returns:
        np.ndarray: Returns bias field corrected MRI volume
    """
    # Ref: https://medium.com/@alexandro.ramr777/how-to-do-bias-field-correction-with-python-156b9d51dd79
    # Ref: https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
    # Convert the NumPy array to SimpleITK image
    vol_sitk = sitk.GetImageFromArray(vol)

    vol_sitk = sitk.Cast(vol_sitk, sitk.sitkFloat64)

    vol_sitk_transformed = sitk.RescaleIntensity(vol_sitk, 0, 255)

    vol_sitk_transformed = sitk.LiThreshold(vol_sitk_transformed, 0, 1)

    head_mask = vol_sitk_transformed

    shrink_factor = 4

    input_img = vol_sitk

    input_img = sitk.Shrink(vol_sitk, [shrink_factor] * input_img.GetDimension())
    mask_img = sitk.Shrink(head_mask, [shrink_factor] * input_img.GetDimension())

    # Perform bias field correction using N4BiasFieldCorrection
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(input_img, mask_img)

    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(vol_sitk)

    log_bias_field = sitk.Cast(log_bias_field, sitk.sitkFloat64)

    corrected_image_full_resolution = vol_sitk / sitk.Exp(log_bias_field)

    # Get the NumPy array representation of the bias-corrected volume
    bias_corrected_vol = sitk.GetArrayFromImage(corrected_image_full_resolution)

    return bias_corrected_vol



def normalise_volume_pixels(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Normalised MRI volume
    """
    # Normalise the volume pixels to the range [0, 1]
    min_value = np.min(vol)
    max_value = np.max(vol)
    normalised_vol = (vol - min_value) / (max_value - min_value)

    return normalised_vol




def center_volume_pixels(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Zero centered MRI volume
    """
    # Calculate the mean value
    mean_value = np.mean(vol)

    # Center the data
    centered_vol = vol - mean_value

    return centered_vol



def standardise_volume_pixels(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Standardised MRI volume
    """
    # Calculate the mean and standard deviation
    mean_value = np.mean(vol)
    std_value = np.std(vol)

    # Standardise the data
    standardised_vol = (vol - mean_value) / std_value

    return standardised_vol


def load_model_from_disk(model_name, model_store_path='Models'):
    """Summary

    Args:
        model_name (str): Model name
        model_store_path (str, optional): Path to load a model from

    Returns:
        TYPE: Description
    """
    file_name = f"{model_store_path}/{model_name}/{model_name}.h5"

    # For running code on Windows
    if platform.system() == "Windows":
        file_name = file_name.replace('/', '\\')

    if os.path.exists(file_name):
        model = load_model(file_name)
    else:
        print(f"ERROR: Model file {file_name} not found.")
        return None

    return model




def preprocess_mri(mri_vol):
    """Summary

    Args:
        mri_vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Returns preprocessed MRI volume
    """
    mri_vol = resize_3D_volume(mri_vol)
    mri_vol = denoise_3D_volume(mri_vol)
    mri_vol = efficient_bias_field_correction_volume(mri_vol)
    mri_vol = normalise_volume_pixels(mri_vol)
    mri_vol = center_volume_pixels(mri_vol)
    mri_vol = standardise_volume_pixels(mri_vol)
    return mri_vol



best_mrnet_model_name = 'MRNet_Model3'
best_mrnet_model_cutoff_threshold = 0.429860  # Determined during evaluation on the test set
mrnet_model = load_model_from_disk(best_mrnet_model_name)

best_kneemri_model_name = 'kneeMRI_Model6'
kneemri_model = load_model_from_disk(best_kneemri_model_name)

st.set_page_config(
    page_title="Medical Image Models",
)

st.title("KNEE LIGAMENT ASSESSMENT")

st.subheader("Welcome to Deep Learning based assessment of Knee MRI")

mri_file = st.file_uploader("Upload MRI",
                            type=['npy', 'pck'],
                            key="mri_file")

mrnet_label = {0: 'Healthy', 1: 'ACL Tear'}
kneemri_label = {0: 'Healthy', 1: 'Partial ACL Tear', 2: 'Complete ACL Tear'}

if mri_file is not None:
    if mri_file.name.endswith('.npy'):
        mri_vol = np.load(mri_file)
    elif mri_file.name.endswith('.pck'):
        mri_vol = pickle.load(mri_file)

    mri_vol = mri_vol.astype(np.float64)  # Change the dtype to float64
    # mri_vol.shape

    predict_button = st.button('Predict')

    if predict_button:
        with st.spinner('Preprocessing...'):
            preprocessed_mri_vol = preprocess_mri(mri_vol)
            # preprocessed_mri_vol.shape

        with st.spinner('Predicting...'):
            mri_vol = np.expand_dims(mri_vol, axis=3)  # Adding extra axis for making it compatible for 3D Convolutions
            # mri_vol.shape
            mrnet_pred_prob = mrnet_model.predict(np.array([preprocessed_mri_vol]))
            print(mrnet_pred_prob)
            mrnet_pred_label = (mrnet_pred_prob[0] >= best_mrnet_model_cutoff_threshold).astype('int')
            print(mrnet_pred_label)

            kneemri_pred_prob = kneemri_model.predict(np.array([preprocessed_mri_vol]))
            print(kneemri_pred_prob)
            kneemri_pred_label = kneemri_pred_prob[0].argmax(axis=-1)
            print(kneemri_pred_label)

            if mrnet_pred_label == 1 and kneemri_pred_label == 0:
                if mrnet_pred_prob[0] > kneemri_pred_prob[0][kneemri_pred_label]:
                    st.write(f'ACL Tear Prediction : **{mrnet_label[mrnet_pred_label[0]]}**')
                    st.warning("Possibility of an ACL tear, unsure about the grade of tear.")
            elif mrnet_pred_label == 0 and kneemri_pred_label > 0:
                if mrnet_pred_prob[0] < kneemri_pred_prob[0][kneemri_pred_label]:
                    st.write(f'ACL Tear Grade Prediction : **{kneemri_label[kneemri_pred_label]}**')
                    st.warning("Possibility of an ACL tear.")
            else:
                st.write(f'ACL Tear Prediction : **{mrnet_label[mrnet_pred_label[0]]}**')
                st.write(f'ACL Tear Grade Prediction : **{kneemri_label[kneemri_pred_label]}**')

    slice_number = st.slider('MRI Slice', min_value=1,
                             max_value=30, value=15) - 1

    img = mri_vol[slice_number, :, :]
    normalized_image_data = (img - img.min()) / (img.max() - img.min())

    with st.columns(3)[1]:
        st.image(normalized_image_data, width=300)

    st.error('Disclaimer : The model predictions are just for reference. Please consult your doctor for treatment.')
