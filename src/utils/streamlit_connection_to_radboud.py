import streamlit as st
from webdav3.client import Client
import logging
import numpy as np
import mne
import os

@st.cache_resource
def get_connection():
    # Configuration for RDR WebDAV access
    options = {
        'webdav_hostname': st.secrets["RDR_HOSTNAME"],
        'webdav_login':    st.secrets["RDR_USERNAME"],
        'webdav_password': st.secrets["RDR_PASSWORD"],
    }
    # Initialize the WebDAV client
    rdr_client = Client(options)
    if not rdr_client:
        logging.error("Connection error.")
        return None, None
    # Get file list from remote directory
    remote_path = f"{st.secrets['RDR_REMOTE_PATH']}/"
    filenames = rdr_client.list(remote_path)
    return rdr_client, filenames

@st.cache_data
def download_dataset_from_repository(experiment_name: str):
    """Load data from the Radboud Data Repository."""
    # Get webDAV client and filenames
    client, filenames = get_connection()
    # Find valid files for the selected experiment
    experiment_files = [filename for filename in filenames if filename.startswith(experiment_name) and (filename.endswith(".npy") or filename.endswith(".edf"))]
    if not experiment_files:
        logging.error(f"No valid files found for experiment {experiment_name}.")
        st.sidebar.error("No valid files found for the selected experiment.")
        return None
    # Process uploaded files
    dataset = {}
    for filename in experiment_files:
        # Fetch files from Radboud Data Repository
        local_path = st.secrets["CACHE_PATH"] + filename
        remote_path = f"{st.secrets['RDR_REMOTE_PATH']}/{filename}"
        client.download(remote_path, local_path)
        # Read the file content according to its type
        if filename.endswith("conf.npy"):
            logging.info(f"Attempting to load scoring file: {filename}")
            dataset["scoring"] = np.load(local_path)
        elif filename.endswith("sleepscoring_manual.edf"):
            logging.info(f"Attempting to load PSG file: {filename}")
            dataset["raw_obj"] = mne.io.read_raw_edf(local_path, preload=False, verbose=False)
        else:
            logging.warning(f"File {filename} is not a valid EDF or NPY file.")
    return dataset

def upload_file_to_repository(file_path: str = None):
    """Sidebar widget to upload a file to the external data repository."""
    client, _ = get_connection()
    if not client:
        logging.error("No connection to repository.")
        return False

    # Define remote path
    filename = os.path.basename(file_path)
    remote_path = os.path.join(st.secrets['RDR_UPLOAD_PATH'], filename)


    # Upload to WebDAV
    try:
        client.upload_sync(remote_path=remote_path, local_path=file_path)
        logging.info(f"Uploaded {file_path} to repository.")
        return True
    except Exception as e:
        logging.error(f"Upload failed: {e}")
        return False