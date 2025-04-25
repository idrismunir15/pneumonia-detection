from kaggle.api.kaggle_api_extended import KaggleApi

#initialise the kaggle api
api=KaggleApi()
api.authenticate()

#download dataset
api.dataset_download_files('rsna-pneumonia-detection-challenge','./data',unzip=False)