from .gdrive.download import download_gdrive_data, initialize_router
from .utils.naming import construct_session_name_iabs

exp_params = {
                'track': 'HT',
                'animal_id': 'CA1_22',
                'session': '3',
}

expname = construct_session_name_iabs(exp_params)
data_router, data_pieces = initialize_router()

# TODO: add support for g-cloud auth from desktop (for pydrive) https://cloud.google.com/docs/authentication/application-default-credentials
# TODO: add force_reload flag and check for data existence

download_gdrive_data(data_router,
                     expname,
                     data_pieces=['Aligned data', 'Computation results'],
                     via_pydrive=True)
