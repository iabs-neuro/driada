from .gdrive.download import download_gdrive_data, initialize_router
from .utils.naming import construct_session_name_iabs

exp_params = {
                'track' : 'AP',
                'animal_id' : '15',
                'session' : '4EM',
}


expname = construct_session_name_iabs(exp_params)
data_router = initialize_router()

download_gdrive_data(data_router,
                     expname,
                     data_pieces = ['Aligned data', 'Computation results'],
                     via_pydrive = False
                     )