import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from encoder_lib import Encoder, GroupScaler, Wrapper
from columns import Columns


def get_encoder(columns: Columns) -> Encoder:
    return Encoder(
        Wrapper(OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), columns.encoder_list()),
        GroupScaler(StandardScaler(with_mean=False), columns.numericals()),
        GroupScaler(StandardScaler(with_mean=False), columns.targets()),
    )
