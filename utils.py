import numpy as np

# util function to change sparse matrix to full np.array
def build_dense(sparse_df, ur_idx=(600, 600, 600), d_matrix_size=None):
    if d_matrix_size is None:
        d_matrix_size = [512, 512, 256]

    ptv_loc = sparse_df['loc'].tolist()
    ptv_data = sparse_df['val'].tolist()

    ptv_loc_UR = np.unravel_index(ptv_loc, ur_idx)

    d_matrix = np.zeros(d_matrix_size)
    d_matrix[ptv_loc_UR[0], ptv_loc_UR[1], ptv_loc_UR[2]] = ptv_data

    return d_matrix