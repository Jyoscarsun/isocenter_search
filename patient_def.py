import numpy as np
from scipy import ndimage
import sparse as sp
from constants_def import ModelParameters


# create class which contains everything needed to define a patient
class Patient:
    def __init__(self, cs: ModelParameters, identifier: str, patient_path: str, dose, rois: list,
                 structure_masks: dict, dij, voxel_dimensions: tuple, isocenters_indices: np.array,
                 ring_sizes: tuple = (0.5, 2)) -> None:

        # list out the necessary patient inputs
        self.cs = cs
        self.identifier = identifier
        self.patient_path = patient_path
        self.dose = dose
        self.rois = rois
        self.structure_masks = structure_masks
        self.dij = dij
        self.voxel_dimensions = voxel_dimensions
        self.isocenters_indices = isocenters_indices
        self.ring_sizes = np.array(ring_sizes)

        self.number_of_beams = 24
        print('making opt structures')
        # make the necessary optimization structures - s_shell, g_shell
        self._make_optimization_structs()

        # output the dose and kernels only for the relevant voxels
        self._get_voxels_of_interest()

    # create the s_shell and g_shell using binary dilation
    def _make_optimization_structs(self) -> None:
        # making the shells for gradient + selectivity
        # get set of voxels included in targets + shells
        self.opt_structures = [{}, {}]

        s_voxels = []
        for i in range(len(self.rois)):
            dense_mask = self.structure_masks[self.rois[i]].todense()
            mask_vol = np.count_nonzero(dense_mask)

            orig_mask = np.copy(dense_mask)
            orig_mask[orig_mask > 1] = 1

            # selectivity shell
            s_ring_vol = mask_vol + self.ring_sizes[0] * mask_vol
            ring = np.copy(dense_mask)
            current_s_ring_vol = 0
            while current_s_ring_vol < s_ring_vol:
                ring = ndimage.binary_dilation(ring).astype(ring.dtype)
                current_s_ring_vol = np.count_nonzero(ring)

            s_final_ring = np.subtract(ring, orig_mask)
            sp_final_s_ring = sp.COO(s_final_ring)
            sp_final_s_ring_DOK = sp.DOK.from_coo(sp_final_s_ring)

            # gradient shell
            g_ring_vol = self.ring_sizes[1] * mask_vol + np.count_nonzero(ring)
            current_g_ring_vol = 0
            while current_g_ring_vol < g_ring_vol:
                ring = ndimage.binary_dilation(ring).astype(ring.dtype)
                current_g_ring_vol = np.count_nonzero(ring)

            g_final_ring = ring - orig_mask - s_final_ring

            sp_final_g_ring = sp.COO(g_final_ring)
            sp_final_g_ring_DOK = sp.DOK.from_coo(sp_final_g_ring)

            self.opt_structures[0].update(sp_final_s_ring_DOK.data)
            self.opt_structures[1].update(sp_final_g_ring_DOK.data)
            # self.opt_structures.append(sp_final_s_ring_DOK.data)
            # self.opt_structures.append(sp_final_g_ring_DOK.data)

            sp_orig_mask_DOK = sp.DOK.from_coo(self.structure_masks[self.rois[i]])

            s_voxels.extend(list(sp_final_s_ring_DOK.data.keys())
                            + list(sp_final_g_ring_DOK.data.keys())
                            + list(sp_orig_mask_DOK.data.keys()))

        # output the list of relevant voxels for the tumor
        self.sampled_voxels = list(dict.fromkeys(s_voxels))

    # only consider the relevant sampled voxels when looking at the predicted dose and the dose kernels
    def _get_voxels_of_interest(self) -> None:

        # sparse_dose = sp.COO(self.dose)
        sparse_dose_DOK = sp.DOK.from_numpy(self.dose)

        self.sampled_dose = {x: sparse_dose_DOK.data[x] for x in self.sampled_voxels if x in sparse_dose_DOK.data}
        for key in self.sampled_voxels:
            if key not in self.sampled_dose:
                self.sampled_dose[key] = 0

        self.sampled_dij = []
        for idx, kernel in enumerate(self.dij):
            print('kernel', idx)
            kernel_DOK = sp.DOK.from_coo(kernel)
            kernel_dict = {x: kernel_DOK.data[x] for x in self.sampled_voxels if x in kernel_DOK.data}
            for key in self.sampled_voxels:
                if key not in kernel_dict:
                    kernel_dict[key] = 0

            self.sampled_dij.append(kernel_dict)
