import pandas as pd
import sparse as sp
import math
import os
import natsort
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import gurobipy as gp
from patient_def import Patient
from utils import build_dense
from constants_def import ModelParameters


def find_local_maxima(data, order, numMax, min_dose):
    size = 1 + 2 * order # Side length of the neighbourhood cube we search in
    fp = np.ones((size, size, size)) # Create a 3D matrix named footprint
    fp[order, order, order] = 0 # The center of the footprint cube is set to 0

    # Replaces each element in data array with the maximum of its neighbourhood
    filtered = ndi.maximum_filter(data, footprint=fp)
    mask = data > filtered
    coords = np.asarray(np.where(mask)).T
    values = data[mask]

    # Sort the neighbourhood by variance
    local_var = []
    for coord in coords:
        slices = tuple(slice(max(0, c-order), min(s, c+order+1)) for c,s in zip(coord, data.shape))
        neighbor = data[slices]
        var = np.var(neighbor)
        local_var.append(var)

    local_var = np.array(local_var)

    # Only take the local maxima indices that has a dosage larger than the minimum
    valid_indices = values >= min_dose
    coords = coords[valid_indices]
    values = values[valid_indices]
    local_var = local_var[valid_indices]

    num_maxima = len(local_var)
    if num_maxima < numMax:
        print(f"Found only {num_maxima} local maxima.")
        top_ind = np.argsort(local_var)[-num_maxima:]
    else:
        # Sort by variance and select top 3
        top_ind = np.argsort(local_var)[-1*numMax:]
    
    # Return the coordinates, values, and variance of  coordinates
    top_coords = coords[top_ind]
    top_val = values[top_ind]
    top_var = local_var[top_ind]

    return top_coords, top_val, top_var

def dist(coord1, coord2):
    # Euclidean distance function
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 + (coord1[2]-coord2[2])**2)

def elbow(coords, ub):
    # Elbow method used to decide how many clusters the kMeans clustering should take on
    data = list(coords)
    if(ub > len(coords)):
        # If upper bound of clusters is smaller than number of coordinates (we can't have 5 clusters with only one point)
        print("Upper bound cluster is greater than number of data points")
        ub = len(coords)
        if ub == 1 or ub == 2:
            return ub
    inertia = []

    for i in range(1,ub+1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    print(inertia)

    diffs = np.diff(inertia)
    second_diff = np.diff(diffs)

    elbow_index = np.argmax(second_diff) + 1
    return elbow_index + 1

def kmeans_get_clusters(data, num_cluster):
    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(data)

    labels = kmeans.labels_
    clusters = []
    cluster = []
    for i in range(num_cluster):
        cluster = [point for point, label in zip(data, labels) if label==i]
        clusters.append(cluster)
    
    return clusters

def dbscan_get_clusters(data, eps, min_samples):
    data = np.array(data)
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = clusters.labels_
    clusters = {}

    for label in np.unique(labels):
        if label != -1:
            clusters[label] = data[labels == label].tolist()
    return clusters

def visualize_clusters(dose_arr, max_ind, clusters, isocenters, identifier):
    # Visualize the different clusters against the clinically selected isocenters
    x, y, z = max_ind
    fig, ax = plt.subplots(figsize=(15, 5))
    cax1 = ax.imshow(dose_arr[:, :, z], cmap='viridis')
    ax.set_title(f'Plane z={z}')
    ax.axis('off')
    fig.colorbar(cax1, ax=ax, orientation='vertical', shrink=0.8)

    colors = ['red', 'green', 'cyan', 'magenta', 'yellow', 'white', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy']
    i = 1
    for cluster in clusters:
        cluster_proj = np.array([point[:2] for point in cluster])
        print(cluster_proj)
        if cluster_proj.size > 0:
            ax.scatter(cluster_proj[:, 1], cluster_proj[:, 0], c=colors[i-1], s=30, label=f"Cluster {i}")
        i += 1

    isocenter_proj = np.array([point[:2] for point in isocenters])
    ax.scatter(isocenter_proj[:, 1], isocenter_proj[:, 0], c='black', s=10, label="Clinical Isocenters")

    ax.legend(loc='upper right')
    output_file = 'isocenter_cluster_' + identifier + '.png'  # Specify the file name and format
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def isocenter_set(dose_arr, clusters):
    isocenters = []

    for cluster in clusters:
        if not cluster:
            continue
        max_val = -np.inf
        best_maxima = None
        for maxima in cluster:
            val = dose_arr[maxima[0], maxima[1], maxima[2]]
            if val > max_val:
                max_val = val
                best_maxima = maxima
        
        if best_maxima is not None:
            isocenters.append(best_maxima)
    
    isocenters_array = np.array(isocenters)
    return isocenters_array

def shift_kernel(patient_path, isocenters, identifier):
    # Inputs include 1. patient_path: folder containing all patient info
    # 2. reference: the reference isocenter from previous iteration and the isocenter previous kernel shifts are based on
    # 3. isocenters: a numpy array of updated isocenter coordinates
    # 4. identifier: a string used for composing the file names, usually in the format of 'C0006'
    past_isocenters = pd.read_csv(patient_path + '\\' + 'adj_isocenters.csv', header=None)
    past_isocenters = past_isocenters.values.tolist()
    num_isocenter_orig = len(past_isocenters)
    reference = past_isocenters[0]

    files = [f for f in os.listdir(patient_path)
            if (os.path.isfile(os.path.join(patient_path, f)) and 'kernel' in f and '_0_' in f)]
    files = natsort.natsorted(files)

    os.chdir(patient_path)

    num_isocenter = len(isocenters)
    npz_delete = []
    if num_isocenter_orig > num_isocenter:
        for i in range(num_isocenter, num_isocenter_orig):
            for j in range(1, 25, 1):
                npz_delete.append("kernels_" + identifier[1:] + "_" + str(i) + "_" + str(j) + ".npz")
        
        for file in npz_delete:
            file_path = os.path.join(patient_path, file)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    if len(isocenters) > 1:
        for i in range(1, len(isocenters)):
            j = 1
            for file in files:
                k_file = sp.load_npz(patient_path + '\\' + file).todense()
                shifted_k = np.roll(k_file, shift=isocenters[i][0] - reference[0], axis=0)
                shifted_k = np.roll(shifted_k, shift=isocenters[i][1] - reference[1], axis=1)
                shifted_k = np.roll(shifted_k, shift=isocenters[i][2] - reference[2], axis=2) 

                sparse_k = sp.COO(shifted_k)
                sp.save_npz('kernels_' + identifier[1:] + '_' + str(i) + "_" + str(j) + '.npz', sparse_k)
                j+=1

    j = 1
    for file in files:
        k_file = sp.load_npz(patient_path + '\\' + file).todense()
        shifted_k = np.roll(k_file, shift=isocenters[0][0] - reference[0], axis=0)
        shifted_k = np.roll(shifted_k, shift=isocenters[0][1] - reference[1], axis=1)
        shifted_k = np.roll(shifted_k, shift=isocenters[0][2] - reference[2], axis=2)

        sparse_k = sp.COO(shifted_k)
        sp.save_npz('kernels_' + identifier[1:] + '_0_' + str(j) + '.npz', sparse_k)
        j+=1
    
    csv_file_path = patient_path + '\\' + 'adj_isocenters.csv'
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)

        for isocenter in isocenters:
            writer.writerow(isocenter)
    return

def optimize(patient_path, isocenters, identifier):
    # The optimization function that returns the duration of each kernel and objective value
    # Inputs include 1. patient_path: folder containing all patient info
    # 2. isocenters: a numpy array of updated isocenter coordinates
    # 3. identifier: patient_identifer such as 'C0006'
    cs = ModelParameters()

    pred_doses = [predictions for predictions in os.listdir(patient_path) if 'predict' in predictions]
    ptvlst = [ptv[7:-12].split('_') for ptv in pred_doses]

    for idx, rois in enumerate(ptvlst):
        dose = sp.COO(build_dense(pd.read_csv(os.path.join(patient_path, pred_doses[idx]))))
        structure_masks = {key: sp.COO(build_dense(pd.read_csv(patient_path + '\\' + key + '.csv')))
                            for key in rois}

        files = [f for f in os.listdir(patient_path)
                    if (os.path.isfile(os.path.join(patient_path, f)) and 'kernel' in f)]
        files = natsort.natsorted(files)

        all_kernels = [sp.load_npz(os.path.join(patient_path, i)) for i in files]

        voxel_dimensions = (0.5, 0.5, 1)

        isocenter_indices = isocenters

        patient = Patient(cs,
                    identifier,
                    patient_path,
                    dose,
                    rois,
                    structure_masks,
                    all_kernels,
                    voxel_dimensions,
                    isocenter_indices)

        # building the inverse optimization model
        # get the list of kernels, only for voxels inside the optimization structures of the tumor
        klst = patient.sampled_dij
        # get the list of targets and convert the voxel locations to dictionary format
        targ_dok_lst = [sp.DOK.from_coo(patient.structure_masks[region]) for region in rois]
        targ_dok = sum(targ_dok_lst)
        # separate the voxels into voxels in the target, s_shell and g_shell
        targ = {x: targ_dok.data[x] for x in patient.sampled_voxels if x in targ_dok.data}
        s_shell = patient.opt_structures[0]
        g_shell = patient.opt_structures[1]
        # get a dose map for only the sampled voxels
        d = patient.sampled_dose

        # get all isocenter indices as a list of tuples
        iso_lst = [tuple(iso) for iso in patient.isocenters_indices]

        # obtain number of isocenters in the target as well as volume of the tumor for time estimation
        num_iso = len({x: targ_dok.data[x] for x in iso_lst if x in targ_dok.data})
        volume = len(targ)*0.25/1000

        # set threshold values for optimization
        dt = list(targ_dok.data.values())[0]
        ds = list(targ_dok.data.values())[0]
        dg = dt/2
        dt2 = dt*2

        nt = len(targ)
        ns = len(s_shell)
        ng = len(g_shell)

        # set values to be used in inverse optimization
        y_minus_t = sum({k: max(dt - d[k], 0) for k in d if k in targ}.values())
        y_plus_s = sum({k: max(d[k] - ds, 0) for k in d if k in s_shell}.values())
        y_plus_g = sum({k: max(d[k] - dg, 0) for k in d if k in g_shell}.values())
        y_time_max = 5.135 * dt - 57.479 * volume + 19.83 * num_iso - 21.5543
        y_plus_t2 = sum({k: max(d[k] - dt2, 0) for k in d if k in targ}.values())

        print('Building inverse model')
        m = gp.Model()
        p_t = m.addVars(targ, lb=-float('inf'), ub=float('inf'), name='targ')
        p_s = m.addVars(s_shell, lb=-float('inf'), ub=float('inf'), name='s_shell')
        p_g = m.addVars(g_shell, lb=-float('inf'), ub=float('inf'), name='g_shell')
        p_t2 = m.addVars(targ, lb=-float('inf'), ub=float('inf'), name='targ2')

        wt = m.addVar(name='wt')
        ws = m.addVar(name='ws')
        wg = m.addVar(name='wg')
        wt2 = m.addVar(name='wt2')
        wbot = m.addVar(name='wbot')

        for kernel in range(len(klst)):
            m.addConstr(wbot >= gp.quicksum([sum(klst[kernel][i] * p_t[i] for i in targ.keys()),
                                                sum(klst[kernel][j] * p_s[j] for j in s_shell.keys()),
                                                sum(klst[kernel][k] * p_g[k] for k in g_shell.keys()),
                                                sum(klst[kernel][pt2l] * p_t2[pt2l] for pt2l in targ.keys())]))

        for key in targ.keys():
            m.addConstr(-p_t[key] <= 0)
            m.addConstr(p_t[key] <= wt / (nt * dt))
            m.addConstr(-p_t2[key] <= wt2 / (nt * dt2))
            m.addConstr(p_t2[key] <= 0)
        for key2 in s_shell.keys():
            m.addConstr(-p_s[key2] <= ws / (ns * ds))
            m.addConstr(p_s[key2] <= 0)
        for key3 in g_shell.keys():
            m.addConstr(-p_g[key3] <= wg / (ng * dg))
            m.addConstr(p_g[key3] <= 0)

        m.addConstr(dt * gp.quicksum(p_t) +
                    ds * gp.quicksum(p_s) +
                    dg * gp.quicksum(p_g) +
                    dt2 * gp.quicksum(p_t2) == 1)

        m.setObjective((wt / (nt * dt)) * y_minus_t +
                        (ws / (ns * ds)) * y_plus_s +
                        (wg / (ng * dg)) * y_plus_g +
                        (wt2 / (ng * dt2)) * y_plus_t2 +
                        wbot * y_time_max)

        m.optimize()

        # output weights from inverse optimization
        weights = [var.x for var in m.getVars() if "w" in var.VarName]
        print('weights =', weights)
        print('obj =', m.getObjective().getValue())

        # build the forward model using the weights from the inverse
        print('Building forward model')
        # forward model
        n = gp.Model()
        yt_minus = n.addVars(targ, name='yt_minus')
        yt_plus = n.addVars(targ, name='yt_plus')
        ys_minus = n.addVars(s_shell, name='ys_minus')
        ys_plus = n.addVars(s_shell, name='ys_plus')
        yg_minus = n.addVars(g_shell, name='yg_minus')
        yg_plus = n.addVars(g_shell, name='yg_plus')
        yt2_minus = n.addVars(targ, name='yt2_minus')
        yt2_plus = n.addVars(targ, name='yt2_plus')

        t = n.addVars(len(klst), name='time')

        for voxel in targ.keys():
            n.addConstr(
                gp.quicksum(klst[i][voxel] * t[i] for i in range(len(klst))) - yt_plus[voxel] + yt_minus[voxel] == dt)

        for voxel in s_shell.keys():
            n.addConstr(
                gp.quicksum(klst[i][voxel] * t[i] for i in range(len(klst))) - ys_plus[voxel] + ys_minus[voxel] == ds)

        for voxel in g_shell.keys():
            n.addConstr(
                gp.quicksum(klst[i][voxel] * t[i] for i in range(len(klst))) - yg_plus[voxel] + yg_minus[voxel] == dg)

        for voxel in targ.keys():
            n.addConstr(
                gp.quicksum(klst[i][voxel] * t[i] for i in range(len(klst))) - yt2_plus[voxel] + yt2_minus[voxel] == dt2)

        n.setObjective((weights[0] / (nt * dt)) * gp.quicksum(yt_minus) +
                        (weights[1] / (ns * ds)) * gp.quicksum(ys_plus) +
                        (weights[2] / (ng * dg)) * gp.quicksum(yg_plus) +
                        (weights[3] / (nt * dt2)) * gp.quicksum(yt2_plus) +
                        (weights[4]) * gp.quicksum(t))

        n.optimize()
        print(f"THE OJECTIVE FUNCTION EVALUATES TO {n.objVal}")
        decisions = [var.x for var in n.getVars() if "time" in var.VarName]
        print(len(decisions))
        # print(decisions)

        # output the times for each beam from the inverse model as a csv file
        os.chdir(patient_path)
        with open('inverse_' + rois[0] + '.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for val in decisions:
                writer.writerow([val])

        return n.objVal

def extract_isocenters(isocenter_path):
    # Return the list of clinically selected isocenters to be graphed against algorithmically searched ones 
    isocenters = pd.read_csv(isocenter_path, header=None)
    isocenter_list = isocenters.values.tolist()
    return isocenter_list

def develop_dose_cloud(patient_path):
    # Function returns a 3D numpy array that represents the dose cloud of the prescription
    # Function also returns the index of the voxel with the highest dosage
    files = [f for f in os.listdir(patient_path)
                if (os.path.isfile(os.path.join(patient_path, f)) and 'kernel' in f)]
    files = natsort.natsorted(files)

    all_kernels = [sp.load_npz(os.path.join(patient_path, i)) for i in files]

    # Load the file containing time duration for each kernel
    duration = pd.read_csv(patient_path + '\\' + 'inverse_PTV0.csv', header=None)

    for idx, kernel in enumerate(all_kernels):
        if idx == 0:
            dose_kernel = kernel * duration.iloc[0, 0]
        
        if duration.iloc[idx, 0] == 0.0:
            continue

        if idx != 0:
            kernel = kernel * duration.iloc[idx, 0]
            dose_kernel = dose_kernel + kernel
    dose_arr = dose_kernel.todense()
    dose_arr = np.array(dose_arr)

    max_ind_flat = np.argmax(dose_arr)
    max_ind = np.unravel_index(max_ind_flat, dose_arr.shape)

    return dose_arr, max_ind

def compute_sc(dose_cloud, pres_dose, min_dose):
    tumour_mask = 0
    inside_tumour = 0
    entire_space = 0
    for i in range(512):
        for j in range(512):
            for k in range(216):
                if(dose_cloud[i, j, k] >= min_dose):
                    entire_space += 1
                if(pres_dose[i, j, k] == min_dose):
                    tumour_mask += 1
                    if(dose_cloud[i, j, k] >= min_dose):
                        inside_tumour += 1
    coverage = inside_tumour / tumour_mask
    selectivity = inside_tumour / entire_space
    conformity = coverage * selectivity
    return coverage, selectivity, conformity

if __name__ == '__main__':
    patient_path = "D:\Summer 24\Research Materials\Gamma Knife Code\GK - IsocenterProject Data\C0000"

    ptv0 = pd.read_csv(patient_path + '\\' + 'PTV0.csv', header=None)
    ptv0 = ptv0.values.tolist()
    min_dose = float(ptv0[2][1])

    print(f"THE MINIMUM DOSE IS {min_dose}")
    obj_diff = np.inf
    obj_val = 0

    pres_dose = build_dense(pd.read_csv(os.path.join(patient_path, "PTV0.csv")))
    dose_arr, max_ind = develop_dose_cloud(patient_path)

    # x, y, z = max_ind
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # cax1 = axs[0].imshow(dose_arr[x, :, :], cmap='viridis')
    # axs[0].set_title(f'Plane x={x}')
    # axs[0].axis('off')
    # fig.colorbar(cax1, ax=axs[0], orientation='vertical', shrink=0.8)

    # cax2 = axs[1].imshow(dose_arr[:, y, :], cmap='viridis')
    # axs[1].set_title(f'Plane y={y}')
    # axs[1].axis('off')
    # fig.colorbar(cax2, ax=axs[1], orientation='vertical', shrink=0.8)

    # cax3 = axs[2].imshow(dose_arr[:, :, z], cmap='viridis')
    # axs[2].set_title(f'Plane z={z}')
    # axs[2].axis('off')
    # fig.colorbar(cax3, ax=axs[2], orientation='vertical', shrink=0.8)

    # output_file = 'dose_cloud_C0006.png'  # Specify the file name and format
    # fig.savefig(output_file, dpi=300, bbox_inches='tight')

    # plt.show()

    coords, value, var = find_local_maxima(dose_arr, 1, 20, min_dose)

    num_cluster = elbow(coords,10)
    clusters = kmeans_get_clusters(coords, num_cluster)
    updated_isocenters = isocenter_set(dose_arr, clusters)
    shift_kernel(patient_path, updated_isocenters, 'C0000')
    optimize(patient_path, updated_isocenters, 'C0000')


    # isocenters = extract_isocenters(patient_path + '\\' + 'adj_isocenters.csv')
    # visualize_clusters(dose_arr, max_ind, clusters, isocenters, "C0094")
    
    # while(obj_val == 0):
        # dose_arr, max_ind = develop_dose_cloud(patient_path)
        # coords, value, var = find_local_maxima(dose_arr, 2, 20, min_dose)

        # num_cluster = elbow(coords,10)
        # clusters = kmeans_get_clusters(coords, num_cluster)

        # If we want to visualize the clusters vs original isocenters
        # isocenters = extract_isocenters(patient_path + '\\' + 'adj_isocenters.csv')
        # visualize_clusters(dose_arr, max_ind, clusters, isocenters)


        # updated_isocenters = isocenter_set(dose_arr, clusters)
        # shift_kernel(patient_path, updated_isocenters, 'C0006')
        # obj_val = optimize(patient_path, updated_isocenters, 'C0006')

    updated_dose_arr, max_ind = develop_dose_cloud(patient_path)
    coverage, selectivity, conformity = compute_sc(updated_dose_arr, pres_dose, min_dose)

    print(f"Coverage: {coverage}")
    print(f"Selectivity: {selectivity}")
    print(f"Conformity: {conformity}")
