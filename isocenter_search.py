import pandas as pd
import sparse as sp
import math
import os
import natsort
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import KDTree
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
    
    # Return the coordinates, values, and variance of coordinates
    top_coords = coords[top_ind]
    top_val = values[top_ind]
    top_var = local_var[top_ind]

    return top_coords, top_val, top_var

def filter_points(data, order, mask, target_num):
    # Data: the predicted dose information
    # Order: construct a neighbourhood such that each point picked must not fall within the same neighbourhood
    # Mask: contain coordinates to all voxels that are within the tumour mask
    # Target_num: the number of maximum areas that we want returned
    val_coord_pair = []
    selected_points = []

    for coord in mask:
        i, j, k = coord
        value = data[i, j, k]
        val_coord_pair.append((value, coord))

    val_coord_pair.sort(reverse=True, key=lambda x: x[0])

    for value, coord in val_coord_pair:
        close = False
        for point in selected_points:
            if (abs(coord[0] - point[0]) < order and abs(coord[1] - point[1]) < order and abs(coord[2] - point[2]) < order):
                close = True
                break

        if close == False:
            selected_points.append(coord)
        if len(selected_points) >= target_num:
            break

    return selected_points

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

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    labels = dbscan.labels_

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

    colors = ['red', 'green', 'cyan', 'magenta', 'yellow', 'white', 'orange', 'purple', 'brown', 
              'pink', 'gray', 'olive', 'navy', 'lightblue', 'darkgreen', 'salmon', 'coral', 'tan', 'teal',
            'lime', 'maroon', 'aqua', 'silver', 'gold', 'indigo',
            'violet', 'beige', 'khaki', 'plum', 'orchid', 'darkred',
            'skyblue', 'peachpuff', 'wheat', 'lightgreen', 'lightgray',
            'lavender', 'thistle', 'burlywood', 'chartreuse', 'cyan',
            'blueviolet', 'tomato', 'powderblue', 'midnightblue', 'darkorange']
    i = 1
    for cluster in clusters.values():
        cluster_proj = np.array([point[:2] for point in cluster])
        print(cluster_proj)
        if cluster_proj.size > 0:
            ax.scatter(cluster_proj[:, 1], cluster_proj[:, 0], c=colors[i-1], s=2, label=f"Cluster {i}")
        i += 1

    # for cluster in clusters:
    #     cluster_proj = np.array([point[:2] for point in cluster])
    #     print(cluster_proj)
    #     if cluster_proj.size > 0:
    #         ax.scatter(cluster_proj[:, 1], cluster_proj[:, 0], c=colors[i-1], s=5, label=f"Cluster {i}")
    #     i += 1

    isocenter_proj = np.array([point[:2] for point in isocenters])
    ax.scatter(isocenter_proj[:, 1], isocenter_proj[:, 0], c='black', s=5, label="Clinical Isocenters")

    ax.legend(loc='upper right', prop={'size': 5})
    output_file = 'isocenter_cluster_' + identifier + '.png'  # Specify the file name and format
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_points(dose_arr, max_ind, points, isocenters, identifier):
    x, y, z = max_ind
    fig, ax = plt.subplots(figsize=(15, 5))
    cax1 = ax.imshow(dose_arr[:, :, z], cmap='viridis')
    ax.set_title(f'Plane z={z}')
    ax.axis('off')
    fig.colorbar(cax1, ax=ax, orientation='vertical', shrink=0.8)

    point_proj = np.array([point[:2] for point in points])
    print(point_proj)
    if point_proj.size > 0:
        ax.scatter(point_proj[:, 1], point_proj[:, 0], c='black', s=1, label="Local max in the tumour mask")
    
    isocenter_proj = np.array([point[:2] for point in isocenters])
    ax.scatter(isocenter_proj[:, 1], isocenter_proj[:, 0], c='red', s=1, label="Clinical Isocenters")
    
    ax.legend(loc='upper right')
    output_file = "local_maxima_" + identifier + '.png'
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
        
        if  duration.iloc[idx, 0] == 0.0:
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
    tm = 0
    it = 0
    es = 0
    for i in range(512):
        for j in range(512):
            for k in range(216):
                if(dose_cloud[i, j, k] >= min_dose):
                    es += 1
                if(pres_dose[i, j, k] == min_dose):
                    tm += 1
                    if(dose_cloud[i, j, k] >= min_dose):
                        it += 1
    cov, sel = it/tm, it/es
    con = cov*sel
    return cov, sel, con

def tumour_mask(pres_dose):
    mask = []
    for i in range(512):
        for j in range(512):
            for k in range(256):
                if(pres_dose[i, j, k] != 0):
                    mask.append([i, j, k])
    return mask

def assign_voxels(mask, isocenters):
    # The indices of the voxels within the tumour mask
    voxel_ind = np.array(mask)
    # Build a KDTree with the isocenter coordinates
    isocenter_tree = KDTree(isocenters)
    # Query the tree with voxel coordinates to find the nearest isocenter
    distances, nearest_isocenter = isocenter_tree.query(voxel_ind)
    r = np.zeros(len(isocenters)) # Radius of influence of each isocenter
    result = {}

    for i, ind in enumerate(voxel_ind):
        isocenter_ind = nearest_isocenter[i]
        distance = distances[i]

        if distance > r[isocenter_ind]:
            r[isocenter_ind] = distance

        result[tuple(ind)] = isocenter_ind
    # Return variables
    # Result is a 3D array with values at each location being the closest isocenter
    # r is a array with values at each location being the radius of influence for that isocenter
    return result, r
    
def shift_isocenter(isocenters, r, dose_cloud, pres_dose, min_dose, cov, sel, con, max_ind):
    """
    The function that takes in isocenters and their respective radius, and return the new set of isocenters
    isocenters: the previous set of isocenters that requires shifting
    r: sphere of influence for each isocenter. Compute relevant metrics within each sphere and shift isocenter accordingly
    """

    """
    Rationale behind isocenter shift: 
    Only move the isocenters in regions that perform worse in comparison to the overall tumour space metrics
    If the regional selectivity is worse off, move the isocenter closer to indices of max dosage (assumed to be the center of entire tumour space)
    If the regional coverage is worse off, move the other isocenters closer to the region but only if those isocenter regions are also performing worse
    """
    metrics = []

    for i in range(len(isocenters)):
        cx, cy, cz = isocenters[i][0], isocenters[i][1], isocenters[i][2]
        x_min = max(0, cx-r[i])
        x_max = min(511, cx+r[i])
        y_min = max(0, cy-r[i])
        y_max = min(511, cy+r[i])
        z_min = max(0, cz-r[i])
        z_max = min(255, cz+r[i])

        x, y, z = np.ogrid[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        dist_squared = (z-cz) ** 2 + (y-cy)**2 + (x-cx)**2
        mask = dist_squared <= r[i]**2

        sphere = (np.array(np.where(mask)).T + [z_min, y_min, x_min]).tolist()

        tm = 0 # Num of voxels within tumour mask
        it = 0 # Num of voxels within tumour mask that has dosage over min
        es = 0 # Num of voxels within entire sphere that has dosage over min
        for coord in sphere:
            if(dose_cloud[coord[0], coord[1], coord[2]] >= min_dose):
                es += 1
            if(pres_dose[coord[0], coord[1], coord[2]] == min_dose):
                tm += 1
                if(dose_cloud[coord[0], coord[1], coord[2]] >= min_dose):
                    it += 1
        cov, sel = it/tm, it/es
        con = cov*sel
        metrics.append([cov, sel, con])

    regions = []
    for i in range(len(metrics)):
        if(metrics[i][2] < con):
            regions.append(i)
    
    for region in regions:
        if(metrics[region][0] < cov): # Coverage is performing worse
            for neighbour in regions:
                if neighbour != region and metrics[neighbour][0] < cov:
                    x_diff, y_diff, z_diff = isocenters[neighbour][0] - isocenters[region][0], isocenters[neighbour][1] - isocenters[region][1], isocenters[neighbour][2] - isocenters[region][2]
                    abs_x = abs(x_diff)
                    abs_y = abs(y_diff)
                    abs_z = abs(z_diff)
                    max_mag = max(abs_x, abs_y, abs_z)

                    def scale(mag, max_mag):
                        if max_mag == 0:
                            return 0
                        return int(1 + 2 * (mag/max_mag))
            
                    isocenters[neighbour][0] += scale(x_diff, max_mag)
                    isocenters[neighbour][1] += scale(y_diff, max_mag)
                    isocenters[neighbour][2] += scale(z_diff, max_mag)

        elif(metrics[region][1] < sel): # Selectivity is performing worse
            x_diff = max_ind[0] - isocenters[region][0]
            y_diff = max_ind[1] - isocenters[region][1]
            z_diff = max_ind[2] - isocenters[region][2]

            abs_x = abs(x_diff)
            abs_y = abs(y_diff)
            abs_z = abs(z_diff)
            max_mag = max(abs_x, abs_y, abs_z)

            def scale(mag, max_mag):
                if max_mag == 0:
                    return 0
                return int(1 + 4 * (mag/max_mag))
            
            isocenters[region][0] += scale(x_diff, max_mag)
            isocenters[region][1] += scale(y_diff, max_mag)
            isocenters[region][2] += scale(z_diff, max_mag)
    return isocenters


if __name__ == '__main__':
    patient_path = "D:\Summer 24\Research Materials\Gamma Knife Code\GK - IsocenterProject Data\C0006"

    ptv0 = pd.read_csv(patient_path + '\\' + 'PTV0.csv', header=None)
    ptv0 = ptv0.values.tolist()
    min_dose = float(ptv0[2][1])

    # Build the prescription dose from multiple tumour kernels
    pres_dose = build_dense(pd.read_csv(os.path.join(patient_path, "PTV0.csv")))
    i = 1
    while(os.path.isfile(os.path.join(patient_path, "PTV" + str(i) + ".csv"))):
        new_mask = build_dense(pd.read_csv(os.path.join(patient_path, "PTV" + str(i) + ".csv")))
        for i in range(512):
            for j in range(512):
                for k in range(256):
                    pres_dose[i, j, k] += new_mask[i, j, k]
        i += 1

    order = 3
    target_num = 100

    mask = tumour_mask(pres_dose)
    dose_arr, max_ind = develop_dose_cloud(patient_path)
    isocenters = extract_isocenters(patient_path + '\\' + 'adj_isocenters.csv')
    # Clinical coverage, selectivity, and conformity to compare later trials to
    clinical_cov, clinical_sel, clinical_con = compute_sc(dose_arr, pres_dose, min_dose)    
    

    points = filter_points(dose_arr, order, mask, target_num)
    clusters = dbscan_get_clusters(points, 3, 2)
    cluster_list = []
    for key in clusters.keys():
        cluster_list.append(clusters[key])
    
    updated_isocenters = isocenter_set(dose_arr, cluster_list)
    shift_kernel(patient_path, updated_isocenters, 'C0006')
    optimize(patient_path, updated_isocenters, 'C0006')
    updated_dose_arr, max_ind = develop_dose_cloud(patient_path)
    cov, sel, con = compute_sc(updated_dose_arr, pres_dose, min_dose)
    best_cov, best_sel, best_con = clinical_cov, clinical_sel, clinical_con # Best coverage, selectivity, and conformity that all belongs 

    # Conditions that we want fulfilled when stopping
    cond1 = cov >= 0.95
    cond2 = cov >= clinical_cov
    cond3 = sel >= clinical_sel
    cond4 = con >= clinical_con
    cond5 = cov >= best_cov
    cond6 = sel >= best_sel
    cond7 = con >= best_con

    cond = [cond1, cond2, cond3, cond4, cond5, cond6, cond7]

    # The stopping condition for this while loop is that there are at least 4 conditions out of 7 that are true
    while(sum(cond) < 4):
        result, r = assign_voxels(mask, isocenters)
        updated_isocenters = shift_isocenter(updated_isocenters, r, updated_dose_arr, mask, min_dose, cov, sel, con, max_ind)
        shift_kernel(patient_path, updated_isocenters, 'C0006')
        optimize(patient_path, updated_isocenters, 'C0006')
        updated_dose_arr, max_ind = develop_dose_cloud(patient_path)
        cov, sel, con = compute_sc(updated_dose_arr, pres_dose, min_dose)
        if(con > best_con):
            best_cov = cov
            best_sel = sel
            best_con = con

        cond1 = cov >= 0.95
        cond2 = cov >= clinical_cov
        cond3 = sel >= clinical_sel
        cond4 = con >= clinical_con
        cond5 = cov >= best_cov
        cond6 = sel >= best_sel
        cond7 = con >= best_con

        cond = [cond1, cond2, cond3, cond4, cond5, cond6, cond7]
        print(f"{sum(cond)} out of 7 conditions are true")
