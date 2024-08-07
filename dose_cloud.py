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

def find_local_maxima(data, order, numMax, min_dose):
    size = 1 + 2 * order # Side length of the cube
    fp = np.ones((size, size, size)) # Create a 3D matrix named footprint
    fp[order, order, order] = 0 # The center of the footprint cube is set to 0

    # Replaces each element in data array with the maximum of its neighbourhood
    filtered = ndi.maximum_filter(data, footprint=fp)
    mask = data > filtered
    coords = np.asarray(np.where(mask)).T
    values = data[mask]

    local_var = []
    for coord in coords:
        slices = tuple(slice(max(0, c-order), min(s, c+order+1)) for c,s in zip(coord, data.shape))
        neighbor = data[slices]
        var = np.var(neighbor)
        local_var.append(var)

    local_var = np.array(local_var)

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
    
    top_coords = coords[top_ind]
    top_val = values[top_ind]
    top_var = local_var[top_ind]

    return top_coords, top_val, top_var

def dist(coord1, coord2):
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 + (coord1[2]-coord2[2])**2)

def elbow(coords, ub):
    data = list(coords)
    if(ub > len(coords)):
        print("Upper bound cluster is greater than number of data points")
        ub = len(coords)
        if ub == 1:
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

def visualize_clusters(dose_arr, max_ind, clusters, isocenters):
    x, y, z = max_ind
    fig, ax = plt.subplots(figsize=(15, 5))
    cax1 = ax.imshow(dose_arr[:, :, z], cmap='viridis')
    ax.set_title(f'Plane z={z}')
    ax.axis('off')
    fig.colorbar(cax1, ax=ax, orientation='vertical', shrink=0.8)

    colors = ['red', 'green', 'cyan', 'black', 'magenta', 'yellow', 'white', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy']
    i = 1
    for cluster in clusters:
        cluster_proj = np.array([point[:2] for point in cluster])
        print(cluster_proj)
        if cluster_proj.size > 0:
            ax.scatter(cluster_proj[:, 0], cluster_proj[:, 1], c=colors[i-1], s=30, label=f"Cluster {i}")
        i += 1

    isocenter_proj = np.array([point[:2] for point in isocenters])
    ax.scatter(isocenter_proj[:, 0], isocenter_proj[:, 1], c='black', s=10, label="Clinical Isocenters")

    ax.legend(loc='upper right')
    output_file = 'isocenter_cluster_C0006.png'  # Specify the file name and format
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

def extract_isocenters(isocenter_path):
    isocenters = pd.read_csv(isocenter_path, header=None)
    isocenter_list = isocenters.values.tolist()
    return isocenter_list

if __name__ == '__main__':
    # Load and process the 24 kernels
    patient_path = "D:\Summer 24\Research Materials\Gamma Knife Code\C0006"
    # patient_path = "D:\Summer 24\Research Materials\Gamma Knife Code\GK - IsocenterProject Data\C0031"

    files = [f for f in os.listdir(patient_path)
                if (os.path.isfile(os.path.join(patient_path, f)) and 'kernel' in f)]
    files = natsort.natsorted(files)

    all_kernels = [sp.load_npz(os.path.join(patient_path, i)) for i in files]

    # Load the file containing time duration for each kernel
    duration = pd.read_csv(patient_path + '\\' + 'inverse_PTV0.csv', header=None)

    for idx, kernel in enumerate(all_kernels):
        print('kernel', idx)
        if idx == 0:
            dose_kernel = kernel * duration.iloc[0, 0]

        if duration.iloc[idx, 0] == 0.0:
            continue

        if idx != 0:
            kernel = kernel * duration.iloc[idx, 0]
            dose_kernel = dose_kernel + kernel
    
    dose_arr = dose_kernel.todense()
    dose_arr = np.array(dose_arr)

    # Find the index of the maixmum value in the flattened array
    max_ind_flat = np.argmax(dose_arr)

    # Convert the flattened index to a tuple of indices for the original 3D array
    max_ind = np.unravel_index(max_ind_flat, dose_arr.shape)
    print("Index of the maximum value: ", max_ind)
    print("Dosage at the maximum location: ", dose_arr[max_ind])

    print(dose_arr.shape)


    coords, value, var = find_local_maxima(dose_arr, 1, 20, 12.5)
    # print(coords)
    # print(len(coords))
    # print(value)
    # print(var)
    # print(elbow(coords,5))

    num_cluster = elbow(coords,5)
    clusters = kmeans_get_clusters(coords, num_cluster)

    print(f"Cluster 0: {clusters[0]}")
    # print(f"Cluster 1: {clusters[1]}")
    # print(f"Coordinates in the first cluster is: {clusters[0]}")
    # print(len(clusters[0]))
    # print(f"Coordinates in the second cluster is: {clusters[1]}")
    # print(len(clusters[1]))
    isocenters = extract_isocenters(patient_path + '\\' + 'adj_isocenters.csv')
    visualize_clusters(dose_arr, max_ind, clusters, isocenters)

    clusters = dbscan_get_clusters(coords, 50, 5)
    for cluster_label, cluster_points in clusters.items():
        print(f"Cluster {cluster_label}: {cluster_points}")

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
