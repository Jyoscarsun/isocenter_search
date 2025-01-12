# Isocenter Search Algorithm for Radiation Therapy

This repository contains the implementation of an isocenter search algorithm designed to optimize radiation therapy by strategically positioning radiation isocenters. The project aims to improve the coverage, selectivity, and conformity of radiation dose distributions for targeted tumor treatment.

## Project Overview

In radiation therapy, determining the optimal isocenter positions is crucial for effectively delivering prescribed doses while minimizing exposure to surrounding healthy tissues. This project implements an iterative approach to:

1. Identify high-dose regions in a 3D dose array.
2. Cluster potential isocenters using machine learning techniques (e.g., KMeans, DBSCAN).
3. Refine isocenter positions based on dose coverage and conformity metrics.
4. Optimize kernel durations for final dose calculation.

## Features

- **3D Dose Array Analysis**: Processes dose distributions to identify high-dose regions.
- **Clustering Algorithms**: Utilizes KMeans and DBSCAN to group potential isocenters.
- **Isocenter Refinement**: Iteratively shifts isocenters to improve coverage and selectivity.
- **Visualization**: Graphically represents dose distributions and isocenter placements.
- **Optimization**: Calculates kernel durations to achieve desired dose objectives.

## Requirements

### Python Dependencies

- `pandas`
- `numpy`
- `sparse`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `gurobipy`
- Custom modules: `patient_def`, `utils`, `constants_def`

## Example Workflow
1. Load kernel files and initial isocenter positions.
2. Process the dose array to identify potential isocenters.
3. Cluster the isocenters using KMeans or DBSCAN.
4. Visualize the clustered isocenters.
5. Refine and optimize isocenter positions iteratively.
6. Output the final dose distribution and associated metrics.

## Acknowledgments
- Applied Optimization Lab Members: For providing domain expertise and developing supportive functionalities.
- Gurobi Optimization: For enabling efficient dose optimization.


