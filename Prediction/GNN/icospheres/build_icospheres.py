#!/usr/bin/env python

import json
import pickle5 as pickle
import numpy as np
from pathlib import Path
import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    import pymesh
except ImportError:
    Warning("pymesh is not installed. Please install it to use icosphere.")


def generate_and_save_icospheres(
    save_path: str = "icospheres_0_1_2_3_4_5_6.json", levels=[0]
) -> None:  # pragma: no cover
    """Generate icospheres from level 0 to 6 (inclusive) and save them to a json file.

    Parameters
    ----------
    path : str
        Path to save the json file.
    """
    radius = 1
    center = np.array((0, 0, 0))
    icospheres = {"vertices": [], "faces": []}

    # Generate icospheres from level 0 to 6 (inclusive)
    cur = 0
    for order in levels:
        icosphere = pymesh.generate_icosphere(radius, center, refinement_order=order)
        icospheres["order_" + str(cur) + "_vertices"] = icosphere.vertices
        icospheres["order_" + str(cur) + "_faces"] = icosphere.faces
        icosphere.add_attribute("face_centroid")
        icospheres["order_" + str(cur) + "_face_centroid"] = (
            icosphere.get_face_attribute("face_centroid")
        )
        print(icosphere.get_face_attribute('face_centroid'))
        cur += 1

    # save icosphere vertices and faces to a json file
    icospheres_dict = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in icospheres.items()
    }
    with open(save_path, "w") as f:
        json.dump(icospheres_dict, f)

def generate_and_save_multi_graph(save_path : str, input_path, levels=[0,1,2,3,4,5,6], base='risk-size-watershed', graph_method='node'):
    radius = 1
    center = np.array((0, 0, 0))
    icospheres = {"vertices": [], "faces": []}
    cur = 0
    for order in levels:
        #icosphere = pymesh.generate_icosphere(radius, center, refinement_order=order)

        prefix = f'graph_{order}_{base}_{graph_method}'
        graphs = pickle.load(open(input_path / f'{prefix}.pkl', 'rb'))
        nodes = graphs.nodes
        latitudes = nodes[:, 4]
        longitudes = nodes[:, 5]
        g_lat_lon_grid = np.asarray([latitudes, longitudes])
        g_lat_lon_grid.shape
        cartesian_grid = latlon2xyz(g_lat_lon_grid)

        icospheres["order_" + str(cur) + "_vertices"] = cartesian_grid
        icospheres["order_" + str(cur) + "_faces"] = nodes
        cur += 1

    # save icosphere vertices and faces to a json file
    icospheres_dict = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in icospheres.items()
    }
    with open(save_path, "w") as f:
        json.dump(icospheres_dict, f)

if __name__ == "__main__":
    """generate_and_save_icospheres("icospheres_0_1.json", [0, 1])
    generate_and_save_icospheres("icospheres_0_1_2.json", [0, 1, 2])
    generate_and_save_icospheres("icospheres_0_1_2_3.json", [0, 1, 2, 3])
    generate_and_save_icospheres("icospheres_0_1_2_3_4.json", [0, 1, 2, 3, 4])
    generate_and_save_icospheres("icospheres_0_1_2_3_4_5.json", [0, 1, 2, 3, 4, 5])
    generate_and_save_icospheres("icospheres_0_1_2_3_4_5_6.json", [0, 1, 2, 3, 4, 5, 6])
    generate_and_save_icospheres("icospheres_1_3_5.json", [1, 3, 5])
    generate_and_save_icospheres("icospheres_2_3_4.json", [2, 3, 4])
    generate_and_save_icospheres("icospheres_2_3_4_5.json", [2, 3, 4, 5])"""

    dataset_name = 'firemen'

    generate_and_save_multi_graph(f'GNN/icospheres/graphs/{dataset_name}/graph_4.json', Path(f'GNN/{dataset_name}/firepoint/2x2/train'), [4], base='risk-size-watershed', graph_method='node')
