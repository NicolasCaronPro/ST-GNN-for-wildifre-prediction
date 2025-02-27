import pickle
import json
import numpy as np
from pathlib import Path

def deg2rad(deg):
    """Converts degrees to radians

    Parameters
    ----------
    deg :
        Tensor of shape (N, ) containing the degrees

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the radians
    """
    return deg * np.pi / 180

def latlon_points_to_xyz(latlon, radius: float = 1, unit: str = "deg"):
    """
    Convertit un ensemble de points latitude-longitude en coordonnées cartésiennes (x, y, z).
    
    - Le x-axis passe par (long, lat) = (0,0).
    - Le y-axis passe par (0,90) (pôle nord).
    - Le z-axis correspond à l'axe des pôles.

    Paramètres
    ----------
    latlon : Tensor
        Tensor de shape (N, 2) contenant les latitudes et longitudes des points.
    radius : float, optional
        Rayon de la sphère, par défaut 1.
    unit : str, optional
        Unité de lat/lon ("deg" pour degrés, "rad" pour radians), par défaut "deg".

    Retour
    ------
    Tensor
        Tensor de shape (N, 3) contenant les coordonnées cartésiennes (x, y, z).
    """
    if latlon.shape[-1] != 2:
        raise ValueError("Le tensor d'entrée doit avoir une shape (N, 2) pour lat/lon.")

    if unit == "deg":
        latlon = np.deg2rad(latlon)  # Convertit en radians
    elif unit != "rad":
        raise ValueError("Unit doit être 'deg' ou 'rad'.")

    lat, lon = latlon[:, 0], latlon[:, 1]

    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)

    return np.stack((x, y, z), axis=1)

def generate_and_save_multi_graph(save_path : str, input_path, levels=[0,1,2,3,4,5,6], base='risk-size-watershed', graph_method='node'):
    icospheres = {"vertices": [], "faces": []}
    cur = 0
    for order in levels:
        #icosphere = pymesh.generate_icosphere(radius, center, refinement_order=order)

        if order == 'departement':
            base = 'None'

        prefix = f'graph_{order}_{base}_{graph_method}'
        graphs = pickle.load(open(input_path / f'{prefix}.pkl', 'rb'))
        nodes = graphs.nodes
        latitudes = nodes[:, 3]
        longitudes = nodes[:, 2]
        g_lat_lon_grid = np.asarray([latitudes, longitudes])
        g_lat_lon_grid = np.moveaxis(g_lat_lon_grid, 0, 1)
        cartesian_grid = latlon_points_to_xyz(g_lat_lon_grid)

        icospheres["order_" + str(cur) + "_vertices"] = cartesian_grid
        icospheres["order_" + str(cur) + "_faces"] = nodes[:, 0]
        icospheres["order_" + str(cur) + "_face_centroid"] = nodes[:, 0]

        cur += 1

    # save icosphere vertices and faces to a json file
    icospheres_dict = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in icospheres.items()
    }
    with open(f'{save_path}_graph_{cur}.json', "w") as f:
        json.dump(icospheres_dict, f)

if __name__ == "__main__":
    dataset_name = 'firemen'

    generate_and_save_multi_graph(f'GNN/icospheres/graphs/{dataset_name}', Path(f'GNN/{dataset_name}/firepoint/2x2/train'), [4,5,6,7,'departement'], base='risk-size-watershed', graph_method='node')