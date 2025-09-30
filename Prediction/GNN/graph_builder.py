#!/usr/bin/env python

import json
import torch
import numpy as np
import logging
import gzip
from dgl import DGLGraph

from torch import Tensor
from sklearn.neighbors import NearestNeighbors

from typing import List
import numpy as np
import torch
import dgl
from torch import Tensor, testing
from dgl.convert import heterograph
import math

def get_edge_len(edge_src: Tensor, edge_dst: Tensor, axis: int = 1):
    """returns the length of the edge

    Parameters
    ----------
    edge_src : Tensor
        Tensor of shape (N, 3) containing the source of the edge
    edge_dst : Tensor
        Tensor of shape (N, 3) containing the destination of the edge
    axis : int, optional
        Axis along which the norm is computed, by default 1

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the length of the edge
    """
    return np.linalg.norm(edge_src - edge_dst, axis=axis)


def cell_to_adj(cells: List[List[int]]):
    """creates adjancy matrix in COO format from mesh cells

    Parameters
    ----------
    cells : List[List[int]]
        List of cells, each cell is a list of 3 vertices

    Returns
    -------
    src, dst : List[int], List[int]
        List of source and destination vertices
    """
    num_cells = np.shape(cells)[0]
    src = [cells[i][indx] for i in range(num_cells) for indx in [0, 1, 2]]
    dst = [cells[i][indx] for i in range(num_cells) for indx in [1, 2, 0]]
    return src, dst


def create_graph(
    src: List,
    dst: List,
    to_bidirected: bool = True,
    add_self_loop: bool = False,
    dtype: torch.dtype = torch.int32,
) -> dgl.DGLGraph:
    """
    Creates a DGL graph from an adj matrix in COO format.

    Parameters
    ----------
    src : List
        List of source nodes
    dst : List
        List of destination nodes
    to_bidirected : bool, optional
        Whether to make the graph bidirectional, by default True
    add_self_loop : bool, optional
        Whether to add self loop to the graph, by default False
    dtype : torch.dtype, optional
        Graph index data type, by default torch.int32

    Returns
    -------
    DGLGraph
        The dgl Graph.
    """
    graph = dgl.graph((src, dst), idtype=dtype)
    if to_bidirected:
        graph = dgl.to_bidirected(graph)
    if add_self_loop:
        graph = dgl.add_self_loop(graph)
    return graph


def create_heterograph(
    src: List, dst: List, labels: str, dtype: torch.dtype = torch.int32
) -> dgl.DGLGraph:
    """Creates a heterogeneous DGL graph from an adj matrix in COO format.

    Parameters
    ----------
    src : List
        List of source nodes
    dst : List
        List of destination nodes
    labels : str
        Label of the edge type
    dtype : torch.dtype, optional
        Graph index data type, by default torch.int32

    Returns
    -------
    DGLGraph
        The dgl Graph.
    """
    graph = heterograph({labels: ("coo", (src, dst))}, idtype=dtype)
    return graph


def add_edge_features(
    graph: dgl.DGLGraph, pos: Tensor, normalize: bool = True
) -> dgl.DGLGraph:
    """Adds edge features to the graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph to add edge features to.
    pos : Tensor
        The node positions.
    normalize : bool, optional
        Whether to normalize the edge features, by default True

    Returns
    -------
    DGLGraph
        The graph with edge features.
    """

    if isinstance(pos, tuple):
        src_pos, dst_pos = pos
    else:
        src_pos = dst_pos = pos
    src, dst = graph.edges()

    src_pos, dst_pos = src_pos[src.long()], dst_pos[dst.long()]
    dst_latlon = xyz2latlon(dst_pos, unit="rad")
    dst_lat, dst_lon = dst_latlon[:, 0], dst_latlon[:, 1]

    # azimuthal & polar rotation
    theta_azimuthal = azimuthal_angle(dst_lon)
    theta_polar = polar_angle(dst_lat)

    src_pos = geospatial_rotation(src_pos, theta=theta_azimuthal, axis="z", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_azimuthal, axis="z", unit="rad")
    # y values should be zero
    if src_pos.ndim == 1:
        src_pos = src_pos[None, :]
        dst_pos = dst_pos[None, :]
    try:
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
    except ValueError:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")
    
    src_pos = geospatial_rotation(src_pos, theta=theta_polar, axis="y", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_polar, axis="y", unit="rad")
    # x values should be one, y & z values should be zero
    if src_pos.ndim == 1:
        src_pos = src_pos[None, :]
        dst_pos = dst_pos[None, :]
    try:
        testing.assert_close(dst_pos[:, 0], torch.ones_like(dst_pos[:, 0]))
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
        testing.assert_close(dst_pos[:, 2], torch.zeros_like(dst_pos[:, 2]))
    except ValueError:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")

    # prepare edge features
    disp = src_pos - dst_pos
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)

    # normalize using the longest edge
    if normalize:
        max_disp_norm = torch.max(disp_norm)
        graph.edata["x"] = torch.cat(
            (disp / max_disp_norm, disp_norm / max_disp_norm), dim=-1
        )
    else:
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)
    return graph

def add_node_features(graph: dgl.DGLGraph, pos: Tensor) -> dgl.DGLGraph:
    """Adds cosine of latitude, sine and cosine of longitude as the node features
    to the graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph to add node features to.
    pos : Tensor
        The node positions.

    Returns
    -------
    graph : DGLGraph
        The graph with node features.
    """
    latlon = xyz2latlon(pos)
    lat, lon = latlon[:, 0], latlon[:, 1]
    graph.ndata["x"] = torch.stack(
        (torch.cos(lat), torch.sin(lon), torch.cos(lon)), dim=-1
    )
    return graph


def latlon2xyz(latlon: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """
    Converts latlon in degrees to xyz
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.

    Parameters
    ----------
    latlon : Tensor
        Tensor of shape (N, 2) containing latitudes and longitudes
    radius : float, optional
        Radius of the sphere, by default 1
    unit : str, optional
        Unit of the latlon, by default "deg"

    Returns
    -------
    Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    """
    if unit == "deg":
        latlon = deg2rad(latlon)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")
    lat, lon = latlon[:, 0], latlon[:, 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack((x, y, z), dim=1)

def latlon_points_to_xyz(latlon: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
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
        latlon = torch.deg2rad(latlon)  # Convertit en radians
    elif unit != "rad":
        raise ValueError("Unit doit être 'deg' ou 'rad'.")

    lat, lon = latlon[:, 0], latlon[:, 1]

    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)

    return torch.stack((x, y, z), dim=1)

def xyz2latlon(xyz: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """
    Converts xyz to latlon in degrees
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.

    Parameters
    ----------
    xyz : Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    radius : float, optional
        Radius of the sphere, by default 1
    unit : str, optional
        Unit of the latlon, by default "deg"

    Returns
    -------
    Tensor
        Tensor of shape (N, 2) containing latitudes and longitudes
    """
    lat = torch.arcsin(xyz[:, 2] / radius)
    lon = torch.arctan2(xyz[:, 1], xyz[:, 0])
    if unit == "deg":
        return torch.stack((rad2deg(lat), rad2deg(lon)), dim=1)
    elif unit == "rad":
        return torch.stack((lat, lon), dim=1)
    else:
        raise ValueError("Not a valid unit")


def deg2rad(deg: Tensor) -> Tensor:
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


def rad2deg(rad):
    """Converts radians to degrees

    Parameters
    ----------
    rad :
        Tensor of shape (N, ) containing the radians

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the degrees
    """
    return rad * 180 / np.pi


def geospatial_rotation(
    invar: Tensor, theta: Tensor, axis: str, unit: str = "rad"
) -> Tensor:
    """Rotation using right hand rule

    Parameters
    ----------
    invar : Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    theta : Tensor
        Tensor of shape (N, ) containing the rotation angle
    axis : str
        Axis of rotation
    unit : str, optional
        Unit of the theta, by default "rad"

    Returns
    -------
    Tensor
        Tensor of shape (N, 3) containing the rotated x, y, z coordinates
    """

    # get the right unit
    if unit == "deg":
        invar = rad2deg(invar)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")

    invar = torch.unsqueeze(invar, -1)
    rotation = torch.zeros((theta.size(0), 3, 3))
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    if axis == "x":
        rotation[:, 0, 0] += 1.0
        rotation[:, 1, 1] += cos
        rotation[:, 1, 2] -= sin
        rotation[:, 2, 1] += sin
        rotation[:, 2, 2] += cos
    elif axis == "y":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 2] += sin
        rotation[:, 1, 1] += 1.0
        rotation[:, 2, 0] -= sin
        rotation[:, 2, 2] += cos
    elif axis == "z":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 1] -= sin
        rotation[:, 1, 0] += sin
        rotation[:, 1, 1] += cos
        rotation[:, 2, 2] += 1.0
    else:
        raise ValueError("Invalid axis")

    outvar = torch.matmul(rotation, invar)
    outvar = outvar.squeeze()
    return outvar


def azimuthal_angle(lon: Tensor) -> Tensor:
    """
    Gives the azimuthal angle of a point on the sphere

    Parameters
    ----------
    lon : Tensor
        Tensor of shape (N, ) containing the longitude of the point

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the azimuthal angle
    """
    angle = torch.where(lon >= 0.0, 2 * np.pi - lon, -lon)
    return angle


def polar_angle(lat: Tensor) -> Tensor:
    """
    Gives the polar angle of a point on the sphere

    Parameters
    ----------
    lat : Tensor
        Tensor of shape (N, ) containing the latitude of the point

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the polar angle
    """
    angle = torch.where(lat >= 0.0, lat, 2 * np.pi + lat)
    return angle


def grid_cell_area(lat: Tensor, unit="deg") -> Tensor:
    """Normalized area of the latitude-longitude grid cell"""
    if unit == "deg":
        lat = deg2rad(lat)
    area = torch.abs(torch.cos(lat))
    return area / torch.mean(area)

class GraphBuilder:
    def __init__(
        self, icospheres_graph_path, lat_lon_grid: Tensor, dtype=torch.float, doPrint=True
    ) -> None:
        self.dtype = dtype
        self.doPrint = doPrint  # Ajout de l'attribut doPrint

        if icospheres_graph_path.endswith(".gz"):
            with gzip.open(icospheres_graph_path, "rt") as f:
                loaded_dict = json.load(f)
        else:
            with open(icospheres_graph_path, "r") as f:
                loaded_dict = json.load(f)

        icospheres = {
            key: (np.array(value) if isinstance(value, list) else value)
            for key, value in loaded_dict.items()
        }
        if self.doPrint:
            print(f"Opened pre-computed graph from {icospheres_graph_path}.")

        self.icospheres = icospheres
        self.max_order = (
            len([key for key in self.icospheres.keys() if "faces" in key]) - 2
        )
        if self.doPrint:
            print("Will use max_order={} icospheres".format(self.max_order))

        # Flatten lat/lon grid
        self.lat_lon_grid_flat = lat_lon_grid.view(-1, 2)
            # --- dans GraphBuilder.__init__ ---
        self._g2m_src = None  # grid indices (numpy int64)
        self._g2m_dst = None  # mesh indices (numpy int64)

    def create_mesh_graph(self, last_graph=None) -> DGLGraph:
        if self.doPrint:
            print("Creating bi-directional mesh graph")

        multimesh_faces = self.icospheres["order_0_faces"]
        """for i in range(1, self.max_order + 1):
            multimesh_faces = np.concatenate(
                (multimesh_faces, self.icospheres["order_" + str(i) + "_faces"])
            )"""

        multimesh_faces = self.icospheres[f"order_{self.max_order}_faces"]
        src, dst = cell_to_adj(multimesh_faces)
        src = np.asarray(src)
        dst = np.asarray(dst)
        #if last_graph is not None:
        #    num_nodes = last_graph.num_nodes()
        #    src += num_nodes
        #    dst += num_nodes

        mesh_graph = create_graph(
            src, dst, to_bidirected=True, add_self_loop=False, dtype=torch.int32
        )
        mesh_pos = torch.tensor(
            self.icospheres["order_" + str(self.max_order) + "_vertices"],
            dtype=torch.float32,
        )

        mesh_graph = add_edge_features(mesh_graph, mesh_pos)
        mesh_graph = add_node_features(mesh_graph, mesh_pos)

        mesh_graph.ndata["x"] = mesh_graph.ndata["x"].to(dtype=self.dtype)
        mesh_graph.edata["x"] = mesh_graph.edata["x"].to(dtype=self.dtype)

        if self.doPrint:
            print("mesh graph={}".format(mesh_graph))
        
        return mesh_graph

    def create_g2m_graph_old(self, last_graph=None, mesh_graph=None) -> DGLGraph:
        if self.doPrint:
            print("Creating grid2mesh bipartite graph")

        edge_len = max([
            np.max(get_edge_len(
                self.icospheres["order_" + str(self.max_order) + "_vertices"][self.icospheres["order_" + str(self.max_order) + "_faces"][:, i]],
                self.icospheres["order_" + str(self.max_order) + "_vertices"][self.icospheres["order_" + str(self.max_order) + "_faces"][:, j]]
            ))
            for i, j in [(0, 1), (0, 2), (1, 2)]
        ])

        if self.doPrint:
            print("Found max edge length = {}".format(edge_len))

        cartesian_grid = latlon_points_to_xyz(self.lat_lon_grid_flat)
        n_nbrs = 4
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(
            self.icospheres["order_" + str(self.max_order) + "_vertices"]
        )
        distances, indices = neighbors.kneighbors(cartesian_grid)

        src, dst = [], []
        for i in range(len(cartesian_grid)):
            for j in range(n_nbrs):
                if distances[i][j] <= 0.6 * edge_len:
                    src.append(i)
                    dst.append(indices[i][j])
        
        node_max = np.max(dst) + 1

        vertices_mesh = torch.tensor(
            self.icospheres["order_" + str(self.max_order) + "_vertices"],
            dtype=torch.float32,
        )[:node_max]

        self.node_max = node_max

        src = np.asarray(src)
        dst = np.asarray(dst)

        #print(f'src create_g2m_graph -> {src}')
        #print(f'dst create_g2m_graph -> {dst}')

        #if last_graph is not None:
        #    num_nodes = last_graph.num_nodes('grid')
        #    src += num_nodes
        #    num_nodes = last_graph.num_nodes('mash')
        #    dst += num_nodes

        if np.max(dst) + 1 != len(self.icospheres["order_" + str(self.max_order) + "_vertices"]) and mesh_graph is not None:
            mesh_graph = mesh_graph.subgraph(np.arange(0, np.max(dst) + 1))
            if self.doPrint:
                print("mesh graph={}".format(mesh_graph))

        g2m_graph = create_heterograph(
            src, dst, ("grid", "g2m", "mesh"), dtype=torch.int32
        )

        g2m_graph.srcdata["pos"] = cartesian_grid.to(torch.float32)
        g2m_graph.dstdata["pos"] = vertices_mesh

        g2m_graph = add_edge_features(
            g2m_graph, (g2m_graph.srcdata["pos"], g2m_graph.dstdata["pos"])
        )

        g2m_graph.srcdata["pos"] = g2m_graph.srcdata["pos"].to(dtype=self.dtype)
        g2m_graph.dstdata["pos"] = g2m_graph.dstdata["pos"].to(dtype=self.dtype)
        g2m_graph.ndata["pos"]["grid"] = g2m_graph.ndata["pos"]["grid"].to(dtype=self.dtype)
        g2m_graph.ndata["pos"]["mesh"] = g2m_graph.ndata["pos"]["mesh"].to(dtype=self.dtype)
        g2m_graph.edata["x"] = g2m_graph.edata["x"].to(dtype=self.dtype)

        if self.doPrint:
            print("grid2mesh bipartite graph={}".format(g2m_graph))

        return (g2m_graph, mesh_graph) if mesh_graph is not None else g2m_graph

    def create_m2g_graph_old(self, last_graph=None) -> DGLGraph:
        if self.doPrint:
            print("Creating mesh2grid bipartite graph")

        #print(self.lat_lon_grid_flat.shape)
        #print(np.unique(self.lat_lon_grid_flat, axis=0).shape)
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        n_nbrs = 1
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(
            self.icospheres["order_" + str(self.max_order) + "_face_centroid"]
        )
        _, indices = neighbors.kneighbors(cartesian_grid)
        indices = indices.flatten()

        src = [
            p for i in indices
            for p in self.icospheres["order_" + str(self.max_order) + "_faces"][i]
        ]
        dst = [i for i in range(len(cartesian_grid)) for _ in range(3)]

        vertices_mesh = torch.tensor(
            self.icospheres["order_" + str(self.max_order) + "_vertices"],
            dtype=torch.float32,
        )[:self.node_max]

        #src = src[:self.node_max]
        dst = [d for i, d in enumerate(dst) if src[i] < self.node_max]
        src = [s for s in src if s < self.node_max]
        src = np.asarray(src)
        dst = np.asarray(dst)

        #print(f'src create_m2g_graph -> {src}')
        #print(f'dst create_m2g_graph -> {dst}')

        #cartesian_grid = cartesian_grid[np.sort(np.unique(dst))]

        #if last_graph is not None:
        #    num_nodes = last_graph.num_nodes('mesh')
        #    src += num_nodes
        #    num_nodes = last_graph.num_nodes('grid')
        #    dst += num_nodes

        m2g_graph = create_heterograph(
            src, dst, ("mesh", "m2g", "grid"), dtype=torch.int32
        )

        m2g_graph.srcdata["pos"] = vertices_mesh
        m2g_graph.dstdata["pos"] = cartesian_grid.to(dtype=torch.float32)

        m2g_graph = add_edge_features(
            m2g_graph, (m2g_graph.srcdata["pos"], m2g_graph.dstdata["pos"])
        )

        m2g_graph.srcdata["pos"] = m2g_graph.srcdata["pos"].to(dtype=self.dtype)
        m2g_graph.dstdata["pos"] = m2g_graph.dstdata["pos"].to(dtype=self.dtype)
        m2g_graph.ndata["pos"]["grid"] = m2g_graph.ndata["pos"]["grid"].to(dtype=self.dtype)
        m2g_graph.ndata["pos"]["mesh"] = m2g_graph.ndata["pos"]["mesh"].to(dtype=self.dtype)
        m2g_graph.edata["x"] = m2g_graph.edata["x"].to(dtype=self.dtype)

        if self.doPrint:
            print("mesh2grid bipartite graph={}".format(m2g_graph))

        return m2g_graph

    def _grid_to_face_edges(self):
        """Construit les arêtes grid->mesh via les 4 sommets (vertices) les plus proches,
        en ne conservant que ceux à distance <= 0.6 * edge_len.
        Retourne (src, dst, node_max, cartesian_grid)."""
        import numpy as np
        from sklearn.neighbors import NearestNeighbors

        # Données du maillage
        faces = self.icospheres[f"order_{self.max_order}_faces"]             # (F, 3)
        vertices = self.icospheres[f"order_{self.max_order}_vertices"]       # (V, 3)
        cartesian_grid = latlon_points_to_xyz(self.lat_lon_grid_flat)        # torch.Tensor (G, 3)
        
        # Longueur d'arête max (comme dans create_g2m_graph) pour fixer le seuil relatif
        edge_len = max([
            np.max(get_edge_len(vertices[faces[:, i]], vertices[faces[:, j]]))
            for i, j in [(0, 1), (0, 2), (1, 2)]
        ])
        thresh = 0.6 * edge_len

        # NN sur les vertices (4 plus proches)
        # -> scikit-learn attend du numpy
        cart_np = cartesian_grid.detach().cpu().numpy() if hasattr(cartesian_grid, "detach") else np.asarray(cartesian_grid)
        nn = NearestNeighbors(n_neighbors=4).fit(vertices)   # vertices est déjà un np.ndarray
        distances, indices = nn.kneighbors(cart_np)          # shapes: (G, 4)

        # Construit les arêtes en appliquant le seuil
        src, dst = [], []
        G = cart_np.shape[0]
        for g in range(G):
            keep = indices[g][distances[g] <= thresh]
            if keep.size > 0:
                src.extend([g] * keep.size)
                dst.extend(keep.tolist())
            else:
                src.append(g)
                dst.append(int(indices[g][0]))

        # Sorties au format attendu
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64) if len(dst) > 0 else np.empty((0,), dtype=np.int64)
        node_max = int(dst.max()) + 1 if dst.size > 0 else 0

        return src, dst, node_max, cartesian_grid

    def create_g2m_graph(self, last_graph=None, mesh_graph=None) -> DGLGraph:
        if self.doPrint:
            print("Creating grid2mesh bipartite graph (face-based, 3 edges/grid)")

        # 1) arêtes via face la plus proche
        src, dst, node_max, cartesian_grid = self._grid_to_face_edges()
        self._g2m_src, self._g2m_dst = src, dst
        self.node_max = node_max

        # 2) optionnel: restreindre le mesh_graph si fourni
        if (np.max(dst) + 1 != len(self.icospheres[f"order_{self.max_order}_vertices"])) and mesh_graph is not None:
            mesh_graph = mesh_graph.subgraph(np.arange(0, np.max(dst) + 1))
            if self.doPrint:
                print("mesh graph={}".format(mesh_graph))

        # 3) construction du hétérographe
        g2m_graph = create_heterograph(src, dst, ("grid", "g2m", "mesh"), dtype=torch.int32)

        vertices_mesh = torch.tensor(
            self.icospheres[f"order_{self.max_order}_vertices"], dtype=torch.float32
        )[:node_max]

        g2m_graph.srcdata["pos"] = cartesian_grid.to(torch.float32)
        g2m_graph.dstdata["pos"] = vertices_mesh

        g2m_graph = add_edge_features(
            g2m_graph, (g2m_graph.srcdata["pos"], g2m_graph.dstdata["pos"])
        )

        # cast dtype
        #g2m_graph.srcdata["pos"] = g2m_graph.srcdata["_grid_to_face_edgespos"].to(dtype=self.dtype)
        g2m_graph.dstdata["pos"] = g2m_graph.dstdata["pos"].to(dtype=self.dtype)
        g2m_graph.ndata["pos"]["grid"] = g2m_graph.ndata["pos"]["grid"].to(dtype=self.dtype)
        g2m_graph.ndata["pos"]["mesh"] = g2m_graph.ndata["pos"]["mesh"].to(dtype=self.dtype)
        g2m_graph.edata["x"] = g2m_graph.edata["x"].to(dtype=self.dtype)

        if self.doPrint:
            print("grid2mesh bipartite graph={}".format(g2m_graph))

        return (g2m_graph, mesh_graph) if mesh_graph is not None else g2m_graph

    def create_m2g_graph(self, last_graph=None) -> DGLGraph:
        if self.doPrint:
            print("Creating mesh2grid bipartite graph (derived from g2m edges)")

        # 1) si on a déjà construit g2m : on réutilise EXACTEMENT les mêmes arêtes, inversées
        if self._g2m_src is None or self._g2m_dst is None:
            # fallback : calcule les mêmes arêtes que g2m (face-based), puis inverse
            src_g2m, dst_g2m, node_max, cartesian_grid = self._grid_to_face_edges()
            self._g2m_src, self._g2m_dst = src_g2m, dst_g2m
            self.node_max = node_max
        else:
            cartesian_grid = latlon_points_to_xyz(self.lat_lon_grid_flat)

        # arêtes inversées : mesh -> grid
        src = self._g2m_dst.copy()
        dst = self._g2m_src.copy()

        m2g_graph = create_heterograph(src, dst, ("mesh", "m2g", "grid"), dtype=torch.int32)

        vertices_mesh = torch.tensor(
            self.icospheres[f"order_{self.max_order}_vertices"], dtype=torch.float32
        )[:self.node_max]

        m2g_graph.srcdata["pos"] = vertices_mesh
        m2g_graph.dstdata["pos"] = cartesian_grid.to(dtype=torch.float32)

        m2g_graph = add_edge_features(
            m2g_graph, (m2g_graph.srcdata["pos"], m2g_graph.dstdata["pos"])
        )

        # cast dtype
        m2g_graph.srcdata["pos"] = m2g_graph.srcdata["pos"].to(dtype=self.dtype)
        m2g_graph.dstdata["pos"] = m2g_graph.dstdata["pos"].to(dtype=self.dtype)
        m2g_graph.ndata["pos"]["grid"] = m2g_graph.ndata["pos"]["grid"].to(dtype=self.dtype)
        m2g_graph.ndata["pos"]["mesh"] = m2g_graph.ndata["pos"]["mesh"].to(dtype=self.dtype)
        m2g_graph.edata["x"] = m2g_graph.edata["x"].to(dtype=self.dtype)

        if self.doPrint:
            print("mesh2grid bipartite graph={}".format(m2g_graph))

        return m2g_graph


class GraphBuilder2:
    def __init__(self, g_lat_lon_grid_scales, Y_scales, graph_scales, date_index, id_index) -> None:
        self.g_lat_lon_grid_scales = g_lat_lon_grid_scales
        self.graph_scales = graph_scales
        self.Y_scales = Y_scales
        self.date_index = date_index
        self.id_index = id_index

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0  # Earth radius
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _add_node_and_edge_features(self, hetero_graph, edge_type, src_pos_list, tgt_pos_list, src, dst):
        src_pos_tensor = torch.tensor(src_pos_list, dtype=torch.float32)
        tgt_pos_tensor = torch.tensor(tgt_pos_list, dtype=torch.float32)

        #print(hetero_graph)
        #print(src_pos_tensor, tgt_pos_tensor)
        # Assign node positions
        #hetero_graph.nodes[edge_type[0]].data["pos"] = src_pos_tensor
        #hetero_graph.nodes[edge_type[2]].data["pos"] = tgt_pos_tensor

        # Compute directional edge vectors
        """src_vecs = src_pos_tensor[torch.tensor(src)]
        tgt_vecs = tgt_pos_tensor[torch.tensor(dst)]
        disp = tgt_vecs - src_vecs
        norm = torch.linalg.norm(disp, dim=1, keepdim=True)
        norm[norm == 0] = 1e-6  # avoid division by zero
        edge_features = torch.cat([disp / norm, norm], dim=1)  # (dx, dy, norm)"""

        num_edges = hetero_graph.num_edges(edge_type)
        hetero_graph.edges[edge_type].data["x"] = torch.ones(num_edges, 1)

    def graph_scale(self):
        graph_list = []
        for i, Y in enumerate(self.Y_scales):
            spatialEdges = self.graph_scales[i].edges
            graph_meta = self.graph_scales[i]
            scale_src = scale_dst = graph_meta.scale  # same scale

            src = []
            dst = []

            for j, node in enumerate(Y):
                spatialNodes = Y[np.argwhere((Y[:, self.date_index, -1] == node[self.date_index][-1]))][:,:, 0, 0]
                if spatialEdges.shape[1] != 0:
                    spatial = spatialEdges[1][
                        (np.isin(spatialEdges[1], spatialNodes[:, 0])) &
                        (spatialEdges[0] == node[self.id_index][0])
                    ]
                    for sp in spatial:
                        target_idx = np.argwhere(
                            (Y[:, self.date_index, -1] == node[self.date_index, -1]) &
                            (Y[:, self.id_index, 0] == sp)
                        )[0][0]
                        src.append(j)
                        dst.append(target_idx)

            edge_type = (f"scale_{scale_src}", "coo", f"scale_{scale_dst}")
            data_dict = {edge_type: (torch.tensor(src, dtype=torch.int32), torch.tensor(dst, dtype=torch.int32))}
            hetero_graph = heterograph(data_dict)

            # Positions (same for src and dst)
            latlon = self.g_lat_lon_grid_scales[i]
            self._add_node_and_edge_features(hetero_graph, edge_type, latlon, latlon, src, dst)
            graph_list.append(hetero_graph)

        return graph_list

    def increase_scale(self, graph_scale_list):
        graph_list = []
        updated_graph_scale_list = []

        for i in range(len(self.Y_scales) - 1):
            src_lat_lon = self.g_lat_lon_grid_scales[i]
            tgt_lat_lon = self.g_lat_lon_grid_scales[i + 1]

            graph_src = self.graph_scales[i]
            graph_target = self.graph_scales[i + 1]
            scale_src = graph_src.scale
            scale_dst = graph_target.scale

            src = []
            dst = []
            for src_idx, (src_lat, src_lon) in enumerate(src_lat_lon):
                for tgt_idx, (tgt_lat, tgt_lon) in enumerate(tgt_lat_lon):
                    #if self.haversine_distance(src_lat, src_lon, tgt_lat, tgt_lon) < 50:
                        src.append(src_idx)
                        dst.append(tgt_idx)

            if len(src) == 0:
                return None, None
            edge_type = (f"scale_{scale_src}", "coo", f"scale_{scale_dst}")
            data_dict = {edge_type: (torch.tensor(src, dtype=torch.int32), torch.tensor(dst, dtype=torch.int32))}
            hetero_graph = heterograph(data_dict)

            # Ajout des features
            self._add_node_and_edge_features(hetero_graph, edge_type, src_lat_lon, tgt_lat_lon, src, dst)
            graph_list.append(hetero_graph)

            # --- Filtrage des graphes sources & cibles selon les noeuds utilisés
            used_src_nodes = np.unique(src)
            used_dst_nodes = np.unique(dst)

            filtered_src_graph = dgl.node_subgraph(graph_scale_list[i], used_src_nodes)
            filtered_dst_graph = dgl.node_subgraph(graph_scale_list[i + 1], used_dst_nodes)

            # On ajoute à la liste mise à jour (attention à l'ordre)
            if i == 0:
                updated_graph_scale_list.append(filtered_src_graph)
            updated_graph_scale_list.append(filtered_dst_graph)

        return graph_list, updated_graph_scale_list
    
    def decrease_scale(self):
        graph_list = []
        for i in range(len(self.Y_scales) - 1, 0, -1):
            src_lat_lon = self.g_lat_lon_grid_scales[i]
            tgt_lat_lon = self.g_lat_lon_grid_scales[i - 1]

            graph_src = self.graph_scales[i]
            graph_target = self.graph_scales[i - 1]

            scale_src = graph_src.scale
            scale_dst = graph_target.scale

            src = []
            dst = []

            for src_idx, (src_lat, src_lon) in enumerate(src_lat_lon):
                for tgt_idx, (tgt_lat, tgt_lon) in enumerate(tgt_lat_lon):
                    #if self.haversine_distance(src_lat, src_lon, tgt_lat, tgt_lon) < 50:
                        src.append(src_idx)
                        dst.append(tgt_idx)

            edge_type = (f"scale_{scale_src}", "coo", f"scale_{scale_dst}")
            data_dict = {edge_type: (torch.tensor(src, dtype=torch.int32), torch.tensor(dst, dtype=torch.int32))}
            hetero_graph = heterograph(data_dict)

            self._add_node_and_edge_features(hetero_graph, edge_type, src_lat_lon[src], tgt_lat_lon[dst], src, dst)
            graph_list.append(hetero_graph)

        return graph_list