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
    try:
        testing.assert_close(dst_pos[:, 1], torch.zeros_like(dst_pos[:, 1]))
    except ValueError:
        raise ValueError("Invalid projection of edge nodes to local ccordinate system")
    src_pos = geospatial_rotation(src_pos, theta=theta_polar, axis="y", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_polar, axis="y", unit="rad")
    # x values should be one, y & z values should be zero
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

    def create_mesh_graph(self, last_graph=None) -> DGLGraph:
        if self.doPrint:
            print("Creating bi-directional mesh graph")

        multimesh_faces = self.icospheres["order_0_faces"]
        for i in range(1, self.max_order + 1):
            multimesh_faces = np.concatenate(
                (multimesh_faces, self.icospheres["order_" + str(i) + "_faces"])
            )

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

    def create_g2m_graph(self, last_graph=None, mesh_graph=None) -> DGLGraph:
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

    def create_m2g_graph(self, last_graph=None) -> DGLGraph:
        if self.doPrint:
            print("Creating mesh2grid bipartite graph")

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
        src = [s for s in src if s < self.node_max]
        src = np.asarray(src)
        dst = dst[:src.shape[0]]
        dst = np.asarray(dst)

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