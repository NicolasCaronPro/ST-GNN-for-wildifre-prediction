import numpy as np
from sklearn.cluster import KMeans
import math
import geopandas as gpd 
from shapely import unary_union
import warnings
from shapely import Point

from models.models import *
from tools import *
from dataloader import *

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Create graph structure from corresponding geoDataframe
class GraphStructure():
    def __init__(self, scale : int,
                 geo : gpd.GeoDataFrame,
                 maxDist: float,
                 numNei : int):

        for col in ['latitude', 'longitude', 'geometry', 'departement']:
            if col not in geo.columns:
                logger.info(f'{col} not in geo columns. Please send a correct geo dataframe')
                exit(2)

        self.nodes = None # All nodes in graph (N, 6) [id, long, lat, dep]
        self.edges = None # All edges in graph (2 x E)
        self.temporalEdges = None # Temporal edges maximum (2 * D * k) where D is the number of dates available,
                                    #K is the number of link by date (minimum 1 for self connection)
        self.model = None # GNN model
        self.train_kmeans = None # Boolean to trace if we creating new scale
        self.scale = scale # current scale define by numUniqueNode // 6
        self.oriLatitues = geo.latitude # all original latitude of interest
        self.oriLongitude = geo.longitude # all original longitude of interest
        self.oriLen = self.oriLatitues.shape[0] # original len of data
        self.oriGeometry = geo['geometry'].values # original geometry
        self.departements = geo.departement # original dept of each geometry
        self.maxDist = maxDist # max dist to link two nodes
        self.numNei = numNei # max connect neighbor a node can have

    def _train_kmeans(self, doRaster: bool, path : Path, sinister : str) -> None:
        self.train_kmeans = True
        logger.info('Create node via KMEANS')
        X = list(zip(self.oriLongitude, self.oriLatitues))
        if self.scale != 0:
            self.numCluster = self.oriLen // (self.scale * 6)
        else:
            self.numCluster = self.oriLen
        self.kmeans = KMeans(self.numCluster, random_state=42, n_init=10)
        self.ids = self.kmeans.fit_predict(X)
        if doRaster:
            self._raster(path=path, sinister=sinister)

    def _predict_node(self, array : list) -> np.array:
        assert self.kmeans is not None
        res = self.kmeans.predict(array)
        return res
    
    def _raster2(self, path, n_pixel_y, n_pixel_x, resStr, doBin, sinister):

        dir_bin = root_target / sinister / 'bin'
        dir_target = root_target / sinister / 'log'

        for dept in np.unique(self.departements):

            maskDept = np.argwhere(self.departements == dept)
            geo = gpd.GeoDataFrame(index=np.arange(maskDept.shape[0]), geometry=self.oriGeometry[maskDept[:,0]])
            geo['scale'+str(self.scale)] = self.ids[maskDept]
            check_and_create_path(Path('log'))
            mask, _, _ = rasterization(geo, n_pixel_y, n_pixel_x, 'scale'+str(self.scale), Path('log'), 'ori')
            mask = mask[0]

            save_object(mask, str2name[dept]+'rasterScale'+str(self.scale)+'.pkl', path / 'raster' / resStr)

            if doBin:
                outputName = str2name[dept]+'binScale0.pkl'
                bin = read_object(outputName, dir_bin)
                if bin is None:
                    continue
                outputName = str2name[dept]+'Influence.pkl'
                influence = read_object(outputName, dir_target)

                binImageScale, influenceImageScale = create_larger_scale_bin(mask, bin, influence)

                save_object(binImageScale, str2name[dept]+'binScale'+str(self.scale)+'.pkl', path / 'bin' / resStr)
                save_object(influenceImageScale, str2name[dept]+'InfluenceScale'+str(self.scale)+'.pkl', path / 'influence' / resStr)

    def _raster(self, path : Path,
                sinister : str) -> None:
        """"
        Create new raster mask
        """
        
        check_and_create_path(path)
        check_and_create_path(path / 'raster')
        check_and_create_path(path / 'bin')
        check_and_create_path(path / 'proba')

        # 1 pixel = 1 Hexagone (2km) use for 1D process
        n_pixel_x = 0.02875215641173088
        n_pixel_y = 0.020721094073767096

        resStr = '2x2'
        self._raster2(path, n_pixel_y, n_pixel_x, resStr, True, sinister)

        # 1 pixel = 1 km, use for 2D process Not suitable for now
        n_pixel_y = 0.009
        n_pixel_x = 0.012
        resStr = '1x1'
        self._raster2(path, n_pixel_y, n_pixel_x, resStr, False, sinister)

    def _read_kmeans(self, path : Path) -> None:
        assert self.scale != 0
        self.train_kmeans = False
        name = 'kmeans'+str(self.scale)+'.pkl'
        self.numCluster = self.oriLen // (self.scale * 6)
        X = list(zip(self.oriLongitude, self.oriLatitues))
        self.kmeans = pickle.load(open(path / name, 'rb'))
        self.ids = self.kmeans.predict(X)

    def _create_predictor(self, start, end, dir):
        dir_influence = root_graph / dir / 'influence' / '2x2'
        dir_predictor = root_graph / dir / 'influenceClustering'
  
        for dep in np.unique(self.departements):

            influence = read_object(str2name[dep]+'InfluenceScale'+str(self.scale)+'.pkl', dir_influence)
            if influence is None:
                continue
            influence = influence[:,:,allDates.index(start):allDates.index(end)]

            predictor = Predictor(5)
            predictor.fit(np.asarray(np.unique(influence[~np.isnan(influence)])))
            save_object(predictor, str2name[dep]+'Predictor'+str(self.scale)+'.pkl', path=dir_predictor)

    def _create_nodes_list(self) -> None:
        """
        Create nodes list in the form (N, 4) where N is the number of Nodes [ID, longitude, latitude, departement of centroid]
        """
        logger.info('Creating node list')
        self.nodes = np.full((self.numCluster, 5), np.nan) # [id, longitude, latitude]
        self.uniqueIDS = np.unique(self.ids)
        for id in self.uniqueIDS:
            mask = np.argwhere(id == self.ids)
            if self.scale != 0:
                nodeGeometry = unary_union(self.oriGeometry[mask[:,0]]).centroid
            else:
                nodeGeometry = self.oriGeometry[mask[:,0]].centroid

            valuesDep, counts = np.unique(self.departements[mask[:,0]], return_counts=True)
            ind = np.argmax(counts)
            self.nodes[id] = np.asarray([id, float(nodeGeometry.x), float(nodeGeometry.y), str2intMaj[valuesDep[ind]], -1])

    def _create_edges_list(self) -> None:
        """
        Create edges list
        """
        logger.info('Creating edges list')
        self.edges = None # (2, E) where E is the number of edges in the graph
        src_nodes = []
        target_nodes = []
        for id in self.uniqueIDS:
            closest = self._distances_from_node(self.nodes[id], self.nodes[self.nodes[:,0] != id])
            closest = closest[closest[:, 1].argsort()]
            for i in range(self.numNei):
                if closest[i, 1] > self.maxDist:
                    break
                src_nodes.append(id)
                target_nodes.append(closest[i,0])

        self.edges = np.row_stack((src_nodes, target_nodes)).astype(int)

    def _create_temporal_edges_list(self, dates : list, k_days : int):
        self.k_days = k_days
        self.temporalEdges = None
        src_date = []
        target_date = []
        leni = len(dates)
        for i, _ in enumerate(dates):
            for k in range(k_days+1):
                if i + k > leni:
                    break
                src_date.append(i)
                target_date.append(i + k)

        self.temporalEdges = np.row_stack((src_date, target_date)).astype(int)

    def _distances_from_node(self, node : np.array,
                            otherNodes: np.array) -> np.array:
        """
        Calculate the distance bewten node and all the other nodes
        """
        res = np.full((otherNodes.shape[0], 2), math.inf)
        ignore = [node[0]]
        for index, (id, lon, lat, _, _) in enumerate(otherNodes):
            if id in ignore:
                continue
            distance = haversine((node[1], node[2]), [lon, lat])
            res[index, 0 ] = id
            res[index, 1] = distance
            ignore.append(id)
        return res
    
    def _assign_department(self, nodes : np.array) -> np.array:
        logger.info('Get node department')
        uniqueDept = np.unique(self.departements)
        dico = {}
        for key in uniqueDept:
            mask = np.argwhere(self.departements == key)
            geoDep = unary_union(self.oriGeometry[mask[:,0]])
            dico[key] = geoDep

        def find_dept(x, y, dico):
            for key, value in dico.items():
                if value.contains(Point(x, y)):
                    return str2intMaj[key]
 
            return np.nan

        nodes[:,3] = np.asarray([find_dept(node[1], node[2], dico) for node in nodes])

        return nodes
    
    def _assign_latitude_longitude(self, nodes : np.array) -> np.array:
        logger.info('Assign latitude longitude')

        uniqueNode = np.unique(nodes[:,0])
        for uN in uniqueNode:
            maskNode = np.argwhere(nodes[:,0] == uN)
            nodes[maskNode[:,0],1:3] = self.nodes[np.argwhere(self.nodes[:,0] == uN)[:,0]][0][1:3]
        return nodes

    def _plot(self, nodes : np.array, time=0) -> None:
        """
        Plotting
        """
        dis = np.unique(nodes[:,4])[:time+1].astype(int)
        if time > 0:
            ax = plt.figure(figsize=(15,15)).add_subplot(projection='3d')
        else:
            _, ax = plt.subplots(figsize=(15,15))
        if self.edges is not None:
            uniqueIDS = np.unique(nodes[:,0]).astype(int)
            for id in uniqueIDS:
                connection = self.edges[1][np.argwhere((self.edges[0] == id) & (np.isin(self.edges[1], nodes[:,0])))[:,0]]
                for target in connection:
                    target = int(target)
                    if target in uniqueIDS:
                        if time > 0:
                            for ti in range(time + 1):
                                ax.plot([dis[ti], dis[ti]],
                                        [nodes[nodes[:,0] == id][0][1], nodes[nodes[:,0] == target][0][1]],
                                        [nodes[nodes[:,0] == id][0][2], nodes[nodes[:,0] == target][0][2]],
                                    color='black')
                        else:
                            ax.plot([nodes[nodes[:,0] == id][0][1], nodes[nodes[:,0] == target][0][1]],
                                    [nodes[nodes[:,0] == id][0][2], nodes[nodes[:,0] == target][0][2]], color='black')
        if time > 0:
            if self.temporalEdges is not None:
                nb = 0
                uniqueDate = np.unique(nodes[:,4]).astype(int)
                uniqueIDS = np.unique(nodes[:,0]).astype(int)
                for di in uniqueDate:
                    connection = self.temporalEdges[1][np.argwhere((self.temporalEdges[0] == di))[:,0]]
                    for target in connection:
                        target = int(target)
                        if target in uniqueDate:
                            for id in uniqueIDS:
                                ax.plot([di, target],
                                         [nodes[nodes[:,0] == id][0][1], nodes[nodes[:,0] == id][0][1]],
                                         [nodes[nodes[:,0] == id][0][2], nodes[nodes[:,0] == id][0][2]],
                                        color='red')
                    nb += 1
                    if nb == time:
                        break
        
        node_x = nodes[nodes[:,4] == dis[0]][:,1]
        node_y = nodes[nodes[:,4] == dis[0]][:,2]
        for ti in range(1, time+1):
            node_x = np.concatenate((node_x, nodes[nodes[:,4] == dis[ti]][:,1]))
            node_y = np.concatenate((node_y, nodes[nodes[:,4] == dis[ti]][:,2]))
   
        if time > 0:
            node_z = nodes[nodes[:,4] == dis[0]][:,4]
            for ti in range(1, time+1):
                node_z = np.concatenate((node_z, nodes[nodes[:,4] == dis[ti]][:,4]))

            ax.scatter(node_z, node_x, node_y)
            ax.set_xlabel('Date')
            ax.set_ylabel('Longitude')
            ax.set_zlabel('Latitude')
        else:
            node_x = nodes[:,1]
            node_y = nodes[:,2]
            ax.scatter(x=node_x, y=node_y, s=60)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')

    def _set_model(self, model) -> None:
        """
        model : the neural network model. Train or not
        """
        self.model = model

    def _load_model_from_path(self, path : Path, model : torch.nn.Module, device) -> None:
        model.load_state_dict(torch.load(path, map_location=device), strict=False)
        self.model = model

    def _set_model_from_path(self, path : Path) -> None:
        self.modelPath = path

    def _predict_np_array(self, X : np.array, edges : np.array, device : torch.device) -> torch.tensor:
        assert self.model is not None
        self.model.eval()
        tensorX = torch.tensor(X, dtype=torch.float32, device=device)
        tensorEdges = torch.tensor(edges, dtype=torch.float32, device=device)
        return self.model(tensorX, tensorEdges)

    def _predict_tensor(self, X : torch.tensor, edges : torch.tensor) -> torch.tensor:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            return self.model(X, edges)

    def _predict_dataset(self, X : Dataset) -> torch.tensor:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            testloader = DataLoader(X, 1)
            for i, data in enumerate(testloader, 0):
                inputs, _, edges = data
                output = self.model(inputs, edges)
                return output
        
    def _predict_test_loader(self, X : DataLoader, features : np.array) -> torch.tensor:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            pred = []
            y = []
         
            for i, data in enumerate(X, 0):
                inputs, labels, edges = data
                inputs = inputs[:, features]
                inputs = inputs.to(device)
                labels = labels.to(device)
                edges = edges.to(device)
                
                weights = labels[:,-3].to(torch.long)

                output = self.model(inputs, edges)

                ids = torch.masked_select(labels[:,0], weights.gt(0))
                target = torch.empty((ids.shape[0], labels.shape[1]), device=device)
                target[:,0] = ids
                for b in range(1, labels.shape[1]):
                    target[:,b] = torch.masked_select(labels[:,b], weights.gt(0))

                #target = labels[weights.gt(0)]
                
                output = torch.masked_select(output, weights.gt(0))
                weights = torch.masked_select(weights, weights.gt(0))

                pred.append(output)
                y.append(target)
            
            pred = torch.cat(pred, 0)
            y = torch.cat(y, 0)

            return pred, y
        
    def predict_model_api_sklearn(self, X : np.array, isBin : bool) -> np.array:
        assert self.model is not None
        if not isBin:
            return self.model.predict(X)
        else:
             return self.model.predict_proba(X)[:,1]
        
    def _predict_perference_with_Y(self, Y : np.array, isBin : bool) -> np.array:
        res = np.empty(Y.shape[0])
        res[0] = 0
        for i, node in enumerate(Y):
            if i == 0:
                continue
            index = np.argwhere((Y[:,4] == node[4] - 1) & (Y[:,0] == node[0]))
            if index.shape[0] == 0:
                res[i] = 0
                continue
            if not isBin: 
                res[i] = Y[index[0], -1]
            else:
                res[i] = Y[index[0], -2] > 0
        return res
        
    def _predict_perference_with_X(self, X : np.array, pos_feature : dict) -> np.array:
        return X[:, pos_feature['Historical'] + 2]

    def _info_on_graph(self, nodes : np.array, output : Path) -> None:
        """
        logger.info informatio on current global graph structure and nodes
        """
        check_and_create_path(output)
        graphIds = np.unique(nodes[:,4])

        size = []
        for graphid in graphIds:
            size.append(nodes[nodes[:,4] == graphid].shape[0])

        logger.info(f'size : {np.unique(size)}')

        logger.info('**************************************************')
        if self.nodes is None:
            logger.info('No nodes found in graph')
        else:
            logger.info(f'The main graph has {self.nodes.shape[0]} nodes')
        if self.edges is None:
            logger.info('No edges found in graph')
        else:
            logger.info(f'The main graph has {self.edges.shape[1]} spatial edges')
        if self.temporalEdges is None:
            logger.info('No temporal edges found in graph')
        else:
            logger.info(f'The main graph has {self.temporalEdges.shape[1]} temporal edges')
        if nodes is not None:
            logger.info(f'We found {np.unique(nodes[:,4]).shape[0]} different graphs in the training set for {nodes.shape[0]} nodes')

        logger.info(f'Mean number of nodes in sub graph : {np.mean(size)}')
        logger.info(f'Max number of nodes in subgraph {np.max(size)}')
        logger.info(f'Minimum number of nodes in subgraph {np.min(size)}')
        logger.info(f'Quantile at 0.05 {np.quantile(size, 0.005)}, at 0.5 {np.quantile(size, 0.5)} and at 0.95 {np.quantile(size, 0.95)}')