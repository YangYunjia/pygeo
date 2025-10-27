'''
CST-based wings

Yunjia. Yang


'''
from collections import OrderedDict
import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree
from pygeo.parameterization.DVGeoSketch import DVGeoSketch
from pygeo.parameterization.designVars import geoDV
from baseclasses.utils import Error

from typing import List, Callable
import json
import copy


class DVGeometryCustom(DVGeoSketch):
    '''
    `self.parameters`  -> used for reconstruct surface mesh (should be int, float, or 1D np.ndarray)
        (when write and read with jso, ndarray should be transferred to list)
    
    
    '''
    
    def __init__(self, fileName, generator: Callable, comm=MPI.COMM_WORLD, scale=1.0, projTol=1e-5, name=None, config=None):
        
        super().__init__(fileName=fileName, comm=comm, scale=scale, projTol=projTol, name=name)
        self.generator = generator
        
        self.parameters = {}
        with open(fileName, 'r') as f:
            load_parameters = json.load(f)
            for k in load_parameters:
                if isinstance(load_parameters[k], list):
                    self.parameters[k] = np.array(load_parameters[k])
                elif isinstance(load_parameters[k], (float, int)):
                    self.parameters[k] = load_parameters[k]
                else:
                    raise Error(f"json key {k} not valid with type {type(load_parameters[k])}")
                
        self._fd_steps = []
        self._neighbor_count = 8
        self._eps = 1e-12
        self._surface_ref = None
        self._surface_tree: cKDTree = None
        self._surface_delta = None      # current generated surface from the reference
        self._surface_valid = False
        self._baseline_initialized = False
        
        self.n_dv = 0
        
        self.config = config
        self.useComposite = False
        self._initialize_baseline_surface()
        
    def _initialize_baseline_surface(self):
        # print('Start Initialize')
        if self._baseline_initialized:
            return
        _ = self._updateModel(self.parameters, cache=True)
        self._baseline_initialized = True
        if self.comm.rank == 0:
            print('Initialize Success')

    def _updateModel(self, dv_dict, cache=False):
        
        surface = np.asarray(self.generator(dv_dict, config=self.config, save_surface=(cache and self.comm.rank == 0)), dtype=float).reshape(-1, 3)
        
        if cache:
            # cache the current surface
            if self._surface_ref is None:
                self._surface_ref = surface.copy()
                self._surface_tree = cKDTree(self._surface_ref)
                
            self._surface_delta = surface - self._surface_ref
            
            self._surface_valid = True
            
        return surface

    def _updateProjectedPts(self, ptSetName, delta):
        """
        Update the stored point-set using the latest surface.
        We always rely on the cached mapping so this works for both
        full-surface and distributed (subset) point sets.
        """
        meta = self.pointSets[ptSetName]
        indices = meta["mapping"]["indices"]
        weights = meta["mapping"]["weights"]
        neighbor_disp = delta[indices]
        disp = np.einsum("ij,ijk->ik", weights, neighbor_disp)
        return meta["ref"] + disp

    def writeJSON(self, fileName):
        write_parameters = {}
        
        for k in self.parameters.keys():
            write_parameters[k] = self.parameters[k].tolist() if isinstance(self.parameters[k], np.ndarray) else self.parameters[k]
        
        with open(fileName, 'w') as f:
            json.dump(write_parameters, f)
        
    @staticmethod
    def _check_shape_consistance(src, targ) -> np.ndarray:
        '''
        get the 1D flatten ndarray for input
        
        '''
        if isinstance(targ, np.ndarray):
            if isinstance(src, (float, int)):
                return src * np.ones(targ.shape)
            else:
                src = np.array(src)
                assert src.shape == targ.shape
                return src.reshape(-1)
        else:
            if isinstance(src, (float, int)):
                return np.array([src])
            else:
                raise Error(f"Type error")

    def addVariable(
        self, name, value=None, lower=None, upper=None, scale=1.0, dh=0.001
    ):
        """
        Add an customed design parameter to the DVGeo problem definition.

        Parameters
        ----------
        name : str or None
            Human-readable name for this design variable, should match param_meta in ShapeGenerator
        value : float or None
            The design variable. If this value is not supplied (None), then
            the current value in the ESP model will be queried and used.
        lower : float or None
            Lower bound for the design variable.
            Use None for no lower bound.
        upper : float or None
            Upper bound for the design variable.
            Use None for no upper bound.
        scale : float
            Scale factor sent to pyOptSparse and used in optimization.
        dh : float
            Finite difference step size.
            Default 0.001.
        """

        if name in self.DVs.keys():
            raise Error("Design variable name " + name + " already in use.")

        # find the design parm index in ESP
        if name not in self.parameters.keys():
            raise Error(
                'User specified design parameter name "' + name + '" which was not found in the Shape Generator "' + self.generator.name + '"'
            )
        
        base_param = self.parameters[name]
        # if value is not None:
        try:
            default = self._check_shape_consistance(value if value is not None else base_param, base_param)
        except AssertionError:
            raise Error(f"Design variable '{name}' default value does not match parameter shape {base_param.shape}")

        if lower is not None:
            try:
                lower = self._check_shape_consistance(lower, base_param)
            except AssertionError:
                raise Error(
                    f"Design variable '{name}' lower bound does not match parameter shape {base_param.shape}"
                )
        if upper is not None:
            try:
                upper = self._check_shape_consistance(upper, base_param)
            except AssertionError:
                raise Error(
                    f"Design variable '{name}' upper bound does not match parameter shape {base_param.shape}"
                )
        
        self.DVs[name] = geoDV(name, default, default.shape[0], lower, upper, scale)
        self.parameters[name] = default if len(default) > 0 else default[0]

        step_array = self._check_shape_consistance(dh, base_param)
        self._fd_steps.extend(step_array.tolist())
        self.n_dv += default.shape[0]
        
    def addPointSet(self, points, ptName, distributed=True, **kwargs):
        
        #### copy from DVGeoESP
        # save this name so that we can zero out the jacobians properly
        self.ptSetNames.append(ptName)
        self.points[ptName] = True  # ADFlow checks self.points to see if something is added or not
        coords = np.array(points).real.astype("d")

        # check that duplicated pointsets are actually the same length
        sizes = np.array(self.comm.allgather(coords.shape[0]), dtype="intc")
        if not distributed:
            all_same_length = np.all(sizes == sizes[0])
            if not all_same_length:
                raise ValueError(
                    "Nondistributed pointsets must be identical on each proc, but these pointsets vary in length per proc. Lengths: ",
                    str(sizes),
                )
        ####
        
        self._initialize_baseline_surface()
        meta = {
            "ref": coords,
            "current": coords.copy(),
            "jac": None,
            "distributed": distributed,
        }
        if self._surface_tree is None:
            raise Error("Reference surface tree not initialized.")

        # find the k nearest points on the surface
        k = min(self._neighbor_count, max(1, self._surface_ref.shape[0]))
        distances, indices = self._surface_tree.query(coords, k=k)
        distances = np.atleast_2d(distances)    # size: N, K (distance betw. 
        indices = np.atleast_2d(indices)        # given point N and its K's neighbor)

        weights = np.zeros_like(distances)
        close_mask = distances[:, 0] <= self.projTol    # the most close neighbor < Tol

        # following code deal with surface subsets and projection pointsets the same;
        # for any points that has a closest reference point in reference, it mapping 
        # that exact point; else it use a k-nearest inverse distance as weights 
        if np.any(close_mask):
            weights[close_mask, 0] = 1.0
        if np.any(~close_mask):
            far_rows = np.where(~close_mask)[0] # select non-close-point points
            inv_dist = 1.0 / np.maximum(distances[far_rows, :], self._eps) # avoid too large inverse dist
            inv_sum = inv_dist.sum(axis=1, keepdims=True)
            inv_sum[inv_sum == 0.0] = 1.0
            weights[far_rows, :] = inv_dist / inv_sum
            
        # here just to mark; they are deal the same in update
        local_allclose = bool(np.all(close_mask))
        global_allclose = self.comm.allreduce(local_allclose, op=MPI.LAND)
        
        meta["type"] = "surface" if global_allclose else "embedded"
        meta["mapping"] = {"indices": indices, "weights": weights}
        
        if self.comm.rank == 0:
            print(f'add pointset "{ptName}" of size {points.shape} with type "{meta["type"]} ({meta["distributed"]})" -- max Dist = {max(distances[:, 0]):.2e}')
        self.pointSets[ptName] = meta
        # Keep a copy in self.points so external callers (e.g. ADflow) know the pointset exists.
        # Store the local coordinates each rank sees; distributed pointsets will therefore hold
        # the per-rank slices.
        self.points[ptName] = coords.copy()
        self.updated[ptName] = False

    def setDesignVars(self, dvDict, updateJacobian=True):
        
        for key, val in dvDict.items():
            if key in self.DVs:
                new_val = self._check_shape_consistance(val, self.parameters[key])
                self.DVs[key].value = new_val
                self.parameters[key] = new_val if len(new_val) > 0 else new_val[0]
        
        # update design vars will invaildify the current surface
        self._surface_valid = False
        self._surface_delta = None
        
        for ptName in self.pointSets:
            self.updated[ptName] = False
            if not updateJacobian:
                self.pointSets[ptName]["jac"] = None
                
        return True

    def update(self, ptSetName, config=None):
        '''
        update the given ptSetName points from input
        
        '''
        self.config = config
        
        if not self._surface_valid:
            self._updateModel(self.parameters, cache=True)
            
        coords = self._updateProjectedPts(ptSetName, self._surface_delta)
        self.pointSets[ptSetName]["current"] = coords
        self.updated[ptSetName] = True
        return coords

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        self.config = config
        
        if self.pointSets[ptSetName]["jac"] is None:
            self._computeSurfJacobian(ptSetName)
            
        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])
        N = dIdpt.shape[0]
        nPt = dIdpt.shape[1]
        # print(comm.rank if comm is not None else 'None', dIdpt.shape)

        # The following code computes the final sensitivity product:
        #
        #        T       T
        #   pXpt     pI
        #  ------  ------
        #   pXdv    pXpt
        #
        # Where I is the objective, Xpt are the externally coordinates
        # supplied in addPointSet

        # Extract just the single dIdpt we are working with. Make
        # a copy because we may need to modify it.

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((N, nPt * 3))

        # # transpose dIdpt and vstack;
        # # Now vstack the result with seamBar as that is far as the
        # # forward FD jacobian went.
        tmp = dIdpt.T

        # we also stack the pointset jacobian
        jac = self.pointSets[ptSetName]["jac"].copy()

        # jac: [NI x NPt * 3]; tmp: [NPt * 3 x NDV]
        dIdxT_local = jac.T.dot(tmp)
        dIdx_local = dIdxT_local.T

        # outer loop will determine whether to use `comm` based on distributed / copy
        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        if self.useComposite:
            dIdx = self.mapSensToComp(dIdx)
        
        dIdxDict = self.convertSensitivityToDict(dIdx, useCompositeNames=self.useComposite)

        return dIdxDict

    def totalSensitivityProd(self, vec, ptSetName, comm=None, config=None):
        self.config = config
        
        if self.pointSets[ptSetName]["jac"] is None:
            self._computeSurfJacobian(ptSetName)

        dv_seed = self.convertDictToSensitivity(vec)
        pts_dot = self.pointSets[ptSetName]["jac"] @ dv_seed
        if comm:
            pts_dot = comm.allreduce(pts_dot, op=MPI.SUM)
        return pts_dot.reshape(-1, 3)

    def _computeSurfJacobian(self, ptSetName):
                
        if not self.updated.get(ptSetName, False):
            base_pts = self.update(ptSetName, config=self.config)
        else:
            base_pts = self.pointSets[ptSetName]["current"]

        base_flat = np.array(base_pts, dtype=float).reshape(-1)
        
        if len(self._fd_steps) != self.n_dv:
            raise Error(f"FD step array size {len(self._fd_steps)} does not match number of DVs {self.n_dv}")
        
        jac = np.zeros((base_flat.size, self.n_dv))

        i = 0
        for key in self.DVs:
            dv = self.DVs[key]
            for i_in_dv in range(dv.nVal):
                
                perturbed_parameters = copy.deepcopy(self.parameters)
                step = self._fd_steps[i + i_in_dv]
                if step == 0.0:
                    raise Error(f"Finite-difference step for DV index {i} is zero.")
                
                if dv.nVal == 1:
                    perturbed_parameters[key] += step
                else:
                    perturbed_parameters[key][i_in_dv] += step
                    
                delta_plus = self._updateModel(perturbed_parameters, cache=False) - self._surface_ref
                pts_plus = self._updateProjectedPts(ptSetName, delta_plus)
            
                jac[:, i + i_in_dv] = (pts_plus.reshape(-1) - base_flat) / step
                
            i += dv.nVal
            
        self.pointSets[ptSetName]["jac"] = jac
        
    def getNDV(self):
        return self.n_dv
