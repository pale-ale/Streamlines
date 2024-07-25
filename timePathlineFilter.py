from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smproxy, smproperty
import vtkmodules.numpy_interface.dataset_adapter as dsa
from vtkmodules.numpy_interface.algorithms import divergence
from dataclasses import dataclass, field
from collections import defaultdict

import time
import numpy as np
import vtk
import itertools
import copy
from vtk.util import numpy_support

import cProfile

Grid = dict[tuple[int,int,int], "GridCell"]

@smproxy.filter(label="TimePathlineFilter")
@smproperty.xml("""
    <OutputPort index="0" name="Initial Lines" />
    <OutputPort index="1" name="Timestep 1 Lines" />
""")
@smproperty.input(name="Input")
class TimePathlineFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
            nInputPorts=1,
            inputType='vtkMultiBlockDataSet',
            nOutputPorts=2,
            outputType='vtkMultiBlockDataSet')
        self.stepcount = 10
        self.stepsizescale = .05
        self.seedoffset = np.zeros(3)
        self.neighbor_distance_min = .05
        self.neighbor_test_distance = .3
        # self.raster = defaultdict(lambda: RasterCell())
        # the raster is indexed by 3 spatial extents, the extens themselves capture time info
        # (type hints are broken for defaultdict, best i can do is dict)
        self.streamlineId = 0
        self.neighbor_count = 0
        self.max_iterations = 10
        self.constrain_to_2d = False
        self.divergence_filter = vtk.vtkImageDivergence()
        self.hausdorff_filter = vtk.vtkHausdorffDistancePointSetFilter()
    
    @smproperty.intvector(name="Step Count", default_values=20)
    def SetStepCount(self, new_count:int):
        self.stepcount = new_count
        self.Modified()
    
    @smproperty.doublevector(name="Step Size Scale", default_values=.25)
    def SetStepSizeScale(self, new_size:float):
        self.stepsizescale = new_size
        self.Modified()
    
    @smproperty.doublevector(name="Minimum Neighbor Distance", default_values=.24)
    def SetMinNeighborDistance(self, new_dist:float):
        self.neighbor_distance_min = new_dist
        self.Modified()
    
    @smproperty.doublevector(name="Neighbor Test Distance", default_values=.25)
    def SetNeighborDistance(self, new_dist:float):
        self.neighbor_test_distance = new_dist
        self.Modified()

    @smproperty.doublevector(name="Seed Offset", number_of_elements=3, default_values=[.5,.5,.9])
    def SetSeedOffset(self, x, y, z):
        self.seedoffset = np.array([x, y, z])
        self.Modified()
    
    @smproperty.intvector(name="Neighbor Count", default_values=5)
    def SetNeighborCount(self, new_count:int):
        self.neighbor_count = new_count
        self.Modified()
    
    @smproperty.intvector(name="Iterations", default_values=10)
    def SetMaxIterations(self, new_iterations:int):
        self.max_iterations = new_iterations
        self.Modified()
    
    @smproperty.doublevector(name="Min Line Length", default_values=.3)
    def SetMinLineLength(self, length:float):
        self.minlinelength = length
        self.Modified()

    @smproperty.xml("""
    <IntVectorProperty name="PadData"
        label="Pad Data"
        command="SetConstrainTo2D"
        number_of_elements="1"
        default_values="1">
        <BooleanDomain name="bool" />
        <Documentation>If checked, dataset will be padded at all timestates
            to ensure consistent block indices</Documentation>
    </IntVectorProperty>
    """)
    def SetConstrainTo2D(self, val):
        '''Set attribute to signal whether data should be padded when RequestData is called
        '''
        self.constrain_to_2d = val
        self.Modified()

    def generate_initial_layout(self, inMultiBlock: vtk.vtkMultiBlockDataSet, initialLayoutMultiBlock: vtk.vtkMultiBlockDataSet):
        timestep_count = inMultiBlock.GetNumberOfBlocks()
        initialLayoutMultiBlock.SetNumberOfBlocks(timestep_count)
        # qualities = vtk.vtkFloatArray()
        # qualities.SetName("Quality")
        # qualities.SetNumberOfValues(timestep_count)
        initial_raster:TimeGrid = TimeGrid()
        imageDataIter = inMultiBlock.NewIterator()
        midpointSeeds:list[list[tuple[float, float, float]]] = []
        while not imageDataIter.IsDoneWithTraversal():
            idx = imageDataIter.GetCurrentFlatIndex() - 1
            imageData:vtk.vtkImageData = imageDataIter.GetCurrentDataObject()
            polyData = vtk.vtkPolyData()
            currentMidSeeds = []
            ##### currently irrelevant
            # nppointdata:vtk.vtkPointData = dsa.WrapDataObject(imageData).GetPointData()
            # field_divergence = divergence(nppointdata["velocity"])
            seeds = self.get_initial_seeds() + self.seedoffset
            points, ids_list = self.generate_streamlines(imageData, idx, initial_raster, seeds)
            polyData.SetPoints(points)
            polyData.Allocate(len(ids_list))
            for ids in ids_list:
                polyData.InsertNextCell(vtk.VTK_POLY_LINE, ids)
                midId = ids.GetId(int(ids.GetNumberOfIds()/2))
                currentMidSeeds.append(points.GetPoint(midId))
            initialLayoutMultiBlock.SetBlock(idx, polyData)
            midpointSeeds.append(currentMidSeeds)
            # qualities.SetValue(idx, self.measure_quality(points, ids_list))
            imageDataIter.GoToNextItem()
        # initialLayoutMultiBlock.GetFieldData().AddArray(qualities)
        return initial_raster, midpointSeeds

    def RequestData(self, request:vtk.vtkInformation, inInfo:tuple[vtk.vtkInformationVector], outInfo:vtk.vtkInformationVector):
        t = time.time()
        inMultiBlock = vtk.vtkMultiBlockDataSet.GetData(inInfo[0])
        timestepCount = inMultiBlock.GetNumberOfBlocks()

        ### Obtain initial lines for every timestep
        initialLayoutMultiBlock = vtk.vtkMultiBlockDataSet.GetData(outInfo.GetInformationObject(0))
        initialRaster, midSeeds = self.generate_initial_layout(inMultiBlock=inMultiBlock, initialLayoutMultiBlock=initialLayoutMultiBlock)
        # cProfile.runctx(
        #     "self.generate_initial_layout(inMultiBlock=inMultiBlock, initialLayoutMultiBlock=initialLayoutMultiBlock)",
        #     globals={},
        #     locals={"self":self, "initialLayoutMultiBlock":initialLayoutMultiBlock, "inMultiBlock":inMultiBlock},
        #     )
        # flattenedMidSeeds = list((seed for timestepseeds in midSeeds for seed in timestepseeds))
        # arr, s = self.get_temporal_importance(initialRaster)
        # initialLayoutMultiBlock.GetFieldData().AddArray(arr)

        ### Use the 1st timestep's seeds to draw lines for every timestep
        # firstTimestepRaster = TimeGrid()
        # firstTimestepMultiBlock = vtk.vtkMultiBlockDataSet.GetData(outInfo.GetInformationObject(1))
        # print("Drawing combined")
        # imageDataIter = inMultiBlock.NewIterator()
        # while not imageDataIter.IsDoneWithTraversal():
        #     idx = imageDataIter.GetCurrentFlatIndex() - 1
        #     imageData:vtk.vtkImageData = imageDataIter.GetCurrentDataObject()
        #     polyData = vtk.vtkPolyData()
        #     points, ids_list = self.generate_streamlines(imageData, idx, firstTimestepRaster, flattenedMidSeeds, autofill=False)
        #     # print("Seeds:", len(flattenedMidSeeds), "Lines:", len(ids_list))
        #     polyData.SetPoints(points)
        #     polyData.Allocate(len(ids_list))
        #     for ids in ids_list:
        #         polyData.InsertNextCell(vtk.VTK_POLY_LINE, ids)
        #     firstTimestepMultiBlock.SetBlock(idx, polyData)
        #     imageDataIter.GoToNextItem()
        
        # timesteps = len(midSeeds)
        # seeds_per_timestep = len(midSeeds[0])
        # for lineid in range(seeds_per_timestep):
        #     temporal_score = vtk.vtkDoubleArray()
        #     temporal_score.SetName(f"Line {lineid} Score")
        #     lines:list[vtk.vtkPoints] = []
        #     for timestep in range(timesteps):
        #         if (line := firstTimestepRaster.get_line_from_timestep(timestep, lineid)) is not None:
        #             lines.append(line)
        #     if lines:
        #         ts = self.get_temporal_coherence(*lines)
        #         temporal_score.InsertNextValue(ts)
        #     initialLayoutMultiBlock.GetFieldData().AddArray(temporal_score)
        print(f"Reloaded after {time.time() - t:.3f}s")
        return 1
    
    def generate_streamlines(self, imagedata:vtk.vtkImageData, timestep:int, grid:"TimeGrid", seeds:np.ndarray, autofill=True):
        vpi = vtk.vtkPointInterpolator()
        vpi.SetKernel(vtk.vtkGaussianKernel())
        vpi.SetSourceData(imagedata)
      
        concat_pts = vtk.vtkPoints()
        line_pointids:list[vtk.vtkIdList] = []
        idx = 0
        self.streamlineId = 0
        imagedata.ComputeBounds()
        seed_candidates:list[np.ndarray] = list(seeds)
        previous_nppts = None
        while seed_candidates and (not autofill or (len(line_pointids) < self.max_iterations)):
            t = time.time()
            seed = seed_candidates.pop(0)
            pts, ids = self.make_streamline(seed, vpi, idx, imagedata.GetBounds(), timestep, grid, ignore_grid=not autofill)
            if pts is None:
                continue
            nppts = numpy_support.vtk_to_numpy(pts.GetData())
            if previous_nppts is None:
                previous_nppts = nppts
            idx += pts.GetNumberOfPoints()
            if autofill: # Choose new points, discard bad ones. Otherwise, just draw the provided seeds
                next_candidates = []
                # Keep points we didnt cover with the current line, discard the rest
                for i, candidate in enumerate(seed_candidates):
                    if all(np.linalg.norm(nppts - candidate, axis=1) >= self.neighbor_distance_min):
                        next_candidates.append(candidate)
                # Fetch new points from our vincinity
                new_candidates = self.get_neighbor_points(pts, self.neighbor_test_distance, self.neighbor_count)
                # Discard points too close to ourselves or the parent line (i.e. the one we just came from)
                for candidate in new_candidates:
                    if np.any(np.linalg.norm(previous_nppts - candidate, axis=1) <= self.neighbor_distance_min): 
                        continue # close to parent
                    if np.any(np.linalg.norm(nppts - candidate, axis=1) <= self.neighbor_distance_min):
                        continue # close to self
                    next_candidates.append(candidate)
                seed_candidates = next_candidates
            concat_pts.InsertPoints(concat_pts.GetNumberOfPoints(), pts.GetNumberOfPoints(), 0, pts)
            line_pointids.append(ids)
        return concat_pts, line_pointids

    def get_temporal_importance(self, grid: "TimeGrid"):
        extents = (int(2.0/self.neighbor_distance_min) + 1)
        temporal_importance_array = vtk.vtkUnsignedIntArray()
        temporal_importance_array.SetName("Temporal Weight")
        temporal_importance_array.SetNumberOfValues(extents ** 3)
        temporal_importance_array.Fill(0.0)
        s = ""
        for (x, y, z), cell in grid.grid.items():
            s += f"{(x, y, z)}: {cell.temporal_importance}, "
            temporal_importance_array.SetValue(z * extents**2 + y * extents + x, cell.temporal_importance)
        return temporal_importance_array, s

    def measure_quality(self, points:vtk.vtkPoints, lineids:list[vtk.vtkIdList]) -> float:
        distancefilter = vtk.vtkHausdorffDistancePointSetFilter()
        pts1, pts2 = vtk.vtkPoints(), vtk.vtkPoints()
        ptset1, ptset2 = vtk.vtkPointSet(), vtk.vtkPointSet()
        distance = float("Inf")
        for ids1, ids2 in itertools.combinations(lineids, 2):
            points.GetPoints(ids1, pts1)
            points.GetPoints(ids2, pts2)
            ptset1.SetPoints(pts1)
            ptset2.SetPoints(pts2)
            distancefilter.SetInputData(0, ptset1)
            distancefilter.SetInputData(1, ptset2)
            distancefilter.Update()
            output:vtk.vtkPointSet = distancefilter.GetOutput()
            arr = output.GetPointData().GetArray(0)
            distance = min(np.min(arr), distance)
        return distance

    def get_temporal_coherence(self, *lines:vtk.vtkPoints):
        start, *rest = lines
        max_dist = 0.0
        ptset1, ptset2 = vtk.vtkPointSet(), vtk.vtkPointSet()
        ptset1.SetPoints(start)
        self.hausdorff_filter.SetInputData(0, ptset1)
        for line in rest:
            ptset2.SetPoints(line)
            self.hausdorff_filter.SetInputData(1, ptset2)
            self.hausdorff_filter.Update()
            output:vtk.vtkPointSet = self.hausdorff_filter.GetOutput()
            arr = output.GetPointData().GetArray(0)
            max_dist = max(np.max(arr), max_dist)
        return max_dist

    def get_initial_seeds(self) -> np.ndarray:
        return np.array([(.0,.0,.0)], dtype=float)
    
    def normalize(self, vecs:np.ndarray):
        if len(vecs.shape) == 1:
            return vecs / np.linalg.norm(vecs)
        return vecs / np.asmatrix(np.linalg.norm(vecs, axis=1)).T

    def roots_of_unity(self, k:int):
        exponents = np.arange(k) * 1j * np.pi * 2.0 / k
        return np.power(np.e, exponents, dtype=complex)
    
    def get_neighbors_normal(self, p:np.ndarray, normals:np.ndarray, num:int, d:float):
        if self.constrain_to_2d: # we have points deltas (normal vectors in 3d) without a z comonent if using 2d only
            xyplane_normals = self.normalize(np.vstack((-normals[:,1], normals[:,0], np.zeros((normals.shape[0])))).T)
            return np.concatenate(((p + xyplane_normals * d), (p - xyplane_normals * d)), axis=0)
        
        norm_normals = self.normalize(normals)
        identity = np.identity(3, dtype=float)
        first_normals:np.ndarray = np.tile(identity[0], (normals.shape[0], 1))
        xdots = np.einsum('ij, j->i', norm_normals, identity[0])
        first_normals[xdots > .99] = identity[1] # if we are too close to [1,0,0], choose [0,1,0]
        ydots = np.einsum('ij, j->i', norm_normals, identity[1])
        first_normals[ydots > .99] = identity[2] # if we are too close to [0,1,0], choose [0,0,1]
        zdots = np.einsum('ij, j->i', norm_normals, identity[2])
        first_normals[zdots > .99] = identity[0] # if we are too close to [0,0,1], choose [1,0,0]
        second_normals = self.normalize(np.cross(normals, first_normals))

        neighbors = np.repeat(p, num, axis=0)
        scaled_roots_of_unity = self.roots_of_unity(num) * d
        re = np.real(scaled_roots_of_unity)
        im = np.imag(scaled_roots_of_unity)

        for i in range(first_normals.shape[0]):
            qr_input = np.vstack((norm_normals[i], first_normals[i], second_normals[i]))
            q, _ = np.linalg.qr(qr_input.T)
            q = q.T
            n1 = np.repeat(q[1], num, axis=0)
            n2 = np.repeat(q[2], num, axis=0)
            neighbors[i*num:i*num+num] += np.einsum("ij, i -> ij", n1, re) + np.einsum("ij, i -> ij", n2, im)
        return neighbors

    def is_close_to_another(self, point:np.ndarray, distance:float, exclude_id:int, timestep:int, raster:"TimeGrid"):
        assert distance <= self.neighbor_distance_min
        for offset in itertools.product(
            [-self.neighbor_distance_min, 0, self.neighbor_distance_min],
            [-self.neighbor_distance_min, 0, self.neighbor_distance_min],
            [-self.neighbor_distance_min, 0, self.neighbor_distance_min] if not self.constrain_to_2d else [0],
        ):
            key = self.get_raster_key(*(point + offset))
            cell = raster.grid.get(key)
            if cell is None:
                continue
            verts, ids = cell.vertex_info.get(timestep, (None, None))
            if verts is None or ids is None:
                continue
            filtered_verts:np.ndarray = verts[ids != exclude_id]
            filtered_ids:np.ndarray   =   ids[ids != exclude_id]
            if filtered_verts.size == 0 or filtered_ids.size == 0:
                continue
            distances = np.linalg.norm(filtered_verts - point, axis=1)
            mindistidx = np.argmin(distances)
            mindist, mindistlineid = distances[mindistidx], filtered_ids[mindistidx]
            if mindist <= distance:
                return True
        return False
    
    def get_neighbor_points(self, linepts:vtk.vtkPoints, distance:float, neighbor_count:int):
        assert neighbor_count > 0
        points:np.ndarray = numpy_support.vtk_to_numpy(linepts.GetData())
        deltas = points[1:] - points[:-1]
        normals = np.copy(self.get_neighbors_normal(points[1:], deltas, neighbor_count,  distance))
        
        return normals
    
    def make_streamline(self, seed, vpi:vtk.vtkPointInterpolator, idstart:int, bounds:tuple, timestep:int, grid:"TimeGrid", ignore_grid=False):
        def is_too_close(pos):
            return not ignore_grid and self.is_close_to_another(pos, self.neighbor_distance_min, self.streamlineId, timestep, grid)
        def is_outside(pos):
            return not TimePathlineFilter.inside_domain(pos, *bounds)
        points_to_interpolate = vtk.vtkPoints()
        data_to_interpolate = vtk.vtkPointSet()
        data_to_interpolate.SetPoints(points_to_interpolate)
        vpi.SetInputData(data_to_interpolate)
        
        backward_points = vtk.vtkPoints()
        backward_points.SetNumberOfPoints(self.stepcount)
        backward_ids = vtk.vtkIdList()
        backward_start_idx = 0
        current_pos = copy.deepcopy(seed)
        if is_outside(current_pos) or is_too_close(current_pos):
            return None, None

        for i in range(self.stepcount):
            backward_points.InsertPoint(self.stepcount - 1 - i, current_pos)
            backward_ids.InsertNextId(i)
            points_to_interpolate.SetNumberOfPoints(1) #why do i need to call this every time?
            points_to_interpolate.SetPoint(0, current_pos)
            vpi.Update()
            out:vtk.vtkPointSet = vpi.GetOutput()
            velocities = out.GetPointData().GetArray(0)
            velocity:np.ndarray = numpy_support.vtk_to_numpy(velocities)[0]
            current_pos -= self.normalize(velocity) * self.stepsizescale
            if is_outside(current_pos) or is_too_close(current_pos):
                backward_start_idx = self.stepcount - i - 1
                break

        backward_count = backward_points.GetNumberOfPoints() - backward_start_idx
        forward_points = vtk.vtkPoints()
        forward_ids = vtk.vtkIdList()
        current_pos = copy.deepcopy(seed)
        for i in range(self.stepcount):
            forward_points.InsertNextPoint(current_pos)
            forward_ids.InsertNextId(backward_count + i) 
            points_to_interpolate.SetNumberOfPoints(1) #why do i need to call this every time?
            points_to_interpolate.SetPoint(0, current_pos)
            vpi.Update()
            out:vtk.vtkPointSet = vpi.GetOutput()
            velocities = out.GetPointData().GetArray(0)
            velocity = numpy_support.vtk_to_numpy(velocities)[0]
            current_pos += self.normalize(velocity) * self.stepsizescale
            if is_outside(current_pos) or is_too_close(current_pos):
                break
        
        forward_count = forward_points.GetNumberOfPoints()
        points = vtk.vtkPoints()
        points.InsertPoints(0, backward_count, backward_start_idx, backward_points) # contains the seed point
        points.InsertPoints(backward_count, forward_count-1, 1, forward_points) # we dont want that twice
        
        if backward_count + forward_count < 2 or points.GetNumberOfPoints() * self.stepsizescale < self.minlinelength:
            return None, None
        
        ids = vtk.vtkIdList()
        for i in range(points.GetNumberOfPoints()):
            ids.InsertNextId(idstart + i)
        np_points:np.ndarray = numpy_support.vtk_to_numpy(points.GetData())
        keys = [(x,y,z) for x,y,z in (np_points / self.neighbor_distance_min).astype(int)]
        grid.add_line(timestep=timestep, lineid=self.streamlineId, points=points, keys=keys)
        self.streamlineId += 1
        return points, ids
    
    def get_raster_key(self, x:float, y:float, z:float):
        return int(x/self.neighbor_distance_min), int(y/self.neighbor_distance_min), int(z/self.neighbor_distance_min)

    @staticmethod
    def inside_domain(point, x1, x2, y1, y2, z1, z2):
        return all((l <= c <= r for (l,c,r) in zip((x1, y1, z1), point, (x2, y2, z2))))

@dataclass
class GridCell():
    vertex_info: dict[int, list[np.ndarray]] = field( #this type hint is a lie, see above
        default_factory=lambda: defaultdict(
            lambda: [
                np.ndarray((0,3), dtype=float),
                np.ndarray((0), dtype=int)
            ]
        )
    )
    temporal_importance: int = 0

    def get_vertex_info(self, timestep: int) -> tuple[np.ndarray, np.ndarray]:
        info = self.vertex_info.get(timestep) # does defaultdict.get trigger the default_factory?
        if not info:
            return None
        return info[0], info[1]

class TimeGrid:
    def __init__(self) -> None:
        self.grid:Grid = defaultdict(lambda: GridCell())
        self._lines:dict[tuple[int, int], vtk.vtkPoints] = {}
    
    def get_line_from_timestep(self, timestep:int, lineid:int):
        return self._lines.get((timestep, lineid), None)
    
    def add_line(self, timestep:int, lineid:int, points:vtk.vtkPoints, keys:np.ndarray):
        if (timestep, lineid) in self._lines:
            print(f"Cannot add line (timestep, lineid): already exists")
            return
        self._lines[(timestep, lineid)] = points
        
        for key, pointidx in zip(keys, range(points.GetNumberOfPoints()), strict=True):
            self._add_vertex(timestep=timestep, key=key, lineid=lineid, point=points.GetPoint(pointidx))

    def _add_vertex(self, timestep:int, key:int, lineid:int, point:tuple[float,float,float]):
        cell = self.grid[key]
        vinfo = cell.vertex_info[timestep]
        vinfo[0] = np.append(vinfo[0], [point], axis=0)
        if lineid not in vinfo[1]:
            cell.temporal_importance += 1
        vinfo[1] = np.append(vinfo[1], [lineid], axis=0)