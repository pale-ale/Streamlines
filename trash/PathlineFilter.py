from vtkmodules.vtkCommonDataModel import vtkDataSet, vtkImageData
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase, VTKAlgorithm
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtk import vtkInterpolatedVelocityField, vtkPointInterpolator, vtkPointSet, vtkGaussianKernel

from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

import math
import numpy as np
import vtk
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support

@smproxy.filter(label="PathlineFilter")
@smproperty.input(name="Input")
class PathlineFilter(VTKPythonAlgorithmBase):
    def __init__(self, nInputPorts=1, inputType='vtkImageData', nOutputPorts=1, outputType='vtkPolyData'):
        VTKPythonAlgorithmBase.__init__(self,
            nInputPorts=1,
            inputType='vtkImageData',
            nOutputPorts=1,
            outputType='vtkPolyData')
        self.stepcount = 10
        self.stepsizescale = .05
        self.seedoffset = np.zeros(3)
        self.neighbor_distance_min = .05
        self.neighbor_test_distance = .3
        self.raster:dict[tuple[int,int,int],int] = {}
        self.streamlineId = 0
        self.neighbor_count = 0
        self.max_iterations = 10
    
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

    def RequestData(self, request, inInfo:tuple[vtk.vtkInformationVector], outInfo):
        imageData = vtkImageData.GetData(inInfo[0])
        seeds = self.get_initial_seeds()
        seeds += self.seedoffset
        vpi = vtkPointInterpolator()
        vpi.SetKernel(vtkGaussianKernel())
        vpi.SetSourceData(imageData)
      
        concat_pts = vtk.vtkPoints()
        line_pointids:list[vtk.vtkIdList] = []
        output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))
        idx = 0
        imageData.ComputeBounds()

        seed_candidates:list[np.ndarray] = [seeds[0]]
        previous_nppts = None
        while seed_candidates and len(line_pointids) < self.max_iterations:
            if not seed_candidates:
                break
            seed = seed_candidates.pop(0)
            pts, ids = self.make_streamline(seed, vpi, idx, imageData.GetBounds())
            if pts is None:
                continue
            nppts = numpy_support.vtk_to_numpy(pts.GetData())
            if previous_nppts is None:
                previous_nppts = nppts
            idx += pts.GetNumberOfPoints()
            next_candidates = []
            # Keep points we didnt cover with the current line, discard the rest
            for i, candidate in enumerate(seed_candidates):
                if all(np.linalg.norm(nppts - candidate, axis=1) >= self.neighbor_distance_min):
                    next_candidates.append(candidate)
            # Fetch new points from our vincinity
            new_candidates = self.get_neighbor_points(pts, self.neighbor_test_distance, self.neighbor_count)
            # Discard points too close to ourselves or the parent line (i.e. the one we just came from)
            # Parent check probably unnecessary, since we are our own parent?
            for candidate in new_candidates:
                close_to_self = any(np.linalg.norm(nppts - candidate, axis=1) <= self.neighbor_distance_min)
                close_to_parent = any(np.linalg.norm(previous_nppts - candidate, axis=1) <= self.neighbor_distance_min)
                if not (close_to_self or close_to_parent):
                    next_candidates.append(candidate)
            concat_pts.InsertPoints(concat_pts.GetNumberOfPoints(), pts.GetNumberOfPoints(), 0, pts)
            line_pointids.append(ids)
            seed_candidates = next_candidates
        output.SetPoints(concat_pts)
        output.Allocate(1,1)
        for ids in line_pointids:
            output.InsertNextCell(vtk.VTK_POLY_LINE, ids)
        print("Reloaded", idx)
        return 1

    def get_initial_seeds(self) -> np.ndarray:
        return np.array([(.0,.0,.0)], dtype=float)

    def normalize(self, vecs:np.ndarray):
        return vecs / np.asmatrix(np.linalg.norm(vecs, axis=1)).T

    def roots_of_unity(self, k:int):
        exponents = np.arange(k) * 1j * np.pi * 2.0 / k
        return np.power(np.e, exponents, dtype=complex) 

    def get_neighbor_points(self, linepts:vtk.vtkPoints, distance:float, neighbor_count:int):
        assert neighbor_count > 0
        points:np.ndarray = numpy_support.vtk_to_numpy(linepts.GetData())
        deltas = points[1:] - points[:-1]
        norm_deltas = self.normalize(deltas)
        identity = np.identity(3)
        first_normals:np.ndarray = np.zeros_like(deltas)
        first_normals[:,0] = 1.0
        dots = np.einsum('ij, ij->i', first_normals, norm_deltas)
        first_normals[dots >= .99] = identity[1] # if we are too close to [1,0,0], choose [0,1,0]
        first_normals = self.normalize(first_normals)
        second_normals = self.normalize(np.cross(deltas, first_normals))
        for i in range(first_normals.shape[0]):
            q, _ = np.linalg.qr(np.vstack((norm_deltas[i], first_normals[i], second_normals[i])))
            first_normals[i], second_normals[i] = q.T[1], q.T[2]
        scaled_roots_of_unity = self.roots_of_unity(neighbor_count) * distance
        arrays = []
        for i in range(neighbor_count):
            re = np.real(scaled_roots_of_unity[i])
            im = np.imag(scaled_roots_of_unity[i])
            arrays.append(points[1:] + re * first_normals + im * second_normals)
        return np.copy(np.concatenate(arrays))
    
    def make_streamline(self, seed, vpi:vtk.vtkPointInterpolator, idstart:int, bounds:tuple):
        points_to_interpolate = vtk.vtkPoints()
        data_to_interpolate = vtk.vtkPointSet()
        data_to_interpolate.SetPoints(points_to_interpolate)
        vpi.SetInputData(data_to_interpolate)
        
        backward_points = vtk.vtkPoints()
        backward_points.SetNumberOfPoints(self.stepcount)
        backward_ids = vtk.vtkIdList()
        backward_start_idx = 0
        current_pos = np.copy(seed)
        if self.raster.get(self.get_raster_key(*current_pos), None) is not None:
            return None, None
        for i in range(self.stepcount):
            backward_points.InsertPoint(self.stepcount - 1 - i, current_pos)
            backward_ids.InsertNextId(i)
            points_to_interpolate.SetNumberOfPoints(1) #why do i need to call this every time?
            points_to_interpolate.SetPoint(0, current_pos)
            vpi.Update()
            out:vtkPointSet = vpi.GetOutput()
            velocities = out.GetPointData().GetArray(0)
            velocity:np.ndarray = numpy_support.vtk_to_numpy(velocities)[0]
            current_pos -= velocity * self.stepsizescale
            colliding_streamline = self.raster.get(self.get_raster_key(*current_pos), None)
            if not PathlineFilter.inside_domain(current_pos, *bounds) or \
                colliding_streamline not in (None, self.streamlineId):
                backward_start_idx = self.stepcount - i - 1
                break
            else:
                self.raster[self.get_raster_key(*current_pos)] = self.streamlineId

        backward_count = backward_points.GetNumberOfPoints() - backward_start_idx
        forward_points = vtk.vtkPoints()
        forward_ids = vtk.vtkIdList()
        current_pos = np.copy(seed)
        for i in range(self.stepcount):
            forward_points.InsertNextPoint(current_pos)
            forward_ids.InsertNextId(backward_count + i) 
            points_to_interpolate.SetNumberOfPoints(1) #why do i need to call this every time?
            points_to_interpolate.SetPoint(0, current_pos)
            vpi.Update()
            out:vtkPointSet = vpi.GetOutput()
            velocities = out.GetPointData().GetArray(0)
            velocity = numpy_support.vtk_to_numpy(velocities)[0]
            current_pos += velocity * self.stepsizescale
            colliding_streamline = self.raster.get(self.get_raster_key(*current_pos), None)
            if not PathlineFilter.inside_domain(current_pos, *bounds) or \
                colliding_streamline not in (None, self.streamlineId):
                break
            else:
                self.raster[self.get_raster_key(*current_pos)] = self.streamlineId
        
        forward_count = forward_points.GetNumberOfPoints()
        points = vtk.vtkPoints()
        points.InsertPoints(0, backward_count, backward_start_idx, backward_points) # contains the seed point
        points.InsertPoints(backward_count, forward_count-1, 1, forward_points) # we dont want that twice
        
        ids = vtk.vtkIdList()
        for i in range(points.GetNumberOfPoints()):
            ids.InsertNextId(idstart + i)
        
        self.streamlineId += 1
        if points.GetNumberOfPoints() * self.stepsizescale < 1:
            return None, None
        return points, ids

    def get_raster_key(self, x:float, y:float, z:float):
        return int(x/self.neighbor_distance_min), int(y/self.neighbor_distance_min), int(z/self.neighbor_distance_min)

    @staticmethod
    def inside_domain(point, x1, x2, y1, y2, z1, z2):
        return all((l <= c <= r for (l,c,r) in zip((x1, y1, z1), point, (x2, y2, z2))))
