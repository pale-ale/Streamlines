from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

from paraview.util.vtkAlgorithm import smproxy, smproperty

import math
import numpy as np
import vtk

@smproxy.source(label="Time Dependent Vector Field Source")
@smproperty.input(name="Input")
class TimeVectorFieldSource(VTKPythonAlgorithmBase):
    def __init__(self, nInputPorts=0, inputType='vtkDataSet', nOutputPorts=1, outputType='vtkMultiBlockDataSet'):
        VTKPythonAlgorithmBase.__init__(self,
                nInputPorts=0,
                nOutputPorts=1,
                outputType='vtkMultiBlockDataSet')
        self.rotation_start = 0
        self.rotation_stop = 1
        self.rotation_steps = 1
        size = 10
        self.xsize, self.ysize, self.zsize = size, size, size
        self.constrain_to_2d = False
        self.np_pts = self.get_points()

    def get_points(self) -> np.ndarray:
        xs, ys, zs = np.linspace(0, 1, self.xsize), np.linspace(0, 1, self.ysize), np.linspace(0, 1, self.zsize)
        if self.constrain_to_2d:
            zs = 0
        ptcount = xs.size*ys.size*zs.size
        return np.reshape(np.meshgrid(xs, ys, zs, indexing="ij"), (3, ptcount)).T
    
    def get_vectors(self, rotation:float = 0) -> np.ndarray:
        arr = np.flip(self.np_pts, 1) # reorder the array due to meshgrid order differing from imagedata
        arr_slice = arr[0:(self.xsize*self.ysize)] # truncate to x0,...,xn,y0,...,yn
        arr = np.tile(arr_slice, (self.zsize, 1)) # tile in z plane, self.zsize number of times
        return self.rotate_z(arr, rotation)

    def rotate_z(self, coords, angle):
        matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        return (np.asmatrix(matrix) * np.asmatrix(coords).T).T
   
    @smproperty.doublevector(Name="Rotation (float, float, int)", default_values=(1.0, np.pi, 3))
    def SetRotation(self, start:float, stop:float, steps:float):
      self.rotation_start, self.rotation_stop, self.rotation_steps = start, stop, int(steps)
      self.Modified()

    def RequestData(self, request:vtk.vtkInformation, inInfo:vtk.vtkInformationVector, outInfo:vtk.vtkMultiBlockDataSet):
        print("Request:", request, "InInfo:", inInfo, "OutInfo.", outInfo)
        multiBlock:vtk.vtkMultiBlockDataSet = dsa.WrapDataObject(vtk.vtkMultiBlockDataSet.GetData(outInfo))
        multiBlock.SetNumberOfBlocks(self.rotation_steps)
        rotations = np.linspace(self.rotation_start, self.rotation_stop, self.rotation_steps)
        vtk_timestamps = dsa.numpyTovtkDataArray(rotations)
        vtk_timestamps.SetName("Rotation")
        for idx, rotation in enumerate(rotations):
            mesh = vtk.vtkImageData()
            mesh.SetSpacing(1/self.xsize, 1/self.ysize, 1/self.zsize)
            mesh.SetDimensions(self.xsize, self.ysize, self.zsize)
            pdata = mesh.GetPointData()
            pdata.SetVectors(dsa.numpyTovtkDataArray(self.get_vectors(rotation), "velocity"))
            multiBlock.SetBlock(idx, mesh)
        it = multiBlock.NewIterator()
        while not it.IsDoneWithTraversal():
            meta:vtk.vtkInformation = it.GetCurrentMetaData()
            meta.Set(vtk.vtkMultiBlockDataSet.NAME(), f"test{it.GetCurrentFlatIndex()}")
            it.GoToNextItem()
        print("Reloaded.")
        return 1

    def RequestInformation(self, request, inInfo, outInfo:vtk.vtkInformationVector):
        # executive = self.GetExecutive()
        # outInfo = executive.GetOutputInformation(0)
        # exts = (0, self.xsize-1, 0, self.ysize-1, 0, self.zsize-1)
        # outInfo.Set(executive.WHOLE_EXTENT(), *exts)
        return 1
