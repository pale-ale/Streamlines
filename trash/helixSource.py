from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

import math
import numpy as np
import vtk
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa

@smproxy.source(label="Helix Source")
@smproperty.input(name="Input")
class HelixSource(VTKPythonAlgorithmBase):
   def __init__(self, nInputPorts=0, inputType='vtkDataSet', nOutputPorts=1, outputType='vtkPolyData'):
      VTKPythonAlgorithmBase.__init__(self,
                nInputPorts=0,
                nOutputPorts=1,
                outputType='vtkPolyData')
      self.rounds = 3.0
   
   @smproperty.doublevector(Name="Rounds", default_values=2.0)
   def SetRounds(self, rounds):
      self.rounds = rounds
      self.Modified()

   def RequestData(self, request, inInfo, outInfo):
      numPts = 80  # Points along Helix
      length = 8.0 # Length of Helix
      rounds = self.rounds # Number of times around

      index = np.arange(0, numPts, dtype=np.int32)
      scalars = index * rounds * 2 * math.pi / numPts
      x = index * length / numPts
      y = np.sin(scalars)
      z = np.cos(scalars)

      coordinates = algs.make_vector(x, y, z)
      pts = vtk.vtkPoints()
      pts.SetData(dsa.numpyTovtkDataArray(coordinates, 'Points'))

      output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))

      output.SetPoints(pts)

      output.PointData.append(index, 'Index')
      output.PointData.append(scalars, 'Scalars')

      ptIds = vtk.vtkIdList()
      ptIds.SetNumberOfIds(numPts)
      for i in range(numPts):
         ptIds.SetId(i, i)

      output.Allocate(1, 1)

      output.InsertNextCell(vtk.VTK_POLY_LINE, ptIds)
      return 1

