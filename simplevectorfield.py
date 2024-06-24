from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from paraview.util.vtkAlgorithm import smproxy, smproperty
from vectorFieldDefs import SinkVectorField

import numpy as np
import vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from paraview import util


@smproxy.source(label="Simple Sink Vector Field Source")

class VectorFieldSourceBase(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkImageData")
        self._vectorFieldDef = SinkVectorField()
        self._bounds = np.array((1,1,1), dtype=float)
        self._extents = np.array((1,1,1), dtype=int)
        self.constrain_to_2d = False
        self.points = self._generate_points()

    def RequestInformation(self, request, inInfo, outInfo: vtk.vtkInformationVector):
        executive = self.GetExecutive()
        # print("Req:", request, "In:", inInfo, "Out:", outInfo)
        self.exts = tuple(item for pair in zip([0,0,0], [i for i in self._get_extents_constrained()]) for item in pair)
        outInfo = executive.GetOutputInformation(0)
        print(self.exts)
        util.SetOutputWholeExtent(self, self.exts)
        # outInfo.Set(executive.WHOLE_EXTENT(), *exts)
        # outInfo.Set(vtk.vtkImageData.ORIGIN(), -1,0,0)
        # imgdata = vtk.vtkImageData.GetData(outInfo)
        # print("Req:", request, "In:", inInfo, "Out:", outInfo)
        # print(imgdata.GetExtent())
        # imgdata.SetSpacing(1,1,1)
    #     outInfo.Remove(executive.TIME_STEPS())
    #     outInfo.Remove(executive.TIME_RANGE())
    #     timesteps = [float(ts[0]) for ts in self.timestep_generator()]
    #     for t in timesteps:
    #         outInfo.Append(executive.TIME_STEPS(), t)
    #     outInfo.Append(executive.TIME_RANGE(), timesteps[0])
    #     outInfo.Append(executive.TIME_RANGE(), timesteps[-1])
        return 1

    def RequestData(self, request: vtk.vtkInformation, inInfo: vtk.vtkInformationVector, outInfo: vtk.vtkInformationVector):
        mesh = vtkImageData.GetData(outInfo)
        spacing = self._get_bounds_constrained() / self._get_extents_constrained()
        print("set spacing to", spacing)
        mesh.SetSpacing(spacing)
        mesh.SetDimensions(self._get_extents_constrained())
        vecs = dsa.numpyTovtkDataArray(self._vectorFieldDef.get_vector(self.points, 0), "velocity")
        mesh.GetPointData().SetVectors(vecs)
        print(mesh.GetNumberOfPoints())
        return 1
   
    # def RequestData(self, request: vtk.vtkInformation, inInfo: vtk.vtkInformationVector, outInfo: vtk.vtkMultiBlockDataSet):
    #     multiBlock: vtk.vtkMultiBlockDataSet = dsa.WrapDataObject(
    #         vtk.vtkMultiBlockDataSet.GetData(outInfo))
    #     self.prepare_data(multiBlock)
    #     for timestep in self.timestep_generator():
    #         self.consume_timestep(timestep, multiBlock)
    #     self.finalize_data(multiBlock)
    #     return 1

    def _generate_points(self) -> np.ndarray:
        bz, by, bx = self._get_bounds_constrained()
        ez, ey, ex = self._get_extents_constrained()
        xs, ys, zs = np.linspace(0, bx, ex + 1), np.linspace(0, by, ey + 1), np.linspace(0, bz, ez + 1)
        ptcount = 8 #xs.size*ys.size*zs.size
        np_pts = np.reshape(np.meshgrid(
            xs, ys, zs, indexing="ij"), (3, ptcount)).T
        # reorder the array due to meshgrid order differing from imagedata
        return np.flip(np_pts, 1)

    def _get_extents_constrained(self):
        return np.array([1,1,1], dtype=int)
        return self._extents if not self.constrain_to_2d else np.array((*self._extents[:2], 1), dtype=int)

    def _get_bounds_constrained(self):
        return np.array([1,1,1], dtype=float)
        return self._bounds if not self.constrain_to_2d else np.array((*self._bounds[:2], 0), dtype=float)

    def timestep_generator(self):
        raise NotImplementedError()

    def consume_timestep(self, timestep, dataset: vtk.vtkMultiBlockDataSet):
        raise NotImplementedError()

    def prepare_data(self, dataset: vtk.vtkMultiBlockDataSet):
        pass

    def finalize_data(self, dataset: vtk.vtkMultiBlockDataSet):
        pass
