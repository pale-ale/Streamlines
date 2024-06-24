from vectorFieldSource import ImageDataVectorField
from vectorFieldDefs import StraightVectorField, SourceVectorField, SinkVectorField
from paraview.util.vtkAlgorithm import smproxy, smproperty
import numpy as np

@smproxy.source(label="Straight Vector Field Source")
@smproperty.input(name="Input")
class StraightVectorFieldSource(ImageDataVectorField):
    def __init__(self):
        super().__init__(StraightVectorField(np.array([1.0, 0.0, 0.0])))
        self._vectorFieldDef: StraightVectorField

    @smproperty.doublevector(Name="Direction", default_values=[1.0, 0.0, 0.0])
    def SetDirection(self, x, y, z):
        self._vectorFieldDef.direction = np.array([x,y,z])
        self._vectors = self._vectorFieldDef.get_vector(self._points)
        self.Modified()


@smproxy.source(label="Source Vector Field Source")
@smproperty.input(name="Input")
class SourceVectorFieldSource(ImageDataVectorField):
    def __init__(self):
        super().__init__(SourceVectorField())
        self._vectorFieldDef: SourceVectorField
    
    @smproperty.doublevector(Name="Source Position", default_values=[1.0, 1.0, 1.0])
    def SetSourcePosition(self, x, y, z):
        self._vectorFieldDef.source_position = np.array([x,y,z])
        self._vectors = self._vectorFieldDef.get_vector(self._points)
        self.Modified()


@smproxy.source(label="Sink Vector Field Source")
@smproperty.input(name="Input")
class SinkVectorFieldSource(ImageDataVectorField):
    def __init__(self):
        super().__init__(SinkVectorField())
        self._vectorFieldDef: SinkVectorField
    
    @smproperty.doublevector(Name="Sink Position", default_values=[1.0, 1.0, 1.0])
    def SetSinkPosition(self, x, y, z):
        self._vectorFieldDef.sink_position = np.array([x,y,z])
        self._vectors = self._vectorFieldDef.get_vector(self._points)
        self.Modified()

# @smproxy.source(label="Curved Vector Field Source")
# @smproperty.input(name="Input")
# class CurvedVectorFieldSource(VTKPythonAlgorithmBase):
#     def __init__(self, nInputPorts=0, inputType='vtkDataSet', nOutputPorts=1, outputType='vtkImageData'):
#         VTKPythonAlgorithmBase.__init__(self,
#                 nInputPorts=0,
#                 nOutputPorts=1,
#                 outputType='vtkImageData')
#         self.z_angle = 0
#         size = 10
#         self.xsize, self.ysize, self.zsize = size, size, size
#         self._vectorFieldDef = StraightVectorField(np.array([1.0,0,0]))
#         self._points = self._generate_points()
#         self._vectors = self._generate_vectors(self._points)
    
#     @property
#     def points(self):
#         return self._points
    
#     @property
#     def vectors(self):
#         return self._vectors

#     def rotate_z(self, coords, angle):
#         matrix = np.array([
#             [np.cos(angle), -np.sin(angle), 0],
#             [np.sin(angle), np.cos(angle), 0],
#             [0, 0, 1],
#         ])
#         return (np.asmatrix(matrix) * np.asmatrix(coords).T).T
   
#     @smproperty.doublevector(Name="Rotation", default_values=1.0)
#     def SetRotation(self, rotation):
#         self.z_angle = rotation
#         self.Modified()
    
#     @smproperty.doublevector(Name="Direction", default_values=[1.0, 0.0, 0.0])
#     def SetDirection(self, x, y, z):
#         self._vectorFieldDef.direction = np.array([x,y,z])
#         self.Modified()

#     def RequestData(self, request, inInfo, outInfo):
#         mesh:vtkImageData = dsa.WrapDataObject(vtkImageData.GetData(outInfo))
#         mesh.SetSpacing(.2, .2, .2)
#         mesh.SetDimensions(self.xsize, self.ysize, self.zsize)
#         mesh.GetPointData().SetVectors(dsa.numpyTovtkDataArray(self.vectors, "velocity"))
#         return 1

#     def RequestInformation(self, request, inInfo, outInfo:vtk.vtkInformationVector):
#         executive = self.GetExecutive()
#         exts = (0, self.xsize-1, 0, self.ysize-1, 0, self.zsize-1)
#         outInfo = executive.GetOutputInformation(0)
#         outInfo.Set(executive.WHOLE_EXTENT(), *exts)
#         return 1

#     def _generate_points(self) -> np.ndarray:
#         xs, ys, zs = np.linspace(0, 1, self.xsize), np.linspace(0, 1, self.ysize), np.linspace(0, 1, self.zsize)
#         ptcount = xs.size*ys.size*zs.size
#         return np.reshape(np.meshgrid(xs, ys, zs, indexing="ij"), (3, ptcount)).T
    
#     def _generate_vectors(self, points:np.ndarray) -> np.ndarray:
#         return self._vectorFieldDef.get_vector(points).reshape((1000,3))