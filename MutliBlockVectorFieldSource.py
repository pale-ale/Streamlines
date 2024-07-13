from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vectorFieldDefs import VectorFieldBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from paraview.util.vtkAlgorithm import smproxy, smproperty

import numpy as np
import vtk

class VectorFieldSourceBase(VTKPythonAlgorithmBase):
    def __init__(self, vectorFieldDef: VectorFieldBase, nInputPorts: int, inputType: str, nOutputPorts: int, outputType: str, bounds: tuple[int, int, int], dimensions: tuple[int, int, int]):
        super().__init__(nInputPorts=nInputPorts, inputType=inputType,
                         nOutputPorts=nOutputPorts, outputType=outputType)
        self._vectorFieldDef = vectorFieldDef
        self._bounds = np.array(bounds, dtype=float)
        self._dimensions = np.array(dimensions, dtype=int)
        self.constrain_to_2d = False
        self.points = self._generate_points()

    def RequestInformation(self, request, inInfo, outInfo: vtk.vtkInformationVector):
        executive = self.GetExecutive()
        exts = tuple(item for pair in zip([0,0,0], [i-1 for i in self._get_dimensions_constrained()]) for item in pair)
        outInfo = executive.GetOutputInformation(0)
        outInfo.Set(executive.WHOLE_EXTENT(), *exts)
        outInfo.Remove(executive.TIME_STEPS())
        outInfo.Remove(executive.TIME_RANGE())
        timesteps = [float(ts[0]) for ts in self.timestep_generator()]
        for t in timesteps:
            outInfo.Append(executive.TIME_STEPS(), t)
        outInfo.Append(executive.TIME_RANGE(), timesteps[0])
        outInfo.Append(executive.TIME_RANGE(), timesteps[-1])
        return 1

    def RequestData(self, request: vtk.vtkInformation, inInfo: vtk.vtkInformationVector, outInfo: vtk.vtkInformationVector):
        executive = self.GetExecutive()
        oo = outInfo.GetInformationObject(0)
        if oo.Has(executive.UPDATE_TIME_STEP()):
            time = oo.Get(executive.UPDATE_TIME_STEP())
        else:
            time = 0
        mesh: vtk.vtkImageData = vtk.vtkImageData.GetData(outInfo)
        mesh.SetDimensions(self._get_dimensions_constrained())
        with np.errstate(divide="ignore", invalid="ignore"):
            spacing = self._get_bounds_constrained() / (self._get_dimensions_constrained() - (1,1,1))
        mesh.SetSpacing(np.nan_to_num(spacing))
        pdata = mesh.GetPointData()
        vectors = self._vectorFieldDef.get_vector(self.points, time)
        pdata.SetVectors(
            dsa.numpyTovtkDataArray(vectors, "velocity")
        )
        return 1

    def _generate_points(self) -> np.ndarray:
        bz, by, bx = self._get_bounds_constrained()
        dz, dy, dx = self._get_dimensions_constrained()
        xs, ys, zs = np.linspace(0, bx, dx), np.linspace(0, by, dy), np.linspace(0, bz, dz)
        ptcount = xs.size*ys.size*zs.size
        np_pts = np.reshape(np.meshgrid(
            xs, ys, zs, indexing="ij"), (3, ptcount)).T
        # reorder the array due to meshgrid order differing from imagedata
        return np.flip(np_pts, 1)

    def _get_dimensions_constrained(self):
        dims = self._dimensions.copy()
        if self.constrain_to_2d:
            dims[2] = 1
        return dims

    def _get_bounds_constrained(self):
        return self._bounds if not self.constrain_to_2d else np.array((*self._bounds[:2], 0), dtype=float)

    def timestep_generator(self):
        raise NotImplementedError()


@smproxy.source(label="Straight Vector Field Source")
@smproperty.input(name="Input")
class StraightVectorFieldSource(VectorFieldSourceBase):
    def __init__(self):
        from vectorFieldDefs import StraightVectorField
        super().__init__(StraightVectorField(), 0, "vtkDataObject", 1, "vtkImageData", bounds=(1,1,1), dimensions=(10,10,1))
        self._vectorFieldDef: StraightVectorField
    
    @smproperty.doublevector(Name="Bounds", default_values=[10.0, 10.0, 0.0])
    def SetBounds(self, x, y, z):
        self._bounds = np.array([x,y,z])
        self.points = self._generate_points()
        self.Modified()
    
    @smproperty.intvector(Name="Dimensions", default_values=[11, 11, 1])
    def SetDimensions(self, x, y, z):
        self._dimensions = np.array([x,y,z])
        self.points = self._generate_points()
        self.Modified()
    
    @smproperty.xml("""
    <IntVectorProperty name="PadData"
        label="Constrain to XY-Plane"
        command="SetConstrainTo2D"
        number_of_elements="1"
        default_values="1">
        <BooleanDomain name="bool" />
    </IntVectorProperty>
    """)
    def SetConstrainTo2D(self, val):
        '''Set attribute to signal whether data should be padded when RequestData is called
        '''
        self.constrain_to_2d = bool(val)
        self.points = self._generate_points()
        self.Modified()
    
    def timestep_generator(self):
        yield [0]



@smproxy.source(label="Moving Sink Vector Field Source")
@smproperty.input(name="Input")
class MovingSinkVectorFieldSource(VectorFieldSourceBase):
    def __init__(self):
        from vectorFieldDefs import SinkVectorField
        super().__init__(SinkVectorField(), 0, "vtkDataObject", 1, "vtkImageData", bounds=(1,1,1), dimensions=(10,10,1))
        self._vectorFieldDef: SinkVectorField
        self._vectorFieldDef.sink_start_position = np.array([0.0, 0.5, 0.5])
        self._vectorFieldDef.sink_stop_position = np.array([1.0, 0.5, 0.5])
        self.steps = 1
    
    @smproperty.doublevector(Name="Sink Start", default_values=[0.0, 0.0, 0.0])
    def SetSinkStart(self, x, y, z):
        self._vectorFieldDef.sink_start_position = np.array([x,y,z])
        self.Modified()

    @smproperty.doublevector(Name="Sink Stop", default_values=[1.0, 0.0, 0.0])
    def SetSinkStop(self, x, y, z):
        self._vectorFieldDef.sink_stop_position = np.array([x,y,z])
        self.Modified()
    
    @smproperty.doublevector(Name="Bounds", default_values=[10.0, 10.0, 0.0])
    def SetBounds(self, x, y, z):
        self._bounds = np.array([x,y,z])
        self.points = self._generate_points()
        self.Modified()
    
    @smproperty.intvector(Name="Dimensions", default_values=[11, 11, 1])
    def SetDimensions(self, x, y, z):
        self._dimensions = np.array([x,y,z])
        self.points = self._generate_points()
        self.Modified()
    
    @smproperty.intvector(Name="Sink Steps", default_values=[1])
    def SetSinkSteps(self, steps):
        self.steps = steps
        self.Modified()
    
    @smproperty.xml("""
    <IntVectorProperty name="PadData"
        label="Constrain to XY-Plane"
        command="SetConstrainTo2D"
        number_of_elements="1"
        default_values="1">
        <BooleanDomain name="bool" />
    </IntVectorProperty>
    """)
    def SetConstrainTo2D(self, val):
        '''Set attribute to signal whether data should be padded when RequestData is called
        '''
        self.constrain_to_2d = bool(val)
        self.points = self._generate_points()
        self.Modified()
    
    def timestep_generator(self):
        yield from enumerate(np.linspace(self._vectorFieldDef.sink_start_position, self._vectorFieldDef.sink_stop_position, self.steps))


@smproxy.source(label="Double Orbit Vector Field Source")
@smproperty.input(name="Input")
class DoubleOrbitVectorFieldSource(VectorFieldSourceBase):
    def __init__(self):
        from vectorFieldDefs import DoubleOrbitVectorField
        super().__init__(DoubleOrbitVectorField(), 0, "vtkDataObject", 1, "vtkImageData", bounds=(1,1,1), dimensions=(10,10,1))
        self._vectorFieldDef: DoubleOrbitVectorField
    
    @smproperty.doublevector(Name="Center 1", default_values=[0.0, 0.0, 0.0])
    def SetCenter1(self, x, y, z):
        self._vectorFieldDef.center_position_1 = np.array([x,y,z])
        self.Modified()
    
    @smproperty.doublevector(Name="Center 2", default_values=[0.0, 10.0, 0.0])
    def SetCenter2(self, x, y, z):
        self._vectorFieldDef.center_position_2 = np.array([x,y,z])
        self.Modified()

    @smproperty.doublevector(Name="Bounds", default_values=[10.0, 10.0, 0.0])
    def SetBounds(self, x, y, z):
        self._bounds = np.array([x,y,z])
        self.points = self._generate_points()
        self.Modified()
    
    @smproperty.intvector(Name="Dimensions", default_values=[11, 11, 1])
    def SetDimensions(self, x, y, z):
        self._dimensions = np.array([x,y,z])
        self.points = self._generate_points()
        self.Modified()
    
    @smproperty.xml("""
    <IntVectorProperty name="PadData"
        label="Constrain to XY-Plane"
        command="SetConstrainTo2D"
        number_of_elements="1"
        default_values="1">
        <BooleanDomain name="bool" />
    </IntVectorProperty>
    """)
    def SetConstrainTo2D(self, val):
        '''Set attribute to signal whether data should be padded when RequestData is called
        '''
        self.constrain_to_2d = bool(val)
        self.points = self._generate_points()
        self.Modified()
    
    def timestep_generator(self):
        yield [0]

