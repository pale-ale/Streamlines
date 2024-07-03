from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smproxy, smproperty
from paraview.simple import GetActiveView, GetDisplayProperties, FindSource, GetSources
import vtkmodules.numpy_interface.dataset_adapter as dsa
from scipy import fftpack

import time
import numpy as np
import math
import scipy
import scipy.ndimage
import vtk
import itertools
import enum
from PIL import Image, ImageDraw
import copy
from vtk.util import numpy_support
import cProfile


RASTER_RESOLUTION_SCALE = 3
HALO_RADIUS = 3
LINE_KILL_LENGTH = 0
LINE_START_LENGTH = 5
TARGET_BRIGHTNESS = 1.0

class Streamline:
    def __init__(self) -> None:
        self.seed: np.ndarray = None
        self.points = vtk.vtkPoints()
        self.backward_segments:int = LINE_START_LENGTH
        self.forward_segments:int = LINE_START_LENGTH
        self.segment_length = .1
        self.needs_reeval = True

    def get_end_points(self, n:int=0):
        """Return the (n+1)th point from the start and end points of the line (in that order)."""
        if self.points.GetNumberOfPoints() < 2:
            return None, None
        return np.array(self.points.GetPoint(n)), np.array(self.points.GetPoint(self.points.GetNumberOfPoints()-1-n))
    
    def get_midpoint(self):
        """Return the floor(n/2)th point, where n is the number of points of the streamline."""
        return np.array(self.points.GetPoint(self.points.GetNumberOfPoints() // 2), dtype=float)

    def get_end_probes(self):
        if self.points.GetNumberOfPoints() < 2:
            return None, None
        back_end, front_end = self.get_end_points()
        second_back, second_front = self.get_end_points(n=1)
        return 2.0 * back_end - second_back, 2.0 * front_end - second_front


class TimeGrid:
    def __init__(self, array_shape:tuple[int,int], key_gen) -> None:
        self._lines:dict[tuple[int, int], tuple[Streamline, np.ndarray]] = {}
        self.footprint_array = np.zeros(array_shape, dtype=float)
        self.low_pass_array = np.zeros(array_shape, dtype=float)
        self.energy_array = np.zeros(array_shape, dtype=float)
        self.energy_target = np.ndarray(array_shape, dtype=float)
        self.energy_target.fill(TARGET_BRIGHTNESS)
        self.key_gen = key_gen
        self._on_reject = None
        self.lower_bound:np.ndarray = None
        self.upper_bound:np.ndarray = None
        self.bound_delta:np.ndarray = None
        self.new_line_segment_length = 1.0
        self.max_join_distance = 1.0
    
    def get_raster_spacing(self):
        delta = self.upper_bound - self.lower_bound
        return delta[0] / self.footprint_array.shape[0]

    def get_line_from_timestep(self, timestep:int, lineid:int):
        return self._lines.get((timestep, lineid), None)
    
    def update_line(self, timestep:int, line:Streamline, needs_reeval=True):
        # self._assert_needs_accept_or_reject()
        self._lines[(timestep, id(line))] = line, self.get_line_footprint(line)
        line.segment_length = self.new_line_segment_length
        line.needs_reeval = needs_reeval

    def get_line_footprint(self, line:Streamline):
        arr = np.zeros_like(self.footprint_array)
        for idx in range(line.points.GetNumberOfPoints()):
            point = line.points.GetPoint(idx)
            x,y,z = self.key_gen(point)
            val = arr[x,y]
            arr[x,y] += 1.0 #if val == 0 else line.segment_length / 2
        return arr#.clip(0., 1.)
    
    def update_footprint_array(self):
        self.footprint_array.fill(0.0)
        for line, footprint in self._lines.values():
            self.footprint_array += footprint
    
    def move(self, key:tuple[int,int], offset:np.ndarray):
        self._assert_needs_accept_or_reject()
        line = self._lines[key][0]
        line.seed += offset
        line.needs_reeval = True
        self._on_reject = lambda: self._reset_move(line, offset)
    
    def change_length(self, key:tuple[int,int], change_front:int, change_back:int):
        self._assert_needs_accept_or_reject()
        line = self._lines[key][0]
        prev_fwd_segs, prev_bwd_segs = line.forward_segments, line.backward_segments
        line.forward_segments = max(0, prev_fwd_segs + change_front)
        line.backward_segments = max(0, prev_bwd_segs + change_back)
        line.needs_reeval = True
        self._on_reject = lambda: self._reset_change_length(
            line,
            line.forward_segments - prev_fwd_segs,
            line.backward_segments - prev_bwd_segs
        )
    
    def get_join_candidates(self, dist:float):
        pairs:list[tuple[Streamline, Streamline]] = []
        for l1, _ in self._lines.values():
            l1_start, l1_end = l1.get_end_points()
            if l1_end is None:
                continue
            for l2, _ in self._lines.values():
                l2_start, l2_end = l2.get_end_points()
                if l2_start is None or l1 == l2: continue
                if np.linalg.norm(np.array((l1_end)) - l2_start) < dist:
                    pairs.append((l1, l2))
        return pairs

    def join_lines(self, l1:Streamline, l2:Streamline):
        """Join the end of l1 and the start of l2."""
        self._assert_needs_accept_or_reject()
        self._lines.pop((0, id(l1)))
        self._lines.pop((0, id(l2)))
        l1_start, l1_end = l1.get_end_points()
        l2_start, l2_end = l2.get_end_points()
        combined = Streamline()
        l1_importance = l1.points.GetNumberOfPoints() / (l1.points.GetNumberOfPoints() + l2.points.GetNumberOfPoints())
        combined.seed = l1_importance * l1_end + (1-l1_importance) * l2_start
        combined.backward_segments = l1.points.GetNumberOfPoints()
        combined.forward_segments = l2.points.GetNumberOfPoints()
        # print("Joining", l1_end, l2_start, combined.seed, combined.forward_segments, combined.backward_segments)
        self.update_line(0, combined)
        self._on_reject = lambda: self._reset_merge(combined, l1, l2)
    
    def shatter(self):
        shards:list[Streamline] = []
        for line, _ in self._lines.values():
            points = line.points
            ptcount = points.GetNumberOfPoints()
            if ptcount < LINE_KILL_LENGTH:
                continue
            current_point_idx = LINE_START_LENGTH + 1
            while current_point_idx < ptcount:
                shard = Streamline()
                shard.seed = np.array(points.GetPoint(current_point_idx))
                shards.append(shard)
                current_point_idx += shard.forward_segments + shard.backward_segments + 1
        self._lines.clear()
        for shard in shards:
            self.update_line(0, shard)

    def reject(self):
        if self._on_reject is not None:
            self._on_reject()
            self._on_reject = None

    def accept(self):
        self._on_reject = None
   
    def _reset_move(self, line:Streamline, offset):
        line.seed -= offset
        line.needs_reeval = True
    
    def _reset_change_length(self, line:Streamline, fw_len, bw_len):
        line.forward_segments -= fw_len
        line.backward_segments -= bw_len
        line.needs_reeval = True

    def _assert_needs_accept_or_reject(self):
        assert self._on_reject is None

    def _reset_merge(self, new:Streamline, l1:Streamline, l2:Streamline):
        self._lines.pop((0, id(new)), None)
        self.update_line(0, l1)
        self.update_line(0, l2)


class Oracle:
    def __init__(self, grid:TimeGrid, key_gen) -> None:
        self.grid = grid
        self.key_gen = key_gen
    
    def make_suggestion(self):
        """Take an educated guess. Return the line, actionID, and values that are probably beneficial."""
        gridx, gridy = self.grid.low_pass_array.shape
        highest_desire = 0
        front = False
        l = None
        for (timestep, lineid), (line, footprint) in self.grid._lines.items():
            ##### Check if we want to grow or shrink
            if line.points.GetNumberOfPoints() < 2:
                continue
            front_probe_spot, back_probe_spot = line.get_end_probes()
            desire_front, desire_back = 0, 0
            if front_probe_spot is not None:
                ### Out of bounds, we just set the desires to 0, thereby skipping the respective side.
                fx, fy = self.key_gen(front_probe_spot)[:2]
                if (0 <= fx < gridx and 0 <= fy < gridy):
                    desire_front = 2*TARGET_BRIGHTNESS - self.grid.low_pass_array[fx, fy]
            if back_probe_spot is not None:
                ### Out of bounds, we just set the desires to 0, thereby skipping the respective side.
                bx, by = self.key_gen(back_probe_spot )[:2]
                if (0 <= bx < gridx and 0 <= by < gridy):
                    desire_back  = 2*TARGET_BRIGHTNESS - self.grid.low_pass_array[bx, by]
            if abs(desire_back) > abs(highest_desire):
                highest_desire = desire_back
                l = line
                front = False
            if abs(desire_front) > abs(highest_desire):
                highest_desire = desire_front
                l = line
                front = True
        back_change = int(np.sign(highest_desire)) * 3 if front else 0
        front_change = int(np.sign(highest_desire)) * 3 if not front else 0
        # print(front_probe_spot, back_probe_spot, front_change, back_change)
        if l is None:
            return None
        return lambda: self.grid.change_length((0, id(l)), front_change, back_change)


class COHERENCE_STRATEGY(enum.Enum):
    BIAS = 0
    MIDSEED = 1
    COMBINED = 2


@smproxy.filter(label="TurkBanksPathlineFilter")
@smproperty.xml("""
    <OutputPort index="0" name="Stream Lines" type="vtkPolyData"/>
    <OutputPort index="1" name="Energy Map" type="vtkImageData"/>
""")
@smproperty.input(name="Input")
class TurkBanksPathlineFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
            nInputPorts=1,
            inputType='vtkImageData',
            # inputType='vtkStructuredGrid',
            nOutputPorts=2,
            # outputType='vtkMultiBlockDataSet')
        )
        self.streamlineId = 0
        ##### Sub-components are init'd in RequestData
        self.timeGrid = None
        self.oracle = None
        self.filter = None
        ##### These are also set in RequestData
        self.bounds = None
        self.rasterx, self.rastery = -1, -1
        self.grid_halo_radius = 1
        self.filter_scale = 1.0
        ##### Time Coherence
        self.make_coherent = False
        self.coherence_strat:COHERENCE_STRATEGY = COHERENCE_STRATEGY.COMBINED
        self.last_frame_low_pass:np.ndarray = None # Needed for COHERENCE_STRATEGY.BIAS
        self.midseeds:np.ndarray = None # Needed for COHERENCE_STRATEGY.MIDSEED
    
    def FillOutputPortInformation(self, port, info):
        """Sets the default output type to OutputType."""
        if port == 0:
            info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkPolyData")
        elif port == 1:
            info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkImageData")
        return 1

    @smproperty.xml("""
    <IntVectorProperty name="Time Coherence"
        label="Cohere to previously selected frame"
        command="SetTimeCoherence"
        number_of_elements="1"
        default_values="0">
        <BooleanDomain name="bool" />
    </IntVectorProperty>
    """)
    def SetTimeCoherence(self, val):
        '''Enable/Disable time coherence.'''
        self.make_coherent = bool(val)
        self.last_frame_low_pass = None
        self.Modified()
    
    def GetUpdateTimestep(self):
        """Returns the requested time value, or None if not present"""
        executive = self.GetExecutive()
        outInfo = executive.GetOutputInformation(0)
        timestep = None
        all_timesteps = []
        if outInfo.Has(executive.UPDATE_TIME_STEP()):
            timestep = outInfo.Get(executive.UPDATE_TIME_STEP())
        if outInfo.Has(executive.TIME_STEPS()):
            all_timesteps = outInfo.Get(executive.TIME_STEPS())
        return timestep, all_timesteps

    def RequestInformation(self, request:vtk.vtkInformation, inInfo:tuple[vtk.vtkInformationVector], outInfo:vtk.vtkInformationVector):
        SDDP = vtk.vtkStreamingDemandDrivenPipeline
        inInfoObj = inInfo[0].GetInformationObject(0)
        xmin, xmax, ymin, ymax, zmin, zmax = inInfoObj.Get(SDDP.WHOLE_EXTENT())
        xsize, ysize = xmax - xmin, ymax - ymin
        self.rasterx = math.ceil((xmax - xmin) * RASTER_RESOLUTION_SCALE) + 1
        self.rastery = math.ceil((ymax - ymin) * RASTER_RESOLUTION_SCALE) + 1
        self.bounds = vtk.vtkImageData().GetData(inInfo[0])

        outInfo0 = outInfo.GetInformationObject(0)
        outInfo0.Set(SDDP.WHOLE_EXTENT(), xmin, xmax, ymin, ymax, zmin, zmax)
        outInfo1 = outInfo.GetInformationObject(1)
        outInfo1.Set(SDDP.WHOLE_EXTENT(), 0, self.rasterx - 1, 0, self.rastery - 1, 0, 0)
        return 1

    def Reset(self):
        self.timeGrid = None
        self.oracle = None

    def RequestData(self, request:vtk.vtkInformation, inInfo:tuple[vtk.vtkInformationVector], outInfo:vtk.vtkInformationVector):
        timestep, all_timesteps = self.GetUpdateTimestep()
        t = time.time()
        inImageData = vtk.vtkImageData.GetData(inInfo[0])
        inImageData.SetSpacing(1,1,1)
        inImageData.Modified()
        self.timeGrid = TimeGrid((self.rasterx, self.rastery), self.get_raster_key)
        self.timeGrid.lower_bound = np.array((inImageData.GetBounds()[::2]), dtype=float)
        self.timeGrid.upper_bound = np.array((inImageData.GetBounds()[1::2]), dtype=float)
        self.timeGrid.bound_delta = self.timeGrid.upper_bound - self.timeGrid.lower_bound
        
        #    self.timeGrid.target += (self.last_frame_low_pass.clip(0, 2*TARGET_BRIGHTNESS) - TARGET_BRIGHTNESS) * .4
        self.oracle = Oracle(self.timeGrid, self.get_raster_key)
        #### Configure the output ImageData object
        outImageData = vtk.vtkImageData.GetData(outInfo.GetInformationObject(1))
        outImageData.SetDimensions(self.rasterx, self.rastery, 1)
        outImageData.SetSpacing(*(self.timeGrid.bound_delta / (self.rasterx - 1, self.rastery - 1, 1)))
        #### Some dials to turn. This hopefully sets them into a decent starting position.
        self.timeGrid.new_line_segment_length = self.timeGrid.get_raster_spacing()
        self.grid_halo_radius = math.floor(HALO_RADIUS / self.timeGrid.get_raster_spacing())
        self.timeGrid.max_join_distance = HALO_RADIUS * .75
        self.join_allowance = self.grid_halo_radius * TARGET_BRIGHTNESS * 4
        print(f"\n\n{'== Pre-Configured Settings ==':^50}")
        print(f"{'HALO_RADIUS'          :<25}:{HALO_RADIUS         :>10}")
        print(f"{'LINE_KILL_LENGTH'     :<25}:{LINE_KILL_LENGTH    :>10}")
        print(f"{'LINE_START_LENGTH'    :<25}:{LINE_START_LENGTH   :>10}")
        print(f"{'RASTER_RESOLUTION_SCALE' :<25}:{RASTER_RESOLUTION_SCALE :>10}")
        print(f"{'TARGET_BRIGHTNESS'    :<25}:{TARGET_BRIGHTNESS   :>10}")
        print(f"{'COHERENCE'            :<25}:{self.coherence_strat.name if self.make_coherent else 'Disabled' :>10}")
        print(f"\n{'== Auto-Configured Settings ==':^50}")
        print(f"{'Raster Size'          :<25}:" + f"{str(self.rasterx) + ' * ' + str(self.rastery):>10}")
        print(f"{'Raster halo radius'   :<25}:{self.grid_halo_radius                :>10}")
        print(f"{'Line segment length'  :<25}:{self.timeGrid.new_line_segment_length:>10}")
        print(f"{'Line join allowance'  :<25}:{self.join_allowance                  :>10}")
        print(f"{'Line join radius'     :<25}:{self.timeGrid.max_join_distance      :>10.3f}")
        print(f"{'': >10}{'':=>30}{'': >10}\n\n")
        
        self.filter_scale = self.get_filter_scale()
        
        vpi = vtk.vtkPointInterpolator() # vtk.vtkProbeFilter()
        vpi.SetSourceData(inImageData)
        
        iterations = 0
        iterations = self.minimize_energy(vpi)
        # cProfile.runctx("energy, iterations = self.minimize_energy(vpi)", {}, {"self":self, "vpi":vpi})
        self.generate_streamlines(self.timeGrid, vpi)
        self.timeGrid.update_footprint_array()
        outPolyData = vtk.vtkPolyData.GetData(outInfo)
        self.draw_streamlines(self.timeGrid, outPolyData)
        low_pass_data = self.timeGrid.low_pass_array
        print("Min/Max brightness:", np.min(low_pass_data), np.max(low_pass_data))
        if self.last_frame_low_pass is not None:
            # Display the previous low-pass image if  we have one
            print("Min/Max old:", np.min(self.last_frame_low_pass), np.max(self.last_frame_low_pass))
            last_img_array = np.zeros((*self.last_frame_low_pass.shape, 3), dtype=np.uint8)
            last_img_array[:,:,0] = (self.last_frame_low_pass / 2).clip(0, 1) * 255
        print("Min/Max footprint:", np.min(self.timeGrid.footprint_array), np.max(self.timeGrid.footprint_array))
        print("Min/Max energy:", np.min(self.timeGrid.energy_array), np.max(self.timeGrid.energy_array))
        print(f"Reached final energy {self.energy_function()} after {iterations} steps.")
        print("Lines:", len(self.timeGrid._lines))
        print(f"Reloaded after {time.time() - t:.3f}s")
        current_img_array = np.zeros((*low_pass_data.shape, 3), dtype=np.uint8)
        current_img_array[:,:,1] = (low_pass_data / 2).clip(0,1) * 255
        summed = current_img_array
        if self.last_frame_low_pass is not None:
            summed += last_img_array
        blended = Image.fromarray(np.rot90(summed))
        # last_frame_img = Image.fromarray(np.rot90(last_img_array))
        # current_frame_img = Image.fromarray(np.rot90(current_img_array))
        # last_frame_img.alpha_composite(current_frame_img)
        # last_frame_img.show()
        if timestep is not None and timestep < 1e-6:
            print("Reset for first frame.")
            self.last_frame_low_pass = None
            self.midseeds = None
            return 1
        if self.make_coherent:
            if self.coherence_strat in (COHERENCE_STRATEGY.BIAS, COHERENCE_STRATEGY.COMBINED):
                pass
            if self.coherence_strat in (COHERENCE_STRATEGY.MIDSEED, COHERENCE_STRATEGY.COMBINED):
                if self.midseeds is not None:
                    raster_keys = self.get_raster_key(self.midseeds)[:,:2]
                    print("Adding midpoints")
                    summed[raster_keys[:,0], raster_keys[:,1]] = [255, 0 ,255]
                    summed.clip(0, 255, out=summed)
                    for seed in self.midseeds:
                        x,y,z = self.get_raster_key(seed)
                        point = np.array((x,y))
                        p1 = (point - (1,1)).clip((0,0), (self.rasterx, self.rastery))
                        p2 = (point + (1,1)).clip((0,0), (self.rasterx, self.rastery))
                        pts = [*p1, *p2]
                        ImageDraw.Draw(blended).ellipse(pts, fill=(255,0,255), width=2)
                if len(self.timeGrid._lines) > 0:
                    self.timeGrid.shatter()
                    self.generate_streamlines(self.timeGrid, vpi)
                    self.midseeds = np.vstack([line.get_midpoint() for line, _ in self.timeGrid._lines.values()])
                else:
                    self.midseeds = None
        self.last_frame_low_pass = low_pass_data
        flipped = np.moveaxis(summed, (0,1), (1,0)).reshape(self.rasterx * self.rastery, 3)
        outImageData.GetPointData().SetVectors(dsa.numpyTovtkDataArray(flipped))
        # blended.save(f"/home/alba/projects/Streamlines/graphics/{timestep}.png")
        return 1
       
    def minimize_energy(self, vpi:vtk.vtkProbeFilter, max_iterations=4000):
        grid = self.timeGrid
        iterations = 0
        self.generate_initial_streamlets(grid, vpi, self.energy_function)
        ###### Add a bad line, this sohuld trigger discard
        # line1 = Streamline()
        # line1.seed = np.array([5, 1, self.timeGrid.lower_bound[2]])
        # grid.update_line(0, line1)
        # line2 = Streamline()
        # line2.seed = np.array([10,10, self.timeGrid.lower_bound[2]])
        # grid.update_line(0, line2)
        # self.generate_streamlines(grid, vpi)
        # print("Post-add", get_energy())
        # grid.update_footprint_array()
        # print("Pre-join:", get_energy())
        # grid.join_lines(line1, line2)
        # self.generate_streamlines(grid, vpi)
        # grid.update_footprint_array()
        # print("Post-join:", get_energy())
        # grid.reject()
        # self.generate_streamlines(grid, vpi)
        # grid.update_footprint_array()
        # print(energy, get_energy())
        # print("Join candidates:", grid.get_join_candidates(grid.get_raster_spacing() * 10))
        ######
        self.generate_streamlines(grid, vpi)
        energy = test_energy = self.energy_function()
        print("Initial streamlet count | quality:", len(grid._lines), self.energy_function())
        rejected = 0
        rejected_oracles = 0
        energy = self.energy_function()
        accepted = len(grid._lines)
        # return 0
        while energy > 1 and iterations < max_iterations:
            if iterations % 2:
                action_id = 3
                action = self.oracle.make_suggestion()
                if action is not None:
                    action()
                    reset_action = lambda: self.timeGrid.reject()
            else:
                # action_id = 3
                action_id = np.random.choice(len(random_actions), replace=True)
                # action_id = np.random.choice([1,2,3,4], replace=True)
                action = random_actions[action_id]
                reset_action = action(grid)  #track how to undo the change we just executed
            self.generate_streamlines(grid, vpi)
            test_energy = self.energy_function()
            allowance = 0
            if action_id == 4: allowance = self.join_allowance * 2
            if iterations % 2 or test_energy - allowance < energy:
                # print("Accepted change.", test_energy, energy)
                grid.accept()
                accepted += 1
                reset_action = None
                energy = test_energy
            elif reset_action is not None: #if the change gets discarded, try to undo it
                # print("Bad energy (", test_energy, "<=", energy, "). Undid last action.")
                rejected += 1
                if iterations % 2:
                    rejected_oracles += 1
                reset_action()
            iterations += 1
        print(f"Reject vs accept: ({rejected} vs {accepted}). Oracle rejctions: {rejected_oracles}")
        return iterations
    
    def generate_initial_streamlets(self, grid:TimeGrid, vpi:vtk.vtkProbeFilter, energy_measure):
        grid._lines.clear()
        self.generate_streamlines(grid, vpi)
        energy = energy_measure()
        if self.make_coherent and self.coherence_strat.MIDSEED and self.midseeds is not None:
            xyzgenerator = self.midseeds
        else:
            gridx, gridy = grid.footprint_array.shape
            lo_x, lo_y, lo_z = self.timeGrid.lower_bound
            hi_x, hi_y, hi_z = self.timeGrid.upper_bound
            xvals = np.linspace(lo_x, hi_x - .00001, math.ceil(self.rasterx / self.grid_halo_radius))
            yvals = np.linspace(lo_y, hi_y - .00001, math.ceil(self.rastery / self.grid_halo_radius))
            xyzgenerator = itertools.product(xvals, yvals, [lo_z])
        for x,y,z in xyzgenerator:
            line = Streamline()
            line.seed = np.array([x,y,z])
            line_id = id(line)
            grid.update_line(0, line)
            self.generate_streamlines(grid, vpi)
            new_energy = energy_measure()
            if new_energy < energy:
                grid.accept()
                energy = new_energy
            else:
                # pop if the line doesnt improve things and has not been removed already
                grid._lines.pop((0, line_id), None) 
        grid.accept()
        self.generate_streamlines(grid, vpi)

    def draw_streamlines(self, grid:TimeGrid, polydata:vtk.vtkPolyData):
        # points = vtk.vtkPoints()
        total_points = vtk.vtkPoints()
        polydata.Allocate(len(grid._lines.values()))
        global_idx = 0
        for (timestep, lineid), (line, footprint) in grid._lines.items():
            # points.InsertPoints(points.GetNumberOfPoints(), line.points.GetNumberOfTuples(), 0, line.points)
            line_point_count = line.points.GetNumberOfPoints()
            total_points.InsertPoints(global_idx, line_point_count, 0, line.points)
            point_ids = vtk.vtkIdList()
            for point_idx in range(line_point_count):
                point_ids.InsertNextId(global_idx + point_idx)
            global_idx += line_point_count
            polydata.InsertNextCell(vtk.VTK_POLY_LINE, point_ids)
        polydata.SetPoints(total_points)
        return polydata

    def generate_streamlines(self, grid:TimeGrid, vpi:vtk.vtkPointInterpolator):
        id_count = 0
        pop = []
        for (timestep, lineid), (line, footprint) in grid._lines.items():
            if not line.needs_reeval:
                continue
            new_id_count = self.generate_streamline_single(line, line.segment_length, vpi, self.bounds)
            new_id_count = 0 if new_id_count is None else new_id_count
            # print(f"New points for line {id(line)}: {new_id_count}")
            if new_id_count < LINE_KILL_LENGTH:
                pop.append((0, id(line)))
            else:
                grid.update_line(0, line, needs_reeval=False)
                id_count += new_id_count
        for key in pop:
            print("Destroyed line")
            self.timeGrid._lines.pop(key)
        grid.update_footprint_array()
        self.gauss_filter(grid.footprint_array, grid.low_pass_array)
        self.energy_array(grid.low_pass_array, grid.energy_array)

    def generate_streamline_single(self, line:Streamline, segmentsize:float, vpi:vtk.vtkImageProbeFilter, bounds:tuple):
        points_to_interpolate = vtk.vtkPoints()
        points_to_interpolate.SetNumberOfPoints(1)
        pointset_to_interpolate = vtk.vtkPointSet()
        pointset_to_interpolate.SetPoints(points_to_interpolate)
        vpi.SetInputData(pointset_to_interpolate)
        seed = line.seed
        if (not self.inside_domain(seed)) or line.forward_segments + line.backward_segments < 2:
            return None
        current_pos = copy.deepcopy(seed)
        backward_count = line.backward_segments
        backward_points = vtk.vtkPoints()
        backward_points.SetNumberOfPoints(backward_count)
        backward_start = backward_count
        while backward_start > 0 and self.inside_domain(current_pos):
            backward_start -= 1
            backward_points.SetPoint(backward_start, current_pos)
            points_to_interpolate.SetNumberOfPoints(1)
            points_to_interpolate.SetPoint(0, current_pos)
            vpi.Update()
            out:vtk.vtkPointSet = vpi.GetOutput()
            velocities = out.GetPointData().GetArray(0)
            if velocities.GetNumberOfTuples() == 0:
                print("No inteprolated value for point:\n", current_pos)
                return None
            velocity:np.ndarray = numpy_support.vtk_to_numpy(velocities)[0]
            if np.linalg.norm(velocity) < 1e-9:
                break
            diff = self.normalize(velocity) * segmentsize
            current_pos -= diff
        # print("backward:", backward_points.GetNumberOfPoints(), backward_count - backward_start - 1)
        current_backward_count = backward_count - backward_start
        points = vtk.vtkPoints()
        points.InsertPoints(0, max(0, current_backward_count - 1), backward_start, backward_points)
        current_pos = copy.deepcopy(seed)
        while points.GetNumberOfPoints() - current_backward_count < line.forward_segments and self.inside_domain(current_pos):
            points.InsertNextPoint(current_pos)
            points_to_interpolate.SetNumberOfPoints(1) #why do i need to call this every time?
            points_to_interpolate.SetPoint(0, current_pos)
            vpi.Update()
            out:vtk.vtkPointSet = vpi.GetOutput()
            velocities:vtk.vtkDoubleArray = out.GetPointData().GetArray(0)
            if velocities.GetNumberOfTuples() == 0:
                print("No inteprolated value for point", current_pos)
                return None
            velocity:np.ndarray = numpy_support.vtk_to_numpy(velocities)[0]
            diff = self.normalize(velocity) * segmentsize
            current_pos += diff
        self.streamlineId += 1
        line.points = points
        return line.points.GetNumberOfPoints()

    def get_raster_key(self, coords:tuple[float, float, float]):
        delta = self.timeGrid.bound_delta
        translated = coords - self.timeGrid.lower_bound
        with np.errstate(divide="ignore"):
            scaled = np.nan_to_num(translated / delta)
        scaled *= self.rasterx, self.rastery, 0
        return scaled.astype(int)

    def normalize(self, vecs:np.ndarray):
        if len(vecs.shape) == 1:
            return vecs / np.linalg.norm(vecs)
        return vecs / np.asmatrix(np.linalg.norm(vecs, axis=1)).T

    def get_filter_scale(self):
        radius = self.grid_halo_radius
        sigma = radius / 3
        filter_size = 1 + radius * 2
        arr = np.zeros((filter_size, filter_size), dtype=float)
        arr[:, radius] = 1.0
        center_value = scipy.ndimage.gaussian_filter(arr, sigma, radius=radius, mode='constant')[radius, radius]
        return  TARGET_BRIGHTNESS * 1.5 / center_value

    def gauss_filter(self, arr:np.ndarray, out:np.ndarray):
        radius = self.grid_halo_radius
        ### gaussian low pass
        sigma = radius / 3
        scipy.ndimage.gaussian_filter(arr, sigma, radius=radius, output=out, mode='constant')
        np.multiply(out, self.filter_scale, out=out)
    
    def hermite_filter(self, arr:np.ndarray, out:np.ndarray):
        # WIP, do not use.
        radius = self.grid_halo_radius
        ### cubic hermite low pass
        if self.filter is None:
            dim = 2 * radius + 1
            xs, ys = np.arange(-radius, radius + 1), np.arange(-radius, radius + 1)
            self.filter = (np.sqrt(xs[:,None] ** 2 + ys[None, :] ** 2) / radius).clip(0, 1.0)
            self.filter = 2 * self.filter ** 3 - 3 * self.filter ** 2 + 1
    
    def energy_array(self, arr:np.ndarray, out:np.ndarray):
        ### The energy compared to our constant grayscale target
        return np.square(arr - self.timeGrid.energy_target, out=out)
    
    def energy_function(self) -> float:
        tc_factor = .5
        # assert 0 <= tc_factor <= 1.0
        if self.last_frame_low_pass is not None and\
                self.coherence_strat in (COHERENCE_STRATEGY.BIAS, COHERENCE_STRATEGY.COMBINED) and\
                self.make_coherent:
            incoherence = np.square(self.last_frame_low_pass - self.timeGrid.low_pass_array)
            total_energy = (1 - tc_factor) * self.timeGrid.energy_array + tc_factor * incoherence
        else:
            total_energy = self.timeGrid.energy_array
        return np.sum(total_energy)

    def inside_domain(self, point):
        arr = (self.timeGrid.lower_bound <= point) & (point <= self.timeGrid.upper_bound)
        return np.all(arr)


def _get_random_line_key(grid:TimeGrid): 
    keys = list(grid._lines.keys())
    return keys[np.random.choice(len(keys))] if keys else None

def _add_line(grid:TimeGrid):
    new_line = Streamline()
    new_line.seed = grid.lower_bound + (grid.upper_bound - grid.lower_bound) * np.random.random(3)
    grid.update_line(0, new_line)
    # print(f"Added line {id(new_line)} at {new_line.seed}.")
    return lambda: grid._lines.pop((0, id(new_line)), None)

def _remove_line(grid:TimeGrid):
    if key := _get_random_line_key(grid):
        line = grid._lines.pop(key)[0]
        # print("Removed line", id(line))
        return lambda: grid.update_line(0, line)
    return None

def _shift_line(grid:TimeGrid):
    if key := _get_random_line_key(grid):
        dist = (np.random.random(3) - (.5,.5,.5)) * (1,1,0) * 10 * grid.get_raster_spacing()
        grid.move(key, dist)
        # print(f"Moved line {id(grid._lines[key][0])} by {dist}.")
        return lambda: grid.reject()
    return None

def _change_length(grid:TimeGrid):
    if key := _get_random_line_key(grid):
        line = grid._lines[key][0]
        # prev_len = line.length
        grid.change_length(key, np.random.choice(10) - 5, np.random.choice(10) - 5)
        # print(f"Changed length of {id(line)}: {prev_len} -> {line.length}")
        return lambda: grid.reject()
    return None

def _join_lines(grid:TimeGrid):
    pairs = grid.get_join_candidates(grid.max_join_distance)
    if len(pairs) < 1:
        return None
    grid.join_lines(*pairs[np.random.choice(len(pairs))])
    return lambda: grid.reject()

random_actions = {
    0: _add_line,
    1: _remove_line,
    2: _shift_line,
    3: _change_length,
    4: _join_lines,
}