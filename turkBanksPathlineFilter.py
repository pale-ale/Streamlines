from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smproxy, smproperty
import vtkmodules.numpy_interface.dataset_adapter as dsa
from scipy import fftpack

import time
import numpy as np
import math
import scipy
import scipy.ndimage
import vtk
import itertools
from PIL import Image
import copy
from vtk.util import numpy_support
import cProfile

NEIGHBOR_DIST_RATIO = .1
RASTER_RESOLUTION = 8
LINE_KILL_LENGTH = 4

class Streamline:
    def __init__(self) -> None:
        self.seed: np.ndarray = None
        self.points = vtk.vtkPoints()
        self.backward_segments:int = 10
        self.forward_segments:int = 10
        self.segment_length = .1
        self.needs_reeval = True

    def get_end_points(self, n:int=0):
        """Return the (n+1)th point from the start and end points of the line (in that order)."""
        if self.points.GetNumberOfPoints() < 2:
            return None, None
        return np.array(self.points.GetPoint(n)), np.array(self.points.GetPoint(self.points.GetNumberOfPoints()-1-n))
    
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
            x,y,z = self.key_gen(*point)
            val = arr[x,y]
            arr[x,y] += 1.0 if val == 0 else line.segment_length / 2
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
        self._lines.pop((0, id(new)))
        self.update_line(0, l1)
        self.update_line(0, l2)
        l1.needs_reeval = True
        l2.needs_reeval = True


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
                fx, fy = self.key_gen(*front_probe_spot)[:2]
                if (0 <= fx < gridx and 0 <= fy < gridy):
                    desire_front = .5 - self.grid.low_pass_array[fx, fy]
            if back_probe_spot is not None:
                ### Out of bounds, we just set the desires to 0, thereby skipping the respective side.
                bx, by = self.key_gen(*back_probe_spot )[:2]
                if (0 <= bx < gridx and 0 <= by < gridy):
                    desire_back  = .5 - self.grid.low_pass_array[bx, by]
                else:
                    desire_back = 0
            if abs(desire_back) > abs(highest_desire):
                highest_desire = desire_back
                l = line
                front = False
            if abs(desire_front) > abs(highest_desire):
                highest_desire = desire_front
                l = line
                front = True
        back_change = int(np.sign(highest_desire)) if front else 0
        front_change = int(np.sign(highest_desire)) if not front else 0
        print(back_change, front_change)
        if l is None:
            return None
        return lambda: self.grid.change_length((0, id(l)), front_change, back_change)


@smproxy.filter(label="TurkBanksPathlineFilter")
# @smproperty.xml("""
#     <OutputPort index="0" name="Initial Lines" />
#     <OutputPort index="1" name="Timestep 1 Lines" />
# """)
@smproperty.input(name="Input")
class TurkBanksPathlineFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
            nInputPorts=1,
            inputType='vtkImageData',
            # inputType='vtkStructuredGrid',
            # nOutputPorts=2,
            # outputType='vtkMultiBlockDataSet')
        )
        self.streamlineId = 0
        self.rasterx, self.rastery = -1, -1
        self.bounds = None
        self.timeGrid = None
        self.halo_radius = 1
        self.oracle = None
        self.filter = None
        self.filter_scale = 1.0
    
    def RequestInformation(self, request, inInfo, outInfo):
        imagedata = vtk.vtkImageData.GetData(inInfo[0])
        self.bounds = imagedata.GetBounds()
        return super().RequestInformation(request, inInfo, outInfo)

    def RequestData(self, request:vtk.vtkInformation, inInfo:tuple[vtk.vtkInformationVector], outInfo:vtk.vtkInformationVector):
        t = time.time()
        imageData = vtk.vtkImageData.GetData(inInfo[0])
        imageData.SetSpacing(1,1,1)
        imageData.Modified()
        _xmin, self.rasterx, _ymin, self.rastery, _zmin, _zmax = np.multiply(imageData.GetExtent(), RASTER_RESOLUTION)
        self.timeGrid = TimeGrid((self.rasterx, self.rastery), self.get_raster_key)
        self.timeGrid.lower_bound = np.array((imageData.GetBounds()[::2]), dtype=float)
        self.timeGrid.upper_bound = np.array((imageData.GetBounds()[1::2]), dtype=float)
        self.timeGrid.bound_delta = self.timeGrid.upper_bound - self.timeGrid.lower_bound

        self.oracle = Oracle(self.timeGrid, self.get_raster_key)
        
        #### Some dials to turn. This hopefully sets them into a decent starting position.
        self.timeGrid.new_line_segment_length = self.timeGrid.get_raster_spacing() * 2
        self.halo_radius = min(self.rasterx, self.rastery) * NEIGHBOR_DIST_RATIO
        self.filter_radius = math.ceil(self.halo_radius)
        self.timeGrid.max_join_distance = self.halo_radius * self.timeGrid.get_raster_spacing()
        self.join_allowance = 10 / self.timeGrid.get_raster_spacing()
        print(f"\n\n{'== Pre-Configured Settings ==':^50}")
        print(f"{'NEIGHBOR_DIST_RATIO'  :<25}:{NEIGHBOR_DIST_RATIO :>10}")
        print(f"{'LINE_KILL_LENGTH'     :<25}:{LINE_KILL_LENGTH    :>10}")
        print(f"{'RASTER_RESOLUTION'    :<25}:{RASTER_RESOLUTION   :>10}")
        print(f"\n{'== Auto-Configured Settings ==':^50}")
        print(f"{'Raster Size'          :<25}:" + f"{str(self.rasterx) + ' * ' + str(self.rastery):>10}")
        print(f"{'Line segment length'  :<25}:{self.timeGrid.new_line_segment_length:>10}")
        print(f"{'Line halo radius'     :<25}:{self.halo_radius                     :>10}")
        print(f"{'Filter halo radius'   :<25}:{self.filter_radius                   :>10}")
        print(f"{'Max line join radius' :<25}:{self.timeGrid.max_join_distance      :>10.3f}")
        print(f"{'Line join allowance'  :<25}:{self.join_allowance                  :>10}")
        print(f"{'': >10}{'':=>30}{'': >10}\n\n")
        
        self.filter_scale = self.get_filter_scale()
        
        vpi = vtk.vtkPointInterpolator() # vtk.vtkProbeFilter()
        vpi.SetSourceData(imageData)
        
        iterations = 0
        iterations = self.minimize_energy(vpi)
        # cProfile.runctx("energy, iterations = self.minimize_energy(vpi)", {}, {"self":self, "vpi":vpi})
        self.generate_streamlines(self.timeGrid, vpi)
        # print(self.timeGrid.lower_bound, self.timeGrid.upper_bound)

        self.timeGrid.update_footprint_array()
        low_pass_data = self.timeGrid.low_pass_array
        img = Image.fromarray(np.rot90((low_pass_data.clip(0,1) * 255).astype(np.uint8)))
        img.show()
        print("Min/Max badness:", np.min(low_pass_data), np.max(low_pass_data))
        print("Min/Max footprint:", np.min(self.timeGrid.footprint_array), np.max(self.timeGrid.footprint_array))
        print(f"Reached final energy {self.energy_function(low_pass_data)} after {iterations} steps.")
        print("Lines:", len(self.timeGrid._lines))
        print(f"Reloaded after {time.time() - t:.3f}s")
        # print(self.timeGrid.footprint_array)
        # print(low_pass_data.round(2))
        outPolyData = vtk.vtkPolyData.GetData(outInfo)
        self.draw_streamlines(self.timeGrid, outPolyData)
        return 1
       
    def minimize_energy(self, vpi:vtk.vtkProbeFilter, max_iterations=5000):
        grid = self.timeGrid
        def get_energy(): return self.energy_function(self.timeGrid.low_pass_array)
        
        iterations = 0
        self.generate_initial_streamlets(grid, vpi, get_energy)
        print("Initial streamlet quality:", get_energy())
        ###### Add a bad line, this sohuld trigger discard
        # line1 = Streamline()
        # line1.seed = np.array([20, 20, self.timeGrid.lower_bound[2]])
        # line1.backward_segments = 30
        # line1.forward_segments = 30
        # grid.update_line(0, line1)
        # line2 = Streamline()
        # line2.seed = np.array([20.4,16, self.timeGrid.lower_bound[2]])
        # grid.update_line(0, line2)
        self.generate_streamlines(grid, vpi)
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
        # energy = test_energy = get_energy()
        ######
        ###### Test oracle's quality
        # for i in  range(100):
        #     suggestion = self.oracle.make_suggestion()
        #     if suggestion is None:
        #         print("none suggestion?")
        #         continue
        #     energy = get_energy()
        #     suggestion()
        #     self.generate_streamlines(grid, vpi)
        #     if energy < get_energy():
        #         print("bad oracle!")
        #     self.timeGrid.accept()
        # print(line1.backward_segments, line1.forward_segments)
        rejected = 0
        rejected_oracles = 0
        self.generate_streamlines(grid, vpi)
        energy = get_energy()
        accepted = len(grid._lines)
        while energy > 1 and iterations < max_iterations:
            if False and iterations % 2:
                action_id = 2
                action = self.oracle.make_suggestion()
                reset_action = lambda: self.timeGrid.reject()
            else:
                # action_id = np.random.choice([4], replace=True)
                # action_id = np.random.choice([2,3,4], replace=True)
                action_id = np.random.choice(len(random_actions), replace=True)
                action = random_actions[action_id]
                reset_action = action(grid)  #track how to undo the change we just executed
            self.generate_streamlines(grid, vpi)
            test_energy = get_energy()
            allowance = 0
            if action_id == 4: allowance = self.join_allowance
            if test_energy - allowance < energy:
                # print("Accepted change.", test_energy, energy)
                grid.accept()
                accepted += 1
                reset_action = None
                energy = test_energy
            elif reset_action is not None: #if the change gets discarded, try to undo it
                # print("Bad energy (", test_energy, "<=", energy, "). Undid last action.")
                rejected += 1
                # if iterations % 2:
                #     rejected_oracles += 1
                reset_action()
            iterations += 1
        print(f"Reject vs accept: ({rejected} vs {accepted}). Oracle rejctions: {rejected_oracles}")
        return iterations
    
    def generate_initial_streamlets(self, grid:TimeGrid, vpi:vtk.vtkProbeFilter, energy_measure):
        grid._lines.clear()
        energy = energy_measure()
        gridx, gridy = grid.footprint_array.shape
        lower = self.timeGrid.lower_bound
        upper = self.timeGrid.upper_bound
        xvals = np.linspace(lower[0], upper[0] - .00001, math.ceil(gridx / 20))
        yvals = np.linspace(lower[1], upper[1] - .00001, math.ceil(gridy / 20))
        for x,y in itertools.product(xvals, yvals):
            line = Streamline()
            line.seed = np.array([x,y,lower[2]])
            line_id = id(line)
            grid.update_line(0, line)
            self.generate_streamlines(grid, vpi)
            new_energy = energy_measure()
            if new_energy < energy:
                grid.accept()
                energy = new_energy
            else:
                grid._lines.pop((0, line_id))
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
        for (timestep, lineid), (line, footprint) in grid._lines.items():
            if not line.needs_reeval:
                continue
            new_id_count = self.generate_streamline_single(line, line.segment_length, vpi, self.bounds)
            # print(f"New points for line {id(line)}: {new_id_count}")
            if new_id_count is None or new_id_count < max(2, LINE_KILL_LENGTH):
                self.timeGrid._lines.pop(0, id(line))
            else:
                grid.update_line(0, line, needs_reeval=False)
                id_count += new_id_count
        grid.update_footprint_array()
        self.low_pass_filter(grid.footprint_array, grid.low_pass_array)

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
        current_backward_count = backward_count - backward_start - 1
        points = vtk.vtkPoints()
        points.InsertPoints(0, current_backward_count, backward_start, backward_points)
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

    def get_raster_key(self, x:float, y:float, z:float):
        delta = self.timeGrid.upper_bound - self.timeGrid.lower_bound
        translated = (x,y,z) - self.timeGrid.lower_bound
        with np.errstate(divide="ignore"):
            scaled = np.nan_to_num(translated / delta)
        return int(scaled[0] * self.rasterx), int(scaled[1]*self.rastery), 0

    def normalize(self, vecs:np.ndarray):
        if len(vecs.shape) == 1:
            return vecs / np.linalg.norm(vecs)
        return vecs / np.asmatrix(np.linalg.norm(vecs, axis=1)).T

    def get_filter_scale(self):
        radius = self.filter_radius
        sigma = radius / 6
        filter_size = 1 + radius * 2
        arr = np.zeros((filter_size, filter_size), dtype=float)
        arr[:, radius] = 1.0
        center_value = scipy.ndimage.gaussian_filter(arr * sigma, sigma, radius=radius, mode='constant')[radius, radius]
        return  1 / center_value

    def low_pass_filter(self, arr:np.ndarray, out:np.ndarray):
        # low_pass = np.zeros((self.rasterx + self.filter_radius*2, self.rastery + self.filter_radius*2), dtype=float)
        # kernel = self.filter[3] # Here you would insert your actual kernel of any size
        # arr = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, arr)
        # return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, arr)
        # cubic_array = (arr * 2) ** 3 + (arr * 1.3)**2
        radius = self.filter_radius
        sigma = radius / 6
        scipy.ndimage.gaussian_filter(arr * sigma, sigma, radius=radius, output=out, mode='constant')
        np.multiply(out, self.filter_scale, out=out)
    
    def energy_function(self, arr:np.ndarray, t:float=.5) -> float:
        return np.sum(np.square(arr - t))

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
    return lambda: grid._lines.pop((0, id(new_line)))

def _remove_line(grid:TimeGrid):
    if key := _get_random_line_key(grid):
        line = grid._lines.pop(key)[0]
        # print("Removed line", id(line))
        return lambda: grid.update_line(0, line)
    return None

def _shift_line(grid:TimeGrid):
    if key := _get_random_line_key(grid):
        dist = (grid.upper_bound - grid.lower_bound) * (np.random.random(3) - (.5,.5,.5)) * .02
        grid.move(key, dist)
        # print(f"Moved line {id(grid._lines[key][0])} by {dist}.")
        return lambda: grid.reject()
    return None

def _change_length(grid:TimeGrid):
    if key := _get_random_line_key(grid):
        line = grid._lines[key][0]
        # prev_len = line.length
        grid.change_length(key, np.random.choice([-1,1]), np.random.choice([-1,1]))
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