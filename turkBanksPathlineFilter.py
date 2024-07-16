import scipy.interpolate
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain
import vtkmodules.numpy_interface.dataset_adapter as dsa
from vtk.util import numpy_support

from scipy import fftpack

from typing import Callable

import time
import numpy as np
import math
import scipy
import scipy.ndimage
import vtk
import random
import itertools
import enum
from PIL import Image, ImageDraw
import copy
from vtk.util import numpy_support
import cProfile

np.seterr(divide='ignore', invalid='ignore')

# These can probably remain constant between different datasets
LINE_SEGMENT_SIZE = 0.005 # factor is 1/0.005 * final image size
SEP_TO_BLUR = 6.0/5.0
BLUR_MIN = RADIUS_LOWPASS = 4.0
DELTA_LEN_FACTOR = 1.125
DELTA_MOVE_FACTOR = .5
SAMPLE_RADIUS = .5

# This one is supposed to be configured
SEPARATION = 0.04
# REAL_SEPARATION = .8
# sep_min = sep_max = sep: desired line separation distance, e.g. 0.04 = 4% of screen width?

# The following ones all derive from it

FILTER_SIZE_X = SEP_TO_BLUR * BLUR_MIN / SEPARATION
# The y-size is computed based on the aspect ratio of the vector field

LINE_START_LENGTH = 2.5 * SEPARATION * .5
# len = birth length = 2.5*sep
DELTA_LENGTH = DELTA_LEN_FACTOR * SEPARATION
DELTA_MOVE = DELTA_MOVE_FACTOR * SEPARATION


TARGET_BRIGHTNESS = 1.

JOIN_DIST = SEPARATION * SEP_TO_BLUR

ALLOW_LEN_CHANGE = True

# join dist = max_join_Dist = sep * sep_to_blur

# Constants:

# move dist = sep * delta_move * random-.5
# len dist = delta_len * random -.5

# Oracle:
# 1/5 change is commpletely at random
# 1/2 rnadomize end change

LINE_REDRAW_COUNTER = 0
FILTER_RECALC_COUNTER = 0

SAMPLE_POINTS = 20

class Streamline:
    def __init__(self) -> None:
        self.seed: np.ndarray = None
        self.points:np.ndarray = None
        self.low_pass_points:np.ndarray = None
        self.velocities: np.ndarray = None
        self.forward_length:float = LINE_START_LENGTH
        self.backward_length:float = LINE_START_LENGTH
        self.current_forward_segments = 0
        self.current_backward_segments = 0
        self.low_pass_footprint:np.ndarray = None
        self.desired_move:np.ndarray = None
        self.desired_move_len = 0.0
        self.desired_front_change = 0.0
        self.desired_back_change = 0.0
        self.total_desire = 0.0
        self.filter_source:Callable[[np.ndarray], float] = None
        self.segment_footprints:list[np.ndarray] = []
        self.segment_bounds:np.ndarray = None
        self.raster_size:np.ndarray = None

    @property
    def ptcount(self):
        return self.points.shape[0]

    def duplicate(self, full=True):
        cp = Streamline()
        # Cheap attributes are always copied
        cp.seed = self.seed.copy()
        cp.forward_length = self.forward_length
        cp.backward_length = self.backward_length
        self.current_forward_segments = self.current_forward_segments
        self.current_backward_segments = self.current_backward_segments
        cp.filter_source = self.filter_source
        cp.raster_size = self.raster_size
        if not full: return cp
        # Copy the expensive ones
        cp.points = self.points.copy()
        cp.velocities = self.velocities.copy()
        cp.low_pass_footprint = self.low_pass_footprint.copy()
        if self.segment_bounds is None:
            print(self.points)
        cp.segment_bounds = self.segment_bounds.copy()
        cp.segment_footprints = self.segment_footprints.copy() #list.copy() is shallow
        return cp

    def calculate_whole_footprint(self):
        global LINE_REDRAW_COUNTER
        LINE_REDRAW_COUNTER += 1
        segment_starts = self.low_pass_points[:-1]
        segment_ends = self.low_pass_points[1:]
        segment_count = segment_starts.shape[0]
        if segment_count == 0:
            return
        # segment_starts = np.array(((40.0, 30),))
        # segment_ends   = np.array(((35.3, 44.2),))
        self.segment_bounds, self.segment_footprints = self.calculate_segments(segment_starts, segment_ends)
        for i in range(len(self.segment_footprints)):
            # print(self.segment_bounds[i])
            xmin, xmax, ymin, ymax = self.segment_bounds[i]
            self.low_pass_footprint[xmin:xmax+1, ymin:ymax+1] += self.segment_footprints[i]
    
    def calculate_segments(self, segment_starts:np.ndarray, segment_ends:np.ndarray):
        segment_count = segment_starts.shape[0]
        a_vals = segment_ends[:,1] - segment_starts[:,1]
        b_vals = segment_starts[:,0] - segment_ends[:,0]
        lengths = np.sqrt(np.square(a_vals) + np.square(b_vals))
        a_vals /= lengths
        b_vals /= lengths
        segments = np.hstack((segment_starts, segment_ends)).reshape((segment_count, 2, -1))
        return self.calculate_segment_footprint(segments, a_vals, b_vals)

    def do_single_change(self, front_change:bool, new_point:np.ndarray|None=None, new_velocity:np.ndarray|None=None):
        assert NotImplementedError
        # Check if we just want to delete a segment
        delete_idx = (len(self.segment_footprints) - 1) if front_change else 0
        if new_point is None:
            if len(self.segment_footprints) == 0:
                print(self.points)
            footprint = self.segment_footprints.pop(delete_idx)
            xmin, xmax, ymin, ymax = self.segment_bounds[delete_idx]
            self.low_pass_footprint[xmin:xmax+1,ymin:ymax+1] -= footprint
            self.segment_bounds = np.delete(self.segment_bounds, delete_idx, 0)
            self.velocities     = np.delete(self.velocities,     delete_idx, 0)
            self.points         = np.delete(self.points,         delete_idx, 0)
            return
        # Otherwise, calculate the segment and add it
        insert_idx_segs = len(self.segment_footprints) if front_change else 0
        insert_idx_pts = len(self.points) if front_change else 0
        self.velocities = np.insert(self.velocities, insert_idx_pts, new_velocity[0], axis=0)
        self.points     = np.insert(self.points,     insert_idx_pts, new_point[0],    axis=0)
        if front_change:
            seg_start = self.points[insert_idx_pts-1,:2][None]
            seg_end = new_point[:,:2]
        else:
            seg_start = new_point[:,:2]
            seg_end = self.points[insert_idx_pts+1,:2][None]
        segment_bounds, footprints = self.calculate_segments(seg_start, seg_end)
        self.segment_footprints.insert(insert_idx_segs, footprints[0])
        self.segment_bounds = np.insert(self.segment_bounds, insert_idx_segs, segment_bounds[0], axis=0)
        xmin, xmax, ymin, ymax = segment_bounds[0]
        self.low_pass_footprint[xmin:xmax+1, ymin:ymax+1] += footprints[0]

    def calculate_segment_pixel(self, seg_start:np.ndarray, seg_end:np.ndarray, x:np.ndarray, y:np.ndarray, a:float, b:float, rad:float):
        # Translate
        pixels = np.dstack(np.meshgrid(x,y, indexing="ij")).astype(float)
        translated_seg_start = seg_start - pixels
        translated_seg_end   =   seg_end - pixels
        # Verticalize
        vert_start_x = np.abs(a * translated_seg_start[:,:,0] + b * translated_seg_start[:,:,1])
        vert_start_y =       -b * translated_seg_start[:,:,0] + a * translated_seg_start[:,:,1]
        vert_end_x   = np.abs(a *   translated_seg_end[:,:,0] + b *   translated_seg_end[:,:,1])
        vert_end_y   =       -b *   translated_seg_end[:,:,0] + a *   translated_seg_end[:,:,1]
        # Scale
        div = 1 / rad
        vert_start_x *= div
        vert_start_y *= div
        vert_end_x   *= div
        vert_end_y   *= div
        # Clip
        vert_start_y = np.clip(vert_start_y, -1, 1)
        vert_end_y   = np.clip(  vert_end_y, -1, 1)
        # Filter contribution
        opposite_sides = vert_start_y * vert_end_y <= 0
        segment_in_range = vert_start_x <= 1.0
        p1 = np.abs(np.dstack((vert_start_x, vert_start_y)))
        p2 = np.abs(np.dstack((vert_end_x, vert_end_y)))
        brightness_start = np.zeros(p1.shape[:-1])
        brightness_end = np.zeros(p2.shape[:-1])
        brightness_start[segment_in_range] = self.filter_source(p1[segment_in_range])
        brightness_end[segment_in_range] = self.filter_source(p2[segment_in_range])
        return np.abs(brightness_start + (np.where(opposite_sides, brightness_end, -brightness_end)))

    def calculate_segment_footprint(self, segs:np.ndarray, a_vals:np.ndarray, b_vals:np.ndarray):
        # Indexing: PairNumber, Start/End, X/Y-coords
        min_x_each_coord = np.floor(np.min(segs[:,:,0], axis=1) - BLUR_MIN).clip(0, self.raster_size[0]-1)
        min_y_each_coord = np.floor(np.min(segs[:,:,1], axis=1) - BLUR_MIN).clip(0, self.raster_size[1]-1)
        max_x_each_coord =  np.ceil(np.max(segs[:,:,0], axis=1) + BLUR_MIN).clip(0, self.raster_size[0]-1)
        max_y_each_coord =  np.ceil(np.max(segs[:,:,1], axis=1) + BLUR_MIN).clip(0, self.raster_size[1]-1)
        # Indexing: PairNumber, minx/maxx/miny/maxy
        pixel_bounds = np.vstack((
            min_x_each_coord,
            max_x_each_coord,
            min_y_each_coord,
            max_y_each_coord
        )).T.astype(int)

        segment_brightnesses = []
        for segmentidx in range(pixel_bounds.shape[0]):
            min_x, max_x, min_y, max_y = pixel_bounds[segmentidx]
            # print("Seg Start/End:", segs[segmentidx][0], segs[segmentidx][1])
            # print("Seg Min/Max:", (min_x, min_y), (max_x, max_y))
            segment_brightnesses.append(
                self.calculate_segment_pixel(
                    segs[segmentidx, 0] - (min_x, min_y),
                    segs[segmentidx, 1] - (min_x, min_y),
                    np.arange(max_x - min_x + 1),
                    np.arange(max_y - min_y + 1),
                    a_vals[segmentidx],
                    b_vals[segmentidx],
                    BLUR_MIN
                )
            )
        return pixel_bounds, segment_brightnesses

    def get_end_points(self, n:int=0):
        """Return the n-th point from the start and end points of the line (in that order)."""
        if self.ptcount < 2:
            return None, None
        return np.vstack((self.points[n], self.points[-(n+1)]))
    
    def get_midpoint(self):
        """Return the floor(n/2)th point, where n is the number of points of the streamline."""
        return np.array(self.points.GetPoint(self.points.GetNumberOfPoints() // 2), dtype=float)

    def get_end_probes(self, dist:float):
        """Return 2 * segmentsize past the line endpoints, order = back, front"""
        if self.ptcount < 2:
            return None, None
        back_end, front_end = self.get_end_points()
        second_back, second_front = self.get_end_points(n=1)
        delta_back = _normalize(back_end-second_back) * dist
        delta_front = _normalize(front_end-second_front) * dist
        return back_end + delta_back, front_end + delta_front

    def get_side_probes(self, dist:float):
        if self.ptcount < 1:
            return None
        random_probe_indices = np.random.choice(self.ptcount, min(self.ptcount, SAMPLE_POINTS), replace=False)
        spaced_pts = self.points[random_probe_indices]
        # print("PTS:", spaced_pts)
        deltas = self.velocities[random_probe_indices]
        # print(self.velocities)
        normals = deltas @ ((0,1,0), (-1,0,0), (0,0,1)) # left-sided normal
        normals = _normalize(normals) * dist
        return np.concatenate((spaced_pts + normals, spaced_pts - normals))


class LowPassFilterGrid:
    def __init__(self, tbpfilter:"TurkBanksPathlineFilter", raster_size:np.ndarray, lower_bound:np.ndarray, upper_bound:np.ndarray, radius:float) -> None:
        self.tbpfilter = tbpfilter

        # Number of pixels (int) in X/Y direction
        self.raster_size = raster_size
        # X/Y-Size of the vector field (float)
        self.lower_bound:np.ndarray = lower_bound
        self.upper_bound:np.ndarray = upper_bound
        # The radius of the blur (float) applied in pixels
        self.radius = radius

        self._lines:dict[int, Streamline] = {}
        self.total_low_pass_footprint = np.zeros(raster_size, dtype=float)
        self.energy_array = np.zeros(raster_size, dtype=float)
        self.energy_target = np.ndarray(raster_size, dtype=float)
        self.energy_target.fill(TARGET_BRIGHTNESS)
        self._on_reject = None
        self.bound_delta:np.ndarray = upper_bound - lower_bound
        self.new_line_segment_length = 1.0
        self.filter = None
        self.join_allowance = 0.0

        print("Filter size in pixels (x,y):", self.raster_size)
        self.setup_filter()

    def get_raster_spacing(self):
        return self.bound_delta[:2] / self.energy_array.shape

    def update_line(self, line:Streamline):
        """Set line boundaries, integrate from the seed, and populate the footprint array of the line."""
        # Finish initializing the line
        line.low_pass_footprint = np.zeros_like(self.energy_array)
        line.filter_source = self.interp
        line.raster_size = self.raster_size
        # Updating the line populates the point array
        self.tbpfilter.generate_streamline(line)
        self.transform_line_filter(line)
        line.calculate_whole_footprint()

    def get_raster_key(self, coords):
        coords = np.atleast_2d(coords)
        translated = coords - self.lower_bound
        with np.errstate(divide="ignore"):
            scaled = np.nan_to_num(translated / self.bound_delta)
        scaled *= (*self.raster_size, 0)
        return scaled.astype(int)

    def setup_filter(self):
        """The filter uses indexing scheme [X,Y] to make it consistent with the coordinates."""
        samples = 30
        delta = 1 / (samples-1)
        xs, ys = np.arange(samples) * delta, np.arange(samples) * delta
        filter = (np.sqrt(xs[:,None] ** 2 + ys[None, :] ** 2)).clip(0, 1.0)
        filter = (2 * filter ** 3 - 3 * filter ** 2 + 1) * delta
        # The first column and last row of the filter should = 0.0
        filter = np.roll(filter, (0,1))
        self.filter = np.cumsum(filter, axis=1)
        self.interp = scipy.interpolate.RegularGridInterpolator((xs,ys), self.filter, bounds_error=False, fill_value=0.0)
    
    def hermite_filter(self, arr:np.ndarray):
        # WIP, do not use.
        ### cubic hermite low pass
        return scipy.ndimage.convolve(arr, self.filter, mode="constant")
        # kernel = self.filter[radius,:]
        # arr = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, arr)
        # return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 1, arr)

    def get_line_footprint(self, line:Streamline):
        arr = np.zeros_like(self.energy_array)
        raster_keys = self.get_raster_key(line.points)
        if np.any(raster_keys) >= self.rasterx:
            print("Bad key:", raster_keys, "of line at", line.seed)
        arr[raster_keys[:,0], raster_keys[:, 1]] += 1
        # TODO: what if line occupies same segment multiple times?
        return arr
    
    def update_filter(self, change:np.ndarray):
        global FILTER_RECALC_COUNTER
        FILTER_RECALC_COUNTER += 1
        self.total_low_pass_footprint += change
        # Update energy relative to the constant grayscale
        np.square(self.total_low_pass_footprint - self.energy_target, out=self.energy_array)

    def shatter(self):
        shards:list[Streamline] = []
        for line in self._lines.values():
            points = line.points
            ptcount = points.GetNumberOfPoints()
            if ptcount < LINE_KILL_SEGMENTS:
                continue
            current_point_idx = LINE_START_LENGTH + 1
            while current_point_idx < ptcount:
                shard = Streamline()
                shard.seed = np.array(points.GetPoint(current_point_idx))
                shards.append(shard)
                current_point_idx += shard.forward_length + shard.backward_length + 1
        self._lines.clear()
        for shard in shards:
            self.update_line(shard)

    def add_line(self, line:Streamline, force=False):
        self._assert_needs_accept_or_reject()
        self.update_line(line)
        return self._change_lines(add_line=line)
    
    def transform_line_filter(self, line:Streamline):
        line.low_pass_points = line.points[..., :2] * self.raster_size - .5
        
    def remove_line(self, line:Streamline):
        self._assert_needs_accept_or_reject()
        return self._change_lines(remove_line=line)

    def move(self, line:Streamline, offset:np.ndarray):
        self._assert_needs_accept_or_reject()
        new_line = Streamline()
        new_line.seed = line.seed + offset
        new_line.forward_length = line.forward_length
        new_line.backward_length = line.backward_length
        self.update_line(new_line)
        return self._change_lines(add_line=new_line, remove_line=line)
    
    def change_length(self, line:Streamline, front:bool, lengthen:bool):
        self._assert_needs_accept_or_reject()
        if line.ptcount < 3 and not lengthen:
            self._lines.pop(id(line))
            return True
        copied_line = line.duplicate(full=True)
        new_point = None

        if lengthen:
            if front:
                insert_idx = -1
                self.tbpfilter.vpi2.SetIntegrationDirectionToForward()
            else:
                insert_idx = 0
                self.tbpfilter.vpi2.SetIntegrationDirectionToBackward()
            new_point, new_velocity = self.tbpfilter.integrate_line_vtk(copied_line.points[insert_idx], 0)
            if new_point.shape[0] < 2:
                return None
            copied_line.do_single_change(front, new_point[1:], new_velocity[1:])
        else:
            copied_line.do_single_change(front)
        return self._change_lines(add_line=copied_line, remove_line=line)
    
    def move_and_change_length(self, line:Streamline, move:bool, front:bool, lengthen:bool, change_both:bool=False, recommended_move=None):
         ############## len change hella weird
        #   if (change & ALL_LEN) {
        #     len1 = len1_orig + delta_length1 * 2 * (drand48() - 0.5);
        #     if (len1 < 0)
        #       len1 = delta_length1 * drand48();

        #     len2 = len2_orig + delta_length2 * 2 * (drand48() - 0.5);
        #     if (len2 < 0)
        #       len2 = delta_length2 * drand48();
        #   }
        # print(line.seed, move, front, lengthen, change_both, recommended_move)
        if move is None and not change_both:
            raise NotImplementedError
            # return self.change_length(line, *np.random.randint(0,2,2))
        new_line = line.duplicate(full=False)
        if move:
            if recommended_move is not None:
                new_line.seed += recommended_move
            else:
                new_line.seed[0:2] += DELTA_MOVE * (np.random.random(2) - .5)
        if ALLOW_LEN_CHANGE:
            change_front = front or change_both
            change_back = (not front) or change_both
            l1 = new_line.forward_length
            l2 = new_line.backward_length
            random1, random2 = np.random.random(2)
            if change_front:
                if change_both:
                    l1 += DELTA_LENGTH * 2 * (random1 - .5)
                elif lengthen:
                    l1 += DELTA_LENGTH * random1
                else:
                    l1 -= DELTA_LENGTH * random1
            if change_back:
                if change_both:
                    l2 += DELTA_LENGTH * 2 * (random2 - .5)
                elif lengthen:
                    l2 += DELTA_LENGTH * random2
                else:
                    l2 -= DELTA_LENGTH * random2
            if l1 < 0: l1 = DELTA_LENGTH * np.random.random()
            if l2 < 0: l2 = DELTA_LENGTH * np.random.random()
            # print(new_line.forward_length, new_line.backward_length, l1, l2)
            new_line.forward_length = l1
            new_line.backward_length = l2
        self.update_line(new_line)
        return self._change_lines(add_line=new_line, remove_line=line, drop_on_improve=True, auto_revert=True)

    def get_join_candidates(self):
        """Get pairs of lines that can be joined head -> tail."""
        lines       = list(self._lines.values())
        linecount   = len(lines)
        heads       = np.ndarray((linecount, 3), dtype=float)
        tails       = np.ndarray((linecount, 3), dtype=float)
        
        for i, line in enumerate(self._lines.values()):
            heads[i], tails[i] = line.get_end_points()
        
        # Get the distances from every point to every other point as the cartesian product
        distances = np.linalg.norm(heads[:,None] - tails[None,:], axis=2)
        
        # Do not join a line with itself by setting the endpoint distance to infinity
        np.fill_diagonal(distances, float("inf"))
        
        # Obtain head-tail pairs from different lines that are close enough
        head_tail_pairs = np.array(np.where(distances <= JOIN_DIST)).T
        # for x,y in head_tail_pairs:
        #     print(f"Join {x}->{y}, dist={distances[x,y]}")
        # Convert the indices back to lines for further use
        return [(lines[i], lines[j]) for i,j in head_tail_pairs]

    def join_lines(self, l1:Streamline, l2:Streamline):
        """Join the head of l1 and the tail of l2."""
        self._assert_needs_accept_or_reject()
        energy = self.tbpfilter.energy_function()
        self._change_lines(remove_line=l1, auto_revert=False)
        self._change_lines(remove_line=l2, auto_revert=False)
        delete_energy = self.tbpfilter.energy_function()
        l1_start = l1.get_end_points()[0]
        l2_end = l2.get_end_points()[1]
        l1_importance = l1.ptcount / (l1.ptcount + l2.ptcount)
        combined = Streamline()
        combined.seed = l1_importance * l1_start + (1-l1_importance) * l2_end
        combined.backward_length = l1.forward_length + l1.backward_length
        combined.forward_length  = l2.forward_length + l2.backward_length
        self.update_line(combined)
        self._change_lines(add_line=combined, auto_revert=False)
        self._on_reject = lambda: self._reset_merge(combined, l1, l2)
        add_energy = self.tbpfilter.energy_function()
        if add_energy < energy or (add_energy - energy) < (delete_energy - energy) * .25:
            self.accept()
            return combined
        # print("Un-Joining", l1_start, l2_end, combined.seed, combined.forward_length, combined.backward_length, add_energy, energy, delete_energy)
        self.reject()

    def accept_or_reject(self, previous_energy:float):
        if (new_energy := self.tbpfilter.energy_function()) < previous_energy:
            self.accept()
            return True
        self.reject()
        return False

    def reject(self):
        if self._on_reject is not None:
            self._on_reject()
            self._on_reject = None

    def accept(self):
        self._on_reject = None

    def _change_lines(self, add_line:Streamline = None, remove_line:Streamline = None, drop_on_improve:bool=False, auto_revert:bool=True):
        """Remove remove_line from the list of lines, then add add_line. Can abort early or revert the changes due to bad energy. 
        
        Returns True if every operation was carried out and we did not revert.
        Use drop_on_improve to not add add_line if removing remove_line reduced the energy.
        In this case, False is returned if add_line was not None.
        Auto_revert is used to re-add remove_line and remove add_line if the energy changed for the worse, and return False.
        """
        energy = self.tbpfilter.energy_function()
        if remove_line is not None:
            self._lines.pop(id(remove_line))
            self.update_filter(-remove_line.low_pass_footprint)
        if drop_on_improve and self.tbpfilter.energy_function() < energy:
            return None
        if add_line is not None:
            self._lines[id(add_line)] = add_line
            self.update_filter(add_line.low_pass_footprint)
        if auto_revert:
            self._assert_needs_accept_or_reject()
            self._on_reject = lambda: self._change_lines(add_line=remove_line, remove_line=add_line, auto_revert=False)
            return add_line if self.accept_or_reject(energy) else None
        return add_line
    
    def _assert_needs_accept_or_reject(self):
        assert self._on_reject is None

    def _reset_merge(self, new:Streamline, l1:Streamline, l2:Streamline):
        # print("reset merge")
        self._change_lines(add_line=l1, remove_line=new, auto_revert=False)
        self._change_lines(add_line=l2, auto_revert=False)


class Oracle:
    def __init__(self, grid:LowPassFilterGrid, key_gen) -> None:
        self.grid = grid
        self.key_gen = key_gen
    
    def update_line_desire(self, line:Streamline):
        self._update_length_desire(line)
        self._update_move_desire(line)
        line.total_desire = abs(line.desired_front_change) + abs(line.desired_back_change) + 4 * abs(line.desired_move_len)

    def _update_move_desire(self, line:Streamline):
        probe_spots = line.get_side_probes(SAMPLE_RADIUS * RADIUS_LOWPASS / FILTER_SIZE_X)
        if probe_spots is None:
            line.desired_move = None
            line.desired_move_len = 0
            return
        probe_count = probe_spots.shape[0]
        grid_vals:np.ndarray = (probe_spots[..., :2] * self.grid.raster_size).astype(int)
        grid_vals.clip((0,0), self.grid.raster_size - 1, out=grid_vals)
        left_spots, right_spots = grid_vals[:probe_count//2], grid_vals[probe_count//2:]
        right_brightnesses = self.grid.total_low_pass_footprint[right_spots[:,0], right_spots[:,1]]
        left_brightnesses = self.grid.total_low_pass_footprint[left_spots[:,0], left_spots[:,1]]
        right_desire = np.average(right_brightnesses) - TARGET_BRIGHTNESS
        left_desire = np.average(left_brightnesses) - TARGET_BRIGHTNESS
        delta = (right_desire - left_desire) * DELTA_MOVE
        move = _normalize(line.velocities[line.current_backward_segments-1])[0] @ ((0,1,0), (-1,0,0), (0,0,1)) * delta
        line.desired_move = move
        line.desired_move_len = np.linalg.norm(move)
    
    def _update_length_desire(self, line:Streamline):
        ##### Check if we want to grow or shrink
        back_probe_spot, front_probe_spot = line.get_end_probes(SAMPLE_RADIUS * RADIUS_LOWPASS / FILTER_SIZE_X)
        # print(back_probe_spot, front_probe_spot)
        # print("Front Probe, Back Probe:", front_probe_spot, back_probe_spot)
        xmax, ymax = self.grid.raster_size
        border = LINE_SEGMENT_SIZE * .5
        if front_probe_spot is not None:
            ### Out of bounds, we just set the desires to 0, thereby skipping the respective side.
            fx, fy = self.key_gen(front_probe_spot)[0,:2]
            # print("front:", fx, fy, border, xmax-border, ymax-border)
            if (border <= fx <= xmax-border and border <= fy <= ymax-border):
                line.desired_front_change = TARGET_BRIGHTNESS - self.grid.total_low_pass_footprint[fx, fy]
            else:
                line.desired_front_change = 0
        if back_probe_spot is not None:
            ### Out of bounds, we just set the desires to 0, thereby skipping the respective side.
            bx, by = self.key_gen(back_probe_spot )[0,:2]
            # print("Back:", bx, by, border, xmax-border, ymax-border)
            if (border <= bx <= xmax-border and border <= by <= ymax-border):
                line.desired_back_change = TARGET_BRIGHTNESS - self.grid.total_low_pass_footprint[bx, by]
            else:
                line.desired_back_change = 0

    def make_suggestion(self):
        """Take an educated guess. Return the line, actionID, and values that are probably beneficial."""
        sorted_lines = list(sorted(self.grid._lines.values(), key=lambda line: line.total_desire, reverse=True))

        if (count := len(sorted_lines)) < 1:
            return False
        line_idx = int(np.random.random() * count / 8)
        line = sorted_lines[line_idx]
        self.update_line_desire(line)
        # print(l.seed, l.desired_front_change, l.desired_back_change, l.desired_move_len)
        front = abs(line.desired_front_change)
        back = abs(line.desired_back_change)
        move = 4 * line.desired_move_len
        # print("Move len, Front change, Back change:", line.desired_move_len, line.desired_front_change, line.desired_back_change)
        do_move = False
        change_front = False
        lengthen = False
        if (not ALLOW_LEN_CHANGE) or (move > front and move > back):
            do_move = True
        else:
            change_front = front > back
            lengthen = np.sign(line.desired_front_change if change_front else line.desired_back_change)

        new_line = self.grid.move_and_change_length(line, do_move, change_front, lengthen > 0, recommended_move=line.desired_move)
        if new_line is not None:
            self.update_line_desire(new_line)

        for i in range(5):
            line = sorted_lines[int(np.random.random() * count)]
            self.update_line_desire(line)
        return new_line is not None


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
        self.rasterx, self.rastery = -1, -1
        ##### Time Coherence
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

    @smproperty.doublevector(name="Time Coherence Weight", default_values=.0)
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetTimeCoherenceWeight(self, val):
        self.time_coherence_weight = val
        self.Modified()
    
    @smproperty.doublevector(name="Time Coherence Factor", default_values=1.0)
    def SetTimeCoherenceFactor(self, val):
        self.time_coherence_factor = val
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
        xmin, xmax, ymin, ymax, zmin, zmax = wholeext = inInfoObj.Get(SDDP.WHOLE_EXTENT())

        self.aspect_ratio = (ymax - ymin) / (xmax - xmin)
        self.rasterx = math.ceil(FILTER_SIZE_X) + 1
        self.rastery = math.ceil(self.rasterx * self.aspect_ratio)
        self.field_extents = vtk.vtkImageData().GetData(inInfo[0]).GetExtent()
        self.field_domain = np.array((xmax-xmin, ymax-ymin))

        outInfo0 = outInfo.GetInformationObject(0)
        outInfo0.Set(SDDP.WHOLE_EXTENT(), xmin, xmax, ymin, ymax, zmin, zmax)
        outInfo1 = outInfo.GetInformationObject(1)
        outInfo1.Set(SDDP.WHOLE_EXTENT(), 0, self.rasterx-1, 0, self.rastery-1, 0, 0)
        return 1

    def Reset(self):
        self.timeGrid = None
        self.oracle = None

    def RequestData(self, request:vtk.vtkInformation, inInfo:tuple[vtk.vtkInformationVector], outInfo:vtk.vtkInformationVector):
        timestep, all_timesteps = self.GetUpdateTimestep()
        t = time.time()
        inImageData = vtk.vtkImageData.GetData(inInfo[0])
     
        self.vpi = vtk.vtkPointInterpolator() # vtk.vtkProbeFilter()
        self.vpi.SetSourceData(inImageData)

        self.vpi2 = vtk.vtkStreamTracer()
        self.vpi2.SetInputData(inImageData)

        # self.vpi2.SetIntegratorTypeToRungeKutta45()
        self.vpi2.SetIntegratorTypeToRungeKutta4()
        self.vpi2.SetIntegrationStepUnit(1)
        self.vpi2.SetMaximumError(1)
        self.vpi2.SetInitialIntegrationStep(LINE_SEGMENT_SIZE * max(self.field_extents[:4]))

        self.timeGrid = LowPassFilterGrid(
            self,
            np.array((self.rasterx, self.rastery)),
            np.array((inImageData.GetBounds()[::2]), dtype=float),
            np.array((inImageData.GetBounds()[1::2]), dtype=float),
            RADIUS_LOWPASS
        )
        self.timeGrid.bound_delta = np.round(self.timeGrid.upper_bound - self.timeGrid.lower_bound, 10)
        self.timeGrid.update_filter(0)
        self.oracle = Oracle(self.timeGrid, self.timeGrid.get_raster_key)
        #### Configure the output ImageData object
        outImageData = vtk.vtkImageData.GetData(outInfo.GetInformationObject(1))
        outImageData.SetDimensions(self.rasterx, self.rastery, 1)
        outImageData.SetSpacing(*(self.timeGrid.bound_delta / (self.rasterx - 1, self.rastery - 1, 1)))
        #### Some dials to turn. This hopefully sets them into a decent starting position.
        print(f"\n\n{'== Pre-Configured Settings ==':^50}")
        print(f"{'HALO_RADIUS'          :<25}:{BLUR_MIN         :>10}")
        print(f"{'LINE_START_LENGTH'    :<25}:{LINE_START_LENGTH   :>10}")
        # print(f"{'CHANGE_SIZE'          :<25}:{CHANG         :>10}")
        # print(f"{'RASTER_RESOLUTION_SCALE' :<25}:{RASTER_RESOLUTION_SCALE :>10}")
        print(f"{'TARGET_BRIGHTNESS'    :<25}:{TARGET_BRIGHTNESS   :>10}")
        print(f"{'COHERENCE'            :<25}:{self.coherence_strat.name  :>10}")
        print(f"\n{'== Auto-Configured Settings ==':^50}")
        print(f"{'Raster Size'          :<25}:" + f"{str(self.rasterx) + ' * ' + str(self.rastery):>10}")
        print(f"{'Raster radius'   :<25}:{BLUR_MIN      :>10}")
        print(f"{'Line join radius'     :<25}:{JOIN_DIST      :>10.3f}")
        print(f"{'': >10}{'':=>30}{'': >10}\n\n")
        
        iterations = 0
        iterations = self.minimize_energy()
        # cProfile.runctx("iterations = self.minimize_energy()", {}, {"self":self})
        final_energy = self.energy_function()
        outPolyData = vtk.vtkPolyData.GetData(outInfo)
        self.draw_streamlines(self.timeGrid, outPolyData)
        low_pass_data = self.timeGrid.total_low_pass_footprint
        print(f"{'Min/Max brightness:':<20}", np.min(low_pass_data), np.max(low_pass_data))
        if self.last_frame_low_pass is not None:
            # Display the previous low-pass image if  we have one
            low = self.last_frame_low_pass
            inc = self.incoherence_array(self.timeGrid.total_low_pass_footprint)
            print(f"{'Min/Max old:'  :<20}", np.min(low), np.max(low))
            print(f"{'Incoherence:'  :<20}", np.min(inc), np.max(inc))
            last_img_array = np.zeros((*self.last_frame_low_pass.shape, 3), dtype=np.uint8)
            last_img_array[:,:,0] = (self.last_frame_low_pass / 2).clip(0, 1) * 255
        # print(f"{'Min/Max footprint:':<20}", np.min(self.timeGrid.footprint_array), np.max(self.timeGrid.footprint_array))
        print(f"{'Min/Max energy:'   :<20}", np.min(self.timeGrid.energy_array), np.max(self.timeGrid.energy_array))
        print(f"{'Lines:'            :<20}", len(self.timeGrid._lines))
        print(f"{'Line Redraws:'     :<20}", LINE_REDRAW_COUNTER)
        print(f"{'Filter Recalcs:'   :<20}", FILTER_RECALC_COUNTER)
        print(f"===== Reached final energy {final_energy:.3f} (={final_energy*100/(self.timeGrid.raster_size[0] * self.timeGrid.raster_size[1]):.3f}%) after {iterations} steps.")
        print(f"> Reloaded after {time.time() - t:.3f}s")
        current_img_array = np.zeros((*low_pass_data.shape, 3), dtype=np.uint8)
        current_img_array[:,:,1] = (low_pass_data / 2).clip(0,1) * 255
        summed = current_img_array
        if self.last_frame_low_pass is not None and False:
            summed += last_img_array
        blended = Image.fromarray(np.rot90(summed))
        if timestep is not None and timestep < 1e-6:
            print("Reset for first frame.")
            self.last_frame_low_pass = None
            self.midseeds = None
        if self.coherence_strat in (COHERENCE_STRATEGY.BIAS, COHERENCE_STRATEGY.COMBINED):
            pass
        if self.coherence_strat in (COHERENCE_STRATEGY.MIDSEED, COHERENCE_STRATEGY.COMBINED):
            if self.midseeds is not None:
                raster_keys = self.timeGrid.get_raster_key(self.midseeds)[:,:2]
                print("Adding midpoints")
                summed[raster_keys[:,0], raster_keys[:,1]] = [255, 0 ,255]
                summed.clip(0, 255, out=summed)
                for seed in self.midseeds:
                    x,y,z = self.timeGrid.get_raster_key(seed)
                    point = np.array((x,y))
                    p1 = (point - (1,1)).clip((0,0), (self.rasterx, self.rastery))
                    p2 = (point + (1,1)).clip((0,0), (self.rasterx, self.rastery))
                    pts = [*p1, *p2]
                    ImageDraw.Draw(blended).ellipse(pts, fill=(255,0,255), width=2)
            # if len(self.timeGrid._lines) > 0:
            #     self.timeGrid.shatter()
            #     self.midseeds = np.vstack([line.get_midpoint() for line in self.timeGrid._lines.values()])
            # else:
            #     self.midseeds = None
        self.last_frame_low_pass = low_pass_data
        flipped = np.moveaxis(summed, (0,1), (1,0)).reshape(self.rasterx * self.rastery, 3)
        outImageData.GetPointData().SetVectors(dsa.numpyTovtkDataArray(flipped))
        # blended.save(f"/home/alba/projects/Streamlines/graphics/{timestep}.png")
        return 1

    def minimize_energy(self, max_iterations=1000):
        global LINE_REDRAW_COUNTER, FILTER_RECALC_COUNTER
        grid = self.timeGrid
        iterations = 0
        t1 = time.time()
        self.generate_initial_streamlets(grid)
        ###### Some quick testing
        # def spawnl(x,y):
        #     l = Streamline()
        #     l.seed = np.array((x,y, self.timeGrid.lower_bound[2]))
        #     grid.add_line(l)
        #     return l
        # line1 = spawnl(.5, .5)
        # line2 = spawnl(.55 , .9)
        # line3 = spawnl(.1 , .4)
        # line4 = spawnl(.9 , .1)
        ######
        energy = self.energy_function()
        print("Initial streamlet count | energy | time:", len(grid._lines), energy, f"{time.time()-t1:.3f}")
        # LINE_REDRAW_COUNTER = FILTER_RECALC_COUNTER = 0
        accepted_oracles = 0
        accepted = 0
        ###### Further placement of inital lines
        # Try birthing lines
        self._try_birth_line()

        # Improve streamlines a number of times
        while iterations < max_iterations:
            # # Try moving a random streamline in a random direction and changing the lengths of both ends
            if self._try_move_and_ls_line():
                accepted += 1
            # # Try birthing lines
            if iterations % 100 == 0 and self._try_birth_line() > 0:
                accepted += 1
            # Try joining two lines
            if self._try_join_lines() is not None:
                accepted += 1
            # TODO: Try the oracle
            if self.oracle.make_suggestion():
                accepted_oracles += 1
            # else:
            #     print("Bad suggestion")
            #     break
            iterations += 1
        if iterations > 0:
            accept_percent = 100 * accepted         / iterations
            oracle_percent = 100 * accepted_oracles / iterations
            print(f"Accepted {accepted} out of {iterations} ({accept_percent:.3f}%). Oracle accepts: {accepted_oracles} ({oracle_percent:.3f}%)")
        return iterations

    def generate_initial_streamlets(self, grid:LowPassFilterGrid):
        random_placement = False # currently broken. dont activate
        lo_x, lo_y, lo_z = self.timeGrid.lower_bound
        hi_x, hi_y, hi_z = self.timeGrid.upper_bound
        d_x, d_y, d_z = self.timeGrid.bound_delta
        xcount = 12 # math.ceil(self.rasterx / (self.timeGrid.grid_halo_radius * 1.5))
        ycount = 12 # math.ceil(self.rastery / (self.timeGrid.grid_halo_radius * 1.5))
        delta = (d_x) / xcount
        if self.midseeds is not None:
            xyzgenerator = self.midseeds
        elif random_placement:
            total_count = 300 # xcount * ycount
            seeds = np.random.random((total_count, 2))
            xyzgenerator = ((x * d_x, y * d_y, 0) for x,y in seeds)
        else:
            xvals = np.linspace(lo_x + delta/2, hi_x - delta/2, xcount+1)
            yvals = np.linspace(lo_y + delta/2, hi_y - delta/2, ycount+1)
            xyzgenerator = itertools.product(xvals, yvals, [lo_z])
        for x,y,z in xyzgenerator:
            line = Streamline()
            line.seed = np.array([x,y,z])
            # Add line if it improves the image
            grid.add_line(line)
        for line in self.timeGrid._lines.values():
            self.oracle.update_line_desire(line)

    def draw_streamlines(self, grid:LowPassFilterGrid, polydata:vtk.vtkPolyData):
        # points = vtk.vtkPoints()
        total_points = vtk.vtkPoints()
        polydata.Allocate(len(grid._lines.values()))
        global_idx = 0
        for line in grid._lines.values():
            # points.InsertPoints(points.GetNumberOfPoints(), line.points.GetNumberOfTuples(), 0, line.points)
            line_point_count = line.ptcount
            vtkpoints = vtk.vtkPoints()
            for point in line.points:
                vtkpoints.InsertNextPoint(point)
            total_points.InsertPoints(global_idx, line_point_count, 0, vtkpoints)
            point_ids = vtk.vtkIdList()
            for point_idx in range(line_point_count):
                point_ids.InsertNextId(global_idx + point_idx)
            global_idx += line_point_count
            polydata.InsertNextCell(vtk.VTK_POLY_LINE, point_ids)
        polydata.SetPoints(total_points)
        return polydata

    def generate_streamline(self, line:Streamline):
        """Calculate and store the points on the given streamline."""
        # Obtain points via integraiton
        self.vpi2.SetIntegrationDirection(1)
        transform = np.array((*self.field_domain, 1))
        transformed_seed = line.seed * transform
        backward_points, backward_velocities = self.integrate_line_vtk(transformed_seed, line.backward_length)
        self.vpi2.SetIntegrationDirection(0)
        forward_points, forward_velocities = self.integrate_line_vtk(transformed_seed, line.forward_length)
        # Save points in order tail >= seed >= head, guaranteed to be inside the domain, with the duplicate seed removed
        line.points     = np.concatenate((np.flip(backward_points,     0), forward_points[1:]    ), axis=0) / transform
        line.velocities = np.concatenate((np.flip(backward_velocities, 0), forward_velocities[1:]), axis=0)
        line.current_backward_segments = backward_points.shape[0]
        line.current_forward_segments = forward_points.shape[0] - 1

    def integrate_line_vtk(self, start:np.ndarray, maxlen:float):
        seedpts = vtk.vtkPoints()
        seedpts.InsertNextPoint(start)
        seedpt_set = vtk.vtkPointSet()
        seedpt_set.SetPoints(seedpts)
        self.vpi2.SetSourceData(seedpt_set)
        self.vpi2.SetMaximumPropagation(maxlen  * max(self.field_extents[:4]))
        self.vpi2.Update()
        lo = (0, 0, 0)
        hi = (*self.field_domain, 0)
        out = self.vpi2.GetOutput()
        outpts = np.array(dsa.WrapDataObject(out).GetPoints())
        #vtkvels = out.GetPointData().GetArray("data")
        vtkvels = out.GetPointData().GetArray("velocity")
        outvels = np.array(numpy_support.vtk_to_numpy(vtkvels))
        valid = (lo <= outpts).all(axis=1) & (outpts <= hi).all(axis=1)
        return outpts[valid], outvels[valid]

    def incoherence_array(self, arr:np.ndarray):
        ### Compare arr to the previous brightness
        return np.square(arr - self.last_frame_low_pass)

    def energy_function(self) -> float:
        if self.last_frame_low_pass is not None and\
                self.coherence_strat in (COHERENCE_STRATEGY.BIAS, COHERENCE_STRATEGY.COMBINED):
            energy = (1 - self.time_coherence_weight) * self.timeGrid.energy_array
            incoherence = self.time_coherence_weight * self.incoherence_array(self.timeGrid.total_low_pass_footprint)
            total_energy = energy + self.time_coherence_factor * incoherence
        else:
            total_energy = self.timeGrid.energy_array
        return np.sum(total_energy)

    def inside_domain(self, point):
        arr = (self.timeGrid.lower_bound <= point) & (point <= self.timeGrid.upper_bound)
        return np.all(arr)

    def _get_random_line(self) -> Streamline:
        return np.random.choice(list(self.timeGrid._lines.values()))

    ### Grid modifier actions

    def _try_move_and_ls_line(self):
        line = self._get_random_line()
        return self.timeGrid.move_and_change_length(line, True, 0, 0, True )

    def _try_add_line(self):
        new_line = Streamline()
        new_line.seed = self.timeGrid.lower_bound + (self.timeGrid.bound_delta) * np.random.random(3)
        return new_line if self.timeGrid.add_line(new_line) else None
    
    def _try_birth_line(self):
        margin = RADIUS_LOWPASS
        inside_border = self.timeGrid.total_low_pass_footprint[math.floor(margin):math.ceil(self.rasterx-margin), math.floor(margin):math.ceil(self.rastery-margin)]
        birth_locations = np.argwhere(inside_border <= .02)
        births = 0
        for i in range(birth_locations.shape[0]):
            if self.timeGrid.total_low_pass_footprint[birth_locations[i,0], birth_locations[i,1]] <= .02:
                s = Streamline()
                s.seed = np.append(birth_locations[i] / self.timeGrid.raster_size, self.timeGrid.lower_bound[2]) 
                if self.timeGrid.add_line(s) is not None:
                    births += 1
        return births

    def _try_join_lines(self):
        pairs = self.timeGrid.get_join_candidates()
        if len(pairs) > 1:
            # Try twice
            return self.timeGrid.join_lines(*pairs[np.random.choice(len(pairs))]) or \
                self.timeGrid.join_lines(*pairs[np.random.choice(len(pairs))])

def _normalize(arr:np.ndarray):
    arr = np.atleast_2d(arr)
    return arr / np.linalg.norm(arr, axis=-1)[:, None]