import PIL.Image
import PIL.ImageDraw2
import PIL.ImageFilter
import scipy.interpolate
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain
import vtkmodules.numpy_interface.dataset_adapter as dsa
from vtk.util import numpy_support

from scipy import fftpack

from typing import Callable
from dataclasses import dataclass, replace

import time
import numpy as np
import math
import scipy
import scipy.ndimage
import vtk
import itertools

from vtk.util import numpy_support

np.seterr(divide='ignore', invalid='ignore')

LOG_PARAMS = True
LOG_STATS  = True

SHATTER_COHERENCE = False
BIAS_COHERENCE    = True
DRAW_COHERENCE    = SHATTER_COHERENCE or BIAS_COHERENCE

ALLOW_LEN_CHANGE = True
ALLOW_MOVE       = True

LINE_REDRAW_COUNTER   = 0
FILTER_RECALC_COUNTER = 0

SAMPLE_POINTS = 20

class Streamline:
    def __init__(self, len_front: float, len_back: float) -> None:
        self.seed: np.ndarray = None
        self.points:np.ndarray = None
        self.low_pass_points:np.ndarray = None
        self.velocities: np.ndarray = None
        self.forward_length:float = len_front
        self.backward_length:float = len_back
        self.current_forward_segments = 0
        self.current_backward_segments = 0
        self.desired_move:np.ndarray = None
        self.desired_move_len = 0.0
        self.desired_front_change = 0.0
        self.desired_back_change = 0.0
        self.total_desire = 0.0
        self.filter_source:Callable[[np.ndarray], float] = None

    @property
    def ptcount(self):
        return self.points.shape[1]

    def duplicate(self, full=True):
        cp = Streamline(0, 0)
        # Cheap attributes are always copied
        cp.seed = self.seed.copy()
        cp.forward_length = self.forward_length
        cp.backward_length = self.backward_length
        self.current_forward_segments = self.current_forward_segments
        self.current_backward_segments = self.current_backward_segments
        cp.filter_source = self.filter_source
        if not full: return cp
        # Copy the expensive ones
        cp.points = self.points.copy()
        cp.velocities = self.velocities.copy()
        return cp

    def do_single_change(self, front_change:bool, new_point:np.ndarray|None=None, new_velocity:np.ndarray|None=None):
        assert NotImplementedError
        # Check if we just want to delete a segment
        delete_idx = (len(self.segment_footprints) - 1) if front_change else 0
        if new_point is None:
            if len(self.segment_footprints) == 0:
                print("No Points:", self.points)
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

    def get_end_points(self, n:int=0):
        """Return the n-th point from the start and end points of the line (in that order)."""
        if self.ptcount < 2:
            return None, None
        return np.vstack((self.points[:,n].T, self.points[:, -(n+1)].T))

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
        spaced_pts = self.points.T[random_probe_indices]
        # print("PTS:", spaced_pts)
        deltas = self.velocities.T[random_probe_indices]
        # print(self.velocities)
        normals = deltas @ ((0,1,0), (-1,0,0), (0,0,1)) # left-sided normal
        normals = _normalize(normals) * dist
        return np.concatenate((spaced_pts + normals, spaced_pts - normals))


@dataclass
class LPFilterConfig:
    target_energy     : float = 1.0
    separation        : float = 0.04
    sep_to_blur       : float = 6.0 / 5.0
    filter_radius     : float = 2.0
    sample_radius     : float =  .5
    line_segment_size : float =  .01
    @property
    def blur_min(self)      : return self.filter_radius
    @property
    def line_start_len(self): return self.separation  * 2.5
    @property
    def delta_len(self)     : return self.separation  * 1.125
    @property
    def delta_move(self)    : return self.separation  *  .5
    @property
    def join_dist(self)     : return self.separation  * self.sep_to_blur
    @property
    def filter_xsize(self)  : return self.sep_to_blur * self.filter_radius / self.separation


class LPFilterTarget:
    def __init__(self, size:tuple[int,int]) -> None:
        self.data = np.zeros(size)
        self.size = np.array(size)


class LPFilterPainter:
    def __init__(self, config:LPFilterConfig, target:LPFilterTarget, filter_source:Callable[[np.ndarray], float]) -> None:
        self.config: LPFilterConfig = config
        self.target: LPFilterTarget = target
        self.filter_source = filter_source
        self.footprints: dict[Streamline, np.ndarray] = {}
        max_seg_distance = config.line_segment_size * config.filter_xsize + 1
        filter_diameter = 2 * (config.filter_radius + 1)
        self.max_seg_len = math.ceil(max_seg_distance + filter_diameter) + 1
    
    def add_line(self, line:Streamline, footprint:np.ndarray|None=None):
        # Generate the footprint if it wasnt already calculated
        if footprint is None:
            self.footprints[line] = np.zeros_like(self.target.data)
            self.write_whole_footprint(line, self.footprints[line])
        else:
            self.footprints[line] = footprint
        
        # Add it to the target
        self.target.data += self.footprints[line]
    
    def remove_line(self, line:Streamline):
        # Remove the line and its footprint
        footprint = self.footprints.pop(line)
        self.target.data -= footprint

    def write_whole_footprint(self, line:Streamline, target:np.ndarray):
        global LINE_REDRAW_COUNTER
        LINE_REDRAW_COUNTER += 1

        segment_count  = line.ptcount - 1
        if segment_count < 1:
            return
        max_segment_side_len = self.max_seg_len        
        # Use the (float) coordinates in range [0,1] transformed into the target frame of reference
        points = self.transform_points_target(line.points)
        segment_starts = points[..., :-1]
        segment_ends   = points[..., 1:]

        # Pre-Generate a range of coordinates to be reused
        mesh = np.meshgrid(np.arange(max_segment_side_len), np.arange(max_segment_side_len))
        self.pixel_indices = np.flip(np.dstack(mesh).T, 0)

        # Obtain factors from the line equation
        a_vals, b_vals = self.calculate_segment_equations(segment_starts, segment_ends)

        # Obtain segemnts in the shape (4, n), ordered (startx, starty, endx, endy)
        combined_segs = np.vstack((segment_starts, segment_ends))

        # Obtain segment bounds in the shape (4, n), ordered (lo_x, lo_y, hi_, hi_y)
        segment_pixel_bounds = self.calculate_segment_bounds(combined_segs)

        # Calculate each segment's brightness levels and write them to the target
        for segmentidx in range(segment_count):
            xmin, xmax, ymin, ymax = segment_pixel_bounds[..., segmentidx]
            footprint = self.calculate_segment_pixel(
                combined_segs[0:2, segmentidx] - (xmin, ymin),
                combined_segs[2:4, segmentidx] - (xmin, ymin),
                xmax - xmin + 1,
                ymax - ymin + 1,
                a_vals[segmentidx],
                b_vals[segmentidx],
                self.config.blur_min
            ).T
            if footprint.shape != (xmax-xmin+1, ymax-ymin+1):
                print(f"Mismatch in line at {line.seed}, FL:{line.forward_length}, BL:{line.backward_length}")
            target[xmin:xmax+1, ymin:ymax+1] += footprint
    
    def calculate_segment_equations(self, segment_starts:np.ndarray, segment_ends:np.ndarray):
        a_vals = segment_ends[1, ...] - segment_starts[1, ...]
        b_vals = segment_starts[0, ...] - segment_ends[0, ...]
        lengths = np.sqrt(np.square(a_vals) + np.square(b_vals))
        a_vals /= lengths
        b_vals /= lengths
        return a_vals, b_vals

    def calculate_segment_pixel(self, seg_start:np.ndarray, seg_end:np.ndarray, x:int, y:int, a:float, b:float, rad:float):
        # Translate
        pixel_indices = self.pixel_indices[:, :y, :x]
        translated_seg_start = seg_start[:, None, None] - pixel_indices
        translated_seg_end   =   seg_end[:, None, None] - pixel_indices
        # Verticalize
        vert_start_x = np.abs(a * translated_seg_start[0, ...] + b * translated_seg_start[1, ...])
        vert_start_y =       -b * translated_seg_start[0, ...] + a * translated_seg_start[1, ...]
        vert_end_x   = np.abs(a *   translated_seg_end[0, ...] + b *   translated_seg_end[1, ...])
        vert_end_y   =       -b *   translated_seg_end[0, ...] + a *   translated_seg_end[1, ...]
        # Scale
        div = 1 / rad
        vert_start_x *= div
        vert_start_y *= div
        vert_end_x   *= div
        vert_end_y   *= div
        # Clip
        np.clip(vert_start_y, -1, 1, out=vert_start_y)
        np.clip(  vert_end_y, -1, 1, out=vert_end_y  )
        # Filter contribution
        opposite_sides = vert_start_y * vert_end_y <= 0
        segment_in_range = vert_start_x <= 1.0
        p1 = np.abs(np.dstack((vert_start_x, vert_start_y)))
        p2 = np.abs(np.dstack((vert_end_x, vert_end_y)))
        brightness_start = np.zeros(p1.shape[:-1])
        brightness_end = np.zeros(p2.shape[:-1])
        brightness_start[segment_in_range] = self.filter_source(p1[segment_in_range])
        brightness_end[segment_in_range] = self.filter_source(p2[segment_in_range])
        final_brightness = np.abs(brightness_start + (np.where(opposite_sides, brightness_end, -brightness_end)))
        # print(final_brightness.round(1))
        return final_brightness

    def calculate_segment_bounds(self, segs:np.ndarray):
        size = self.target.size
        # Indexing: X/Y, PairNumber, Start/End
        min_x_each_segment = np.floor(np.min(segs[ ::2, ...], axis=0) - self.config.blur_min).clip(0, size[0]-1)
        min_y_each_segment = np.floor(np.min(segs[1::2, ...], axis=0) - self.config.blur_min).clip(0, size[1]-1)
        max_x_each_segment =  np.ceil(np.max(segs[ ::2, ...], axis=0) + self.config.blur_min).clip(0, size[0]-1)
        max_y_each_segment =  np.ceil(np.max(segs[1::2, ...], axis=0) + self.config.blur_min).clip(0, size[1]-1)
        # Indexing: PairNumber, minx/maxx/miny/maxy
        pixel_bounds = np.vstack((
            min_x_each_segment,
            max_x_each_segment,
            min_y_each_segment,
            max_y_each_segment
        )).astype(int)
        return pixel_bounds

    def transform_points_target(self, points:np.ndarray):
        return points[:2, ...] * self.target.size[:,None] - .5


class LowPassFilter:
    def __init__(
            self,
            line_integrator:Callable[[Streamline], None],
            energy_function:Callable[[], float],
            painters: list[LPFilterPainter]) -> None:
        # Call this with a streamline to integrate and populate it in-place
        self.line_integrator = line_integrator
        # Obtain the energy measure from an energy array
        self.energy_function = energy_function
        # Number of pixels (int) in X/Y direction
        self.painters = painters
        # The radius of the blur (float) applied in pixels
        self._lines:dict[int, Streamline] = {}
        self._filter_targets:dict[LPFilterConfig, LPFilterTarget] = {}
        self._on_reject = None

    def update_line(self, line:Streamline):
        """Set line boundaries, integrate from the seed, and populate the footprint array of the line."""
        # Finish initializing the line
        # Updating the line populates the point array
        self.line_integrator(line)
    
    def update_painters(self, line:Streamline, add:bool):
        global FILTER_RECALC_COUNTER
        FILTER_RECALC_COUNTER += 1
        for painter in self.painters:
            if add:
                painter.add_line(line)
            else:
                painter.remove_line(line)

    def shatter(self):
        # Max number of pts of the shards
        start_len = self._head_config.line_start_len
        seg_size = self._head_config.line_segment_size
        shards:list[Streamline] = []
        for line in self._lines.values():
            line_len = line.ptcount * seg_size
            shard_count = math.ceil(line_len / start_len)
            split_pts = np.array_split(line.points, shard_count, axis=1)
            for pts in split_pts:
                center_idx = pts.shape[1] // 2
                shard = Streamline(
                    (pts.shape[1] - center_idx) * seg_size,
                    center_idx * seg_size
                )
                shard.seed = pts[...,center_idx].T
                shards.append(shard)
        return shards

    def add_line(self, line:Streamline, force=False):
        self._assert_needs_accept_or_reject()
        self.update_line(line)
        return self._change_lines(add_line=line, auto_revert=not force)

    def remove_line(self, line:Streamline):
        self._assert_needs_accept_or_reject()
        return self._change_lines(remove_line=line)

    def move(self, line:Streamline, offset:np.ndarray):
        self._assert_needs_accept_or_reject()
        new_line = Streamline(line.forward_length, line.backward_length)
        new_line.seed = line.seed + offset
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
                self.tbpfilter.stracer.SetIntegrationDirectionToForward()
            else:
                insert_idx = 0
                self.tbpfilter.stracer.SetIntegrationDirectionToBackward()
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
        cfg = self._head_config
        new_line = line.duplicate(full=False)
        if move:
            if recommended_move is not None:
                new_line.seed += recommended_move
            else:
                new_line.seed[0:2] += cfg.delta_move * (np.random.random(2) - .5)
        if ALLOW_LEN_CHANGE:
            change_front = front or change_both
            change_back = (not front) or change_both
            l1 = new_line.forward_length
            l2 = new_line.backward_length
            random1, random2 = np.random.random(2)
            if change_front:
                if change_both:
                    l1 += cfg.delta_len * 2 * (random1 - .5)
                elif lengthen:
                    l1 += cfg.delta_len * random1
                else:
                    l1 -= cfg.delta_len * random1
            if change_back:
                if change_both:
                    l2 += cfg.delta_len * 2 * (random2 - .5)
                elif lengthen:
                    l2 += cfg.delta_len * random2
                else:
                    l2 -= cfg.delta_len * random2
            if l1 < 0: l1 = cfg.delta_len * np.random.random()
            if l2 < 0: l2 = cfg.delta_len * np.random.random()
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
        head_tail_pairs = np.array(np.where(distances <= self._head_config.join_dist)).T
        # for x,y in head_tail_pairs:
        #     print(f"Join {x}->{y}, dist={distances[x,y]}")
        # Convert the indices back to lines for further use
        return [(lines[i], lines[j]) for i,j in head_tail_pairs]

    def join_lines(self, l1:Streamline, l2:Streamline):
        """Join the head of l1 and the tail of l2."""
        self._assert_needs_accept_or_reject()
        energy = self.energy_function()
        self._change_lines(remove_line=l1, auto_revert=False)
        self._change_lines(remove_line=l2, auto_revert=False)
        delete_energy = self.energy_function()
        l1_start = l1.get_end_points()[0]
        l2_end = l2.get_end_points()[1]
        l1_importance = l1.ptcount / (l1.ptcount + l2.ptcount)
        combined = Streamline(l2.forward_length + l2.backward_length, l1.forward_length + l1.backward_length)
        combined.seed = l1_importance * l1_start + (1-l1_importance) * l2_end
        self.update_line(combined)
        self._change_lines(add_line=combined, auto_revert=False)
        self._on_reject = lambda: self._reset_merge(combined, l1, l2)
        add_energy = self.energy_function()
        if add_energy < energy or (add_energy - energy) < (delete_energy - energy) * .65:
            self.accept()
            return combined
        # print("Un-Joining", l1_start, l2_end, combined.seed, combined.forward_length, combined.backward_length, add_energy, energy, delete_energy)
        self.reject()

    def accept_or_reject(self, previous_energy:float):
        if (new_energy := self.energy_function()) < previous_energy:
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

    @property
    def _head_config(self):
        return self.painters[0].config

    def _change_lines(self, add_line:Streamline = None, remove_line:Streamline = None, drop_on_improve:bool=False, auto_revert:bool=True):
        """Remove remove_line from the list of lines, then add add_line. Can abort early or revert the changes due to bad energy. 
        
        Returns True if every operation was carried out and we did not revert.
        Use drop_on_improve to not add add_line if removing remove_line reduced the energy.
        In this case, False is returned if add_line was not None.
        Auto_revert is used to re-add remove_line and remove add_line if the energy changed for the worse, and return False.
        """
        energy = self.energy_function()
        if remove_line is not None:
            self._lines.pop(id(remove_line))
            self.update_painters(remove_line, False)
        if drop_on_improve and self.energy_function() < energy:
            return None
        if add_line is not None:
            self._lines[id(add_line)] = add_line
            self.update_painters(add_line, True)
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
    def __init__(self, filters: "FilterStack") -> None:
        self.filters = filters
    
    def update_line_desire(self, active_filter:LowPassFilter, line:Streamline):
        self._update_length_desire(active_filter, line)
        self._update_move_desire(active_filter, line)
        line.total_desire = 4 * abs(line.desired_move_len) + abs(line.desired_front_change) + abs(line.desired_back_change)

    def _update_move_desire(self, active_filter:LowPassFilter, line:Streamline):
        painter = active_filter.painters[0]
        config = painter.config
        # Find out where to probe
        probe_spots = line.get_side_probes(config.sample_radius * config.filter_radius / config.filter_xsize)
        # If the line has no points yet, we skip it
        if probe_spots is None:
            line.desired_move = None
            line.desired_move_len = 0
            return
        
        # Get the the raster coordinates of the probe spots
        grid_vals:np.ndarray = painter.transform_points_target(probe_spots)[..., :2].astype(int)
        grid_vals.clip((0,0), painter.target.size - 1, out=grid_vals)
        
        # The probe spot count is always n * 2, where n is the number of front and back probes
        probe_count = probe_spots.shape[0]
        left_spots, right_spots = grid_vals[:probe_count//2], grid_vals[probe_count//2:]

        # Get the energy value on both sides
        right_brightnesses = painter.target.data[right_spots[:,0], right_spots[:,1]]
        left_brightnesses  = painter.target.data[left_spots[:,0], left_spots[:,1]]

        # The difference is how much we want to move
        right_desire = np.average(right_brightnesses) - config.target_energy
        left_desire  = np.average(left_brightnesses)  - config.target_energy
        delta = (right_desire - left_desire) * config.delta_move

        # Calculate the herustic move that improves the line.
        move = _normalize(line.velocities.T[line.current_backward_segments-1])[0] @ ((0,1,0), (-1,0,0), (0,0,1)) * delta
        line.desired_move = move

        # The higher the magnitude, the more we want to move
        line.desired_move_len = np.linalg.norm(move)
    
    def _update_length_desire(self, active_filter:LowPassFilter, line:Streamline):
        ##### Check if we want to grow or shrink
        painter = active_filter.painters[0]
        config = painter.config

        back_probe_spot, front_probe_spot = line.get_end_probes(config.sample_radius * config.filter_radius / config.filter_xsize)
        # print(back_probe_spot, front_probe_spot)
        # print("Front Probe, Back Probe:", front_probe_spot, back_probe_spot)
        xmax, ymax = painter.target.size
        border = config.line_segment_size * .5
        if front_probe_spot is not None:
            ### Out of bounds, we just set the desires to 0, thereby skipping the respective side.
            fx, fy = painter.transform_points_target(front_probe_spot)[0, :2].astype(int)
            # print("front:", fx, fy, border, xmax-border, ymax-border)
            if (border <= fx <= xmax-border and border <= fy <= ymax-border):
                line.desired_front_change = config.target_energy - painter.target.data[fx, fy]
            else:
                line.desired_front_change = 0
        if back_probe_spot is not None:
            ### Out of bounds, we just set the desires to 0, thereby skipping the respective side.
            bx, by = painter.transform_points_target(back_probe_spot )[0, :2].astype(int)
            # print("Back:", bx, by, border, xmax-border, ymax-border)
            if (border <= bx <= xmax-border and border <= by <= ymax-border):
                line.desired_back_change = config.target_energy - painter.target.data[bx, by]
            else:
                line.desired_back_change = 0

    def make_suggestion(self):
        """Take an educated guess. Return the line, actionID, and values that are probably beneficial."""
        active_filter = self.filters.active_filter
        sorted_lines:list[Streamline] = list(sorted(active_filter._lines.values(), key=lambda line: line.total_desire, reverse=True))

        if (count := len(sorted_lines)) < 1:
            return False
        line_idx = int(np.random.random() * count / 8)
        line = sorted_lines[line_idx]
        self.update_line_desire(active_filter, line)
        # print(l.seed, l.desired_front_change, l.desired_back_change, l.desired_move_len)
        front = abs(line.desired_front_change)
        back = abs(line.desired_back_change)
        move = 4 * line.desired_move_len
        # print("Move len, Front change, Back change:", line.desired_move_len, line.desired_front_change, line.desired_back_change)
        do_move = False
        change_front = False
        lengthen = False
        if ALLOW_MOVE and((not ALLOW_LEN_CHANGE) or (move > front and move > back)):
            do_move = True
        else:
            change_front = front > back
            lengthen = np.sign(line.desired_front_change if change_front else line.desired_back_change)

        new_line = active_filter.move_and_change_length(line, do_move, change_front, lengthen > 0, recommended_move=line.desired_move)
        if new_line is not None:
            self.update_line_desire(active_filter, new_line)

        for i in range(5):
            line = sorted_lines[int(np.random.random() * count)]
            self.update_line_desire(active_filter, line)
        return new_line is not None


class FilterStack():
    def __init__(self) -> None:
        # Contains the last low pass filter
        self.old_filter:LowPassFilter|None = None
        # The filter we are currently using
        self.active_filter:LowPassFilter|None = None
        #### How strong we should try to cohere to the previous low pass filter's image
        # Range is 0 (no coherence) to 1.0 (full coherence)
        self.coherence_weight: float = 0.0
        # A line integrator used to integrate a streamline in-place
        self.line_integrator: Callable[[Streamline], None] = None
        # Keep track of the size of the fitlers
        self.aspect_ratio:float = 1.0
        self.xsize: int = -1
        self.interp = self.setup_blur_filter_values()

    @property
    def last_filter(self) -> LowPassFilter|None:
        """Get the last filter, or None if no last filter exists."""
        # return self.old_filters[-1] if len(self.old_filters) > 0 else None
        return self.old_filter #[-1] if len(self.old_filters) > 0 else None

    def get_filter_size(self, config:LPFilterConfig) -> tuple[int,int]:
        x = math.ceil(config.filter_xsize)
        y = math.ceil(config.filter_xsize * self.aspect_ratio)
        return x, y
    
    def new_filter(self, configs: list[LPFilterConfig]) -> LowPassFilter:
        assert len(configs) > 0
        painters: list[LPFilterPainter] = []
        for config in configs:
            assert config.filter_xsize == self.xsize
            painters.append(LPFilterPainter(config, LPFilterTarget(self.get_filter_size(config)), self.interp))
        # We set up the energy measure to use the first target
        space_target = painters[0].target
        return LowPassFilter(self.line_integrator, lambda: self._energy_function(space_target, *self._get_time_targets()), painters)

    def push_filter(self, new_filter:LowPassFilter):
        """Add a new filter and set it as the active one. If we already have an active filter, add it to old_filters first."""
        if self.active_filter is not None:
            self.old_filter = self.active_filter
        self.active_filter = new_filter
        return new_filter

    def get_highres_image(self, last=False):
        """Return a high-res image of the currently active filter or the previous one."""
        # TODO: Make it actually high-res
        return self.last_filter.total_low_pass_footprint if last else self.active_filter.total_low_pass_footprint

    def setup_blur_filter_values(self):
        """The filter uses indexing scheme [X,Y] to make it consistent with the coordinates."""
        samples = 30
        delta = 1 / (samples-1)
        xs, ys = np.arange(samples) * delta, np.arange(samples) * delta
        filter = (np.sqrt(xs[:,None] ** 2 + ys[None, :] ** 2)).clip(0, 1.0)
        filter = (2 * filter ** 3 - 3 * filter ** 2 + 1) * delta
        # The first column and last row of the filter should = 0.0
        values = np.cumsum(np.roll(filter, (0,1)), axis=1)
        return scipy.interpolate.RegularGridInterpolator((xs,ys), values, bounds_error=False, fill_value=0.0)

    def _get_time_targets(self):
        if not BIAS_COHERENCE or self.last_filter is None:
            return None, None
        return self.active_filter.painters[1].target, self.last_filter.painters[1].target
    
    def _spatial_energy(self, space_current_target:LPFilterTarget) -> float:
        return np.sum(np.square(space_current_target.data - self.active_filter._head_config.target_energy))

    def _temporal_energy(self, time_current_target:LPFilterTarget, time_last_target:LPFilterTarget) -> float:
        return np.sum(np.clip(np.square(time_current_target.data - time_last_target.data), 0, 1))

    def _energy_function(
            self,
            space_current_target:LPFilterTarget,
            time_current_target :LPFilterTarget|None,
            time_last_target    :LPFilterTarget|None
        ) -> float:

        spacial_energy = self._spatial_energy(space_current_target)
        
        if BIAS_COHERENCE and time_last_target is not None:
            # Get the time footprint, not the spatial
            temporal_energy = self._temporal_energy(time_current_target, time_last_target)
            a = 1 - self.coherence_weight
            b = self.coherence_weight
            return a * spacial_energy + b * temporal_energy
        return spacial_energy

# TODO: Oracle for timecoherence

@smproxy.filter(label="TurkBanksPathlineFilter")
@smproperty.xml("""
    <OutputPort index="0" name="Stream Lines" type="vtkPolyData"/>
    <OutputPort index="1" name="Energy Map" type="vtkImageData"/>
""")
@smproperty.input(name="Input")
class TurkBanksPathlineFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, inputType='vtkImageData', nOutputPorts=2)
        ##### Sub-components are fully init'd in RequestData
        self.oracle       : Oracle      = None
        self.filter_stack : FilterStack = FilterStack()
        self.filter_stack.line_integrator = self.generate_streamline
        
        ##### These are set in RequestInformation to account for things like size of the vector field/viewport
        # The X and Y pixel count of the filters
        self.rasterx      : int   = -1
        self.rastery      : int   = -1
        
        # The lo(w) and hi(gh) bounds and extents.
        # The extents are sample coordinates on the grid (ints), the bounds are the size of the vector field (float) inside VTK
        self.lo_extent    : np.ndarray|None = None
        self.hi_extent    : np.ndarray|None = None
        self.delta_extent : np.ndarray|None = None
        self.lo_bounds    : np.ndarray|None = None
        self.hi_bounds    : np.ndarray|None = None
        self.delta_bounds : np.ndarray|None = None

        ##### Time Coherence
        self.shards:list[Streamline] = None # Needed for shatter coherence
    
    def FillOutputPortInformation(self, port, info):
        """Set the ports' respective output types (port 0: vtkPolyData, port 1: vtkImageData)."""
        if port == 0:
            info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkPolyData")
        elif port == 1:
            info.Set(vtk.vtkDataObject.DATA_TYPE_NAME(), "vtkImageData")
        else:
            raise NotImplementedError(f"Got a request for port {port}, which is not set up.")
        return 1

    @smproperty.doublevector(name="Time Coherence Weight", default_values=.0)
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetTimeCoherenceWeight(self, val):
        """Set how much to cohere to the previous frame (0 = No coherence, 1 = Full coherence)."""
        self.filter_stack.coherence_weight = val
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
        """Fill the ouput port information and perofmr initial setup."""
        SDDP = vtk.vtkStreamingDemandDrivenPipeline
        inImageData = vtk.vtkImageData().GetData(inInfo[0])
        vtk_extent = inImageData.GetExtent()
        vtk_bounds = inImageData.GetBounds()

        ### Calculate bounds and extents of the vector field
        self.lo_extent = np.array(vtk_extent[::2])
        self.hi_extent = np.array(vtk_extent[1::2])
        self.delta_extent = self.hi_extent - self.lo_extent
        self.lo_bounds = np.array(vtk_bounds[::2], dtype=np.float32)
        self.hi_bounds = np.array(vtk_bounds[1::2], dtype=np.float32)
        self.delta_bounds = self.hi_bounds - self.lo_bounds

        self._inital_config = LPFilterConfig(separation=0.1, filter_radius=10, line_segment_size=0.01)

        ### Obtain the aspect ratio
        self.filter_stack.xsize = self._inital_config.filter_xsize
        self.filter_stack.aspect_ratio = self.delta_extent[1] / self.delta_extent[0]
        self.rasterx = math.ceil(self.filter_stack.xsize)
        self.rastery = math.ceil(self.filter_stack.xsize * self.filter_stack.aspect_ratio)
        
        ### Fill the output port information
        outInfo0 = outInfo.GetInformationObject(0)
        outInfo0.Set(SDDP.WHOLE_EXTENT(), *vtk_extent)
        outInfo1 = outInfo.GetInformationObject(1)
        outInfo1.Set(SDDP.WHOLE_EXTENT(), 0, self.rasterx-1, 0, self.rastery-1, 0, 0)
        return 1

    def RequestData(self, request:vtk.vtkInformation, inInfo:tuple[vtk.vtkInformationVector], outInfo:vtk.vtkInformationVector):
        # Check what timestep we are currently at
        # global ALLOW_MOVE
        # ALLOW_MOVE = self.filter_stack.coherence_weight > 0
        timestep, _all_timesteps = self.GetUpdateTimestep()

        ##### Add a new filter to the stack and keep a reference to it
        configs:list[LPFilterConfig] = []
        configs.append(replace(self._inital_config))
        if BIAS_COHERENCE:
            # the filter_xsize, defined as sep_to_blur * filter_radius / separation
            # must equal the other configs' size, otherwise an exception is raised.
            default = configs[-1]
            _coh_factor = 2
            time_config:LPFilterConfig = replace(
                default,
                sep_to_blur=default.sep_to_blur * _coh_factor,
                filter_radius=default.filter_radius / _coh_factor
            )
            configs.append(time_config)
        self.filter_stack.push_filter(self.filter_stack.new_filter(configs))
        # if self.filter_stack.last_filter is not None:
        #     self.filter_stack.last_filter.painters[1].target.data.fill(0.0)
        current_filter = self.filter_stack.active_filter
        current_config = current_filter._head_config
        
        # Set up a quick runtime measurement
        t = time.time()

        # Obtain the vector field we want to visualize
        inImageData = vtk.vtkImageData.GetData(inInfo[0])
        
        # Set up the oracle
        self.oracle = Oracle(self.filter_stack)

        # Setup the vtkStreamTracer used to integrate lines
        self.stracer = vtk.vtkStreamTracer()
        self.stracer.SetInputData(inImageData)
        self.stracer.SetIntegratorTypeToRungeKutta4()
        self.stracer.SetIntegrationStepUnit(1)
        self.stracer.SetMaximumError(1)
        self.stracer.SetInitialIntegrationStep(current_config.line_segment_size * max(self.delta_bounds[:4]))
        
        # Configure the output ImageData object
        outImageData = vtk.vtkImageData.GetData(outInfo.GetInformationObject(1))
        outImageData.SetDimensions(self.rasterx, self.rastery, 1)
        outImageData.SetSpacing(1/(self.rasterx-1), 1/(self.rastery-1) * self.filter_stack.aspect_ratio, 1)

        # Sometimes, we want to automatically reset for the first frame
        if timestep is not None and timestep < 1e-6:
            print("Reset for first frame.")
            self.filter_stack.old_filter = None
            self.shards = None
        

        #### Some Information
        if LOG_PARAMS:
            print(f"\n\n{'== Pre-Configured Settings ==':^50}")
            print(f"{'HALO_RADIUS'       :<25}:{current_config.blur_min      :>10}")
            print(f"{'LINE_START_LENGTH' :<25}:{current_config.line_start_len:>10}")
            print(f"{'TARGET_BRIGHTNESS' :<25}:{current_config.target_energy :>10}")
            print(f"{'MOVE_DELTA'        :<25}:{current_config.delta_move    :>10}")
            print(f"{'LEN_DELTA'         :<25}:{current_config.delta_len     :>10}")
            print(f"{'BIAS'              :<25}:{BIAS_COHERENCE    :>10}")
            print(f"{'SHATTER'           :<25}:{SHATTER_COHERENCE :>10}")
            print(f"\n{'== Auto-Configured Settings ==':^50}")
            print(f"{'Raster Size'       :<25}:" + f"{str(self.rasterx) + ' * ' + str(self.rastery):>10}")
            print(f"{'Line join radius'  :<25}:{current_filter._head_config.join_dist:>10.3f}")
            print(f"{'': >10}{'':=>30}{'': >10}\n\n")
        
        iterations = 0
        iterations = self.minimize_energy()
        # cProfile.runctx("iterations = self.minimize_energy()", {}, {"self":self})
        final_energy = current_filter.energy_function()
        
        # Add the final lines into our polydata instance to display them
        outPolyData = vtk.vtkPolyData.GetData(outInfo)
        self.draw_streamlines(current_filter, outPolyData)
        
        # Get the final data for some statistics and time coherence
        if BIAS_COHERENCE:
            target = current_filter.painters[1].target
        else:
            target = current_filter.painters[0].target
        if LOG_STATS:
            print(f"{'Min/Max brightness:':<20}", np.min(target.data), np.max(target.data))
            print(f"{'Lines:'             :<20}", len(current_filter._lines))
            print(f"{'Line Redraws:'      :<20}", LINE_REDRAW_COUNTER)
            print(f"{'Filter Recalcs:'    :<20}", FILTER_RECALC_COUNTER)
        print(f"===== Reached final energy {final_energy:.3f} (={final_energy*100/(target.size[0] * target.size[1]):.3f}%) after {iterations} steps.")
        print(f"> Reloaded after {time.time() - t:.3f}s")
       
        ### Draw the energies to the output ImageData object
        out_img_array = np.zeros((*target.data.shape, 3), dtype=np.uint8)
        
        # Draw current energy via green channel. Optimal level is 127 out of 255
        out_img_array[:,:,1] = (target.data / (2 * current_config.target_energy)).clip(0,1) * 255

        # Take a sample along the x or y axis
        if False:
            sample_vals = target.data[20, :].round(3)
            print(sample_vals)
        
        # If we want coherence, add the last frame's energy as the red channel => yellow for perfect time coherence
        if BIAS_COHERENCE and self.filter_stack.last_filter is not None:
            last_filter = self.filter_stack.last_filter
            last_target = last_filter.painters[1].target
            # Draw the previous energy if we use the bias. Actually draw it anyway since it is useful
            # inc = self.incoherence_array(current_filter.total_low_pass_footprint)
            out_img_array[:,:,0] = (last_target.data / (2 * last_filter._head_config.target_energy)).clip(0, 1) * 255
            # Some stats if we want them
            # if LOG_STATS:
            #     print(f"{'Min/Max old:'  :<20}", np.min(last), np.max(last))
            #     print(f"{'Incoherence:'  :<20}", np.min(inc), np.max(inc))
            
        # If we got our seeds by shattering, draw the shattered seeds
        if SHATTER_COHERENCE:
            if self.shards is not None:
                for shard in self.shards:
                    seed = shard.seed
                    # Draw shattered seeds
                    filter_pts = current_filter.painters[0].transform_points_target(seed[:,None]).round(0).astype(int)
                    out_img_array[filter_pts[0,...], filter_pts[1,...]] = [255, 0 ,255]
            self.shards = current_filter.shatter()
        flipped = np.moveaxis(out_img_array, (0,1), (1,0)).reshape(self.rasterx * self.rastery, 3)
        outImageData.GetPointData().SetVectors(dsa.numpyTovtkDataArray(flipped))
        # blended.save(f"/home/alba/projects/Streamlines/graphics/{timestep}.png")
        return 1

    def minimize_energy(self, max_iterations=750, allow_generation=True, allow_birth=True):
        np.random.seed(0)
        global LINE_REDRAW_COUNTER, FILTER_RECALC_COUNTER, ALLOW_LEN_CHANGE, ALLOW_MOVE
        ALLOW_LEN_CHANGE = True
        ALLOW_MOVE = True
        active_filter = self.filter_stack.active_filter
        iterations = 0
        t1 = time.time()
        if allow_generation or self.shards is not None:
            self.generate_initial_streamlets(active_filter)
        else:
            ###### Some quick testing
            def spawnline(x,y):
                l = Streamline(active_filter._head_config.line_start_len/2, active_filter._head_config.line_start_len/2)
                l.seed = np.array((x,y, self.lo_bounds[2]))
                active_filter.add_line(l, force=False)
                return l
            line1 = spawnline(.5,.5)
            # line1 = spawnline(.5,.65)
            # line1 = spawnline(.5,.35)
            # line1 = spawnline(.6,.5)
        # if self.shards is not None:
        #     return 0
        energy = active_filter.energy_function()
        print("Initial streamlet count | energy | time:", len(active_filter._lines), energy, f"{time.time()-t1:.3f}")
        accepted_oracles = 0
        accepted = 0

        # Try birthing lines at the start to fill gaps from the inital seeding
        if allow_birth:
            self._try_birth_line()

        # svals = []
        # tvals = []
        # Improve streamlines a number of times
        while iterations < max_iterations:
            ###### print time cohrence vs image quality
            # cts = self.filter_stack.active_filter.painters[0].target
            # svals.append(round(self.filter_stack._spatial_energy(cts), 3))
            # if self.filter_stack.last_filter is not None:
            #     ctt, ltt = self.filter_stack.active_filter.painters[1].target, self.filter_stack.last_filter.painters[1].target
            #     tvals.append(round(self.filter_stack._temporal_energy(ctt, ltt), 3))


            # Try moving a random streamline in a random direction and changing the lengths of both ends
            if self._try_move_and_ls_line():
                accepted += 1
            # Try birthing lines
            if allow_birth and iterations % 100 == 0 and self._try_birth_line() > 0:
                accepted += 1
            # Try joining two lines
            if ALLOW_LEN_CHANGE and self._try_join_lines() is not None:
                accepted += 1
            # Try the oracle's suggestion
            if self.oracle.make_suggestion():
                accepted_oracles += 1
            iterations += 1
        if iterations > 0:
            accept_percent = 100 * accepted         / iterations
            oracle_percent = 100 * accepted_oracles / iterations
            print(f"Accepted {accepted} out of {iterations} ({accept_percent:.3f}%). Oracle accepts: {accepted_oracles} ({oracle_percent:.3f}%)")
        # print("Svals:", svals)
        # print("Tvals:", tvals)
        ###### Print the lines to re-use them if necessary.
        # for line in self.filter_stack.active_filter._lines.values():
        #     print(f"{line.seed, line.forward_length, line.backward_length}")


        return iterations

    def generate_initial_streamlets(self, grid:LowPassFilter):
        active_filter = self.filter_stack.active_filter
        start_len = active_filter._head_config.line_start_len
        random_placement = False # Generate 300 random lines instead
        lo_x, lo_y, lo_z = self.lo_bounds
        hi_x, hi_y, hi_z = self.hi_bounds
        d_x, d_y, d_z = self.delta_bounds
        xcount = 12 # math.ceil(self.rasterx / (self.timeGrid.grid_halo_radius * 1.5))
        ycount = 9 # math.ceil(self.rastery / (self.timeGrid.grid_halo_radius * 1.5))
        delta = (d_x) / xcount
        if SHATTER_COHERENCE and self.shards is not None:
            for shard in self.shards:
                grid.add_line(shard)
        else:
            if random_placement:
                total_count = 300 # xcount * ycount
                seeds = np.random.random((total_count, 2))
                xyzgenerator = ((x * d_x, y * d_y, 0) for x,y in seeds)
            else:
                xvals = np.linspace(delta/2, 1 - delta/2, xcount+1)
                yvals = np.linspace(delta/2, 1 - delta/2, ycount+1)
                xyzgenerator = itertools.product(xvals, yvals, [lo_z])
            for x,y,z in xyzgenerator:
                line = Streamline(start_len/2, start_len/2)
                line.seed = np.array([x,y,z])
                # Add line if it improves the image
                grid.add_line(line)
        for line in active_filter._lines.values():
            self.oracle.update_line_desire(active_filter, line)

    def draw_streamlines(self, grid:LowPassFilter, polydata:vtk.vtkPolyData):
        # points = vtk.vtkPoints()
        total_points = vtk.vtkPoints()
        polydata.Allocate(len(grid._lines.values()))
        global_idx = 0
        for line in grid._lines.values():
            # points.InsertPoints(points.GetNumberOfPoints(), line.points.GetNumberOfTuples(), 0, line.points)
            line_point_count = line.ptcount
            vtkpoints = vtk.vtkPoints()
            for point in (line.points.T * (1, self.filter_stack.aspect_ratio, 1)):
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
        self.stracer.SetIntegrationDirection(1)
        transform = np.array((*self.delta_bounds[:2], 1))
        transformed_seed = line.seed * transform
        # print("Integrating seed:", transformed_seed)
        length_transform = abs(np.max(transform))
        backward_points, backward_velocities = self.integrate_line_vtk(transformed_seed, line.backward_length)
        self.stracer.SetIntegrationDirection(0)
        forward_points, forward_velocities = self.integrate_line_vtk(transformed_seed, line.forward_length)
        # print(forward_points.shape, backward_points.shape)
        # Save points in order tail >= seed >= head, guaranteed to be inside the domain, with the duplicate seed removed
        line.points     = np.concatenate((np.flip(backward_points,     1), forward_points[..., 1:]    ), axis=1) / (transform[:, None])
        line.velocities = np.concatenate((np.flip(backward_velocities, 1), forward_velocities[..., 1:]), axis=1)
        line.current_backward_segments = backward_points.shape[1]
        line.current_forward_segments = forward_points.shape[1] - 1

    def integrate_line_vtk(self, start:np.ndarray, maxlen:float):
        seedpts = vtk.vtkPoints()
        seedpts.InsertNextPoint(start)
        seedpt_set = vtk.vtkPointSet()
        seedpt_set.SetPoints(seedpts)
        self.stracer.SetSourceData(seedpt_set)
        self.stracer.SetMaximumPropagation(maxlen * max(self.delta_bounds[:4]))
        self.stracer.Update()
        out = self.stracer.GetOutput()
        outpts = np.array(dsa.WrapDataObject(out).GetPoints()).T
        vtkvels = out.GetPointData().GetArray("data")
        if vtkvels is None:
            vtkvels = out.GetPointData().GetArray("velocity")
        outvels = np.array(numpy_support.vtk_to_numpy(vtkvels)).T
        valid = (self.lo_bounds[:,None] <= outpts).all(axis=0) & (outpts <= self.hi_bounds[:,None]).all(axis=0)
        return outpts[..., valid], outvels[..., valid]

    def incoherence_array(self, arr:np.ndarray):
        ### Compare arr to the previous brightness
        return np.square(arr - self.filter_stack.last_filter.total_low_pass_footprint)

    ### Grid modifier actions

    def _get_random_line(self) -> Streamline:
        return np.random.choice(list(self.filter_stack.active_filter._lines.values()))

    def _try_move_and_ls_line(self):
        line = self._get_random_line()
        return self.filter_stack.active_filter.move_and_change_length(line, ALLOW_MOVE, 0, 0, True )

    def _try_birth_line(self):
        fil = self.filter_stack.active_filter
        painter = fil.painters[0]
        target = painter.target
        margin = fil._head_config.filter_radius / 5
        inside_border = target.data[math.floor(margin):math.ceil(self.rasterx-margin), math.floor(margin):math.ceil(self.rastery-margin)]
        birth_locations = np.argwhere(inside_border <= .02)
        births = 0
        for i in range(birth_locations.shape[0]):
            if target.data[birth_locations[i,0], birth_locations[i,1]] <= .02:
                s = Streamline(fil._head_config.line_start_len/2, fil._head_config.line_start_len/2)
                s.seed = np.append(birth_locations[i] / (target.size), self.lo_bounds[2]) 
                if fil.add_line(s) is not None:
                    births += 1
        return births

    def _try_join_lines(self):
        fil = self.filter_stack.active_filter
        pairs = fil.get_join_candidates()
        if len(pairs) > 1:
            # Try twice
            return fil.join_lines(*pairs[np.random.choice(len(pairs))]) or fil.join_lines(*pairs[np.random.choice(len(pairs))])

def _normalize(arr:np.ndarray):
    arr = np.atleast_2d(arr)
    return arr / np.linalg.norm(arr, axis=-1)[:, None]