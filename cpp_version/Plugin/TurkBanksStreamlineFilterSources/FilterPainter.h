#pragma once

#include <map>
#include <functional>
#include <memory>
#include <cmath>

#include "FilterTarget.h"
#include "FilterConfig.h"
#include "Streamline.h"

class FilterPainter{
    public:
    FilterPainter(
        const FilterConfig& config,
        const FilterTarget& target
    );

    FilterConfig config;
    FilterTarget target;
    std::map<const Streamline*, std::shared_ptr<std::vector<double>>> footprints;
    std::map<const Streamline*, std::shared_ptr<std::vector<double>>> cachedFootprints;
    
    void AddLine    (std::shared_ptr<const Streamline> streamline);
    void CacheLine  (std::shared_ptr<const Streamline> streamline);
    void RestoreLine(std::shared_ptr<const Streamline> streamline);
    void DropCaches (){cachedFootprints.clear();}
    int filterSizePixels = 30;
    std::vector<std::vector<double>> filter;

    private:
    double maxSegmentDistance;

    void CalcFilter();
    float GetGrayscale(double x, double y);
    void CalcSegmentEquations(
        const std::vector<double>& x1,
        const std::vector<double>& y1,
        const std::vector<double>& x2,
        const std::vector<double>& y2,
        std::vector<double>& a,
        std::vector<double>& b,
        int count
    );
    void DrawLineSegmented(
        const std::vector<double>& x1,
        const std::vector<double>& y1,
        const std::vector<double>& x2,
        const std::vector<double>& y2,
        const std::vector<double>& a,
        const std::vector<double>& b,
        const Streamline* streamline,
        int count
    );
    void DrawLine(const Streamline *streamline);


    float radial_value(float x, float y)
    {
        x = fabs(x);
        y = fabs(y);

        int i = (int) floor((filterSizePixels-1) * x);
        int j = (int) floor((filterSizePixels-1) * y);

        if (i == filterSizePixels - 1)
            i = filterSizePixels - 2;

        if (j == filterSizePixels - 1)
            j = filterSizePixels - 2;

        float tx = (filterSizePixels-1) * x - i;
        float ty = (filterSizePixels-1) * y - j;
        
        float s00 = filter[i][j];
        float s01 = filter[i][j+1];
        float s10 = filter[i+1][j];
        float s11 = filter[i+1][j+1];

        float s0 = s00 + ty * (s01 - s00);
        float s1 = s10 + ty * (s11 - s10);
        float s  = s0  + tx * (s1 - s0);

        return (s);
    }


};

// class LPFilterPainter:
//     def __init__(self, config:LPFilterConfig, target:LPFilterTarget, filter_source:Callable[[np.ndarray], float]) -> None:
//         self.config: LPFilterConfig = config
//         self.target: LPFilterTarget = target
//         self.filter_source = filter_source
//         self.footprints: dict[Streamline, np.ndarray] = {}
//         max_seg_distance = config.line_segment_size * config.filter_xsize + 1
//         filter_diameter = 2 * (config.filter_radius + 1)
//         self.max_seg_len = math.ceil(max_seg_distance + filter_diameter) + 1
    
//     def add_line(self, line:Streamline, footprint:np.ndarray|None=None):
//         # Generate the footprint if it wasnt already calculated
//         if footprint is None:
//             self.footprints[line] = np.zeros_like(self.target.data)
//             self.write_whole_footprint(line, self.footprints[line])
//         else:
//             self.footprints[line] = footprint
        
//         # Add it to the target
//         self.target.data += self.footprints[line]
    
//     def remove_line(self, line:Streamline):
//         # Remove the line and its footprint
//         footprint = self.footprints.pop(line)
//         self.target.data -= footprint

//     def write_whole_footprint(self, line:Streamline, target:np.ndarray):
//         global LINE_REDRAW_COUNTER
//         LINE_REDRAW_COUNTER += 1

//         segment_count  = line.ptcount - 1
//         if segment_count < 1:
//             return
//         max_segment_side_len = self.max_seg_len        
//         # Use the (float) coordinates in range [0,1] transformed into the target frame of reference
//         points = self.transform_points_target(line.points)
//         segment_starts = points[..., :-1]
//         segment_ends   = points[..., 1:]

//         # Pre-Generate a range of coordinates to be reused
//         mesh = np.meshgrid(np.arange(max_segment_side_len), np.arange(max_segment_side_len))
//         self.pixel_indices = np.flip(np.dstack(mesh).T, 0)

//         # Obtain factors from the line equation
//         a_vals, b_vals = self.calculate_segment_equations(segment_starts, segment_ends)

//         # Obtain segemnts in the shape (4, n), ordered (startx, starty, endx, endy)
//         combined_segs = np.vstack((segment_starts, segment_ends))

//         # Obtain segment bounds in the shape (4, n), ordered (lo_x, lo_y, hi_, hi_y)
//         segment_pixel_bounds = self.calculate_segment_bounds(combined_segs)

//         # Calculate each segment's brightness levels and write them to the target
//         for segmentidx in range(segment_count):
//             xmin, xmax, ymin, ymax = segment_pixel_bounds[..., segmentidx]
//             footprint = self.calculate_segment_pixel(
//                 combined_segs[0:2, segmentidx] - (xmin, ymin),
//                 combined_segs[2:4, segmentidx] - (xmin, ymin),
//                 xmax - xmin + 1,
//                 ymax - ymin + 1,
//                 a_vals[segmentidx],
//                 b_vals[segmentidx],
//                 self.config.blur_min
//             ).T
//             if footprint.shape != (xmax-xmin+1, ymax-ymin+1):
//                 print(f"Mismatch in line at {line.seed}, FL:{line.forward_length}, BL:{line.backward_length}")
//             target[xmin:xmax+1, ymin:ymax+1] += footprint
    
//     def calculate_segment_equations(self, segment_starts:np.ndarray, segment_ends:np.ndarray):
//         a_vals = segment_ends[1, ...] - segment_starts[1, ...]
//         b_vals = segment_starts[0, ...] - segment_ends[0, ...]
//         lengths = np.sqrt(np.square(a_vals) + np.square(b_vals))
//         a_vals /= lengths
//         b_vals /= lengths
//         return a_vals, b_vals

//     def calculate_segment_pixel(self, seg_start:np.ndarray, seg_end:np.ndarray, x:int, y:int, a:float, b:float, rad:float):
//         # Translate
//         pixel_indices = self.pixel_indices[:, :y, :x]
//         translated_seg_start = seg_start[:, None, None] - pixel_indices
//         translated_seg_end   =   seg_end[:, None, None] - pixel_indices
//         # Verticalize
//         vert_start_x = np.abs(a * translated_seg_start[0, ...] + b * translated_seg_start[1, ...])
//         vert_start_y =       -b * translated_seg_start[0, ...] + a * translated_seg_start[1, ...]
//         vert_end_x   = np.abs(a *   translated_seg_end[0, ...] + b *   translated_seg_end[1, ...])
//         vert_end_y   =       -b *   translated_seg_end[0, ...] + a *   translated_seg_end[1, ...]
//         # Scale
//         div = 1 / rad
//         vert_start_x *= div
//         vert_start_y *= div
//         vert_end_x   *= div
//         vert_end_y   *= div
//         # Clip
//         np.clip(vert_start_y, -1, 1, out=vert_start_y)
//         np.clip(  vert_end_y, -1, 1, out=vert_end_y  )
//         # Filter contribution
//         opposite_sides = vert_start_y * vert_end_y <= 0
//         segment_in_range = vert_start_x <= 1.0
//         p1 = np.abs(np.dstack((vert_start_x, vert_start_y)))
//         p2 = np.abs(np.dstack((vert_end_x, vert_end_y)))
//         brightness_start = np.zeros(p1.shape[:-1])
//         brightness_end = np.zeros(p2.shape[:-1])
//         brightness_start[segment_in_range] = self.filter_source(p1[segment_in_range])
//         brightness_end[segment_in_range] = self.filter_source(p2[segment_in_range])
//         final_brightness = np.abs(brightness_start + (np.where(opposite_sides, brightness_end, -brightness_end)))
//         # print(final_brightness.round(1))
//         return final_brightness

//     def calculate_segment_bounds(self, segs:np.ndarray):
//         size = self.target.size
//         # Indexing: X/Y, PairNumber, Start/End
//         min_x_each_segment = np.floor(np.min(segs[ ::2, ...], axis=0) - self.config.blur_min).clip(0, size[0]-1)
//         min_y_each_segment = np.floor(np.min(segs[1::2, ...], axis=0) - self.config.blur_min).clip(0, size[1]-1)
//         max_x_each_segment =  np.ceil(np.max(segs[ ::2, ...], axis=0) + self.config.blur_min).clip(0, size[0]-1)
//         max_y_each_segment =  np.ceil(np.max(segs[1::2, ...], axis=0) + self.config.blur_min).clip(0, size[1]-1)
//         # Indexing: PairNumber, minx/maxx/miny/maxy
//         pixel_bounds = np.vstack((
//             min_x_each_segment,
//             max_x_each_segment,
//             min_y_each_segment,
//             max_y_each_segment
//         )).astype(int)
//         return pixel_bounds

//     def transform_points_target(self, points:np.ndarray):
//         return points[:2, ...] * self.target.size[:,None] - .5