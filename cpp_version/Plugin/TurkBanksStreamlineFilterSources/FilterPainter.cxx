#include "FilterPainter.h"
#include <cmath>
#include <cstdio>
#include <iostream>

FilterPainter::FilterPainter(
    const FilterConfig& config,
    const FilterTarget& target
):
    config{config}, target{target},
    maxSegmentDistance{config.segmentSize * config.filterSizeX() + 1}
{
    CalcFilter();
}

void FilterPainter::AddLine(std::shared_ptr<const Streamline> streamline) {
    footprints[streamline.get()] = std::make_shared<std::vector<double>>(target.data.size());
    DrawLine(streamline.get());
}

void FilterPainter::CacheLine(std::shared_ptr<const Streamline> streamline) {
    auto& footprint = footprints[streamline.get()];
    cachedFootprints[streamline.get()] = footprint;
    for (int i = 0; i < footprint->size(); i++){
        target.data[i] -= (*footprint)[i];
    }
    footprints.erase(streamline.get());
}

void FilterPainter::RestoreLine(std::shared_ptr<const Streamline> streamline){
    auto& footprint = cachedFootprints[streamline.get()];
    footprints[streamline.get()] = footprint;
    for (int i = 0; i < footprint->size(); i++){
        target.data[i] += (*footprint)[i];
    }
    cachedFootprints.erase(streamline.get());
}

void FilterPainter::CalcSegmentEquations(
    const std::vector<double>& x1,
    const std::vector<double>& y1,
    const std::vector<double>& x2,
    const std::vector<double>& y2,
    std::vector<double>& a,
    std::vector<double>& b,
    int count
){
    double tmpA, tmpB, len;
    for (int i = 0; i < count; ++i){
        tmpA = y2[i] - y1[i];
        tmpB = x1[i] - x2[i];
        len = sqrt(tmpA*tmpA + tmpB*tmpB);
        if (len < 1e-10){
            len = 1.0;
        }
        a[i] = tmpA / len;
        b[i] = tmpB / len;
    }
}

void FilterPainter::DrawLineSegmented(
    const std::vector<double>& x1,
    const std::vector<double>& y1,
    const std::vector<double>& x2,
    const std::vector<double>& y2,
    const std::vector<double>& a,
    const std::vector<double>& b,
    const Streamline* streamline,
    int count
){
    double r = config.filterRadius();
    // Draw each segment
    auto& footprint = *footprints[streamline];
    double tx1, tx2, ty1, ty2; //translated
    double vx1, vx2, vy1, vy2; //verticalized
    int xMin, xMax, yMin, yMax;
    for (int i = 0; i < count; ++i){
        // Get each segment's min/max x/y bounds, including the filter radius
        double _x1 = x1[i] * target.targetX - .5;
        double _x2 = x2[i] * target.targetX - .5;
        double _y1 = y1[i] * target.targetY - .5;
        double _y2 = y2[i] * target.targetY - .5;
        // cout << x1[i] << ", " << x2[i] << endl;
        // cout << y1[i] << ", " << y2[i] << endl;
        xMin = std::clamp<int>(std::floor(std::min(_x1, _x2) - r), 0, target.targetX);
        xMax = std::clamp<int>(std::ceil (std::max(_x1, _x2) + r), 0, target.targetX);
        yMin = std::clamp<int>(std::floor(std::min(_y1, _y2) - r), 0, target.targetY);
        yMax = std::clamp<int>(std::ceil (std::max(_y1, _y2) + r), 0, target.targetY);
        for (int x = xMin; x < xMax; ++x){
            for (int y = yMin; y < yMax; ++y){
                // Translate segment bounds so that pixel (x,y) is the origin
                tx1 = x1[i] * target.targetX - x - .5;
                tx2 = x2[i] * target.targetX - x - .5;
                ty1 = y1[i] * target.targetY - y - .5;
                ty2 = y2[i] * target.targetY - y - .5;
                // Verticalize segments
                vx1 = fabs(a[i] * tx1 + b[i] * ty1);
                vx2 = fabs(a[i] * tx2 + b[i] * ty2);
                vy1 =      a[i] * ty1 - b[i] * tx1;
                vy2 =      a[i] * ty2 - b[i] * tx2;
                // Scale
                double div = 1 / r;
                vx1 *= div;
                vx2 *= div;
                vy1 *= div;
                vy2 *= div;
                // Clip to filter radius in y-direction
                vy1 = std::clamp<double>(vy1, -1, 1);
                vy2 = std::clamp<double>(vy2, -1, 1);
                // If the pixel is outside the filter radius, skip it
                if (vx1 > 1.0 || vx2 > 1.0){
                    continue;
                }
                // Filter values
                double b1 = radial_value(vx1, vy1);
                double b2 = radial_value(vx2, vy2);
                // If both y values lie on opposite sides, add them, otherwise subtract them
                int idx = y * target.targetX + x;
                if (vy1 * vy2 < 0){
                    target.data[idx] += fabs(b1 + b2);
                    footprint[idx]   += fabs(b1 + b2);
                } else {
                    target.data[idx] += fabs(b1 - b2);  
                    footprint[idx]   += fabs(b1 - b2);
                }
            }
        }
    }
}

void FilterPainter::DrawLine(const Streamline *streamline){
    vtkPoints *streamlinePoints = streamline->mPoints;
    int pointcount = streamlinePoints->GetNumberOfPoints();
    if (pointcount < 2){
        return;
    }
    int segmentcount = pointcount - 1;
    std::vector<double> x1 (segmentcount);
    std::vector<double> y1 (segmentcount);
    std::vector<double> x2 (segmentcount);
    std::vector<double> y2 (segmentcount);
    std::vector<double>  a (segmentcount);
    std::vector<double>  b (segmentcount);
    double* point;
    for (int i = 0; i < segmentcount; ++i){
        point = streamlinePoints->GetPoint(i);
        x1[i] = point[0];
        y1[i] = point[1];
        point = streamlinePoints->GetPoint(i+1);
        x2[i] = point[0];
        y2[i] = point[1];
    }
    CalcSegmentEquations(x1, y1, x2, y2, a, b, segmentcount);
    DrawLineSegmented(   x1, y1, x2, y2, a, b, streamline, segmentcount);
}

void FilterPainter::CalcFilter(){
    float delta = 1.0 / (filterSizePixels-1);
    for (int i = 0; i < filterSizePixels; i++)
        filter.push_back(std::vector<double>(filterSizePixels));
    for (int i = 0; i < filterSizePixels; i++) {
        float r = i * delta;
        float sum = 0;
        for (int j = 0; j < filterSizePixels; j++) {
            float h = j * delta;
            float t = sqrt (r*r + h*h);
            float f = (2 * t - 3) * t * t + 1;
            if (t > 1)
                f = 0;
            filter[i][j] = sum;
            sum += f * delta;
        }
    }
}
