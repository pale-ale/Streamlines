#pragma once

#include <vector>
#include <memory>
#include <functional>

class FilterConfig;
class LowPassFilter;
class Streamline;
class VectorField;

class Oracle{
    public:
    // Set this proxy to a lambda computing the difference between actual and target brightness at a location
    std::function<double(double, double)> energyDiffProxy;
    
    std::shared_ptr<Streamline> GetBadLine(LowPassFilter* filter);
    void UpdateLineDesire(    LowPassFilter *filter, std::shared_ptr<Streamline> streamline);
    void UpdateLineEndDesire( LowPassFilter *filter, std::shared_ptr<Streamline> streamline);
    void UpdateLineSideDesire(LowPassFilter *filter, std::shared_ptr<Streamline> streamline);

    private:
    double _iterQuality(
        const double &startX, const double &minX, const double &maxX,
        const double &startY, const double &minY, const double &maxY,
        const double &stepSize, const double &stepCount,
        const VectorField *vectorField,
        const LowPassFilter *filter,
        const FilterConfig *config,
        const bool &useGreater
    );
};
