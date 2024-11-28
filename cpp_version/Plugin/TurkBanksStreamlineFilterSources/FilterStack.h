#pragma once

#include "LowPassFilter.h"
#include "FilterConfig.h"
#include "FilterPainter.h"
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>
#include <functional>
#include <array>


class FilterStack{
    typedef std::tuple<int, int> FilterSize;

    public:
    FilterStack() = default;

    float AspectRatio = 1.0;
    float CoherenceWeight = 0.0;
    std::shared_ptr<LowPassFilter> ActiveFilter = nullptr;
    std::shared_ptr<LowPassFilter> PreviousFilter = nullptr;

    constexpr FilterSize getFilterSize (const FilterConfig& config){
        return {ceil(config.filterSizeX()), ceil(AspectRatio * config.filterSizeX())};
    }

    std::shared_ptr<LowPassFilter> CreateFilter(std::vector<FilterConfig> configs, double coherenceWeight){
        std::vector<std::shared_ptr<FilterPainter>> painters;
        for (auto config : configs){
            auto [x, y]  = getFilterSize(config);
            painters.push_back(std::make_shared<FilterPainter>(config, FilterTarget(x, y)));
        }
        return std::make_shared<LowPassFilter>(painters, nullptr, coherenceWeight);
    }

    std::shared_ptr<LowPassFilter> CopyFilter(std::shared_ptr<const LowPassFilter> filter){
        return std::make_shared<LowPassFilter>(*filter);
    }

    std::shared_ptr<LowPassFilter> PushFilter(std::shared_ptr<LowPassFilter> newFilter){
        if (ActiveFilter){
            PreviousFilter = ActiveFilter;
        }
        if (PreviousFilter){
            newFilter->oldFilter = PreviousFilter.get();
        }
        ActiveFilter = newFilter;
        return newFilter;
    }
};
