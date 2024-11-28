#pragma once

#include <vector>
#include <memory>
#include <numeric>

#include "FilterPainter.h"

class Streamline;

class LowPassFilter{
    public:
    LowPassFilter(
        std::vector<std::shared_ptr<FilterPainter>> painters,
        LowPassFilter *oldFilter = nullptr,
        double coherenceWeight = 0.0
        ):
        painters{painters},
        oldFilter{oldFilter},
        coherenceWeight{coherenceWeight}
    {}
    LowPassFilter(const LowPassFilter &filter);

    std::vector<std::shared_ptr<FilterPainter>> painters;
    std::vector<std::shared_ptr<Streamline>> mStreamlines;
    std::map<std::shared_ptr<Streamline>, double> desires;
    LowPassFilter *oldFilter = nullptr;
    double coherenceWeight;

    void AddLine(std::shared_ptr<Streamline> streamline);
    void RemoveLine(std::shared_ptr<Streamline> streamline);
    void CacheLine(std::shared_ptr<Streamline> streamline);
    void RestoreLine(std::shared_ptr<Streamline> streamline);
    void DropCaches(){for (auto&& painter : painters){painter->DropCaches();}}
    std::shared_ptr<Streamline> ChangeLine(
        std::shared_ptr<Streamline> streamline,
        double frontChange,
        double backChange,
        double xChange,
        double yChange
    );
    std::vector<std::pair<std::shared_ptr<Streamline>, std::shared_ptr<Streamline>>> GetJoinCandidates();
    std::shared_ptr<Streamline> JoinLines(std::shared_ptr<Streamline> l1, std::shared_ptr<Streamline> l2);
    std::vector<std::tuple<std::array<double, 3>, double, double>> Shatter() const;
    
    double GetTotalEnergy() const {
        const std::vector<double> *spatialData  = &painters[0]->target.data;
        const FilterPainter *temporalPainter    = GetTemporalPainter();
        const FilterPainter *oldTemporalPainter = GetOldTemporalPainter();

        // Calculate the energies        
        double spatialEnergy = GetSpatialEnergy(spatialData, painters[0]->config.targetEnergy);
        double temporalEnergy = spatialEnergy;
        if (temporalPainter && oldTemporalPainter){
            temporalEnergy = GetTemporalEnergy(&temporalPainter->target.data, &oldTemporalPainter->target.data);
        }
        double totalEnergy = (1 - coherenceWeight) * spatialEnergy + coherenceWeight * temporalEnergy;
        return totalEnergy;
    }
    
    double GetSpatialEnergy(const std::vector<double> *spacialArray, double target) const{
        double spatialEnergy = 0;
        for (auto v: *spacialArray)
            spatialEnergy += std::pow(v-target, 2);
        return spatialEnergy;
    }

    double GetTemporalEnergy(const std::vector<double> *temporalArray1, const std::vector<double> *temporalArray2) const{
        double temporalEnergy = 0;
        if (temporalArray1->size() != temporalArray2->size()){
            return -1;
        }
        for (int i = 0; i < temporalArray1->size(); i++)
            temporalEnergy += std::pow((*temporalArray1)[i] - (*temporalArray2)[i], 2);
        return temporalEnergy;
        //  Perhaps sth like this faster (if done correctly?)
        // return std::accumulate(arr.begin(), arr.end(), 0, 
        //     [target](double sum, double a){return sum + (a-target)*(a-target);}
        // );
    }

    FilterPainter *GetTemporalPainter() const{
        if (painters.size() > 1){
            return painters[1].get();
        }
        return nullptr;
    }
    
    FilterPainter *GetOldTemporalPainter() const{
        if (oldFilter && oldFilter->painters.size() > 1){
            return oldFilter->painters[1].get();
        }
        return nullptr;
    }

    double GetEnergyDiffAt(const double &x, const double &y) const{
        FilterPainter *spatialPainter = painters[0].get();
        FilterPainter *temporalPainter = GetTemporalPainter();
        FilterPainter *oldTemporalPainter = GetOldTemporalPainter();
        double spatialDiff = spatialPainter->target.Get(x, y) - spatialPainter->config.targetEnergy;
        if (!temporalPainter || !oldTemporalPainter)
            return spatialDiff;
        double temporalDiff = temporalPainter->target.Get(x,y) - oldTemporalPainter->target.Get(x,y);
        return (1 - coherenceWeight) * spatialDiff + coherenceWeight * temporalDiff;
    }
};
