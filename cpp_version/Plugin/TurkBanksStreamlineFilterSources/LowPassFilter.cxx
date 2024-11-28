#include "LowPassFilter.h"
#include "Streamline.h"
#include "TurkBanksStreamlineFilterSourcesModule.h"
#include <vtkPoints.h>
#include <algorithm>

LowPassFilter::LowPassFilter(const LowPassFilter &filter):LowPassFilter(filter.painters, nullptr, filter.coherenceWeight){
    this->painters.clear();
    for (auto &painter_shptr : filter.painters){
        auto newPainter = std::make_shared<FilterPainter>(*painter_shptr);
        this->painters.push_back(newPainter);
    }
    this->mStreamlines.clear();
    for (auto &streamline_shptr : filter.mStreamlines){
        auto newStreamline = std::make_shared<Streamline>(*streamline_shptr);
        this->mStreamlines.push_back(newStreamline);
        for (auto &painter : painters){
            painter->footprints[newStreamline.get()] = painter->footprints[streamline_shptr.get()];
            painter->footprints.erase(streamline_shptr.get());
        }   
    }
    this->coherenceWeight = filter.coherenceWeight;
    this->desires.clear();
}

void LowPassFilter::AddLine(std::shared_ptr<Streamline> streamline){
    mStreamlines.push_back(streamline);
    for (auto&& painter : painters){
        painter->AddLine(streamline);
    }
}

void LowPassFilter::RestoreLine(std::shared_ptr<Streamline> streamline){
    mStreamlines.push_back(streamline);
    for (auto&& painter : painters){
        painter->RestoreLine(streamline);
    }
}

void LowPassFilter::CacheLine(std::shared_ptr<Streamline> streamline){
    mStreamlines.erase(std::remove(mStreamlines.begin(), mStreamlines.end(), streamline), mStreamlines.end());
    for (auto&& painter : painters){
        painter->CacheLine(streamline);
    }
}

void LowPassFilter::RemoveLine(std::shared_ptr<Streamline> streamline){
    mStreamlines.erase(std::remove(mStreamlines.begin(), mStreamlines.end(), streamline), mStreamlines.end());
    CacheLine(streamline);
    DropCaches();
}

std::shared_ptr<Streamline> LowPassFilter::ChangeLine(std::shared_ptr<Streamline> streamline, double frontChange,
                               double backChange, double xChange, double yChange) {
    // Store the energy before we do anything
    double energy = GetTotalEnergy();

    // If removing the line improves the energy, leave it out
    CacheLine(streamline);
    if (GetTotalEnergy() < energy){
        DropCaches();
        return nullptr;
    }
    double newSeed[3];
    newSeed[0] = std::clamp<double>(streamline->mSeed[0] + xChange, 0, 1);
    newSeed[1] = std::clamp<double>(streamline->mSeed[1] + yChange, 0, 1);
    newSeed[2] = streamline->mSeed[2];
    auto newLine = std::make_shared<Streamline>(
        newSeed,
        streamline->mVectorField,
        streamline->mFrontLength + frontChange,
        streamline->mBackLength  + backChange
    );
    AddLine(newLine);

    // If the change was good, keep the new line and return it
    if (GetTotalEnergy() < energy){
        return newLine; 
    }

    // Otherwise, add the old one back and return that instead
    CacheLine(newLine);
    RestoreLine(streamline);
    DropCaches();
    return streamline;
}

constexpr bool dist(double* a, double* b, double d){
    return std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2) < std::pow(d, 2);
}

std::vector<std::pair<std::shared_ptr<Streamline>, std::shared_ptr<Streamline>>> LowPassFilter::GetJoinCandidates(){
    // Get Pairs of lines that are close, join order is [head, tail]
    std::vector<std::pair<std::shared_ptr<Streamline>, std::shared_ptr<Streamline>>> pairs;
    auto&& painter = painters[0];
    for (auto&& streamline1 : mStreamlines){
        for (auto&& streamline2 : mStreamlines){
            if (streamline1 == streamline2)
                continue;
            auto&& pts1 = streamline1->mPoints;
            auto&& pts2 = streamline2->mPoints;
            int pts1len = pts1->GetNumberOfPoints();
            int pts2len = pts2->GetNumberOfPoints();
            if (pts1len < 1 || pts2len < 1)
                continue;
            if (dist(pts1->GetPoint(pts1len-1), pts2->GetPoint(0), painter->config.joinDistance())){
                pairs.push_back({streamline1, streamline2});
            }
        }
    }
    return pairs;
}

std::shared_ptr<Streamline> LowPassFilter::JoinLines(std::shared_ptr<Streamline> l1, std::shared_ptr<Streamline> l2){
    // Join l1's head with l2's tail
    double newSeed[3];
    auto&& painter = painters[0];
    auto&& pts1 = l1->mPoints;
    auto&& pts2 = l2->mPoints;
    int pts1len = pts1->GetNumberOfPoints();
    int pts2len = pts2->GetNumberOfPoints();
    double weight = pts2len / ((double)pts1len + pts2len);
    double headX, headY, tailX, tailY;
    l1->GetHead(headX, headY);
    l2->GetTail(tailX, tailY);
    newSeed[0] = (1-weight) * headX + weight * tailX;
    newSeed[1] = (1-weight) * headY + weight * tailY;
    newSeed[2] = l1->mSeed[2];
    auto newLine = std::make_shared<Streamline>(newSeed, l1->mVectorField, l2->mBackLength + l2->mFrontLength, l1->mBackLength + l1->mFrontLength);
    return newLine;
}

std::vector<std::tuple<std::array<double, 3>, double, double>> LowPassFilter::Shatter() const{
    std::vector<std::tuple<std::array<double, 3>, double, double>> shards;
    const double segSize = painters[0]->config.segmentSize;
    const double shardLenHalf = 0.03;
    const int idxStart = (shardLenHalf / segSize);
    const int idxStep  = std::ceil(2 * (shardLenHalf / segSize));
    for (const auto &streamline : mStreamlines){
        const auto &pts = streamline->mPoints;
        const int ptcount = pts->GetNumberOfPoints();
        if (ptcount < 2){continue;}
        if (ptcount < idxStep){
            double coords[3];
            pts->GetPoint(ptcount/2, coords);
            shards.push_back(std::tuple(std::array<double, 3>{coords[0], coords[1], coords[2]}, ptcount * segSize/2, ptcount * segSize/2));
            continue;
        }
        for (int i=idxStart; i<ptcount; i+=idxStep){
            double coords[3];
            pts->GetPoint(i, coords);
            shards.push_back(std::tuple(std::array<double, 3>{coords[0], coords[1], coords[2]}, shardLenHalf, shardLenHalf));
        }
    }
    return shards;
}