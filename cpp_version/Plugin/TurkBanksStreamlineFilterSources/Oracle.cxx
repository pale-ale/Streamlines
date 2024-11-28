#include "Oracle.h"
#include "LowPassFilter.h"
#include "Streamline.h"
#include "VectorField.h"


#define square(a) (a)*(a)
static bool _inRange(const double &lo, const double &hi, const double &val){return lo <= val && val <= hi;}

double Oracle::_iterQuality(
    const double &startX, const double &minX, const double &maxX,
    const double &startY, const double &minY, const double &maxY,
    const double &stepSize, const double &stepCount,
    const VectorField *vectorField,
    const LowPassFilter *filter,
    const FilterConfig *config,
    const bool &useGreater
) {
    double x = startX, y = startY;
    double xVel, yVel;
    double quality = 0.0;
    int samples;
    for (samples = 0; samples < stepCount; ++samples){
        vectorField->Get(x, y, xVel, yVel);
        double factor = stepSize / sqrt(xVel * xVel + yVel * yVel);
        double brightnessDiff = filter->GetEnergyDiffAt(x, y); //target.Get(x, y) - config.targetEnergy;
        if ((useGreater && brightnessDiff > 0) || ((! useGreater) && brightnessDiff < 0)){
            quality += square(brightnessDiff);
        }
        x += xVel * factor;
        y += yVel * factor;
        ++samples;
        if (! (_inRange(minX, maxX, x) && _inRange(minY, maxY, y))){
            // cout << "x: " << minX << " < " << x << " < " << maxX << endl;
            // cout << "y: " << minY << " < " << y << " < " << maxY << endl;
            return 0;
        }
    }
    return samples > 0 ? quality / samples : 0;
}


std::shared_ptr<Streamline> Oracle::GetBadLine(LowPassFilter* filter){
    std::vector<std::shared_ptr<Streamline>> lines(filter->mStreamlines);
    if (lines.size() < 1){
        cout << "No lines to choose from!" << endl;
        return nullptr;
    }
    std::sort(lines.begin(), lines.end(), [](auto l1, auto l2){return l1->mTotalDesire > l2->mTotalDesire;});
    // for (auto&& l : lines){
    //     cout << *l << ": " << l->mFrontDesire << ", " << l->mBackDesire << ", " << l->mSideDesire << ", " << l->mTotalDesire << "\n";
    // }
    // cout << "\n";
    int randomIdx = drand48() * (lines.size()-1) / 8;
    return lines[randomIdx];
}

void Oracle::UpdateLineDesire(LowPassFilter *filter, std::shared_ptr<Streamline> streamline){
    UpdateLineEndDesire(filter, streamline);
    UpdateLineSideDesire(filter, streamline);
    streamline->mTotalDesire = fabs(streamline->mBackDesire) + fabs(streamline->mFrontDesire) + 4 * fabs(streamline->mSideDesire);
}

void Oracle::UpdateLineEndDesire(LowPassFilter *filter, std::shared_ptr<Streamline> streamline){
    const auto *config = &filter->painters[0]->config;
    const auto *vectorField = streamline->mVectorField;
    
    // Image borders
    const double minX =     2*config->segmentSize;
    const double minY =     2*config->segmentSize;
    const double maxX = 1 - 2*config->segmentSize;
    const double maxY = 1 - 2*config->segmentSize;
    // double minX = 2.0 / target.targetX; //2*config.segmentSize;
    // double minY = 2.0 / target.targetY; //2*config.segmentSize;
    // double maxX = 1 - 2.0 / target.targetX; //1 - 2*config.segmentSize;
    // double maxY = 1 - 2.0 / target.targetY; //1 - 2*config.segmentSize;

    // Sample info
    const int maxSampleCount = 5;
    double sampleStep = config->sampleRadius() / maxSampleCount / config->filterSizeX();
    double frontLengthenQuality = 0.0;
    double frontShortenQuality = 0.0;
    double backLengthenQuality = 0.0;
    double backShortenQuality = 0.0;

    // Get the start and end points
    double head[2];
    double tail[2];
    streamline->GetHead(head[0], head[1]);
    streamline->GetTail(tail[0], tail[1]);

    // Check forward lengthen quality
    frontLengthenQuality = _iterQuality(
        head[0], minX, maxX,
        head[1], minY, maxY,
        sampleStep, maxSampleCount,
        vectorField, filter, config, false
    );
    // Check backward shorten quality
    backShortenQuality = _iterQuality(
        tail[0], minX, maxX,
        tail[1], minY, maxY,
        sampleStep, maxSampleCount,
        vectorField, filter, config, true
    );
    // Change the integration direction to backward
    sampleStep *= -1;
    // Check forward shorten quality
    frontShortenQuality = _iterQuality(
        head[0], minX, maxX,
        head[1], minY, maxY,
        sampleStep, maxSampleCount,
        vectorField, filter, config, true
    );
    // Check backward lengthen quality
    backLengthenQuality = _iterQuality(
        tail[0], minX, maxX,
        tail[1], minY, maxY,
        sampleStep, maxSampleCount,
        vectorField, filter, config, false
    );
    // cout << frontLengthenQuality << ", " << backLengthenQuality << ", " << frontShortenQuality << ", " << backShortenQuality << "\n";
    // Set the desires accordingly
    streamline->mFrontDesire = frontShortenQuality > frontLengthenQuality ? -frontShortenQuality : frontLengthenQuality;
    streamline->mBackDesire  = backShortenQuality  > backLengthenQuality  ? -backShortenQuality  : backLengthenQuality;
}

void Oracle::UpdateLineSideDesire( LowPassFilter *filter, std::shared_ptr<Streamline> streamline){
    if (streamline->mPoints->GetNumberOfPoints() == 0){
        streamline->mSideDesire = 0;
        return;
    }
    const auto& config = filter->painters[0]->config;
    const auto& target = filter->painters[0]->target;

    // How many samples to take
    int sideSamples = 10; // ceil(streamline->mPoints->GetNumberOfPoints() / 3.0);

    // Sum of image brightness for samples on the left/right
    double leftBrightness;
    double rightBrightness;

    // Sample positions and the respective field velocities and normals
    double samplePoint[3];
    double sampleVel[3];
    double sampleNormal[2];

    // Image borders
    const double minX = 2*config.segmentSize;
    const double minY = 2*config.segmentSize;
    const double maxX = 1 - 2*config.segmentSize;
    const double maxY = 1 - 2*config.segmentSize;

    // Radius for samples to be taken
    const double sampleRadius = config.sampleRadius() / 5;
    // cout << endl;
    for (int i = 0; i < sideSamples; i++){
        int randomIdx = drand48() * streamline->mPoints->GetNumberOfPoints();
        streamline->mPoints->GetPoint(randomIdx, samplePoint);// i * 3, samplePoint);
        if (! (_inRange(minX, maxX, samplePoint[0]) && _inRange(minY, maxY, samplePoint[1]))){
            continue;
        }
        streamline->mVelocities->GetTuple(randomIdx, sampleVel);
        sampleNormal[0] = -sampleVel[1];
        sampleNormal[1] =  sampleVel[0];
        const double norm = sqrt(sampleNormal[0] * sampleNormal[0] + sampleNormal[1] * sampleNormal[1]);
        if (norm == 0){continue;}
        sampleNormal[0] *= sampleRadius / norm;
        sampleNormal[1] *= sampleRadius / norm;
        const double newL = std::pow(filter->GetEnergyDiffAt(samplePoint[0] + sampleNormal[0], samplePoint[1] + sampleNormal[1]), 2);
        const double newR = std::pow(filter->GetEnergyDiffAt(samplePoint[0] - sampleNormal[0], samplePoint[1] - sampleNormal[1]), 2);
        leftBrightness  += newL;
        rightBrightness += newR;
        // cout << newL << ", " << newR << ", " << sampleNormal[0] << ", " << sampleNormal[1] << endl;
        // cout << " sample: " << samplePoint[0] << ", " << samplePoint[1];
        // cout << "\n normal: "   << normal[0] << ", " << normal[1] << endl;
        // cout << "\n lx: "   << samplePoint[0] + normal[0] << " ly: " << samplePoint[1] + normal[1] << " lb: " << lb;
        // cout << "\n rx: "   << samplePoint[0] - normal[0] << " ry: " << samplePoint[1] - normal[1] << " rb: " << rb << endl;
    }
    // mSideDesire gets multiplied with a left normal, hence values > 0 produce a left shift
    // cout << *streamline << " LB:" << leftBrightness << " RB: " << rightBrightness << endl;
    streamline->mSideDesire = (rightBrightness - leftBrightness) / sideSamples;
}
