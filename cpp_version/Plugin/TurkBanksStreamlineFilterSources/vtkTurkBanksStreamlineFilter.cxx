#include <vtkCellArray.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>
#include <vtkTurkBanksStreamlineFilter.h>
#include <vtkPoints.h>
#include <vtkImageData.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <vtkCompositeDataPipeline.h>
#include <vtkStreamTracer.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <memory>
#include <random>
#include <ranges>

#include "LowPassFilter.h"
#include "FilterPainter.h"
#include "Oracle.h"
#include "VectorField.h"

vtkStandardNewMacro(vtkTurkBanksStreamlineFilter);

//------------------------------------------------------------------------------
vtkTurkBanksStreamlineFilter::vtkTurkBanksStreamlineFilter()
{
    this->SetNumberOfInputPorts(1);
    this->SetNumberOfOutputPorts(3);
    oracle = new Oracle();
}

//------------------------------------------------------------------------------
bool vtkTurkBanksStreamlineFilter::GetNeedsUpdate()
{
    if (this->CurIteration < this->Iterations)
    {
        this->Modified();
        return true;
    }
        return false;
}

int vtkTurkBanksStreamlineFilter::FillInputPortInformation(int vtkNotUsed(port), vtkInformation* info)
{
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    return 1;
}

int vtkTurkBanksStreamlineFilter::FillOutputPortInformation(int port, vtkInformation* info)
{
    if (port == 0){
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkPolyData");
        return 1;
    }
    if (port == 1){
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
        return 1;
    }
    if (port == 2){
        info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
        return 1;
    }
    return 0;
}

int vtkTurkBanksStreamlineFilter::RequestUpdateExtent(vtkInformation* req, vtkInformationVector**inInfo, vtkInformationVector* outInfo){
    // Configure the update extent for requests in upstream direction
    int exts[6];
    int port = req->Get(vtkStreamingDemandDrivenPipeline::FROM_OUTPUT_PORT());
    auto inInfoObj = inInfo[0]->GetInformationObject(0);
    vtkStreamingDemandDrivenPipeline::GetWholeExtent(inInfoObj, exts);
    inInfoObj->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), exts, 6);
    // cout << "Set upstream update extent to " <<exts[0]<< " " <<exts[1]<< " " <<exts[2]<< " " <<exts[3]<< " " <<exts[4]<< " " <<exts[5]<< endl;
    return 1;
}

int vtkTurkBanksStreamlineFilter::RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector* outInfo){
    typedef vtkStreamingDemandDrivenPipeline SDDP;
    
    // Setup the filter configs
    m_spatialConf.separation = Separation;
    m_spatialConf.joinDistanceFactor = JoinDistanceFactor;
    m_spatialConf.lengthenFactor = LengthenFactor;
    m_spatialConf.segmentSize = IntegrationStepSize;
    m_spatialConf.resolutionFactor = ResolutionFactor;

    m_temporalConf = m_spatialConf;
    m_temporalConf.filterRadiusFactor = TemporalFilterRadius;
    NeedsRedraw = true;

    // Get the respective information objects
    auto iobj0 = outInfo->GetInformationObject(0);
    auto iobj1 = outInfo->GetInformationObject(1);
    auto iobj2 = outInfo->GetInformationObject(2);

    // Set the extents for the vtkPolyData object on output port 0
    int exts [6];
    SDDP::GetWholeExtent(iobj0, exts);
    SDDP::SetWholeExtent(iobj0, exts);
    double aspectRatio = (double)(exts[3] - exts[2]) / (exts[1] - exts[0]);
    // cout << "Set whole extent on port 0 to " <<exts[0]<< " " <<exts[1]<< " " <<exts[2]<< " " <<exts[3]<< " " <<exts[4]<< " " <<exts[5]<< endl;
    
    // Modify the extents for the spatial energy (vtkImageData object on output port 1)
    SDDP::GetWholeExtent(iobj1, exts);
    exts[1] = exts[0] + ceil(m_spatialConf.filterSizeX()) - 1;
    exts[3] = exts[2] + ceil(m_spatialConf.filterSizeX() * aspectRatio) - 1;
    exts[4] = exts[5] = 0;
    SDDP::SetWholeExtent(iobj1, exts);
    // cout << "Set whole extent on port 1 to " <<exts[0]<< " " <<exts[1]<< " " <<exts[2]<< " " <<exts[3]<< " " <<exts[4]<< " " <<exts[5]<< endl;
    
    // Modify the extents for the temporal energy (vtkImageData object on output port 2)
    SDDP::GetWholeExtent(iobj2, exts);
    exts[1] = exts[0] + ceil(m_temporalConf.filterSizeX()) - 1;
    exts[3] = exts[2] + ceil(m_temporalConf.filterSizeX() * aspectRatio) - 1;
    exts[4] = exts[5] = 0;
    SDDP::SetWholeExtent(iobj2, exts);
    // cout << "Set whole extent on port 2 to " <<exts[0]<< " " <<exts[1]<< " " <<exts[2]<< " " <<exts[3]<< " " <<exts[4]<< " " <<exts[5]<< endl;

    // Same timestep info for port 0
    iobj0->Remove(SDDP::TIME_STEPS());
    for (int i = 0; i <= 10; ++i){
        iobj1->Append(SDDP::TIME_STEPS(), i/10.0);
    }
    iobj0->Remove(SDDP::TIME_RANGE());
    iobj0->Append(SDDP::TIME_RANGE(), 0);
    iobj0->Append(SDDP::TIME_RANGE(), 1);

    // Set timestep information for port 1
    iobj1->Remove(SDDP::TIME_STEPS());
    for (int i = 0; i <= 10; ++i){
        iobj1->Append(SDDP::TIME_STEPS(), i/10.0);
    }
    iobj1->Remove(SDDP::TIME_RANGE());
    iobj1->Append(SDDP::TIME_RANGE(), 0);
    iobj1->Append(SDDP::TIME_RANGE(), 1);
    
    // Set timestep information for port 1
    iobj2->Remove(SDDP::TIME_STEPS());
    for (int i = 0; i <= 10; ++i){
        iobj2->Append(SDDP::TIME_STEPS(), i/10.0);
    }
    iobj2->Remove(SDDP::TIME_RANGE());
    iobj2->Append(SDDP::TIME_RANGE(), 0);
    iobj2->Append(SDDP::TIME_RANGE(), 1);

    // Set some values for the filter stack
    mFilterStack.AspectRatio = aspectRatio;
    return 1;
}

void vtkTurkBanksStreamlineFilter::GenerateInitialStreamlines(int linesX, int linesY){
    if (JoinDistanceFactor > 0){
        double stepX = 1.0 / (linesX+1);
        double stepY = 1.0 / (linesY+1);
        for (int i = 1; i < linesX+1; ++i){
            for (int j = 1; j < linesY+1; ++j){
                double seed[3] = {i*stepX, j*stepY, 0};
                auto streamline = std::make_shared<Streamline>(seed, vectorField, .075);
                mFilterStack.ActiveFilter->AddLine(streamline);
            }
        }
    }else{
        std::array<double, 3> seed;
        seed = {.5, .5, 0};
        auto streamline = std::make_shared<Streamline>(seed.data(), vectorField, .1);
        mFilterStack.ActiveFilter->AddLine(streamline);
        // seed = {.8, .5, 0};
        // streamline = std::make_shared<Streamline>(seed.data(), vectorField, .1);
        // mFilterStack.ActiveFilter->AddLine(streamline);
        // seed = {.6, .65, 0};
        // streamline = std::make_shared<Streamline>(seed.data(), m_integrator, .1);
        // mFilterStack.ActiveFilter->AddLine(streamline);
    }
}

void vtkTurkBanksStreamlineFilter::GenerateInitialStreamlines(std::vector<std::tuple<std::array<double, 3>, double, double >> shards){
    for (auto& [seed, frontlen, backlen] : shards){
        auto streamline = std::make_shared<Streamline>(seed.data(), vectorField, frontlen, backlen);
        mFilterStack.ActiveFilter->AddLine(streamline);
    }
}

bool vtkTurkBanksStreamlineFilter::MoveLineRandomly(LowPassFilter* filter, std::shared_ptr<Streamline> streamline, const FilterConfig* config){
    // Choose new lengths on both ends
    const double deltaLen  = config->deltaLen();
    double frontChange = (drand48() - .5) * 2 * deltaLen;
    double backChange  = (drand48() - .5) * 2 * deltaLen;
    if (streamline->mFrontLength + frontChange < 0){
        // If the front change would make the length negative, choose a random positive length
        frontChange = drand48() * deltaLen - streamline->mFrontLength;
    }
    if (streamline->mBackLength + backChange < 0){
        // If the back change would make the length negative, choose a random positive length
        backChange  = drand48() * deltaLen - streamline->mBackLength;
    }

    // Try the new line. Return true if it is better or the old line was fast deleted
    const double deltaMove = config->deltaMove();
    return streamline != filter->ChangeLine(
        streamline, frontChange, backChange,
        (drand48() - .5) * deltaMove,
        (drand48() - .5) * deltaMove
    );
}

void vtkTurkBanksStreamlineFilter::TryBirthLines(LowPassFilter* filter, const FilterConfig* config){
    bool ranWithoutBirth = false;
    bool gotBirth = false;
    double energy = filter->GetTotalEnergy();
    auto& target = filter->painters[0]->target;
    
    // Save the coordinate indices where the image is too dark
    std::vector<int> indices;
    for (int i = 0; i < target.data.size(); i++){
        if (target.data[i] < BirthThreshold){
            indices.push_back(i);
        }
    }

    // Shuffle those indices to not introduce a bias
    std::random_shuffle(indices.begin(), indices.end());

    // Walk along the indices and try to birth lines until they are exhausted, updating them every birth
    while (!indices.empty()){
        bool birth = false;
        for (auto idx : indices){
            int pixelX, pixelY;
            double seed[3];
            target.Coord(idx, pixelX, pixelY);
            seed[0] = (pixelX + .5) / target.targetX;
            seed[1] = (pixelY + .5) / target.targetY;
            seed[2] = 0;
            auto streamline = std::make_shared<Streamline>(seed, vectorField, .1);
            energy = filter->GetTotalEnergy();
            filter->AddLine(streamline);
            if (filter->GetTotalEnergy() < energy)
            {
                birth = true;
                break;
            } else {
                filter->RemoveLine(streamline);
            }
        }
        if (birth){
            std::vector<int> newIndices;
            for (int idx : indices){
                if (target.data[idx] <= BirthThreshold){
                    newIndices.push_back(idx);
                }
            }
            indices = newIndices;
        } else {
            break;
        }
    }
}

bool vtkTurkBanksStreamlineFilter::TryJoinLines(LowPassFilter* filter, const FilterConfig* config){
    double energy = filter->GetTotalEnergy();
    auto &&joinOptions = filter->GetJoinCandidates();
    if (joinOptions.size() > 0){
        auto candidate = joinOptions[drand48() * joinOptions.size()];
        auto sl1 = candidate.first;
        auto sl2 = candidate.second;
        auto joinedLine = filter->JoinLines(sl1, sl2);
        filter->CacheLine(sl1);
        filter->CacheLine(sl2);
        double removeEnergy = filter->GetTotalEnergy();
        filter->AddLine(joinedLine);
        double addEnergy = filter->GetTotalEnergy();
        if (addEnergy < energy || (addEnergy - energy) < (removeEnergy - energy) * .25){
            return true;
        } else {
            filter->CacheLine(joinedLine);
            filter->RestoreLine(sl1);
            filter->RestoreLine(sl2);
            filter->DropCaches();
        }
    }
    return false;
}

//------------------------------------------------------------------------------
int vtkTurkBanksStreamlineFilter::RequestData(vtkInformation*, vtkInformationVector** inputVector,
    vtkInformationVector* outputVector)
{
    // Obtain input vtkImageData
    vtkInformation* imageDataInInfo = inputVector[0]->GetInformationObject(0);
    imgData = vtkImageData::SafeDownCast(imageDataInInfo->Get(vtkDataObject::DATA_OBJECT()));

    // Setup our vector field with the input vtkImageData
    vectorField = new VectorField(imgData, &m_spatialConf);

    // Obtain output vtkPolyData (port 0) and vtkImageData (port 1) objects
    vtkInformation* polyDataOutInfo          = outputVector->GetInformationObject(0);
    vtkInformation* spatialImageDataOutInfo  = outputVector->GetInformationObject(1);
    vtkInformation* temporalImageDataOutInfo = outputVector->GetInformationObject(2);
    vtkPolyData*    polyDataOut          = vtkPolyData::SafeDownCast(polyDataOutInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkImageData*   spatialImageDataOut  = vtkImageData::SafeDownCast(spatialImageDataOutInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkImageData*   temporalImageDataOut = vtkImageData::SafeDownCast(temporalImageDataOutInfo->Get(vtkDataObject::DATA_OBJECT()));

    // Obtain the animation time requested by paraview and write it to the output information objects
    double time = outputVector->GetInformationObject(0)->Get(vtkStreamingDemandDrivenPipeline::UPDATE_TIME_STEP());

    // Check the objects for nullptrs
    if (! (imgData && polyDataOut && spatialImageDataOut && temporalImageDataOut)){
        vtkErrorMacro("Input ImageData or output PolyData/ImageData is null.");
        return 1;
    }

    // Set output time for paraview
    GetOutputInformation(0)->Set(vtkDataObject::DATA_TIME_STEP(), time);
    GetOutputInformation(1)->Set(vtkDataObject::DATA_TIME_STEP(), time);
    GetOutputInformation(2)->Set(vtkDataObject::DATA_TIME_STEP(), time);

    // Run the main loop or skip an update
    bool doOptimization = true; // NeedsRedraw;
    // Reset the runtime state of the plugin
    bool reset = !SingleFrameMode ;//|| time == 0;
    // Create a new, empty filter each update
    bool createFilter = (!m_shardInfo.empty()) || mFilterStack.ActiveFilter == nullptr || reset;
    // Spawn initial lines
    bool makeInitialLines = createFilter;

    // Setup the output objects
    if (reset){
        // Keeps track of how many time steps we passed
        CurIteration = 0;

        // Get the vector field size
        double vSpacing [3];
        int vDims [3]; // Dims = points, Exts = Dims - 1 = Cells
        imgData->GetSpacing(vSpacing);
        imgData->GetDimensions(vDims);
        VFieldX = vSpacing[0] * (vDims[0] - 1);
        VFieldY = vSpacing[1] * (vDims[1] - 1);
        // if (SingleFrameMode){
        //     mFilterStack.ActiveFilter   = nullptr;
        //     mFilterStack.PreviousFilter = nullptr;
        // }
        // cout << "Vector Field Size: " << VFieldX << ", " << VFieldY << endl;
    }

    NeedsRedraw = false;

    // Build a new filter from scratch
    if (createFilter){
        std::vector<FilterConfig> configs;
        if (CoherenceWeight != 0){
            configs = {m_spatialConf, m_temporalConf};
        } else {
            configs = {m_spatialConf};
        }
        mFilterStack.PushFilter(mFilterStack.CreateFilter(configs, CoherenceWeight));
    }

    // Make the initial lines
    if (!m_shardInfo.empty()){
        // Use the shards from the last frame
        GenerateInitialStreamlines(m_shardInfo);
        m_shardInfo.clear();
    } else if (makeInitialLines){
        // Fill image with initial lines on a rectilinear grid 
        GenerateInitialStreamlines(12, 10);
    }

    // Useful references
    auto *const filter = mFilterStack.ActiveFilter.get();
    auto *const spatialTarget = &filter->painters[0]->target;
    auto *const temporalTarget = filter->painters.size() > 1 ? &filter->painters[1]->target : nullptr;
    auto *const config = &filter->painters[0]->config;

    // Target dimensions for convenience
    const int spatialTargetX = spatialTarget->targetX;
    const int spatialTargetY = spatialTarget->targetY;
    const int temporalTargetX = temporalTarget ? temporalTarget->targetX : -1;
    const int temporalTargetY = temporalTarget ? temporalTarget->targetY : -1;

    // Set the output spatial imagedata to the target dimensions
    {
        int dimX = spatialTargetX;
        int dimY = spatialTargetY;
        int dimZ = 1;
        // cout << "Setting output vtkImageData dimensions to " << dimX << ", " << dimY << ", " << dimZ << endl;
        spatialImageDataOut->SetDimensions(dimX, dimY, dimZ);
        spatialImageDataOut->SetSpacing(VFieldX / (dimX-1), VFieldY / (dimY-1), 1);
    }

    // Set the output temporal imagedata to the target dimensions
    if (temporalTarget){
        int dimX = temporalTargetX;
        int dimY = temporalTargetY;
        int dimZ = 1;
        // cout << "Setting output vtkImageData dimensions to " << dimX << ", " << dimY << ", " << dimZ << endl;
        temporalImageDataOut->SetDimensions(dimX, dimY, dimZ);
        temporalImageDataOut->SetSpacing(VFieldX / (dimX-1), VFieldY / (dimY-1), 1);
    }

    // Setup statistics counters
    m_statistics.reset();

    // We sometimes want to skip optimization, e.g. for shattering in single frame mode
    if (doOptimization){
        // To animate this in paraview while still being responsive with few accepted optimizations
        Optimize(SingleFrameMode ? 100 : Iterations);
    }
    
    // Print info like final energy, line count, and runtime
    if (!SingleFrameMode || (SingleFrameMode && CurIteration & 100 == 0)){
        PrintStats();
    }
   
    // Draw to the respective output vtkImageDatas
    DrawTargetsToOutputs(spatialImageDataOut, temporalImageDataOut);

    // Show the streamlines via the output vtkPolyData object
    DrawLines(filter, filter->painters[0].get(), polyDataOut);

    if (bShatter){
        m_shardInfo = mFilterStack.ActiveFilter->Shatter();
    }

    // For an animation, Paraview sets the first frame to zero, then skips to the start frame due to ??? 
    // if (time == 0 && !SingleFrameMode){
    //     mFilterStack.ActiveFilter = nullptr;
    //     mFilterStack.PreviousFilter = nullptr;
    // }
    return 1;
}

void vtkTurkBanksStreamlineFilter::Optimize(int maxIterations){
    auto *const filter = mFilterStack.ActiveFilter.get();
    auto *const config = &filter->painters[0]->config;
    const int finalIteration = CurIteration + maxIterations;

    for (; CurIteration < finalIteration; ++CurIteration){
        // Pick a random line and move / lengthen it using random values
        if (MoveLineRandomly(filter, GetRandomLine(filter), config)){
            m_statistics.randomAccepts++;
        } else {
            m_statistics.randomRejects++;
        }

        // Random Join
        if (config->joinDistanceFactor > 0)
        {
            if (TryJoinLines(filter, config)){
                m_statistics.joinAccepts++;
            } else {
                m_statistics.joinRejects++;
            }
        }
        
        // Birth lines
        if (CurIteration % 100 == 1 && BirthThreshold > 0){
            TryBirthLines(filter, config);
        }

        // Update every line's quality
        for (auto &line : filter->mStreamlines){
            oracle->UpdateLineDesire(filter, line);
        }

        // Pick a bad line and do a fitting change
        {
            auto badLine = oracle->GetBadLine(filter);
            if (drand48() < .2){
                // Maybe do a completely random change
                if (MoveLineRandomly(filter, GetRandomLine(filter), config)){
                    m_statistics.oracleAccepts++;
                } else {
                    m_statistics.oracleRejects++;
                }
            } else {
                // Otherwise, recommend a sensible change to maybe move the line and lengthen/shorten one end
                double moveX = 0.0;
                double moveY = 0.0;
                double frontChange = 0.0;
                double backChange = 0.0;
                double fDesire = fabs(badLine->mFrontDesire);
                double bDesire = fabs(badLine->mBackDesire);
                double sDesire = fabs(badLine->mSideDesire) * 4;

                // Move the line sideways if necessary
                if (sDesire > fDesire && sDesire > bDesire){
                    moveX = badLine->mSeedNormal[0] * badLine->mSideDesire * config->deltaMove() * 2*drand48();
                    moveY = badLine->mSeedNormal[1] * badLine->mSideDesire * config->deltaMove() * 2*drand48();
                }

                // Also change one end
                if (fDesire > bDesire){
                    frontChange = copysign(std::max(config->segmentSize, drand48() * config->deltaLen()), badLine->mFrontDesire);
                } else {
                    backChange  = copysign(std::max(config->segmentSize, drand48() * config->deltaLen()), badLine->mBackDesire);
                }

                // Maybe scramble the length change
                if (drand48() < .5){
                    std::swap(frontChange, backChange);
                }

                // Try the change
                auto newLine = filter->ChangeLine(badLine, frontChange, backChange, moveX, moveY);
                if (newLine == nullptr){
                    // Line was fast-deleted
                    m_statistics.oracleAccepts++;
                } else if (newLine != badLine){
                    // Moving the line improved the image
                    oracle->UpdateLineDesire(filter, newLine);
                    m_statistics.oracleAccepts++;
                } else {
                    // Moving the line made the image worse
                    m_statistics.oracleRejects++;
                }
            }
        }

        // To keep the filter responsive in a single-step animation scenario
        if (SingleFrameMode && m_statistics.randomAccepts + m_statistics.joinAccepts + m_statistics.oracleAccepts > 0){
            return;
        }
    }
}

void vtkTurkBanksStreamlineFilter::DrawLines(const LowPassFilter *filter, const FilterPainter *painter, vtkPolyData *polyData){
    vtkNew<vtkPoints> pts;
    vtkNew<vtkDoubleArray> energy;
    vtkNew<vtkDoubleArray> velocities;
    const int streamlineCount = filter->mStreamlines.size();
    polyData->Allocate(streamlineCount);
    //energies->SetNumberOfTuples(streamlineCount);
    energy->SetNumberOfComponents(3);
    velocities->SetNumberOfComponents(3);
    //energy->SetNumberOfTuples(streamlineCount);
    energy->SetName("Front|Back|Side Energy");
    velocities->SetName("Velocity");
    double point[3];
    double velocity[3];
    for (auto& line : filter->mStreamlines){
        vtkNew<vtkIdList> ids;
        for (int i = 0; i < line->mPoints->GetNumberOfPoints(); i++){
            line->mPoints->GetPoint(i, point);
            line->mVelocities->GetTuple(i, velocity);
            point[0] *= VFieldX;
            point[1] *= VFieldY;
            pts->InsertNextPoint(point);
            velocities->InsertNextTuple(velocity);
            ids->InsertNextId(pts->GetNumberOfPoints()-1);
        }
        polyData->InsertNextCell(VTK_POLY_LINE, ids);
        energy->InsertNextTuple3(line->mFrontDesire, line->mBackDesire, line->mSideDesire);
    }
    polyData->SetPoints(pts);
    polyData->GetPointData()->AddArray(velocities);
    polyData->GetCellData()->AddArray(energy);
}

void vtkTurkBanksStreamlineFilter::DrawTargetsToOutputs(vtkImageData* spatial, vtkImageData* temporal)
{
    const auto *currentFilter = mFilterStack.ActiveFilter.get();
    const auto *previousFilter = mFilterStack.PreviousFilter.get();

    const auto *currentSpatialTarget = currentFilter && currentFilter->painters.size() > 0 ? &currentFilter->painters[0]->target : nullptr;
    
    const auto *currentTemporalTarget = currentFilter && currentFilter->painters.size() > 1 ? &currentFilter->painters[1]->target : nullptr;
    const auto *previousTemporalTarget = previousFilter && previousFilter->painters.size() > 1 ? &previousFilter->painters[1]->target : nullptr;
    
    // Fill the current spatial output vtkImageData's cells with the filter values
    {
        vtkNew<vtkPoints> vectors;
        for (double i : currentSpatialTarget->data){
            vectors->InsertNextPoint(0, i, 0);
        }
        spatial->GetPointData()->SetVectors(vectors->GetData());
    }

    // Fill the temporal output vtkImageData's cells with the filter values
    if (currentTemporalTarget) {
        const int pixelCount = currentTemporalTarget->data.size();
        vtkNew<vtkPoints> vectors;
        // vectors->SetNumberOfPoints(pixelCount);
        for (int i = 0; i < pixelCount; ++i){
            // If we have a previous filter, draw that to the red channel
            if (previousTemporalTarget){
                vectors->InsertNextPoint(previousTemporalTarget->data[i], currentTemporalTarget->data[i], 0);
            // Otherwise, just draw the current one
            } else {
                vectors->InsertNextPoint(0, currentTemporalTarget->data[i], 0);
            }
        }
        temporal->GetPointData()->SetVectors(vectors->GetData());
    }
}

// ----------------------------------------------------------------------------
void vtkTurkBanksStreamlineFilter::PrintSelf(std::ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
}

