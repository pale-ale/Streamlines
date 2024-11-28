#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>

#include "VectorField.h"
#include "Streamline.h"
#include "FilterConfig.h"

/* 
Get the velocity at point (x, y) and write it to (outX, outY). 
Bilinear interpolation is used to determine the value form the samples.
*/
void VectorField::Get(const double &x, const double &y, double &outX, double& outY) const {
    // Get the extents
    int extents[6];
    imageData->GetExtent(extents);

    // Get the coordinates in index space (i,j,k)
    double indexcoords[3];
    imageData->TransformPhysicalPointToContinuousIndex(x * xSize, y * ySize, 0, indexcoords);

    // Clamp the i and j values to next lower integer, k is discarded
    int i = std::clamp<int>((int)indexcoords[0], 0, xExtent-1);
    int j = std::clamp<int>((int)indexcoords[1], 0, yExtent-1);

    // Get the remainders in range (0,1)
    float tx = indexcoords[0] - i;
    float ty = indexcoords[1] - j;
    
    // Obtain the four sample points
    int coord[3] = {i , j, extents[4]};
    double vel00[3];
    double vel01[3];
    double vel10[3];
    double vel11[3];
    auto vectors = imageData->GetPointData()->GetVectors();
    vectors->GetTuple(imageData->GetTupleIndex(vectors, coord), vel00);
    ++coord[0];
    vectors->GetTuple(imageData->GetTupleIndex(vectors, coord), vel10);
    ++coord[1];
    vectors->GetTuple(imageData->GetTupleIndex(vectors, coord), vel11);
    --coord[0];
    vectors->GetTuple(imageData->GetTupleIndex(vectors, coord), vel01);

    // Perform the bilinear interpolation and return the result
    double vel0[2] = {vel00[0] + ty * (vel01[0] - vel00[0]), vel00[1] + ty * (vel01[1] - vel00[1])};
    double vel1[2] = {vel10[0] + ty * (vel11[0] - vel10[0]), vel10[1] + ty * (vel11[1] - vel10[1])};
    outX = vel0[0] + tx * (vel1[0] - vel0[0]);
    outY = vel0[1] + tx * (vel1[1] - vel0[1]);
}

/*
Generate the points and velocities for a streamline from its seed and forward/backward lengths.
The streamline is modified in place and the points/velocities are saved inside it.
*/
void VectorField::SetStreamlinePoints(Streamline *streamline) const {
    // Clear residual points and velocities of the streamline
    auto &points = streamline->mPoints;
    auto &velocities = streamline->mVelocities;
    points->SetNumberOfPoints(0);
    velocities->SetNumberOfTuples(0);
    velocities->SetNumberOfComponents(3);

    // Integrate in backward direction first
    {
        auto [backPoints, backVelocities] = _GetStreamlinePointsSingle(streamline, false);
        // Add the points in reverse order, leaving the seed for the forward iteration
        for (int i = backPoints->GetNumberOfPoints() - 1; i > 0; --i){
            points->InsertNextPoint(backPoints->GetPoint(i));
            velocities->InsertNextTuple(backVelocities->GetTuple3(i-1));
        }
    }

    // Integrate in forward direction
    {
        auto [frontPoints, frontVelocities] = _GetStreamlinePointsSingle(streamline, true);
        if (frontPoints){
            points->InsertPoints(points->GetNumberOfPoints(), frontPoints->GetNumberOfPoints(), 0, frontPoints);
            velocities->InsertTuples(velocities->GetNumberOfTuples(), frontVelocities->GetNumberOfTuples(), 0, frontVelocities);
        }
        // Set the seed velocity
        if (frontVelocities->GetNumberOfValues() > 0) {
            streamline->mSeedNormal[0] = - frontVelocities->GetTuple3(0)[1];
            streamline->mSeedNormal[1] =   frontVelocities->GetTuple3(0)[0];
        }
    }
}

/*
Gaussian integration routine
*/
std::tuple<vtkSmartPointer<vtkPoints>, vtkSmartPointer<vtkDataArray>> VectorField::_GetStreamlinePointsSingle(
    const Streamline *streamline, bool forward
) const {
    double ssize = defaultConf->segmentSize;
    vtkSmartPointer<vtkPoints>       pts = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkDoubleArray> vels = vtkSmartPointer<vtkDoubleArray>::New();
    pts->SetNumberOfPoints(0);
    vels->SetNumberOfTuples(0);
    vels->SetNumberOfComponents(3);
    double x = streamline->mSeed[0];
    double y = streamline->mSeed[1];
    double vx, vy, norm;
    int maxPoints = (forward ? streamline->mFrontLength : streamline->mBackLength) / ssize;
    int points = 0;
    while (x > 0 && x < 1 && y > 0 && y < 1 && points < maxPoints){
        Get(x, y, vx, vy);
        norm = sqrt(vx * vx + vy * vy);
        if (norm <= 1e-8){
            return {pts, vels};
        }
        pts->InsertNextPoint(x, y, 0);
        vels->InsertNextTuple3(vx, vy, 0);
        vx *= (ssize / norm);
        vy *= (ssize / norm) / aspectRatio;
        if (forward){
            x += vx;
            y += vy;
        } else {
            x -= vx;
            y -= vy;
        }
        ++points;
    }
    return {pts, vels};
}
