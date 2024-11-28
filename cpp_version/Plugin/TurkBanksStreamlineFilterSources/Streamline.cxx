#include "Streamline.h"
#include "VectorField.h"

Streamline::Streamline(const double seed[3], const VectorField* vectorField, double length):
    Streamline(seed, vectorField,  length/2, length/2)
{}

Streamline::Streamline(const double seed[3], const VectorField* vectorField, double frontLength, double backLength):
    mSeed{seed[0], seed[1], seed[2]}, mVectorField{vectorField}, mFrontLength{frontLength}, mBackLength{backLength}
{
    vectorField->SetStreamlinePoints(this);
}
