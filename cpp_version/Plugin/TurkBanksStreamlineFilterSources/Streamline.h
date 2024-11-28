#pragma once

#include <functional>
#include <tuple>
#include <vector>
#include <array>
#include <vtkPoints.h>
#include <vtkNew.h>
#include <vtkDoubleArray.h>

class VectorField;

class Streamline{
    public:
    
    Streamline(const double seed[3], const VectorField* vectorField, double length);
    Streamline(const double seed[3], const VectorField* vectorField, double frontLength, double backLength);
    Streamline(const Streamline &sl):Streamline(sl.mSeed, sl.mVectorField, sl.mFrontLength, sl.mBackLength){};
    void GetTail(double &x, double &y){
        int ptCount = mPoints->GetNumberOfPoints();
        double *point;
        if (ptCount > 0){
            point = mPoints->GetPoint(0);
            x = point[0];
            y = point[1];
        }
    }

    void GetHead(double &x, double &y){
        int ptCount = mPoints->GetNumberOfPoints();
        double *point;
        if (ptCount > 0){
            point = mPoints->GetPoint(ptCount-1);
            x = point[0];
            y = point[1];
        }
    }

    friend ostream& operator<<(ostream& os, const Streamline &streamline)
    {
        os << "Line Seed: " << streamline.mSeed[0] << " " << streamline.mSeed[1];
        return os;
    }

    double mBackLength;
    double mFrontLength ;
    double mFrontDesire = 0.0;
    double mBackDesire  = 0.0;
    double mSideDesire  = 0.0;
    double mTotalDesire = 0.0;
    const VectorField* mVectorField;

    double mSeed[3];
    double mSeedNormal[2];

    vtkNew<vtkPoints> mPoints;
    vtkNew<vtkDoubleArray> mVelocities;
};
