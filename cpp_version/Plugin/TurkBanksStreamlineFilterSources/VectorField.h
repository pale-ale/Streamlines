#pragma once

#include <vtkImageData.h>

class Streamline;
class FilterConfig;

class VectorField{
  public:
    VectorField(vtkImageData *imageData, FilterConfig *defaultConf):
    imageData{imageData},
    defaultConf{defaultConf},
    xExtent(imageData->GetExtent()[1] - imageData->GetExtent()[0]),
    yExtent(imageData->GetExtent()[3] - imageData->GetExtent()[2]),
    xSize(imageData->GetSpacing()[0] * xExtent),
    ySize(imageData->GetSpacing()[1] * yExtent),
    aspectRatio((double)yExtent/xExtent)
    {}

    void Get(const double &x, const double &y, double &outX, double& outY) const;
    void SetStreamlinePoints(Streamline *streamline) const;
    std::tuple<vtkSmartPointer<vtkPoints>, vtkSmartPointer<vtkDataArray>> _GetStreamlinePointsSingle(const Streamline *streamline, bool forward) const;

    // How many steps we can take in every direction (=nPoints - 1)
    const int xExtent, yExtent;
    // Absolute size of the field w.r.t. spacing
    const double xSize, ySize;
    const double aspectRatio;

  private:
    vtkImageData* imageData = nullptr;
    FilterConfig* defaultConf = nullptr;
};
