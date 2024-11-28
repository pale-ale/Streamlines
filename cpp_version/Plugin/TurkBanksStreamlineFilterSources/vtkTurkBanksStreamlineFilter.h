#pragma once

#include "TurkBanksStreamlineFilterSourcesModule.h"
#include "vtkPolyDataAlgorithm.h"

#include <array>
#include <chrono>
#include <vector>
#include <tuple>

#include "FilterConfig.h"
#include "FilterPainter.h"
#include "FilterStack.h"

class vtkPoints;
class vtkCellArray;
class vtkImageData;
class Oracle;
class VectorField;


struct Statistics{
    std::chrono::system_clock::time_point startTime;
    int randomAccepts = 0;
    int randomRejects = 0;
    int joinAccepts = 0;
    int joinRejects = 0;
    int oracleAccepts = 0;
    int oracleRejects = 0;

    void reset(){
        startTime = std::chrono::system_clock::now();
        randomAccepts = 0;
        randomRejects = 0;
        joinAccepts = 0;
        joinRejects = 0;
        oracleAccepts = 0;
        oracleRejects = 0;
    }
};

class TURKBANKSSTREAMLINEFILTERSOURCES_EXPORT vtkTurkBanksStreamlineFilter : public vtkPolyDataAlgorithm
{
public:
  typedef std::array<double,3> Point;
  static vtkTurkBanksStreamlineFilter* New();
  vtkTypeMacro(vtkTurkBanksStreamlineFilter, vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  ///@{
  /**
   * Set/Get the maximum number of iteration before the live source stops.
   */
  vtkSetMacro(Iterations, int);
  vtkGetMacro(Iterations, int);
  
  vtkSetMacro(JoinDistanceFactor, double);
  vtkGetMacro(JoinDistanceFactor, double);
  
  vtkSetMacro(LengthenFactor, double);
  vtkGetMacro(LengthenFactor, double);
  
  vtkSetMacro(Separation, double);
  vtkGetMacro(Separation, double);
  
  vtkSetMacro(BirthThreshold, double);
  vtkGetMacro(BirthThreshold, double);

  vtkSetMacro(IntegrationStepSize, double);
  vtkGetMacro(IntegrationStepSize, double);
  
  vtkSetMacro(CoherenceWeight, double);
  vtkGetMacro(CoherenceWeight, double);
  
  vtkSetMacro(SingleFrameMode, bool);
  vtkGetMacro(SingleFrameMode, bool);
  
  vtkSetMacro(ResolutionFactor, double);
  vtkGetMacro(ResolutionFactor, double);
  
  vtkSetMacro(bShatter, bool);
  vtkGetMacro(bShatter, bool);
  
  vtkSetMacro(TemporalFilterRadius, double);
  vtkGetMacro(TemporalFilterRadius, double);

  void WriteTemporalData(){
    if (!mFilterStack.ActiveFilter){
        cout << "No active filter. Cannot write to temp." << endl;
        return;
    }
    mFilterStack.PushFilter(mFilterStack.CopyFilter(mFilterStack.ActiveFilter));
    Modified();
  }
  
  void ClearData(){
    mFilterStack.ActiveFilter = nullptr;
    mFilterStack.PreviousFilter = nullptr;
    Modified();
  }

  void Shatter(){
    if (mFilterStack.ActiveFilter){
        m_shardInfo = mFilterStack.ActiveFilter->Shatter();
        Modified();
    } 
  }
  ///@}

  /**
   * Check if the RequestUpdateExtent/RequestData need to be called again to refresh the output.
   * Return true if an update is needed.
   *
   * This method is required for Live Source.
   */
  bool GetNeedsUpdate();

protected:
  vtkTurkBanksStreamlineFilter();
  ~vtkTurkBanksStreamlineFilter() override = default;

  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
  int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
  int RequestUpdateExtent(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

  int FillInputPortInformation(int port, vtkInformation* info) override;
  int FillOutputPortInformation(int port, vtkInformation* info) override;


private:
  int Iterations = 2000;
  double JoinDistanceFactor = 1.0;
  double LengthenFactor = 1.0;
  double BirthThreshold = 0.02;
  double Separation = 0.04;
  double IntegrationStepSize = 0.005;
  double CoherenceWeight = 0.5;
  double ResolutionFactor = 1.0;
  double TemporalFilterRadius = 0.7;
  double LastUpdateMTime = -1;
  bool SingleFrameMode = true;
  bool bShatter = true;
  bool NeedsRedraw = false;

  // Needed for image generation
  double VFieldX = 0.0;
  double VFieldY = 0.0;
  int CurIteration = 0;

  vtkNew<vtkPoints> mPoints;
  vtkNew<vtkCellArray> mCells;
  FilterStack mFilterStack;
  Oracle *oracle;
  VectorField* vectorField;
  vtkImageData* imgData;

  vtkTurkBanksStreamlineFilter(const vtkTurkBanksStreamlineFilter&) = delete;
  void operator=(const vtkTurkBanksStreamlineFilter&) = delete;

  void GenerateInitialStreamlines(int linesX, int linesY);
  void GenerateInitialStreamlines(std::vector<std::tuple<std::array<double, 3>, double, double >>);
  void DrawLines(const LowPassFilter *filter, const FilterPainter *painter, vtkPolyData *polyData);
  void DrawTargetsToOutputs(vtkImageData* spatial, vtkImageData* temporal);
  std::shared_ptr<Streamline> GetRandomLine(LowPassFilter* filter){
    int randomLineIdx = drand48() * filter->mStreamlines.size();
    return filter->mStreamlines[randomLineIdx];
  }
  bool MoveLineRandomly(LowPassFilter* filter, std::shared_ptr<Streamline> streamline, const FilterConfig* config);
  bool TryJoinLines(LowPassFilter* filter, const FilterConfig* config);
  void TryBirthLines(LowPassFilter* filter, const FilterConfig* config);
  void Optimize(int maxIterations);
  void PrintStats(){
    auto runtimeMSecs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_statistics.startTime).count();
    cout <<
         "Energy: " << mFilterStack.ActiveFilter->GetTotalEnergy()    <<
        "\nLines: " << mFilterStack.ActiveFilter->mStreamlines.size() <<
        "\nElapsed Time (seconds): "  << runtimeMSecs / 1000.0        <<
        "\nRandom Accept vs Reject: " << m_statistics.randomAccepts   << "|" << m_statistics.randomRejects << 
        "\n  Join Accept vs reject: " << m_statistics.joinAccepts     << "|" << m_statistics.joinRejects   << 
        "\nOracle Accept vs Reject: " << m_statistics.oracleAccepts   << "|" << m_statistics.oracleRejects <<
    endl;
  }

private:
  FilterConfig m_spatialConf;
  FilterConfig m_temporalConf;
  std::vector<std::tuple<std::array<double, 3>, double, double >> m_shardInfo;
  Statistics m_statistics;
};
