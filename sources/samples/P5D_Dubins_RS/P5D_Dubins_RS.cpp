#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <helperOC/helperOC.hpp>
#include <helperOC/Legacy/ExtractCostates.hpp>
#include <helperOC/DynSys/P5D_Dubins/P5D_Dubins.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>

int main(int argc, char *argv[])
{
  bool dump_file = false;
  if (argc >= 2) {
    dump_file = (atoi(argv[1]) == 0) ? false : true;
  }
  size_t line_length_of_chunk = 1;
  if (argc >= 3) {
    line_length_of_chunk = atoi(argv[2]);
  }
  size_t model_size = 2;
  if (argc >= 4) {
    model_size = atoi(argv[3]);
  }
  // Time stamps
  FLOAT_TYPE tMax = 15.;
  if (argc >= 5) {
    tMax = static_cast<FLOAT_TYPE>(atof(argv[4]));
  }
  FLOAT_TYPE dt = (FLOAT_TYPE).05;
  if (argc >= 6) {
    dt = static_cast<FLOAT_TYPE>(atof(argv[5]));
  }
  bool useTempFile = false;
  if (argc >= 7) {
    useTempFile = (atoi(argv[6]) == 0) ? false : true;
  }
  bool keepLast = true;
  if (argc >= 8) {
    keepLast = (atoi(argv[7]) == 0) ? true : false;
  }
  bool calculateTTRduringSolving = false;
  if (argc >= 9) {
    calculateTTRduringSolving = (atoi(argv[8]) == 0) ? false : true;
  }
  bool useCuda = false;
  if (argc >= 10) {
    useCuda = (atoi(argv[9]) == 0) ? false : true;
  }
  int num_of_threads = 0;
  if (argc >= 11) {
    num_of_threads = atoi(argv[10]);
  }
  int num_of_gpus = 0;
  if (argc >= 12) {
    num_of_gpus = atoi(argv[11]);
  }
  levelset::DelayedDerivMinMax_Type delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable;
  if (argc >= 13) {
    switch (atoi(argv[12])) {
    default:
    case 0:
      delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable;
      break;
    case 1:
      delayedDerivMinMax = levelset::DelayedDerivMinMax_Always;
      break;
    case 2:
      delayedDerivMinMax = levelset::DelayedDerivMinMax_Adaptive;
      break;
    }
  }
  bool enable_user_defined_dynamics_on_gpu = true;
  if (argc >= 14) {
    enable_user_defined_dynamics_on_gpu = (atoi(argv[13]) == 0) ? false : true;
  }
  // Create grid
  beacls::IntegerVec  Ns;
  std::cout << "Model size option :" << model_size <<std::endl;
  switch (model_size) {
  case 0:
    Ns = beacls::IntegerVec{ 11, 11, 11, 11, 11};
    break;
  case 1:
    Ns = beacls::IntegerVec{ 31, 31, 21, 21, 21};
  case 2:
    break;
    Ns = beacls::IntegerVec{ 51, 51, 31, 31, 31};
  case 3:
    break;
    Ns = beacls::IntegerVec{ 151, 151, 101, 101, 101};
    break;
  case 4:
    Ns = beacls::IntegerVec{ 501, 501, 301, 301, 301 };
    break;
  default:
    Ns = beacls::IntegerVec{ 1501, 1501, 1001, 1001, 1001 };
    break;
  }
  std::cout << "Grid dimensions : ["
    << Ns[0] << ", " << Ns[1] << ", "
    << Ns[2] << ", " << Ns[3] << ", " << Ns[4] << "]" << std::endl;

  levelset::HJI_Grid* g = helperOC::createGrid( // Grid limits (HARD-CODED)
    beacls::FloatVec{
        (FLOAT_TYPE)(-21), (FLOAT_TYPE)(-18), (FLOAT_TYPE)(-M_PI),
         (FLOAT_TYPE)1, (FLOAT_TYPE)(-M_PI)}, 
    beacls::FloatVec{
        (FLOAT_TYPE)15, (FLOAT_TYPE)18, (FLOAT_TYPE)(M_PI),
         (FLOAT_TYPE)5, (FLOAT_TYPE)(M_PI)}, 
    Ns, beacls::IntegerVec{2}); // Only periodic dimension is theta_rel

  beacls::FloatVec tau = generateArithmeticSequence<FLOAT_TYPE>(0., dt, tMax);
  // Generate P5D_Dubins object (take default parameter values in .hpp)
  helperOC::P5D_Dubins* p5D_Dubins = new helperOC::P5D_Dubins(
    beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0,
      (FLOAT_TYPE)0, (FLOAT_TYPE)0});
  

  // Dynamical system parameters
  helperOC::DynSysSchemeData* schemeData = new helperOC::DynSysSchemeData;
  schemeData->set_grid(g);
  schemeData->dynSys = p5D_Dubins;
  schemeData->accuracy = helperOC::ApproximationAccuracy_veryHigh;
  schemeData->uMode = helperOC::DynSys_UMode_Max;
  schemeData->dMode = helperOC::DynSys_DMode_Min;

  // Target set and visualization
  std::vector<beacls::FloatVec> targets(1);
  const size_t numel = g->get_numel();
  const size_t num_dim = g->get_num_of_dimensions();
  
  targets[0].assign(numel, 0.);

  for (size_t dim = 0; dim < num_dim; ++dim) {
    const beacls::FloatVec &xs = g->get_xs(dim);

    if (dim == 0 || dim == 1) { 
      // target(1) = - x0^2 - x1^2
      std::transform(xs.cbegin(), xs.cend(), targets[0].begin(), 
          targets[0].begin(), [](const auto &xs_i, const auto &tar_i) {
          return tar_i - std::pow(xs_i, 2); });
    }
  }
  
  helperOC::HJIPDE_extraArgs extraArgs;
  helperOC::HJIPDE_extraOuts extraOuts;
  extraArgs.visualize = false;
  extraArgs.execParameters.line_length_of_chunk = line_length_of_chunk;
  extraArgs.execParameters.calcTTR = calculateTTRduringSolving;
  extraArgs.keepLast = keepLast;
  extraArgs.execParameters.useCuda = useCuda;
  extraArgs.execParameters.num_of_gpus = num_of_gpus;
  extraArgs.execParameters.num_of_threads = num_of_threads;
  extraArgs.execParameters.delayedDerivMinMax = delayedDerivMinMax;
  extraArgs.execParameters.enable_user_defined_dynamics_on_gpu = enable_user_defined_dynamics_on_gpu;

  helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

  beacls::FloatVec stoptau;
  std::vector<beacls::FloatVec> data;
  std::cout << "Begin HJI PDE Solve." <<std::endl;

  
  hjipde->solve(data, stoptau, extraOuts, targets, tau, schemeData, 
    helperOC::HJIPDE::MinWithType_None, extraArgs);

  // std::vector<beacls::FloatVec> P, derivL, derivR;

  // Compute gradient
  // if (!TTR.empty()) {
  //   helperOC::ExtractCostates* extractCostates = new helperOC::ExtractCostates(schemeData->accuracy);
  //   extractCostates->operator()(P, derivL, derivR, g, TTR, TTR.size(), false, extraArgs.execParameters);
  //   if (extractCostates) delete extractCostates;
  // }

  // if (dump_file) {
  //   std::string P5D_Dubins_RS_RS_filename("P5D_Dubins_RS_RS.mat");
  //   beacls::MatFStream* P5D_Dubins_RS_RS_fs = beacls::openMatFStream(P5D_Dubins_RS_RS_filename, beacls::MatOpenMode_Write);
  //   g->save_grid(std::string("g"), P5D_Dubins_RS_RS_fs);
  //   if (!data.empty()) save_vector_of_vectors(data, std::string("data"), Ns, false, P5D_Dubins_RS_RS_fs);
  //   if (!tau.empty()) save_vector(tau, std::string("tau"), beacls::IntegerVec(), false, P5D_Dubins_RS_RS_fs);
  //   if (!P.empty()) save_vector_of_vectors(P, std::string("P"), Ns, false, P5D_Dubins_RS_RS_fs);
  //   beacls::closeMatFStream(P5D_Dubins_RS_RS_fs);

  //   std::string P5D_Dubins_RS_RS_smaller_filename("P5D_Dubins_RS_RS_smaller.mat");
  //   beacls::MatFStream* P5D_Dubins_RS_RS_smaller_fs = beacls::openMatFStream(P5D_Dubins_RS_RS_smaller_filename, beacls::MatOpenMode_Write);
  //   g->save_grid(std::string("g"), P5D_Dubins_RS_RS_smaller_fs);
  //   if (!P.empty()) save_vector_of_vectors(P, std::string("P"), Ns, false, P5D_Dubins_RS_RS_smaller_fs);
  //   if (!tau.empty()) save_vector(tau, std::string("tau"), beacls::IntegerVec(), false, P5D_Dubins_RS_RS_smaller_fs);
  //   beacls::closeMatFStream(P5D_Dubins_RS_RS_smaller_fs);
  // }

  if (hjipde) delete hjipde;
  if (schemeData) delete schemeData;
  if (p5D_Dubins) delete p5D_Dubins;
  if (g) delete g;
  return 0;
}