/*******************************************************************************
 Copyright 2006-2012 Lukas Käll <lukas.kall@scilifelab.se>

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 *******************************************************************************/
// #include <boost/filesystem.hpp>
#include "CrossValidation.h"

// number of folds for cross validation
const unsigned int CrossValidation::numFolds_ = 3u;
#ifdef _OPENMP
#include <algorithm>
#include <omp.h>
const unsigned int CrossValidation::numAlgInObjects_ = CrossValidation::numFolds_;
#else
const unsigned int CrossValidation::numAlgInObjects_ = 1u;
#endif
// checks cross validation convergence in case of quickValidation_
const double CrossValidation::requiredIncreaseOver2Iterations_ = 0.01; 

CrossValidation::CrossValidation(bool quickValidation, 
  bool reportPerformanceEachIteration, double testFdr, double selectionFdr, 
  double initialSelectionFdr, double selectedCpos, double selectedCneg, int niter, bool usePi0,
  int nestedXvalBins, bool trainBestPositive, unsigned int numThreads, bool skipNormalizeScores, std::string psmInfluencerDIR) :
    quickValidation_(quickValidation), usePi0_(usePi0),
    reportPerformanceEachIteration_(reportPerformanceEachIteration), 
    testFdr_(testFdr), selectionFdr_(selectionFdr), initialSelectionFdr_(initialSelectionFdr),
    selectedCpos_(selectedCpos), selectedCneg_(selectedCneg), niter_(niter),
    nestedXvalBins_(nestedXvalBins), trainBestPositive_(trainBestPositive),
  numThreads_(numThreads), skipNormalizeScores_(skipNormalizeScores), psmInfluencerDIR_(psmInfluencerDIR){}

CrossValidation::~CrossValidation() { 
  for (unsigned int set = 0; set < numFolds_ * nestedXvalBins_; ++set) {
    if (svmInputs_[set]) {
      delete svmInputs_[set];
    }
    svmInputs_[set] = NULL;
  }
}

/** 
 * Sets up the SVM classifier: 
 * - divide dataset into training and test sets for each fold
 * - set parameters (fdr, soft margin)
 * @param w_ vector of SVM weights
 * @return number of positives for initial setup
 */
int CrossValidation::preIterationSetup(Scores& fullset, SanityCheck* pCheck, 
    Normalizer* pNorm, FeatureMemoryPool& featurePool) {
  assert(nestedXvalBins_ >= 1u);
  
  // initialize weights vector for all folds
  w_ = vector<vector<double> >(numFolds_, 
           vector<double> (FeatureNames::getNumFeatures() + 1));

  // One input set, to be reused multiple times
  for (unsigned int set = 0; set < numFolds_ * nestedXvalBins_; ++set) {
    svmInputs_.push_back(new AlgIn(fullset.size(), FeatureNames::getNumFeatures() + 1));
    assert( svmInputs_.back() );
  }
  
  trainScores_.resize(numFolds_, Scores(usePi0_));
  testScores_.resize(numFolds_, Scores(usePi0_));
  
  fullset.createXvalSetsBySpectrum(trainScores_, testScores_, numFolds_, featurePool);
  
  if (selectionFdr_ <= 0.0) {
    selectionFdr_ = testFdr_;
  }
  if (selectedCpos_ > 0) {
    candidatesCpos_.push_back(selectedCpos_);
  } else {
    candidatesCpos_.push_back(10);
    candidatesCpos_.push_back(1);
    candidatesCpos_.push_back(0.1);
    if (VERB > 0) {
      cerr << "Selecting Cpos by cross-validation." << endl;
    }
  }
  if (selectedCpos_ > 0 && selectedCneg_ > 0) {
    candidatesCfrac_.push_back(selectedCneg_ / selectedCpos_);
  } else {
    candidatesCfrac_.push_back(1.0 * fullset.getTargetDecoySizeRatio());
    candidatesCfrac_.push_back(3.0 * fullset.getTargetDecoySizeRatio());
    candidatesCfrac_.push_back(10.0 * fullset.getTargetDecoySizeRatio());
    if (VERB > 0) {
      cerr << "Selecting Cneg by cross-validation." << endl;
    }
  }
  int numPositive = pCheck->getInitDirection(testScores_, trainScores_, pNorm, w_, 
                                         testFdr_, initialSelectionFdr_);

  // Write header for target/decoy train set size file
  writeTargetDecoyTrainSizesHeader();
  // Write out test sets
  writeTestSets();
  // Write out normalizers
  writeNormalizers(pNorm);

  // Form cpos, cneg pairs per nested CV fold
  candidateCposCfrac cpCnFold;
  for (int set = 0; set < numFolds_; ++set) {
    for (int nestedSet = 0; nestedSet < nestedXvalBins_; nestedSet++) {
      if(!quickValidation_) {
        std::vector<double>::const_iterator itCpos = candidatesCpos_.begin();
        for ( ; itCpos != candidatesCpos_.end(); ++itCpos) {
          double cpos = *itCpos;
          std::vector<double>::const_iterator itCfrac = candidatesCfrac_.begin();
          for ( ; itCfrac != candidatesCfrac_.end(); ++itCfrac) {
            double cfrac = *itCfrac;
            cpCnFold.cpos = cpos;
            cpCnFold.cfrac = cfrac;
            cpCnFold.set = set;
            cpCnFold.nestedSet = nestedSet;
            cpCnFold.tp = 0;
            for (int i = FeatureNames::getNumFeatures() + 1; i--;) {
              cpCnFold.ww.push_back(0);
            }     
            classWeightsPerFold_.push_back(cpCnFold);
          }
        }
      } else {
        cpCnFold.cpos = 1;
        cpCnFold.cfrac = 1;
        cpCnFold.set = set;
        cpCnFold.nestedSet = nestedSet;
        cpCnFold.tp = 0;
        for (int i = FeatureNames::getNumFeatures() + 1; i--;) {
          cpCnFold.ww.push_back(0);
        }         
        classWeightsPerFold_.push_back(cpCnFold);
      }
    }
  }
  
  if (DataSet::getCalcDoc()) {
    for (int set = 0; set < numFolds_; ++set) {
      trainScores_[set].calcScores(w_[set], selectionFdr_);
    }
  #pragma omp parallel for schedule(dynamic, 1)
    for (int set = 0; set < numFolds_; ++set) {
      trainScores_[set].recalculateDescriptionOfCorrect(selectionFdr_);
      testScores_[set].getDOC().copyDOCparameters(trainScores_[set].getDOC());
      testScores_[set].setDOCFeatures(pNorm);
    }
  }
  
  return numPositive;
}

/** 
 * Train the SVM using several cross validation iterations
 * @param pNorm Normalization object
 */
void CrossValidation::train(Normalizer* pNorm) {

  if (VERB > 0) {
    cerr << "---Training with Cpos";
    if (selectedCpos_ > 0) {
      cerr << "=" << selectedCpos_;
    } else {
      cerr << " selected by cross validation";
    }
    cerr << ", Cneg";
    if (selectedCneg_ > 0) {
      cerr << "=" << selectedCneg_;
    } else {
      cerr << " selected by cross validation";
    }
    cerr << ", initial_fdr=" << initialSelectionFdr_;
    cerr << ", fdr=" << selectionFdr_ << endl;
  }
  // iterate
  int foundPositivesOldOld = 0, foundPositivesOld = 0, foundPositives = 0; 
  for (unsigned int i = 0; i < niter_; i++) {
    if (VERB > 1) {
      cerr << "Iteration " << i + 1 << ":\t";
    }
    
    bool updateDOC = true;
    double selectionFdr = selectionFdr_;
    if (i == 0u) {
      selectionFdr = initialSelectionFdr_;
    }
    foundPositives = doStep(updateDOC, pNorm, selectionFdr, i);
    
    if (reportPerformanceEachIteration_) {
      int foundTestPositives = 0;
      for (size_t set = 0; set < numFolds_; ++set) {
        foundTestPositives += testScores_[set].calcScores(w_[set], testFdr_);
      }
      if (VERB > 1) {
        std::cerr << "Found " << foundTestPositives << " test set PSMs with q<" 
             << testFdr_ << endl;
      }
    } else if (VERB > 1) {
      cerr << "Estimated " << foundPositives << " PSMs with q<" << testFdr_ << endl;
    }
    
    if (VERB > 2) {
      printAllWeightsColumns(cerr);
    }
    if (VERB > 3) {
      printAllRawWeightsColumns(cerr, pNorm);
    }
    if (foundPositives > 0 && foundPositivesOldOld > 0 && quickValidation_ &&
           (static_cast<double>(foundPositives - foundPositivesOldOld) <= 
           foundPositivesOldOld * requiredIncreaseOver2Iterations_)) {
      if (VERB > 0) {
        std::cerr << "Performance increase over the last two iterations " <<
            "indicate that the algorithm has converged\n" <<
            "(" << foundPositives << " vs " << foundPositivesOldOld << ")" << 
            std::endl;
      }
      break;
    }
    foundPositivesOldOld = foundPositivesOld;    
    foundPositivesOld = foundPositives;
  }
  if (VERB == 2) {
    printAllWeightsColumns(cerr);
  }
  if (VERB == 3) {
    printAllRawWeightsColumns(cerr, pNorm);
  }
  foundPositives = 0;
  for (size_t set = 0; set < numFolds_; ++set) {
    foundPositives += testScores_[set].calcScores(w_[set], testFdr_);
  }
  if (VERB > 0) {
    std::cerr << "Found " << foundPositives << 
                 " test set PSMs with q<" << testFdr_ << "." << std::endl;
  }  
}

/** 
 * Executes a cross validation step
 * @param w_ list of the bins' normal vectors (in linear algebra sense) of the 
 *        hyperplane from SVM
 * @param updateDOC boolean deciding to recalculate retention features 
 *        @see DescriptionOfCorrect
 * @return Estimation of number of true positives
 */
int CrossValidation::doStep(bool updateDOC, Normalizer* pNorm, double selectionFdr, int trainingIter) {
  // Setup
  struct options* pOptions = new options;
  pOptions->lambda = 1.0;
  pOptions->lambda_u = 1.0;
  pOptions->epsilon = EPSILON;
  pOptions->cgitermax = CGITERMAX;
  pOptions->mfnitermax = MFNITERMAX;
  int estTruePos = 0;
  
  // for determining an appropriate positive training set, the decoys+1 in the 
  // FDR estimates is too restrictive for small datasets
  bool skipDecoysPlusOne = true; 
  for (int set = 0; set < numFolds_; ++set) {
    trainScores_[set].calcScores(w_[set], selectionFdr, skipDecoysPlusOne);
  }
  
  if (DataSet::getCalcDoc() && updateDOC) {
  #pragma omp parallel for schedule(dynamic, 1)
    for (int set = 0; set < numFolds_; ++set) {
      trainScores_[set].recalculateDescriptionOfCorrect(selectionFdr);
      //trainScores_[set].setDOCFeatures(pNorm); // this overwrites features of overlapping training folds...
      testScores_[set].getDOC().copyDOCparameters(trainScores_[set].getDOC());
      testScores_[set].setDOCFeatures(pNorm);
    }
  }

  // Below implements the series of speedups detailed in the following:
  // ////////////////////////////////
  // Speeding Up Percolator
  // John T. Halloran, Hantian Zhang, Kaan Kara, Cédric Renggli, Matthew The, Ce Zhang, David M. Rocke, Lukas Käll, and William Stafford Noble
  //   Journal of Proteome Research 2019 18 (9), 3353-3359
  // ////////////////////////////////
  // Note that the implementation further improves on the speedups in the paper by: 
  //   -implementing a single threadpool using OMP for SVM training per each cpos,cneg pair per nested CV fold
  //   -has a much smaller memory footprint by fixing memory leaks in L2_SVM_MFN and more efficient validation of the learned SVM parameters
  // ////

  // Create SVM input data for parallelization
   std::vector<AlgIn*> svmInputsVec;
   std::vector< std::vector< Scores > > nestedTestScoresVec;
   for (int set = 0; set < numFolds_; ++set) {
     std::vector<Scores> nestedTrainScores(nestedXvalBins_, usePi0_), nestedTestScores(nestedXvalBins_, usePi0_);
     if (nestedXvalBins_ > 1) {
       FeatureMemoryPool featurePool;
       trainScores_[set].createXvalSetsBySpectrum(nestedTrainScores, nestedTestScores, nestedXvalBins_, featurePool);
     } else {
       // sub-optimal cross validation
       nestedTrainScores[0] = trainScores_[set];
       nestedTestScores[0] = trainScores_[set];
     }
     nestedTestScoresVec.push_back(nestedTestScores);
     // Set SVM input data for L2-SVM-MFN
     for (int nestedFold = 0; nestedFold < nestedXvalBins_; ++nestedFold)
       {
         AlgIn* svmInput = svmInputs_[set * nestedXvalBins_ + nestedFold];
         if ((VERB > 2) && (nestedFold==0)){
           cerr << "Split " << set + 1 << ": Training with " 
                << svmInput->positives << " positives and "
                << svmInput->negatives << " negatives" << std::endl;
         }
         nestedTrainScores[nestedFold].generateNegativeTrainingSet(*svmInput, 1.0);
         nestedTrainScores[nestedFold].generatePositiveTrainingSet(*svmInput, selectionFdr, 1.0, trainBestPositive_);
	 // Write training set sizes
	 writeTargetDecoyTrainSizes(trainingIter, set * nestedXvalBins_ + nestedFold, svmInput->positives, svmInput->negatives, svmInput->allPositives);

         svmInputsVec.push_back(svmInput);
       }
   }

#pragma omp parallel for schedule(dynamic, 1) ordered 
   for (int pairIdx = 0; pairIdx < classWeightsPerFold_.size(); pairIdx++){
    candidateCposCfrac* cpCnFold = &classWeightsPerFold_[pairIdx];
    AlgIn* svmInput = svmInputsVec[cpCnFold->set * nestedXvalBins_  + cpCnFold->nestedSet];
    trainCpCnPair(*cpCnFold, pOptions,svmInput);
  }

  estTruePos = mergeCpCnPairs(selectionFdr, pOptions, nestedTestScoresVec, candidatesCpos_, 
                              candidatesCfrac_, trainingIter);
  delete pOptions;
  return estTruePos;
}

/** 
 * Train SVM over a single (cpos, cneg) pair
 * @param cpCnFold contains cpos, cneg pair and SVM learned weights
 * @param pOptions options for the SVM algorithm
 * @param svmInput training data for this particular nested CV fold
*/
void CrossValidation::trainCpCnPair(candidateCposCfrac& cpCnFold,
      options * pOptions, AlgIn* svmInput) {

  struct vector_double* pWeights = new vector_double;
  pWeights->d = FeatureNames::getNumFeatures() + 1;
  pWeights->vec = new double[pWeights->d];

  unsigned int set = cpCnFold.set;
  unsigned int nestedFold = cpCnFold.nestedSet;
  double cpos = cpCnFold.cpos;
  double cfrac = cpCnFold.cfrac;
    
  // Create storage vector for SVM algorithm
  struct vector_double* Outputs = new vector_double;
  size_t numInputs = svmInput->positives + svmInput->negatives;
  Outputs->vec = new double[numInputs];
  Outputs->d = numInputs;

  if (VERB > 3) cerr << "- cross-validation with Cpos=" << cpos
                     << ", Cneg=" << cfrac * cpos << endl;
  int tp = 0;
  for (int ix = 0; ix < pWeights->d; ix++) {
    pWeights->vec[ix] = 0;
  }
  for (int ix = 0; ix < Outputs->d; ix++) {
    Outputs->vec[ix] = 0;
  }
        
  // Call SVM algorithm (see ssl.cpp)
  L2_SVM_MFN(*svmInput, pOptions, pWeights, Outputs, cpos, cfrac * cpos);
        
  for (int i = FeatureNames::getNumFeatures() + 1; i--;) {
    cpCnFold.ww[i] = pWeights->vec[i];
  }
  // Update container for supportVector informatio
  cpCnFold.supportVectors.resize(numInputs);
  // Record support vectors
  for (int ix = 0; ix < numInputs; ix++) {
    cpCnFold.supportVectors[ix] = svmInput->supportVectors[ix];
  }

  delete[] Outputs->vec;
  delete Outputs;
  delete[] pWeights->vec;
  delete pWeights;
}

/* Write out psm influencer info for current CV bin.  Note that this function is called after the maximal 
   (cpos, cneg) SVM is identified and receives the corresponding sotred vector of bools, i.e., when nestedXvalBins_ <= 1
*/
void CrossValidation::writeSupportVectors(const AlgIn& data, int fold, int trainingIter, vector<bool> &supportVectors){
#ifndef WIN32
      std::string str = psmInfluencerDIR_ + "/supportVectors";
#else
      std::string str = psmInfluencerDIR_ + "\supportVectors";
#endif
      char buffer [30];
      sprintf(buffer,"_iteration%d_fold%d", trainingIter, fold);
      str.append(buffer);
      str.append(".txt");

      PSMDescription* pPSM;
      double* setRow;

      // double** set = data.vals;
      const double* Y = data.Y;
      const int n = data.n;
      const int m = data.m;

      ofstream featFile;
      featFile.open(str.c_str());
      // Write header
      // featFile << "PSMId\tLabel\tpeptide\tproteinIds";
      featFile << "PSMId\tLabel";
      for(int i = 0; i < FeatureNames::getNumFeatures(); i++){
	featFile << "\t" << DataSet::getFeatureNames().getFeatureName(i);
      }
      featFile << "\tpeptide\tproteinIds" << endl;

      // Write support vectors
      for(int i = 0; i < m; i++){
      	if(!supportVectors[i]){
      	  continue;
      	}
	pPSM = data.pPSMs[i];
	setRow = pPSM->features;
	// Peptide info
	std::ostringstream out;
	pPSM->printProteins(out);
	featFile << pPSM->getId() <<  "\t" << Y[i];
	// PSM feature values
      	for(int j = 0; j < FeatureNames::getNumFeatures(); j++){
	  featFile << "\t" << setRow[j];
      	}
      	featFile << "\t" << pPSM->peptide << out.str() << endl;
      }
      featFile.close();
}

/* Write out psm influencer info for current set of nested folds.  Note that this function uses the 
   bool variables directly from ssl.cpp, since L2_SVM_MFN is called just prior to this function, i.e., 
   when nestedXvalBins_ > 1
*/
void CrossValidation::writeSupportVectors(const AlgIn& data, int fold, int trainingIter){
#ifndef WIN32
      std::string str = psmInfluencerDIR_ + "/supportVectors";
#else
      std::string str = psmInfluencerDIR_ + "\supportVectors";
#endif
      char buffer [30];
      sprintf(buffer,"_iteration%d_fold%d", trainingIter, fold);
      str.append(buffer);
      str.append(".txt");

      PSMDescription* pPSM;
      double* setRow;

      const bool* supportVectors = data.supportVectors;
      const double* Y = data.Y;
      const int n = data.n;
      const int m = data.m;

      ofstream featFile;
      featFile.open(str.c_str());
      // Write header
      // featFile << "PSMId\tLabel\tpeptide\tproteinIds";
      featFile << "PSMId\tLabel";
      for(int i = 0; i < FeatureNames::getNumFeatures(); i++){
	featFile << "\t" << DataSet::getFeatureNames().getFeatureName(i);
      }
      featFile << "\tpeptide\tproteinIds" << endl;

      // Write support vectors
      for(int i = 0; i < m; i++){
      	if(!supportVectors[i]){
      	  continue;
      	}
	pPSM = data.pPSMs[i];
	setRow = pPSM->features;
	// Peptide info
	std::ostringstream out;
	pPSM->printProteins(out);
	featFile << pPSM->getId() <<  "\t" << Y[i];
	// PSM feature values
      	for(int j = 0; j < FeatureNames::getNumFeatures(); j++){
	  featFile << "\t" << setRow[j];
      	}
      	featFile << "\t" << pPSM->peptide << out.str() << endl;
      }
      featFile.close();
}

// Write training set sizes
void CrossValidation::writeTargetDecoyTrainSizesHeader(){
#ifndef WIN32
  std::string str = psmInfluencerDIR_ + "/trainSetSizes.txt";
#else
  std::string str = psmInfluencerDIR_ + "\trainSetSizes.txt";
#endif
  ofstream featFile;
  featFile.open(str.c_str());
  featFile << "Iterations\tFold\tNumTargets\tNumDecoys\tAllTargets" << endl;
  featFile.close();
}

// Write training set sizes
 void CrossValidation::writeTargetDecoyTrainSizes(int iteration, int fold, int positives, int negatives, int allPositives){
#ifndef WIN32
  std::string str = psmInfluencerDIR_ + "/trainSetSizes.txt";
#else
  std::string str = psmInfluencerDIR_ + "\trainSetSizes.txt";
#endif
  ofstream featFile;
  featFile.open(str.c_str(), ofstream::app);
  featFile << iteration << "\t" << fold << "\t" << positives << "\t" << negatives << "\t" << allPositives << endl;
  featFile.close();
}

/* Write out test sets
*/
void CrossValidation::writeTestSets(){
      char buffer [30];
      PSMDescription* pPSM;
      double* setRow;
      double y;

      for (int set = 0; set < numFolds_; ++set) {
#ifndef WIN32
	std::string str = psmInfluencerDIR_ + "/testSet";
#else
	std::string str = psmInfluencerDIR_ + "\testSet";
#endif
	sprintf(buffer,"_fold%d", set);
	str.append(buffer);
	str.append(".txt");

	ofstream featFile;
	featFile.open(str.c_str());
	// Write header
	// featFile << "PSMId\tLabel\tpeptide\tproteinIds";
	featFile << "PSMId\tLabel";
	for(int i = 0; i < FeatureNames::getNumFeatures(); i++){
	  featFile << "\t" << DataSet::getFeatureNames().getFeatureName(i);
	}
	featFile << "\tpeptide\tproteinIds" << endl;

	// Write test features
	std::vector<ScoreHolder>::const_iterator scoreIt = testScores_[set].begin();
	for ( ; scoreIt != testScores_[set].end(); ++scoreIt) {
	  if (scoreIt->isTarget()) {
	    y = 1;
	  }else{
	    y=-1;
	  }
	  pPSM = scoreIt->pPSM;
	  setRow = scoreIt->pPSM->features;
	  // Peptide info
	  std::ostringstream out;
	  pPSM->printProteins(out);
	  featFile << pPSM->getId() <<  "\t" << y;
	  // PSM feature values
	  for(int j = 0; j < FeatureNames::getNumFeatures(); j++){
	    featFile << "\t" << setRow[j];
	  }
	  featFile << "\t" << pPSM->peptide << out.str() << endl;
	}
	featFile.close();
      }
}

/* Write normalizer weights
*/
void CrossValidation::writeNormalizers(Normalizer* pnorm){
  double* denoms = pnorm->getDiv();
  double* mus = pnorm->getSub();
  size_t offset = 0;
  if (DataSet::getCalcDoc()) {
    offset = FeatureNames::getNumFeatures() -DescriptionOfCorrect::numDOCFeatures();
  }
#ifndef WIN32
	std::string str = psmInfluencerDIR_ + "/normalizers.txt";
#else
	std::string str = psmInfluencerDIR_ + "\normalizers";
#endif
	ofstream featFile;
	featFile.open(str.c_str());
	featFile << "mu\tdenom" << endl;
	for(int i = 0; i < FeatureNames::getNumFeatures(); i++){
	  featFile << fixed << setprecision(8) << mus[offset + i];
	  featFile << "\t";
	  featFile << fixed << setprecision(8) << denoms[offset + i];
	  featFile << endl;
	}
	featFile.close();
}

/* Within a Percolator iteration, write out top (cpos, cneg) SVM learned weights for each fold 
*/
void CrossValidation::writeSvmWeights(std::vector< std::vector<double> > weightMatrix, int trainingIter){
#ifndef WIN32
      std::string str = psmInfluencerDIR_ + "/svmWeights";
#else
      std::string str = psmInfluencerDIR_ + "\svmWeights";
#endif
      char buffer [30];
      sprintf(buffer,"_iteration%d", trainingIter);
      str.append(buffer);
      str.append(".txt");

      ofstream weightFile;
      weightFile.open(str.c_str());
      // Write header
      weightFile << "CVFold";
      for(int i = 0; i < FeatureNames::getNumFeatures(); i++){
	weightFile << "\t" << DataSet::getFeatureNames().getFeatureName(i);
      }
      weightFile << "\tbias" << endl;
      // Write weights per CV fold
      for (int set = 0; set < numFolds_; ++set) {
	weightFile << set;
	for(int i = 0; i < FeatureNames::getNumFeatures() + 1; i++){
	  weightFile << "\t" << std::setprecision(10) << weightMatrix[set][i];
	}
	weightFile << endl;
      }

      weightFile.close();
}

/** 
 * Validate and merge weights learned per cpos,cneg pairs per nested CV fold per CV fold
 * @param pWeights results vector from the SVM algorithm
 * @param pOptions options for the SVM algorithm
 * @param nestedTestScoresVec 2D vector, test sets per nested CV fold per CV fold
*/
int CrossValidation::mergeCpCnPairs(double selectionFdr,
                                    options * pOptions, vector< vector<Scores> >& nestedTestScoresVec,
                                    const vector<double>& cposCandidates, const vector<double>& cfracCandidates, int trainingIter) {
  // for determining the number of positives, the decoys+1 in the FDR estimates 
  // is too restrictive for small datasets
  bool skipDecoysPlusOne = true;

  vector<int> bestTruePoses(numFolds_, 0);
  vector<double> bestCposes(numFolds_, 1);
  vector<double> bestCfracs(numFolds_, 1);
  
  int set = 0;
  // Validate learned parameters per (cpos,cneg) pair per nested CV fold
  // Note: this cannot be done in trainCpCnPair without setting a critical pragma, due to the 
  //       scoring calculation in calcScores.
  unsigned int numCpCnPairsPerSet = classWeightsPerFold_.size() / numFolds_;
#pragma omp parallel for schedule(dynamic, 1) ordered
  for (set = 0; set < numFolds_; ++set) {
    unsigned int a = set * numCpCnPairsPerSet;
    unsigned int b = (set+1) * numCpCnPairsPerSet;
    int counter = -1;
    int bestInd = 0;
    int tp = 0;
    std::vector<candidateCposCfrac>::iterator itCpCnPair;
    std::map<std::pair<double, double>, int> intermediateResults;
    for (itCpCnPair = classWeightsPerFold_.begin() + a; itCpCnPair < classWeightsPerFold_.begin() + b; itCpCnPair++) {
      counter++;
      tp = nestedTestScoresVec[set][itCpCnPair->nestedSet].calcScores(itCpCnPair->ww, testFdr_, skipDecoysPlusOne);
      intermediateResults[std::make_pair(itCpCnPair->cpos, itCpCnPair->cfrac)] += tp;
      itCpCnPair->tp = tp;
      if (nestedXvalBins_ <= 1) {
        if(tp >= bestTruePoses[set]){
          bestTruePoses[set] = tp;
          w_[set] = itCpCnPair->ww;
          bestCposes[set] = itCpCnPair->cpos;
          bestCfracs[set] = itCpCnPair->cfrac;
        }
      }
    }
    if (nestedXvalBins_ > 1) {     // Check nestedXvalBins, which collapse (accumulate) tp estimated for each CV bin
      // Now check which achieved best performance among cpos, cneg pairs
      std::vector<double>::const_iterator itCpos = cposCandidates.begin();
      for ( ; itCpos != cposCandidates.end(); ++itCpos) {
        double cpos = *itCpos;  
        std::vector<double>::const_iterator itCfrac = cfracCandidates.begin();
        for ( ; itCfrac != cfracCandidates.end(); ++itCfrac) {
          double cfrac = *itCfrac;
          tp = intermediateResults[std::make_pair(cpos, cfrac)];
          if(tp >= bestTruePoses[set]){
            bestTruePoses[set] = tp;
            bestCposes[set] = cpos;
            bestCfracs[set] = cfrac;
	    bestInd = counter;
          }
        }
      }
    }

    // Write out psm influencer info for optimal (cpos,cneg) SVMs per set
    if (nestedXvalBins_ <= 1) {
      AlgIn* svmInput = svmInputs_[set];
      vector<bool> *svs = &classWeightsPerFold_[bestInd].supportVectors;
      writeSupportVectors(*svmInput, set, trainingIter, *svs);
    }

  }
  
  if (nestedXvalBins_ > 1) {
#pragma omp parallel for schedule(dynamic, 1) ordered
    for (set = 0; set < numFolds_; ++set) {
      struct vector_double* pWeights = new vector_double;
      pWeights->d = FeatureNames::getNumFeatures() + 1;
      pWeights->vec = new double[pWeights->d];

      AlgIn* svmInput = svmInputs_[set * nestedXvalBins_];
      trainScores_[set].generateNegativeTrainingSet(*svmInput, 1.0);
      trainScores_[set].generatePositiveTrainingSet(*svmInput, selectionFdr, 1.0, trainBestPositive_);
    
      // Create storage vector for SVM algorithm
      struct vector_double* Outputs = new vector_double;
      size_t numInputs = svmInput->positives + svmInput->negatives;
      Outputs->vec = new double[numInputs];
      Outputs->d = numInputs;
    
      for (int ix = 0; ix < pWeights->d; ix++) {
        pWeights->vec[ix] = 0;
      }
      for (int ix = 0; ix < Outputs->d; ix++) {
        Outputs->vec[ix] = 0;
      }
      // Call SVM algorithm (see ssl.cpp)
      L2_SVM_MFN(*svmInput, pOptions, pWeights, Outputs, bestCposes[set], bestCposes[set] * bestCfracs[set]);
      writeSupportVectors(*svmInput, set, trainingIter);
      for (int i = FeatureNames::getNumFeatures() + 1; i--;) {
        w_[set][i] = pWeights->vec[i];
      }
      delete[] pWeights->vec;
      delete pWeights;
      delete[] Outputs->vec;
      delete Outputs;
    }
  }

  writeSvmWeights(w_, trainingIter);

  double bestTruePos = 0;
  for (set = 0; set < numFolds_; ++set) {
    bestTruePos += trainScores_[set].calcScores(w_[set], testFdr_);
  }
  return bestTruePos / (numFolds_ - 1);
}

void CrossValidation::postIterationProcessing(Scores& fullset,
                                              SanityCheck* pCheck) {
  if (!pCheck->validateDirection(w_)) {
    for (int set = 0; set < numFolds_; ++set) {
      testScores_[set].calcScores(w_[0], selectionFdr_);
    }
  }
  if (DataSet::getCalcDoc()) {
    // TODO: take the average instead of the first DOC model?
    fullset.getDOC().copyDOCparameters(testScores_[0].getDOC());
  }
  fullset.merge(testScores_, selectionFdr_, skipNormalizeScores_);
}

/**
 * Prints the weights of the normalized vector to a stream
 * @param weightStream stream where the weights are written to
 * @param w_ normal vector
 */
void CrossValidation::printSetWeights(ostream & weightStream, unsigned int set) {
  weightStream << DataSet::getFeatureNames().getFeatureNames() << 
                  "\tm0" << std::endl;
  weightStream.precision(3);
  weightStream << w_[set][0];
  for (unsigned int ix = 1; ix < FeatureNames::getNumFeatures() + 1; ix++) {
    weightStream << "\t" << fixed << setprecision(4) << w_[set][ix];
  }
  weightStream << endl;
}

/**
 * Prints the weights of the raw vector to a stream
 * @param weightStream stream where the weights are written to
 * @param w_ normal vector
 */
void CrossValidation::printRawSetWeights(ostream & weightStream, unsigned int set, 
                                      Normalizer * pNorm) {
  vector<double> ww(FeatureNames::getNumFeatures() + 1);
  pNorm->unnormalizeweight(w_[set], ww);
  weightStream << ww[0];
  for (unsigned int ix = 1; ix < FeatureNames::getNumFeatures() + 1; ix++) {
    weightStream << "\t" << fixed << setprecision(4) << ww[ix];
  }
  weightStream << endl;
}

// used to save weights for reuse as weight input file
void CrossValidation::printAllWeights(ostream & weightStream, 
                                      Normalizer * pNorm) {
  for (unsigned int ix = 0; ix < numFolds_; ++ix) {
    printSetWeights(weightStream, ix);
    printRawSetWeights(weightStream, ix, pNorm);
  }
}

void CrossValidation::getAvgWeights(std::vector<double>& weights, 
                                    Normalizer * pNorm) {
  vector<double> ww(FeatureNames::getNumFeatures() + 1);
  
  weights.resize(FeatureNames::getNumFeatures() + 1);
  for (size_t set = 0; set < w_.size(); ++set) {
    pNorm->unnormalizeweight(w_[set], ww);
    for (unsigned int ix = 0; ix < FeatureNames::getNumFeatures() + 1; ix++) {
      weights[ix] += ww[ix] / w_.size();
    }
  }
}

void CrossValidation::printAllWeightsColumns(
    std::vector< std::vector<double> > weightMatrix, ostream & outputStream) {
  // write to intermediate stream to prevent the fixed precision from sticking
  ostringstream weightStream;
  for (unsigned int set = 0; set < numFolds_; ++set) {
    weightStream << " Split" << set + 1 << '\t'; // right-align with weights
  }
  weightStream << "FeatureName" << std::endl;
  size_t numRows = FeatureNames::getNumFeatures() + 1;
  for (unsigned int ix = 0; ix < numRows; ix++) {
    for (unsigned int set = 0; set < numFolds_; ++set) {
      // align positive and negative weights
      if (weightMatrix[set][ix] >= 0) weightStream << " ";
      weightStream << fixed << setprecision(4) << weightMatrix[set][ix];
      weightStream << "\t";
    }
    if (ix == numRows - 1) {
      weightStream << "m0";
    } else {
      weightStream << DataSet::getFeatureNames().getFeatureName(ix);
    }
    weightStream << std::endl;
  }
  outputStream << weightStream.str();
}

void CrossValidation::printAllWeightsColumns(ostream & weightStream) {
  std::cerr << "Learned normalized SVM weights for the " << numFolds_ 
            << " cross-validation splits:" << std::endl;
  printAllWeightsColumns(w_, weightStream);
}

void CrossValidation::printAllRawWeightsColumns(ostream & weightStream, 
                                      Normalizer * pNorm) {
  std::cerr << "Learned raw SVM weights for the " << numFolds_ 
            << " cross-validation splits:" << std::endl;
  std::vector< std::vector<double> > ww = w_;
  for (size_t set = 0; set < w_.size(); ++set) {
    pNorm->unnormalizeweight(w_[set], ww[set]);
  }
  printAllWeightsColumns(ww, weightStream);
}

void CrossValidation::printDOC() {
  cerr << "For the cross validation sets the average deltaMass are ";
  for (size_t ix = 0; ix < testScores_.size(); ix++) {
    cerr << testScores_[ix].getDOC().getAvgDeltaMass() << " ";
  }
  cerr << "and average pI are ";
  for (size_t ix = 0; ix < testScores_.size(); ix++) {
    cerr << testScores_[ix].getDOC().getAvgPI() << " ";
  }
  cerr << endl;
}
