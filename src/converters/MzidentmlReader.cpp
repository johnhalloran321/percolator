#include "MzidentmlReader.h"
#include "DataSet.h"

static const XMLCh sequenceCollectionStr[] = { chLatin_S, chLatin_e, chLatin_q, chLatin_u, chLatin_e,chLatin_n, 
						  chLatin_c, chLatin_e, chLatin_C, chLatin_o, chLatin_l, chLatin_l, chLatin_e, 
						  chLatin_c, chLatin_t, chLatin_i, chLatin_o, chLatin_n, chNull };
						  
						  

MzidentmlReader::MzidentmlReader(ParseOptions po):Reader(po)
{
  hashparams["sequest:PeptideRankSp"] = 0;
  hashparams["sequest:deltacn"] = 1;
  hashparams["sequest:xcorr"] = 2;
  hashparams["sequest:PeptideSp"] = 3;
  hashparams["sequest:matched ions"] = 4;
  hashparams["sequest:total ions"] = 5;
  hashparams["sequest:PeptideIdnumber"] = 6;
  hashparams["sequest:PeptideNumber"] = 7;
}

MzidentmlReader::~MzidentmlReader()
{
      


}

void MzidentmlReader::cleanHashMaps()
{
    peptideMapType::iterator iter;
    for(iter = peptideMap.begin(); iter != peptideMap.end(); ++iter) 
    { 
      delete iter->second; iter->second=0;
    }
        
    proteinMapType::iterator iter2;
    for(iter2 = proteinMap.begin(); iter2 != proteinMap.end(); ++iter2) 
    { 
      delete iter2->second; iter2->second=0;
    }
    
        
    peptideEvidenceMapType::iterator iter3;
    for(iter3 = peptideEvidenceMap.begin(); iter3 != peptideEvidenceMap.end(); ++iter3) 
    { 
      delete iter3->second; iter3->second=0;
    }
}

bool MzidentmlReader::checkValidity(string file)
{
  bool ismeta = true;
  std::ifstream fileIn(file.c_str(), std::ios::in);
  if (!fileIn) {
    std::cerr << "Could not open file " << file << std::endl;
    exit(-1);
  }
  std::string line;
  if (!getline(fileIn, line)) {
    std::cerr << "Could not read file " << file << std::endl;
    exit(-1);
  }
  fileIn.close();
  if (line.size() > 1 && line[0]=='<' && line[1]=='?') {
    //TODO here I should check that the file is xml and has the tag <MzIdentML id="SEQUEST_use_case"
    if (line.find("xml") == std::string::npos) {
      std::cerr << "file is not xml format " << file << std::endl;
      exit(-1);
    }
  }
  else
  {
    ismeta = false;
  }
  return ismeta;
}


void MzidentmlReader::addFeatureDescriptions(bool doEnzyme, const string& aaAlphabet, std::string fn)
{
  //NOTE from now lets assume the features all always SEQUEST features, ideally I would create my list of features from the 
  //     features description of the XSD
  push_backFeatureDescription("lnrSp");
  push_backFeatureDescription("deltCn");
  push_backFeatureDescription("Xcorr");
  push_backFeatureDescription("Sp");
  push_backFeatureDescription("IonFrac");
  push_backFeatureDescription("Mass");
  push_backFeatureDescription("PepLen");
  for (int charge = minCharge; charge <= maxCharge; ++charge) 
  {
    std::ostringstream cname;
    cname << "Charge" << charge;
    push_backFeatureDescription(cname.str().c_str());

  }
  if (doEnzyme) 
  {
    push_backFeatureDescription("enzN");
    push_backFeatureDescription("enzC");
    push_backFeatureDescription("enzInt");
  }
    
  push_backFeatureDescription("dM");
  push_backFeatureDescription("absdM");
  
  if (po.calcPTMs) 
  {
    push_backFeatureDescription("ptm");
  }
  if (po.pngasef) 
  {
    push_backFeatureDescription("PNGaseF");
  }
  if (!aaAlphabet.empty()) 
  {
    for (std::string::const_iterator it = aaAlphabet.begin(); it != aaAlphabet.end(); it++)
    {
      push_backFeatureDescription(*it + "-Freq");
    }
  }
    
}


void MzidentmlReader::getMaxMinCharge(string fn)
{

  bool foundFirstChargeState = false;
  ifstream ifs;
  ifs.exceptions (ifstream::badbit | ifstream::failbit);
  try
  {
    ifs.open (fn.c_str());
    parser p;
    string schemaDefinition = MZIDENTML_SCHEMA_LOCATION + string("mzIdentML1.1.0.xsd");
    string scheme_namespace = MZIDENTML_NAMESPACE;
    string schema_major = "";
    string schema_minor = "";
    xml_schema::dom::auto_ptr<DOMDocument> doc (p.start (ifs, fn.c_str(), true, schemaDefinition, schema_major, schema_minor, scheme_namespace));
    
    //NOTE wouldnt be  better to use the get tag by Name to jump this?
    for (doc = p.next(); doc.get() != 0 && !XMLString::equals(spectrumIdentificationResultStr,doc->getDocumentElement()->getTagName()); doc = p.next ()) 
    {
      // Let's skip some sub trees that we are not interested, e.g. AnalysisCollection
    }
    
    ::mzIdentML_ns::SpectrumIdentificationResultType specIdResult(*doc->getDocumentElement ());
    for (; doc.get () != 0 && XMLString::equals( spectrumIdentificationResultStr, doc->getDocumentElement ()->getTagName() ); doc = p.next ()) 
    {
      ::mzIdentML_ns::SpectrumIdentificationResultType specIdResult(*doc->getDocumentElement ());
      
      assert(specIdResult.SpectrumIdentificationItem().size() > 0);
      ::percolatorInNs::fragSpectrumScan::experimentalMassToCharge_type experimentalMassToCharge = specIdResult.SpectrumIdentificationItem()[0].experimentalMassToCharge();
      
      BOOST_FOREACH( const ::mzIdentML_ns::SpectrumIdentificationItemType & item, specIdResult.SpectrumIdentificationItem() )
      {
	if ( ! foundFirstChargeState ) 
	{
	  minCharge = item.chargeState();
	  maxCharge = item.chargeState();
	  foundFirstChargeState = true;
	}
	minCharge = std::min(item.chargeState(),minCharge);
	maxCharge = std::max(item.chargeState(),maxCharge);
      }
    }
  }catch (ifstream::failure e) {
    cerr << "Exception opening/reading file :" << fn <<endl;
  }
  catch (const xercesc::DOMException& e)
  {
    char * tmpStr = XMLString::transcode(e.getMessage());
    std::cerr << "catch xercesc_3_1::DOMException=" << tmpStr << std::endl;  
    XMLString::release(&tmpStr);

  }
  catch (const xml_schema::exception& e)
  {
    cerr << e << endl;
  }
  catch(std::exception e){
    cerr << e.what() <<endl;
  }
  
  assert( foundFirstChargeState );
  return;
}

void MzidentmlReader::read(const std::string fn, bool isDecoy, boost::shared_ptr<FragSpectrumScanDatabase> database) 
{
  namespace xml = xsd::cxx::xml;
  int scanNumber=0;
  scanNumberMapType scanNumberMap;
  
  try
  {
    ifstream ifs;
    ifs.exceptions (ifstream::badbit | ifstream::failbit);
    ifs.open (fn.c_str());
    parser p;
    string schemaDefinition = MZIDENTML_SCHEMA_LOCATION + string("mzIdentML1.1.0.xsd");
    string scheme_namespace = MZIDENTML_NAMESPACE;
    string schema_major = "";
    string schema_minor = "";
    xml_schema::dom::auto_ptr<DOMDocument> doc (p.start (ifs, fn.c_str(), true, schemaDefinition, schema_major, schema_minor, scheme_namespace));
    
    //NOTE wouldnt be  better to use the get tag by Name to jump SequenceCollenction directly?
    while (doc.get () != 0 && ! XMLString::equals( sequenceCollectionStr, doc->getDocumentElement ()->getTagName())) 
    {
      doc = p.next ();// Let's skip some sub trees that we are not interested, e.g. AuditCollection
    }
    
    assert(doc.get());
    mzIdentML_ns::SequenceCollectionType sequenceCollection(*doc->getDocumentElement ());
    
    peptideMap.clear();
    proteinMap.clear();
    peptideEvidenceMap.clear();
    
    //NOTE probably I can get rid of these hash tables with a proper access to elements by tag and id
    BOOST_FOREACH( const mzIdentML_ns::SequenceCollectionType::Peptide_type &peptide, sequenceCollection.Peptide() ) 
    {
      //PEPTIDE
      mzIdentML_ns::SequenceCollectionType::Peptide_type *pept = new mzIdentML_ns::SequenceCollectionType::Peptide_type(peptide);
      peptideMap.insert( std::make_pair(peptide.id(), pept));      
    }
    
    BOOST_FOREACH( const mzIdentML_ns::SequenceCollectionType::DBSequence_type &protein, sequenceCollection.DBSequence() ) 
    {
      //PROTEIN
      mzIdentML_ns::SequenceCollectionType::DBSequence_type *prot = new mzIdentML_ns::SequenceCollectionType::DBSequence_type(protein);
      proteinMap.insert( std::make_pair(protein.id(), prot));      
    }
    
    BOOST_FOREACH( const ::mzIdentML_ns::PeptideEvidenceType &peptideE, sequenceCollection.PeptideEvidence() ) 
    {
      //PEPTIDE EVIDENCE
      ::mzIdentML_ns::PeptideEvidenceType *peptE = new mzIdentML_ns::PeptideEvidenceType(peptideE);
      //peptideEvidenceMap_peptideid.insert(std::make_pair(peptideE.peptide_ref(),peptE));
      peptideEvidenceMap.insert(std::make_pair(peptideE.id(),peptE)); 
    }
    
    //NOTE wouldnt be  better to use the get tag by Name to jump to Spectrum collection?
    for (doc = p.next (); doc.get () != 0 && !XMLString::equals( spectrumIdentificationResultStr, doc->getDocumentElement ()->getTagName() ); doc = p.next ()) 
    {
      // Let's skip some sub trees that we are not interested, e.g. AnalysisCollection
    }
    
    ::mzIdentML_ns::SpectrumIdentificationResultType specIdResult(*doc->getDocumentElement ());
    assert( specIdResult.SpectrumIdentificationItem().size() > 0 );
    unsigned scanNumber = 0;
    for (; doc.get () != 0 && XMLString::equals( spectrumIdentificationResultStr, doc->getDocumentElement ()->getTagName() ); doc = p.next ()) 
    {
      ::mzIdentML_ns::SpectrumIdentificationResultType specIdResult(*doc->getDocumentElement ());
      assert(specIdResult.SpectrumIdentificationItem().size() > 0);
      ::percolatorInNs::fragSpectrumScan::experimentalMassToCharge_type experimentalMassToCharge = specIdResult.SpectrumIdentificationItem()[0].experimentalMassToCharge();
      
      BOOST_FOREACH( const ::mzIdentML_ns::SpectrumIdentificationItemType & item, specIdResult.SpectrumIdentificationItem() )  
      {
        createPSM(item, experimentalMassToCharge, isDecoy, ++scanNumber, database);
      }
      
    }
    
    cleanHashMaps();

  }
  catch (const xercesc::DOMException& e)
  {
    char * tmpStr = XMLString::transcode(e.getMessage());
    std::cerr << "catch xercesc_3_1::DOMException=" << tmpStr << std::endl;  
    XMLString::release(&tmpStr);
    exit(-1);
  }
  catch (const xml_schema::exception& e)
  {
    cerr << e << endl;
    exit(-1);
  }
  catch (const ios_base::failure&)
  {
    cerr << "io failure" << endl;
    exit(-1);
  }
}

void MzidentmlReader::createPSM(const ::mzIdentML_ns::SpectrumIdentificationItemType & item, 
				  ::percolatorInNs::fragSpectrumScan::experimentalMassToCharge_type experimentalMassToCharge,
				  bool isDecoy,unsigned useScanNumber,boost::shared_ptr<FragSpectrumScanDatabase> database) 
{
  


  std::auto_ptr< percolatorInNs::features >  features_p( new percolatorInNs::features ());
  percolatorInNs::features::feature_sequence & f_seq =  features_p->feature();
 
  if ( ! item.calculatedMassToCharge().present() ) 
  { 
    std::cerr << "error: calculatedMassToCharge attribute is needed for percolator" << std::endl; 
    exit(EXIT_FAILURE); 
  }
  
  std::string peptideSeq =  peptideMap[item.peptide_ref().get()]->PeptideSequence();
  std::string peptideId = item.peptide_ref().get();
  
  /*std::vector<mzIdentML_ns::PeptideEvidenceType *> peptide_evidences;
  std::transform(peptideEvidenceMap_peptideid.lower_bound(peptideId),peptideEvidenceMap_peptideid.upper_bound(peptideId),
		 std::inserter(peptide_evidences,peptide_evidences.begin()), RetrieveValue());*/
  
  std::vector< std::string > proteinIds;
  std::string __flankN = "";
  std::string __flankC = "";
  
  //NOTE I should notify the authors of mzIdentML to notify this bug
  /*if(item.PeptideEvidenceRef().size() != peptide_evidences.size())
  {
    std::cerr << "Warning : something extrange happened, the number of Peptide Evidences of PSM "
              << boost::lexical_cast<string>(item.id()) << " found in the Spectrum tag does not "
	      << "correspond to the number of evidences found in the PeptideEvidence tag." << std::endl;
  }*/
  
  
  BOOST_FOREACH(const ::mzIdentML_ns::PeptideEvidenceRefType &pepEv_ref, item.PeptideEvidenceRef())
  {
    std::string ref_id = pepEv_ref.peptideEvidence_ref().c_str();
    ::mzIdentML_ns::PeptideEvidenceType *pepEv = peptideEvidenceMap[ref_id];
    __flankN = boost::lexical_cast<string>(pepEv->pre());
    __flankC = boost::lexical_cast<string>(pepEv->post());
    std::string proteinid = boost::lexical_cast<string>(pepEv->dBSequence_ref());
    mzIdentML_ns::SequenceCollectionType::DBSequence_type *proteinObj = proteinMap[proteinid];
    std::string proteinName = boost::lexical_cast<string>(proteinObj->accession());
    proteinIds.push_back(proteinName);
  }
  
  /*BOOST_FOREACH(const ::mzIdentML_ns::PeptideEvidenceType *pepEv, peptide_evidences)
  {
    __flankN = boost::lexical_cast<string>(pepEv->pre());
    __flankC = boost::lexical_cast<string>(pepEv->post());
    std::string proteinid = boost::lexical_cast<string>(pepEv->dBSequence_ref());
    mzIdentML_ns::SequenceCollectionType::DBSequence_type *proteinObj = proteinMap[proteinid];
    std::string proteinName = boost::lexical_cast<string>(proteinObj->accession());
    proteinIds.push_back(proteinName);
  }*/
   
  if(po.iscombined && !po.reversedFeaturePattern.empty())
  {
    //NOTE taking the highest ranked PSM protein for combined search
    isDecoy = proteinIds.front().find(po.reversedFeaturePattern, 0) != std::string::npos;
  }
  
  double rank = item.rank();
  double PI = boost::lexical_cast<double>(item.calculatedPI().get());
  double lnrSP = 0.0;
  double deltaCN = 0.0;
  double xCorr = 0.0;
  double Sp = 0.0;
  int charge = item.chargeState();
  double ionMatched = 0.0;
  double ionTotal = 0.0;
  double theoretic_mass = boost::lexical_cast<double>(item.calculatedMassToCharge());
  double observed_mass = boost::lexical_cast<double>(item.experimentalMassToCharge());
  std::string peptideSeqWithFlanks = __flankN + std::string(".") + peptideSeq + std::string(".") + __flankC;
  assert(peptideSeqWithFlanks.size() >= po.peptidelength );
  unsigned peptide_length = DataSet::peptideLength(peptideSeqWithFlanks);
  double dM = MassHandler::massDiff(observed_mass,theoretic_mass,charge, peptideSeq );
  std::map<char,int> ptmMap = po.ptmScheme;
  std::string psmid = boost::lexical_cast<string>(item.id()) + "_" + boost::lexical_cast<string>(useScanNumber) + "_" + 
		       boost::lexical_cast<string>(charge) + "_" + boost::lexical_cast<string>(rank);
  
  BOOST_FOREACH( const ::mzIdentML_ns::CVParamType & cv, item.cvParam() )
  {
    if ( cv.value().present() )
    {
      //NOTE this is risky, check for key fail so I know some new parameter are included
      switch(hashparams[std::string(cv.name().c_str())])
      {
	case 0: lnrSP = boost::lexical_cast<double>(cv.value().get().c_str());break;
	case 1: deltaCN = boost::lexical_cast<double>(cv.value().get().c_str());break;
	case 2: xCorr = boost::lexical_cast<double>(cv.value().get().c_str());break;
	case 3: Sp = boost::lexical_cast<double>(cv.value().get().c_str());break;
	case 4: ionMatched = boost::lexical_cast<double>(cv.value().get().c_str());break;
	case 5: ionTotal = boost::lexical_cast<double>(cv.value().get().c_str());break;  
      }
    }
  }

  f_seq.push_back( log(max(1.0,lnrSP)) );
  f_seq.push_back( deltaCN );
  f_seq.push_back( xCorr );
  f_seq.push_back( Sp );
  f_seq.push_back( ionMatched/ionTotal );
  f_seq.push_back( observed_mass ); 
  f_seq.push_back( DataSet::peptideLength(peptideSeqWithFlanks) ); 

  for (int c = minCharge; c <= maxCharge; c++) 
  {
    f_seq.push_back( charge == c ? 1.0 : 0.0); // Charge
  }
  if ( Enzyme::getEnzymeType() != Enzyme::NO_ENZYME ) 
  {
    f_seq.push_back( Enzyme::isEnzymatic(peptideSeqWithFlanks.at(0),peptideSeqWithFlanks.at(2)) ? 1.0: 0.0);
    f_seq.push_back( Enzyme::isEnzymatic(peptideSeqWithFlanks.at(peptideSeqWithFlanks.size() - 3),peptideSeqWithFlanks.at(peptideSeqWithFlanks.size() - 1)) ? 1.0: 0.0);
    f_seq.push_back( (double)Enzyme::countEnzymatic(peptideSeq) );
  }
  
  f_seq.push_back( dM ); 
  f_seq.push_back( abs(dM) ); 
  
  if (po.calcPTMs ) 
  { 
    f_seq.push_back(  DataSet::cntPTMs(peptideSeqWithFlanks)); 
  }
  if (po.pngasef ) 
  { 
    f_seq.push_back( DataSet::isPngasef(peptideSeqWithFlanks, isDecoy)); 
  }
  if (po.calcAAFrequencies ) 
  { 
    computeAAFrequencies(peptideSeqWithFlanks, f_seq); 
  }
  
  percolatorInNs::occurence::flankN_type flankN = peptideSeqWithFlanks.substr(0,1);
  percolatorInNs::occurence::flankC_type flankC = peptideSeqWithFlanks.substr(peptideSeqWithFlanks.size() - 1,1);
  
  // Strip peptide from termini and modifications 
  std::string peptideS = peptideSeq;
  for(unsigned int ix=0;ix<peptideSeq.size();++ix) 
  {
    if (aaAlphabet.find(peptideSeq[ix])==string::npos && ambiguousAA.find(peptideSeq[ix])==string::npos && modifiedAA.find(peptideSeq[ix])==string::npos)
    {
      if (ptmMap.count(peptideSeq[ix])==0) 
      {
	cerr << "Peptide sequence " << peptideSeqWithFlanks << " contains modification " << peptideSeq[ix] << " that is not specified by a \"-p\" argument" << endl;
        exit(-1);
      }
      peptideSeq.erase(ix,1);
    }  
  }
  
  std::auto_ptr< percolatorInNs::peptideType >  peptide_p( new percolatorInNs::peptideType(peptideSeq));
  // Register the ptms
  for(unsigned int ix=0;ix<peptideS.size();++ix) 
  {
    if (aaAlphabet.find(peptideS[ix])==string::npos) 
    {
      int accession = ptmMap[peptideS[ix]];
      std::auto_ptr< percolatorInNs::uniMod > um_p (new percolatorInNs::uniMod(accession));
      std::auto_ptr< percolatorInNs::modificationType >  mod_p( new percolatorInNs::modificationType(um_p,ix));
      peptide_p->modification().push_back(mod_p);      
      peptideS.erase(ix,1);      
    }  
  }
  
  ::percolatorInNs::peptideSpectrumMatch* tmp_psm = new ::percolatorInNs::peptideSpectrumMatch(features_p, peptide_p, psmid, isDecoy, observed_mass, theoretic_mass, charge);
  std::auto_ptr< ::percolatorInNs::peptideSpectrumMatch > psm_p(tmp_psm);
  
  for ( std::vector< std::string >::const_iterator i = proteinIds.begin(); i != proteinIds.end(); ++i ) 
  {
    std::auto_ptr< percolatorInNs::occurence >  oc_p( new percolatorInNs::occurence (*i,flankN, flankC)  );
    psm_p->occurence().push_back(oc_p);
  }
  
  /*std::cerr << " Saved PSM " << psmid << " " << isDecoy << " " << observed_mass << " " << theoretic_mass 
	    << " " << charge << " " << peptideSeqWithFlanks << " " << proteinIds.front() << " " << deltaCN << " " << xCorr << std::endl;*/
  
  database->savePsm(useScanNumber, psm_p);
  
  return;
}