/**
 * 
 */
package openacc.analysis;

import java.util.List;

import openacc.hir.ACCAnnotation;
import cetus.analysis.AnalysisPass;
import cetus.analysis.ArrayPrivatization;
import cetus.analysis.DDTDriver;
import cetus.analysis.LoopParallelizationPass;
import cetus.analysis.Reduction;
import cetus.hir.Annotatable;
import cetus.hir.CetusAnnotation;
import cetus.hir.IRTools;
import cetus.hir.Program;
import cetus.transforms.IVSubstitution;
import cetus.transforms.TransformPass;
import cetus.exec.Driver;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class AccParallelization extends AnalysisPass {
	private int optionLevel;
	private boolean IRSymbolOnly = true;
	
	/**
	 * @param program
	 */
	public AccParallelization(Program program, int option, boolean IRSymOnly) {
		super(program);
		optionLevel = option;
		IRSymbolOnly = IRSymOnly;
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return "[AccParallelization]";
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		if( optionLevel <= 0 ) {
			return;
		}
		//Run missing prerequisite passes necessary for LoopParallelizationPass.
		//[DEBUG] Temporarily disabled since this may change OpenACC work-sharing
		//loops, resulting in changes in the overall semantics.
/*        if (Driver.getOptionValue("induction") == null) {
			Driver.setOptionValue("induction", "1");
            TransformPass.run(new IVSubstitution(program));
        }*/
        if (Driver.getOptionValue("privatize") == null) {
        	//This pass will be called before AccPrivatization pass.
 /*       	String value = Driver.getOptionValue("AccPrivatization");
        	if( (value == null) || (Integer.valueOf(value).intValue() == 0) ) {
        		Driver.setOptionValue("privatize", "1");
        		AnalysisPass.run(new ArrayPrivatization(program));
        	}*/
        	String value = Driver.getOptionValue("AccPrivatization");
        	if( (value == null) || value.equals("0") ) {
        		value = "1";
        	}
        	Driver.setOptionValue("privatize", value);
        	AnalysisPass.run(new ArrayPrivatization(program));
        }
        if (Driver.getOptionValue("ddt") == null) {
        	Driver.setOptionValue("ddt", "1");
            AnalysisPass.run(new DDTDriver(program));
        }
        if (Driver.getOptionValue("reduction") == null) {
        	//This pass will be called before AccReduction pass.
/*        	String value = Driver.getOptionValue("AccReduction");
        	if( (value == null) || (Integer.valueOf(value).intValue() == 0) ) {
        		Driver.setOptionValue("reduction", "1");
        		AnalysisPass.run(new Reduction(program));
        	}*/
        	String value = Driver.getOptionValue("AccReduction");
        	if( (value == null) || value.equals("0") ) {
        		value = "1";
        	}
        	Driver.setOptionValue("reduction", value);
        	AnalysisPass.run(new Reduction(program));
        }
        //Run the main parallelization loop.
        if( optionLevel == 1 ) {
        	Driver.setOptionValue("parallelize-loops", "2");
        } else if( optionLevel == 2 ) {
        	Driver.setOptionValue("parallelize-loops", "4");
        } else if( optionLevel <= 4 ){
        	Driver.setOptionValue("parallelize-loops", Integer.toString(optionLevel));
        } else {
        	return;
        }
        //Annotate independent clause to OpenACC loop directive if it does not have any
        //work-sharing clauses or seq/independent clause.
        AnalysisPass.run(new LoopParallelizationPass(program));
        
        List<ACCAnnotation>  cRegionAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
        if( cRegionAnnots != null ) {
        	for( ACCAnnotation cAnnot : cRegionAnnots ) {
        		Annotatable at = cAnnot.getAnnotatable();
        		List<ACCAnnotation> loopAnnots = AnalysisTools.ipCollectPragmas(at, ACCAnnotation.class, "loop", null); 
        		if( (loopAnnots != null) && (!loopAnnots.isEmpty()) ) {
        			for( ACCAnnotation lAnnot : loopAnnots ) {
        				//Do not modify user-provided information.
        				if( lAnnot.containsKey("gang") || lAnnot.containsKey("worker") || 
        						lAnnot.containsKey("vector") || lAnnot.containsKey("seq") || lAnnot.containsKey("independent") ) {
        					continue;
        				} else {
        					Annotatable lat = lAnnot.getAnnotatable();
        					CetusAnnotation cetusAnnot = lat.getAnnotation(CetusAnnotation.class, "parallel");
        					if( cetusAnnot != null ) {
        						lAnnot.put("independent", "_clause");
        						//lAnnot.remove("auto");
        					}
        				}
        			}
        		} else {
        			List<CetusAnnotation> cetusAnnots = AnalysisTools.ipCollectPragmas(at, CetusAnnotation.class, "parallel", null); 
        			if( cetusAnnots != null ) {
        				for( CetusAnnotation cetusAnnot : cetusAnnots ) {
        					Annotatable cat = cetusAnnot.getAnnotatable();
        					if( at.equals(cat) ) {
        						cAnnot.put("loop", "_directive");
        						cAnnot.put("independent", "_clause");
        					} else {
        						ACCAnnotation aAnnot = new ACCAnnotation("loop", "_directive");
        						aAnnot.put("independent", "_clause");
        						cat.annotate(aAnnot);
        					}
        				}
        			}
        		}
        	}
		}
	}

}
