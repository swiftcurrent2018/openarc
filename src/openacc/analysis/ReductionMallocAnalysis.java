/**
 * 
 */
package openacc.analysis;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

import openacc.analysis.AnalysisTools;
import openacc.hir.KernelFunctionCall;
import openacc.hir.ReductionOperator;
import openacc.hir.ACCAnnotation;

import cetus.analysis.AnalysisPass;
import cetus.analysis.LoopTools;
import cetus.hir.*;

/**
 * This analysis finds the optimal malloc point for reduction-related GPU
 * variables, and the optimal point is annotated with a rcreate(redSymList) clause 
 * in internal annotations
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *
 */
public class ReductionMallocAnalysis extends AnalysisPass {
	private boolean IRSymbolOnly = true;
	/**
	 * @param program
	 */
	public ReductionMallocAnalysis(Program program, boolean IRSymOnly) {
		super(program);
		IRSymbolOnly = IRSymOnly;
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return "[ReductionMallocAnalysis]";
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		List<ACCAnnotation>  cRegionAnnots = 
			AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
		//DEBUG: This pass assumes that kernels regions are split into each kernels loops.
		for( ACCAnnotation cAnnot : cRegionAnnots ) {
			Annotatable at = cAnnot.getAnnotatable();
			if( at.containsAnnotation(ACCAnnotation.class, "reduction") ) {
				//Find reduction symbols.
				Set<Symbol> reductionSymbols = AnalysisTools.getReductionSymbols(at, IRSymbolOnly);
				//Find symbols used in iterspace expressions.
				Set<Symbol> itrSymbols = new HashSet<Symbol>();
				List<ACCAnnotation> itsAnnots = IRTools.collectPragmas(at, ACCAnnotation.class, "iterspace");
				for( ACCAnnotation itsAnnot : itsAnnots ) {
					Expression exp = (Expression)itsAnnot.get("iterspace");
					itrSymbols.addAll(SymbolTools.getAccessedSymbols(exp));
				}
				///////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//Find optimal reduction-malloc point.                                                                       //
				//If iteration space size is not changed, the reduction-malloc point can be moved up to enclosing statement. //
				//FIXME: If reduction symbol is local, the malloc point can not go beyond the symbol scope, but not checked  //
				//in the current implementation.
				///////////////////////////////////////////////////////////////////////////////////////////////////////////////
				Traversable refstmt = at; //Refer to the optimal reduction-malloc insertion point.
				Traversable tChild = at;
				Traversable tt = tChild.getParent();
				while( (tt != null) && !(tt instanceof Procedure) ) {
					if( !(tt instanceof CompoundStatement) ) {
						tChild = tt;
						tt = tt.getParent();
						continue;
					}
					//TODO: for now, optimal point is searched within a procedure boundary, but it can be further optimized
					//across procedure boundary.
					if( tt.getParent() instanceof Procedure ) {
						refstmt = tChild;
						break;
					}
					Set<Symbol> defSet = DataFlowTools.getDefSymbol(tt);
					boolean gridSizeNotChanged = false;
					defSet.retainAll(itrSymbols);
					if( defSet.isEmpty() ) {
						gridSizeNotChanged = true;
						for( Traversable t : tt.getChildren() ) {
							if( (t != tChild) && (t instanceof Annotatable) ) {
								if( ((Annotatable)t).containsAnnotation(ACCAnnotation.class, "reduction") ) {
									Set<Symbol> tRedSyms = AnalysisTools.getReductionSymbols((Annotatable)t, IRSymbolOnly);
									tRedSyms.retainAll(reductionSymbols);
									if( tRedSyms.size() != 0 ) {
										//TODO: too conservative; fine-control per symbol will be more efficient.
										gridSizeNotChanged = false;
										break;
									}
								}
							}
						}
						if( gridSizeNotChanged ) {
							//////////////////////////////////////////////////////////////////////////
							//If a function called in the tt has compute regions, conservatively    //
							//assume that the grid-size may be changed in the called function.      //
							//FIXME: To be precise, we should check whether itrSymbols are modified //
							//       or not interprocedurally, but not checked here.                //
							//////////////////////////////////////////////////////////////////////////
							List<FunctionCall> callList = IRTools.getFunctionCalls(tt);
							for( FunctionCall fcall : callList ) {
								if( (fcall instanceof KernelFunctionCall) || AnalysisTools.isCudaCall(fcall) 
										|| StandardLibrary.contains(fcall) ) {
									continue;
								}
								Procedure cProc = fcall.getProcedure();
								if( cProc == null ) {
									continue;
								} else {
									List<ACCAnnotation> cList = 
										AnalysisTools.collectPragmas(cProc, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
									if( cList.size() > 0 ) {
										gridSizeNotChanged = false;
										break;
									}
								}
							}
						}
					}
					if( gridSizeNotChanged ) {
						Traversable tGrandChild = tChild;
						tChild = tt;
						tt = tt.getParent();
						if( tt instanceof IfStatement ) {
							refstmt = tt;
						} else if( (tt instanceof DoLoop) || (tt instanceof WhileLoop) ) {
							refstmt = tt;
						} else if( tt instanceof ForLoop ) {
							Expression iVar = LoopTools.getIndexVariable((ForLoop)tt);
							Set<Symbol> iSyms = SymbolTools.getAccessedSymbols(iVar);
							iSyms.retainAll(itrSymbols);
							if( iSyms.isEmpty() ) {
								refstmt = tt;
							} else {
								refstmt = tGrandChild;
								break;
							}
						} else {
							refstmt = tChild;
						}
					} else {
						break;
					}
				}
				//Add rcreate clauses to the refstmt.
				Annotatable rAt = (Annotatable)refstmt;
				Annotation iAnnot = rAt.getAnnotation(ACCAnnotation.class, "internal");
				if( iAnnot == null ) {
					iAnnot = new ACCAnnotation("internal", "_directive");
					iAnnot.setSkipPrint(true);
					at.annotate(iAnnot);
				}
				Set<Symbol> rCreateSet = iAnnot.get("rcreate");
				if( rCreateSet == null ) {
					rCreateSet = new HashSet<Symbol>();
					iAnnot.put("rcreate", rCreateSet);
				}
				rCreateSet.addAll(reductionSymbols);
			}
		}
	}
	
}
