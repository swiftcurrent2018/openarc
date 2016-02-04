/**
 * 
 */
package openacc.transforms;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.hir.Annotatable;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.DataFlowTools;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FunctionCall;
import cetus.hir.IRTools;
import cetus.hir.IntegerLiteral;
import cetus.hir.Loop;
import cetus.hir.NameID;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import cetus.transforms.TransformPass;
import openacc.hir.ACCAnnotation;
import openacc.analysis.ACCAnalysis;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *
 */
public class PipeTransformation extends TransformPass {
	private boolean enablePipeTransformation = false;
	private long pipe_async_offset = 1000;

	/**
	 * @param program
	 */
	public PipeTransformation(Program program, boolean enablePipeTr) {
		super(program);
		enablePipeTransformation = enablePipeTr;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return "[PipeTransformation]";
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		List<ACCAnnotation>  pipeAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "pipe");
		for( ACCAnnotation tAnnot : pipeAnnots ) {
			Annotatable pAt = tAnnot.getAnnotatable();
			ACCAnalysis.updateSymbolsInACCAnnotations(pAt, null);
			List<ACCAnnotation>  computeAnnots = AnalysisTools.ipCollectPragmas(pAt, ACCAnnotation.class, 
					ACCAnnotation.pipeIOClauses, false, null);
			if( tAnnot.containsKey("declare") ) {
				Procedure proc = IRTools.getParentProcedure(pAt);
				if( proc != null ) {
					//Declare directive in a procedure.
					computeAnnots = AnalysisTools.ipCollectPragmas(proc.getBody(), ACCAnnotation.class, 
							ACCAnnotation.pipeIOClauses, false, null);
				} else {
					computeAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.pipeIOClauses, false);
				}
			}
			if( tAnnot.containsKey("pipe") ) {
				if( enablePipeTransformation ) {
					Set<Annotatable> visitedRegions = new HashSet<Annotatable>();
					long i=pipe_async_offset;
					for( ACCAnnotation cAnnot : computeAnnots ) {
						Annotatable cAt = cAnnot.getAnnotatable();
						if( !visitedRegions.contains(cAt) ) {
							visitedRegions.add(cAt);
							ACCAnalysis.updateSymbolsInACCAnnotations(cAt, null);
							Map<Expression, Set<Integer>> useExpMap = DataFlowTools.getUseMap(cAt);
							Map<Expression, Set<Integer>> defExpMap = DataFlowTools.getDefMap(cAt);
							Map<Symbol, Set<Integer>> useSymMap = DataFlowTools.convertExprMap2SymbolMap(useExpMap);
							Map<Symbol, Set<Integer>> defSymMap = DataFlowTools.convertExprMap2SymbolMap(defExpMap);
							Set<Symbol> useSymSet = useSymMap.keySet();
							Set<Symbol> defSymSet = defSymMap.keySet();
							int useCnt = 0;
							int defCnt = 0;
							ACCAnnotation pipeAnnot = cAt.getAnnotation(ACCAnnotation.class, "pipein");
							if( pipeAnnot != null ) {
								Set<SubArray> pipeSubArrays = pipeAnnot.get("pipein");
								Set<Symbol> pipeSymSet = AnalysisTools.subarraysToSymbols(pipeSubArrays, true);
								for(Symbol pSym : pipeSymSet ) {
									useCnt = 0;
									defCnt = 0;
									if( useSymSet.contains(pSym) ) {
										useCnt = useSymMap.get(pSym).size();
									}
									if( defSymSet.contains(pSym) ) {
										defCnt = defSymMap.get(pSym).size();
									}
									if( defCnt > 0 ) {
										Tools.exit("[ERROR in PipeTransformation] the variable (" + pSym.getSymbolName() + 
												") in the pipein clause should be read-only but modified in the following region; exit!\n"
												+ "ACC annotation: " + pipeAnnot + AnalysisTools.getEnclosingContext(cAt));
									}
									if( useCnt > 1 ) {
										Tools.exit("[ERROR in PipeTransformation] the variable (" + pSym.getSymbolName() + 
												") in the pipein clause should be read only once in a target region; exit!\n"
												+ "ACC annotation: " + pipeAnnot + AnalysisTools.getEnclosingContext(cAt));
									}
								}
							}
							pipeAnnot = cAt.getAnnotation(ACCAnnotation.class, "pipeout");
							if( pipeAnnot != null ) {
								Set<SubArray> pipeSubArrays = pipeAnnot.get("pipeout");
								Set<Symbol> pipeSymSet = AnalysisTools.subarraysToSymbols(pipeSubArrays, true);
								for(Symbol pSym : pipeSymSet ) {
									useCnt = 0;
									defCnt = 0;
									if( useSymSet.contains(pSym) ) {
										useCnt = useSymMap.get(pSym).size();
									}
									if( defSymSet.contains(pSym) ) {
										defCnt = defSymMap.get(pSym).size();
									}
									if( useCnt > 0 ) {
										Tools.exit("[ERROR in PipeTransformation] the variable (" + pSym.getSymbolName() + 
												") in the pipeout clause should be write-only but read in the following region; exit!\n"
												+ "ACC annotation: " + pipeAnnot + AnalysisTools.getEnclosingContext(cAt));
									}
									if( defCnt > 1 ) {
										Tools.exit("[ERROR in PipeTransformation] the variable (" + pSym.getSymbolName() + 
												") in the pipeout clause should be modified only once in a target region; exit!\n"
												+ "ACC annotation: " + pipeAnnot + AnalysisTools.getEnclosingContext(cAt));
									}
								}
							}
							ACCAnnotation asyncAnnot = cAt.getAnnotation(ACCAnnotation.class, "async");
							if( asyncAnnot != null ) {
								Expression asyncExp = (Expression)asyncAnnot.get("async");
								if( asyncExp instanceof IntegerLiteral ) {
									long asyncID = ((IntegerLiteral)asyncExp).getValue() + i;
									asyncAnnot.put("async", new IntegerLiteral(asyncID));
								} else {
									asyncAnnot.put("async", new BinaryExpression(asyncExp.clone(), BinaryOperator.ADD, new IntegerLiteral(i)));
								}
							} else {
								cAnnot.put("async", new IntegerLiteral(i));
							}
							i++;
						}
					}
				} else {
					Object tValue = tAnnot.remove("pipe");
					Set<SubArray> createSet = tAnnot.get("create");
					if( createSet == null ) {
						createSet = new HashSet<SubArray>();
						tAnnot.put("create", createSet);
					}
					createSet.addAll((Set<SubArray>)tValue);
					Set<Annotatable> visitedRegions = new HashSet<Annotatable>();
					for( ACCAnnotation cAnnot : computeAnnots ) {
						Annotatable ttAt = cAnnot.getAnnotatable();
						if( !visitedRegions.contains(ttAt) ) {
							visitedRegions.add(ttAt);
							if( cAnnot.containsKey("pipein") ) {
								Object cValue = cAnnot.remove("pipein");
								Set<SubArray> presentSet = cAnnot.get("present");
								if( presentSet == null ) {
									presentSet = new HashSet<SubArray>();
									cAnnot.put("present", presentSet);
								}
								presentSet.addAll((Set<SubArray>)cValue);
							}
							if( cAnnot.containsKey("pipeout") ) {
								Object cValue = cAnnot.remove("pipeout");
								Set<SubArray> presentSet = cAnnot.get("present");
								if( presentSet == null ) {
									presentSet = new HashSet<SubArray>();
									cAnnot.put("present", presentSet);
								}
								presentSet.addAll((Set<SubArray>)cValue);
							}
						}
					}
				}
			}
		}
	}

}
