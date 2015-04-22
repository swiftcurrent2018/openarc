/**
 * 
 */
package openacc.transforms;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;

import cetus.hir.Annotatable;
import cetus.hir.AnnotationStatement;
import cetus.hir.ArraySpecifier;
import cetus.hir.CompoundStatement;
import cetus.hir.DataFlowTools;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Loop;
import cetus.hir.NameID;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.StringLiteral;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import cetus.hir.Typecast;
import cetus.transforms.TransformPass;

/**
 * This pass add profile-related runtime APIs according to the inserted profile directives.
 * This transformation is based on the following assumptions:
 * 1) profile region directive can be annotated to any structured blocks.
 * 2) profile track directive can be annotated only to loop structures that are also annotated with
 * profile region directive.
 * 3) profile event directives are standalone, but they can exist only within structured blocks annotated
 * with profile track directive.
 *     - Due to 2), current implementation allows profile event directives only within loop structures.
 * 4) Current implementation assumes that all profile event directives within a profile region are lexically
 * included within the profile region.
 *     - To avoid this limit, input program should be inlined using tinline OpenARC commandline option, or 
 *     this pass should be extended with interprocedural analysis/transformation capability.
 *     (Main issues in supporting profile measure directives across function boundary is to make loop variables 
 *     visible to the the profile measure directive in (nestedly) called functions.)
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class CustomProfilingTransformation extends TransformPass {
	static public String mode_all = "all";
	static public String mode_occupancy = "occupancy";
	static public String mode_throughput = "throughput";
	static public String mode_memorytransfer = "memorytransfer";
	static public String HINoInduction = new String("HINOINDUCTION");
	static public Expression booleanTrue = new IntegerLiteral(1);
	static public Expression booleanFalse = new IntegerLiteral(0);
	private boolean IRSymbolOnly;
	private String passName = "[CustomProfilingTransformation]";
	private String defaultProfileMode = mode_all;
	private IntegerLiteral defaultVerbosity = new IntegerLiteral(0);
	private String profLoopNameBase = "__HIProfLoopName";

	/**
	 * @param program
	 */
	public CustomProfilingTransformation(Program program, boolean IRSymOnly) {
		super(program);
		IRSymbolOnly = IRSymOnly;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return passName;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		//Step1: Add profile header include (#include "profile.h"), and add HI_profile_init() and HI_profile_shutdown() calls.
		//       ====> This will be done by ACC2GPUTranslator (ACC2CUDATranslator and ACC2OpenCLTranslator).
		
		//Step2: Add profile APIs for each profile directive.
		List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);
		List<ARCAnnotation> profileAnnots = IRTools.collectPragmas(program, ARCAnnotation.class, "profile");
		if( (profileAnnots != null) && (!profileAnnots.isEmpty()) ) {
			Set<String> modeSet = new HashSet<String>();
			Set<Expression> eventSet = new HashSet<Expression>();
			int counter = 0;
			for(ARCAnnotation profAnnot : profileAnnots) {
				Statement at = (Statement)profAnnot.getAnnotatable();
				CompoundStatement pStmt = (CompoundStatement)at.getParent();
				boolean isComputeRegion = false;
				Procedure cProc = IRTools.getParentProcedure(at);
				StringLiteral label = null;
				Expression profcond = null;
				Expression verbosity = defaultVerbosity.clone();
				Expression induction = null;
				modeSet.clear();
				eventSet.clear();
				if( at.containsAnnotation(ACCAnnotation.class, "kernels") 
						|| at.containsAnnotation(ACCAnnotation.class, "parallel") ) {
					isComputeRegion = true;
				}
				if( profAnnot.containsKey("label") ) {
					label = new StringLiteral((String)profAnnot.get("label"));
				}
				if( profAnnot.containsKey("profcond") ) {
					profcond = profAnnot.get("profcond");
				}
				if( profAnnot.containsKey("verbosity") ) {
					verbosity = profAnnot.get("verbosity");
				}
				if( profAnnot.containsKey("induction") ) {
					induction = profAnnot.get("induction");
				}
				if( profAnnot.containsKey("mode") ) {
					Set<String> strSet = profAnnot.get("mode");
					modeSet.addAll(strSet);
				}
				if( modeSet.isEmpty() ) {
					modeSet.add(defaultProfileMode); //add default mode if user didn't provide anything.
				}
				if( profAnnot.containsKey("event") ) {
					eventSet.addAll((Set<Expression>)profAnnot.get("event"));
				}
				if( profAnnot.containsKey("region") ) { //profile region
					//If label is missing, assign a unique label.
					if( label == null ) {
						label = new StringLiteral(cProc.getSymbolName() + "_pregion" + counter++);
					}
					
					//Step2-1: Add HI_profile_track() call before this region.
					//e.g., HI_profile_track("mylabel", "instruction(0) occupancy(0)", 
					//"HINOINDUCTION HINOINDUCTION", 1);
					FunctionCall fCall = new FunctionCall(new NameID("HI_profile_track"));
					fCall.addArgument(label.clone());
					StringBuilder str1 = new StringBuilder(80);
					StringBuilder str2 = new StringBuilder(80);
					boolean isFirst = true;
					for( String tmode : modeSet ) {
						if( isFirst ) {
							isFirst = false;
						} else {
							str1.append(" ");
							str2.append(" ");
						}
						str1.append(tmode + "(" + verbosity + ")");
						str2.append(HINoInduction);
					}
					fCall.addArgument(new StringLiteral(str1.toString()));
					fCall.addArgument(new StringLiteral(str2.toString()));
					fCall.addArgument(booleanTrue.clone());
					Statement fCallStmt = new ExpressionStatement(fCall);
					pStmt.addStatementBefore(at, fCallStmt);
					
					//Step2-2: Add HI_profile_start() before profile region and HI_profile_stop() 
					//         after the region.
					fCall = new FunctionCall(new NameID("HI_profile_start"));
					fCall.addArgument(label.clone());
					fCallStmt = new ExpressionStatement(fCall);
					pStmt.addStatementBefore(at, fCallStmt);
					if( isComputeRegion ) {
						//Add device synchronization API before HI_profile_stop() call.
						//It seems that this is not needed in the new ACC2GPUTranslator, since it always inserts
						//HI_synchronize() for synchronous kernel calls.
					}
					fCall = new FunctionCall(new NameID("HI_profile_stop"));
					fCall.addArgument(label.clone());
					fCallStmt = new ExpressionStatement(fCall);
					pStmt.addStatementAfter(at, fCallStmt);
					
					//Step2-3: If the mode clause has memorytransfer argument, it should be handled by
					//later ACC2GPUTranslator pass.
					
				} else if( profAnnot.containsKey("track") ) { //profile track
					//profLoopName should be unique, due to possible nested tracking.
					String profLoopNameStr = null;
					//If label is missing, assign a unique label.
					if( label == null ) {
						profLoopNameStr = profLoopNameBase + counter;
						label = new StringLiteral(cProc.getSymbolName() + "_ptrack" + counter++);
					} else {
						profLoopNameStr = profLoopNameBase + counter++;
					}
					//Create a local variable to hold profile Loop name for each track directive.
					List<Specifier> specs = new ArrayList<Specifier>(1);
					specs.add(Specifier.CHAR);
					ArraySpecifier aspec = new ArraySpecifier(new IntegerLiteral(256));
					Identifier profLoopName = SymbolTools.getArrayTemp(cProc.getBody(), specs, aspec, profLoopNameStr);
					//If induction variable is missing, error.
					if( induction == null ) {
						Tools.exit("[ERROR in CustomProfilingTransformation()] Profile track directive should have an induction clause; exit.\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + profAnnot + "\n");
					}
					CompoundStatement loopBody = null;
					boolean isForLoop = false;
					if( at instanceof Loop ) {
						loopBody = (CompoundStatement)((Loop)at).getBody();
						if( at instanceof ForLoop ) {
							isForLoop = true;
						}
					} else {
						Tools.exit("[ERROR in CustomProfilingTransformation()] Profile track directive is allowed only to loops; exit.\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + profAnnot + "\n");
					}
					//Step2-1: Add HI_profile_track() call before this region.
					//e.g., HI_profile_track("track1", "occupancy error", "iter iter", 0);
					FunctionCall fCall = new FunctionCall(new NameID("HI_profile_track"));
					fCall.addArgument(label.clone());
					StringBuilder str1 = new StringBuilder(80);
					StringBuilder str2 = new StringBuilder(80);
					boolean isFirst = true;
					for( String tmode : modeSet ) {
						if( isFirst ) {
							isFirst = false;
						} else {
							str1.append(" ");
							str2.append(" ");
						}
						str1.append(tmode);
						if( induction == null ) {
							str2.append(HINoInduction);
						} else {
							str2.append(induction.clone());
						}
					}
					for( Expression tExp : eventSet ) {
						if( isFirst ) {
							isFirst = false;
						} else {
							str1.append(" ");
							str2.append(" ");
						}
						str1.append(tExp.toString());
						if( induction == null ) {
							str2.append(HINoInduction);
						} else {
							str2.append(induction.clone());
						}
						
					}
					fCall.addArgument(new StringLiteral(str1.toString()));
					fCall.addArgument(new StringLiteral(str2.toString()));
					fCall.addArgument(booleanFalse.clone());
					Statement fCallStmt = new ExpressionStatement(fCall);
					pStmt.addStatementBefore(at, fCallStmt);
					
					//Step2-2: Add HI_profile_start() at the beginning of the attached loop and HI_profile_stop() 
					//         at the end of the attached loop
					//e.g., if(iter%2==0) {
					//          sprintf(__HIProfLoopName, "track1: iter index: ", iter);
					//          HI_profile_start(__HIProfLoopName);
					//      }
					fCall = new FunctionCall(new NameID("sprintf"));
					fCall.addArgument(profLoopName.clone());
					fCall.addArgument(new StringLiteral("%s%d"));
					str1 = new StringBuilder(80);
					str1.append(label.getValue());
					str1.append(": ");
					str1.append(induction.toString());
					str1.append(" index: ");
					fCall.addArgument(new StringLiteral(str1.toString()));
					fCall.addArgument(induction.clone());
					fCallStmt = new ExpressionStatement(fCall);
					fCall = new FunctionCall(new NameID("HI_profile_start"));
					fCall.addArgument(profLoopName.clone());
					Statement fCallStmt2 = new ExpressionStatement(fCall);
					Statement nonDeclStmt = IRTools.getFirstNonDeclarationStatement(loopBody);
					if( profcond == null ) {
						if( nonDeclStmt == null ) {
							loopBody.addStatement(fCallStmt);
							loopBody.addStatement(fCallStmt2);
						} else {
							loopBody.addStatementBefore(nonDeclStmt, fCallStmt);
							loopBody.addStatementBefore(nonDeclStmt, fCallStmt2);
						}
					} else {
						CompoundStatement ifBody = new CompoundStatement();
						ifBody.addStatement(fCallStmt);
						ifBody.addStatement(fCallStmt2);
						IfStatement ifStmt = new IfStatement(profcond.clone(), ifBody);
						loopBody.addStatementBefore(nonDeclStmt, ifStmt);
					}
					//e.g., if(iter%2==0) {
					//          HI_profile_stop(__HIProfLoopName);
					//      }
					fCall = new FunctionCall(new NameID("HI_profile_stop"));
					fCall.addArgument(profLoopName.clone());
					fCallStmt = new ExpressionStatement(fCall);
					Statement refStmt = null;
					if( !isForLoop && (induction != null) ) {
						//Find the first statemenent where the iteration variable is modified.
						for( Traversable child : loopBody.getChildren() ) {
							Set<Expression> defSet = DataFlowTools.getDefSet(child);
							if( defSet.contains(induction)) {
								refStmt = (Statement)child;
								break;
							}
						}
					}
					Statement stopStmt = null;
					if( profcond == null ) {
						stopStmt = fCallStmt;
					} else {
						CompoundStatement ifBody = new CompoundStatement();
						ifBody.addStatement(fCallStmt);
						IfStatement ifStmt = new IfStatement(profcond.clone(), ifBody);
						stopStmt = ifStmt;
					}
					if( refStmt == null ) {
						loopBody.addStatement(stopStmt);
					} else {
						loopBody.addStatementBefore(refStmt, stopStmt);
					}
				} else if( profAnnot.containsKey("measure") ) { //profile measure
					//If label is missing, assign a unique label.
					if( label == null ) {
						label = new StringLiteral(cProc.getSymbolName() + "_pmeasure" + counter++);
					}
					if( eventSet.isEmpty() ) {
						Tools.exit("[ERROR in CustomProfilingTransformation()] Profile measure directive has no argument " +
								"in the event clause; exit.\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + profAnnot + "\n");
					}
					StringBuilder str1 = null;
					FunctionCall fCall = null;
					//Step2-1: Add HI_profile_track() call before this region for each event argument.
					//e.g., HI_profile_track("track1: measure1", "error", "iter", 0);
					for( Expression tExp : eventSet ) {
						fCall = new FunctionCall(new NameID("HI_profile_track"));
						//Find a parent region containing track directive whose event clause contains events in this
						//measure directive.
						Statement pRegion = null;
						CompoundStatement gpRegion = null;
						String pLabel = null;
						Traversable tt = at.getParent();
						while ((tt != null) && (tt instanceof Annotatable)) {
							Annotatable atObj = (Annotatable)tt;
							ARCAnnotation tAnnot = atObj.getAnnotation(ARCAnnotation.class, "track");
							if( tAnnot != null ) {
								if( tAnnot.containsKey("event") ) {
									Set<Expression> tExpSet = tAnnot.get("event");
									if( tExpSet.contains(tExp) ) {
										pRegion = (Statement)atObj;
										pLabel = tAnnot.get("label");
										break;
									}
								}
							}
							tt = tt.getParent();
						}
						//[TODO] If parent track directive is not found, it can be searched interprocedurally.
						
						if( (pRegion != null) && (pLabel != null) ) {
							gpRegion = (CompoundStatement)pRegion.getParent();
							str1 = new StringBuilder(80);
							str1.append(pLabel);
							str1.append(": ");
							str1.append(label.getValue());
							fCall.addArgument(new StringLiteral(str1.toString()));
							fCall.addArgument(new StringLiteral(tExp.toString()));
							if( induction == null ) {
								fCall.addArgument(new StringLiteral(HINoInduction));
							} else {
								fCall.addArgument(new StringLiteral(induction.toString()));
							}
							fCall.addArgument(booleanFalse.clone());
							Statement fCallStmt = new ExpressionStatement(fCall);
							gpRegion.addStatementBefore(pRegion, fCallStmt);
						} else {
							Tools.exit("[ERROR in CustomProfilingTransformation()] Can not find a parent profile track directive containing " +
									"event argument, " + tExp.toString() + ", for the following profile measure directive:\n" +
									profAnnot + 
									"\nEnclosing procedure: " + cProc.getSymbolName() + "\n");
						}
					}
					//Step2-1: Add HI_profile_measure_userevent() call.
					//e.g., HI_profile_measure_userevent("measure1: error", error);
					for( Expression tExp : eventSet ) {
						fCall = new FunctionCall(new NameID("HI_profile_measure_userevent"));
						str1 = new StringBuilder(80);
						str1.append(label.getValue());
						str1.append(": ");
						str1.append(tExp.toString());
						fCall.addArgument(new StringLiteral(str1.toString()));
						if( (tExp instanceof Identifier) && 
								((Identifier)tExp).getSymbol().getTypeSpecifiers().contains(Specifier.DOUBLE) ) {
							fCall.addArgument(tExp.clone());
						} else {
							List specs = new ArrayList(1);
							specs.add(Specifier.DOUBLE);
							fCall.addArgument(new Typecast(specs, tExp.clone()));
						}
						Statement fCallStmt = new ExpressionStatement(fCall);
						if( profcond == null ) {
							pStmt.addStatementAfter(at, fCallStmt);
						} else {
							CompoundStatement ifBody = new CompoundStatement();
							ifBody.addStatement(fCallStmt);
							IfStatement ifStmt = new IfStatement(profcond.clone(), ifBody);
							pStmt.addStatementAfter(at, ifStmt);
						}
					}
				} else {
					Tools.exit("[ERROR in CustomProfilingTransformation()] Profile directive should have one subdirective: " +
							"region, track, or measure.\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + profAnnot + "\n");
				}
			}
			
		}

	}

}
