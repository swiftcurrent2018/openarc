/**
 * 
 */
package openacc.transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;
import openacc.analysis.AnalysisTools;
import openacc.analysis.ConfList;
import openacc.analysis.SubArray;
import openacc.analysis.SubArrayWithConf;
import cetus.analysis.LoopTools;
import cetus.hir.*;
import cetus.transforms.TransformPass;

/**
 * Example transform directives: 
 *     #pragma openarc transform transpose(A[0:N1][0:N2]::[2,1], B[0:N1][0:N2]::[2,1])
 * <p>
 *     #pragma openarc transform redim(A[0:4*X*Y*Z]::[Z,Y,X,4], B[0:4:X*Y*Z]::[4,Z,Y,X])
 * <p>
 *     #pragma openarc transform expand(A[0:N1][0:N2], B[0:N1][0:N2])
 * <p>
 *     #pragma openarc transform redim_transpose(A[0:4*X*Y*Z]::[Z,Y,X,4]::[2,1])
 * <p>
 *     #pragma openarc transform expand_transpose(A[0:N1][0:N2]::[2,1])
 * <p>
 *     #pragma openarc transform transpose_expand(A[0:N1][0:N2]::[2,1])
 * <p>
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *         
 *         Tetsuya Hoshino <hoshino@matsulab.is.titech.ac.jp>
 *         Tokyo Institute of Technology
 *
 */
public class DataLayoutTransform extends TransformPass {
	private static String pass_name = "[DataLayoutTransform]";
	private boolean IRSymbolOnly = true;

	/**
	 * @param program
	 */
	public DataLayoutTransform(Program program, boolean IRSymOnly) {
		super(program);
		IRSymbolOnly = IRSymOnly;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return pass_name;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */

	@Override

	public void start() {
		//for each transform, 
		//step0: ([TODO]): if ifcond exists and its argument is 0, skip the current transform region. 
		//step1: If the transform region is inside of compute regions, error in current version. 
		//step2: if the transposed array isn't exist device memory, step2.1. else, step2.2
		//step2.1: Create new transposed array
		//step2.2: 
		//step3: transpose function call
		//step4: re-transpose function call
		//step5: replace the variables
		//step5.5: recalculate the index of the new array
		//step6: ([TODO]) deal with function calls
		//step7: ([TODO]) deal with copy clause and update directive in the transform region.

		List<ARCAnnotation> transformAnnots = IRTools.collectPragmas(program, ARCAnnotation.class, "transform");
		if( transformAnnots != null ) {
			List<ARCAnnotation> transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redimAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expandAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redim_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expand_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> transpose_expandAnnots = new LinkedList<ARCAnnotation>();

			for( ARCAnnotation arcAnnot : transformAnnots ) {
				if( arcAnnot.containsKey("transpose") ) {
					transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim") ) {
					redimAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand") ) {
					expandAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim_transpose") ) {
					redim_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand_transpose") ) {
					expand_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("transpose_expand") ) {
					transpose_expandAnnots.add(arcAnnot);
				}
			}


			List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);

			//////////////////////////////////////////
			// Performs "transpose" transformation. //
			//////////////////////////////////////////
			for( ARCAnnotation tAnnot : transposeAnnots ) {

				Annotatable at = tAnnot.getAnnotatable();
				Set<SubArrayWithConf> sArrayConfSet = tAnnot.get("transpose");

				Statement transposeRegion = (Statement)tAnnot.getAnnotatable();
				//				CompoundStatement cStmt = (CompoundStatement)transposeRegion.getParent();
				//				CompoundStatement cStmt = (CompoundStatement)transposeRegion.getChildren().get(0);
				CompoundStatement cStmt = null;
				Statement firstNonDeclStmt = null;
				if (transposeRegion instanceof CompoundStatement ){
					cStmt = (CompoundStatement)transposeRegion;
					firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(transposeRegion);
				} else {
					// [FIXME] is it correct?
					cStmt = new CompoundStatement();
					firstNonDeclStmt = transposeRegion;
				}



				Procedure cProc = IRTools.getParentProcedure(transposeRegion);
				CompoundStatement cBody = cProc.getBody();

				System.out.println("test");

				//step1 
				ACCAnnotation compAnnot = null;
				compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "kernels");
				if( compAnnot == null ){
					compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "parallel");
				}
				ACCAnnotation pcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(transposeRegion, ACCAnnotation.class, ACCAnnotation.computeRegions, false, funcCallList, null);
				if( pcompAnnot != null ) {
					Tools.exit("[ERROR in DataLayoutTransformation()] transform pragma can not exist inside of " +
							"any compute regions (kerenls/parallel regions):\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + tAnnot + "\n");
				}

				List<FunctionCall> funcCallList2 = IRTools.getFunctionCalls(cStmt);


				for(SubArrayWithConf sArrayConf : sArrayConfSet ) {

					//step2.0 get pragma information
					SubArray sArray = sArrayConf.getSubArray();
					//In transpose clause, each SubArrayWithConf has only on configuration list ([2,1,3]).
					//CF: redim_transpose, each SubArrayWithConf will have two configuration lists 
					//(one for redim (e.g, [X,Y,Z]), and the other for transpose(e.g., [2,1,3])).
					ConfList confL = sArrayConf.getConfList(0);
					//...
					List<Expression> startIndecies = sArray.getStartIndices();
					List<Expression> exList = sArray.getLengths();
					Expression size_1D = new IntegerLiteral(1);
					for(Expression ex : exList){
						size_1D = Symbolic.multiply(size_1D, ex);
					}
					int dimension = sArray.getArrayDimension();
					Expression sArrayNameEx = sArray.getArrayName();
					String sArrayName = sArray.getArrayName().toString();
					String transposedArrayName = "transposed__" + sArrayName;
					System.out.println("size = " + dimension + " name = " + sArrayName.toString());

					Set<Symbol> symSet = cBody.getSymbols();
					SymbolTable global_table = (SymbolTable) cProc.getParent();
					Set<Symbol> symSet2 = global_table.getSymbols();
					Symbol transpose_sym = AnalysisTools.findsSymbol(symSet, sArrayName);
					if( transpose_sym == null ){
						transpose_sym = AnalysisTools.findsSymbol(symSet2, sArrayName);
					}
					System.out.println(transpose_sym.getSymbolName());
					System.out.println(transpose_sym.getArraySpecifiers().get(0).toString());
					System.out.println(transpose_sym.getTypeSpecifiers().get(0).toString());

					/* get host variable */
					Expression hostVar = new Identifier(transpose_sym);
					//					Set<Symbol> outSymbols = new HashSet<Symbol>();


					/* New variable declaration and allocation */
					Expression newVar = null;
					// Create a new transpose-host variable.
					// The type of the transpose-host symbol should be a pointer type 
					VariableDeclarator transpose_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(transposedArrayName));
					List<Specifier> transpose_new_specs = transpose_sym.getTypeSpecifiers();
					transpose_new_specs.remove(Specifier.STATIC);
//					VariableDeclaration transpose_decl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
//							transpose_declarator);
					VariableDeclaration transpose_decl = new VariableDeclaration(transpose_new_specs, //[FIXME]
							transpose_declarator);
					newVar = new Identifier(transpose_declarator);
					Symbol transpose_dec = transpose_declarator;


					/* make transposition kernels */

					//size and rule information
					Expression sizeInfo = null;
					Expression ruleInfo = null;
					NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
					NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
					List<Expression> arryDimList = new ArrayList<Expression>();
					arryDimList.add(new IntegerLiteral(dimension));
					ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
					VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
					VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
					VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
					VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
					sizeInfo = new Identifier(sizeInfo_declarator);
					ruleInfo = new Identifier(ruleInfo_declarator);

					//transpose_kernel function for CPU
					NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
					FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
					Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
					transpose_funcCall.addArgument(newVar.clone());
					transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					transpose_funcCall.addArgument(sizeInfo.clone());
					transpose_funcCall.addArgument(ruleInfo.clone());
					transpose_funcCall.addArgument(new IntegerLiteral(dimension));

					//re_transpose_kernel function for CPU
					NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
					FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
					Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
					retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					retranspose_funcCall.addArgument(newVar.clone());
					retranspose_funcCall.addArgument(sizeInfo.clone());
					retranspose_funcCall.addArgument(ruleInfo.clone());
					retranspose_funcCall.addArgument(new IntegerLiteral(dimension));

					//declarations for device address pointer
					Expression gpuVar = null; 
					Expression gpuNewVar = null;
					String gpuVarName = "deviceAddressFor__" + sArrayName;
					String gpuNewVarName = "transposedDeviceAddressFor__" + sArrayName;
					VariableDeclarator gpuVarDeclarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(gpuVarName));
					VariableDeclarator gpuNewVarDeclarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(gpuNewVarName));
//					VariableDeclaration gpuVarDecl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
//							gpuVarDeclarator);
//					VariableDeclaration gpuNewVarDecl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
//							gpuNewVarDeclarator);
					VariableDeclaration gpuVarDecl = new VariableDeclaration(transpose_new_specs, //[FIXME]
							gpuVarDeclarator);
					VariableDeclaration gpuNewVarDecl = new VariableDeclaration(transpose_new_specs, //[FIXME]
							gpuNewVarDeclarator);
					gpuVar = new Identifier(gpuVarDeclarator);
					gpuNewVar = new Identifier(gpuNewVarDeclarator);
					cBody.addDeclaration(gpuVarDecl);
					cBody.addDeclaration(gpuNewVarDecl);

					//transpose_kernel function for GPU
					NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
					FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
					Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
					gpu_transpose_funcCall.addArgument(gpuNewVar.clone());
					gpu_transpose_funcCall.addArgument(gpuVar.clone());
					gpu_transpose_funcCall.addArgument(sizeInfo.clone());
					gpu_transpose_funcCall.addArgument(ruleInfo.clone());
					gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));

					//re_transpose_kernel function for GPU
					NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
					FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
					Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
					gpu_retranspose_funcCall.addArgument(gpuVar.clone());
					gpu_retranspose_funcCall.addArgument(gpuNewVar.clone());
					gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
					gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
					gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));


					//					CompoundStatement ifPresentTrueBodyStmt = new CompoundStatement();
					//					CompoundStatement checkPresentStmt = //new IfStatement(checkPresent, ifPresentTrueBodyStmt, ifPresentElseBodyStmt);
					//							new CompoundStatement();
					//					checkPresentStmt.addStatement(ifPresentTrueBodyStmt);
					cBody.addDeclaration(transpose_decl);

					Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
					for(Expression ex : exList){
						size = new BinaryExpression(size, BinaryOperator.MULTIPLY, ex.clone());
					}
					FunctionCall malloc_call = new FunctionCall(new NameID("malloc"));
					malloc_call.addArgument(size);
					List<Specifier> specs = new ArrayList<Specifier>(4);
					//specs.addAll(transpose_dec.getTypeSpecifiers());
					specs.addAll(transpose_dec.getTypeSpecifiers());
					Statement malloc_stmt = new ExpressionStatement(new AssignmentExpression(newVar.clone(),
							AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
					//ifPresentTrueBodyStmt.addStatement(malloc_stmt.clone());
					cStmt.addStatementBefore(firstNonDeclStmt, malloc_stmt.clone());

					//if already exist, add a data directive for new device variable
//					ACCAnnotation dataDirective = new ACCAnnotation("data", null);
//					SubArray newDeviceArray = new SubArray(newVar.clone(), new IntegerLiteral(0), size_1D.clone());
//					dataDirective.put("create", newDeviceArray.clone());
//					Statement dataDirectiveStmt = new AnnotationStatement(dataDirective);
//					CompoundStatement dataDirectiveRegion = new CompoundStatement();
//					dataDirectiveRegion.addStatement(cStmt.clone());
//					cStmt.addStatement(dataDirectiveStmt.clone());
//					cStmt.addStatement(dataDirectiveRegion.clone());
//					cStmt.addStatement(dataDirectiveStmt.clone());
//					cStmt.addStatement(dataDirectiveRegion.clone());
					//ifPresentElseBodyStmt.addStatement(cStmt.clone());
					//cStmt.addStatementBefore(firstNonDeclStmt, checkPresentStmt);
					//cStmt.swapWith(checkPresentStmt);


					//step2.1 already exist 

					//List<ARCAnnotation> trueAnnotationListARC =
					//IRTools.collectPragmas(ifPresentTrueBodyStmt, ARCAnnotation.class, "transform");
					//for(ARCAnnotation trueTransAnnot : trueAnnotationListARC){
					//						Statement trueTransposeRegion = (Statement)trueTransAnnot.getAnnotatable();
					//						CompoundStatement trueCStmt = null;
					//						Statement trueFirstNonDeclStmt = null;
					//						if (trueTransposeRegion instanceof CompoundStatement ){
					//							trueCStmt = (CompoundStatement)trueTransposeRegion;
					//							trueFirstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(trueTransposeRegion);
					//						} else {
					//							// [FIXME] is it correct?
					//							trueCStmt = new CompoundStatement();
					//							trueFirstNonDeclStmt = trueTransposeRegion;
					//						}
					List<ACCAnnotation> annotationListACC =
							AnalysisTools.collectPragmas(transposeRegion, ACCAnnotation.class);
										
					//step5 replace all variables
					Statement enclosingCompRegion = null;
					for(ACCAnnotation annot : annotationListACC)
					{
						Annotatable att = annot.getAnnotatable();
						if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
								|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
							enclosingCompRegion = (Statement)att;
							IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
						}

						for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
							//Set<SubArray> transposeSArraySet = annot.get(dataClause);
							Set<Object> transposeSArraySet = annot.get(dataClause);
							if(transposeSArraySet != null){
								System.out.println(transposeSArraySet.toString());
								//for(SubArray subArray : transposeSArraySet){
								for(Object subArray : transposeSArraySet){
									if(subArray instanceof SubArray){
										System.out.println(subArray.toString());
										System.out.println(((SubArray)subArray).getArrayName().toString());
										System.out.println(sArrayName);
										if(((SubArray)subArray).getArrayName().equals(sArrayNameEx)){
											SubArray newVarSubArray = ((SubArray)subArray).clone();
											newVarSubArray.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(newVarSubArray.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												newVarSubArray.setRange(i, ExList.get(i));
											}
											transposeSArraySet.add(newVarSubArray);
											//transposeSArraySet.remove(subArray);
											break;
										}
									}
								}
							}

						}
						for(String accinter : ACCAnnotation.internalDataClauses){  
							//Set<Symbol> transposeSArrays = annot.get(accinter);
							Set<Object> transposeSArrays = annot.get(accinter);
							//Set<SubArray> transposeSArrays = annot.get(accinter);
							if(transposeSArrays != null){
								System.out.println(accinter + " " + transposeSArrays.toString() + " " + transposeSArrays.size());
								//								for(SubArray subArray : transposeSArrays){ //[FIXME]
								for(Object subArray : transposeSArrays){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											transposeSArrays.add(transpose_dec);
											//transposeSArrays.remove(subArray);
											break;
										}
									}
								}
							}
						}
					}
					if(enclosingCompRegion == null) {
						continue;
					}


					List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
					for(Expression e : newVarExpressions){
						if(e.getParent().getClass().equals(ArrayAccess.class)){
							ArrayAccess aa = (ArrayAccess)e.getParent();
							//							System.out.println("array access name" + aa.getArrayName().toString());
							List<Expression> oldIndices = aa.getIndices();
							int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
							Expression newIndex = oldIndices.get(newIndexIndex);
							for(int i = 1; i < dimension;i++){
								newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
								newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
										BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
							}
							List<Expression> newIndices = new ArrayList<Expression>();
							newIndices.add(newIndex);
							aa.setIndices(newIndices);
						}
					}

					cStmt.addDeclaration(sizeInfo_decl.clone());
					cStmt.addDeclaration(ruleInfo_decl.clone());
					for(int i = 0;i < dimension;i++){
						Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
						Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
						cStmt.addStatementBefore(firstNonDeclStmt,sizeInfo_Statement);
						cStmt.addStatementBefore(firstNonDeclStmt,ruleInfo_Statement);
					}


					CompoundStatement presentErrorCode = new CompoundStatement();
					FunctionCall printfCall = new FunctionCall(new NameID("printf"));
					printfCall.addArgument(new StringLiteral("[ERROR] GPU memory for the host variable, "+hostVar.toString()+
							", does not exist. \\n"));
					presentErrorCode.addStatement(new ExpressionStatement(printfCall));
					printfCall = new FunctionCall(new NameID("printf"));
					printfCall.addArgument(new StringLiteral("Enclosing annotation: \\n" + tAnnot.toString() + " \\n"));//[FIXME]
					presentErrorCode.addStatement(new ExpressionStatement(printfCall));
					FunctionCall exitCall = new FunctionCall(new NameID("exit"));
					exitCall.addArgument(new IntegerLiteral(1));
					presentErrorCode.addStatement(new ExpressionStatement(exitCall));

					NameID getGpuVarAddressName = new NameID("HI_get_device_address");
					FunctionCall getGpuVarAddressFuncCall = new FunctionCall(getGpuVarAddressName);
					getGpuVarAddressFuncCall.addArgument(hostVar.clone());
					getGpuVarAddressFuncCall.addArgument(gpuVar.clone());
					Expression asyncID = new NameID("acc_async_noval");
					getGpuVarAddressFuncCall.addArgument(asyncID.clone());
					Expression getGpuVarAddressExp = new BinaryExpression(getGpuVarAddressFuncCall, BinaryOperator.COMPARE_NE, new NameID("HI_success"));
					IfStatement getGpuVarAddressIfStmt = new IfStatement(getGpuVarAddressExp.clone(), presentErrorCode.clone());
					cStmt.addStatementBefore(firstNonDeclStmt, getGpuVarAddressIfStmt.clone());


					CompoundStatement transposedPresentErrorCode = new CompoundStatement();
					FunctionCall transposedPrintfCall = new FunctionCall(new NameID("printf"));
					transposedPrintfCall.addArgument(new StringLiteral("[ERROR] GPU memory for the host variable, "+newVar.toString()+
							", does not exist. \\n"));
					transposedPresentErrorCode.addStatement(new ExpressionStatement(transposedPrintfCall));
					transposedPrintfCall = new FunctionCall(new NameID("printf"));
					transposedPrintfCall.addArgument(new StringLiteral("Enclosing annotation: \\n" + tAnnot.toString() + " \\n"));
					transposedPresentErrorCode.addStatement(new ExpressionStatement(transposedPrintfCall));
					FunctionCall transposedExitCall = new FunctionCall(new NameID("exit"));
					transposedExitCall.addArgument(new IntegerLiteral(1));
					transposedPresentErrorCode.addStatement(new ExpressionStatement(transposedExitCall));

					NameID getGpuNewVarAddressName = new NameID("HI_get_device_address");
					FunctionCall getGpuNewVarAddressFuncCall = new FunctionCall(getGpuNewVarAddressName);
					getGpuNewVarAddressFuncCall.addArgument(newVar.clone());
					getGpuNewVarAddressFuncCall.addArgument(gpuNewVar.clone());
					getGpuNewVarAddressFuncCall.addArgument(asyncID.clone());
					Expression getGpuNewVarAddressExp = new BinaryExpression(getGpuNewVarAddressFuncCall, BinaryOperator.COMPARE_NE, new NameID("HI_success"));
					IfStatement getGpuNewVarAddressIfStmt = new IfStatement(getGpuNewVarAddressExp.clone(), transposedPresentErrorCode.clone());
					cStmt.addStatementBefore(firstNonDeclStmt, getGpuNewVarAddressIfStmt.clone());		

					cStmt.addStatementBefore(firstNonDeclStmt, gpu_transpose_func_stmt);
					cStmt.addStatement(gpu_retranspose_func_stmt);
					//step free
					FunctionCall free_call = new FunctionCall(new NameID("free"));
					free_call.addArgument(newVar.clone());
					Statement free_stmt = new ExpressionStatement(free_call);
					cStmt.addStatement(free_stmt);


					//step   function calls
					for(FunctionCall func : funcCallList2){
						System.out.println(func.toString() + ", " + func.getStatement().toString());

						Traversable t = func.getParent();
						while ( !(t instanceof CompoundStatement)){
							t = t.getParent();
						}
						CompoundStatement funcParent = (CompoundStatement)t;
						funcParent.addStatementBefore(func.getStatement(), gpu_retranspose_func_stmt.clone());
						funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
					}


					//step update directive 
					for(ACCAnnotation annot : annotationListACC)
					{
						Annotatable att = annot.getAnnotatable();
						if(att.containsAnnotation(ACCAnnotation.class, "update")){
							if(att.containsAnnotation(ACCAnnotation.class, "host")){
								boolean flag = false;
								Set<Object> hostList = annot.get("host");
								for(Object subArray : hostList){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											hostList.add(transpose_dec);
											hostList.remove(subArray);
											flag = true;
											break;
										}
									}
									if(subArray instanceof SubArray){
										SubArray sa = (SubArray)subArray;
										if(sa.getArrayName().equals(sArrayNameEx)){
											sa.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(sa.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												sa.setRange(i, ExList.get(i));
											}
											flag = true;
											break;
										}
									}
								}
								if(flag){
									Traversable t = att;
									while ( !(t instanceof CompoundStatement)){
										t = t.getParent();
									}
									CompoundStatement attParent = (CompoundStatement)t;
									attParent.addStatementAfter((Statement)att, retranspose_func_stmt.clone());
								}

							}
							if(att.containsAnnotation(ACCAnnotation.class, "device")){
								boolean flag = false;
								Set<Object> deviceList = annot.get("device");
								for(Object subArray : deviceList){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											deviceList.add(transpose_dec);
											//System.out.println("transsym " + transpose_sym.getSymbolName());
											deviceList.remove(subArray);
											flag = true;
											break;
										}
									}
									if(subArray instanceof SubArray){
										SubArray sa = (SubArray)subArray;
										if(sa.getArrayName().equals(sArrayNameEx)){
											sa.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(sa.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												sa.setRange(i, ExList.get(i));
											}
											flag = true;
											break;
										}
									}
								}
								if(flag){
									Traversable t = att;
									while ( !(t instanceof CompoundStatement)){
										t = t.getParent();
									}
									CompoundStatement attParent = (CompoundStatement)t;
									attParent.addStatementBefore((Statement)att, transpose_func_stmt.clone());
								}
							}
						}
					}
					//					}

					//					//step2.2 else 
					//					List<ARCAnnotation> elseAnnotationListARC =
					//							IRTools.collectPragmas(ifPresentElseBodyStmt, ARCAnnotation.class, "transform");
					//
					//					for(ARCAnnotation elseTransAnnot : elseAnnotationListARC){
					//						Statement elseTransposeRegion = (Statement)elseTransAnnot.getAnnotatable();
					//						CompoundStatement elseCStmt = null;
					//						Statement elseFirstNonDeclStmt = null;
					//						if (elseTransposeRegion instanceof CompoundStatement ){
					//							elseCStmt = (CompoundStatement)elseTransposeRegion;
					//							elseFirstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(elseTransposeRegion);
					//						} else {
					//							// [FIXME] is it correct?
					//							elseCStmt = new CompoundStatement();
					//							elseFirstNonDeclStmt = elseTransposeRegion;
					//						}
					//
					//						List<ACCAnnotation> annotationListACC =
					//								AnalysisTools.collectPragmas(elseTransposeRegion, ACCAnnotation.class);
					//						List<FunctionCall> elseFuncCallList = IRTools.getFunctionCalls(elseCStmt);
					//
					//						//step5 replace all variables
					//						Statement enclosingCompRegion = null;
					//						for(ACCAnnotation annot : annotationListACC)
					//						{
					//							Annotatable att = annot.getAnnotatable();
					//							if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
					//									|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
					//								enclosingCompRegion = (Statement)att;
					//								IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
					//							}
					//
					//							for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
					//								Set<SubArray> transposeSArraySet = annot.get(dataClause);
					//								if(transposeSArraySet != null){
					//									System.out.println(transposeSArraySet.toString());
					//									for(SubArray subArray : transposeSArraySet){
					//										System.out.println(subArray.toString());
					//										System.out.println(subArray.getArrayName().toString());
					//										System.out.println(sArrayName);
					//										if(subArray.getArrayName().equals(sArrayNameEx)){
					//											SubArray newVarSubArray = subArray.clone();
					//											newVarSubArray.setArrayName(newVar.clone());
					//											int[] newIndexIndex = new int[dimension];
					//											for(int i = 0;i < dimension;i++){
					//												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
					//											}
					//											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
					//											for(int i = 0;i < dimension;i++){
					//												ExList.add(newVarSubArray.getRange(newIndexIndex[i]));
					//											}
					//											for(int i = 0;i < dimension;i++){
					//												newVarSubArray.setRange(i, ExList.get(i));
					//											}
					//											transposeSArraySet.add(newVarSubArray);
					//											transposeSArraySet.remove(subArray);
					//											break;
					//										}
					//									}
					//								}
					//							}
					//							for(String accinter : ACCAnnotation.internalDataClauses){  
					//								Set<Object> transposeSArrays = annot.get(accinter);
					//								if(transposeSArrays != null){
					//									System.out.println(accinter + " " + transposeSArrays.toString() + " " + transposeSArrays.size());
					//									for(Object subArray : transposeSArrays){
					//										if(subArray instanceof Symbol){
					//											if(((Symbol)subArray).equals(transpose_sym)){
					//												transposeSArrays.add(transpose_dec);
					//												transposeSArrays.remove(subArray);
					//												break;
					//											}
					//										}
					//									}
					//								}
					//							}
					//						}
					//						if(enclosingCompRegion == null) {
					//							continue;
					//						}
					//
					//
					//						List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
					//						for(Expression e : newVarExpressions){
					//							if(e.getParent().getClass().equals(ArrayAccess.class)){
					//								ArrayAccess aa = (ArrayAccess)e.getParent();
					//								//							System.out.println("array access name" + aa.getArrayName().toString());
					//								List<Expression> oldIndices = aa.getIndices();
					//								int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
					//								Expression newIndex = oldIndices.get(newIndexIndex);
					//								for(int i = 1; i < dimension;i++){
					//									newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
					//									newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
					//											BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
					//								}
					//								List<Expression> newIndices = new ArrayList<Expression>();
					//								newIndices.add(newIndex);
					//								aa.setIndices(newIndices);
					//							}
					//						}
					//
					//
					//						//size and rule information
					//						//						Expression sizeInfo = null;
					//						//						Expression ruleInfo = null;
					//						//						NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
					//						//						NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
					//						//						List<Expression> arryDimList = new ArrayList<Expression>();
					//						//						arryDimList.add(new IntegerLiteral(dimension));
					//						//						ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
					//						//						VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
					//						//						VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
					//						//						VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
					//						//						VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
					//						//						sizeInfo = new Identifier(sizeInfo_declarator);
					//						//						ruleInfo = new Identifier(ruleInfo_declarator);
					//						elseCStmt.addDeclaration(sizeInfo_decl.clone());
					//						elseCStmt.addDeclaration(ruleInfo_decl.clone());
					//						for(int i = 0;i < dimension;i++){
					//							Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
					//							Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
					//							elseCStmt.addStatementBefore(elseFirstNonDeclStmt,sizeInfo_Statement);
					//							elseCStmt.addStatementBefore(elseFirstNonDeclStmt,ruleInfo_Statement);
					//						}
					//
					//						//step3 make transpose_kernel function for CPU
					//						//						NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
					//						//						//CompoundStatement funcBody = new CompoundStatement();
					//						//						//Function transpose_kernel = new Function();
					//						//						FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
					//						//						Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
					//						elseCStmt.addStatementBefore(elseFirstNonDeclStmt, transpose_func_stmt);
					//						//						transpose_funcCall.addArgument(newVar.clone());
					//						//						//transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					//						//						transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					//						//						transpose_funcCall.addArgument(sizeInfo.clone());
					//						//						transpose_funcCall.addArgument(ruleInfo.clone());
					//						//						transpose_funcCall.addArgument(new IntegerLiteral(dimension));
					//						//	
					//						//						//step4 make re_transpose_kernel function for CPU
					//						//						NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
					//						//						FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
					//						//						Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
					//						elseCStmt.addStatement(retranspose_func_stmt);
					//						//						//retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					//						//						retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					//						//						retranspose_funcCall.addArgument(newVar.clone());
					//						//						retranspose_funcCall.addArgument(sizeInfo.clone());
					//						//						retranspose_funcCall.addArgument(ruleInfo.clone());
					//						//						retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
					//						//	
					//						//	
					//						//						//step5 make transpose_kernel function for GPU
					//						//						NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
					//						//						FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
					//						//						Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
					//						//						gpu_transpose_funcCall.addArgument(newVar.clone());
					//						//						//gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					//						//						gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					//						//						gpu_transpose_funcCall.addArgument(sizeInfo.clone());
					//						//						gpu_transpose_funcCall.addArgument(ruleInfo.clone());
					//						//						gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));
					//						//	
					//						//						//step4 make re_transpose_kernel function for GPU
					//						//						NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
					//						//						FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
					//						//						Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
					//						//						//gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					//						//						gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					//						//						gpu_retranspose_funcCall.addArgument(newVar.clone());
					//						//						gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
					//						//						gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
					//						//						gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
					//						//	
					//
					//						//step free
					//						FunctionCall free_call = new FunctionCall(new NameID("free"));
					//						free_call.addArgument(newVar.clone());
					//						Statement free_stmt = new ExpressionStatement(free_call);
					//						elseCStmt.addStatement(free_stmt);
					//
					//
					//						//step   function calls
					//						for(FunctionCall func : elseFuncCallList){
					//							System.out.println(func.toString() + ", " + func.getStatement().toString());
					//
					//							Traversable t = func.getParent();
					//							while ( !(t instanceof CompoundStatement)){
					//								t = t.getParent();
					//							}
					//							CompoundStatement funcParent = (CompoundStatement)t;
					//							funcParent.addStatementBefore(func.getStatement(), gpu_retranspose_func_stmt.clone());
					//							funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
					//							//							System.out.println(funcParent.toString());
					//							//							ACCAnnotation useDeviceDirective = new ACCAnnotation("host_data", null);
					//							//							List<Expression> useDeviceList = new ArrayList<Expression>();
					//							//							useDeviceList.add(newVar);
					//							//							useDeviceList.add(hostVar);
					//							//							useDeviceDirective.put("use_device", useDeviceList);
					//							//							Statement useDeviceStmt = new AnnotationStatement(useDeviceDirective);
					//							//							CompoundStatement useDeviceRegion = new CompoundStatement(); 
					//							//							CompoundStatement re_useDeviceRegion = new CompoundStatement(); 
					//							//							useDeviceRegion.addStatement(gpu_retranspose_func_stmt.clone());
					//							//							re_useDeviceRegion.addStatement(gpu_transpose_func_stmt.clone());
					//							//							funcParent.addStatementBefore(func.getStatement(), useDeviceStmt.clone());
					//							//							funcParent.addStatementBefore(func.getStatement(), useDeviceRegion);
					//							//							funcParent.addStatementAfter(func.getStatement(), re_useDeviceRegion);
					//							//							funcParent.addStatementAfter(func.getStatement(), useDeviceStmt.clone());
					//							////							funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
					//						}
					//
					//
					//						//step update directive 
					//						for(ACCAnnotation annot : annotationListACC)
					//						{
					//							Annotatable att = annot.getAnnotatable();
					//							if(att.containsAnnotation(ACCAnnotation.class, "update")){
					//								if(att.containsAnnotation(ACCAnnotation.class, "host")){
					//									boolean flag = false;
					//									Set<Object> hostList = annot.get("host");
					//									for(Object subArray : hostList){
					//										if(subArray instanceof Symbol){
					//											if(((Symbol)subArray).equals(transpose_sym)){
					//												hostList.add(transpose_dec);
					//												//System.out.println("transsym " + transpose_sym.getSymbolName());
					//												hostList.remove(subArray);
					//												flag = true;
					//												break;
					//											}
					//										}
					//										if(subArray instanceof SubArray){
					//											SubArray sa = (SubArray)subArray;
					//											if(sa.getArrayName().equals(sArrayNameEx)){
					//												sa.setArrayName(newVar.clone());
					//												int[] newIndexIndex = new int[dimension];
					//												for(int i = 0;i < dimension;i++){
					//													newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
					//												}
					//												List < List<Expression>> ExList = new ArrayList< List<Expression>>();
					//												for(int i = 0;i < dimension;i++){
					//													ExList.add(sa.getRange(newIndexIndex[i]));
					//												}
					//												for(int i = 0;i < dimension;i++){
					//													sa.setRange(i, ExList.get(i));
					//												}
					//												flag = true;
					//												break;
					//											}
					//										}
					//									}
					//									if(flag){
					//										Traversable t = att;
					//										while ( !(t instanceof CompoundStatement)){
					//											t = t.getParent();
					//										}
					//										CompoundStatement attParent = (CompoundStatement)t;
					//										attParent.addStatementAfter((Statement)att, retranspose_func_stmt.clone());
					//									}
					//
					//								}
					//								if(att.containsAnnotation(ACCAnnotation.class, "device")){
					//									boolean flag = false;
					//									Set<Object> deviceList = annot.get("device");
					//									for(Object subArray : deviceList){
					//										if(subArray instanceof Symbol){
					//											if(((Symbol)subArray).equals(transpose_sym)){
					//												deviceList.add(transpose_dec);
					//												//System.out.println("transsym " + transpose_sym.getSymbolName());
					//												deviceList.remove(subArray);
					//												flag = true;
					//												break;
					//											}
					//										}
					//										if(subArray instanceof SubArray){
					//											SubArray sa = (SubArray)subArray;
					//											if(sa.getArrayName().equals(sArrayNameEx)){
					//												sa.setArrayName(newVar.clone());
					//												int[] newIndexIndex = new int[dimension];
					//												for(int i = 0;i < dimension;i++){
					//													newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
					//												}
					//												List < List<Expression>> ExList = new ArrayList< List<Expression>>();
					//												for(int i = 0;i < dimension;i++){
					//													ExList.add(sa.getRange(newIndexIndex[i]));
					//												}
					//												for(int i = 0;i < dimension;i++){
					//													sa.setRange(i, ExList.get(i));
					//												}
					//												flag = true;
					//												break;
					//											}
					//										}
					//									}
					//									if(flag){
					//										Traversable t = att;
					//										while ( !(t instanceof CompoundStatement)){
					//											t = t.getParent();
					//										}
					//										CompoundStatement attParent = (CompoundStatement)t;
					//										attParent.addStatementBefore((Statement)att, transpose_func_stmt.clone());
					//									}
					//								}
					//							}
					//						}
					//					}
				}
			}



			//			//////////////////////////////////////////
			//			// Performs "redim" transformation. //
			//			//////////////////////////////////////////
			//			for( ARCAnnotation tAnnot : transposeAnnots ) {
			//
			//				Annotatable at = tAnnot.getAnnotatable();
			//				Set<SubArrayWithConf> sArrayConfSet = tAnnot.get("redim");
			//
			//				Statement redimStmt = (Statement)at;
			//				CompoundStatement redimBody = null;
			//				Statement firstNonDeclStmt = null;
			//				if (redimStmt instanceof CompoundStatement ){
			//					redimBody = (CompoundStatement)redimStmt;
			//					firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(redimStmt);
			//				} else {
			//					System.err.println("error: transform directive shuld have a region {}"); //[FIXME] English
			//				}
			//
			//				Procedure redimParentProc = IRTools.getParentProcedure(redimStmt);
			//				CompoundStatement redimParentBody = redimParentProc.getBody();
			//
			//				//step1 
			//				ACCAnnotation compAnnot = null;
			//				compAnnot = redimStmt.getAnnotation(ACCAnnotation.class, "kernels");
			//				if( compAnnot == null ){
			//					compAnnot = redimStmt.getAnnotation(ACCAnnotation.class, "parallel");
			//				}
			//				ACCAnnotation pcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(redimStmt, ACCAnnotation.class, ACCAnnotation.computeRegions, false, funcCallList, null);
			//				if( pcompAnnot != null ) {
			//					// [FIXME] is it correct?
			//					Tools.exit("[ERROR in DataLayoutTransformation()] transform pragma can not exist inside of " +
			//							"any compute regions (kerenls/parallel regions):\n" +
			//							"Enclosing procedure: " + redimParentProc.getSymbolName() + "\nOpenACC annotation: " + tAnnot + "\n");
			//				}
			//
			//
			//				List<FunctionCall> funcCallList2 = IRTools.getFunctionCalls(redimBody);
			//
			//				for(SubArrayWithConf sArrayConf : sArrayConfSet ) {
			//					//step2
			//
			//					SubArray sArray = sArrayConf.getSubArray();
			//					ConfList confL = sArrayConf.getConfList(0);
			//					List<Expression> confExList = confL.getConfigList();
			//					int dimension = confExList.size();
			//
			//					
			//					
			//					List<Expression> startIndecies = sArray.getStartIndices();
			//					List<Expression> exList = sArray.getLengths();
			//					Expression sArrayNameEx = sArray.getArrayName();
			//					String sArrayName = sArray.getArrayName().toString();
			//					String transposedArrayName = "transposed__" + sArrayName;
			//					System.out.println("size = " + dimension + " name = " + sArrayName.toString());
			//
			//					Set<Symbol> symSet = redimParentBody.getSymbols();
			//					Symbol transpose_sym = AnalysisTools.findsSymbol(symSet, sArrayName);
			//					System.out.println(transpose_sym.getSymbolName());
			//					System.out.println(transpose_sym.getArraySpecifiers().get(0).toString());
			//					System.out.println(transpose_sym.getTypeSpecifiers().get(0).toString());
			//
			//
			//					List<ACCAnnotation> annotationListACC =
			//							AnalysisTools.collectPragmas(redimStmt, ACCAnnotation.class);
			//
			//					/* get host variable */
			//					Expression hostVar = new Identifier(transpose_sym);
			//					//					Set<Symbol> outSymbols = new HashSet<Symbol>();
			//
			//					/* New variable declaration and allocation */
			//					Expression newVar = null;
			//					// Create a new transpose-host variable.
			//					// The type of the transpose-host symbol should be a pointer type 
			//					VariableDeclarator transpose_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
			//							new NameID(transposedArrayName));
			//					VariableDeclaration transpose_decl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
			//							transpose_declarator);
			//					newVar = new Identifier(transpose_declarator);
			//					Symbol transpose_dec = transpose_declarator;
			//					//transpose_sym = transpose_declarator;
			//
			//					//step5 replace all variables
			//					Statement enclosingCompRegion = null;
			//					for(ACCAnnotation annot : annotationListACC)
			//					{
			//						Annotatable att = annot.getAnnotatable();
			//						if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
			//								|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
			//							enclosingCompRegion = (Statement)att;
			//							IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
			//							//IRTools.replaceAll(cStmt, hostVar, newVar);
			//						}
			//
			//						for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
			//							//							Set <Set<SubArray>> transposeSArraySet = annot.get(dataClause);
			//							Set<SubArray> transposeSArraySet = annot.get(dataClause);
			//							if(transposeSArraySet != null){
			//								System.out.println(transposeSArraySet.toString());
			//								for(SubArray subArray : transposeSArraySet){
			//									System.out.println(subArray.toString());
			//									System.out.println(subArray.getArrayName().toString());
			//									System.out.println(sArrayName);
			//									if(subArray.getArrayName().equals(sArrayNameEx)){
			//										SubArray newVarSubArray = subArray.clone();
			//										newVarSubArray.setArrayName(newVar.clone());
			//										int[] newIndexIndex = new int[dimension];
			//										for(int i = 0;i < dimension;i++){
			//											newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//										}
			//										List < List<Expression>> ExList = new ArrayList< List<Expression>>();
			//										for(int i = 0;i < dimension;i++){
			//											ExList.add(newVarSubArray.getRange(newIndexIndex[i]));
			//										}
			//										for(int i = 0;i < dimension;i++){
			//											newVarSubArray.setRange(i, ExList.get(i));
			//										}
			//										transposeSArraySet.add(newVarSubArray);
			//									}
			//									//}
			//									//}
			//								}
			//							}
			//
			//						}
			//						for(String accinter : ACCAnnotation.internalDataClauses){  
			//							//Set<Symbol> transposeSArrays = annot.get(accinter);
			//							Set<Object> transposeSArrays = annot.get(accinter);
			//							//Set<SubArray> transposeSArrays = annot.get(accinter);
			//							if(transposeSArrays != null){
			//								System.out.println(accinter + " " + transposeSArrays.toString() + " " + transposeSArrays.size());
			//								//								for(SubArray subArray : transposeSArrays){ //[FIXME]
			//								for(Object subArray : transposeSArrays){
			//									if(subArray instanceof Symbol){
			//										if(((Symbol)subArray).equals(transpose_sym)){
			//											transposeSArrays.add(transpose_dec);
			//											//transposeSArrays.remove(subArray);
			//											break;
			//										}
			//									}
			//								}
			//							}
			//						}
			//					}
			//					if(enclosingCompRegion == null) {
			//						continue;
			//					}
			//
			//
			//					List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
			//					for(Expression e : newVarExpressions){
			//						if(e.getParent().getClass().equals(ArrayAccess.class)){
			//							ArrayAccess aa = (ArrayAccess)e.getParent();
			//							//							System.out.println("array access name" + aa.getArrayName().toString());
			//							List<Expression> oldIndices = aa.getIndices();
			//							int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
			//							Expression newIndex = oldIndices.get(newIndexIndex);
			//							for(int i = 1; i < dimension;i++){
			//								newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//								newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
			//										BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
			//							}
			//							List<Expression> newIndices = new ArrayList<Expression>();
			//							newIndices.add(newIndex);
			//							aa.setIndices(newIndices);
			//						}
			//					}
			//
			//					//					List<ArrayAccess> array_accesses = AnalysisTools.getArrayAccesses(newVar);
			//					//					for(ArrayAccess aa : array_accesses){
			//					//					System.out.println("array access name" + aa.getArrayName().toString());
			//					//					}
			//
			//					redimBody.addDeclaration(transpose_decl);
			//					//Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
			//					Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
			//					for(Expression ex : exList){
			//						size = new BinaryExpression(size, BinaryOperator.MULTIPLY, ex);
			//					}
			//					FunctionCall malloc_call = new FunctionCall(new NameID("malloc"));
			//					malloc_call.addArgument(size);
			//					List<Specifier> specs = new ArrayList<Specifier>(4);
			//					//specs.addAll(transpose_dec.getTypeSpecifiers());
			//					specs.addAll(transpose_dec.getTypeSpecifiers());
			//					Statement malloc_stmt = new ExpressionStatement(new AssignmentExpression(newVar.clone(),
			//							AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
			//					redimBody.addStatementBefore(firstNonDeclStmt, malloc_stmt);
			//
			//
			//					//size and rule information
			//					Expression sizeInfo = null;
			//					Expression ruleInfo = null;
			//					NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
			//					NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
			//					List<Expression> arryDimList = new ArrayList<Expression>();
			//					arryDimList.add(new IntegerLiteral(dimension));
			//					ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
			//					VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
			//					VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
			//					VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
			//					VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
			//					sizeInfo = new Identifier(sizeInfo_declarator);
			//					ruleInfo = new Identifier(ruleInfo_declarator);
			//					redimBody.addDeclaration(sizeInfo_decl);
			//					redimBody.addDeclaration(ruleInfo_decl);
			//					for(int i = 0;i < dimension;i++){
			//						Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
			//						Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
			//						redimBody.addStatementBefore(firstNonDeclStmt,sizeInfo_Statement);
			//						redimBody.addStatementBefore(firstNonDeclStmt,ruleInfo_Statement);
			//					}
			//
			//					//step3 make transpose_kernel function for CPU
			//					NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
			//					//CompoundStatement funcBody = new CompoundStatement();
			//					//Function transpose_kernel = new Function();
			//					FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
			//					Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
			//					redimBody.addStatementBefore(firstNonDeclStmt, transpose_func_stmt);
			//					transpose_funcCall.addArgument(newVar.clone());
			//					//transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					transpose_funcCall.addArgument(sizeInfo.clone());
			//					transpose_funcCall.addArgument(ruleInfo.clone());
			//					transpose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//					//step4 make re_transpose_kernel function for CPU
			//					NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
			//					FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
			//					Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
			//					redimBody.addStatement(retranspose_func_stmt);
			//					//retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					retranspose_funcCall.addArgument(newVar.clone());
			//					retranspose_funcCall.addArgument(sizeInfo.clone());
			//					retranspose_funcCall.addArgument(ruleInfo.clone());
			//					retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//
			//					//step5 make transpose_kernel function for GPU
			//					NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
			//					FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
			//					Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
			//					gpu_transpose_funcCall.addArgument(newVar.clone());
			//					//gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_transpose_funcCall.addArgument(sizeInfo.clone());
			//					gpu_transpose_funcCall.addArgument(ruleInfo.clone());
			//					gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//					//step4 make re_transpose_kernel function for GPU
			//					NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
			//					FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
			//					Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
			//					//gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_retranspose_funcCall.addArgument(newVar.clone());
			//					gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
			//					gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
			//					gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//
			//					//step free
			//					FunctionCall free_call = new FunctionCall(new NameID("free"));
			//					free_call.addArgument(newVar.clone());
			//					Statement free_stmt = new ExpressionStatement(free_call);
			//					redimBody.addStatement(free_stmt);
			//
			//
			//					//step   function calls
			//					for(FunctionCall func : funcCallList2){
			//						System.out.println(func.toString() + ", " + func.getStatement().toString());
			//
			//						Traversable t = func.getParent();
			//						while ( !(t instanceof CompoundStatement)){
			//							t = t.getParent();
			//						}
			//						CompoundStatement funcParent = (CompoundStatement)t;
			//						System.out.println(funcParent.toString());
			//						ACCAnnotation useDeviceDirective = new ACCAnnotation("host_data", null);
			//						List<Expression> useDeviceList = new ArrayList<Expression>();
			//						useDeviceList.add(newVar);
			//						useDeviceList.add(hostVar);
			//						useDeviceDirective.put("use_device", useDeviceList);
			//						Statement useDeviceStmt = new AnnotationStatement(useDeviceDirective);
			//						CompoundStatement useDeviceRegion = new CompoundStatement(); 
			//						CompoundStatement re_useDeviceRegion = new CompoundStatement(); 
			//						useDeviceRegion.addStatement(gpu_retranspose_func_stmt.clone());
			//						re_useDeviceRegion.addStatement(gpu_transpose_func_stmt.clone());
			//						funcParent.addStatementBefore(func.getStatement(), useDeviceStmt.clone());
			//						funcParent.addStatementBefore(func.getStatement(), useDeviceRegion);
			//						funcParent.addStatementAfter(func.getStatement(), re_useDeviceRegion);
			//						funcParent.addStatementAfter(func.getStatement(), useDeviceStmt.clone());
			////						funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
			//					}
			//
			//
			//					//step update directive 
			//					for(ACCAnnotation annot : annotationListACC)
			//					{
			//						Annotatable att = annot.getAnnotatable();
			//						if(att.containsAnnotation(ACCAnnotation.class, "update")){
			//							if(att.containsAnnotation(ACCAnnotation.class, "host")){
			//								boolean flag = false;
			//								Set<Object> hostList = annot.get("host");
			//								for(Object subArray : hostList){
			//									if(subArray instanceof Symbol){
			//										if(((Symbol)subArray).equals(transpose_sym)){
			//											hostList.add(transpose_dec);
			//											//System.out.println("transsym " + transpose_sym.getSymbolName());
			//											hostList.remove(subArray);
			//											flag = true;
			//											break;
			//										}
			//									}
			//									if(subArray instanceof SubArray){
			//										SubArray sa = (SubArray)subArray;
			//										if(sa.getArrayName().equals(sArrayNameEx)){
			//											sa.setArrayName(newVar.clone());
			//											int[] newIndexIndex = new int[dimension];
			//											for(int i = 0;i < dimension;i++){
			//												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//											}
			//											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
			//											for(int i = 0;i < dimension;i++){
			//												ExList.add(sa.getRange(newIndexIndex[i]));
			//											}
			//											for(int i = 0;i < dimension;i++){
			//												sa.setRange(i, ExList.get(i));
			//											}
			//											flag = true;
			//											break;
			//										}
			//									}
			//								}
			//								if(flag){
			//									Traversable t = att;
			//									while ( !(t instanceof CompoundStatement)){
			//										t = t.getParent();
			//									}
			//									CompoundStatement attParent = (CompoundStatement)t;
			//									attParent.addStatementAfter((Statement)att, retranspose_func_stmt.clone());
			//								}
			//
			//							}
			//							if(att.containsAnnotation(ACCAnnotation.class, "device")){
			//								boolean flag = false;
			//								Set<Object> deviceList = annot.get("device");
			//								for(Object subArray : deviceList){
			//									if(subArray instanceof Symbol){
			//										if(((Symbol)subArray).equals(transpose_sym)){
			//											deviceList.add(transpose_dec);
			//											//System.out.println("transsym " + transpose_sym.getSymbolName());
			//											deviceList.remove(subArray);
			//											flag = true;
			//											break;
			//										}
			//									}
			//									if(subArray instanceof SubArray){
			//										SubArray sa = (SubArray)subArray;
			//										if(sa.getArrayName().equals(sArrayNameEx)){
			//											sa.setArrayName(newVar.clone());
			//											int[] newIndexIndex = new int[dimension];
			//											for(int i = 0;i < dimension;i++){
			//												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//											}
			//											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
			//											for(int i = 0;i < dimension;i++){
			//												ExList.add(sa.getRange(newIndexIndex[i]));
			//											}
			//											for(int i = 0;i < dimension;i++){
			//												sa.setRange(i, ExList.get(i));
			//											}
			//											flag = true;
			//											break;
			//										}
			//									}
			//								}
			//								if(flag){
			//									Traversable t = att;
			//									while ( !(t instanceof CompoundStatement)){
			//										t = t.getParent();
			//									}
			//									CompoundStatement attParent = (CompoundStatement)t;
			//									attParent.addStatementBefore((Statement)att, transpose_func_stmt.clone());
			//								}
			//
			//							}
			//						}
			//					}
			//				}
			//			}
		}
		SymbolTools.linkSymbol(program);
		//System.out.println(program.toString());
	}



	public void start_tmp() {
		//for each transform, 
		//step0: ([TODO]): if ifcond exists and its argument is 0, skip the current transform region. 
		//step1: If the transform region is inside of compute regions, error in current version. 
		//step2: Create new transposed array
		//step2.1: Get the original data type by directive parsing
		//step3: transpose function call
		//step4: re-transpose function call
		//step5: replace the variables
		//step5.5: recalculate the index of the new array
		//step6: ([TODO]) deal with function calls
		//step7: ([TODO]) deal with copy clause and update directive in the transform region.

		List<ARCAnnotation> transformAnnots = IRTools.collectPragmas(program, ARCAnnotation.class, "transform");
		if( transformAnnots != null ) {
			List<ARCAnnotation> transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redimAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expandAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redim_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expand_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> transpose_expandAnnots = new LinkedList<ARCAnnotation>();

			for( ARCAnnotation arcAnnot : transformAnnots ) {
				if( arcAnnot.containsKey("transpose") ) {
					transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim") ) {
					redimAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand") ) {
					expandAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim_transpose") ) {
					redim_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand_transpose") ) {
					expand_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("transpose_expand") ) {
					transpose_expandAnnots.add(arcAnnot);
				}
			}


			List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);

			//////////////////////////////////////////
			// Performs "transpose" transformation. //
			//////////////////////////////////////////
			for( ARCAnnotation tAnnot : transposeAnnots ) {

				Annotatable at = tAnnot.getAnnotatable();
				Set<SubArrayWithConf> sArrayConfSet = tAnnot.get("transpose");

				Statement transposeRegion = (Statement)tAnnot.getAnnotatable();
				//				CompoundStatement cStmt = (CompoundStatement)transposeRegion.getParent();
				//				CompoundStatement cStmt = (CompoundStatement)transposeRegion.getChildren().get(0);
				CompoundStatement cStmt = null;
				Statement firstNonDeclStmt = null;
				if (transposeRegion instanceof CompoundStatement ){
					cStmt = (CompoundStatement)transposeRegion;
					firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(transposeRegion);
				} else {
					// [FIXME] is it correct?
					cStmt = new CompoundStatement();
					firstNonDeclStmt = transposeRegion;
				}



				Procedure cProc = IRTools.getParentProcedure(transposeRegion);
				CompoundStatement cBody = cProc.getBody();

				System.out.println("test");

				//step1 
				ACCAnnotation compAnnot = null;
				compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "kernels");
				if( compAnnot == null ){
					compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "parallel");
				}
				ACCAnnotation pcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(transposeRegion, ACCAnnotation.class, ACCAnnotation.computeRegions, false, funcCallList, null);
				if( pcompAnnot != null ) {
					Tools.exit("[ERROR in DataLayoutTransformation()] transform pragma can not exist inside of " +
							"any compute regions (kerenls/parallel regions):\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + tAnnot + "\n");
				}





				List<FunctionCall> funcCallList2 = IRTools.getFunctionCalls(cStmt);

				for(SubArrayWithConf sArrayConf : sArrayConfSet ) {
					//step2

					SubArray sArray = sArrayConf.getSubArray();
					//In transpose clause, each SubArrayWithConf has only on configuration list ([2,1,3]).
					//CF: redim_transpose, each SubArrayWithConf will have two configuration lists 
					//(one for redim (e.g, [X,Y,Z]), and the other for transpose(e.g., [2,1,3])).
					ConfList confL = sArrayConf.getConfList(0);
					//...
					List<Expression> startIndecies = sArray.getStartIndices();
					List<Expression> exList = sArray.getLengths();
					int dimension = sArray.getArrayDimension();
					Expression sArrayNameEx = sArray.getArrayName();
					String sArrayName = sArray.getArrayName().toString();
					String transposedArrayName = "transposed__" + sArrayName;
					System.out.println("size = " + dimension + " name = " + sArrayName.toString());

					Set<Symbol> symSet = cBody.getSymbols();
					Symbol transpose_sym = AnalysisTools.findsSymbol(symSet, sArrayName);
					System.out.println(transpose_sym.getSymbolName());
					System.out.println(transpose_sym.getArraySpecifiers().get(0).toString());
					System.out.println(transpose_sym.getTypeSpecifiers().get(0).toString());


					List<ACCAnnotation> annotationListACC =
							AnalysisTools.collectPragmas(transposeRegion, ACCAnnotation.class);

					/* get host variable */
					Expression hostVar = new Identifier(transpose_sym);
					//					Set<Symbol> outSymbols = new HashSet<Symbol>();

					/* New variable declaration and allocation */
					Expression newVar = null;
					// Create a new transpose-host variable.
					// The type of the transpose-host symbol should be a pointer type 
					VariableDeclarator transpose_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(transposedArrayName));
					VariableDeclaration transpose_decl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							transpose_declarator);
					newVar = new Identifier(transpose_declarator);
					Symbol transpose_dec = transpose_declarator;
					//transpose_sym = transpose_declarator;


					//declarations for device address pointer
					Expression gpuVar = null; 
					Expression gpuNewVar = null;
					String gpuVarName = "deviceAddressFor__" + sArrayName;
					String gpuNewVarName = "transposedDeviceAddressFor__" + sArrayName;
					VariableDeclarator gpuVarDeclarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(gpuVarName));
					VariableDeclarator gpuNewVarDeclarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(gpuNewVarName));
					VariableDeclaration gpuVarDecl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							gpuVarDeclarator);
					VariableDeclaration gpuNewVarDecl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							gpuNewVarDeclarator);
					gpuVar = new Identifier(gpuVarDeclarator);
					gpuNewVar = new Identifier(gpuNewVarDeclarator);
					cBody.addDeclaration(gpuVarDecl);
					cBody.addDeclaration(gpuNewVarDecl);




					//step5 replace all variables
					Statement enclosingCompRegion = null;
					for(ACCAnnotation annot : annotationListACC)
					{
						Annotatable att = annot.getAnnotatable();
						if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
								|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
							enclosingCompRegion = (Statement)att;
							IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
							//IRTools.replaceAll(cStmt, hostVar, newVar);
						}

						for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
							//							Set <Set<SubArray>> transposeSArraySet = annot.get(dataClause);
							Set<SubArray> transposeSArraySet = annot.get(dataClause);
							if(transposeSArraySet != null){
								System.out.println(transposeSArraySet.toString());
								//for(Set<SubArray> subArrays : transposeSArraySet){
								for(SubArray subArray : transposeSArraySet){
									//if(subArrays != null){
									//for(SubArray subArray : subArrays){
									System.out.println(subArray.toString());
									System.out.println(subArray.getArrayName().toString());
									System.out.println(sArrayName);
									if(subArray.getArrayName().equals(sArrayNameEx)){
										subArray.setArrayName(newVar.clone());
										int[] newIndexIndex = new int[dimension];
										for(int i = 0;i < dimension;i++){
											newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
										}
										List < List<Expression>> ExList = new ArrayList< List<Expression>>();
										for(int i = 0;i < dimension;i++){
											ExList.add(subArray.getRange(newIndexIndex[i]));
										}
										for(int i = 0;i < dimension;i++){
											subArray.setRange(i, ExList.get(i));
										}
									}
									//}
									//}
								}
							}

						}
						for(String accinter : ACCAnnotation.internalDataClauses){  
							//Set<Symbol> transposeSArrays = annot.get(accinter);
							Set<Object> transposeSArrays = annot.get(accinter);
							//Set<SubArray> transposeSArrays = annot.get(accinter);
							if(transposeSArrays != null){
								System.out.println(accinter + " " + transposeSArrays.toString() + " " + transposeSArrays.size());
								//								for(SubArray subArray : transposeSArrays){ //[FIXME]
								for(Object subArray : transposeSArrays){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											//transposeSArrays.add(transpose_dec);
											transposeSArrays.add(transpose_dec);
											//System.out.println("transsym " + transpose_sym.getSymbolName());
											transposeSArrays.remove(subArray);
											break;
										}
									}
								}
							}
						}
					}
					if(enclosingCompRegion == null) {
						continue;
					}


					List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
					for(Expression e : newVarExpressions){
						if(e.getParent().getClass().equals(ArrayAccess.class)){
							ArrayAccess aa = (ArrayAccess)e.getParent();
							//							System.out.println("array access name" + aa.getArrayName().toString());
							List<Expression> oldIndices = aa.getIndices();
							int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
							Expression newIndex = oldIndices.get(newIndexIndex);
							for(int i = 1; i < dimension;i++){
								newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
								newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
										BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
							}
							List<Expression> newIndices = new ArrayList<Expression>();
							newIndices.add(newIndex);
							aa.setIndices(newIndices);
						}
					}

					//					List<ArrayAccess> array_accesses = AnalysisTools.getArrayAccesses(newVar);
					//					for(ArrayAccess aa : array_accesses){
					//					System.out.println("array access name" + aa.getArrayName().toString());
					//					}

					cStmt.addDeclaration(transpose_decl);
					//Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
					Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
					for(Expression ex : exList){
						size = new BinaryExpression(size, BinaryOperator.MULTIPLY, ex);
					}
					FunctionCall malloc_call = new FunctionCall(new NameID("malloc"));
					malloc_call.addArgument(size);
					List<Specifier> specs = new ArrayList<Specifier>(4);
					//specs.addAll(transpose_dec.getTypeSpecifiers());
					specs.addAll(transpose_dec.getTypeSpecifiers());
					Statement malloc_stmt = new ExpressionStatement(new AssignmentExpression(newVar.clone(),
							AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
					cStmt.addStatementBefore(firstNonDeclStmt, malloc_stmt);



					//[TODO] To know how can I get access to the variables which have same name of the symbols
					//IRTools.replaceAll((Traversable)cBody, (Expression)transpose_sym, newVar.clone());

					//get kernels/parallel region



					//size and rule information
					Expression sizeInfo = null;
					Expression ruleInfo = null;
					NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
					NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
					List<Expression> arryDimList = new ArrayList<Expression>();
					arryDimList.add(new IntegerLiteral(dimension));
					ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
					VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
					VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
					VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
					VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
					sizeInfo = new Identifier(sizeInfo_declarator);
					ruleInfo = new Identifier(ruleInfo_declarator);
					cStmt.addDeclaration(sizeInfo_decl);
					cStmt.addDeclaration(ruleInfo_decl);
					for(int i = 0;i < dimension;i++){
						Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
						Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
						cStmt.addStatementBefore(firstNonDeclStmt,sizeInfo_Statement);
						cStmt.addStatementBefore(firstNonDeclStmt,ruleInfo_Statement);
					}


					//					Identifier loopIndex = TransformTools.getNewTempIndex(cStmt);
					//					CompoundStatement loopBody = new CompoundStatement();
					//					Statement loopInit = new ExpressionStatement(new BinaryExpression(loopIndex.clone(), AssignmentOperator.NORMAL,new IntegerLiteral(0) ));  
					//					Expression loopCond = new BinaryExpression(loopIndex.clone(), BinaryOperator.COMPARE_LT, new IntegerLiteral(dimension));
					//					Expression loopStep = new UnaryExpression(UnaryOperator.POST_INCREMENT,loopIndex.clone());
					//					ForLoop sizeInfo_ForStatement = new ForLoop(loopInit, loopCond, loopStep, loopBody);
					//					cStmt.addStatementBefore(firstNonDeclStmt, sizeInfo_ForStatement);



					//step3 make transpose_kernel function for CPU
					NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
					//CompoundStatement funcBody = new CompoundStatement();
					//Function transpose_kernel = new Function();
					FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
					Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
					cStmt.addStatementBefore(firstNonDeclStmt, transpose_func_stmt);
					transpose_funcCall.addArgument(newVar.clone());
					//transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					transpose_funcCall.addArgument(sizeInfo.clone());
					transpose_funcCall.addArgument(ruleInfo.clone());
					transpose_funcCall.addArgument(new IntegerLiteral(dimension));

					//step4 make re_transpose_kernel function for CPU
					NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
					FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
					Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
					cStmt.addStatement(retranspose_func_stmt);
					//retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					retranspose_funcCall.addArgument(newVar.clone());
					retranspose_funcCall.addArgument(sizeInfo.clone());
					retranspose_funcCall.addArgument(ruleInfo.clone());
					retranspose_funcCall.addArgument(new IntegerLiteral(dimension));


					//step5 make transpose_kernel function for GPU
					NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
					FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
					Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
					gpu_transpose_funcCall.addArgument(newVar.clone());
					//gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					gpu_transpose_funcCall.addArgument(sizeInfo.clone());
					gpu_transpose_funcCall.addArgument(ruleInfo.clone());
					gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));

					//step4 make re_transpose_kernel function for GPU
					NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
					FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
					Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
					//gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					gpu_retranspose_funcCall.addArgument(newVar.clone());
					gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
					gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
					gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));


					//step free
					FunctionCall free_call = new FunctionCall(new NameID("free"));
					free_call.addArgument(newVar.clone());
					Statement free_stmt = new ExpressionStatement(free_call);
					cStmt.addStatement(free_stmt);


					//step   function calls
					for(FunctionCall func : funcCallList2){

						Traversable t = func.getParent();
						while ( !(t instanceof CompoundStatement)){
							t = t.getParent();
						}
						CompoundStatement funcParent = (CompoundStatement)t;
						funcParent.addStatementBefore(func.getStatement(), gpu_retranspose_func_stmt.clone());
						funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
					}
					//					for(FunctionCall func : funcCallList2){
					//						System.out.println(func.toString() + ", " + func.getStatement().toString());
					//
					//						Traversable t = func.getParent();
					//						while ( !(t instanceof CompoundStatement)){
					//							t = t.getParent();
					//						}
					//						CompoundStatement funcParent = (CompoundStatement)t;
					//						System.out.println(funcParent.toString());
					//						ACCAnnotation useDeviceDirective = new ACCAnnotation("host_data", null);
					//						List<Expression> useDeviceList = new ArrayList<Expression>();
					//						useDeviceList.add(newVar);
					//						useDeviceList.add(hostVar);
					//						useDeviceDirective.put("use_device", useDeviceList);
					//						Statement useDeviceStmt = new AnnotationStatement(useDeviceDirective);
					//						CompoundStatement useDeviceRegion = new CompoundStatement(); 
					//						CompoundStatement re_useDeviceRegion = new CompoundStatement(); 
					//						useDeviceRegion.addStatement(gpu_retranspose_func_stmt.clone());
					//						re_useDeviceRegion.addStatement(gpu_transpose_func_stmt.clone());
					//						funcParent.addStatementBefore(func.getStatement(), useDeviceStmt.clone());
					//						funcParent.addStatementBefore(func.getStatement(), useDeviceRegion);
					//						funcParent.addStatementAfter(func.getStatement(), re_useDeviceRegion);
					//						funcParent.addStatementAfter(func.getStatement(), useDeviceStmt.clone());
					//						//						funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
					//					}


					//step update directive 
					for(ACCAnnotation annot : annotationListACC)
					{
						Annotatable att = annot.getAnnotatable();
						if(att.containsAnnotation(ACCAnnotation.class, "update")){
							if(att.containsAnnotation(ACCAnnotation.class, "host")){
								boolean flag = false;
								Set<Object> hostList = annot.get("host");
								for(Object subArray : hostList){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											hostList.add(transpose_dec);
											//System.out.println("transsym " + transpose_sym.getSymbolName());
											hostList.remove(subArray);
											flag = true;
											break;
										}
									}
									if(subArray instanceof SubArray){
										SubArray sa = (SubArray)subArray;
										if(sa.getArrayName().equals(sArrayNameEx)){
											sa.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(sa.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												sa.setRange(i, ExList.get(i));
											}
											flag = true;
											break;
										}
									}
								}
								if(flag){
									Traversable t = att;
									while ( !(t instanceof CompoundStatement)){
										t = t.getParent();
									}
									CompoundStatement attParent = (CompoundStatement)t;
									attParent.addStatementAfter((Statement)att, retranspose_func_stmt.clone());
								}

							}
							if(att.containsAnnotation(ACCAnnotation.class, "device")){
								boolean flag = false;
								Set<Object> deviceList = annot.get("device");
								for(Object subArray : deviceList){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											deviceList.add(transpose_dec);
											//System.out.println("transsym " + transpose_sym.getSymbolName());
											deviceList.remove(subArray);
											flag = true;
											break;
										}
									}
									if(subArray instanceof SubArray){
										SubArray sa = (SubArray)subArray;
										if(sa.getArrayName().equals(sArrayNameEx)){
											sa.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(sa.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												sa.setRange(i, ExList.get(i));
											}
											flag = true;
											break;
										}
									}
								}
								if(flag){
									Traversable t = att;
									while ( !(t instanceof CompoundStatement)){
										t = t.getParent();
									}
									CompoundStatement attParent = (CompoundStatement)t;
									attParent.addStatementBefore((Statement)att, transpose_func_stmt.clone());
								}

							}
						}
					}
				}
			}
		}
		SymbolTools.linkSymbol(program);
	}



	public void start_20141112_bug() {
		//for each transform, 
		//step0: ([TODO]): if ifcond exists and its argument is 0, skip the current transform region. 
		//step1: If the transform region is inside of compute regions, error in current version. 
		//step2: if the transposed array isn't exist device memory, step2.1. else, step2.2
		//step2.1: Create new transposed array
		//step2.2: 
		//step3: transpose function call
		//step4: re-transpose function call
		//step5: replace the variables
		//step5.5: recalculate the index of the new array
		//step6: ([TODO]) deal with function calls
		//step7: ([TODO]) deal with copy clause and update directive in the transform region.

		List<ARCAnnotation> transformAnnots = IRTools.collectPragmas(program, ARCAnnotation.class, "transform");
		if( transformAnnots != null ) {
			List<ARCAnnotation> transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redimAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expandAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redim_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expand_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> transpose_expandAnnots = new LinkedList<ARCAnnotation>();

			for( ARCAnnotation arcAnnot : transformAnnots ) {
				if( arcAnnot.containsKey("transpose") ) {
					transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim") ) {
					redimAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand") ) {
					expandAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim_transpose") ) {
					redim_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand_transpose") ) {
					expand_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("transpose_expand") ) {
					transpose_expandAnnots.add(arcAnnot);
				}
			}


			List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);

			//////////////////////////////////////////
			// Performs "transpose" transformation. //
			//////////////////////////////////////////
			for( ARCAnnotation tAnnot : transposeAnnots ) {

				Set<SubArrayWithConf> sArrayConfSet = tAnnot.get("transpose");
				Statement transposeRegion = (Statement)tAnnot.getAnnotatable();
				CompoundStatement cStmt = null;
				Statement firstNonDeclStmt = null;
				if (transposeRegion instanceof CompoundStatement ){
					cStmt = (CompoundStatement)transposeRegion;
					firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(transposeRegion);
				} else {
					// [FIXME] is it correct?
					cStmt = new CompoundStatement();
					firstNonDeclStmt = transposeRegion;
				}

				Procedure cProc = IRTools.getParentProcedure(transposeRegion);
				CompoundStatement cBody = cProc.getBody();

				//step1 
				ACCAnnotation compAnnot = null;
				compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "kernels");
				if( compAnnot == null ){
					compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "parallel");
				}
				ACCAnnotation pcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(transposeRegion, ACCAnnotation.class, ACCAnnotation.computeRegions, false, funcCallList, null);
				if( pcompAnnot != null ) {
					Tools.exit("[ERROR in DataLayoutTransformation()] transform pragma can not exist inside of " +
							"any compute regions (kerenls/parallel regions):\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + tAnnot + "\n");
				}

				for(SubArrayWithConf sArrayConf : sArrayConfSet ) {

					//step2.0 get pragma information
					SubArray sArray = sArrayConf.getSubArray();
					//In transpose clause, each SubArrayWithConf has only on configuration list ([2,1,3]).
					//CF: redim_transpose, each SubArrayWithConf will have two configuration lists 
					//(one for redim (e.g, [X,Y,Z]), and the other for transpose(e.g., [2,1,3])).
					ConfList confL = sArrayConf.getConfList(0);
					List<Expression> startIndecies = sArray.getStartIndices();
					List<Expression> exList = sArray.getLengths();
					Expression size_1D = new IntegerLiteral(1);
					for(Expression ex : exList){
						size_1D = Symbolic.multiply(size_1D, ex);
					}
					int dimension = sArray.getArrayDimension();
					Expression sArrayNameEx = sArray.getArrayName();
					String sArrayName = sArray.getArrayName().toString();
					String transposedArrayName = "transposed__" + sArrayName;
					System.out.println("size = " + dimension + " name = " + sArrayName.toString());

					Set<Symbol> symSet = cBody.getSymbols();
					Symbol transpose_sym = AnalysisTools.findsSymbol(symSet, sArrayName);

					/* get host variable */
					Expression hostVar = new Identifier(transpose_sym);

					/* New variable declaration and allocation */
					Expression newVar = null;
					// Create a new transpose-host variable.
					// The type of the transpose-host symbol should be a pointer type 
					VariableDeclarator transpose_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(transposedArrayName));
					VariableDeclaration transpose_decl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							transpose_declarator);
					newVar = new Identifier(transpose_declarator);
					Symbol transpose_dec = transpose_declarator;

					/* make transposition kernels */

					//size and rule information
					Expression sizeInfo = null;
					Expression ruleInfo = null;
					NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
					NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
					List<Expression> arryDimList = new ArrayList<Expression>();
					arryDimList.add(new IntegerLiteral(dimension));
					ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
					VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
					VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
					VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
					VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
					sizeInfo = new Identifier(sizeInfo_declarator);
					ruleInfo = new Identifier(ruleInfo_declarator);

					//transpose_kernel function for CPU
					NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
					FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
					Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
					transpose_funcCall.addArgument(newVar.clone());
					transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					transpose_funcCall.addArgument(sizeInfo.clone());
					transpose_funcCall.addArgument(ruleInfo.clone());
					transpose_funcCall.addArgument(new IntegerLiteral(dimension));

					//re_transpose_kernel function for CPU
					NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
					FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
					Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
					retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					retranspose_funcCall.addArgument(newVar.clone());
					retranspose_funcCall.addArgument(sizeInfo.clone());
					retranspose_funcCall.addArgument(ruleInfo.clone());
					retranspose_funcCall.addArgument(new IntegerLiteral(dimension));

					//declarations for device address pointer
					Expression gpuVar = null; 
					Expression gpuNewVar = null;
					String gpuVarName = "deviceAddressFor__" + sArrayName;
					String gpuNewVarName = "transposedDeviceAddressFor__" + sArrayName;
					VariableDeclarator gpuVarDeclarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(gpuVarName));
					VariableDeclarator gpuNewVarDeclarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(gpuNewVarName));
					VariableDeclaration gpuVarDecl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							gpuVarDeclarator);
					VariableDeclaration gpuNewVarDecl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							gpuNewVarDeclarator);
					gpuVar = new Identifier(gpuVarDeclarator);
					gpuNewVar = new Identifier(gpuNewVarDeclarator);
					cBody.addDeclaration(gpuVarDecl);
					cBody.addDeclaration(gpuNewVarDecl);


					//transpose_kernel function for GPU
					NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
					FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
					Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
					gpu_transpose_funcCall.addArgument(gpuNewVar.clone());
					gpu_transpose_funcCall.addArgument(gpuVar.clone());
					gpu_transpose_funcCall.addArgument(sizeInfo.clone());
					gpu_transpose_funcCall.addArgument(ruleInfo.clone());
					gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));

					//re_transpose_kernel function for GPU
					NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
					FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
					Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
					gpu_retranspose_funcCall.addArgument(gpuVar.clone());
					gpu_retranspose_funcCall.addArgument(gpuNewVar.clone());
					gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
					gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
					gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));




					//					//step2 check if the transposed array is already exist on the device or not,
					//					NameID checkPresentName = new NameID("acc_is_present");
					//					FunctionCall checkPresentFuncCall = new FunctionCall(checkPresentName);
					//					checkPresentFuncCall.addArgument(hostVar);
					//					checkPresentFuncCall.addArgument(size_1D);
					//					Expression checkPresent = new BinaryExpression(checkPresentFuncCall, BinaryOperator.COMPARE_EQ, new IntegerLiteral(1));
					//					CompoundStatement ifPresentTrueBodyStmt = new CompoundStatement();
					//					CompoundStatement ifPresentElseBodyStmt = new CompoundStatement();
					//					Statement checkPresentStmt = new IfStatement(checkPresent, ifPresentTrueBodyStmt, ifPresentElseBodyStmt);
					//
					//					cBody.addDeclaration(transpose_decl);

					Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
					for(Expression ex : exList){
						size = new BinaryExpression(size, BinaryOperator.MULTIPLY, ex.clone());
					}
					FunctionCall malloc_call = new FunctionCall(new NameID("malloc"));
					malloc_call.addArgument(size);
					List<Specifier> specs = new ArrayList<Specifier>(4);
					//specs.addAll(transpose_dec.getTypeSpecifiers());
					specs.addAll(transpose_dec.getTypeSpecifiers());
					Statement malloc_stmt = new ExpressionStatement(new AssignmentExpression(newVar.clone(),
							AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
					cStmt.addStatementBefore(firstNonDeclStmt, malloc_stmt.clone());
					//					ifPresentTrueBodyStmt.addStatement(malloc_stmt.clone());
					//					ifPresentElseBodyStmt.addStatement(malloc_stmt.clone());

					//if already exist, add a data directive for new device variable
					ACCAnnotation dataDirective = new ACCAnnotation("data", null);
					SubArray newDeviceArray = new SubArray(newVar.clone(), new IntegerLiteral(0), size_1D.clone());
					dataDirective.put("create", newDeviceArray.clone());
					Statement dataDirectiveStmt = new AnnotationStatement(dataDirective);
					CompoundStatement dataDirectiveRegion = new CompoundStatement();
					dataDirectiveRegion.addStatement(cStmt.clone());
					cStmt.addStatement(dataDirectiveStmt.clone());
					cStmt.addStatement(dataDirectiveRegion.clone());
					//ifPresentElseBodyStmt.addStatement(cStmt.clone());
					//cStmt.addStatementBefore(firstNonDeclStmt, checkPresentStmt);

					List<ACCAnnotation> annotationListACC =
							AnalysisTools.collectPragmas(cStmt, ACCAnnotation.class);
					List<FunctionCall> FuncCallList = IRTools.getFunctionCalls(cStmt);

					//step5 replace all variables
					Statement enclosingCompRegion = null;
					for(ACCAnnotation annot : annotationListACC)
					{
						Annotatable att = annot.getAnnotatable();
						if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
								|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
							enclosingCompRegion = (Statement)att;
							IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
						}

						for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
							//Set<SubArray> transposeSArraySet = annot.get(dataClause);
							Set<Object> transposeSArraySet = annot.get(dataClause);
							if(transposeSArraySet != null){
								System.out.println(transposeSArraySet.toString());
								for(Object subArray : transposeSArraySet){
									if(subArray instanceof SubArray){
										if(((SubArray)subArray).getArrayName().equals(sArrayNameEx)){
											SubArray newVarSubArray = ((SubArray)subArray).clone();
											newVarSubArray.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(newVarSubArray.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												newVarSubArray.setRange(i, ExList.get(i));
											}
											transposeSArraySet.add(newVarSubArray);
											//transposeSArraySet.remove(subArray);
											break;
										}
									}
								}
							}

						}
						for(String accinter : ACCAnnotation.internalDataClauses){  
							//Set<Symbol> transposeSArrays = annot.get(accinter);
							Set<Object> transposeSArrays = annot.get(accinter);
							//Set<SubArray> transposeSArrays = annot.get(accinter);
							if(transposeSArrays != null){
								System.out.println(accinter + " " + transposeSArrays.toString() + " " + transposeSArrays.size());
								//								for(SubArray subArray : transposeSArrays){ //[FIXME]
								for(Object subArray : transposeSArrays){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											transposeSArrays.add(transpose_dec);
											//transposeSArrays.remove(subArray);
											break;
										}
									}
								}
							}
						}
					}
					if(enclosingCompRegion == null) {
						continue;
					}


					List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
					for(Expression e : newVarExpressions){
						if(e.getParent().getClass().equals(ArrayAccess.class)){
							ArrayAccess aa = (ArrayAccess)e.getParent();
							//							System.out.println("array access name" + aa.getArrayName().toString());
							List<Expression> oldIndices = aa.getIndices();
							int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
							Expression newIndex = oldIndices.get(newIndexIndex);
							for(int i = 1; i < dimension;i++){
								newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
								newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
										BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
							}
							List<Expression> newIndices = new ArrayList<Expression>();
							newIndices.add(newIndex);
							aa.setIndices(newIndices);
						}
					}



					//						//size and rule information
					cStmt.addDeclaration(sizeInfo_decl.clone());
					cStmt.addDeclaration(ruleInfo_decl.clone());
					for(int i = 0;i < dimension;i++){
						Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
						Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
						cStmt.addStatementBefore(firstNonDeclStmt,sizeInfo_Statement);
						cStmt.addStatementBefore(firstNonDeclStmt,ruleInfo_Statement);
					}

					CompoundStatement presentErrorCode = new CompoundStatement();
					FunctionCall printfCall = new FunctionCall(new NameID("printf"));
					printfCall.addArgument(new StringLiteral("[ERROR] GPU memory for the host variable, "+hostVar.toString()+
							", does not exist. \\n"));
					presentErrorCode.addStatement(new ExpressionStatement(printfCall));
					printfCall = new FunctionCall(new NameID("printf"));
					printfCall.addArgument(new StringLiteral("Enclosing annotation: \\n" + transposeRegion.toString() + " \\n"));
					presentErrorCode.addStatement(new ExpressionStatement(printfCall));
					FunctionCall exitCall = new FunctionCall(new NameID("exit"));
					exitCall.addArgument(new IntegerLiteral(1));
					presentErrorCode.addStatement(new ExpressionStatement(exitCall));

					NameID getGpuVarAddressName = new NameID("HI_get_device_address");
					FunctionCall getGpuVarAddressFuncCall = new FunctionCall(getGpuVarAddressName);
					getGpuVarAddressFuncCall.addArgument(hostVar.clone());
					getGpuVarAddressFuncCall.addArgument(gpuVar.clone());
					Expression asyncID = new NameID("acc_async_noval");
					getGpuVarAddressFuncCall.addArgument(asyncID.clone());

					Expression getGpuVarAddressExp = new BinaryExpression(getGpuVarAddressFuncCall, BinaryOperator.COMPARE_NE, new NameID("HI_success"));
					IfStatement getGpuVarAddressIfStmt = new IfStatement(getGpuVarAddressExp.clone(), presentErrorCode.clone());
					cStmt.addStatementBefore(firstNonDeclStmt, getGpuVarAddressIfStmt.clone());


					CompoundStatement transposedPresentErrorCode = new CompoundStatement();
					FunctionCall transposedPrintfCall = new FunctionCall(new NameID("printf"));
					transposedPrintfCall.addArgument(new StringLiteral("[ERROR] GPU memory for the host variable, "+newVar.toString()+
							", does not exist. \\n"));
					transposedPresentErrorCode.addStatement(new ExpressionStatement(transposedPrintfCall));
					transposedPrintfCall = new FunctionCall(new NameID("printf"));
					transposedPrintfCall.addArgument(new StringLiteral("Enclosing annotation: \\n" + transposeRegion.toString() + " \\n"));
					transposedPresentErrorCode.addStatement(new ExpressionStatement(transposedPrintfCall));
					FunctionCall transposedExitCall = new FunctionCall(new NameID("exit"));
					transposedExitCall.addArgument(new IntegerLiteral(1));
					transposedPresentErrorCode.addStatement(new ExpressionStatement(transposedExitCall));

					NameID getGpuNewVarAddressName = new NameID("HI_get_device_address");
					FunctionCall getGpuNewVarAddressFuncCall = new FunctionCall(getGpuNewVarAddressName);
					getGpuNewVarAddressFuncCall.addArgument(newVar.clone());
					getGpuNewVarAddressFuncCall.addArgument(gpuNewVar.clone());
					getGpuNewVarAddressFuncCall.addArgument(asyncID.clone());
					Expression getGpuNewVarAddressExp = new BinaryExpression(getGpuNewVarAddressFuncCall, BinaryOperator.COMPARE_NE, new NameID("HI_success"));
					IfStatement getGpuNewVarAddressIfStmt = new IfStatement(getGpuNewVarAddressExp.clone(), transposedPresentErrorCode.clone());
					cStmt.addStatementBefore(firstNonDeclStmt, getGpuNewVarAddressIfStmt.clone());		

					cStmt.addStatementBefore(firstNonDeclStmt, gpu_transpose_func_stmt);
					cStmt.addStatement(gpu_retranspose_func_stmt);

					//step free
					FunctionCall free_call = new FunctionCall(new NameID("free"));
					free_call.addArgument(newVar.clone());
					Statement free_stmt = new ExpressionStatement(free_call);
					cStmt.addStatement(free_stmt);


					//step   function calls
					for(FunctionCall func : funcCallList){

						Traversable t = func.getParent();
						while ( !(t instanceof CompoundStatement)){
							t = t.getParent();
						}
						CompoundStatement funcParent = (CompoundStatement)t;
						funcParent.addStatementBefore(func.getStatement(), gpu_retranspose_func_stmt.clone());
						funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
					}


					//step update directive 
					for(ACCAnnotation annot : annotationListACC)
					{
						Annotatable att = annot.getAnnotatable();
						if(att.containsAnnotation(ACCAnnotation.class, "update")){
							if(att.containsAnnotation(ACCAnnotation.class, "host")){
								boolean flag = false;
								Set<Object> hostList = annot.get("host");
								for(Object subArray : hostList){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											hostList.add(transpose_dec);
											hostList.remove(subArray);
											flag = true;
											break;
										}
									}
									if(subArray instanceof SubArray){
										SubArray sa = (SubArray)subArray;
										if(sa.getArrayName().equals(sArrayNameEx)){
											sa.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(sa.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												sa.setRange(i, ExList.get(i));
											}
											flag = true;
											break;
										}
									}
								}
								if(flag){
									Traversable t = att;
									while ( !(t instanceof CompoundStatement)){
										t = t.getParent();
									}
									CompoundStatement attParent = (CompoundStatement)t;
									attParent.addStatementAfter((Statement)att, retranspose_func_stmt.clone());
								}

							}
							if(att.containsAnnotation(ACCAnnotation.class, "device")){
								boolean flag = false;
								Set<Object> deviceList = annot.get("device");
								for(Object subArray : deviceList){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											deviceList.add(transpose_dec);
											//System.out.println("transsym " + transpose_sym.getSymbolName());
											deviceList.remove(subArray);
											flag = true;
											break;
										}
									}
									if(subArray instanceof SubArray){
										SubArray sa = (SubArray)subArray;
										if(sa.getArrayName().equals(sArrayNameEx)){
											sa.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(sa.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												sa.setRange(i, ExList.get(i));
											}
											flag = true;
											break;
										}
									}
								}
								if(flag){
									Traversable t = att;
									while ( !(t instanceof CompoundStatement)){
										t = t.getParent();
									}
									CompoundStatement attParent = (CompoundStatement)t;
									attParent.addStatementBefore((Statement)att, transpose_func_stmt.clone());
								}
							}
						}
					}
				}
			}



			//			//////////////////////////////////////////
			//			// Performs "redim" transformation. //
			//			//////////////////////////////////////////
			//			for( ARCAnnotation tAnnot : transposeAnnots ) {
			//
			//				Annotatable at = tAnnot.getAnnotatable();
			//				Set<SubArrayWithConf> sArrayConfSet = tAnnot.get("redim");
			//
			//				Statement redimStmt = (Statement)at;
			//				CompoundStatement redimBody = null;
			//				Statement firstNonDeclStmt = null;
			//				if (redimStmt instanceof CompoundStatement ){
			//					redimBody = (CompoundStatement)redimStmt;
			//					firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(redimStmt);
			//				} else {
			//					System.err.println("error: transform directive shuld have a region {}"); //[FIXME] English
			//				}
			//
			//				Procedure redimParentProc = IRTools.getParentProcedure(redimStmt);
			//				CompoundStatement redimParentBody = redimParentProc.getBody();
			//
			//				//step1 
			//				ACCAnnotation compAnnot = null;
			//				compAnnot = redimStmt.getAnnotation(ACCAnnotation.class, "kernels");
			//				if( compAnnot == null ){
			//					compAnnot = redimStmt.getAnnotation(ACCAnnotation.class, "parallel");
			//				}
			//				ACCAnnotation pcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(redimStmt, ACCAnnotation.class, ACCAnnotation.computeRegions, false, funcCallList, null);
			//				if( pcompAnnot != null ) {
			//					// [FIXME] is it correct?
			//					Tools.exit("[ERROR in DataLayoutTransformation()] transform pragma can not exist inside of " +
			//							"any compute regions (kerenls/parallel regions):\n" +
			//							"Enclosing procedure: " + redimParentProc.getSymbolName() + "\nOpenACC annotation: " + tAnnot + "\n");
			//				}
			//
			//
			//				List<FunctionCall> funcCallList2 = IRTools.getFunctionCalls(redimBody);
			//
			//				for(SubArrayWithConf sArrayConf : sArrayConfSet ) {
			//					//step2
			//
			//					SubArray sArray = sArrayConf.getSubArray();
			//					ConfList confL = sArrayConf.getConfList(0);
			//					List<Expression> confExList = confL.getConfigList();
			//					int dimension = confExList.size();
			//
			//					
			//					
			//					List<Expression> startIndecies = sArray.getStartIndices();
			//					List<Expression> exList = sArray.getLengths();
			//					Expression sArrayNameEx = sArray.getArrayName();
			//					String sArrayName = sArray.getArrayName().toString();
			//					String transposedArrayName = "transposed__" + sArrayName;
			//					System.out.println("size = " + dimension + " name = " + sArrayName.toString());
			//
			//					Set<Symbol> symSet = redimParentBody.getSymbols();
			//					Symbol transpose_sym = AnalysisTools.findsSymbol(symSet, sArrayName);
			//					System.out.println(transpose_sym.getSymbolName());
			//					System.out.println(transpose_sym.getArraySpecifiers().get(0).toString());
			//					System.out.println(transpose_sym.getTypeSpecifiers().get(0).toString());
			//
			//
			//					List<ACCAnnotation> annotationListACC =
			//							AnalysisTools.collectPragmas(redimStmt, ACCAnnotation.class);
			//
			//					/* get host variable */
			//					Expression hostVar = new Identifier(transpose_sym);
			//					//					Set<Symbol> outSymbols = new HashSet<Symbol>();
			//
			//					/* New variable declaration and allocation */
			//					Expression newVar = null;
			//					// Create a new transpose-host variable.
			//					// The type of the transpose-host symbol should be a pointer type 
			//					VariableDeclarator transpose_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
			//							new NameID(transposedArrayName));
			//					VariableDeclaration transpose_decl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
			//							transpose_declarator);
			//					newVar = new Identifier(transpose_declarator);
			//					Symbol transpose_dec = transpose_declarator;
			//					//transpose_sym = transpose_declarator;
			//
			//					//step5 replace all variables
			//					Statement enclosingCompRegion = null;
			//					for(ACCAnnotation annot : annotationListACC)
			//					{
			//						Annotatable att = annot.getAnnotatable();
			//						if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
			//								|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
			//							enclosingCompRegion = (Statement)att;
			//							IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
			//							//IRTools.replaceAll(cStmt, hostVar, newVar);
			//						}
			//
			//						for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
			//							//							Set <Set<SubArray>> transposeSArraySet = annot.get(dataClause);
			//							Set<SubArray> transposeSArraySet = annot.get(dataClause);
			//							if(transposeSArraySet != null){
			//								System.out.println(transposeSArraySet.toString());
			//								for(SubArray subArray : transposeSArraySet){
			//									System.out.println(subArray.toString());
			//									System.out.println(subArray.getArrayName().toString());
			//									System.out.println(sArrayName);
			//									if(subArray.getArrayName().equals(sArrayNameEx)){
			//										SubArray newVarSubArray = subArray.clone();
			//										newVarSubArray.setArrayName(newVar.clone());
			//										int[] newIndexIndex = new int[dimension];
			//										for(int i = 0;i < dimension;i++){
			//											newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//										}
			//										List < List<Expression>> ExList = new ArrayList< List<Expression>>();
			//										for(int i = 0;i < dimension;i++){
			//											ExList.add(newVarSubArray.getRange(newIndexIndex[i]));
			//										}
			//										for(int i = 0;i < dimension;i++){
			//											newVarSubArray.setRange(i, ExList.get(i));
			//										}
			//										transposeSArraySet.add(newVarSubArray);
			//									}
			//									//}
			//									//}
			//								}
			//							}
			//
			//						}
			//						for(String accinter : ACCAnnotation.internalDataClauses){  
			//							//Set<Symbol> transposeSArrays = annot.get(accinter);
			//							Set<Object> transposeSArrays = annot.get(accinter);
			//							//Set<SubArray> transposeSArrays = annot.get(accinter);
			//							if(transposeSArrays != null){
			//								System.out.println(accinter + " " + transposeSArrays.toString() + " " + transposeSArrays.size());
			//								//								for(SubArray subArray : transposeSArrays){ //[FIXME]
			//								for(Object subArray : transposeSArrays){
			//									if(subArray instanceof Symbol){
			//										if(((Symbol)subArray).equals(transpose_sym)){
			//											transposeSArrays.add(transpose_dec);
			//											//transposeSArrays.remove(subArray);
			//											break;
			//										}
			//									}
			//								}
			//							}
			//						}
			//					}
			//					if(enclosingCompRegion == null) {
			//						continue;
			//					}
			//
			//
			//					List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
			//					for(Expression e : newVarExpressions){
			//						if(e.getParent().getClass().equals(ArrayAccess.class)){
			//							ArrayAccess aa = (ArrayAccess)e.getParent();
			//							//							System.out.println("array access name" + aa.getArrayName().toString());
			//							List<Expression> oldIndices = aa.getIndices();
			//							int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
			//							Expression newIndex = oldIndices.get(newIndexIndex);
			//							for(int i = 1; i < dimension;i++){
			//								newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//								newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
			//										BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
			//							}
			//							List<Expression> newIndices = new ArrayList<Expression>();
			//							newIndices.add(newIndex);
			//							aa.setIndices(newIndices);
			//						}
			//					}
			//
			//					//					List<ArrayAccess> array_accesses = AnalysisTools.getArrayAccesses(newVar);
			//					//					for(ArrayAccess aa : array_accesses){
			//					//					System.out.println("array access name" + aa.getArrayName().toString());
			//					//					}
			//
			//					redimBody.addDeclaration(transpose_decl);
			//					//Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
			//					Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
			//					for(Expression ex : exList){
			//						size = new BinaryExpression(size, BinaryOperator.MULTIPLY, ex);
			//					}
			//					FunctionCall malloc_call = new FunctionCall(new NameID("malloc"));
			//					malloc_call.addArgument(size);
			//					List<Specifier> specs = new ArrayList<Specifier>(4);
			//					//specs.addAll(transpose_dec.getTypeSpecifiers());
			//					specs.addAll(transpose_dec.getTypeSpecifiers());
			//					Statement malloc_stmt = new ExpressionStatement(new AssignmentExpression(newVar.clone(),
			//							AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
			//					redimBody.addStatementBefore(firstNonDeclStmt, malloc_stmt);
			//
			//
			//					//size and rule information
			//					Expression sizeInfo = null;
			//					Expression ruleInfo = null;
			//					NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
			//					NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
			//					List<Expression> arryDimList = new ArrayList<Expression>();
			//					arryDimList.add(new IntegerLiteral(dimension));
			//					ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
			//					VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
			//					VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
			//					VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
			//					VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
			//					sizeInfo = new Identifier(sizeInfo_declarator);
			//					ruleInfo = new Identifier(ruleInfo_declarator);
			//					redimBody.addDeclaration(sizeInfo_decl);
			//					redimBody.addDeclaration(ruleInfo_decl);
			//					for(int i = 0;i < dimension;i++){
			//						Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
			//						Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
			//						redimBody.addStatementBefore(firstNonDeclStmt,sizeInfo_Statement);
			//						redimBody.addStatementBefore(firstNonDeclStmt,ruleInfo_Statement);
			//					}
			//
			//					//step3 make transpose_kernel function for CPU
			//					NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
			//					//CompoundStatement funcBody = new CompoundStatement();
			//					//Function transpose_kernel = new Function();
			//					FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
			//					Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
			//					redimBody.addStatementBefore(firstNonDeclStmt, transpose_func_stmt);
			//					transpose_funcCall.addArgument(newVar.clone());
			//					//transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					transpose_funcCall.addArgument(sizeInfo.clone());
			//					transpose_funcCall.addArgument(ruleInfo.clone());
			//					transpose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//					//step4 make re_transpose_kernel function for CPU
			//					NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
			//					FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
			//					Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
			//					redimBody.addStatement(retranspose_func_stmt);
			//					//retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					retranspose_funcCall.addArgument(newVar.clone());
			//					retranspose_funcCall.addArgument(sizeInfo.clone());
			//					retranspose_funcCall.addArgument(ruleInfo.clone());
			//					retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//
			//					//step5 make transpose_kernel function for GPU
			//					NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
			//					FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
			//					Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
			//					gpu_transpose_funcCall.addArgument(newVar.clone());
			//					//gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_transpose_funcCall.addArgument(sizeInfo.clone());
			//					gpu_transpose_funcCall.addArgument(ruleInfo.clone());
			//					gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//					//step4 make re_transpose_kernel function for GPU
			//					NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
			//					FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
			//					Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
			//					//gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_retranspose_funcCall.addArgument(newVar.clone());
			//					gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
			//					gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
			//					gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//
			//					//step free
			//					FunctionCall free_call = new FunctionCall(new NameID("free"));
			//					free_call.addArgument(newVar.clone());
			//					Statement free_stmt = new ExpressionStatement(free_call);
			//					redimBody.addStatement(free_stmt);
			//
			//
			//					//step   function calls
			//					for(FunctionCall func : funcCallList2){
			//						System.out.println(func.toString() + ", " + func.getStatement().toString());
			//
			//						Traversable t = func.getParent();
			//						while ( !(t instanceof CompoundStatement)){
			//							t = t.getParent();
			//						}
			//						CompoundStatement funcParent = (CompoundStatement)t;
			//						System.out.println(funcParent.toString());
			//						ACCAnnotation useDeviceDirective = new ACCAnnotation("host_data", null);
			//						List<Expression> useDeviceList = new ArrayList<Expression>();
			//						useDeviceList.add(newVar);
			//						useDeviceList.add(hostVar);
			//						useDeviceDirective.put("use_device", useDeviceList);
			//						Statement useDeviceStmt = new AnnotationStatement(useDeviceDirective);
			//						CompoundStatement useDeviceRegion = new CompoundStatement(); 
			//						CompoundStatement re_useDeviceRegion = new CompoundStatement(); 
			//						useDeviceRegion.addStatement(gpu_retranspose_func_stmt.clone());
			//						re_useDeviceRegion.addStatement(gpu_transpose_func_stmt.clone());
			//						funcParent.addStatementBefore(func.getStatement(), useDeviceStmt.clone());
			//						funcParent.addStatementBefore(func.getStatement(), useDeviceRegion);
			//						funcParent.addStatementAfter(func.getStatement(), re_useDeviceRegion);
			//						funcParent.addStatementAfter(func.getStatement(), useDeviceStmt.clone());
			////						funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
			//					}
			//
			//
			//					//step update directive 
			//					for(ACCAnnotation annot : annotationListACC)
			//					{
			//						Annotatable att = annot.getAnnotatable();
			//						if(att.containsAnnotation(ACCAnnotation.class, "update")){
			//							if(att.containsAnnotation(ACCAnnotation.class, "host")){
			//								boolean flag = false;
			//								Set<Object> hostList = annot.get("host");
			//								for(Object subArray : hostList){
			//									if(subArray instanceof Symbol){
			//										if(((Symbol)subArray).equals(transpose_sym)){
			//											hostList.add(transpose_dec);
			//											//System.out.println("transsym " + transpose_sym.getSymbolName());
			//											hostList.remove(subArray);
			//											flag = true;
			//											break;
			//										}
			//									}
			//									if(subArray instanceof SubArray){
			//										SubArray sa = (SubArray)subArray;
			//										if(sa.getArrayName().equals(sArrayNameEx)){
			//											sa.setArrayName(newVar.clone());
			//											int[] newIndexIndex = new int[dimension];
			//											for(int i = 0;i < dimension;i++){
			//												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//											}
			//											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
			//											for(int i = 0;i < dimension;i++){
			//												ExList.add(sa.getRange(newIndexIndex[i]));
			//											}
			//											for(int i = 0;i < dimension;i++){
			//												sa.setRange(i, ExList.get(i));
			//											}
			//											flag = true;
			//											break;
			//										}
			//									}
			//								}
			//								if(flag){
			//									Traversable t = att;
			//									while ( !(t instanceof CompoundStatement)){
			//										t = t.getParent();
			//									}
			//									CompoundStatement attParent = (CompoundStatement)t;
			//									attParent.addStatementAfter((Statement)att, retranspose_func_stmt.clone());
			//								}
			//
			//							}
			//							if(att.containsAnnotation(ACCAnnotation.class, "device")){
			//								boolean flag = false;
			//								Set<Object> deviceList = annot.get("device");
			//								for(Object subArray : deviceList){
			//									if(subArray instanceof Symbol){
			//										if(((Symbol)subArray).equals(transpose_sym)){
			//											deviceList.add(transpose_dec);
			//											//System.out.println("transsym " + transpose_sym.getSymbolName());
			//											deviceList.remove(subArray);
			//											flag = true;
			//											break;
			//										}
			//									}
			//									if(subArray instanceof SubArray){
			//										SubArray sa = (SubArray)subArray;
			//										if(sa.getArrayName().equals(sArrayNameEx)){
			//											sa.setArrayName(newVar.clone());
			//											int[] newIndexIndex = new int[dimension];
			//											for(int i = 0;i < dimension;i++){
			//												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//											}
			//											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
			//											for(int i = 0;i < dimension;i++){
			//												ExList.add(sa.getRange(newIndexIndex[i]));
			//											}
			//											for(int i = 0;i < dimension;i++){
			//												sa.setRange(i, ExList.get(i));
			//											}
			//											flag = true;
			//											break;
			//										}
			//									}
			//								}
			//								if(flag){
			//									Traversable t = att;
			//									while ( !(t instanceof CompoundStatement)){
			//										t = t.getParent();
			//									}
			//									CompoundStatement attParent = (CompoundStatement)t;
			//									attParent.addStatementBefore((Statement)att, transpose_func_stmt.clone());
			//								}
			//
			//							}
			//						}
			//					}
			//				}
			//			}
		}
		SymbolTools.linkSymbol(program);
	}







	public void start_20141107() {
		//for each transform, 
		//step0: ([TODO]): if ifcond exists and its argument is 0, skip the current transform region. 
		//step1: If the transform region is inside of compute regions, error in current version. 
		//step2: if the transposed array isn't exist device memory, step2.1. else, step2.2
		//step2.1: Create new transposed array
		//step2.2: 
		//step3: transpose function call
		//step4: re-transpose function call
		//step5: replace the variables
		//step5.5: recalculate the index of the new array
		//step6: ([TODO]) deal with function calls
		//step7: ([TODO]) deal with copy clause and update directive in the transform region.

		List<ARCAnnotation> transformAnnots = IRTools.collectPragmas(program, ARCAnnotation.class, "transform");
		if( transformAnnots != null ) {
			List<ARCAnnotation> transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redimAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expandAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redim_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expand_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> transpose_expandAnnots = new LinkedList<ARCAnnotation>();

			for( ARCAnnotation arcAnnot : transformAnnots ) {
				if( arcAnnot.containsKey("transpose") ) {
					transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim") ) {
					redimAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand") ) {
					expandAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim_transpose") ) {
					redim_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand_transpose") ) {
					expand_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("transpose_expand") ) {
					transpose_expandAnnots.add(arcAnnot);
				}
			}


			List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);

			//////////////////////////////////////////
			// Performs "transpose" transformation. //
			//////////////////////////////////////////
			for( ARCAnnotation tAnnot : transposeAnnots ) {

				Annotatable at = tAnnot.getAnnotatable();
				Set<SubArrayWithConf> sArrayConfSet = tAnnot.get("transpose");

				Statement transposeRegion = (Statement)tAnnot.getAnnotatable();
				//				CompoundStatement cStmt = (CompoundStatement)transposeRegion.getParent();
				//				CompoundStatement cStmt = (CompoundStatement)transposeRegion.getChildren().get(0);
				CompoundStatement cStmt = null;
				Statement firstNonDeclStmt = null;
				if (transposeRegion instanceof CompoundStatement ){
					cStmt = (CompoundStatement)transposeRegion;
					firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(transposeRegion);
				} else {
					// [FIXME] is it correct?
					cStmt = new CompoundStatement();
					firstNonDeclStmt = transposeRegion;
				}



				Procedure cProc = IRTools.getParentProcedure(transposeRegion);
				CompoundStatement cBody = cProc.getBody();

				System.out.println("test");

				//step1 
				ACCAnnotation compAnnot = null;
				compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "kernels");
				if( compAnnot == null ){
					compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "parallel");
				}
				ACCAnnotation pcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(transposeRegion, ACCAnnotation.class, ACCAnnotation.computeRegions, false, funcCallList, null);
				if( pcompAnnot != null ) {
					Tools.exit("[ERROR in DataLayoutTransformation()] transform pragma can not exist inside of " +
							"any compute regions (kerenls/parallel regions):\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + tAnnot + "\n");
				}



				for(SubArrayWithConf sArrayConf : sArrayConfSet ) {

					//step2.0 get pragma information
					SubArray sArray = sArrayConf.getSubArray();
					//In transpose clause, each SubArrayWithConf has only on configuration list ([2,1,3]).
					//CF: redim_transpose, each SubArrayWithConf will have two configuration lists 
					//(one for redim (e.g, [X,Y,Z]), and the other for transpose(e.g., [2,1,3])).
					ConfList confL = sArrayConf.getConfList(0);
					//...
					List<Expression> startIndecies = sArray.getStartIndices();
					List<Expression> exList = sArray.getLengths();
					Expression size_1D = new IntegerLiteral(1);
					for(Expression ex : exList){
						size_1D = Symbolic.multiply(size_1D, ex);
					}
					int dimension = sArray.getArrayDimension();
					Expression sArrayNameEx = sArray.getArrayName();
					String sArrayName = sArray.getArrayName().toString();
					String transposedArrayName = "transposed__" + sArrayName;
					System.out.println("size = " + dimension + " name = " + sArrayName.toString());

					Set<Symbol> symSet = cBody.getSymbols();
					Symbol transpose_sym = AnalysisTools.findsSymbol(symSet, sArrayName);
					System.out.println(transpose_sym.getSymbolName());
					System.out.println(transpose_sym.getArraySpecifiers().get(0).toString());
					System.out.println(transpose_sym.getTypeSpecifiers().get(0).toString());

					/* get host variable */
					Expression hostVar = new Identifier(transpose_sym);
					//					Set<Symbol> outSymbols = new HashSet<Symbol>();


					/* New variable declaration and allocation */
					Expression newVar = null;
					// Create a new transpose-host variable.
					// The type of the transpose-host symbol should be a pointer type 
					VariableDeclarator transpose_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(transposedArrayName));
					VariableDeclaration transpose_decl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							transpose_declarator);
					//					VariableDeclarator transpose_declarator2 = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
					//							new NameID(transposedArrayName));
					//					VariableDeclaration transpose_decl2 = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
					//							transpose_declarator2); // [FIXME]
					newVar = new Identifier(transpose_declarator);
					Symbol transpose_dec = transpose_declarator;


					/* make transposition kernels */

					//size and rule information
					Expression sizeInfo = null;
					Expression ruleInfo = null;
					NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
					NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
					List<Expression> arryDimList = new ArrayList<Expression>();
					arryDimList.add(new IntegerLiteral(dimension));
					ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
					VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
					VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
					VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
					VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
					sizeInfo = new Identifier(sizeInfo_declarator);
					ruleInfo = new Identifier(ruleInfo_declarator);

					//transpose_kernel function for CPU
					NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
					FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
					Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
					transpose_funcCall.addArgument(newVar.clone());
					transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					transpose_funcCall.addArgument(sizeInfo.clone());
					transpose_funcCall.addArgument(ruleInfo.clone());
					transpose_funcCall.addArgument(new IntegerLiteral(dimension));

					//re_transpose_kernel function for CPU
					NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
					FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
					Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
					retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					retranspose_funcCall.addArgument(newVar.clone());
					retranspose_funcCall.addArgument(sizeInfo.clone());
					retranspose_funcCall.addArgument(ruleInfo.clone());
					retranspose_funcCall.addArgument(new IntegerLiteral(dimension));

					//declarations for device address pointer
					Expression gpuVar = null; 
					Expression gpuNewVar = null;
					String gpuVarName = "deviceAddressFor__" + sArrayName;
					String gpuNewVarName = "transposedDeviceAddressFor__" + sArrayName;
					VariableDeclarator gpuVarDeclarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(gpuVarName));
					VariableDeclarator gpuNewVarDeclarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(gpuNewVarName));
					VariableDeclaration gpuVarDecl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							gpuVarDeclarator);
					VariableDeclaration gpuNewVarDecl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							gpuNewVarDeclarator);
					gpuVar = new Identifier(gpuVarDeclarator);
					gpuNewVar = new Identifier(gpuNewVarDeclarator);
					cBody.addDeclaration(gpuVarDecl);
					cBody.addDeclaration(gpuNewVarDecl);


					//transpose_kernel function for GPU
					NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
					FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
					Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
					gpu_transpose_funcCall.addArgument(gpuNewVar.clone());
					gpu_transpose_funcCall.addArgument(gpuVar.clone());
					gpu_transpose_funcCall.addArgument(sizeInfo.clone());
					gpu_transpose_funcCall.addArgument(ruleInfo.clone());
					gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));

					//re_transpose_kernel function for GPU
					NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
					FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
					Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
					gpu_retranspose_funcCall.addArgument(gpuVar.clone());
					gpu_retranspose_funcCall.addArgument(gpuNewVar.clone());
					gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
					gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
					gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));




					//step2 check if the transposed array is already exist on the device or not,
					NameID checkPresentName = new NameID("acc_is_present");
					FunctionCall checkPresentFuncCall = new FunctionCall(checkPresentName);
					checkPresentFuncCall.addArgument(hostVar);
					checkPresentFuncCall.addArgument(size_1D);
					Expression checkPresent = new BinaryExpression(checkPresentFuncCall, BinaryOperator.COMPARE_EQ, new IntegerLiteral(1));
					CompoundStatement ifPresentTrueBodyStmt = new CompoundStatement();
					CompoundStatement ifPresentElseBodyStmt = new CompoundStatement();
					Statement checkPresentStmt = new IfStatement(checkPresent, ifPresentTrueBodyStmt, ifPresentElseBodyStmt);

					//insert new variable declaration and malloc statement
					//ifPresentTrueBodyStmt.addDeclaration(transpose_decl);
					//ifPresentElseBodyStmt.addDeclaration(transpose_decl2);
					cBody.addDeclaration(transpose_decl);
					//SymbolTools.linkSymbol(ifPresentTrueBodyStmt);
					//SymbolTools.linkSymbol(ifPresentElseBodyStmt);

					Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
					for(Expression ex : exList){
						size = new BinaryExpression(size, BinaryOperator.MULTIPLY, ex.clone());
					}
					FunctionCall malloc_call = new FunctionCall(new NameID("malloc"));
					malloc_call.addArgument(size);
					List<Specifier> specs = new ArrayList<Specifier>(4);
					//specs.addAll(transpose_dec.getTypeSpecifiers());
					specs.addAll(transpose_dec.getTypeSpecifiers());
					Statement malloc_stmt = new ExpressionStatement(new AssignmentExpression(newVar.clone(),
							AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
					ifPresentTrueBodyStmt.addStatement(malloc_stmt.clone());
					ifPresentElseBodyStmt.addStatement(malloc_stmt.clone());

					//if already exist, add a data directive for new device variable
					ACCAnnotation dataDirective = new ACCAnnotation("data", null);
					SubArray newDeviceArray = new SubArray(newVar.clone(), new IntegerLiteral(0), size_1D.clone());
					dataDirective.put("create", newDeviceArray.clone());
					Statement dataDirectiveStmt = new AnnotationStatement(dataDirective);
					CompoundStatement dataDirectiveRegion = new CompoundStatement();
					dataDirectiveRegion.addStatement(cStmt.clone());
					ifPresentTrueBodyStmt.addStatement(dataDirectiveStmt.clone());
					ifPresentTrueBodyStmt.addStatement(dataDirectiveRegion.clone());
					ifPresentElseBodyStmt.addStatement(cStmt.clone());
					//cStmt.addStatementBefore(firstNonDeclStmt, checkPresentStmt);
					cStmt.swapWith(checkPresentStmt);


					//step2.1 already exist 

					List<ARCAnnotation> trueAnnotationListARC =
							IRTools.collectPragmas(ifPresentTrueBodyStmt, ARCAnnotation.class, "transform");
					for(ARCAnnotation trueTransAnnot : trueAnnotationListARC){
						Statement trueTransposeRegion = (Statement)trueTransAnnot.getAnnotatable();
						CompoundStatement trueCStmt = null;
						Statement trueFirstNonDeclStmt = null;
						if (trueTransposeRegion instanceof CompoundStatement ){
							trueCStmt = (CompoundStatement)trueTransposeRegion;
							trueFirstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(trueTransposeRegion);
						} else {
							// [FIXME] is it correct?
							trueCStmt = new CompoundStatement();
							trueFirstNonDeclStmt = trueTransposeRegion;
						}
						List<ACCAnnotation> annotationListACC =
								AnalysisTools.collectPragmas(trueTransposeRegion, ACCAnnotation.class);
						List<FunctionCall> elseFuncCallList = IRTools.getFunctionCalls(trueCStmt);

						//step5 replace all variables
						Statement enclosingCompRegion = null;
						for(ACCAnnotation annot : annotationListACC)
						{
							Annotatable att = annot.getAnnotatable();
							if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
									|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
								enclosingCompRegion = (Statement)att;
								IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
							}

							for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
								Set<SubArray> transposeSArraySet = annot.get(dataClause);
								if(transposeSArraySet != null){
									System.out.println(transposeSArraySet.toString());
									for(SubArray subArray : transposeSArraySet){
										System.out.println(subArray.toString());
										System.out.println(subArray.getArrayName().toString());
										System.out.println(sArrayName);
										if(subArray.getArrayName().equals(sArrayNameEx)){
											SubArray newVarSubArray = subArray.clone();
											newVarSubArray.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(newVarSubArray.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												newVarSubArray.setRange(i, ExList.get(i));
											}
											transposeSArraySet.add(newVarSubArray);
											//transposeSArraySet.remove(subArray);
											break;
										}
									}
								}

							}
							for(String accinter : ACCAnnotation.internalDataClauses){  
								//Set<Symbol> transposeSArrays = annot.get(accinter);
								Set<Object> transposeSArrays = annot.get(accinter);
								//Set<SubArray> transposeSArrays = annot.get(accinter);
								if(transposeSArrays != null){
									System.out.println(accinter + " " + transposeSArrays.toString() + " " + transposeSArrays.size());
									//								for(SubArray subArray : transposeSArrays){ //[FIXME]
									for(Object subArray : transposeSArrays){
										if(subArray instanceof Symbol){
											if(((Symbol)subArray).equals(transpose_sym)){
												transposeSArrays.add(transpose_dec);
												//transposeSArrays.remove(subArray);
												break;
											}
										}
									}
								}
							}
						}
						if(enclosingCompRegion == null) {
							continue;
						}


						List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
						for(Expression e : newVarExpressions){
							if(e.getParent().getClass().equals(ArrayAccess.class)){
								ArrayAccess aa = (ArrayAccess)e.getParent();
								//							System.out.println("array access name" + aa.getArrayName().toString());
								List<Expression> oldIndices = aa.getIndices();
								int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
								Expression newIndex = oldIndices.get(newIndexIndex);
								for(int i = 1; i < dimension;i++){
									newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
									newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
											BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
								}
								List<Expression> newIndices = new ArrayList<Expression>();
								newIndices.add(newIndex);
								aa.setIndices(newIndices);
							}
						}



						//						//size and rule information
						//						Expression sizeInfo = null;
						//						Expression ruleInfo = null;
						//						NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
						//						NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
						//						List<Expression> arryDimList = new ArrayList<Expression>();
						//						arryDimList.add(new IntegerLiteral(dimension));
						//						ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
						//						VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
						//						VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
						//						VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
						//						VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
						//						sizeInfo = new Identifier(sizeInfo_declarator);
						//						ruleInfo = new Identifier(ruleInfo_declarator);
						trueCStmt.addDeclaration(sizeInfo_decl.clone());
						trueCStmt.addDeclaration(ruleInfo_decl.clone());
						for(int i = 0;i < dimension;i++){
							Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
							Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
							trueCStmt.addStatementBefore(trueFirstNonDeclStmt,sizeInfo_Statement);
							trueCStmt.addStatementBefore(trueFirstNonDeclStmt,ruleInfo_Statement);
						}

						//						//step3 make transpose_kernel function for CPU
						//						NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
						//						//CompoundStatement funcBody = new CompoundStatement();
						//						//Function transpose_kernel = new Function();
						//						FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
						//						Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
						//						//trueCStmt.addStatementBefore(trueFirstNonDeclStmt, transpose_func_stmt);
						//						transpose_funcCall.addArgument(newVar.clone());
						//						transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						transpose_funcCall.addArgument(sizeInfo.clone());
						//						transpose_funcCall.addArgument(ruleInfo.clone());
						//						transpose_funcCall.addArgument(new IntegerLiteral(dimension));
						//	
						//						//step4 make re_transpose_kernel function for CPU
						//						NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
						//						FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
						//						Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
						//						//trueCStmt.addStatement(retranspose_func_stmt);
						//						retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						retranspose_funcCall.addArgument(newVar.clone());
						//						retranspose_funcCall.addArgument(sizeInfo.clone());
						//						retranspose_funcCall.addArgument(ruleInfo.clone());
						//						retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
						//	
						//
						//

						CompoundStatement presentErrorCode = new CompoundStatement();
						FunctionCall printfCall = new FunctionCall(new NameID("printf"));
						printfCall.addArgument(new StringLiteral("[ERROR] GPU memory for the host variable, "+hostVar.toString()+
								", does not exist. \\n"));
						presentErrorCode.addStatement(new ExpressionStatement(printfCall));
						printfCall = new FunctionCall(new NameID("printf"));
						printfCall.addArgument(new StringLiteral("Enclosing annotation: \\n" + trueTransAnnot.toString() + " \\n"));
						presentErrorCode.addStatement(new ExpressionStatement(printfCall));
						FunctionCall exitCall = new FunctionCall(new NameID("exit"));
						exitCall.addArgument(new IntegerLiteral(1));
						presentErrorCode.addStatement(new ExpressionStatement(exitCall));

						NameID getGpuVarAddressName = new NameID("HI_get_device_address");
						FunctionCall getGpuVarAddressFuncCall = new FunctionCall(getGpuVarAddressName);
						getGpuVarAddressFuncCall.addArgument(hostVar.clone());
						getGpuVarAddressFuncCall.addArgument(gpuVar.clone());
						Expression asyncID = new NameID("acc_async_noval");
						getGpuVarAddressFuncCall.addArgument(asyncID.clone());
						Expression getGpuVarAddressExp = new BinaryExpression(getGpuVarAddressFuncCall, BinaryOperator.COMPARE_NE, new NameID("HI_success"));
						IfStatement getGpuVarAddressIfStmt = new IfStatement(getGpuVarAddressExp.clone(), presentErrorCode.clone());
						trueCStmt.addStatementBefore(trueFirstNonDeclStmt, getGpuVarAddressIfStmt.clone());


						CompoundStatement transposedPresentErrorCode = new CompoundStatement();
						FunctionCall transposedPrintfCall = new FunctionCall(new NameID("printf"));
						transposedPrintfCall.addArgument(new StringLiteral("[ERROR] GPU memory for the host variable, "+newVar.toString()+
								", does not exist. \\n"));
						transposedPresentErrorCode.addStatement(new ExpressionStatement(transposedPrintfCall));
						transposedPrintfCall = new FunctionCall(new NameID("printf"));
						transposedPrintfCall.addArgument(new StringLiteral("Enclosing annotation: \\n" + trueTransAnnot.toString() + " \\n"));
						transposedPresentErrorCode.addStatement(new ExpressionStatement(transposedPrintfCall));
						FunctionCall transposedExitCall = new FunctionCall(new NameID("exit"));
						transposedExitCall.addArgument(new IntegerLiteral(1));
						transposedPresentErrorCode.addStatement(new ExpressionStatement(transposedExitCall));

						NameID getGpuNewVarAddressName = new NameID("HI_get_device_address");
						FunctionCall getGpuNewVarAddressFuncCall = new FunctionCall(getGpuNewVarAddressName);
						getGpuNewVarAddressFuncCall.addArgument(newVar.clone());
						getGpuNewVarAddressFuncCall.addArgument(gpuNewVar.clone());
						getGpuNewVarAddressFuncCall.addArgument(asyncID.clone());
						Expression getGpuNewVarAddressExp = new BinaryExpression(getGpuNewVarAddressFuncCall, BinaryOperator.COMPARE_NE, new NameID("HI_success"));
						IfStatement getGpuNewVarAddressIfStmt = new IfStatement(getGpuNewVarAddressExp.clone(), transposedPresentErrorCode.clone());
						trueCStmt.addStatementBefore(trueFirstNonDeclStmt, getGpuNewVarAddressIfStmt.clone());		

						//						//step5 make transpose_kernel function for GPU
						//						NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
						//						FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
						//						Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
						trueCStmt.addStatementBefore(trueFirstNonDeclStmt, gpu_transpose_func_stmt);
						//						gpu_transpose_funcCall.addArgument(newVar.clone());
						//						gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						gpu_transpose_funcCall.addArgument(sizeInfo.clone());
						//						gpu_transpose_funcCall.addArgument(ruleInfo.clone());
						//						gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));
						//	
						//						//step4 make re_transpose_kernel function for GPU
						//						NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
						//						FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
						//						Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
						trueCStmt.addStatement(gpu_retranspose_func_stmt);
						//						gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						gpu_retranspose_funcCall.addArgument(newVar.clone());
						//						gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
						//						gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
						//						gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
						//	
						//	
						//						TransformTools.getNewTempIndex(((ForLoop)enclosingCompRegion).getBody())

						//step free
						FunctionCall free_call = new FunctionCall(new NameID("free"));
						free_call.addArgument(newVar.clone());
						Statement free_stmt = new ExpressionStatement(free_call);
						trueCStmt.addStatement(free_stmt);


						//step   function calls
						for(FunctionCall func : elseFuncCallList){
							System.out.println(func.toString() + ", " + func.getStatement().toString());

							Traversable t = func.getParent();
							while ( !(t instanceof CompoundStatement)){
								t = t.getParent();
							}
							CompoundStatement funcParent = (CompoundStatement)t;
							funcParent.addStatementBefore(func.getStatement(), gpu_retranspose_func_stmt.clone());
							funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
							//							ACCAnnotation useDeviceDirective = new ACCAnnotation("host_data", null);
							//							List<Expression> useDeviceList = new ArrayList<Expression>();
							//							useDeviceList.add(newVar);
							//							useDeviceList.add(hostVar);
							//							useDeviceDirective.put("use_device", useDeviceList);
							//							Statement useDeviceStmt = new AnnotationStatement(useDeviceDirective);
							//							CompoundStatement useDeviceRegion = new CompoundStatement(); 
							//							CompoundStatement re_useDeviceRegion = new CompoundStatement(); 
							//							useDeviceRegion.addStatement(gpu_retranspose_func_stmt.clone());
							//							re_useDeviceRegion.addStatement(gpu_transpose_func_stmt.clone());
							//							funcParent.addStatementBefore(func.getStatement(), useDeviceStmt.clone());
							//							funcParent.addStatementBefore(func.getStatement(), useDeviceRegion);
							//							funcParent.addStatementAfter(func.getStatement(), re_useDeviceRegion);
							//							funcParent.addStatementAfter(func.getStatement(), useDeviceStmt.clone());
						}


						//step update directive 
						for(ACCAnnotation annot : annotationListACC)
						{
							Annotatable att = annot.getAnnotatable();
							if(att.containsAnnotation(ACCAnnotation.class, "update")){
								if(att.containsAnnotation(ACCAnnotation.class, "host")){
									boolean flag = false;
									Set<Object> hostList = annot.get("host");
									for(Object subArray : hostList){
										if(subArray instanceof Symbol){
											if(((Symbol)subArray).equals(transpose_sym)){
												hostList.add(transpose_dec);
												hostList.remove(subArray);
												flag = true;
												break;
											}
										}
										if(subArray instanceof SubArray){
											SubArray sa = (SubArray)subArray;
											if(sa.getArrayName().equals(sArrayNameEx)){
												sa.setArrayName(newVar.clone());
												int[] newIndexIndex = new int[dimension];
												for(int i = 0;i < dimension;i++){
													newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
												}
												List < List<Expression>> ExList = new ArrayList< List<Expression>>();
												for(int i = 0;i < dimension;i++){
													ExList.add(sa.getRange(newIndexIndex[i]));
												}
												for(int i = 0;i < dimension;i++){
													sa.setRange(i, ExList.get(i));
												}
												flag = true;
												break;
											}
										}
									}
									if(flag){
										Traversable t = att;
										while ( !(t instanceof CompoundStatement)){
											t = t.getParent();
										}
										CompoundStatement attParent = (CompoundStatement)t;
										attParent.addStatementAfter((Statement)att, retranspose_func_stmt.clone());
									}

								}
								if(att.containsAnnotation(ACCAnnotation.class, "device")){
									boolean flag = false;
									Set<Object> deviceList = annot.get("device");
									for(Object subArray : deviceList){
										if(subArray instanceof Symbol){
											if(((Symbol)subArray).equals(transpose_sym)){
												deviceList.add(transpose_dec);
												//System.out.println("transsym " + transpose_sym.getSymbolName());
												deviceList.remove(subArray);
												flag = true;
												break;
											}
										}
										if(subArray instanceof SubArray){
											SubArray sa = (SubArray)subArray;
											if(sa.getArrayName().equals(sArrayNameEx)){
												sa.setArrayName(newVar.clone());
												int[] newIndexIndex = new int[dimension];
												for(int i = 0;i < dimension;i++){
													newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
												}
												List < List<Expression>> ExList = new ArrayList< List<Expression>>();
												for(int i = 0;i < dimension;i++){
													ExList.add(sa.getRange(newIndexIndex[i]));
												}
												for(int i = 0;i < dimension;i++){
													sa.setRange(i, ExList.get(i));
												}
												flag = true;
												break;
											}
										}
									}
									if(flag){
										Traversable t = att;
										while ( !(t instanceof CompoundStatement)){
											t = t.getParent();
										}
										CompoundStatement attParent = (CompoundStatement)t;
										attParent.addStatementBefore((Statement)att, transpose_func_stmt.clone());
									}
								}
							}
						}
					}

					//step2.2 else 
					List<ARCAnnotation> elseAnnotationListARC =
							IRTools.collectPragmas(ifPresentElseBodyStmt, ARCAnnotation.class, "transform");

					for(ARCAnnotation elseTransAnnot : elseAnnotationListARC){
						Statement elseTransposeRegion = (Statement)elseTransAnnot.getAnnotatable();
						CompoundStatement elseCStmt = null;
						Statement elseFirstNonDeclStmt = null;
						if (elseTransposeRegion instanceof CompoundStatement ){
							elseCStmt = (CompoundStatement)elseTransposeRegion;
							elseFirstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(elseTransposeRegion);
						} else {
							// [FIXME] is it correct?
							elseCStmt = new CompoundStatement();
							elseFirstNonDeclStmt = elseTransposeRegion;
						}

						List<ACCAnnotation> annotationListACC =
								AnalysisTools.collectPragmas(elseTransposeRegion, ACCAnnotation.class);
						List<FunctionCall> elseFuncCallList = IRTools.getFunctionCalls(elseCStmt);

						//step5 replace all variables
						Statement enclosingCompRegion = null;
						for(ACCAnnotation annot : annotationListACC)
						{
							Annotatable att = annot.getAnnotatable();
							if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
									|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
								enclosingCompRegion = (Statement)att;
								IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
							}

							for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
								Set<SubArray> transposeSArraySet = annot.get(dataClause);
								if(transposeSArraySet != null){
									System.out.println(transposeSArraySet.toString());
									for(SubArray subArray : transposeSArraySet){
										System.out.println(subArray.toString());
										System.out.println(subArray.getArrayName().toString());
										System.out.println(sArrayName);
										if(subArray.getArrayName().equals(sArrayNameEx)){
											SubArray newVarSubArray = subArray.clone();
											newVarSubArray.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(newVarSubArray.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												newVarSubArray.setRange(i, ExList.get(i));
											}
											transposeSArraySet.add(newVarSubArray);
											transposeSArraySet.remove(subArray);
											break;
										}
									}
								}
							}
							for(String accinter : ACCAnnotation.internalDataClauses){  
								Set<Object> transposeSArrays = annot.get(accinter);
								if(transposeSArrays != null){
									System.out.println(accinter + " " + transposeSArrays.toString() + " " + transposeSArrays.size());
									for(Object subArray : transposeSArrays){
										if(subArray instanceof Symbol){
											if(((Symbol)subArray).equals(transpose_sym)){
												transposeSArrays.add(transpose_dec);
												transposeSArrays.remove(subArray);
												break;
											}
										}
									}
								}
							}
						}
						if(enclosingCompRegion == null) {
							continue;
						}


						List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
						for(Expression e : newVarExpressions){
							if(e.getParent().getClass().equals(ArrayAccess.class)){
								ArrayAccess aa = (ArrayAccess)e.getParent();
								//							System.out.println("array access name" + aa.getArrayName().toString());
								List<Expression> oldIndices = aa.getIndices();
								int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
								Expression newIndex = oldIndices.get(newIndexIndex);
								for(int i = 1; i < dimension;i++){
									newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
									newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
											BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
								}
								List<Expression> newIndices = new ArrayList<Expression>();
								newIndices.add(newIndex);
								aa.setIndices(newIndices);
							}
						}


						//size and rule information
						//						Expression sizeInfo = null;
						//						Expression ruleInfo = null;
						//						NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
						//						NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
						//						List<Expression> arryDimList = new ArrayList<Expression>();
						//						arryDimList.add(new IntegerLiteral(dimension));
						//						ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
						//						VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
						//						VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
						//						VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
						//						VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
						//						sizeInfo = new Identifier(sizeInfo_declarator);
						//						ruleInfo = new Identifier(ruleInfo_declarator);
						elseCStmt.addDeclaration(sizeInfo_decl.clone());
						elseCStmt.addDeclaration(ruleInfo_decl.clone());
						for(int i = 0;i < dimension;i++){
							Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
							Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
							elseCStmt.addStatementBefore(elseFirstNonDeclStmt,sizeInfo_Statement);
							elseCStmt.addStatementBefore(elseFirstNonDeclStmt,ruleInfo_Statement);
						}

						//step3 make transpose_kernel function for CPU
						//						NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
						//						//CompoundStatement funcBody = new CompoundStatement();
						//						//Function transpose_kernel = new Function();
						//						FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
						//						Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
						elseCStmt.addStatementBefore(elseFirstNonDeclStmt, transpose_func_stmt);
						//						transpose_funcCall.addArgument(newVar.clone());
						//						//transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						transpose_funcCall.addArgument(sizeInfo.clone());
						//						transpose_funcCall.addArgument(ruleInfo.clone());
						//						transpose_funcCall.addArgument(new IntegerLiteral(dimension));
						//	
						//						//step4 make re_transpose_kernel function for CPU
						//						NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
						//						FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
						//						Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
						elseCStmt.addStatement(retranspose_func_stmt);
						//						//retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						retranspose_funcCall.addArgument(newVar.clone());
						//						retranspose_funcCall.addArgument(sizeInfo.clone());
						//						retranspose_funcCall.addArgument(ruleInfo.clone());
						//						retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
						//	
						//	
						//						//step5 make transpose_kernel function for GPU
						//						NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
						//						FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
						//						Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
						//						gpu_transpose_funcCall.addArgument(newVar.clone());
						//						//gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						gpu_transpose_funcCall.addArgument(sizeInfo.clone());
						//						gpu_transpose_funcCall.addArgument(ruleInfo.clone());
						//						gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));
						//	
						//						//step4 make re_transpose_kernel function for GPU
						//						NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
						//						FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
						//						Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
						//						//gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
						//						gpu_retranspose_funcCall.addArgument(newVar.clone());
						//						gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
						//						gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
						//						gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
						//	

						//step free
						FunctionCall free_call = new FunctionCall(new NameID("free"));
						free_call.addArgument(newVar.clone());
						Statement free_stmt = new ExpressionStatement(free_call);
						elseCStmt.addStatement(free_stmt);


						//step   function calls
						for(FunctionCall func : elseFuncCallList){
							System.out.println(func.toString() + ", " + func.getStatement().toString());

							Traversable t = func.getParent();
							while ( !(t instanceof CompoundStatement)){
								t = t.getParent();
							}
							CompoundStatement funcParent = (CompoundStatement)t;
							funcParent.addStatementBefore(func.getStatement(), gpu_retranspose_func_stmt.clone());
							funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
							//							System.out.println(funcParent.toString());
							//							ACCAnnotation useDeviceDirective = new ACCAnnotation("host_data", null);
							//							List<Expression> useDeviceList = new ArrayList<Expression>();
							//							useDeviceList.add(newVar);
							//							useDeviceList.add(hostVar);
							//							useDeviceDirective.put("use_device", useDeviceList);
							//							Statement useDeviceStmt = new AnnotationStatement(useDeviceDirective);
							//							CompoundStatement useDeviceRegion = new CompoundStatement(); 
							//							CompoundStatement re_useDeviceRegion = new CompoundStatement(); 
							//							useDeviceRegion.addStatement(gpu_retranspose_func_stmt.clone());
							//							re_useDeviceRegion.addStatement(gpu_transpose_func_stmt.clone());
							//							funcParent.addStatementBefore(func.getStatement(), useDeviceStmt.clone());
							//							funcParent.addStatementBefore(func.getStatement(), useDeviceRegion);
							//							funcParent.addStatementAfter(func.getStatement(), re_useDeviceRegion);
							//							funcParent.addStatementAfter(func.getStatement(), useDeviceStmt.clone());
							////							funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
						}


						//step update directive 
						for(ACCAnnotation annot : annotationListACC)
						{
							Annotatable att = annot.getAnnotatable();
							if(att.containsAnnotation(ACCAnnotation.class, "update")){
								if(att.containsAnnotation(ACCAnnotation.class, "host")){
									boolean flag = false;
									Set<Object> hostList = annot.get("host");
									for(Object subArray : hostList){
										if(subArray instanceof Symbol){
											if(((Symbol)subArray).equals(transpose_sym)){
												hostList.add(transpose_dec);
												//System.out.println("transsym " + transpose_sym.getSymbolName());
												hostList.remove(subArray);
												flag = true;
												break;
											}
										}
										if(subArray instanceof SubArray){
											SubArray sa = (SubArray)subArray;
											if(sa.getArrayName().equals(sArrayNameEx)){
												sa.setArrayName(newVar.clone());
												int[] newIndexIndex = new int[dimension];
												for(int i = 0;i < dimension;i++){
													newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
												}
												List < List<Expression>> ExList = new ArrayList< List<Expression>>();
												for(int i = 0;i < dimension;i++){
													ExList.add(sa.getRange(newIndexIndex[i]));
												}
												for(int i = 0;i < dimension;i++){
													sa.setRange(i, ExList.get(i));
												}
												flag = true;
												break;
											}
										}
									}
									if(flag){
										Traversable t = att;
										while ( !(t instanceof CompoundStatement)){
											t = t.getParent();
										}
										CompoundStatement attParent = (CompoundStatement)t;
										attParent.addStatementAfter((Statement)att, retranspose_func_stmt.clone());
									}

								}
								if(att.containsAnnotation(ACCAnnotation.class, "device")){
									boolean flag = false;
									Set<Object> deviceList = annot.get("device");
									for(Object subArray : deviceList){
										if(subArray instanceof Symbol){
											if(((Symbol)subArray).equals(transpose_sym)){
												deviceList.add(transpose_dec);
												//System.out.println("transsym " + transpose_sym.getSymbolName());
												deviceList.remove(subArray);
												flag = true;
												break;
											}
										}
										if(subArray instanceof SubArray){
											SubArray sa = (SubArray)subArray;
											if(sa.getArrayName().equals(sArrayNameEx)){
												sa.setArrayName(newVar.clone());
												int[] newIndexIndex = new int[dimension];
												for(int i = 0;i < dimension;i++){
													newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
												}
												List < List<Expression>> ExList = new ArrayList< List<Expression>>();
												for(int i = 0;i < dimension;i++){
													ExList.add(sa.getRange(newIndexIndex[i]));
												}
												for(int i = 0;i < dimension;i++){
													sa.setRange(i, ExList.get(i));
												}
												flag = true;
												break;
											}
										}
									}
									if(flag){
										Traversable t = att;
										while ( !(t instanceof CompoundStatement)){
											t = t.getParent();
										}
										CompoundStatement attParent = (CompoundStatement)t;
										attParent.addStatementBefore((Statement)att, transpose_func_stmt.clone());
									}
								}
							}
						}
					}
				}
			}



			//			//////////////////////////////////////////
			//			// Performs "redim" transformation. //
			//			//////////////////////////////////////////
			//			for( ARCAnnotation tAnnot : transposeAnnots ) {
			//
			//				Annotatable at = tAnnot.getAnnotatable();
			//				Set<SubArrayWithConf> sArrayConfSet = tAnnot.get("redim");
			//
			//				Statement redimStmt = (Statement)at;
			//				CompoundStatement redimBody = null;
			//				Statement firstNonDeclStmt = null;
			//				if (redimStmt instanceof CompoundStatement ){
			//					redimBody = (CompoundStatement)redimStmt;
			//					firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(redimStmt);
			//				} else {
			//					System.err.println("error: transform directive shuld have a region {}"); //[FIXME] English
			//				}
			//
			//				Procedure redimParentProc = IRTools.getParentProcedure(redimStmt);
			//				CompoundStatement redimParentBody = redimParentProc.getBody();
			//
			//				//step1 
			//				ACCAnnotation compAnnot = null;
			//				compAnnot = redimStmt.getAnnotation(ACCAnnotation.class, "kernels");
			//				if( compAnnot == null ){
			//					compAnnot = redimStmt.getAnnotation(ACCAnnotation.class, "parallel");
			//				}
			//				ACCAnnotation pcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(redimStmt, ACCAnnotation.class, ACCAnnotation.computeRegions, false, funcCallList, null);
			//				if( pcompAnnot != null ) {
			//					// [FIXME] is it correct?
			//					Tools.exit("[ERROR in DataLayoutTransformation()] transform pragma can not exist inside of " +
			//							"any compute regions (kerenls/parallel regions):\n" +
			//							"Enclosing procedure: " + redimParentProc.getSymbolName() + "\nOpenACC annotation: " + tAnnot + "\n");
			//				}
			//
			//
			//				List<FunctionCall> funcCallList2 = IRTools.getFunctionCalls(redimBody);
			//
			//				for(SubArrayWithConf sArrayConf : sArrayConfSet ) {
			//					//step2
			//
			//					SubArray sArray = sArrayConf.getSubArray();
			//					ConfList confL = sArrayConf.getConfList(0);
			//					List<Expression> confExList = confL.getConfigList();
			//					int dimension = confExList.size();
			//
			//					
			//					
			//					List<Expression> startIndecies = sArray.getStartIndices();
			//					List<Expression> exList = sArray.getLengths();
			//					Expression sArrayNameEx = sArray.getArrayName();
			//					String sArrayName = sArray.getArrayName().toString();
			//					String transposedArrayName = "transposed__" + sArrayName;
			//					System.out.println("size = " + dimension + " name = " + sArrayName.toString());
			//
			//					Set<Symbol> symSet = redimParentBody.getSymbols();
			//					Symbol transpose_sym = AnalysisTools.findsSymbol(symSet, sArrayName);
			//					System.out.println(transpose_sym.getSymbolName());
			//					System.out.println(transpose_sym.getArraySpecifiers().get(0).toString());
			//					System.out.println(transpose_sym.getTypeSpecifiers().get(0).toString());
			//
			//
			//					List<ACCAnnotation> annotationListACC =
			//							AnalysisTools.collectPragmas(redimStmt, ACCAnnotation.class);
			//
			//					/* get host variable */
			//					Expression hostVar = new Identifier(transpose_sym);
			//					//					Set<Symbol> outSymbols = new HashSet<Symbol>();
			//
			//					/* New variable declaration and allocation */
			//					Expression newVar = null;
			//					// Create a new transpose-host variable.
			//					// The type of the transpose-host symbol should be a pointer type 
			//					VariableDeclarator transpose_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
			//							new NameID(transposedArrayName));
			//					VariableDeclaration transpose_decl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
			//							transpose_declarator);
			//					newVar = new Identifier(transpose_declarator);
			//					Symbol transpose_dec = transpose_declarator;
			//					//transpose_sym = transpose_declarator;
			//
			//					//step5 replace all variables
			//					Statement enclosingCompRegion = null;
			//					for(ACCAnnotation annot : annotationListACC)
			//					{
			//						Annotatable att = annot.getAnnotatable();
			//						if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
			//								|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
			//							enclosingCompRegion = (Statement)att;
			//							IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
			//							//IRTools.replaceAll(cStmt, hostVar, newVar);
			//						}
			//
			//						for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
			//							//							Set <Set<SubArray>> transposeSArraySet = annot.get(dataClause);
			//							Set<SubArray> transposeSArraySet = annot.get(dataClause);
			//							if(transposeSArraySet != null){
			//								System.out.println(transposeSArraySet.toString());
			//								for(SubArray subArray : transposeSArraySet){
			//									System.out.println(subArray.toString());
			//									System.out.println(subArray.getArrayName().toString());
			//									System.out.println(sArrayName);
			//									if(subArray.getArrayName().equals(sArrayNameEx)){
			//										SubArray newVarSubArray = subArray.clone();
			//										newVarSubArray.setArrayName(newVar.clone());
			//										int[] newIndexIndex = new int[dimension];
			//										for(int i = 0;i < dimension;i++){
			//											newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//										}
			//										List < List<Expression>> ExList = new ArrayList< List<Expression>>();
			//										for(int i = 0;i < dimension;i++){
			//											ExList.add(newVarSubArray.getRange(newIndexIndex[i]));
			//										}
			//										for(int i = 0;i < dimension;i++){
			//											newVarSubArray.setRange(i, ExList.get(i));
			//										}
			//										transposeSArraySet.add(newVarSubArray);
			//									}
			//									//}
			//									//}
			//								}
			//							}
			//
			//						}
			//						for(String accinter : ACCAnnotation.internalDataClauses){  
			//							//Set<Symbol> transposeSArrays = annot.get(accinter);
			//							Set<Object> transposeSArrays = annot.get(accinter);
			//							//Set<SubArray> transposeSArrays = annot.get(accinter);
			//							if(transposeSArrays != null){
			//								System.out.println(accinter + " " + transposeSArrays.toString() + " " + transposeSArrays.size());
			//								//								for(SubArray subArray : transposeSArrays){ //[FIXME]
			//								for(Object subArray : transposeSArrays){
			//									if(subArray instanceof Symbol){
			//										if(((Symbol)subArray).equals(transpose_sym)){
			//											transposeSArrays.add(transpose_dec);
			//											//transposeSArrays.remove(subArray);
			//											break;
			//										}
			//									}
			//								}
			//							}
			//						}
			//					}
			//					if(enclosingCompRegion == null) {
			//						continue;
			//					}
			//
			//
			//					List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
			//					for(Expression e : newVarExpressions){
			//						if(e.getParent().getClass().equals(ArrayAccess.class)){
			//							ArrayAccess aa = (ArrayAccess)e.getParent();
			//							//							System.out.println("array access name" + aa.getArrayName().toString());
			//							List<Expression> oldIndices = aa.getIndices();
			//							int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
			//							Expression newIndex = oldIndices.get(newIndexIndex);
			//							for(int i = 1; i < dimension;i++){
			//								newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//								newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
			//										BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
			//							}
			//							List<Expression> newIndices = new ArrayList<Expression>();
			//							newIndices.add(newIndex);
			//							aa.setIndices(newIndices);
			//						}
			//					}
			//
			//					//					List<ArrayAccess> array_accesses = AnalysisTools.getArrayAccesses(newVar);
			//					//					for(ArrayAccess aa : array_accesses){
			//					//					System.out.println("array access name" + aa.getArrayName().toString());
			//					//					}
			//
			//					redimBody.addDeclaration(transpose_decl);
			//					//Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
			//					Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
			//					for(Expression ex : exList){
			//						size = new BinaryExpression(size, BinaryOperator.MULTIPLY, ex);
			//					}
			//					FunctionCall malloc_call = new FunctionCall(new NameID("malloc"));
			//					malloc_call.addArgument(size);
			//					List<Specifier> specs = new ArrayList<Specifier>(4);
			//					//specs.addAll(transpose_dec.getTypeSpecifiers());
			//					specs.addAll(transpose_dec.getTypeSpecifiers());
			//					Statement malloc_stmt = new ExpressionStatement(new AssignmentExpression(newVar.clone(),
			//							AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
			//					redimBody.addStatementBefore(firstNonDeclStmt, malloc_stmt);
			//
			//
			//					//size and rule information
			//					Expression sizeInfo = null;
			//					Expression ruleInfo = null;
			//					NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
			//					NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
			//					List<Expression> arryDimList = new ArrayList<Expression>();
			//					arryDimList.add(new IntegerLiteral(dimension));
			//					ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
			//					VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
			//					VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
			//					VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
			//					VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
			//					sizeInfo = new Identifier(sizeInfo_declarator);
			//					ruleInfo = new Identifier(ruleInfo_declarator);
			//					redimBody.addDeclaration(sizeInfo_decl);
			//					redimBody.addDeclaration(ruleInfo_decl);
			//					for(int i = 0;i < dimension;i++){
			//						Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
			//						Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
			//						redimBody.addStatementBefore(firstNonDeclStmt,sizeInfo_Statement);
			//						redimBody.addStatementBefore(firstNonDeclStmt,ruleInfo_Statement);
			//					}
			//
			//					//step3 make transpose_kernel function for CPU
			//					NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
			//					//CompoundStatement funcBody = new CompoundStatement();
			//					//Function transpose_kernel = new Function();
			//					FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
			//					Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
			//					redimBody.addStatementBefore(firstNonDeclStmt, transpose_func_stmt);
			//					transpose_funcCall.addArgument(newVar.clone());
			//					//transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					transpose_funcCall.addArgument(sizeInfo.clone());
			//					transpose_funcCall.addArgument(ruleInfo.clone());
			//					transpose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//					//step4 make re_transpose_kernel function for CPU
			//					NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
			//					FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
			//					Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
			//					redimBody.addStatement(retranspose_func_stmt);
			//					//retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					retranspose_funcCall.addArgument(newVar.clone());
			//					retranspose_funcCall.addArgument(sizeInfo.clone());
			//					retranspose_funcCall.addArgument(ruleInfo.clone());
			//					retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//
			//					//step5 make transpose_kernel function for GPU
			//					NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
			//					FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
			//					Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
			//					gpu_transpose_funcCall.addArgument(newVar.clone());
			//					//gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_transpose_funcCall.addArgument(sizeInfo.clone());
			//					gpu_transpose_funcCall.addArgument(ruleInfo.clone());
			//					gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//					//step4 make re_transpose_kernel function for GPU
			//					NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
			//					FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
			//					Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
			//					//gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
			//					gpu_retranspose_funcCall.addArgument(newVar.clone());
			//					gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
			//					gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
			//					gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));
			//
			//
			//					//step free
			//					FunctionCall free_call = new FunctionCall(new NameID("free"));
			//					free_call.addArgument(newVar.clone());
			//					Statement free_stmt = new ExpressionStatement(free_call);
			//					redimBody.addStatement(free_stmt);
			//
			//
			//					//step   function calls
			//					for(FunctionCall func : funcCallList2){
			//						System.out.println(func.toString() + ", " + func.getStatement().toString());
			//
			//						Traversable t = func.getParent();
			//						while ( !(t instanceof CompoundStatement)){
			//							t = t.getParent();
			//						}
			//						CompoundStatement funcParent = (CompoundStatement)t;
			//						System.out.println(funcParent.toString());
			//						ACCAnnotation useDeviceDirective = new ACCAnnotation("host_data", null);
			//						List<Expression> useDeviceList = new ArrayList<Expression>();
			//						useDeviceList.add(newVar);
			//						useDeviceList.add(hostVar);
			//						useDeviceDirective.put("use_device", useDeviceList);
			//						Statement useDeviceStmt = new AnnotationStatement(useDeviceDirective);
			//						CompoundStatement useDeviceRegion = new CompoundStatement(); 
			//						CompoundStatement re_useDeviceRegion = new CompoundStatement(); 
			//						useDeviceRegion.addStatement(gpu_retranspose_func_stmt.clone());
			//						re_useDeviceRegion.addStatement(gpu_transpose_func_stmt.clone());
			//						funcParent.addStatementBefore(func.getStatement(), useDeviceStmt.clone());
			//						funcParent.addStatementBefore(func.getStatement(), useDeviceRegion);
			//						funcParent.addStatementAfter(func.getStatement(), re_useDeviceRegion);
			//						funcParent.addStatementAfter(func.getStatement(), useDeviceStmt.clone());
			////						funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
			//					}
			//
			//
			//					//step update directive 
			//					for(ACCAnnotation annot : annotationListACC)
			//					{
			//						Annotatable att = annot.getAnnotatable();
			//						if(att.containsAnnotation(ACCAnnotation.class, "update")){
			//							if(att.containsAnnotation(ACCAnnotation.class, "host")){
			//								boolean flag = false;
			//								Set<Object> hostList = annot.get("host");
			//								for(Object subArray : hostList){
			//									if(subArray instanceof Symbol){
			//										if(((Symbol)subArray).equals(transpose_sym)){
			//											hostList.add(transpose_dec);
			//											//System.out.println("transsym " + transpose_sym.getSymbolName());
			//											hostList.remove(subArray);
			//											flag = true;
			//											break;
			//										}
			//									}
			//									if(subArray instanceof SubArray){
			//										SubArray sa = (SubArray)subArray;
			//										if(sa.getArrayName().equals(sArrayNameEx)){
			//											sa.setArrayName(newVar.clone());
			//											int[] newIndexIndex = new int[dimension];
			//											for(int i = 0;i < dimension;i++){
			//												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//											}
			//											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
			//											for(int i = 0;i < dimension;i++){
			//												ExList.add(sa.getRange(newIndexIndex[i]));
			//											}
			//											for(int i = 0;i < dimension;i++){
			//												sa.setRange(i, ExList.get(i));
			//											}
			//											flag = true;
			//											break;
			//										}
			//									}
			//								}
			//								if(flag){
			//									Traversable t = att;
			//									while ( !(t instanceof CompoundStatement)){
			//										t = t.getParent();
			//									}
			//									CompoundStatement attParent = (CompoundStatement)t;
			//									attParent.addStatementAfter((Statement)att, retranspose_func_stmt.clone());
			//								}
			//
			//							}
			//							if(att.containsAnnotation(ACCAnnotation.class, "device")){
			//								boolean flag = false;
			//								Set<Object> deviceList = annot.get("device");
			//								for(Object subArray : deviceList){
			//									if(subArray instanceof Symbol){
			//										if(((Symbol)subArray).equals(transpose_sym)){
			//											deviceList.add(transpose_dec);
			//											//System.out.println("transsym " + transpose_sym.getSymbolName());
			//											deviceList.remove(subArray);
			//											flag = true;
			//											break;
			//										}
			//									}
			//									if(subArray instanceof SubArray){
			//										SubArray sa = (SubArray)subArray;
			//										if(sa.getArrayName().equals(sArrayNameEx)){
			//											sa.setArrayName(newVar.clone());
			//											int[] newIndexIndex = new int[dimension];
			//											for(int i = 0;i < dimension;i++){
			//												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
			//											}
			//											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
			//											for(int i = 0;i < dimension;i++){
			//												ExList.add(sa.getRange(newIndexIndex[i]));
			//											}
			//											for(int i = 0;i < dimension;i++){
			//												sa.setRange(i, ExList.get(i));
			//											}
			//											flag = true;
			//											break;
			//										}
			//									}
			//								}
			//								if(flag){
			//									Traversable t = att;
			//									while ( !(t instanceof CompoundStatement)){
			//										t = t.getParent();
			//									}
			//									CompoundStatement attParent = (CompoundStatement)t;
			//									attParent.addStatementBefore((Statement)att, transpose_func_stmt.clone());
			//								}
			//
			//							}
			//						}
			//					}
			//				}
			//			}
		}
		SymbolTools.linkSymbol(program);
	}


	public void start_20141030() {
		//for each transform, 
		//step0: ([TODO]): if ifcond exists and its argument is 0, skip the current transform region. 
		//step1: If the transform region is inside of compute regions, error in current version. 
		//step2: Create new transposed array
		//step2.1: Get the original data type by directive parsing
		//step3: transpose function call
		//step4: re-transpose function call
		//step5: replace the variables
		//step5.5: recalculate the index of the new array
		//step6: ([TODO]) deal with function calls
		//step7: ([TODO]) deal with copy clause and update directive in the transform region.

		List<ARCAnnotation> transformAnnots = IRTools.collectPragmas(program, ARCAnnotation.class, "transform");
		if( transformAnnots != null ) {
			List<ARCAnnotation> transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redimAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expandAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redim_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expand_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> transpose_expandAnnots = new LinkedList<ARCAnnotation>();

			for( ARCAnnotation arcAnnot : transformAnnots ) {
				if( arcAnnot.containsKey("transpose") ) {
					transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim") ) {
					redimAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand") ) {
					expandAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim_transpose") ) {
					redim_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand_transpose") ) {
					expand_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("transpose_expand") ) {
					transpose_expandAnnots.add(arcAnnot);
				}
			}


			List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);

			//////////////////////////////////////////
			// Performs "transpose" transformation. //
			//////////////////////////////////////////
			for( ARCAnnotation tAnnot : transposeAnnots ) {

				Annotatable at = tAnnot.getAnnotatable();
				Set<SubArrayWithConf> sArrayConfSet = tAnnot.get("transpose");

				Statement transposeRegion = (Statement)tAnnot.getAnnotatable();
				//				CompoundStatement cStmt = (CompoundStatement)transposeRegion.getParent();
				//				CompoundStatement cStmt = (CompoundStatement)transposeRegion.getChildren().get(0);
				CompoundStatement cStmt = null;
				Statement firstNonDeclStmt = null;
				if (transposeRegion instanceof CompoundStatement ){
					cStmt = (CompoundStatement)transposeRegion;
					firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(transposeRegion);
				} else {
					// [FIXME] is it correct?
					cStmt = new CompoundStatement();
					firstNonDeclStmt = transposeRegion;
				}



				Procedure cProc = IRTools.getParentProcedure(transposeRegion);
				CompoundStatement cBody = cProc.getBody();

				System.out.println("test");

				//step1 
				ACCAnnotation compAnnot = null;
				compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "kernels");
				if( compAnnot == null ){
					compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "parallel");
				}
				ACCAnnotation pcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(transposeRegion, ACCAnnotation.class, ACCAnnotation.computeRegions, false, funcCallList, null);
				if( pcompAnnot != null ) {
					Tools.exit("[ERROR in DataLayoutTransformation()] transform pragma can not exist inside of " +
							"any compute regions (kerenls/parallel regions):\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + tAnnot + "\n");
				}


				List<FunctionCall> funcCallList2 = IRTools.getFunctionCalls(cStmt);

				for(SubArrayWithConf sArrayConf : sArrayConfSet ) {
					//step2

					SubArray sArray = sArrayConf.getSubArray();
					//In transpose clause, each SubArrayWithConf has only on configuration list ([2,1,3]).
					//CF: redim_transpose, each SubArrayWithConf will have two configuration lists 
					//(one for redim (e.g, [X,Y,Z]), and the other for transpose(e.g., [2,1,3])).
					ConfList confL = sArrayConf.getConfList(0);
					//...
					List<Expression> startIndecies = sArray.getStartIndices();
					List<Expression> exList = sArray.getLengths();
					int dimension = sArray.getArrayDimension();
					Expression sArrayNameEx = sArray.getArrayName();
					String sArrayName = sArray.getArrayName().toString();
					String transposedArrayName = "transposed__" + sArrayName;
					System.out.println("size = " + dimension + " name = " + sArrayName.toString());

					Set<Symbol> symSet = cBody.getSymbols();
					Symbol transpose_sym = AnalysisTools.findsSymbol(symSet, sArrayName);
					System.out.println(transpose_sym.getSymbolName());
					System.out.println(transpose_sym.getArraySpecifiers().get(0).toString());
					System.out.println(transpose_sym.getTypeSpecifiers().get(0).toString());


					List<ACCAnnotation> annotationListACC =
							AnalysisTools.collectPragmas(transposeRegion, ACCAnnotation.class);

					/* get host variable */
					Expression hostVar = new Identifier(transpose_sym);
					//					Set<Symbol> outSymbols = new HashSet<Symbol>();

					/* New variable declaration and allocation */
					Expression newVar = null;
					// Create a new transpose-host variable.
					// The type of the transpose-host symbol should be a pointer type 
					VariableDeclarator transpose_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(transposedArrayName));
					VariableDeclaration transpose_decl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							transpose_declarator);
					newVar = new Identifier(transpose_declarator);
					Symbol transpose_dec = transpose_declarator;
					//transpose_sym = transpose_declarator;

					//step5 replace all variables
					Statement enclosingCompRegion = null;
					for(ACCAnnotation annot : annotationListACC)
					{
						Annotatable att = annot.getAnnotatable();
						if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
								|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
							enclosingCompRegion = (Statement)att;
							IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
							//IRTools.replaceAll(cStmt, hostVar, newVar);
						}

						for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
							//							Set <Set<SubArray>> transposeSArraySet = annot.get(dataClause);
							Set<SubArray> transposeSArraySet = annot.get(dataClause);
							if(transposeSArraySet != null){
								System.out.println(transposeSArraySet.toString());
								//for(Set<SubArray> subArrays : transposeSArraySet){
								for(SubArray subArray : transposeSArraySet){
									//if(subArrays != null){
									//for(SubArray subArray : subArrays){
									System.out.println(subArray.toString());
									System.out.println(subArray.getArrayName().toString());
									System.out.println(sArrayName);
									if(subArray.getArrayName().equals(sArrayNameEx)){
										subArray.setArrayName(newVar.clone());
										int[] newIndexIndex = new int[dimension];
										for(int i = 0;i < dimension;i++){
											newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
										}
										List < List<Expression>> ExList = new ArrayList< List<Expression>>();
										for(int i = 0;i < dimension;i++){
											ExList.add(subArray.getRange(newIndexIndex[i]));
										}
										for(int i = 0;i < dimension;i++){
											subArray.setRange(i, ExList.get(i));
										}
									}
									//}
									//}
								}
							}

						}
						for(String accinter : ACCAnnotation.internalDataClauses){  
							//Set<Symbol> transposeSArrays = annot.get(accinter);
							Set<Object> transposeSArrays = annot.get(accinter);
							//Set<SubArray> transposeSArrays = annot.get(accinter);
							if(transposeSArrays != null){
								System.out.println(accinter + " " + transposeSArrays.toString() + " " + transposeSArrays.size());
								//								for(SubArray subArray : transposeSArrays){ //[FIXME]
								for(Object subArray : transposeSArrays){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											//transposeSArrays.add(transpose_dec);
											transposeSArrays.add(transpose_dec);
											//System.out.println("transsym " + transpose_sym.getSymbolName());
											transposeSArrays.remove(subArray);
											break;
										}
									}
								}
							}
						}
					}
					if(enclosingCompRegion == null) {
						continue;
					}


					List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
					for(Expression e : newVarExpressions){
						if(e.getParent().getClass().equals(ArrayAccess.class)){
							ArrayAccess aa = (ArrayAccess)e.getParent();
							//							System.out.println("array access name" + aa.getArrayName().toString());
							List<Expression> oldIndices = aa.getIndices();
							int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
							Expression newIndex = oldIndices.get(newIndexIndex);
							for(int i = 1; i < dimension;i++){
								newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
								newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
										BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
							}
							List<Expression> newIndices = new ArrayList<Expression>();
							newIndices.add(newIndex);
							aa.setIndices(newIndices);
						}
					}

					//					List<ArrayAccess> array_accesses = AnalysisTools.getArrayAccesses(newVar);
					//					for(ArrayAccess aa : array_accesses){
					//					System.out.println("array access name" + aa.getArrayName().toString());
					//					}

					cStmt.addDeclaration(transpose_decl);
					//Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
					Expression size = new SizeofExpression(transpose_dec.getTypeSpecifiers()); //[FIXME]
					for(Expression ex : exList){
						size = new BinaryExpression(size, BinaryOperator.MULTIPLY, ex);
					}
					FunctionCall malloc_call = new FunctionCall(new NameID("malloc"));
					malloc_call.addArgument(size);
					List<Specifier> specs = new ArrayList<Specifier>(4);
					//specs.addAll(transpose_dec.getTypeSpecifiers());
					specs.addAll(transpose_dec.getTypeSpecifiers());
					Statement malloc_stmt = new ExpressionStatement(new AssignmentExpression(newVar.clone(),
							AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
					cStmt.addStatementBefore(firstNonDeclStmt, malloc_stmt);



					//[TODO] To know how can I get access to the variables which have same name of the symbols
					//IRTools.replaceAll((Traversable)cBody, (Expression)transpose_sym, newVar.clone());

					//get kernels/parallel region



					//size and rule information
					Expression sizeInfo = null;
					Expression ruleInfo = null;
					NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
					NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
					List<Expression> arryDimList = new ArrayList<Expression>();
					arryDimList.add(new IntegerLiteral(dimension));
					ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
					VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
					VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
					VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
					VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
					sizeInfo = new Identifier(sizeInfo_declarator);
					ruleInfo = new Identifier(ruleInfo_declarator);
					cStmt.addDeclaration(sizeInfo_decl);
					cStmt.addDeclaration(ruleInfo_decl);
					for(int i = 0;i < dimension;i++){
						Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
						Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
						cStmt.addStatementBefore(firstNonDeclStmt,sizeInfo_Statement);
						cStmt.addStatementBefore(firstNonDeclStmt,ruleInfo_Statement);
					}


					//					Identifier loopIndex = TransformTools.getNewTempIndex(cStmt);
					//					CompoundStatement loopBody = new CompoundStatement();
					//					Statement loopInit = new ExpressionStatement(new BinaryExpression(loopIndex.clone(), AssignmentOperator.NORMAL,new IntegerLiteral(0) ));  
					//					Expression loopCond = new BinaryExpression(loopIndex.clone(), BinaryOperator.COMPARE_LT, new IntegerLiteral(dimension));
					//					Expression loopStep = new UnaryExpression(UnaryOperator.POST_INCREMENT,loopIndex.clone());
					//					ForLoop sizeInfo_ForStatement = new ForLoop(loopInit, loopCond, loopStep, loopBody);
					//					cStmt.addStatementBefore(firstNonDeclStmt, sizeInfo_ForStatement);



					//step3 make transpose_kernel function for CPU
					NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
					//CompoundStatement funcBody = new CompoundStatement();
					//Function transpose_kernel = new Function();
					FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
					Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
					cStmt.addStatementBefore(firstNonDeclStmt, transpose_func_stmt);
					transpose_funcCall.addArgument(newVar.clone());
					//transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					transpose_funcCall.addArgument(sizeInfo.clone());
					transpose_funcCall.addArgument(ruleInfo.clone());
					transpose_funcCall.addArgument(new IntegerLiteral(dimension));

					//step4 make re_transpose_kernel function for CPU
					NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
					FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
					Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
					cStmt.addStatement(retranspose_func_stmt);
					//retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					retranspose_funcCall.addArgument(newVar.clone());
					retranspose_funcCall.addArgument(sizeInfo.clone());
					retranspose_funcCall.addArgument(ruleInfo.clone());
					retranspose_funcCall.addArgument(new IntegerLiteral(dimension));


					//step5 make transpose_kernel function for GPU
					NameID gpu_transposeFuncName = new NameID("HI_device_array_transposition_kernel__");
					FunctionCall gpu_transpose_funcCall = new FunctionCall(gpu_transposeFuncName);
					Statement gpu_transpose_func_stmt = new ExpressionStatement(gpu_transpose_funcCall);
					gpu_transpose_funcCall.addArgument(newVar.clone());
					//gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					gpu_transpose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					gpu_transpose_funcCall.addArgument(sizeInfo.clone());
					gpu_transpose_funcCall.addArgument(ruleInfo.clone());
					gpu_transpose_funcCall.addArgument(new IntegerLiteral(dimension));

					//step4 make re_transpose_kernel function for GPU
					NameID gpu_retransposeFuncName = new NameID("HI_device_re_array_transposition_kernel__");
					FunctionCall gpu_retranspose_funcCall = new FunctionCall(gpu_retransposeFuncName);
					Statement gpu_retranspose_func_stmt = new ExpressionStatement(gpu_retranspose_funcCall);
					//gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					gpu_retranspose_funcCall.addArgument(new Typecast(transpose_dec.getTypeSpecifiers(),hostVar.clone()));
					gpu_retranspose_funcCall.addArgument(newVar.clone());
					gpu_retranspose_funcCall.addArgument(sizeInfo.clone());
					gpu_retranspose_funcCall.addArgument(ruleInfo.clone());
					gpu_retranspose_funcCall.addArgument(new IntegerLiteral(dimension));


					//step free
					FunctionCall free_call = new FunctionCall(new NameID("free"));
					free_call.addArgument(newVar.clone());
					Statement free_stmt = new ExpressionStatement(free_call);
					cStmt.addStatement(free_stmt);


					//step   function calls
					for(FunctionCall func : funcCallList2){
						System.out.println(func.toString() + ", " + func.getStatement().toString());

						Traversable t = func.getParent();
						while ( !(t instanceof CompoundStatement)){
							t = t.getParent();
						}
						CompoundStatement funcParent = (CompoundStatement)t;
						System.out.println(funcParent.toString());
						ACCAnnotation useDeviceDirective = new ACCAnnotation("host_data", null);
						List<Expression> useDeviceList = new ArrayList<Expression>();
						useDeviceList.add(newVar);
						useDeviceList.add(hostVar);
						useDeviceDirective.put("use_device", useDeviceList);
						Statement useDeviceStmt = new AnnotationStatement(useDeviceDirective);
						CompoundStatement useDeviceRegion = new CompoundStatement(); 
						CompoundStatement re_useDeviceRegion = new CompoundStatement(); 
						useDeviceRegion.addStatement(gpu_retranspose_func_stmt.clone());
						re_useDeviceRegion.addStatement(gpu_transpose_func_stmt.clone());
						funcParent.addStatementBefore(func.getStatement(), useDeviceStmt.clone());
						funcParent.addStatementBefore(func.getStatement(), useDeviceRegion);
						funcParent.addStatementAfter(func.getStatement(), re_useDeviceRegion);
						funcParent.addStatementAfter(func.getStatement(), useDeviceStmt.clone());
						//						funcParent.addStatementAfter(func.getStatement(), gpu_transpose_func_stmt.clone());
					}


					//step update directive 
					for(ACCAnnotation annot : annotationListACC)
					{
						Annotatable att = annot.getAnnotatable();
						if(att.containsAnnotation(ACCAnnotation.class, "update")){
							if(att.containsAnnotation(ACCAnnotation.class, "host")){
								boolean flag = false;
								Set<Object> hostList = annot.get("host");
								for(Object subArray : hostList){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											hostList.add(transpose_dec);
											//System.out.println("transsym " + transpose_sym.getSymbolName());
											hostList.remove(subArray);
											flag = true;
											break;
										}
									}
									if(subArray instanceof SubArray){
										SubArray sa = (SubArray)subArray;
										if(sa.getArrayName().equals(sArrayNameEx)){
											sa.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(sa.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												sa.setRange(i, ExList.get(i));
											}
											flag = true;
											break;
										}
									}
								}
								if(flag){
									Traversable t = att;
									while ( !(t instanceof CompoundStatement)){
										t = t.getParent();
									}
									CompoundStatement attParent = (CompoundStatement)t;
									attParent.addStatementAfter((Statement)att, retranspose_func_stmt.clone());
								}

							}
							if(att.containsAnnotation(ACCAnnotation.class, "device")){
								boolean flag = false;
								Set<Object> deviceList = annot.get("device");
								for(Object subArray : deviceList){
									if(subArray instanceof Symbol){
										if(((Symbol)subArray).equals(transpose_sym)){
											deviceList.add(transpose_dec);
											//System.out.println("transsym " + transpose_sym.getSymbolName());
											deviceList.remove(subArray);
											flag = true;
											break;
										}
									}
									if(subArray instanceof SubArray){
										SubArray sa = (SubArray)subArray;
										if(sa.getArrayName().equals(sArrayNameEx)){
											sa.setArrayName(newVar.clone());
											int[] newIndexIndex = new int[dimension];
											for(int i = 0;i < dimension;i++){
												newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
											}
											List < List<Expression>> ExList = new ArrayList< List<Expression>>();
											for(int i = 0;i < dimension;i++){
												ExList.add(sa.getRange(newIndexIndex[i]));
											}
											for(int i = 0;i < dimension;i++){
												sa.setRange(i, ExList.get(i));
											}
											flag = true;
											break;
										}
									}
								}
								if(flag){
									Traversable t = att;
									while ( !(t instanceof CompoundStatement)){
										t = t.getParent();
									}
									CompoundStatement attParent = (CompoundStatement)t;
									attParent.addStatementBefore((Statement)att, transpose_func_stmt.clone());
								}

							}
						}
					}
				}
			}
		}
		SymbolTools.linkSymbol(program);
	}


	public void start_org() {
		//for each transform, 
		//step0: ([TODO]): if ifcond exists and its argument is 0, skip the current transform region. 
		//step1: If the transform region is inside of compute regions, error in current version. 
		//step2: Create new transposed array
		//step2.1: Get the original data type by directive parsing
		//step3: transpose function call
		//step4: re-transpose function call
		//step5: replace the variables
		//step5.5: recalculate the index of the new array
		//step6: ([TODO]) deal with function calls
		//step7: ([TODO]) deal with copy clause and update directive in the transform region.

		List<ARCAnnotation> transformAnnots = IRTools.collectPragmas(program, ARCAnnotation.class, "transform");
		if( transformAnnots != null ) {
			List<ARCAnnotation> transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redimAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expandAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> redim_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> expand_transposeAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> transpose_expandAnnots = new LinkedList<ARCAnnotation>();

			for( ARCAnnotation arcAnnot : transformAnnots ) {
				if( arcAnnot.containsKey("transpose") ) {
					transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim") ) {
					redimAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand") ) {
					expandAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("redim_transpose") ) {
					redim_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("expand_transpose") ) {
					expand_transposeAnnots.add(arcAnnot);
				} else if( arcAnnot.containsKey("transpose_expand") ) {
					transpose_expandAnnots.add(arcAnnot);
				}
			}


			List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);

			//////////////////////////////////////////
			// Performs "transpose" transformation. //
			//////////////////////////////////////////
			for( ARCAnnotation tAnnot : transposeAnnots ) {

				Annotatable at = tAnnot.getAnnotatable();
				Set<SubArrayWithConf> sArrayConfSet = tAnnot.get("transpose");

				Statement transposeRegion = (Statement)tAnnot.getAnnotatable();
				//				CompoundStatement cStmt = (CompoundStatement)transposeRegion.getParent();
				//				CompoundStatement cStmt = (CompoundStatement)transposeRegion.getChildren().get(0);
				CompoundStatement cStmt = null;
				Statement firstNonDeclStmt = null;
				if (transposeRegion instanceof CompoundStatement ){
					cStmt = (CompoundStatement)transposeRegion;
					firstNonDeclStmt = IRTools.getFirstNonDeclarationStatement(transposeRegion);
				} else {
					// [FIXME] is it correct?
					cStmt = new CompoundStatement();
					firstNonDeclStmt = transposeRegion;
				}



				Procedure cProc = IRTools.getParentProcedure(transposeRegion);
				CompoundStatement cBody = cProc.getBody();

				System.out.println("test");

				//step1 
				ACCAnnotation compAnnot = null;
				compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "kernels");
				if( compAnnot == null ){
					compAnnot = transposeRegion.getAnnotation(ACCAnnotation.class, "parallel");
				}
				ACCAnnotation pcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(transposeRegion, ACCAnnotation.class, ACCAnnotation.computeRegions, false, funcCallList, null);
				if( pcompAnnot != null ) {
					Tools.exit("[ERROR in DataLayoutTransformation()] transform pragma can not exist inside of " +
							"any compute regions (kerenls/parallel regions):\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + tAnnot + "\n");
				}



				for(SubArrayWithConf sArrayConf : sArrayConfSet ) {
					//step2

					SubArray sArray = sArrayConf.getSubArray();
					//In transpose clause, each SubArrayWithConf has only on configuration list ([2,1,3]).
					//CF: redim_transpose, each SubArrayWithConf will have two configuration lists 
					//(one for redim (e.g, [X,Y,Z]), and the other for transpose(e.g., [2,1,3])).
					ConfList confL = sArrayConf.getConfList(0);
					//...
					List<Expression> startIndecies = sArray.getStartIndices();
					List<Expression> exList = sArray.getLengths();
					int dimension = sArray.getArrayDimension();
					Expression sArrayNameEx = sArray.getArrayName();
					String sArrayName = sArray.getArrayName().toString();
					String transposedArrayName = "transposed__" + sArrayName;
					System.out.println("size = " + dimension + " name = " + sArrayName.toString());

					Set<Symbol> symSet = cBody.getSymbols();
					Symbol transpose_sym = AnalysisTools.findsSymbol(symSet, sArrayName);
					System.out.println(transpose_sym.getSymbolName());
					System.out.println(transpose_sym.getArraySpecifiers().get(0).toString());
					System.out.println(transpose_sym.getTypeSpecifiers().get(0).toString());


					List<ACCAnnotation> annotationListACC =
							AnalysisTools.collectPragmas(transposeRegion, ACCAnnotation.class);
					List<ARCAnnotation> annotationListARC =
							AnalysisTools.collectPragmas(transposeRegion, ARCAnnotation.class);

					/* get host variable */
					Expression hostVar = new Identifier(transpose_sym);
					//					Set<Symbol> outSymbols = new HashSet<Symbol>();

					/* New variable declaration and allocation */
					Expression newVar = null;
					// Create a new transpose-host variable.
					// The type of the transpose-host symbol should be a pointer type 
					VariableDeclarator transpose_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
							new NameID(transposedArrayName));
					VariableDeclaration transpose_decl = new VariableDeclaration(transpose_sym.getTypeSpecifiers(), //[FIXME]
							transpose_declarator);
					newVar = new Identifier(transpose_declarator);
					transpose_sym = transpose_declarator;

					//step5 replace all variables
					Statement enclosingCompRegion = null;
					for(ACCAnnotation annot : annotationListACC)
					{
						Annotatable att = annot.getAnnotatable();
						if(att.containsAnnotation(ACCAnnotation.class, "kernels") 
								|| att.containsAnnotation(ACCAnnotation.class, "parallel")){
							enclosingCompRegion = (Statement)att;
							IRTools.replaceAll(enclosingCompRegion, hostVar, newVar);
							//IRTools.replaceAll(cStmt, hostVar, newVar);
						}

						for(String dataClause : ACCAnnotation.dataClauses){  //[FIXME] is it enough? How about ARCannotations?
							//							Set <Set<SubArray>> transposeSArraySet = annot.get(dataClause);
							Set<SubArray> transposeSArraySet = annot.get(dataClause);
							if(transposeSArraySet != null){
								System.out.println(transposeSArraySet.toString());
								//for(Set<SubArray> subArrays : transposeSArraySet){
								for(SubArray subArray : transposeSArraySet){
									//if(subArrays != null){
									//for(SubArray subArray : subArrays){
									System.out.println(subArray.toString());
									System.out.println(subArray.getArrayName().toString());
									System.out.println(sArrayName);
									if(subArray.getArrayName().equals(sArrayNameEx)){
										subArray.setArrayName(newVar.clone());
										int[] newIndexIndex = new int[dimension];
										for(int i = 0;i < dimension;i++){
											newIndexIndex[i] = Integer.parseInt(confL.getConfig(i).toString()) -1;
										}
										List < List<Expression>> ExList = new ArrayList< List<Expression>>();
										for(int i = 0;i < dimension;i++){
											ExList.add(subArray.getRange(newIndexIndex[i]));
										}
										for(int i = 0;i < dimension;i++){
											subArray.setRange(i, ExList.get(i));
										}
									}
									//}
									//}
								}
							}

						}
						for(String accinter : ACCAnnotation.internalDataClauses){  
							Set<Symbol> transposeSArrays = annot.get(accinter);
							//Set<SubArray> transposeSArrays = annot.get(accinter);
							if(transposeSArrays != null){
								System.out.println(transposeSArrays.toString());
								//								for(SubArray subArray : transposeSArrays){ //[FIXME]
								for(Symbol subArray : transposeSArrays){ //[FIXME]
									//									System.out.println(subArray.toString());
									//									System.out.println("symbolName " + subArray.getSymbolName());
									if(subArray.getSymbolName().equals(sArrayName)){
										transposeSArrays.add(transpose_sym);
										//System.out.println("transsym " + transpose_sym.getSymbolName());
										transposeSArrays.remove(subArray);
									}
								}
							}
						}
					}
					if(enclosingCompRegion == null) {
						continue;
					}


					List<Expression> newVarExpressions = IRTools.findExpressions(enclosingCompRegion, newVar);
					for(Expression e : newVarExpressions){
						if(e.getParent().getClass().equals(ArrayAccess.class)){
							ArrayAccess aa = (ArrayAccess)e.getParent();
							//							System.out.println("array access name" + aa.getArrayName().toString());
							List<Expression> oldIndices = aa.getIndices();
							int newIndexIndex = Integer.parseInt(confL.getConfig(0).toString()) -1;
							Expression newIndex = oldIndices.get(newIndexIndex);
							for(int i = 1; i < dimension;i++){
								newIndexIndex = Integer.parseInt(confL.getConfig(i).toString()) -1;
								newIndex = new BinaryExpression (new BinaryExpression(newIndex.clone(), BinaryOperator.MULTIPLY, exList.get(newIndexIndex).clone()),
										BinaryOperator.ADD, oldIndices.get(newIndexIndex).clone());
							}
							List<Expression> newIndices = new ArrayList<Expression>();
							newIndices.add(newIndex);
							aa.setIndices(newIndices);
						}
					}

					//					List<ArrayAccess> array_accesses = AnalysisTools.getArrayAccesses(newVar);
					//					for(ArrayAccess aa : array_accesses){
					//					System.out.println("array access name" + aa.getArrayName().toString());
					//					}

					cStmt.addDeclaration(transpose_decl);
					Expression size = new SizeofExpression(transpose_sym.getTypeSpecifiers()); //[FIXME]
					for(Expression ex : exList){
						size = new BinaryExpression(size, BinaryOperator.MULTIPLY, ex);
					}
					FunctionCall malloc_call = new FunctionCall(new NameID("malloc"));
					malloc_call.addArgument(size);
					List<Specifier> specs = new ArrayList<Specifier>(4);
					specs.addAll(transpose_sym.getTypeSpecifiers());
					Statement malloc_stmt = new ExpressionStatement(new AssignmentExpression(newVar.clone(),
							AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
					cStmt.addStatementBefore(firstNonDeclStmt, malloc_stmt);



					//[TODO] To know how can I get access to the variables which have same name of the symbols
					//IRTools.replaceAll((Traversable)cBody, (Expression)transpose_sym, newVar.clone());

					//get kernels/parallel region



					//size and rule information
					Expression sizeInfo = null;
					Expression ruleInfo = null;
					NameID sizeInfo_name = new NameID("transpose_size_info_" + sArrayName);
					NameID ruleInfo_name = new NameID("transpose_rule_info_" + sArrayName);
					List<Expression> arryDimList = new ArrayList<Expression>();
					arryDimList.add(new IntegerLiteral(dimension));
					ArraySpecifier arraySpecs = new ArraySpecifier(arryDimList);
					VariableDeclarator sizeInfo_declarator = new VariableDeclarator(sizeInfo_name, arraySpecs);
					VariableDeclaration sizeInfo_decl = new VariableDeclaration(ArraySpecifier.INT, sizeInfo_declarator);
					VariableDeclarator ruleInfo_declarator = new VariableDeclarator(ruleInfo_name, arraySpecs);
					VariableDeclaration ruleInfo_decl = new VariableDeclaration(ArraySpecifier.INT, ruleInfo_declarator);
					sizeInfo = new Identifier(sizeInfo_declarator);
					ruleInfo = new Identifier(ruleInfo_declarator);
					cStmt.addDeclaration(sizeInfo_decl);
					cStmt.addDeclaration(ruleInfo_decl);
					for(int i = 0;i < dimension;i++){
						Statement sizeInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(sizeInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, exList.get(i).clone()));
						Statement ruleInfo_Statement = new ExpressionStatement(new BinaryExpression(new ArrayAccess(ruleInfo.clone(),new IntegerLiteral(i)), AssignmentOperator.NORMAL, confL.getConfig(i).clone()));
						cStmt.addStatementBefore(firstNonDeclStmt,sizeInfo_Statement);
						cStmt.addStatementBefore(firstNonDeclStmt,ruleInfo_Statement);
					}


					//					Identifier loopIndex = TransformTools.getNewTempIndex(cStmt);
					//					CompoundStatement loopBody = new CompoundStatement();
					//					Statement loopInit = new ExpressionStatement(new BinaryExpression(loopIndex.clone(), AssignmentOperator.NORMAL,new IntegerLiteral(0) ));  
					//					Expression loopCond = new BinaryExpression(loopIndex.clone(), BinaryOperator.COMPARE_LT, new IntegerLiteral(dimension));
					//					Expression loopStep = new UnaryExpression(UnaryOperator.POST_INCREMENT,loopIndex.clone());
					//					ForLoop sizeInfo_ForStatement = new ForLoop(loopInit, loopCond, loopStep, loopBody);
					//					cStmt.addStatementBefore(firstNonDeclStmt, sizeInfo_ForStatement);



					//step3 make transpose_kernel function

					NameID transposeFuncName = new NameID("HI_array_transposition_kernel__");
					//CompoundStatement funcBody = new CompoundStatement();
					//Function transpose_kernel = new Function();

					FunctionCall transpose_funcCall = new FunctionCall(transposeFuncName);
					Statement transpose_func_stmt = new ExpressionStatement(transpose_funcCall);
					cStmt.addStatementBefore(firstNonDeclStmt, transpose_func_stmt);
					transpose_funcCall.addArgument(newVar.clone());
					transpose_funcCall.addArgument(new Typecast(transpose_sym.getTypeSpecifiers(),hostVar.clone()));
					transpose_funcCall.addArgument(sizeInfo.clone());
					transpose_funcCall.addArgument(ruleInfo.clone());
					transpose_funcCall.addArgument(new IntegerLiteral(dimension));




					//step4
					NameID retransposeFuncName = new NameID("HI_re_array_transposition_kernel__");
					FunctionCall retranspose_funcCall = new FunctionCall(retransposeFuncName);
					Statement retranspose_func_stmt = new ExpressionStatement(retranspose_funcCall);
					cStmt.addStatement(retranspose_func_stmt);
					retranspose_funcCall.addArgument(new Typecast(transpose_sym.getTypeSpecifiers(),hostVar.clone()));
					retranspose_funcCall.addArgument(newVar.clone());
					retranspose_funcCall.addArgument(sizeInfo.clone());
					retranspose_funcCall.addArgument(ruleInfo.clone());
					retranspose_funcCall.addArgument(new IntegerLiteral(dimension));

					//step free
					FunctionCall free_call = new FunctionCall(new NameID("free"));
					free_call.addArgument(newVar.clone());
					Statement free_stmt = new ExpressionStatement(free_call);
					cStmt.addStatement(free_stmt);




				}
				//...
			}

			//...
		}
		SymbolTools.linkSymbol(program);

	}



}
