/**
 * 
 */
package openacc.transforms;

import cetus.transforms.TransformPass;
import cetus.hir.*;
import java.util.*;

/**
 * This pass assumes that SingleDeclarator pass is executed before this.
 * 
 * @author f6l
 *
 */
public class DeclarationInitSeparator extends TransformPass {

	private static String pass_name = "[DeclarationInitSeparator]";
	private static Statement insertedInitStmt = null;
	
	public DeclarationInitSeparator(Program program) {
		super(program);
	}
	
	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return pass_name;
	}
	
    private void separateDeclInitialization(VariableDeclaration decl) {
    	//This pass assumes each declaration contains only one declarator, which can be enforced by SingleDeclarator pass.
    	if ( decl.getNumDeclarators() != 1 ) {
			return;
		}
    	Declarator declr = decl.getDeclarator(0);
    	if( !(declr instanceof Symbol) || (declr instanceof ProcedureDeclarator) ) {
    		return;
    	}
		Initializer lsm_init = null;
    	if( declr instanceof VariableDeclarator ) {
    		lsm_init = ((VariableDeclarator)declr).getInitializer();
    	} else if( declr instanceof NestedDeclarator ) {
    		lsm_init = ((NestedDeclarator)declr).getInitializer();
    	}
		if( lsm_init == null ) {
			return;
		}
		//Add array dimension information if missing.
		ArraySpecifier aspec = null;
		int dimsize = 0;
		List aspecs = declr.getArraySpecifiers();
		if( (aspecs != null) && (!aspecs.isEmpty()) ) {
			aspec = (ArraySpecifier)aspecs.get(0);
			dimsize = aspec.getNumDimensions();
		}
		int listSize = lsm_init.getChildren().size();
		if( dimsize == 1 ) { //1D array variable
			Expression tdim = aspec.getDimension(0);
			if( tdim == null ) {
				aspec.setDimension(0, new IntegerLiteral(listSize));
			}
		} else if( dimsize > 1 ) { //multi-dimensional array
			for(int i=0; i<dimsize; i++) {
				Expression tdim = aspec.getDimension(i);
				if( tdim == null ) {
					if( i == 0 ) {
						aspec.setDimension(i, new IntegerLiteral(listSize));
					} else {
						Object tobj = lsm_init.getChildren().get(0);
						int t = 1;
						while( (t<i) && (tobj != null) && (tobj instanceof Initializer) ) {
							tobj = ((Initializer)tobj).getChildren().get(0);
							t++;
						}
						if( (t==i) && (tobj != null) && (tobj instanceof Initializer) ) {
							aspec.setDimension(i, new IntegerLiteral(((Initializer)tobj).getChildren().size()));
						}
					}
				}
			}
		}
        PrintTools.printlnStatus(4, pass_name, "separating the initialization of declaration, ",decl);
        CompoundStatement cStmt = null;
        Statement declStmt = null;
        Traversable parent = decl.getParent();
        if (parent instanceof SymbolTable) {
        	//This declaration is for a global variable; the initialization of a global variable is not separated.
        	return;
        } else if (parent instanceof DeclarationStatement) {
            declStmt = (Statement)parent;
            cStmt = (CompoundStatement)parent.getParent();
        } else {
            return;
        }
        /* now parent is a symbol table and child is either decl or declstmt. */
		List lspecs = new ChainedList();
		lspecs.addAll(decl.getSpecifiers());
		List declrSpecs = new ChainedList();
    	declrSpecs.addAll(declr.getSpecifiers());
		if( declrSpecs.contains(PointerSpecifier.CONST) || declrSpecs.contains(PointerSpecifier.CONST_RESTRICT) || 
				declrSpecs.contains(PointerSpecifier.CONST_RESTRICT_VOLATILE) || declrSpecs.contains(PointerSpecifier.CONST_VOLATILE) ||
				(lspecs.contains(Specifier.CONST)&&!SymbolTools.isPointer((Symbol)declr)) || (lspecs.contains(Specifier.STATIC)) ) {
			//Initialization of constant/static variable/pointer should not be separated.
			//System.err.println("Found constant/static variable/pointer: " + decl);
			return;
		} else {
			if( listSize == 1 ) {
				Object initObj = lsm_init.getChildren().get(0);
				if( initObj instanceof Expression ) {
					Expression initValue = (Expression)initObj;
					if( initValue instanceof Literal ) {
						//We don't separate initialization if its value is constant.
						return;
					} else {
						if( declr instanceof VariableDeclarator ) {
							((VariableDeclarator)declr).setInitializer(null);
						} else if( declr instanceof NestedDeclarator ) {
							if( dimsize > 0 ) {
								//Do not separate declaration of a pointer-to-arrays.
								return;
							} else {
							((NestedDeclarator)declr).setInitializer(null);
							}

						} else {
							Tools.exit("[ERROR in DeclarationInitSeparator] unexpected type of declarator: " + declr);
						}
						initValue.setParent(null);
						AssignmentExpression lAssignExp = null;
						if( dimsize == 0 ) {
							lAssignExp = new AssignmentExpression(new Identifier((Symbol)declr),AssignmentOperator.NORMAL,
								initValue);
						} else {
							lAssignExp = new AssignmentExpression(new UnaryExpression(UnaryOperator.DEREFERENCE, new Identifier((Symbol)declr)),AssignmentOperator.NORMAL,
								initValue);
						}
						Statement lAssignStmt = new ExpressionStatement(lAssignExp);
						cStmt.addStatementAfter(declStmt, lAssignStmt);
						/*
					Statement fStmt = IRTools.getFirstNonDeclarationStatement(cStmt);
					if( fStmt == null ) {
						cStmt.addStatement(lAssignStmt);
					} else {
						if( insertedInitStmt == null ) {
							cStmt.addStatementBefore(fStmt, lAssignStmt);
						} else if(insertedInitStmt.getParent().equals(cStmt)) {
							cStmt.addStatementAfter(insertedInitStmt, lAssignStmt);
						} else {
							cStmt.addStatementBefore(fStmt, lAssignStmt);
						}
					}
					insertedInitStmt = lAssignStmt;
						 */
					}
				}
			} else {
				//DEBUG: we don't know how to handle this case yet.
				return;
			}
		}
    }

    /* (non-Javadoc)
     * @see cetus.transforms.TransformPass#start()
     */
    @Override
    public void start() {
    	DFIterator<Declaration> iter =
    			new DFIterator<Declaration>(program, Declaration.class);
    	while (iter.hasNext()) {
    		Declaration d = iter.next();
    		if (d instanceof Procedure) {
    			PrintTools.printlnStatus(2, pass_name, "examining procedure",
    					"\"", ((Procedure)d).getName(), "\"");
    		} else if (d instanceof VariableDeclaration) {
    			separateDeclInitialization((VariableDeclaration)d);
    		}
    	}
    }

}
