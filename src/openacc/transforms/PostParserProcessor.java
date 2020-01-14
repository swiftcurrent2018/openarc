/**
 * 
 */
package openacc.transforms;

import cetus.analysis.LoopTools;
import cetus.hir.Annotation;
import cetus.hir.AnnotationStatement;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declarator;
import cetus.hir.DFIterator;
import cetus.hir.IDExpression;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.Initializer;
import cetus.hir.IRTools;
import cetus.hir.Loop;
import cetus.hir.NullStatement;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.ForLoop;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import cetus.transforms.TransformPass;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.LinkedList;
import java.util.Set;

import openacc.analysis.AnalysisTools;

/**
 * @author f6l
 *
 */
public class PostParserProcessor extends TransformPass {
	private static String pass_name = "[PostParserProcessor]";

	/**
	 * @param program
	 */
	public PostParserProcessor(Program program) {
		super(program);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		// TODO Auto-generated method stub
		return pass_name;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		// Check whether all declaration statements exist at the beginning of a compoundStatement.
		// If not, show warning messages.
		// FIXME: if declarations are inserted after non-decl statements, Cetus parser may parse them as non-decl statements.
		//        If so, below checking will not work.
		//        The above problem occurs if variable is delared with user-specifier, such as struct.
		//        ==> Fixed
		// DEBUG: Cetus parser seems to reorganize declaration statements mixed with expression statements so that 
		// all declaration statements come before non-decl statements. If this is true, below check is useless.
		Set<String> mixedProcSet = new HashSet<String>();
		DFIterator<CompoundStatement> citer =
			new DFIterator<CompoundStatement>(program, CompoundStatement.class);
		while (citer.hasNext()) {
			CompoundStatement cStmt = citer.next();
			//System.err.println("cStmt\n" + cStmt + "\n");
			List<Traversable> children = cStmt.getChildren();
			boolean found_non_decl_stmt = false;
			boolean has_non_decl_stmt_before_decl = false;
			int children_size = children.size();
			for (int i = 0; i < children_size; i++) {
				Traversable child = children.get(i);
				//if (child instanceof DeclarationStatement) {
					//System.err.println("decl_child\n" + child + "\n");
				//}
				if (!(child instanceof DeclarationStatement) &&
						!(child instanceof AnnotationStatement)) {
					found_non_decl_stmt = true;
					//System.err.println("non_decl_child\n" + child + "\n");
				} else if( found_non_decl_stmt && (child instanceof DeclarationStatement) ) {
					has_non_decl_stmt_before_decl = true;
					break;
				}
			}
			if( has_non_decl_stmt_before_decl ) {
				Procedure pProc = IRTools.getParentProcedure(cStmt);
				mixedProcSet.add(pProc.getSymbolName());
			}
		}
		if( !mixedProcSet.isEmpty() ) {
			StringBuilder str = new StringBuilder();
			str.append("\n\n[WARNING] the current OpenARC implementation assumes that input programs follow " +
					"ANSI C standard (a.k.a C89), which does not allow mixed declarations and code. However, " +
					"the following procedures have mixed declarations. Please change the procedures such that all " +
					"delaration statements comes before any expression statements; otherwise, incorrect translation " +
			"may occur!\nList of procedures with mixed declaration: ");
			str.append(PrintTools.collectionToString(mixedProcSet, ", "));
			str.append("\n\n");
			PrintTools.println(str.toString(), 0);
		}
		
		//If forLoop is wrapped by a CompoundStatement due to variable declaration in the forLoop init statement,
		//revert it back to original forLoop and put the declaration statement into the original parent CompoundStatement
		//of the loop.
		List<ForLoop> forLoopList = new LinkedList<ForLoop>();
		DFIterator<ForLoop> iter =
				new DFIterator<ForLoop>(program, ForLoop.class);
		while (iter.hasNext()) {
			ForLoop tLoop = iter.next();
			//PrintTools.println("ForLoop1: " + tLoop, 0);
			Statement initStmt = tLoop.getInitialStatement();
			if( (initStmt == null) || (initStmt instanceof NullStatement) ) {
				//PrintTools.println("ForLoop2: " + tLoop, 0);
				Expression indexVar = LoopTools.getIndexVariable(tLoop);
				CompoundStatement cStmt = (CompoundStatement)tLoop.getParent();
				if( cStmt.getChildren().size() == 2 ) {
					//PrintTools.println("ForLoop3: " + tLoop, 0);
					Declaration lastDecl = IRTools.getLastDeclaration(cStmt);
					if( lastDecl != null ) {
						List<IDExpression> IDs = lastDecl.getDeclaredIDs();
						if( (IDs.size() == 1) && IDs.get(0).equals(indexVar) ) {
							//PrintTools.println("ForLoop4: " + tLoop, 0);
							forLoopList.add(tLoop);
						}
					}
				}
			}
        }
        for(ForLoop tLoop : forLoopList ) {
        	Expression indexVar = LoopTools.getIndexVariable(tLoop);
        	CompoundStatement cStmt = (CompoundStatement)tLoop.getParent();
            Declaration lastDecl = IRTools.getLastDeclaration(cStmt);
            Traversable declStmt = lastDecl.getParent();
            if( lastDecl instanceof VariableDeclaration ) {
            	Declarator declr = ((VariableDeclaration)lastDecl).getDeclarator(0);
            	if( declr instanceof VariableDeclarator ) {
            		Initializer init = declr.getInitializer();
            		Object initObj = init.getChildren().get(0);
            		if( initObj instanceof Expression ) {
            			Expression initVal = (Expression)initObj;
            			declr.setInitializer(null);
            			AssignmentExpression initExp = new AssignmentExpression(indexVar.clone(), 
            					AssignmentOperator.NORMAL, initVal.clone());
            			Statement stmt = new ExpressionStatement(initExp);
            			tLoop.setInitialStatement(stmt);
            			cStmt.removeChild(declStmt);
            			cStmt.removeStatement(tLoop);
            			if( cStmt.getParent() instanceof Loop ) {
            				//If cStmt is a loop body, add a new wrapper body to contain the cStmt. 
            				CompoundStatement tCStmt = new CompoundStatement();
            				cStmt.swapWith(tCStmt);
            				tCStmt.addStatement(cStmt);
            			}
            			cStmt.swapWith(tLoop);
            			List<Annotation> aAnnots = cStmt.getAnnotations();
            			if( aAnnots != null ) {
            				for(Annotation aAn : aAnnots ) {
            					tLoop.annotate(aAn);
            				}
            			}
            			cStmt.removeAnnotations();
            			lastDecl.setParent(null);
            			Traversable t = tLoop;
            			Traversable p = t.getParent();
            			Traversable pp = p.getParent();
            			while( pp instanceof ForLoop ) {
            				boolean isPerfectlyNested = true;
            				List<Traversable> children = p.getChildren();
            				for( Traversable tt : children ) {
            					if( !tt.equals(t) && !(tt instanceof AnnotationStatement) ) {
            						isPerfectlyNested = false;
            						break;
            					}
            				}
            				if( !isPerfectlyNested ) {
            					break;
            				} else {
            					t = pp;
            					p = t.getParent();
            					pp = p.getParent();
            				}
            			}
            			if( !(p instanceof CompoundStatement) ) {
            				Tools.exit("[ERROR in the PostParserProcessor()] The following statement is parsed wrongly.\n"
            						+ "For correct parsing, an if-statement or for-loop, which contains a single child statement with an attached directive, "
            						+ "should use brackets to include the single body statement.\n" + AnalysisTools.getEnclosingContext(p));
            			} else {
            				CompoundStatement cpStmt = (CompoundStatement)p;
            				Set<Symbol> symbols = cpStmt.getSymbols();
            				Symbol tSym = AnalysisTools.getSymbol(symbols, indexVar.toString());
            				if( tSym == null ) {
            					cpStmt.addDeclaration(lastDecl);
            				} else {
            					Traversable tt = tSym.getDeclaration();
            					while ( (tt!= null) && !(tt instanceof Statement) ) {
            						tt = tt.getParent();
            					}
            					if( tt instanceof Statement ) {
            						Statement symStmt = (Statement)tt;
            						Statement loopStmt = (Statement)t;
            						Declaration lastDeclBeforeTheLoop = null;
            						List<Traversable> children = cpStmt.getChildren();
            						boolean foundCurrentLoop = false;
            						boolean foundCurrentSymbol = false;
            						boolean moveDeclaration = false;
            						for( Traversable tChild : children ) {
            							if( !foundCurrentLoop ) {
            								if( tChild instanceof DeclarationStatement ) {
            									lastDeclBeforeTheLoop = ((DeclarationStatement)tChild).getDeclaration();
            								}
            								if( tChild.equals(t) ) {
            									foundCurrentLoop = true;
            									if( foundCurrentSymbol ) {
            										break;
            									}
            								}
            							}
            							if( tChild.equals(tt) ) {
            								if( foundCurrentLoop ) {
            									//Move the current symbol definition before the current loop.
            									moveDeclaration = true;
            								}
            								break;
            							}
            						}
            						if( moveDeclaration ) {
            							cpStmt.removeStatement(symStmt);
            							Declaration symDecl = ((DeclarationStatement)symStmt).getDeclaration();
            							symDecl.setParent(null);
            							if( lastDeclBeforeTheLoop != null ) {
            								cpStmt.addDeclarationAfter(lastDeclBeforeTheLoop, symDecl);
            							} else {
            								cpStmt.addDeclaration(symDecl);
            							}
            						}
            					}
            				}
            			}
            		}
            	}
            }

        }

	}

}
