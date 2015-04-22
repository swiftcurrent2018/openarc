package openacc.transforms;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.HashSet;
import java.util.HashMap;
import java.util.TreeMap;

import openacc.analysis.ACCAnalysis;
import openacc.analysis.AnalysisTools;
import openacc.hir.ACCAnnotation;
import cetus.exec.Driver;
import cetus.hir.Annotatable;
import cetus.hir.AnnotationDeclaration;
import cetus.hir.ChainedList;
import cetus.hir.CompoundStatement;
import cetus.hir.OmpAnnotation;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declaration;
import cetus.hir.Declarator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.PragmaAnnotation;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.TranslationUnit;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import cetus.hir.NestedDeclarator;
import cetus.transforms.TransformPass;

/**
 * Convert static variables in procedures except for main into global variables.
 * CAVEAT: this conversion may break correctness if the initialization of static variables
 * should be done in a specific order.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class ConvertStatic2Global extends TransformPass {

	public ConvertStatic2Global(Program program) {
		super(program);
	}

	@Override
	public String getPassName() {
		return new String("[convertStatic2Global]");
	}

	@Override
	public void start() {
		String mainEntryFunc = null;
		String value = Driver.getOptionValue("SetAccEntryFunction");
		if( (value != null) && !value.equals("1") ) {
			mainEntryFunc = value;
		}
		List<Procedure> procList = IRTools.getProcedureList(program);
		HashMap<String, String> static2globalMap = new HashMap<String, String>();
		for( Procedure proc : procList ) {
			Declaration orgFirstDecl = null;
			String pname = proc.getSymbolName();
			if ( ((mainEntryFunc != null) && pname.equals(mainEntryFunc)) || 
					((mainEntryFunc == null) && (pname.equals("main") || pname.equals("MAIN__"))) ) {
				// Skip main procedure.
				continue;
			}
			static2globalMap.clear();
			CompoundStatement pBody = proc.getBody();
			CompoundStatement pCStmt = null;
			TranslationUnit tu = (TranslationUnit)proc.getParent();
			//Set<Symbol> symSet = SymbolTools.getVariableSymbols(pBody);
			Set<Symbol> symSet = SymbolTools.getLocalSymbols(pBody);
			Set<Symbol> staticSyms = AnalysisTools.getStaticVariables(symSet);
		    TreeMap<String, Symbol> sortedMap = new TreeMap<String, Symbol>();
			for( Symbol static_var : staticSyms ) {
				String symName = static_var.getSymbolName();
				while( sortedMap.containsKey(symName) && !static_var.equals(sortedMap.get(symName)) ) {
					symName = symName.concat("_");
				}
				sortedMap.put(symName, static_var);
			}
			Collection<Symbol> sortedSet = sortedMap.values();
			int randomnumber = 0;
			for( Symbol sSym : sortedSet ) {
				if( !(sSym instanceof VariableDeclarator) && !(sSym instanceof NestedDeclarator) ) {
					PrintTools.println("\n[WARNING in convertStatic2Global()] unexpected type of symbol : "+sSym + "\n",0);
					continue;
				}
				String oldName = sSym.getSymbolName();
				String newName = oldName.concat("__").concat(pname);
				VariableDeclaration decl = null;
				Declarator cloned_declarator = null;
				DeclarationStatement dStmt = null;
				int linenumber = 0;
				if( sSym instanceof VariableDeclarator ) {
					VariableDeclarator declr = (VariableDeclarator)sSym;
					decl = (VariableDeclaration)declr.getDeclaration();
					dStmt = (DeclarationStatement)decl.getParent();
					linenumber = dStmt.where();
					if( linenumber >= 0 ) {
						newName = newName.concat("_LN") + linenumber;
					} else {
						newName = newName.concat("_RN") + randomnumber++;
					}
					declr.setName(newName); //this change names of identifiers referring to this static symbol.
					cloned_declarator = declr.clone();
				} else if( sSym instanceof NestedDeclarator ) {
					NestedDeclarator declr = (NestedDeclarator)sSym;
					decl = (VariableDeclaration)declr.getDeclaration();
					dStmt = (DeclarationStatement)decl.getParent();
					linenumber = dStmt.where();
					if( linenumber >= 0 ) {
						newName = newName.concat("_LN") + linenumber;
					} else {
						newName = newName.concat("_RN") + randomnumber++;
					}
					declr.setName(newName); //this change names of identifiers referring to this static symbol.
					cloned_declarator = declr.clone();
				}
				static2globalMap.put(oldName, newName);
				// Remove current static symbol from the procedure.
				//pBody.removeChild(dStmt);
				pCStmt = (CompoundStatement)dStmt.getParent();
				pCStmt.removeChild(dStmt);
				//DEBUG: removing a child from DeclarationStatement is not allowed.
				//dStmt.removeChild(decl);
				
				// 
				// Create a cloned Declaration of the static variable.
				//	
				List<Specifier> clonedspecs = new ChainedList<Specifier>();
				clonedspecs.addAll(decl.getSpecifiers());
				//Remove static specifier to be visible to kernel function if the enclosing function is not in the
				//translation unit containing calling compute region.
				clonedspecs.remove(Specifier.STATIC);
				VariableDeclaration cloned_decl = new VariableDeclaration(clonedspecs, cloned_declarator);
				// Create a new global variable in the enclosing translation unit.
				Declaration firstDecl = tu.getFirstDeclaration();
				if( orgFirstDecl == null ) {
					orgFirstDecl = firstDecl;
				}
				tu.addDeclarationBefore(firstDecl, cloned_decl);
				// Replace old ID with the new ID.
				Identifier orgID = new Identifier(sSym);
				Identifier cloned_ID = new Identifier((Symbol)cloned_declarator);
				//DEBUG: below replace is OK even if there exists a local variable, which has the same name as the
				//original static variable, in a nested block, since identifiers referring to the original static
				//symbol is already changed with new name.
				//TransformTools.replaceAll(pBody, orgID, cloned_ID);
				TransformTools.replaceAll(pCStmt, orgID, cloned_ID);
			}
			if( !static2globalMap.isEmpty() ) {
				// Update ACCAnnotation contained in the current procedure.
				ACCAnalysis.updateSymbolsInACCAnnotations(proc.getBody(), static2globalMap);
			}
		}
	}
	
}
