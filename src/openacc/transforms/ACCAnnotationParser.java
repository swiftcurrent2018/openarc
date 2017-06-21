package openacc.transforms;

import java.io.*;
import java.util.*;

import cetus.analysis.*;
import cetus.hir.*;
import cetus.exec.*;
import cetus.transforms.*;
import openacc.hir.*;
import openacc.analysis.*;

/**
 * This pass is used to parse OpenACC annotations that might be
 * added in the C source code input to Cetus and convert them to
 * internal ACCAnnotations. 
 * This pass should be called after cetus.transforms.AnnotationParser
 * is executed.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *         
 */
public class ACCAnnotationParser extends TransformPass
{

	public ACCAnnotationParser(Program program)
	{
		super(program);
	}

	public String getPassName()
	{
		return new String("[ACCAnnotParser]");
	}

	public void start()
	{
		Annotation new_annot = null;
		boolean attach_to_next_annotatable = false;
		HashMap<String, Object> new_map = null;
		HashMap<String, String> macroMap = null;
		LinkedList<Annotation> annots_to_be_attached = new LinkedList<Annotation>();
		HashMap<String, String> gMacroMap = parseCmdLineMacros();
		
		/* Iterate over the program in Depth First order and search for Annotations */
		/* Change this such that each TranslationUnit is searched one-by-one. */
		for( Traversable trUnt : program.getChildren() ) {
			macroMap = new HashMap<String, String>();
			macroMap.putAll(gMacroMap); //Add global commandline macros.
			DFIterator<Traversable> iter = new DFIterator<Traversable>(trUnt);

			while(iter.hasNext())
			{
				Object obj = iter.next();

				////////////////////////////////////////////////////////////////////////
				// AnnotationParser store OpenACC annotations as PragmaAnnotations in //
				// either AnnotationStatements or AnnotationDeclaration, depending on //
				// their locations.                                                   //
				////////////////////////////////////////////////////////////////////////
				if ((obj instanceof AnnotationStatement) || (obj instanceof AnnotationDeclaration) )
				{
					Annotatable annot_container = (Annotatable)obj;
					List<PragmaAnnotation> annot_list = 
						annot_container.getAnnotations(PragmaAnnotation.class);
					if( (annot_list == null) || (annot_list.size() == 0) ) {
						continue;
					}
					////////////////////////////////////////////////////////////////////////////
					// AnnotationParser creates one AnnotationStatement/AnnotationDeclaration //
					// for each PragmaAnnotation.                                             //
					////////////////////////////////////////////////////////////////////////////
					PragmaAnnotation pAnnot = annot_list.get(0);
					//////////////////////////////////////////////////////////////////////////
					// The above pAnnot may be a stand alone CetusAnnotation, OmpAnnotation, //
					// InlineAnnotation, PragmaAnnotation.Event, or PragmaAnnotation.Range. //
					// If so, skip it.                                                      //
					// (The below annotations are child classes of PragmaAnnotation.)       //
					// DEBUG: If new child class of the PragmaAnnotation is added, below    //
					// should be updated too.                                               //
					// (ex: If CudaAnnotation is added, it should be checked here.)         //
					//////////////////////////////////////////////////////////////////////////
					if( pAnnot instanceof CetusAnnotation || pAnnot instanceof OmpAnnotation ||
							pAnnot instanceof InlineAnnotation || pAnnot instanceof PragmaAnnotation.Event ||
							pAnnot instanceof ACCAnnotation || pAnnot instanceof ARCAnnotation ||
							pAnnot instanceof ASPENAnnotation || pAnnot instanceof NVLAnnotation ||
							pAnnot instanceof PragmaAnnotation.Range) {
						continue;
					}
					String old_annot = pAnnot.getName();
					if( old_annot == null ) {
						PrintTools.println("\n[WARNING in ACCAnnotationParser] Pragma annotation, "
								+ pAnnot +", does not have name.\n", 0);
						continue;
					} else if( old_annot.startsWith("unroll") ) {
						//Recognize #pragma unroll N as a PragmaAnnotation and attach it to the following
						//annotatable IR.
						new_annot = pAnnot.clone();
						attach_to_next_annotatable = true;
					} else {
						old_annot = modifyAnnotationString(old_annot);

						/* -------------------------------------------------------------------------
						 * STEP 1:
						 * Find the annotation type by parsing the text in the input annotation and
						 * create a new Annotation of the corresponding type
						 * -------------------------------------------------------------------------
						 */
						String[] token_array = old_annot.split("\\s+");
						// If old_annot string has a leading space, the 2nd token should be checked.
						//String old_annot_key = token_array[1];
						String old_annot_key = token_array[0];
						/* Check for OpenACC annotations */
						if (old_annot_key.compareTo("acc")==0) {
							/* ---------------------------------------------------------------------
							 * Parse the contents:
							 * ACCParser puts the OpenACC directive parsing results into new_map
							 * ---------------------------------------------------------------------
							 */
							new_map = new HashMap<String, Object>();
							attach_to_next_annotatable = ACCParser.parse_acc_pragma(new_map, token_array, macroMap);
							/* Create an ACCAnnotation and copy the parsed contents from new_map
							 * into a new ACCAnnotation */
							new_annot = new ACCAnnotation();
							for (String key : new_map.keySet())
								new_annot.put(key, new_map.get(key));
						} else if ((old_annot_key.compareTo("openarc")==0)) {
							String old_annot_key2 = token_array[1];
							if( old_annot_key2.equals("#") ) { //Preprocess macros on OpenACC/OpenARC directives.
								ACCParser.preprocess_acc_pragma(token_array, macroMap);
								continue;
							} else {
								/* ---------------------------------------------------------------------
								 * Parse the contents:
								 * ACCParser puts the OpenARC directive parsing results into new_map
								 * ---------------------------------------------------------------------
								 */
								new_map = new HashMap<String, Object>();
								attach_to_next_annotatable = ACCParser.parse_openarc_pragma(new_map, token_array, macroMap);
								/* Create an ACCAnnotation and copy the parsed contents from new_map
								 * into a new ACCAnnotation */
								new_annot = new ARCAnnotation();
								for (String key : new_map.keySet())
									new_annot.put(key, new_map.get(key));
							}
						} else if ((old_annot_key.compareTo("aspen")==0)) {
							/* ---------------------------------------------------------------------
							 * Parse the contents:
							 * ACCParser puts the Aspen directive parsing results into new_map
							 * ---------------------------------------------------------------------
							 */
							new_map = new HashMap<String, Object>();
							attach_to_next_annotatable = ACCParser.parse_aspen_pragma(new_map, token_array, macroMap);
							/* Create an ACCAnnotation and copy the parsed contents from new_map
							 * into a new ACCAnnotation */
							new_annot = new ASPENAnnotation();
							for (String key : new_map.keySet())
								new_annot.put(key, new_map.get(key));
						} else if ((old_annot_key.compareTo("nvl")==0)) {
							/* ---------------------------------------------------------------------
							 * Parse the contents:
							 * ACCParser puts the NVL directive parsing results into new_map
							 * ---------------------------------------------------------------------
							 */
							new_map = new HashMap<String, Object>();
							attach_to_next_annotatable = ACCParser.parse_nvl_pragma(new_map, token_array, macroMap);
							/* Create an ACCAnnotation and copy the parsed contents from new_map
							 * into a new ACCAnnotation */
							new_annot = new NVLAnnotation();
							for (String key : new_map.keySet())
								new_annot.put(key, new_map.get(key));
						}
						else {
							//Check whether current annotation accidentally missed acc prefix; if so print error.
							if( ACCAnnotation.OpenACCDirectiveSet.contains(old_annot_key) ) {
								Tools.exit("[ACCAnnotationParsing Error] the following annotation seems to be an OpenACC directive, but" +
										" \"acc\" prefix seems to be omitted. If so, please add it.\nAnnotation: " + pAnnot + "\n");
							} else if( ARCAnnotation.OpenARCDirectiveSet.contains(old_annot_key) ) {
								Tools.exit("[ACCAnnotationParsing Error] the following annotation seems to be an OpenARC directive, but" +
										" \"openarc\" prefix seems to be omitted. If so, please add it.\nAnnotation: " + pAnnot + "\n");
							} else if( ASPENAnnotation.aspen_directives.contains(old_annot_key) ) {
								Tools.exit("[ACCAnnotationParsing Error] the following annotation seems to be an ASPEN directive, but" +
										" \"aspen\" prefix seems to be omitted. If so, please add it.\nAnnotation: " + pAnnot + "\n");
							} else if( NVLAnnotation.nvl_directives.contains(old_annot_key) ) {
								Tools.exit("[ACCAnnotationParsing Error] the following annotation seems to be a NVL directive, but" +
										" \"nvl\" prefix seems to be omitted. If so, please add it.\nAnnotation: " + pAnnot + "\n");
							} else {
								continue;
							}
						}
					}

					/* ----------------------------------------------------------------------------------
					 * STEP 2:
					 * Based on whether the newly created annotation needs to be attached to an Annotatable
					 * object or needs to be inserted as a standalone Annotation contained within
					 * AnnotationStatement or AnnotationDeclaration, perform the following IR
					 * insertion and deletion operations
					 * ----------------------------------------------------------------------------------
					 */
					/* If the annotation doesn't need to be attached to an existing Annotatable object,
					 * remove old PragmaAnnotation and insert the new ACCAnnotation into the existing
					 * container.
					 */
					if (!attach_to_next_annotatable)
					{
						annot_container.removeAnnotations(PragmaAnnotation.class);
						annot_container.annotate(new_annot);

						/* In order to allow non-attached annotations mixed with attached annotations,
						 * check if the to_be_attached list is not empty. If it isn't, some annotations still
						 * exist that need to attached to the very next Annotatable. Hence, ... */
						if ( !annots_to_be_attached.isEmpty() )
							attach_to_next_annotatable = true;

					}
					else 
					{
						/* Add the newly created Annotation to a list of Annotations that will be attached
						 * to the required Annotatable object in the IR
						 */
						annots_to_be_attached.add(new_annot);
						/* Remove the old annotation container from the IR */
						Traversable container_parent = (Traversable)annot_container.getParent();
						container_parent.removeChild(annot_container);
					}
				}
				/* -----------------------------------------------------------------------------------
				 * STEP 3:
				 * A list of newly created Annotations to be attached has been created. Attach it to
				 * the instance of Annotatable object that does not already contain an input Annotation, 
				 * this is encountered next
				 * -----------------------------------------------------------------------------------
				 */
				else if ((obj instanceof DeclarationStatement) &&
						(IRTools.containsClass((Traversable)obj, PreAnnotation.class))) {
					continue;
				}
				else if ((attach_to_next_annotatable) && (obj instanceof Annotatable))
				{
					List<Annotation> attachedSet =  new LinkedList<Annotation>();
					Annotatable container = (Annotatable)obj;
					if (!annots_to_be_attached.isEmpty() && container != null)
					{
						/* Attach the new annotations to this container if valid*/
						for (Annotation annot_to_be_attached : annots_to_be_attached)
							if( annot_to_be_attached instanceof ACCAnnotation ) {
								if( ((ACCAnnotation)annot_to_be_attached).isValidTo(container) ) {
									container.annotate(annot_to_be_attached);
									attachedSet.add(annot_to_be_attached);
								}
							} else if( annot_to_be_attached instanceof ASPENAnnotation ) {
								if( ((ASPENAnnotation)annot_to_be_attached).isValidTo(container) ) {
									container.annotate(annot_to_be_attached);
									attachedSet.add(annot_to_be_attached);
								}
							} else if( annot_to_be_attached instanceof NVLAnnotation ) {
								if( ((NVLAnnotation)annot_to_be_attached).isValidTo(container) ) {
									container.annotate(annot_to_be_attached);
									attachedSet.add(annot_to_be_attached);
								}
							} else {
								container.annotate(annot_to_be_attached);
								attachedSet.add(annot_to_be_attached);
							}
					} 
					else
					{
						System.err.println("[Error ACCAnnotationParser()] unexpected control sequence found");
						System.exit(0);
					}
					annots_to_be_attached.removeAll(attachedSet);
					if( annots_to_be_attached.isEmpty() ) {
						/* reset the flag to false, we've attached all annotations */
						attach_to_next_annotatable = false;
					}
				}
			}
			if( !annots_to_be_attached.isEmpty() ) {
				System.err.println("[Error ACCAnnotationParser()] the following annotations should have been attached " +
						"to annotables; exit!");
				for(Annotation tAnnot : annots_to_be_attached) {
					System.err.println(tAnnot);
				}
				System.exit(0);
			}
		} //end of trUnt loop
	}
	
	static public String modifyAnnotationString(String old_annotation_str)
	{
		String str = null;
		// The delimiter for split operation is white space(\s).
		// Parenthesis, comma, bracket, and colon are delimiters, too. However, we want to leave 
		// them in the pragma token array. Thus, we append a space before and after the
		// parenthesis and colons so that the split operation can recognize them as
		// independent tokens.
		// Sharp (#) is used as a special delimiter for directive preprocessing.
		// Binary operators (+, -, *, /, %) are also used as a delimiter for macro preprocessing.
		// FIXME: below split will not work on class/struct member access expression.
		old_annotation_str = old_annotation_str.replace("(", " ( ");
		old_annotation_str = old_annotation_str.replace(")", " ) ");
		old_annotation_str = old_annotation_str.replace("[", " [ ");
		old_annotation_str = old_annotation_str.replace("]", " ] ");
		old_annotation_str = old_annotation_str.replace(":", " : ");
		old_annotation_str = old_annotation_str.replace(",", " , ");
		old_annotation_str = old_annotation_str.replace("#", " # ");
		old_annotation_str = old_annotation_str.replace("+", " + ");
		old_annotation_str = old_annotation_str.replace("-", " - ");
		old_annotation_str = old_annotation_str.replace("*", " * ");
		old_annotation_str = old_annotation_str.replace("/", " / ");
		old_annotation_str = old_annotation_str.replace("%", " % ");
		old_annotation_str = old_annotation_str.replace("<", " < ");
		old_annotation_str = old_annotation_str.replace(">", " > ");
		old_annotation_str = old_annotation_str.replace("{", " { ");
		old_annotation_str = old_annotation_str.replace("}", " } ");

		str = old_annotation_str;
		return str;
	}
	
	/*
	 * Parse commandline macros and store them into a global map to be used for OpenACC/OpenARC annotation parsing.
	 */
	static public HashMap<String, String> parseCmdLineMacros() {
		HashMap<String, String> gMacroMap = new HashMap<String, String>();
		String macro = Driver.getOptionValue("macro");
		if (macro == null)
			return gMacroMap;
		String[] macro_list = macro.split(",");
		for (int i=0; i<macro_list.length; i++) {
			String tString = macro_list[i];
			String mname = "";
			String mvalue = "1";
			int idx = tString.indexOf("=");
			if( (idx > 0) && (idx < tString.length()) ) { //macro value exists.
				mname = tString.substring(0, idx);
				if( idx+1 == tString.length() ) {
					mvalue = "";
				} else {
					mvalue = tString.substring(idx+1);
				}
			} else if ( idx < 0 ){ //only macro name is defined.
				mname = tString;
			} else {
				System.out.println("Syntax Error in OpenARC Commandline Macro Parsing: " + tString);
				System.out.println("Exit the OpenACC translator!!");
				System.exit(0);
			}
			String openarc_macro_definition = modifyAnnotationString("openarc # define " + mname + " " + mvalue);
			String[] token_array = openarc_macro_definition.split("\\s+");
			
			ACCParser.preprocess_acc_pragma(token_array, gMacroMap);
		}
		return gMacroMap;
	}
	
}


