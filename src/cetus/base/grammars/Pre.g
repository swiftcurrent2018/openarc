header {
package cetus.base.grammars;

}

{
import java.io.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
@SuppressWarnings({"unchecked", "cast"})
}
class PreCParser extends Parser;

options
        {
        k = 2;
        //exportVocab = PreC;
        //buildAST = true;
        //ASTLabelType = "TNode";

        // Copied following options from java grammar.
        codeGenMakeSwitchThreshold = 2;
        codeGenBitsetTestThreshold = 3;
        }

{

}


programUnit [PrintStream out,String filename]
        {
			// set line 1 and original file name
			out.println("#line 1 \""+filename+"\"");
		}
		:
		(
			in:Include
			{
					String s = in.getText();

					// non global include clause
					if(s.startsWith("internal")){
						out.print(s.substring(8));
					}
					// global include clause
					else{
					    // marker for start of header inclusion
						out.print("#pragma startinclude "+in.getText());
						// adjust line numbering
						out.println("#line "+in.getLine());
						//The actual include clause
						//[Fixed by Joel E. Denny to update relative includes]
						//The output file is sometimes placed in a different
						//directory than the input file, but that can break
						//relative includes, so adjust relative includes.
						//TODO: This does not address the case where the
						//includee does not exist relative to the old
						//directory but does exist relative to the new
						//directory. Perhaps GCC's #include_next can help
						//there.
						Pattern pat
						  = Pattern.compile("([^\"]*\")([^\"]*)(\"[^\"]*)",
						                    Pattern.DOTALL);
						Matcher mat = pat.matcher(s);
						if (mat.matches()) {
							//We have "name" not <name>, so includees are first
							//sought relative to includers.
							String start = mat.group(1);
							String includeeName = mat.group(2);
							String end = mat.group(3);
							if (!new File(includeeName).isAbsolute()) {
								//Includee is specified relatively.
								File includer = new File(filename);
								File includee = new File(includer.getParent(),
								                         includeeName);
								if (includee.exists()) {
									//Includee relative to includer exists, so
									//make sure it's found there.
									s = start + includee.getAbsolutePath()
									          + end;
								}
							}
						}
						out.print(s);
						// marker for end of header inclusion
						out.println("#pragma endinclude");
						// adjust line numbering
						out.println("#line "+(in.getLine()+1));

					}

			}

		| 	pre:PreprocDirective
			{
				out.print(pre.getText());
			}
			
		|	re:Line
			{
				out.print(re.getText());
			}
			
		)+
		;

{
@SuppressWarnings({"unchecked", "cast"})
}

// [Lexical specification rewritten by Joel E. Denny to eliminate ambiguities
// reported by ANTLR and thus various parser bugs.]
//
// TODO: Preprocessor directives within comments should be skipped.

class PreCLexer extends Lexer;

options
        {
        k = 2;
        //exportVocab = PreC;
        //testLiterals = false;
        charVocabulary = '\3'..'\377';
        }



{
	int openCount = 0;

}

Line :
        (Space)*
        (
          '#' {$setType(PreprocDirective);}
          (Space)* c:NotSpaceNewline rest:Rest {
            if ((c.getText() + rest.getText()).startsWith("include")) {
              $setType(Include);
              if (openCount != 0) {
                String text = getText();
                setText("internal"+text);
              }
            }
          }
          | '#' Newline
          | NotHashSpaceNewline Rest
          | Newline
        )
        ;

protected Rest
        : ( '#' | Space | NotHashSpaceNewline )* Newline
        ;

protected NotSpaceNewline
        : '#' | NotHashSpaceNewline
        ;

protected NotHashSpaceNewline
        : 
        ~( '#' | '\n' | '\r' | '{' | '}' | ' ' | '\t' | '\014' )
        | Lcurly
        | Rcurly
        ;

protected Newline
        :       (
				"\r\n"
                | '\n'
				| '\r'
                )
                {newline();}
        ;

protected  Space:
        ( ' ' | '\t' | '\014')
        ;

protected Lcurly
		: '{'	{ openCount ++;}
		;
protected Rcurly
		: '}'   { openCount --;}
		;
