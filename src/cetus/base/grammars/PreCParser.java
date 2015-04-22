// $ANTLR 2.7.7 (2010-12-23): "Pre.g" -> "PreCParser.java"$

package cetus.base.grammars;


import antlr.TokenBuffer;
import antlr.TokenStreamException;
import antlr.TokenStreamIOException;
import antlr.ANTLRException;
import antlr.LLkParser;
import antlr.Token;
import antlr.TokenStream;
import antlr.RecognitionException;
import antlr.NoViableAltException;
import antlr.MismatchedTokenException;
import antlr.SemanticException;
import antlr.ParserSharedInputState;
import antlr.collections.impl.BitSet;

import java.io.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
@SuppressWarnings({"unchecked", "cast"})

public class PreCParser extends antlr.LLkParser       implements PreCParserTokenTypes
 {



protected PreCParser(TokenBuffer tokenBuf, int k) {
  super(tokenBuf,k);
  tokenNames = _tokenNames;
}

public PreCParser(TokenBuffer tokenBuf) {
  this(tokenBuf,2);
}

protected PreCParser(TokenStream lexer, int k) {
  super(lexer,k);
  tokenNames = _tokenNames;
}

public PreCParser(TokenStream lexer) {
  this(lexer,2);
}

public PreCParser(ParserSharedInputState state) {
  super(state,2);
  tokenNames = _tokenNames;
}

	public final void programUnit(
		PrintStream out,String filename
	) throws RecognitionException, TokenStreamException {
		
		Token  in = null;
		Token  pre = null;
		Token  re = null;
		
					// set line 1 and original file name
					out.println("#line 1 \""+filename+"\"");
				
		
		try {      // for error handling
			{
			int _cnt3=0;
			_loop3:
			do {
				switch ( LA(1)) {
				case Include:
				{
					in = LT(1);
					match(Include);
					
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
					
								
					break;
				}
				case PreprocDirective:
				{
					pre = LT(1);
					match(PreprocDirective);
					
									out.print(pre.getText());
								
					break;
				}
				case Line:
				{
					re = LT(1);
					match(Line);
					
									out.print(re.getText());
								
					break;
				}
				default:
				{
					if ( _cnt3>=1 ) { break _loop3; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				}
				_cnt3++;
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			reportError(ex);
			recover(ex,_tokenSet_0);
		}
	}
	
	
	public static final String[] _tokenNames = {
		"<0>",
		"EOF",
		"<2>",
		"NULL_TREE_LOOKAHEAD",
		"Include",
		"PreprocDirective",
		"Line",
		"Rest",
		"NotSpaceNewline",
		"NotHashSpaceNewline",
		"Newline",
		"Space",
		"Lcurly",
		"Rcurly"
	};
	
	private static final long[] mk_tokenSet_0() {
		long[] data = { 2L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_0 = new BitSet(mk_tokenSet_0());
	
	}
