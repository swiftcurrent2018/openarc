// $ANTLR 2.7.7 (2010-12-23): "Pre.g" -> "PreCLexer.java"$

package cetus.base.grammars;


import java.io.InputStream;
import antlr.TokenStreamException;
import antlr.TokenStreamIOException;
import antlr.TokenStreamRecognitionException;
import antlr.CharStreamException;
import antlr.CharStreamIOException;
import antlr.ANTLRException;
import java.io.Reader;
import java.util.Hashtable;
import antlr.CharScanner;
import antlr.InputBuffer;
import antlr.ByteBuffer;
import antlr.CharBuffer;
import antlr.Token;
import antlr.CommonToken;
import antlr.RecognitionException;
import antlr.NoViableAltForCharException;
import antlr.MismatchedCharException;
import antlr.TokenStream;
import antlr.ANTLRHashString;
import antlr.LexerSharedInputState;
import antlr.collections.impl.BitSet;
import antlr.SemanticException;

@SuppressWarnings({"unchecked", "cast"})

public class PreCLexer extends antlr.CharScanner implements PreCParserTokenTypes, TokenStream
 {

	int openCount = 0;

public PreCLexer(InputStream in) {
	this(new ByteBuffer(in));
}
public PreCLexer(Reader in) {
	this(new CharBuffer(in));
}
public PreCLexer(InputBuffer ib) {
	this(new LexerSharedInputState(ib));
}
public PreCLexer(LexerSharedInputState state) {
	super(state);
	caseSensitiveLiterals = true;
	setCaseSensitive(true);
	literals = new Hashtable();
}

public Token nextToken() throws TokenStreamException {
	Token theRetToken=null;
tryAgain:
	for (;;) {
		Token _token = null;
		int _ttype = Token.INVALID_TYPE;
		resetText();
		try {   // for char stream error handling
			try {   // for lexical error handling
				if (((LA(1) >= '\u0003' && LA(1) <= '\u00ff'))) {
					mLine(true);
					theRetToken=_returnToken;
				}
				else {
					if (LA(1)==EOF_CHAR) {uponEOF(); _returnToken = makeToken(Token.EOF_TYPE);}
				else {throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());}
				}
				
				if ( _returnToken==null ) continue tryAgain; // found SKIP token
				_ttype = _returnToken.getType();
				_ttype = testLiteralsTable(_ttype);
				_returnToken.setType(_ttype);
				return _returnToken;
			}
			catch (RecognitionException e) {
				throw new TokenStreamRecognitionException(e);
			}
		}
		catch (CharStreamException cse) {
			if ( cse instanceof CharStreamIOException ) {
				throw new TokenStreamIOException(((CharStreamIOException)cse).io);
			}
			else {
				throw new TokenStreamException(cse.getMessage());
			}
		}
	}
}

	public final void mLine(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Line;
		int _saveIndex;
		Token c=null;
		Token rest=null;
		
		{
		_loop6:
		do {
			if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ')) {
				mSpace(false);
			}
			else {
				break _loop6;
			}
			
		} while (true);
		}
		{
		if ((LA(1)=='#') && (_tokenSet_0.member(LA(2)))) {
			match('#');
			_ttype = PreprocDirective;
			{
			_loop9:
			do {
				if ((LA(1)=='\t'||LA(1)=='\u000c'||LA(1)==' ')) {
					mSpace(false);
				}
				else {
					break _loop9;
				}
				
			} while (true);
			}
			mNotSpaceNewline(true);
			c=_returnToken;
			mRest(true);
			rest=_returnToken;
			
			if ((c.getText() + rest.getText()).startsWith("include")) {
			_ttype = Include;
			if (openCount != 0) {
			String text = getText();
			setText("internal"+text);
			}
			}
			
		}
		else if ((LA(1)=='#') && (LA(2)=='\n'||LA(2)=='\r')) {
			match('#');
			mNewline(false);
		}
		else if ((_tokenSet_1.member(LA(1)))) {
			mNotHashSpaceNewline(false);
			mRest(false);
		}
		else if ((LA(1)=='\n'||LA(1)=='\r')) {
			mNewline(false);
		}
		else {
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mSpace(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Space;
		int _saveIndex;
		
		{
		switch ( LA(1)) {
		case ' ':
		{
			match(' ');
			break;
		}
		case '\t':
		{
			match('\t');
			break;
		}
		case '\u000c':
		{
			match('\014');
			break;
		}
		default:
		{
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mNotSpaceNewline(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = NotSpaceNewline;
		int _saveIndex;
		
		if ((LA(1)=='#')) {
			match('#');
		}
		else if ((_tokenSet_1.member(LA(1)))) {
			mNotHashSpaceNewline(false);
		}
		else {
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mRest(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Rest;
		int _saveIndex;
		
		{
		_loop12:
		do {
			switch ( LA(1)) {
			case '#':
			{
				match('#');
				break;
			}
			case '\t':  case '\u000c':  case ' ':
			{
				mSpace(false);
				break;
			}
			default:
				if ((_tokenSet_1.member(LA(1)))) {
					mNotHashSpaceNewline(false);
				}
			else {
				break _loop12;
			}
			}
		} while (true);
		}
		mNewline(false);
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mNewline(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Newline;
		int _saveIndex;
		
		{
		if ((LA(1)=='\r') && (LA(2)=='\n')) {
			match("\r\n");
		}
		else if ((LA(1)=='\n')) {
			match('\n');
		}
		else if ((LA(1)=='\r') && (true)) {
			match('\r');
		}
		else {
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		
		}
		newline();
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mNotHashSpaceNewline(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = NotHashSpaceNewline;
		int _saveIndex;
		
		switch ( LA(1)) {
		case '{':
		{
			mLcurly(false);
			break;
		}
		case '}':
		{
			mRcurly(false);
			break;
		}
		default:
			if ((_tokenSet_2.member(LA(1)))) {
				{
				match(_tokenSet_2);
				}
			}
		else {
			throw new NoViableAltForCharException((char)LA(1), getFilename(), getLine(), getColumn());
		}
		}
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mLcurly(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Lcurly;
		int _saveIndex;
		
		match('{');
		openCount ++;
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	protected final void mRcurly(boolean _createToken) throws RecognitionException, CharStreamException, TokenStreamException {
		int _ttype; Token _token=null; int _begin=text.length();
		_ttype = Rcurly;
		int _saveIndex;
		
		match('}');
		openCount --;
		if ( _createToken && _token==null && _ttype!=Token.SKIP ) {
			_token = makeToken(_ttype);
			_token.setText(new String(text.getBuffer(), _begin, text.length()-_begin));
		}
		_returnToken = _token;
	}
	
	
	private static final long[] mk_tokenSet_0() {
		long[] data = new long[8];
		data[0]=-9224L;
		for (int i = 1; i<=3; i++) { data[i]=-1L; }
		return data;
	}
	public static final BitSet _tokenSet_0 = new BitSet(mk_tokenSet_0());
	private static final long[] mk_tokenSet_1() {
		long[] data = new long[8];
		data[0]=-38654719496L;
		for (int i = 1; i<=3; i++) { data[i]=-1L; }
		return data;
	}
	public static final BitSet _tokenSet_1 = new BitSet(mk_tokenSet_1());
	private static final long[] mk_tokenSet_2() {
		long[] data = new long[8];
		data[0]=-38654719496L;
		data[1]=-2882303761517117441L;
		for (int i = 2; i<=3; i++) { data[i]=-1L; }
		return data;
	}
	public static final BitSet _tokenSet_2 = new BitSet(mk_tokenSet_2());
	
	}
