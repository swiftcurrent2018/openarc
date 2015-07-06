// $ANTLR 2.7.7 (2010-12-23): "NewCParser.g" -> "NewCParser.java"$

package cetus.base.grammars;

public interface NEWCTokenTypes {
	int EOF = 1;
	int NULL_TREE_LOOKAHEAD = 3;
	int LITERAL_typedef = 4;
	int SEMI = 5;
	int VARARGS = 6;
	int LCURLY = 7;
	int LITERAL_asm = 8;
	int LITERAL_volatile = 9;
	int RCURLY = 10;
	int LITERAL_struct = 11;
	int LITERAL_union = 12;
	int LITERAL_enum = 13;
	int LITERAL_auto = 14;
	int LITERAL_register = 15;
	int LITERAL_extern = 16;
	int LITERAL_static = 17;
	int LITERAL_inline = 18;
	int LITERAL_const = 19;
	int LITERAL_restrict = 20;
	int LITERAL___nvl__ = 21;
	int LITERAL___nvl_wp__ = 22;
	int LITERAL_void = 23;
	int LITERAL_char = 24;
	int LITERAL_short = 25;
	int LITERAL_int = 26;
	int LITERAL_long = 27;
	int LITERAL_float = 28;
	int LITERAL_double = 29;
	int LITERAL_signed = 30;
	int LITERAL_unsigned = 31;
	int LITERAL__Bool = 32;
	int LITERAL__Complex = 33;
	int LITERAL__Imaginary = 34;
	int LITERAL___complex = 35;
	int ID = 36;
	int COMMA = 37;
	int COLON = 38;
	int ASSIGN = 39;
	int LITERAL___declspec = 40;
	int LPAREN = 41;
	int RPAREN = 42;
	int Number = 43;
	int StringLiteral = 44;
	int LITERAL___attribute = 45;
	int LITERAL___asm = 46;
	int STAR = 47;
	int LBRACKET = 48;
	int RBRACKET = 49;
	int DOT = 50;
	int LITERAL___label__ = 51;
	int LITERAL_while = 52;
	int LITERAL_do = 53;
	int LITERAL_for = 54;
	int LITERAL_goto = 55;
	int LITERAL_continue = 56;
	int LITERAL_break = 57;
	int LITERAL_return = 58;
	int LITERAL_case = 59;
	int LITERAL_default = 60;
	int LITERAL_if = 61;
	int LITERAL_else = 62;
	int LITERAL_switch = 63;
	int DIV_ASSIGN = 64;
	int PLUS_ASSIGN = 65;
	int MINUS_ASSIGN = 66;
	int STAR_ASSIGN = 67;
	int MOD_ASSIGN = 68;
	int RSHIFT_ASSIGN = 69;
	int LSHIFT_ASSIGN = 70;
	int BAND_ASSIGN = 71;
	int BOR_ASSIGN = 72;
	int BXOR_ASSIGN = 73;
	int LOR = 74;
	int LAND = 75;
	int BOR = 76;
	int BXOR = 77;
	int BAND = 78;
	int EQUAL = 79;
	int NOT_EQUAL = 80;
	int LT = 81;
	int LTE = 82;
	int GT = 83;
	int GTE = 84;
	int LSHIFT = 85;
	int RSHIFT = 86;
	int PLUS = 87;
	int MINUS = 88;
	int DIV = 89;
	int MOD = 90;
	int PTR = 91;
	int INC = 92;
	int DEC = 93;
	int QUESTION = 94;
	int LITERAL_sizeof = 95;
	int LITERAL___alignof__ = 96;
	int LITERAL___builtin_va_arg = 97;
	int LITERAL___builtin_offsetof = 98;
	int BNOT = 99;
	int LNOT = 100;
	int LITERAL___real = 101;
	int LITERAL___imag = 102;
	int CharLiteral = 103;
	int IntOctalConst = 104;
	int LongOctalConst = 105;
	int UnsignedOctalConst = 106;
	int IntIntConst = 107;
	int LongIntConst = 108;
	int UnsignedIntConst = 109;
	int IntHexConst = 110;
	int LongHexConst = 111;
	int UnsignedHexConst = 112;
	int FloatDoubleConst = 113;
	int DoubleDoubleConst = 114;
	int LongDoubleConst = 115;
	int NTypedefName = 116;
	int NInitDecl = 117;
	int NDeclarator = 118;
	int NStructDeclarator = 119;
	int NDeclaration = 120;
	int NCast = 121;
	int NPointerGroup = 122;
	int NExpressionGroup = 123;
	int NFunctionCallArgs = 124;
	int NNonemptyAbstractDeclarator = 125;
	int NInitializer = 126;
	int NStatementExpr = 127;
	int NEmptyExpression = 128;
	int NParameterTypeList = 129;
	int NFunctionDef = 130;
	int NCompoundStatement = 131;
	int NParameterDeclaration = 132;
	int NCommaExpr = 133;
	int NUnaryExpr = 134;
	int NLabel = 135;
	int NPostfixExpr = 136;
	int NRangeExpr = 137;
	int NStringSeq = 138;
	int NInitializerElementLabel = 139;
	int NLcurlyInitializer = 140;
	int NAsmAttribute = 141;
	int NGnuAsmExpr = 142;
	int NTypeMissing = 143;
	int LITERAL___extension__ = 144;
	int Vocabulary = 145;
	int Whitespace = 146;
	int Comment = 147;
	int CPPComment = 148;
	int PREPROC_DIRECTIVE = 149;
	int Space = 150;
	int LineDirective = 151;
	int BadStringLiteral = 152;
	int Escape = 153;
	int IntSuffix = 154;
	int NumberSuffix = 155;
	int Digit = 156;
	int HexDigit = 157;
	int HexFloatTail = 158;
	int Exponent = 159;
	int IDMEAT = 160;
	int WideCharLiteral = 161;
	int WideStringLiteral = 162;
}
