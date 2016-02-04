#! /usr/bin/perl

$argn = @ARGV;
if( $argn != 2 ) 
{
	print STDOUT "$#ARGV\n";
	print STDOUT "====> Incorrect number of inputs! \n";
	print STDOUT "====> Usage: \n";
	print STDOUT "      compareResults.pl reffile compfile\n";
	exit;
}
else 
{
	$reffile = $ARGV[0];
	$compfile = $ARGV[1];
}

unless( -e $compfile ) {
	print STDOUT "Output file does not exist: $compfile\n";
}

open(REFIN, $reffile);
open(COMPIN, $compfile);

my $numErrors = 0;
while (my $refline = <REFIN> ) {
	my $compline = <COMPIN>;
	my @refwords = split(/\s+/, $refline);
	my @compwords = split(/\s+/, $compline);
	my $i = 0;
	foreach my $refV (@refwords) {
		my $compV = $compwords[$i];
		if( "$refV" ne "$compV" ) {
			$numErrors++;
			#print STDOUT "Ref: $refV Comp: $compV\n";
		}
		$i++;
	}
}

print STDOUT "Number of errors: $numErrors\n";
