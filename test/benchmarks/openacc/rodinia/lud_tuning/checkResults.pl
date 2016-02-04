#! /usr/bin/perl
$argn = @ARGV;
if( $argn == 0 ) {
	print STDOUT "====> Commandline input missing; exit!\n";
	exit;
}

foreach $ifile (@ARGV)
{
	if( $ifile =~ /tuning_Run/ ) {
		printf STDOUT "Input file: $ifile\n";
		$ofile1 = "${ifile}_extracted1";
		printf STDOUT "Output files: $ofile1\n";
		open(INFO, $ifile);
		open(INFO1, ">$ofile1");
		my $mode = 1;
		my $prevmode = 0;
		my $tbinary = "";
		my $ptbinary = "";
		while (my $line = <INFO>) {
			if( $line =~ /Execution Command:/ ) {
				my @words = split(/\s+/, $line);
				$tbinary = @words[3];
				if( $prevmode == $mode ) {
					print INFO1 "$ptbinary : Abnormal Exit\n";
				}
				$prevmode = $mode;
				$ptbinary = $tbinary;
			}
			if( $line =~ /Verification Failed/ ) {
				print INFO1 "$tbinary : Verification Fail\n";
				++$mode; 
			} elsif( $line =~ /Verification Successful/ ) {
				print INFO1 "$tbinary : Verification Success\n";
				++$mode; 
			} elsif( $line =~ /executable does not exist/ ) {
				print INFO1 "$tbinary : Compilation Fail\n";
				++$mode; 
			}
		}
		if( $prevmode == $mode ) {
			print INFO1 "$ptbinary : Abnormal Exit\n";
		}
		close(INFO);
		close(INFO1);
	}
}
