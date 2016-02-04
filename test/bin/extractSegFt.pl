#! /usr/bin/perl
$argn = @ARGV;
if ( $argn == 0 ) {
	print STDOUT "====> Commanline Input missing!\n";
	exit;
}

$bitVec1 = 0x0001;
$bitVec2 = 0x0002;
$bitVec3 = 0x0004;
$bitVec4 = 0x0008;
$bitVec5 = 0x0010;
$bitVec6 = 0x0020;
$bitVec7 = 0x0040;
$bitVec8 = 0x0080;
$bitVec9 = 0x0100;
$bitVec10 = 0x0200;
$bitVec11 = 0x0400;
$bitVec12 = 0x0800;

foreach $ifile (@ARGV)
{
	if( $ifile !~ /\.txt/ ) {
		print STDOUT "Input file: $ifile\n";
		$ofile = "${ifile}_segfaults.txt";
		print STDOUT "Output file: $ofile\n";
		open(INFO, $ifile);
		open(INFO2, ">$ofile");
		my @results = ();
		my $segFault = 0x0000;
		my $numTests = 0;
		my $numNormalExits = 0;
		my $numSegFaults = 0;
		my $numSysHangs = 0;
		my $numCheckedErrors = 0;
		my $numAbnormalExits = 0;
		my $numOutputErrors = 0;
		my $numRecoveredErrors = 0;
		my $numRecoveredErrors2 = 0;
		my $numRecoveredErrors3 = 0;
		my $exitNormally = 0;
		while (my $line = <INFO>) {
			if( $line =~ /Segmentation fault/ ) {
				$segFault = $segFault | $bitVec1;
			}
			if( $line =~ /system hang/ ) {
				$segFault = $segFault | $bitVec2;
			}
			if( $line =~ /Error detected/ ) {
				$segFault = $segFault | $bitVec3;
			}
			if( $line =~ /Error recovered/ ) {
				$segFault = $segFault | $bitVec4;
			} 
			if( $line =~ /^restarting/ ) {
				#print STDOUT "==> Checkpoint restarted\n";
				$segFault = $segFault | $bitVec5;
			} 
			if( $line =~ /online restarting/ ) {
				#print STDOUT "==> Online checkpoint restarted\n";
				$segFault = $segFault | $bitVec6;
			} 
			if( $line =~ /LXIM Error detected/ ) {
				#print STDOUT "==> LXIM/LXIP error detection/recovery\n";
				$segFault = $segFault | $bitVec7;
			}
			if( $line =~ /Output file does not exist/ ) {
				$segFault = $segFault | $bitVec8;
			}
			if( $line =~ /test failed/ ) {
				#print STDOUT "==> Symmetry-based verification detects errors\n";
				$segFault = $segFault | $bitVec9;
			}
			if( $line =~ /Verification/ ) {
				#print STDOUT "==> exit normally\n";
				$exitNormally = 1;
			}
			if( $line =~ /Verification Fail/ ) {
				#print STDOUT "==> built-in verification detects errors\n";
				$segFault = $segFault | $bitVec10;
				$exitNormally = 1;
			}
			if( $line =~ /Verification: Failed/ ) {
				#print STDOUT "==> built-in verification detects errors\n";
				$segFault = $segFault | $bitVec10;
				$exitNormally = 1;
			}
			if( $line =~ /VERIFICATION/ ) {
				#print STDOUT "==> exit normally\n";
				$exitNormally = 1;
			}
			if( $line =~ /VERIFICATION FAILED/ ) {
				#print STDOUT "==> built-in verification detects errors\n";
				$segFault = $segFault | $bitVec10;
				$exitNormally = 1;
			}
			if( $line =~ /SUCCESSFUL/ ) {
				#print STDOUT "==> exit normally\n";
				$exitNormally = 1;
			}
			if( $line =~ /UNSUCCESSFUL/ ) {
				#print STDOUT "==> built-in verification detects errors\n";
				$segFault = $segFault | $bitVec10;
				$exitNormally = 1;
			}
			if( $line =~ /dismatch at/ ) {
				#print STDOUT "==> built-in verification detects errors\n";
				$segFault = $segFault | $bitVec10;
				$exitNormally = 1;
			}
			if( $line =~ /Number of errors/ ) {
				#print STDOUT "==> Number of errors exist!\n";
				$exitNormally = 1;
				@words = split(/\s+/, $line);
				$eValue = $words[$#words];
				if( $eValue > 0 ) {
					#print STDOUT "==> Visible output errors\n";
					$segFault = $segFault | $bitVec10;
				}
			}
			if( $line =~ /__END__ Benchmark/ ) {
				$numTests++;
				#print INFO2 "$segFault\n";
				push(@results, $segFault);
				if( $segFault == 0 ) {
					#print STDOUT "Program exits normally without errors\n";
					$numNormalExits++;
				}
				if( ($segFault & $bitVec1) != 0 ) {
					#print STDOUT "Segmentation fault found\n";
					$numSegFaults++;
				}
				if( ($segFault & $bitVec2) != 0 ) {
					#print STDOUT "System-hang found\n";
					$numSysHangs++;
				}
				if( ($segFault & $bitVec3) != 0 ) {
					#print STDOUT "Detects errors\n";
					$numCheckedErrors++;
				}
				if( ($segFault & $bitVec4) != 0 ) {
					#print STDOUT "ABFT recover errors\n";
					$numRecoveredErrors++;
				}
				if( ($segFault & $bitVec5) != 0 ) {
					#print STDOUT "Checkpoint recover\n";
					$numRecoveredErrors2++;
				}
				if( ($segFault & $bitVec6) != 0 ) {
					#print STDOUT "Online checkpoint recover\n";
					$numRecoveredErrors3++;
				}
				if( ($segFault & $bitVec7) != 0 ) {
					#print STDOUT "==> LXIM/LXIP error detection/recovery\n";
				}
				if( (($segFault & $bitVec8) != 0) || ($exitNormally == 0) ) {
					#print STDOUT "Program exits abnormally\n";
					$numAbnormalExits++;
				}
				if( ($segFault & $bitVec9) != 0 ) {
					#print STDOUT "==> Symmetry-based verification detects errors\n";
				}
				if( ($segFault & $bitVec10) != 0 ) {
					#print STDOUT "==> Visible output errors\n";
					if( ($segFault & $bitVec8) == 0 ) {
						#print STDOUT "==> Exit normally but visible output errors\n";
						$numOutputErrors++;
					}
				}
				$segFault = 0x0000;
				$exitNormally = 0;
			}	
		}
		print STDOUT "# of total tests: $numTests\n";
		print STDOUT "# of normal exits w/o errors: $numNormalExits\n";
		print STDOUT "# of segmentation faults: $numSegFaults\n";
		print STDOUT "# of system hangs: $numSysHangs\n";
		print STDOUT "# of detected faults: $numCheckedErrors\n";
		print STDOUT "# of recovered faults: $numRecoveredErrors\n";
		print STDOUT "# of checkpoint-recovered faults: $numRecoveredErrors2\n";
		print STDOUT "# of online checkpoint-recovered faults: $numRecoveredErrors3\n";
		#$numAbnormalExits = $numTests - $numNormalExits;
		print STDOUT "# of abnormal exits: $numAbnormalExits\n";
		print STDOUT "# of normal exits with output errors: $numOutputErrors\n";

		print INFO2 "NUMTESTS: $numTests";
		print INFO2 " NUMNORMALEXITS: $numNormalExits";
		print INFO2 " NUMSEGFAULTS: $numSegFaults";
		print INFO2 " NUMSYSHANGS: $numSysHangs";
		print INFO2 " NUMERRORDETECTS: $numCheckedErrors";
		print INFO2 " NUMERRORRECOVERS: $numRecoveredErrors";
		print INFO2 " NUMERRORRECOVERS2: $numRecoveredErrors2";
		print INFO2 " NUMERRORRECOVERS3: $numRecoveredErrors3";
		print INFO2 " NUMABNORMALEXITS: $numAbnormalExits";
		print INFO2 " NUMNORMALEXITSWOERRORS: $numOutputErrors\n";
		foreach (@results) {
			#print INFO2 "$_\n";
			$binaryNum = sprintf("%b", $_);
			print INFO2 "$binaryNum\n";
		}

		close(INFO);
		close(INFO2);
	}
}
