########################################
#Program to generate a training dataset to be used by the classifiers/clustering methods 
#No parallelization...
########################################

#!/usr/bin/perl
use strict;
use warnings;

#Using in-built perl modules
use Getopt::Long;
use Cwd 'realpath';
use Cwd;
use Math::Round;
#Defining prototype for sub-routines
sub commandLine_options;
sub sysRequirement;
sub prtUsage;
sub prtError;
sub makeFeatureHash4;
sub roundOff;

my $logFile=getcwd();
my $string="";
open(LOGF,">",$logFile."/error.log") || die "cant open file";

#Checking systems requirement
&sysRequirement();

my $installation_path=realpath($0);
$installation_path=~s/\/([^\/]+)$//;

#print "installation path $installation_path\n";exit;

my $command_line = join (" ", @ARGV);
#print LOGF "Command used is: perl $0 $command_line\n\n";
print "Command used is: perl $0 $command_line\n\n";

#Retrieve variables for the options provided to User
my ($chrSizeFile,$listFeatureFile,$output_folder,$negFile) = commandLine_options();

#Getting real file path for all the files
$chrSizeFile = realpath($chrSizeFile);
$listFeatureFile = realpath($listFeatureFile);
$negFile = realpath($negFile);

#print "$chrSizeFile\n$listFeatureFile\n$gmsize\n$tss\n$gbFile\n";exit;

`mkdir $output_folder/Kwasnieski_temp`;

`awk '{print \$_"\t"1}' $chrSizeFile > $output_folder/Kwasnieski_temp/pos` ;
`awk '{print \$_"\t"0}' $negFile > $output_folder/Kwasnieski_temp/neg` ;
	
`cat $output_folder/Kwasnieski_temp/pos $output_folder/Kwasnieski_temp/neg > $output_folder/Kwasnieski_temp/training`;

print LOGF "Instances creation are finished, now generating training dataset\n";

$string=$string." "."$output_folder/Kwasnieski_temp/training";
#Processing each feature file
open(LIST,"$listFeatureFile") || die "cant open feature list file";
#If format of the files are different

while(defined(my $_=<LIST>)){
	chomp($_);
	my @s=split(/\t/,$_);
	
	#Check if the file is in BigBed format
	if($s[0]=~/\.bb/){
		my $oFile=$s[0];
		$oFile=~s/.bb/.bed/;
		print LOGF "The file is BigBed format, so convert it into Bed\n";
		`/users/GD/tools/bigbed_to_bed/bigBedToBed $s[0] $oFile`;
		$s[0]=$oFile;
		
	}	 	
	if($s[0]){
		print "$output_folder/Kwasnieski_temp/training\n";
		print "$s[0]\n";
		#`intersectBed -a $output_folder/Kwasnieski_temp/training -b $s[0] -wo -f $fracOverlap | awk -F"\t" '{ if (\$11 != 0) print \$0 }' > $output_folder/Kwasnieski_temp/Kwasnieski_temp_$s[1]`;
		`intersectBed -a $output_folder/Kwasnieski_temp/training -b $s[0] -wo  > $output_folder/Kwasnieski_temp/Kwasnieski_temp_$s[1]`;
	}
	
	else{
		print "Error: input file does not exist \n";
		exit;
	}
	my %chrHash=();
	my $colNo="";
	my @stat_calculate=();
	
	my $chrHash=makeFeatureHash4("$output_folder/Kwasnieski_temp/Kwasnieski_temp_$s[1]");
	%chrHash = %$chrHash;
	open(I,"$output_folder/Kwasnieski_temp/training") || die "can't open file";
	open(OUT,">$output_folder/Kwasnieski_temp/norm_$s[1]") || die "can't open output file\n";	
	while(defined(my $_=<I>))
	{
		chomp($_);
		$_=~/([^\t]+\t[^\t]+\t[^\t]+)/;
		
		if($chrHash{$1}){
			print OUT 1,"\n";
		} 
			#print OUT 1,"\n";
		else{
			#print "$_\tNA\n";
			#push(@stat_calculate, "NA");
			print OUT 0,"\n";
		}
	}
	close(I);
	#Perform normatization with z-score by ignoring NA values as in R
	#open(OUT,">$output_folder/Kwasnieski_temp/norm_$s[1]") || die "can't open output file\n";
	$string=$string." "."$output_folder/Kwasnieski_temp/norm_$s[1]";
	close(OUT);


}

#Concetenate all the files to form a matrix:
print LOGF "Files are: $string\n";

`paste $string | sed 's/\t/_/' | sed 's/\t/_/' > $output_folder/matrix.txt`;
my $header = `cat $listFeatureFile | cut -f2 | tr '\n' '\t' | sed 's/^/Position\tClass\t/'`;
`sed -i '1i$header' $output_folder/matrix.txt`;
exit;

sub makeFeatureHash4(){
	my ($featureKwasnieski_tempFile) = $_[0];
	 
	open(FEA,$featureKwasnieski_tempFile) || die "cant open file";
	
	my %featureHash = ();
	while(defined(my $_=<FEA>)){
		chomp($_);
		my @split_anno_line=split(/\t/,$_);
		
		#if($split_anno_line[0] eq $key){
		my $h_key=$split_anno_line[0]."\t".$split_anno_line[1]."\t".$split_anno_line[2];
		if(!$featureHash{$h_key}){
			 ##Pushing overlap values in the second last column
			$featureHash{$h_key}=1;
		}
		else{
			$featureHash{$h_key}=1; ##Pushing overlap values in the second last column
		}
	}
	return(\%featureHash);
}
	
	

sub commandLine_options{
	my $helpUsage;
	my $output_folder;
	my $chrSizeFile;
	my $listFeatureFile;
	my $negFile;
	$command_line=GetOptions(
			"h|help" => \$helpUsage,
			"chrSize|chrSizeFile=s" => \$chrSizeFile,
			"o|outDir=s" => \$output_folder,
			"l|listFeatureFile=s" => \$listFeatureFile,
			"n|negFile=s" => \$negFile,
          );
	
	if(defined($helpUsage)){
		prtUsage();
	}
	if(!defined($output_folder)){
		$output_folder= getcwd(); #Default: current folder to generate all output files
		$output_folder=$output_folder."/outputFolder";
		`mkdir $output_folder`;
	
		open (LOGF, ">error.log");
		print LOGF "Output dir is not mentioned. So default output directory is $output_folder \(current directory\)\n";
		if(`ls $output_folder | wc -l | cut -d" " -f1` > 0){
			print LOGF "Warn: $output_folder is not empty, please provide another output folder or rename the existing folder\n";
			print "Warn: $output_folder is not empty, please provide another output folder or rename the existing folder\n";exit;
			#`rm -rf $output_folder/*`;
		}
	}
	else{ #Check dir exist or not
		$output_folder=realpath($output_folder);
		if(-d $output_folder){
			unless ($output_folder =~ /\/$/){
				$output_folder =~ s/$/\//;
			}
		}
		else{
			print LOGF "Output dir does not exist, so creating output directory $output_folder\n";
			`mkdir $output_folder`;
			$output_folder =~ s/$/\//;
		}
		if(`ls $output_folder | wc -l | cut -d" " -f1` > 0){
			print LOGF "Warn: $output_folder is not empty, please provide another output folder or rename the existing folder\n";
			print "Warn: $output_folder is not empty, please provide another output folder or rename the existing folder\n";exit;
                        #`rm -rf $output_folder/*`;
		}
	}
	
		
	if(!$listFeatureFile){
		prtError("Feature description file is missing");
	}
	if(!$chrSizeFile){
		prtError("File containing enhancer coordinates is missing");
	}
	if(!$negFile){
		prtError("File containing negative instances is missing");
	}
	return($chrSizeFile,$listFeatureFile,$output_folder,$negFile);
}


sub prtError {
	my $msg = $_[0];
	
	print STDERR "+===================================================================================================================+\n";
	printf STDERR "|%-115s|\n", "  Error:";
	printf STDERR "|%-115s|\n", "       $msg";
	print STDERR "+===================================================================================================================+\n";
	prtUsage();
	exit;
}



sub prtUsage{ # This sub will provide the Usage of the program.
	 print << "HOW_TO";

Description: Form your own training dataset of the feature to try clustering methods

System requirements:
		Perl:
		 Module - Cwd
		 bedtools - Assumed it in the path
Usage:

	Example:perl buildTrainingData.pl -chrSize CHROMSIZEFILE.txt -l FeatureFileList <optional parameters>
	

### Required parameters:
	--chrSize | --chrSizeFile		<A tab delimited file containing chrName, start and end>
					
	--l | --listFeatureFile			<A tab delimited file containing the name of the files (along with the path) and the name of the feature to be displayed>

	--n | --negFile				<A tab delimited file containing chrName, start and end for the negative class>

### Optional parameters:	


	--h | --help				<Print help usage>
	
	--o | --outDir				<output_folder: All the output files will be saved in the output folder>
						default output folder:current folder/output_folder
											
	
	This script was last edited on 29th July 2014.

HOW_TO
print "\n";exit;
}

sub sysRequirement{

	my $bedtools=`bedtools`;
	prtError("bedtools either not installed or not found in the path") if($bedtools =~ /command not found/);
	
	eval {
       		#require Parallel::ForkManager;
        	require Cwd; #Required for shell and R (and so if("xtable" %in% rownames(installed.packages()) == FALSE) {install.packages("xtable")})
	};
	if($@) {
        	my $errorText = join("", $@);
        	#prtError("Parallel Fork manager is not installed; Try to install and run again")if($errorText =~ /Parallel/); 
		prtError("Cwd perl module is not installed; Try to install and run again")if($errorText =~ /Cwd/);
		
	}	
	else{
		print LOGF "System requirements are fine :)\n";
	}
}
  
exit;
	

