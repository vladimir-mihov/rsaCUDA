#include "parser.hpp"
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <exception>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;

parser::parser()
{

}

programOptions parser::parse( int ac, char **av )
{
	programOptions opts;
	string size,rect,output;
	int tCount;

	po::options_description desc("Allowed options");
	desc.add_options()
		("size,s", po::value<string>(&size), "size of the generated image. Format: WIDTHxHEIGHT. Defaults to 640x480.")
		("rect,r", po::value<string>(&rect), "Part of 2D space. Format - a:b:c:d => x from (a,b) and y from (c,d). Defaults to -1.0:3.0:-2.0:2.0.")
		("output,o", po::value<string>(&output), "Output PNG image. Defaults to zad15.png.")
		("tasks,t", po::value<int>(&tCount), "Number of threads per block. Defaults to 1.")	
		("verbose,v", "Verbose mode. Default behavious is quiet-mode.")
		("help,h", "Prints this help massage.")
		;

	po::variables_map vm;
	po::store( po::parse_command_line(ac,av,desc), vm );
	po::notify( vm );

	if( vm.count("help") )
	{
		cout << desc << endl;
		throw logic_error("");
	}

	if( vm.count("size") )
	{
		vector<int> dimensions = split( size, "x", &atoi );
		if( dimensions.size() != 2 ) throw invalid_argument("Format for size is '-s 2000x2000' for example.\n");
		opts.width = dimensions[0]; opts.height = dimensions[1];
	}
	else
	{
		opts.width = 640;
		opts.height = 480;
	}

	if( vm.count("rect") )
	{
		vector<double> area = split( rect, ":", &atof );
		if( area.size() != 4 ) throw invalid_argument("Format for rect is '-r -1:1:-1:1' for example.\n");
		opts.startX = area[0];
		opts.endX = area[1];
		opts.startY = area[2];
		opts.endY = area[3]; 
	}
	else
	{
		opts.startX = -1.0;
		opts.startY = 2.0;
		opts.endX = 3.0;
		opts.endY = -2.0;
	}

	if( vm.count("output") )
	{
		if( output.substr(output.size()-4) != ".png" ) throw invalid_argument("Output image can only be in png format.\n");
		opts.outputFilename = output;
	}
	else
		opts.outputFilename = "zad15.png";

	if( vm.count("tasks") )
		opts.tCount = tCount;
	else
		opts.tCount = 1;

	if( vm.count("verbose") )
		opts.verbose = true;
	else
		opts.verbose = false;

	return opts;
}

template< class T>
vector<T> parser::split( string& input, const char *delims, T (*callback)( const char *str ) )
{
	char *inputCstr = new char[input.size()+1],
		 *part = NULL;
	vector<T> returnVector;

	strcpy( inputCstr, input.c_str() );

	part = strtok( inputCstr, delims );
	while( part != NULL )
	{
		returnVector.push_back( callback(part) );
		part = strtok( NULL, delims );
	}

	delete[] inputCstr;
	return returnVector;
}
