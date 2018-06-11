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
	int tCount,zoom;
	vector<string> colors;

	int setDefault = 0, nonSet1Default = 0xffdab9, nonSet2Default = 0x4a708b;

	po::options_description desc("Allowed options");
	desc.add_options()
		("size,s", po::value<string>(&size), "Size of the generated image. Format: WIDTHxHEIGHT. Defaults to 640x480.")
		("rect,r", po::value<string>(&rect), "Part of 2D space. Format - a:b:c:d => x from (a,b) and y from (c,d). Defaults to -1.0:3.0:-2.0:2.0.")
		("zoom,z", po::value<int>(&zoom), "Positive integer indicating the level of zoom.")
		("colors,c", po::value<vector<string>>(&colors)->multitoken(),"3 colors in hex RGB. Format is FFFFFF FFFFFF FFFFFF. First two are main colors of points outside of the set. Third is the color of points in the set. Place 'x' for default color." )
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
		if( dimensions.size() != 2 ) throw invalid_argument("Format for size is -s 2000x2000 for example.\n");
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
		if( area.size() != 4 || area[0] > area[1] || area[2] > area[3] ) throw invalid_argument("Format for rect is -r -1:1:-1:1 for example.\n");
		opts.startX = area[0];
		opts.endX = area[1];
		opts.startY = area[3];
		opts.endY = area[2];
	}
	else
	{
		opts.startX = -1.0;
		opts.startY = 2.0;
		opts.endX = 3.0;
		opts.endY = -2.0;
	}

	if( vm.count("zoom") )
	{
		double	xOffset = (opts.endX-opts.startX)*(1.0-1.0/zoom)/2.0,
				yOffset = (opts.startY-opts.endY)*(1.0-1.0/zoom)/2.0;
		opts.startX += xOffset;
		opts.endX -= xOffset;
		opts.startY -= yOffset;
		opts.endY += yOffset;
	}

	if( vm.count("output") )
	{
		if( output.substr(output.size()-4) != ".png" ) throw invalid_argument("Output image can only be in png format.\n");
		opts.outputFilename = output;
	}
	else
		opts.outputFilename = "zad15.png";

	if( vm.count("tasks") )
		opts.threadsPerBlock = tCount;
	else
		opts.threadsPerBlock = 1;

	if( vm.count("verbose") )
		opts.verbose = true;
	else
		opts.verbose = false;

	if( vm.count("colors") )
	{
		if( colors.size() != 3 ) throw invalid_argument("Format for colors is -c x ffffff x for example.\n");
		opts.nonSetColor1 = colors[0] == "x" || colors[0] == "X" ? nonSet1Default : strtoul(colors[0].c_str(), NULL, 16);
		opts.nonSetColor2 = colors[1] == "x" || colors[1] == "X" ? nonSet2Default : strtoul(colors[1].c_str(), NULL, 16);
		opts.setColor = colors[2] == "x" || colors[2] == "X" ? setDefault : strtoul(colors[2].c_str(), NULL, 16);
	}
	else
	{
		opts.nonSetColor1 = nonSet1Default;
		opts.nonSetColor2 = nonSet2Default;
		opts.setColor = setDefault;
	}

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
