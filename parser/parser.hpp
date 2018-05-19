#ifndef PARSER_HPP
#define PARSER_HPP

#include <iostream>
#include <vector>

struct programOptions
{
	int width, height, tCount;
	float startX, endX, startY, endY;
	std::string outputFilename;
	bool verbose;
};

class parser
{
public:
	parser();
	programOptions parse( int ac, char **av );
private:
	template< class T >
	std::vector<T> split( std::string& input, const char *delims, T (*callback)( const char *str ) );
};


#endif //PARSER_HPP
