#include <iostream>
#include <chrono>
#include <math.h>
#include <vector>
#include "mandelbrotData.hpp"
#include "parser/parser.hpp"
#include "lodepng/lodepng.h"

#define NOW chrono::high_resolution_clock::now()

using uchar = unsigned char;
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}

__global__ void mandelbrot( mandelbrotData *data, uchar *result ) {
	int	index = blockDim.x*blockIdx.x + threadIdx.x,
		col = index % data->width,
		row = index / data->width;


	if( col > data->width-1 || row > data->height-1 ) return;

	double	c_re = data->startX + col*data->stepX,
			c_im = data->startY - row*data->stepY,
			x = 0, y = 0;

	uchar iterations = 0;

	while( x*x+y*y < 4 && iterations < 255 )
	{
		double expX = exp(x), sinY, cosY;
		sincos(y,&sinY,&cosY);
		
		double xNew = expX*cosY - c_re;
		y = expX*sinY - c_im;
		x = xNew;
		iterations++;
	}
	result[index] = iterations;
}

#ifdef DRAW
void writePNG( uchar *result, string& outputFilename, mandelbrotData& data );
#endif

int main( int argc, char ** argv ) {
	// initial variables
	uchar *result, *d_result;
	mandelbrotData data, *d_data;
	int threadsPerBlock;
	string outputFilename;
	bool verbose;

	// initialize variables without leaving junk
	{
		parser cmd;
		programOptions opts;
		try {
			opts = cmd.parse(argc,argv);
		} catch ( const exception& e ) {
			cerr << e.what();
			return 1;
		}
		data = mandelbrotData( opts.width, opts.height, opts.startX, opts.endX, opts.startY, opts.endY );
		threadsPerBlock = opts.tCount;
		outputFilename = opts.outputFilename;
		verbose = opts.verbose;
	}
	cout << (verbose ? "Done parsing command line arguments.\n" : "");

	result = new uchar[data.pixels];

	cout << (verbose ? "Allocating memory on the GPU.\n" : "");
	gpuErrchk( cudaMalloc((void **)&d_result, data.pixels*sizeof(uchar)) );
	gpuErrchk( cudaMalloc((void **)&d_data, sizeof(mandelbrotData)) );

	gpuErrchk( cudaMemcpy( d_data, &data, sizeof(mandelbrotData), cudaMemcpyHostToDevice ) );

	cout << (verbose ? "Calculating mandelbrot.\n" : "");
	auto t1 = NOW;
	mandelbrot<<<(data.pixels+threadsPerBlock-1)/threadsPerBlock,threadsPerBlock>>>(d_data,d_result);
	cudaDeviceSynchronize();
	auto t2 = NOW;
	cout << (verbose ? "Done. It took " : "") << chrono::duration<double,milli>(t2-t1).count() << " ms.\n";

	gpuErrchk( cudaMemcpy( result, d_result, data.pixels*sizeof(uchar), cudaMemcpyDeviceToHost ) );

#ifdef DRAW
	cout << (verbose ? "Generating png image.\n" : "");
	auto t3 = NOW;
	writePNG( result, outputFilename, data );
	auto t4 = NOW;
	if( verbose )
		cout << "Done. It took " << chrono::duration<double,milli>(t4-t3).count() << " ms.\n";
#endif

	gpuErrchk( cudaFree(d_result) );
	gpuErrchk( cudaFree(d_data) );
	delete[] result;

	return 0;
}

#ifdef DRAW
void writePNG( uchar *result, string& outputFilename, mandelbrotData& data )
{
	int w = data.width, h = data.height;
	vector<uchar> rawPixelData(w*h*4);
	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x)
		{
			int index = 4*w*y + 4*x;
			uchar resultElement = result[y*w+x];
			rawPixelData[index] = resultElement == 255 ? 0 : resultElement+17;
			rawPixelData[index+1] = resultElement == 255 ? 0 : resultElement+20;
			rawPixelData[index+2] = resultElement == 255 ? 0 : resultElement+40;
			rawPixelData[index+3] = 255;
		}
	unsigned int error = lodepng::encode( outputFilename.c_str(), rawPixelData, w, h );
	if( error ) cerr << "encoder error " << error << ": " << lodepng_error_text(error) << endl;
}
#endif
