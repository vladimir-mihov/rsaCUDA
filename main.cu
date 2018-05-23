#include <iostream>
#include <chrono>
#include <math.h>
#include <vector>
#include <cstdint>
#include "mandelbrotData.hpp"
#include "parser/parser.hpp"
#include "lodepng/lodepng.h"

#define NOW chrono::high_resolution_clock::now()
#define DRAW

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
   }
}

__global__ void mandelbrot( mandelbrotData *data, uint8_t *result ) {
	int	index = blockDim.x*blockIdx.x + threadIdx.x,
		col = index % data->width,
		row = index / data->width;


	if( col > data->width-1 || row > data->height-1 ) return;

	double	c_re = data->startX + col*data->stepX,
			c_im = data->startY - row*data->stepY,
			x = 0, y = 0;

	uint8_t iterations = 0;

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
void writePNG( uint8_t *result, programOptions& opts );
#endif

int main( int argc, char ** argv ) {
	// initial variables
	uint8_t *result, *d_result;
	mandelbrotData data, *d_data;
	programOptions opts;

	// initialize variables without leaving junk
	{
		parser cmd;
		try {
			opts = cmd.parse(argc,argv);
		} catch ( const exception& e ) {
			cerr << e.what();
			return 1;
		}
		data = mandelbrotData( opts.width, opts.height, opts.startX, opts.endX, opts.startY, opts.endY );
	}
	cout << (opts.verbose ? "Done parsing command line arguments.\n" : "");

	result = new uint8_t[data.pixels];

	cout << (opts.verbose ? "Allocating memory on the GPU.\n" : "");
	gpuErrchk( cudaMalloc((void **)&d_result, data.pixels) );
	gpuErrchk( cudaMalloc((void **)&d_data, sizeof(mandelbrotData)) );

	gpuErrchk( cudaMemcpy( d_data, &data, sizeof(mandelbrotData), cudaMemcpyHostToDevice ) );

	cout << (opts.verbose ? "Calculating mandelbrot.\n" : "");
	auto t1 = NOW;
	mandelbrot<<<(data.pixels+opts.threadsPerBlock-1)/opts.threadsPerBlock,opts.threadsPerBlock>>>(d_data,d_result);
	cudaDeviceSynchronize();
	auto t2 = NOW;
	cout << (opts.verbose ? "Done. It took " : "") << chrono::duration<double,milli>(t2-t1).count() << " ms.\n";

	gpuErrchk( cudaMemcpy( result, d_result, data.pixels, cudaMemcpyDeviceToHost ) );

#ifdef DRAW
	cout << (opts.verbose ? "Generating png image.\n" : "");
	auto t3 = NOW;
	writePNG( result, opts );
	auto t4 = NOW;
	if( opts.verbose )
		cout << "Done. It took " << chrono::duration<double,milli>(t4-t3).count() << " ms.\n";
#endif

	gpuErrchk( cudaFree(d_result) );
	gpuErrchk( cudaFree(d_data) );
	delete[] result;

	return 0;
}

#ifdef DRAW
void writePNG( uint8_t *result, programOptions& opts )
{
	int w = opts.width, h = opts.height;
	vector<unsigned char> rawPixelData(w*h*4);

	uint32_t	setColor =  (opts.setColor & 0xff0000)>>16 | (opts.setColor & 0xff00) | (opts.setColor & 0xff)<<16 | 0xff<<24,
				nonSetColor1 = (opts.nonSetColor1 & 0xff0000)>>16 | (opts.nonSetColor1 & 0xff00) | (opts.nonSetColor1 & 0xff)<<16 | 0xff<<24,
				nonSetColor2 = (opts.nonSetColor2 & 0xff0000)>>16 | (opts.nonSetColor2 & 0xff00) | (opts.nonSetColor2 & 0xff)<<16 | 0xff<<24;

	for (int y = 0; y < h; ++y)
		for (int x = 0; x < w; ++x)
		{
			int index = 4*w*y + 4*x;
			uint8_t resultElement = result[y*w+x];
			*reinterpret_cast<uint32_t*>(&rawPixelData[index]) = resultElement == 255 ? setColor : ( resultElement % 2 ? nonSetColor1 : nonSetColor2 );
		}
	unsigned int error = lodepng::encode( opts.outputFilename.c_str(), rawPixelData, w, h );
	if( error ) cerr << "encoder error " << error << ": " << lodepng_error_text(error) << endl;
}
#endif
