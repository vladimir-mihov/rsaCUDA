#ifndef MANDELBROTDATA_HPP
#define MANDELBROTDATA_HPP

struct mandelbrotData
{
	mandelbrotData() {}
	mandelbrotData( int w, int h, double startX, double endX, double startY, double endY ) : 
			width(w),
			height(h),
			pixels(w*h),
			startX(startX),
			startY(startY),
			stepX((endX-startX)/w),
			stepY((startY-endY)/h) {}

	int width, height, pixels;
	double startX, startY, stepX, stepY;
};

#endif //MANDELBROTDATA_HPP