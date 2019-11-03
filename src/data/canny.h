/*
                     Copyright Oliver Kowalke 2018.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

#include <opencv2/opencv.hpp>

double median( cv::Mat & src) {
    int hist_size = 256;
    float range[] = { 0, 256 };
    const float* hist_range = { range };
    bool uniform = true, accumulate = false;
    cv::Mat hist;
    calcHist( & src, 1, 0, cv::Mat(), hist, 1, & hist_size, & hist_range, uniform, accumulate);
    cv::Mat cdf;
    hist.copyTo( cdf);
    for ( int i = 1; i < hist_size; ++i) {
        cdf.at< float >( i) += cdf.at< float >( i - 1);
    }
    cdf /= src.total();
    double m;
    for ( int i = 0; i < hist_size; ++i){
        if ( 0.5 <= cdf.at< float >( i) ) {
            m = i;
            break;
        }
    }

    return m;
}

cv::Mat auto_canny( cv::Mat & img, float sigma = 0.33, int kernel_size = 3) {
    // compute the median of the single channel pixel intensities
    double m = median( img);
    // apply automatic Canny Edge Detection using the computed median
    int lower = std::max( 0, static_cast< int >((1.0 - sigma) * m));
    int upper = std::min( 255, static_cast< int >((1.0 + sigma) * m));
    cv::Mat edged;
    cv::Canny( img, edged, lower, upper, kernel_size);
    return edged;
}
