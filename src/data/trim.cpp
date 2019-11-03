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

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <opencv2/opencv.hpp>

#include "canny.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

namespace {

cv::Mat trim( fs::path const& infile) {
    cv::Mat img = cv::imread( infile.string() );
    cv::Mat gray;
    cv::cvtColor( img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::GaussianBlur( gray, blurred, cv::Size{ 3, 3 }, 0);
    cv::Mat edged = auto_canny( blurred);
    int height = img.rows;
    int width = img.cols;
    // horizontal
    std::vector< cv::Vec4i > rows;
    cv::HoughLinesP( edged, rows, 1, CV_PI/2, 500, 0.9 * width, 20);
    BOOST_ASSERT( ! rows.empty() );
    std::sort( rows.begin(), rows.end(), [](auto const& l, auto const& r) { return l[1] < r[1]; });
    cv::Vec4i rstart = rows[0], rend = rows[rows.size() - 1];
    // vertical
    std::vector< cv::Vec4i > cols;
    cv::HoughLinesP( edged, cols, 1, CV_PI, 500, 0.3 * height, 20);
    BOOST_ASSERT( ! cols.empty() );
    std::sort( cols.begin(), cols.end(), [](auto const& l, auto const& r) { return l[0] < r[0]; });
    cv::Vec4i cstart = cols[0], cend = cols[cols.size() - 1];
    cv::Rect roi{ cstart[0], rstart[1], cend[0] - cstart[0], rend[1] - rstart[1] };
    return { img( roi) };
}

}

int main( int argc, char** argv) {
    try {
        fs::path infile;
        po::options_description desc("allowed options");
            desc.add_options()
                ("help,h", "help message")
                ("file,f", po::value< fs::path >( & infile), "file name");
        po::variables_map vm;
        po::store(
            po::parse_command_line(
                argc,
                argv,
                desc),
            vm);
        po::notify( vm);
        if ( infile.empty() ) {
            throw std::invalid_argument("no file");
        }
        if ( ! fs::exists( infile) ) {
            throw std::invalid_argument( boost::str( boost::format("'%s' does not exist") % infile) );
        }
        // Loads an image
        cv::imwrite( infile.string(), trim( infile) );
        return EXIT_SUCCESS;
    } catch ( std::exception const& e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unhandled exception" << std::endl;
    }
    return EXIT_FAILURE;
}
