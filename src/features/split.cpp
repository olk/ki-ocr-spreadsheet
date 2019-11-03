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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <boost/assert.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include "canny.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

std::vector< cv::Vec4i > remove_duplicates( std::vector< cv::Vec4i > & lines, std::size_t pos, std::size_t threshold = 5) {
    BOOST_ASSERT( ! lines.empty() );
    // sort lines ascending (position == horizontal/vertical)
    std::sort( lines.begin(), lines.end(), [pos](auto const& l, auto const& r) { return l[pos] < r[pos]; });
    std::vector< cv::Vec4i > result;
    // push first/top line to results
    result.push_back( lines[0]);
    std::size_t i = 0, size = lines.size();
    while ( i < size) {
        std::size_t match = 0;
        auto p1 = lines[i];
        for ( std::size_t k = i + 1; k < size; ++k) {
            auto p2 = lines[k];
            if ( threshold < std::abs( p2[pos] - p1[pos]) ) {
                match = k;
                break;
            }
        }
        if (0 < match) {
            result.push_back( lines[match]);
            i = match;
        } else {
            ++i;
        }
    }
    return result;
}

bool has_intersection( cv::Vec4i const& h, cv::Vec4i const& v) {
    return h[0] <= v[0] && v[0] <= h[2];
}

std::vector< std::vector< cv::Mat > > split( fs::path const& infile) {
    // load image
    cv::Mat img = cv::imread( infile.string() );
    cv::Mat gray;
    // convert to gray scale
    cv::cvtColor( img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    // apply blur filter to get smoother edges from Canny Edge Detector
    cv::GaussianBlur( gray, blurred, cv::Size{ 3, 3 }, 0);
    // apply Canny Edge Detector
    cv::Mat edged = auto_canny( blurred);
    int height = img.rows;
    int width = img.cols;
    // find horizontal lines
    std::vector< cv::Vec4i > rows;
    cv::HoughLinesP( edged, rows, 1, CV_PI/2, 500, 0.9 * width, 20);
    rows = remove_duplicates( rows, 1);
    BOOST_ASSERT( 28 == rows.size() );
    // veritcal
    std::vector< cv::Vec4i > cols;
    cv::HoughLinesP( edged, cols, 1, CV_PI, 500, 0.3 * height, 20);
    BOOST_ASSERT( ! cols.empty() );
    cols = remove_duplicates( cols, 0);
    BOOST_ASSERT( 18 == cols.size() );
    // horizontal pairs
    std::vector< std::tuple< cv::Vec4i, cv::Vec4i > > horiz{ rows.size() - 1 };
    for ( std::size_t idx = 0, size = horiz.size(); idx < size; ++idx) {
        cv::Vec4i c = rows[idx];
        cv::Vec4i n = rows[idx+1];
        horiz[idx] = std::make_tuple( c, n);
    }
    // vertical pairs
    std::vector< std::tuple< cv::Vec4i, cv::Vec4i > > vertic{ cols.size() - 1 };
    for ( std::size_t idx = 0, size = vertic.size(); idx < size; ++idx) {
        cv::Vec4i c = cols[idx];
        cv::Vec4i n = cols[idx+1];
        vertic[idx] = std::make_tuple( c, n);
    }
    std::size_t idx = 0;
    std::vector< std::vector< cv::Mat > > rois;
    for ( auto& [h1,h2]: horiz) {
        switch ( idx++) {
            // skipp header rows
            case 0:
            case 1:
            case 2:
            // skipp `best 10th` row
            case 13:
            // skipp footer rows
            case 24:
            case 25:
            case 26:
                continue;
            default:
                break;
        }
        std::vector< cv::Mat > segments;
        for ( auto& [v1,v2]: vertic) {
            if ( has_intersection( h1, v1) &&
                    has_intersection( h1, v2) &&
                    has_intersection( h2, v1) &&
                    has_intersection( h2, v2) ) {
                cv::Rect roi{ v1[0], h1[1], v2[0] - v1[0], h2[1] - h1[1] };
                cv::Mat m = img( roi);
                // overwrite remaining boarder lines
                cv::line( m, cv::Point{ 1, 0 }, cv::Point{ 1, m.rows }, cv::Scalar{ 255, 255, 255 }, 2, 8);
                cv::line( m, cv::Point{ 0, 2 }, cv::Point{ m.cols, 2 }, cv::Scalar{ 255, 255, 255 }, 2, 8);
                segments.push_back( m);
            }
        }
        rois.push_back( segments);
    }
    return rois;
}

int main( int argc, char** argv) {
    try {
        fs::path infile, outdir;
        po::options_description desc("allowed options");
            desc.add_options()
                ("help,h", "help message")
                ("file,f", po::value< fs::path >( & infile), "file name")
                ("outdir,o", po::value< fs::path >( & outdir), "output directory");
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
        if ( outdir.empty() ) {
            throw std::invalid_argument("no outdir");
        }
        if ( ! fs::exists( infile) ) {
            throw std::invalid_argument( boost::str( boost::format("'%s' does not exist") % infile) );
        }
        if ( ! fs::exists( outdir) ) {
            throw std::invalid_argument( boost::str( boost::format("'%s' does not exist") % infile) );
        }
        if ( ! outdir.is_absolute() ) {
            outdir = fs::absolute( outdir);
        }
        if ( ! fs::is_directory( outdir) ) {
            throw std::invalid_argument( boost::str( boost::format("'%s' is not a directory") % outdir) );
        }
        fs::path stem = infile.stem();
        std::vector< std::vector< cv::Mat > > lines = split( infile);
        std::size_t row = 1;
        for ( auto && line: lines) {
            std::size_t col = 1, skipped = 0;
            for ( auto const& segment: line) {
                // skipp `of` column
                if ( 13 == col) {
                    ++col;
                    skipped += 1;
                    continue;
                }
                fs::path outfile = outdir / boost::str( boost::format("%s-%d-%d.jpg")
                        % stem.string() % row % (col - skipped));
                cv::imwrite( outfile.string(), segment);
                ++col;
            }
            ++row;
        }
        return EXIT_SUCCESS;
    } catch ( std::exception const& e) {
        std::cerr << "exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "unhandled exception" << std::endl;
    }
    return EXIT_FAILURE;
}
