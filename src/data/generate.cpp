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

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/process.hpp>

#include <com/sun/star/beans/XPropertySet.hpp>
#include <com/sun/star/bridge/XUnoUrlResolver.hpp>
#include <com/sun/star/container/XIndexAccess.hpp>
#include <com/sun/star/container/XNameContainer.hpp>
#include <com/sun/star/frame/Desktop.hpp>
#include <com/sun/star/frame/XComponentLoader.hpp>
#include <com/sun/star/frame/XStorable.hpp>
#include <com/sun/star/lang/XMultiComponentFactory.hpp>
#include <com/sun/star/lang/XMultiServiceFactory.hpp>
#include <com/sun/star/registry/XSimpleRegistry.hpp>
#include <com/sun/star/sheet/XSpreadsheetDocument.hpp>
#include <com/sun/star/sheet/XSpreadsheet.hpp>
#include <com/sun/star/style/XStyleFamiliesSupplier.hpp>
#include <com/sun/star/style/XStyle.hpp>
#include <com/sun/star/table/BorderLine.hpp>
#include <com/sun/star/table/CellHoriJustify.hpp>
#include <com/sun/star/table/TableBorder.hpp>
#include <com/sun/star/table/XCell.hpp>
#include <com/sun/star/table/XCell.hpp>
#include <com/sun/star/table/XColumnRowRange.hpp>
#include <com/sun/star/table/XTableColumns.hpp>
#include <com/sun/star/table/XTableRows.hpp>
#include <com/sun/star/text/XTextCursor.hpp>
#include <com/sun/star/text/XText.hpp>
#include <com/sun/star/uno/Any.hxx>
#include <com/sun/star/uno/Reference.hxx>
#include <com/sun/star/util/Color.hpp>
#include <com/sun/star/util/XMergeable.hpp>
#include <com/sun/star/view/PaperOrientation.hpp>
#include <com/sun/star/view/XPrintable.hpp>
#include <cppuhelper/bootstrap.hxx>
#include <rtl/process.h>
#include <rtl/string.hxx>
#include <sal/main.h>

using namespace com::sun::star::beans;
using namespace com::sun::star::bridge;
using namespace com::sun::star::container;
using namespace com::sun::star::frame;
using namespace com::sun::star::lang;
using namespace com::sun::star::sheet;
using namespace com::sun::star::style;
using namespace com::sun::star::table;
using namespace com::sun::star::text;
using namespace com::sun::star::uno;
using namespace com::sun::star::util;
using namespace com::sun::star::view;
using ::rtl::OUString;
using ::rtl::OString;

namespace fs = boost::filesystem;
namespace bp = boost::process;

void
export_document_ods( Reference< XMultiServiceFactory > const& xMainComponent,
                     std::string const& documentFile) {
    Reference< XStorable > xStore{ xMainComponent, UNO_QUERY_THROW };
    Sequence< PropertyValue > storeProps{ 1 };
    storeProps[0].Name = OUString::createFromAscii("Overwrite");
    storeProps[0].Value <<= sal_True;
    xStore->storeToURL( OUString::createFromAscii( documentFile.c_str() ), storeProps);
}

void
export_document_csv( Reference< XMultiServiceFactory > const& xMainComponent,
                     std::string const& documentFile) {
    Reference< XStorable > xStore{ xMainComponent, UNO_QUERY_THROW };
    Sequence< PropertyValue > storeProps{ 2 };
    storeProps[0].Name = OUString::createFromAscii("FilterName");
    storeProps[0].Value <<= OUString::createFromAscii("Text - txt - csv (StarCalc)");
    storeProps[1].Name = OUString::createFromAscii("FilterOptions");
    storeProps[1].Value <<= OUString::createFromAscii("59,34,0,1,1/5/2/1/3/1/4/1");
    xStore->storeToURL( OUString::createFromAscii( documentFile.c_str() ), storeProps);
}

void
export_document_jpeg( Reference< XMultiServiceFactory > const& xMainComponent,
                      std::string const& documentFile) {
    Reference< XStorable > xStore{ xMainComponent, UNO_QUERY_THROW };

    // Sequence< PropertyValue > filterParameters{ 1 };
    Sequence< PropertyValue > filterParameters{ 3 };

    filterParameters[0].Name  = OUString::createFromAscii("Quality");
    filterParameters[0].Value <<= (sal_uInt32) (100);

    // 300 DPI
    sal_Int32 pixelWidth =  2481;
    sal_Int32 pixelHeight = 3507;

    filterParameters[0].Name  = OUString::createFromAscii("PixelWidth");
    filterParameters[0].Value <<= pixelWidth;
    filterParameters[1].Name  = OUString::createFromAscii("PixelHeight");
    filterParameters[1].Value <<= pixelHeight;
    filterParameters[2].Name  = OUString::createFromAscii("Quality");
    filterParameters[2].Value <<= (sal_uInt32) (100);

    Sequence< PropertyValue > storeProps{ 2 };
    storeProps[0].Name = OUString::createFromAscii("FilterName");
    storeProps[0].Value <<= OUString::createFromAscii("calc_jpg_Export");
    storeProps[1].Name  = OUString::createFromAscii("FilterData");
    storeProps[1].Value <<= filterParameters;
    xStore->storeToURL( OUString::createFromAscii( documentFile.c_str() ), storeProps);
}

void
export_document_pdf( Reference< XMultiServiceFactory > const& xMainComponent,
                     std::string const& documentFile) {
    Reference< XStorable > xStore{ xMainComponent, UNO_QUERY_THROW };
    Sequence< PropertyValue > storeProps{ 1 };
    storeProps[0].Name = OUString::createFromAscii("FilterName");
    storeProps[0].Value <<= OUString::createFromAscii("calc_pdf_Export");
    xStore->storeToURL( OUString::createFromAscii( documentFile.c_str() ), storeProps);
}

Reference< XSpreadsheet >
get_sheet( Reference< XSpreadsheetDocument > const& xSpreadsheetDoc, int idx) {
    Reference < XSpreadsheets > xSheets = xSpreadsheetDoc->getSheets() ;
    Reference < XIndexAccess > xIA{ xSheets, UNO_QUERY_THROW };
	return Reference < XSpreadsheet >{ xIA->getByIndex( idx), UNO_QUERY_THROW };
}

void
setColumnWidth( Reference< XSpreadsheet > const& xSheet,
                int colX, int colWidth) {
    Reference< XColumnRowRange > xColumnRowRange{ xSheet, UNO_QUERY_THROW };
    Reference< XTableColumns > xTableColumns = xColumnRowRange->getColumns();
    Any column = xTableColumns->getByIndex( colX);
    Reference < XPropertySet > xColProp{ column, UNO_QUERY_THROW };
    xColProp->setPropertyValue( OUString::createFromAscii("Width"), makeAny( static_cast< long >(10 * colWidth) ) );
}

void
setOptimalColumnWidth( Reference< XSpreadsheet > const& xSheet,
                       int colX) {
    Reference< XColumnRowRange > xColumnRowRange{ xSheet, UNO_QUERY_THROW };
    Reference< XTableColumns > xTableColumns = xColumnRowRange->getColumns();
    Any column = xTableColumns->getByIndex( colX);
    Reference < XPropertySet > xColProp{ column, UNO_QUERY_THROW };
    xColProp->setPropertyValue( OUString::createFromAscii("OptimalWidth"), makeAny( sal_True) );
}

void
setRowHeight( Reference< XSpreadsheet > const& xSheet,
                int rowX, int rowHeight) {
    Reference< XColumnRowRange > xColumnRowRange{ xSheet, UNO_QUERY_THROW };
    Reference< XTableRows > xTableRows = xColumnRowRange->getRows();
    Any row = xTableRows->getByIndex( rowX);
    Reference < XPropertySet > xRowProp{ row, UNO_QUERY_THROW };
    xRowProp->setPropertyValue( OUString::createFromAscii("Height"), makeAny( static_cast< long >(10 * rowHeight) ) );
}

void
setOptimalRowHeight( Reference< XSpreadsheet > const& xSheet,
                int rowX) {
    Reference< XColumnRowRange > xColumnRowRange{ xSheet, UNO_QUERY_THROW };
    Reference< XTableRows > xTableRows = xColumnRowRange->getRows();
    Any row = xTableRows->getByIndex( rowX);
    Reference < XPropertySet > xRowProp{ row, UNO_QUERY_THROW };
    xRowProp->setPropertyValue( OUString::createFromAscii("OptimalHeight"), makeAny( sal_True) );
}

void
setCellFont( Reference< XCell > const& xCell,
             int heightValue) {
    Reference< XPropertySet > xCellsProperties{ xCell, UNO_QUERY_THROW };
    xCellsProperties->setPropertyValue( OUString::createFromAscii("CharHeight"), makeAny( (sal_Int32)heightValue ) );
    //xCellsProperties->setPropertyValue( OUString::createFromAscii("CharWeight"), makeAny( ::com::sun::star::awt::FontWeight::BOLD  ) );
    //xCellsProperties->setPropertyValue( OUString::createFromAscii("CharColor"), makeAny( (sal_Int32)colorValue ) );
}

void
setCellBorder( Reference< XCell > const& xCell,
               int LineWidth,
               int colorValue ) {
    Reference< XPropertySet > xCellsProperties{ xCell, UNO_QUERY_THROW };

    BorderLine theBorderLine;
    theBorderLine.Color = colorValue;
    theBorderLine.OuterLineWidth = LineWidth;

    TableBorder theBorderTable;
    theBorderTable.LeftLine = theBorderLine;
    theBorderTable.TopLine = theBorderLine;
    theBorderTable.RightLine = theBorderLine;
    theBorderTable.BottomLine = theBorderLine;
    theBorderTable.VerticalLine = theBorderTable.HorizontalLine = theBorderLine;
    theBorderTable.IsVerticalLineValid = true;
    theBorderTable.IsHorizontalLineValid = true;
    theBorderTable.IsLeftLineValid = true;
    theBorderTable.IsRightLineValid = true;
    theBorderTable.IsTopLineValid = true;
    theBorderTable.IsBottomLineValid = true;
    xCellsProperties->setPropertyValue( OUString::createFromAscii("TableBorder"), makeAny( theBorderTable) );
}

void
setCellAlignment( Reference< XCell > const& xCell) {
    Reference< XPropertySet > xCellsProperties{ xCell, UNO_QUERY_THROW };
    xCellsProperties->setPropertyValue(
            OUString::createFromAscii("HoriJustify"),
            makeAny( CellHoriJustify::CellHoriJustify_CENTER) );
}

void
setCellText( Reference< XCell > const& xCell,
             std::string const& txt) {
    Reference< XText > xCellText{ xCell, UNO_QUERY_THROW };;
    Reference< XTextCursor > xTextCursor = xCellText->createTextCursor();
    xCellText->insertString( xTextCursor, OUString::createFromAscii( txt.c_str() ), false );
}

double
getCellValue( Reference< XSpreadsheet > const& xSheet,
         int CellX,
         int CellY) {
    Reference< XCell > xCell = xSheet->getCellByPosition( CellX, CellY);
    return xCell->getValue();
}

void
setCell( Reference< XSpreadsheet > const& xSheet,
         int CellX,
         int CellY,
         std::string const& txt) {
    Reference< XCell > xCell = xSheet->getCellByPosition( CellX, CellY);
    setCellAlignment( xCell);
    setCellText( xCell, txt);
    setCellBorder( xCell, 2, 0);
    setCellFont( xCell, 8);
}

void
mergeCells( Reference< XSpreadsheet > const& xSheet,
            std::string const& range) {
    Reference< XCellRange > xCellRange = xSheet->getCellRangeByName( OUString::createFromAscii( range.c_str() ) );
    Reference< XMergeable > xMerge{ xCellRange, UNO_QUERY_THROW };
    xMerge->merge( true);
}

char
create_random_upper_letter() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint8_t > distribution{ 65, 90 };
    const std::uint8_t v = distribution( generator);
    return static_cast< char >( v);
}

std::string
to_currency( double d) {
    std::stringstream ss;
    ss.imbue( std::locale{ "en_US.utf8" } );
    ss << std::showbase << std::put_money( d);
    return ss.str();
}

std::string
to_percent( std::string const& str) {
    return str + "%";
}

std::string
create_symbol() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint8_t > distribution{ 0, 1 };
    const std::uint8_t v = distribution( generator);
    std::string symbol;
    std::uint8_t count = 3 + v;
    for ( std::uint8_t i = 0; i < count; ++i) {
        symbol.push_back( create_random_upper_letter() );
    }
    return symbol;
}

std::tuple< std::uint32_t, std::uint32_t, std::uint32_t >
create_prices() {
    static std::minstd_rand generator_current{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution_current{ 1000, 300000 };
    std::uint32_t now = distribution_current( generator_current);
    static std::minstd_rand generator_offset{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution_offset{ 1, 30 };
    std::uint32_t offset = distribution_offset( generator_offset);
    return std::make_tuple(
            now * (1 + offset/100.)/100.,
            now * (1 - offset/100.)/100.,
            now/100.);
}

std::string
create_potential( std::uint32_t high, std::uint32_t now) {
    char buf[10];
    float f = (static_cast< float >( high)/now - 1) * 100.;
    std::snprintf( buf, sizeof( buf), "%.2f", f);
    return to_percent( buf);
}

std::string
create_drawdown() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::int32_t > distribution{ 500, 2000 };
    char buf[10];
    float f = distribution( generator)/100.;
    std::snprintf( buf, sizeof( buf), "-%.2f", f);
    return to_percent( buf);
}

std::string
create_rangeindex() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::int32_t > distribution{ -20, 100 };
    return std::to_string( distribution( generator) );
}

std::string
create_winodds() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution{ 1, 100 };
    return std::to_string( distribution( generator) );
}

std::string
create_payoff() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution{ 100, 5000 };
    char buf[10];
    float f = distribution( generator)/100.;
    std::snprintf( buf, sizeof( buf), "%.2f", f);
    return to_percent( buf);
}

std::string
create_held() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution{ 1, 100 };
    return std::to_string( distribution( generator) );
}

std::string
create_anualreturn() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution{ 1000, 70000 };
    char buf[10];
    float f = distribution( generator)/100.;
    std::snprintf( buf, sizeof( buf), "%.2f", f);
    return to_percent( buf);
}

std::tuple< std::string, std::string >
create_samplesize() {
    static std::minstd_rand generator_smpl{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution_smpl{ 1, 700 };
    static std::minstd_rand generator_max{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution_max{ 200, 1261 };
    return std::make_tuple(
            std::to_string( distribution_smpl( generator_smpl) ),
            std::to_string( distribution_max( generator_max) ) );
}

std::string
create_creditable() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution{ 0, 500 };
    char buf[10];
    float f = distribution( generator)/100.;
    std::snprintf( buf, sizeof( buf), "%.2f", f);
    return buf;
}

std::string
create_rwdrsk() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution{ 0, 500 };
    char buf[10];
    float f = distribution( generator)/100.;
    std::snprintf( buf, sizeof( buf), "%.1f", f);
    return buf;
}

std::string
create_weighted() {
    static std::minstd_rand generator{ std::random_device{}() };
    static std::uniform_int_distribution< std::uint32_t > distribution{ 100, 10000 };
    char buf[10];
    float f = distribution( generator)/100.;
    std::snprintf( buf, sizeof( buf), "%.1f", f);
    return buf;
}

void
create_data( Reference< XSpreadsheet > const& xSheet, std::size_t row) {
    setCell( xSheet, 0, row, create_symbol() );
    auto [ high, low, now ] = create_prices();
    setCell( xSheet, 1, row, to_currency( high) );
    setCell( xSheet, 2, row, to_currency( low) );
    setCell( xSheet, 3, row, to_currency( now) );
    setCell( xSheet, 4, row, create_potential( high, now) );
    setCell( xSheet, 5, row, create_drawdown() );
    setCell( xSheet, 6, row, create_rangeindex() );
    setCell( xSheet, 7, row, create_winodds() );
    setCell( xSheet, 8, row, create_payoff() );
    setCell( xSheet, 9, row, create_held() );
    setCell( xSheet, 10, row, create_anualreturn() );
    auto [ smplsz, maxsz ] = create_samplesize();
    setCell( xSheet, 11, row, smplsz );
    //setCell( xSheet, 12, row, "of");
    setCell( xSheet, 13, row, maxsz);
    setCell( xSheet, 14, row, create_creditable() );
    setCell( xSheet, 15, row, create_rwdrsk() );
    setCell( xSheet, 16, row, create_weighted() );
}

void
create_header( Reference< XSpreadsheet > const& xSheet) {
    // first row
    setCell( xSheet, 0, 0, "");
    for ( int i = 1; i < 17; ++i) {
        std::stringstream ss;
        ss << "(" << std::to_string(i + 1) << ")";
        setCell( xSheet, i, 0, ss.str() );
    }
    // second row
    mergeCells( xSheet, "L1:N1");
    mergeCells( xSheet, "B2:C2");
    mergeCells( xSheet, "H2:I2");
    mergeCells( xSheet, "L2:N2");
    mergeCells( xSheet, "L3:N3");
    setCell( xSheet, 0, 1, "4/9/2018");
    setCell( xSheet, 1, 1, "Forcast");
    setCell( xSheet, 3, 1, "Price");
    setCell( xSheet, 4, 1, "Sell Target");
    setCell( xSheet, 5, 1, "Worst-Case");
    setCell( xSheet, 6, 1, "Range");
    setCell( xSheet, 7, 1, "From Prior Range Indexes");
    setCell( xSheet, 9, 1, "Days");
    setCell( xSheet, 10, 1, "Annual Rate");
    setCell( xSheet, 11, 1, "Sample");
    setCell( xSheet, 14, 1, "Credible");
    setCell( xSheet, 15, 1, "Rwd~Rsk");
    setCell( xSheet, 16, 1, "(8)-Wghted");
    // third row
    setCell( xSheet, 0, 2, "Symbol");
    setCell( xSheet, 1, 2, "High");
    setCell( xSheet, 2, 2, "Low");
    setCell( xSheet, 3, 2, "Now");
    setCell( xSheet, 4, 2, "Potential");
    setCell( xSheet, 5, 2, "Drawdowns");
    setCell( xSheet, 6, 2, "Index");
    setCell( xSheet, 7, 2, "Win Odds /100");
    setCell( xSheet, 8, 2, "% Payoff");
    setCell( xSheet, 9, 2, "Held");
    setCell( xSheet, 10, 2, "of Return");
    setCell( xSheet, 11, 2, "Size");
    setCell( xSheet, 14, 2, "Ratio");
    setCell( xSheet, 15, 2, "Ratio");
    setCell( xSheet, 16, 2, "(5) & (6)");
}

void
create_first_summary( Reference< XSpreadsheet > const& xSheet) {
    mergeCells( xSheet, "B14:D14");
    setCell( xSheet, 0, 13, "10");
    setCell( xSheet, 1, 13, "Best-Odds Forecast Price Ranges");
    setCell( xSheet, 4, 13, "12.2%");
    setCell( xSheet, 5, 13, "-7.4%");
    setCell( xSheet, 6, 13, "34");
    setCell( xSheet, 7, 13, "83");
    setCell( xSheet, 8, 13, "11.1%");
    setCell( xSheet, 9, 13, "41");
    setCell( xSheet, 10, 13, "93%");
    setCell( xSheet, 11, 13, "173");
    setCell( xSheet, 12, 13, "of");
    setCell( xSheet, 13, 13, "1206");
    setCell( xSheet, 14, 13, "0.9");
    setCell( xSheet, 15, 13, "2.0");
    setCell( xSheet, 16, 13, "43");
}

void
create_last_summary( Reference< XSpreadsheet > const& xSheet) {
    mergeCells( xSheet, "B25:D25");
    setCell( xSheet, 0, 24, "20");
    setCell( xSheet, 1, 24, "Best-Odds Forecast Price Ranges");
    setCell( xSheet, 4, 24, "12.2%");
    setCell( xSheet, 5, 24, "-7.4%");
    setCell( xSheet, 6, 24, "34");
    setCell( xSheet, 7, 24, "83");
    setCell( xSheet, 8, 24, "11.1%");
    setCell( xSheet, 9, 24, "41");
    setCell( xSheet, 10, 24, "93%");
    setCell( xSheet, 11, 24, "173");
    setCell( xSheet, 12, 24, "of");
    setCell( xSheet, 13, 24, "1206");
    setCell( xSheet, 14, 24, "0.9");
    setCell( xSheet, 15, 24, "2.0");
    setCell( xSheet, 16, 24, "43");

    mergeCells( xSheet, "B26:D26");
    setCell( xSheet, 0, 25, "2819");
    setCell( xSheet, 1, 25, "Current Forecast Price Ranges");
    setCell( xSheet, 4, 25, "12.2%");
    setCell( xSheet, 5, 25, "-7.4%");
    setCell( xSheet, 6, 25, "34");
    setCell( xSheet, 7, 25, "83");
    setCell( xSheet, 8, 25, "11.1%");
    setCell( xSheet, 9, 25, "41");
    setCell( xSheet, 10, 25, "93%");
    setCell( xSheet, 11, 25, "173");
    setCell( xSheet, 12, 25, "of");
    setCell( xSheet, 13, 25, "1206");
    setCell( xSheet, 14, 25, "0.9");
    setCell( xSheet, 15, 25, "2.0");
    setCell( xSheet, 16, 25, "43");

    setCell( xSheet, 0, 26, "SPY");
    setCell( xSheet, 1, 26, "$293.51");
    setCell( xSheet, 2, 26, "$245.13");
    setCell( xSheet, 3, 26, "$260.55");
    setCell( xSheet, 4, 26, "12.2%");
    setCell( xSheet, 5, 26, "-7.4%");
    setCell( xSheet, 6, 26, "34");
    setCell( xSheet, 7, 26, "83");
    setCell( xSheet, 8, 26, "11.1%");
    setCell( xSheet, 9, 26, "41");
    setCell( xSheet, 10, 26, "93%");
    setCell( xSheet, 11, 26, "173");
    setCell( xSheet, 12, 26, "of");
    setCell( xSheet, 13, 26, "1206");
    setCell( xSheet, 14, 26, "0.9");
    setCell( xSheet, 15, 26, "2.0");
    setCell( xSheet, 16, 26, "43");
}

SAL_IMPLEMENT_MAIN_WITH_ARGS( argc, argv) {
    // start LibreOffice server with: soffice "-accept=socket,host=localhost,port=2083;urp;StarOffice.ServiceManager"
    OUString sConnectionString = OUString::createFromAscii("uno:socket,host=localhost,port=2083;urp;StarOffice.ServiceManager");
    sal_Int32 nCount = static_cast< sal_Int32 >( rtl_getAppCommandArgCount() );
    if ( 5 != nCount) {
        std::printf("using: generator <number> -env:URE_MORE_TYPES=<office_types_rdb_url> [<uno_connection_url>]\n\n"
                "example: generator 100 -env:URE_MORE_TYPES=\"file:///.../program/offapi.rdb\" \"uno:socket,host=localhost,port=2083;urp;StarOffice.ServiceManager\"\n");
        return -1;
    }
    OUString cstr, nstr, template_uri, outdirstr;
    rtl_getAppCommandArg( 0, & cstr.pData);
    rtl_getAppCommandArg( 1, & nstr.pData);
    rtl_getAppCommandArg( 2, & template_uri.pData);
    rtl_getAppCommandArg( 3, & outdirstr.pData);
    rtl_getAppCommandArg( 4, & sConnectionString.pData);
    std::size_t count = cstr.toInt32();
    std::atomic_int32_t n = nstr.toInt32();
    bp::group g;
    bp::child c{ bp::search_path("soffice"), "--invisible", "--accept=socket,host=localhost,port=2083;urp;StarOffice.ServiceManager", g };
    assert( c.running() );
    Reference< XComponentContext > xComponentContext = cppu::defaultBootstrap_InitialComponentContext();
    // Gets the service manager instance to be used (or null). This method has
    // been added for convenience, because the service manager is a often used
    // object.
    Reference< XMultiComponentFactory > xMultiComponentFactoryClient = xComponentContext->getServiceManager();
    // Creates an instance of a component which supports the services specified by the factory.
    Reference< XInterface > xInterface = xMultiComponentFactoryClient->createInstanceWithContext(
            OUString::createFromAscii("com.sun.star.bridge.UnoUrlResolver"), xComponentContext);
    // Resolves the component context from the office, on the uno URL given by argv[1].
    Reference< XUnoUrlResolver > resolver{ xInterface, UNO_QUERY_THROW };
    bool resolved = false;
    do {
        try {
            xInterface = Reference< XInterface >{ resolver->resolve( sConnectionString ), UNO_QUERY };
            resolved = true;
        } catch ( Exception const& e) {
            std::printf("Error: cannot establish a connection using '%s':\n       %s\n",
                    OUStringToOString( sConnectionString, RTL_TEXTENCODING_ASCII_US).getStr(),
                    OUStringToOString( e.Message, RTL_TEXTENCODING_ASCII_US).getStr() );
            using namespace std::chrono_literals;
            std::this_thread::sleep_for( 250ms);
        }
    } while ( ! resolved);
    // gets the server component context as property of the office component factory
    Reference< XPropertySet > xPropSet{ xInterface, UNO_QUERY_THROW };
    xPropSet->getPropertyValue( OUString::createFromAscii("DefaultContext") ) >>= xComponentContext;
    // gets the service manager from the office
    Reference< XMultiComponentFactory > xMultiComponentFactoryServer = xComponentContext->getServiceManager();
    // Creates an instance of a component which supports the services specified by the factory.
    // Important: using the office component context.
    Reference < XDesktop2 > xComponentLoader = Desktop::create( xComponentContext);
    Sequence< PropertyValue > startProps{ 1 };
    startProps[0].Name = OUString::createFromAscii("Hidden");
    startProps[0].Value <<= true;
    std::vector< std::thread > worker;
    std::mutex m;
    for ( std::size_t i = 0; i < count; ++i) {
        worker.emplace_back( [&](){
            std::int32_t i = 0;
            while ( 0 < (i = n--) ) {
                std::unique_lock< std::mutex > lk{ m };
                Reference< XComponent > xComponent = xComponentLoader->loadComponentFromURL(
                        template_uri, //OUString::createFromAscii("private:factory/scalc"),
                        OUString{ "_blank" },
                        0,
                        startProps);
                lk.unlock();
                Reference< XMultiServiceFactory > xMainComponent = Reference< XMultiServiceFactory >{ xComponent, UNO_QUERY_THROW }; 
                Reference< XSpreadsheetDocument > xSpreadsheetDoc{ xMainComponent, UNO_QUERY_THROW }; 
                Reference< XStyleFamiliesSupplier > xStyleFamiliesSupplier{ xSpreadsheetDoc, UNO_QUERY_THROW };
                Reference< XNameAccess > xNameAccess = xStyleFamiliesSupplier->getStyleFamilies();
                Reference< XNameContainer > xNameContainer{ xNameAccess->getByName( OUString::createFromAscii("PageStyles") ), UNO_QUERY_THROW };
                Reference< XNameAccess > xNameAccess_2{ xNameContainer, UNO_QUERY_THROW };
                Reference< XStyle > xStyle{ xNameAccess_2->getByName( OUString::createFromAscii("Default") ), UNO_QUERY_THROW };
                Reference< XPropertySet > xStyleProps{ xStyle, UNO_QUERY_THROW };
                xStyleProps->setPropertyValue( OUString::createFromAscii("ScaleToPages"), makeAny( static_cast< sal_Int16 >( 1) ) );
                xStyleProps->setPropertyValue( OUString::createFromAscii("LeftMargin"), makeAny( static_cast< sal_Int16 >( 25) ) );
                xStyleProps->setPropertyValue( OUString::createFromAscii("RightMargin"), makeAny( static_cast< sal_Int16 >( 0) ) );
                xStyleProps->setPropertyValue( OUString::createFromAscii("BottomMargin"), makeAny( static_cast< sal_Int16 >( 0) ) );
                xStyleProps->setPropertyValue( OUString::createFromAscii("TopMargin"), makeAny( static_cast< sal_Int16 >( 0) ) );
                // get the first spreadsheet (index 0)
                Reference< XSpreadsheet > xSheet = get_sheet( xSpreadsheetDoc, 0);
                // create header
                //create_header( xSheet);
                // create data: first part
                create_data( xSheet, 3);
                create_data( xSheet, 4);
                create_data( xSheet, 5);
                create_data( xSheet, 6);
                create_data( xSheet, 7);
                create_data( xSheet, 8);
                create_data( xSheet, 9);
                create_data( xSheet, 10);
                create_data( xSheet, 11);
                create_data( xSheet, 12);
                // summary: first part
                //create_first_summary( xSheet);
                create_data( xSheet, 14);
                create_data( xSheet, 15);
                create_data( xSheet, 16);
                create_data( xSheet, 17);
                create_data( xSheet, 18);
                create_data( xSheet, 19);
                create_data( xSheet, 20);
                create_data( xSheet, 21);
                create_data( xSheet, 22);
                create_data( xSheet, 23);
                // create data: second part
                // summary: last part
                //create_last_summary( xSheet);
                // change paper orientation (portrait -> landscape)
                Reference< XPrintable > xPrintable{ xSpreadsheetDoc, UNO_QUERY_THROW };
                Sequence< PropertyValue > printerProps{ 1 };
                printerProps[0].Name = OUString::createFromAscii("PaperOrientation");
                printerProps[0].Value <<= PaperOrientation::PaperOrientation_LANDSCAPE;
                xPrintable->setPrinter( printerProps);
                fs::path outdir{ OUStringToOString( outdirstr, RTL_TEXTENCODING_ASCII_US).pData->buffer };
                if ( ! fs::exists( outdir) ) {
                    fs::create_directory( outdir);
                }
                // export spreadsheet
        //        export_document_ods( xMainComponent, boost::str( boost::format("file://%s/%d.ods") % outdir.string() % i) );
                export_document_csv( xMainComponent, boost::str( boost::format("file://%s/%d.csv") % outdir.string() % i) );
        //        export_document_jpeg( xMainComponent, boost::str( boost::format("file://%s/%d.jpeg") % outdir.string() % i) );
                export_document_pdf( xMainComponent, boost::str( boost::format("file://%s/%d.pdf") % outdir.string() % i) );
                // close spreadsheet
                lk.lock();
                xComponent->dispose();
            }
        });
    }
    for ( auto && w: worker) {
        if ( w.joinable() ) {
            w.join();
        }
    }
    try {
        g.terminate();
        c.wait();
    } catch ( boost::process::process_error const& e) {
        if ( e.code() ) {
            std::cerr << "process_error: " << e.what() << std::endl;
        }
    }
    return 0;
}
