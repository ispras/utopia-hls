module main(	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2:5
  input         clock, reset,
  input  [15:0] n_in_0_0_x, n_in_0_1_x, n_in_0_2_x, n_in_0_3_x, n_in_0_4_x, n_in_0_5_x,
  input  [15:0] n_in_0_6_x, n_in_0_7_x, n_in_1_0_x, n_in_1_1_x, n_in_1_2_x, n_in_1_3_x,
  input  [15:0] n_in_1_4_x, n_in_1_5_x, n_in_1_6_x, n_in_1_7_x, n_in_2_0_x, n_in_2_1_x,
  input  [15:0] n_in_2_2_x, n_in_2_3_x, n_in_2_4_x, n_in_2_5_x, n_in_2_6_x, n_in_2_7_x,
  input  [15:0] n_in_3_0_x, n_in_3_1_x, n_in_3_2_x, n_in_3_3_x, n_in_3_4_x, n_in_3_5_x,
  input  [15:0] n_in_3_6_x, n_in_3_7_x, n_in_4_0_x, n_in_4_1_x, n_in_4_2_x, n_in_4_3_x,
  input  [15:0] n_in_4_4_x, n_in_4_5_x, n_in_4_6_x, n_in_4_7_x, n_in_5_0_x, n_in_5_1_x,
  input  [15:0] n_in_5_2_x, n_in_5_3_x, n_in_5_4_x, n_in_5_5_x, n_in_5_6_x, n_in_5_7_x,
  input  [15:0] n_in_6_0_x, n_in_6_1_x, n_in_6_2_x, n_in_6_3_x, n_in_6_4_x, n_in_6_5_x,
  input  [15:0] n_in_6_6_x, n_in_6_7_x, n_in_7_0_x, n_in_7_1_x, n_in_7_2_x, n_in_7_3_x,
  input  [15:0] n_in_7_4_x, n_in_7_5_x, n_in_7_6_x, n_in_7_7_x,
  output [15:0] n_out_0_0_x, n_out_0_1_x, n_out_0_2_x, n_out_0_3_x, n_out_0_4_x,
  output [15:0] n_out_0_5_x, n_out_0_6_x, n_out_0_7_x, n_out_1_0_x, n_out_1_1_x,
  output [15:0] n_out_1_2_x, n_out_1_3_x, n_out_1_4_x, n_out_1_5_x, n_out_1_6_x,
  output [15:0] n_out_1_7_x, n_out_2_0_x, n_out_2_1_x, n_out_2_2_x, n_out_2_3_x,
  output [15:0] n_out_2_4_x, n_out_2_5_x, n_out_2_6_x, n_out_2_7_x, n_out_3_0_x,
  output [15:0] n_out_3_1_x, n_out_3_2_x, n_out_3_3_x, n_out_3_4_x, n_out_3_5_x,
  output [15:0] n_out_3_6_x, n_out_3_7_x, n_out_4_0_x, n_out_4_1_x, n_out_4_2_x,
  output [15:0] n_out_4_3_x, n_out_4_4_x, n_out_4_5_x, n_out_4_6_x, n_out_4_7_x,
  output [15:0] n_out_5_0_x, n_out_5_1_x, n_out_5_2_x, n_out_5_3_x, n_out_5_4_x,
  output [15:0] n_out_5_5_x, n_out_5_6_x, n_out_5_7_x, n_out_6_0_x, n_out_6_1_x,
  output [15:0] n_out_6_2_x, n_out_6_3_x, n_out_6_4_x, n_out_6_5_x, n_out_6_6_x,
  output [15:0] n_out_6_7_x, n_out_7_0_x, n_out_7_1_x, n_out_7_2_x, n_out_7_3_x,
  output [15:0] n_out_7_4_x, n_out_7_5_x, n_out_7_6_x, n_out_7_7_x);

  wire [15:0] _delay_INT16_6_2111_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2120:113
  wire [15:0] _delay_INT16_6_2110_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2119:113
  wire [15:0] _delay_INT16_6_2109_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2118:113
  wire [15:0] _delay_INT16_6_2108_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2117:113
  wire [15:0] _delay_INT16_4_2107_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2116:113
  wire [15:0] _delay_INT16_4_2106_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2115:113
  wire [15:0] _delay_INT16_4_2105_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2114:113
  wire [15:0] _delay_INT16_4_2104_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2113:113
  wire [15:0] _delay_INT16_2_2103_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2112:113
  wire [15:0] _delay_INT16_2_2102_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2111:113
  wire [15:0] _delay_INT16_1_2101_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2110:113
  wire [15:0] _delay_INT16_1_2100_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2109:113
  wire [15:0] _delay_INT16_2_2099_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2108:113
  wire [15:0] _delay_INT16_2_2098_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2107:113
  wire [15:0] _delay_INT16_2_2097_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2106:113
  wire [15:0] _delay_INT16_2_2096_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2105:113
  wire [15:0] _delay_INT16_6_2095_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2104:113
  wire [15:0] _delay_INT16_6_2094_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2103:113
  wire [15:0] _delay_INT16_6_2093_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2102:113
  wire [15:0] _delay_INT16_6_2092_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2101:113
  wire [15:0] _delay_INT16_4_2091_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2100:113
  wire [15:0] _delay_INT16_4_2090_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2099:113
  wire [15:0] _delay_INT16_4_2089_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2098:113
  wire [15:0] _delay_INT16_4_2088_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2097:113
  wire [15:0] _delay_INT16_2_2087_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2096:113
  wire [15:0] _delay_INT16_2_2086_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2095:113
  wire [15:0] _delay_INT16_1_2085_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2094:113
  wire [15:0] _delay_INT16_1_2084_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2093:113
  wire [15:0] _delay_INT16_2_2083_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2092:113
  wire [15:0] _delay_INT16_2_2082_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2091:113
  wire [15:0] _delay_INT16_6_2081_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2090:113
  wire [15:0] _delay_INT16_6_2080_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2089:113
  wire [15:0] _delay_INT16_6_2079_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2088:113
  wire [15:0] _delay_INT16_6_2078_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2087:113
  wire [15:0] _delay_INT16_4_2077_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2086:113
  wire [15:0] _delay_INT16_4_2076_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2085:113
  wire [15:0] _delay_INT16_4_2075_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2084:113
  wire [15:0] _delay_INT16_4_2074_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2083:113
  wire [15:0] _delay_INT16_2_2073_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2082:113
  wire [15:0] _delay_INT16_2_2072_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2081:113
  wire [15:0] _delay_INT16_1_2071_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2080:113
  wire [15:0] _delay_INT16_1_2070_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2079:113
  wire [15:0] _delay_INT16_2_2069_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2078:113
  wire [15:0] _delay_INT16_2_2068_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2077:113
  wire [15:0] _delay_INT16_2_2067_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2076:113
  wire [15:0] _delay_INT16_2_2066_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2075:113
  wire [15:0] _delay_INT16_6_2065_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2074:113
  wire [15:0] _delay_INT16_6_2064_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2073:113
  wire [15:0] _delay_INT16_6_2063_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2072:113
  wire [15:0] _delay_INT16_6_2062_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2071:113
  wire [15:0] _delay_INT16_4_2061_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2070:113
  wire [15:0] _delay_INT16_4_2060_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2069:113
  wire [15:0] _delay_INT16_4_2059_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2068:113
  wire [15:0] _delay_INT16_4_2058_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2067:113
  wire [15:0] _delay_INT16_2_2057_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2066:113
  wire [15:0] _delay_INT16_2_2056_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2065:113
  wire [15:0] _delay_INT16_1_2055_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2064:113
  wire [15:0] _delay_INT16_1_2054_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2063:113
  wire [15:0] _delay_INT16_2_2053_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2062:113
  wire [15:0] _delay_INT16_2_2052_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2061:113
  wire [15:0] _delay_INT16_2_2051_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2060:113
  wire [15:0] _delay_INT16_2_2050_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2059:113
  wire [15:0] _delay_INT16_6_2049_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2058:113
  wire [15:0] _delay_INT16_6_2048_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2057:113
  wire [15:0] _delay_INT16_6_2047_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2056:113
  wire [15:0] _delay_INT16_6_2046_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2055:113
  wire [15:0] _delay_INT16_4_2045_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2054:113
  wire [15:0] _delay_INT16_4_2044_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2053:113
  wire [15:0] _delay_INT16_4_2043_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2052:113
  wire [15:0] _delay_INT16_4_2042_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2051:113
  wire [15:0] _delay_INT16_2_2041_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2050:113
  wire [15:0] _delay_INT16_2_2040_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2049:113
  wire [15:0] _delay_INT16_1_2039_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2048:113
  wire [15:0] _delay_INT16_1_2038_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2047:113
  wire [15:0] _delay_INT16_2_2037_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2046:113
  wire [15:0] _delay_INT16_2_2036_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2045:113
  wire [15:0] _delay_INT16_2_2035_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2044:113
  wire [15:0] _delay_INT16_2_2034_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2043:113
  wire [15:0] _delay_INT16_6_2033_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2042:113
  wire [15:0] _delay_INT16_6_2032_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2041:113
  wire [15:0] _delay_INT16_6_2031_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2040:113
  wire [15:0] _delay_INT16_6_2030_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2039:113
  wire [15:0] _delay_INT16_4_2029_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2038:113
  wire [15:0] _delay_INT16_4_2028_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2037:113
  wire [15:0] _delay_INT16_4_2027_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2036:113
  wire [15:0] _delay_INT16_4_2026_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2035:113
  wire [15:0] _delay_INT16_2_2025_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2034:113
  wire [15:0] _delay_INT16_2_2024_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2033:113
  wire [15:0] _delay_INT16_1_2023_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2032:113
  wire [15:0] _delay_INT16_1_2022_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2031:113
  wire [15:0] _delay_INT16_2_2021_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2030:113
  wire [15:0] _delay_INT16_2_2020_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2029:113
  wire [15:0] _delay_INT16_2_2019_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2028:113
  wire [15:0] _delay_INT16_2_2018_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2027:113
  wire [15:0] _delay_INT16_6_2017_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2026:113
  wire [15:0] _delay_INT16_6_2016_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2025:113
  wire [15:0] _delay_INT16_6_2015_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2024:113
  wire [15:0] _delay_INT16_6_2014_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2023:113
  wire [15:0] _delay_INT16_4_2013_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2022:113
  wire [15:0] _delay_INT16_4_2012_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2021:113
  wire [15:0] _delay_INT16_4_2011_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2020:113
  wire [15:0] _delay_INT16_4_2010_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2019:113
  wire [15:0] _delay_INT16_2_2009_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2018:113
  wire [15:0] _delay_INT16_2_2008_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2017:113
  wire [15:0] _delay_INT16_1_2007_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2016:113
  wire [15:0] _delay_INT16_1_2006_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2015:113
  wire [15:0] _delay_INT16_2_2005_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2014:113
  wire [15:0] _delay_INT16_2_2004_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2013:113
  wire [15:0] _delay_INT16_2_2003_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2012:113
  wire [15:0] _delay_INT16_2_2002_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2011:113
  wire [15:0] _delay_INT16_6_2001_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2010:113
  wire [15:0] _delay_INT16_6_2000_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2009:113
  wire [15:0] _delay_INT16_6_1999_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2008:113
  wire [15:0] _delay_INT16_6_1998_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2007:113
  wire [15:0] _delay_INT16_4_1997_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2006:113
  wire [15:0] _delay_INT16_4_1996_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2005:113
  wire [15:0] _delay_INT16_4_1995_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2004:113
  wire [15:0] _delay_INT16_4_1994_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2003:113
  wire [15:0] _delay_INT16_2_1993_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2002:113
  wire [15:0] _delay_INT16_2_1992_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2001:113
  wire [15:0] _delay_INT16_1_1991_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2000:113
  wire [15:0] _delay_INT16_1_1990_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1999:113
  wire [15:0] _delay_INT16_2_1989_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1998:113
  wire [15:0] _delay_INT16_2_1988_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1997:113
  wire [15:0] _delay_INT16_2_1987_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1996:113
  wire [15:0] _delay_INT16_2_1986_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1995:113
  wire [15:0] _delay_INT16_6_1985_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1994:113
  wire [15:0] _delay_INT16_6_1984_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1993:113
  wire [15:0] _delay_INT16_6_1983_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1992:113
  wire [15:0] _delay_INT16_6_1982_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1991:113
  wire [15:0] _delay_INT16_2_1981_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1990:113
  wire [15:0] _delay_INT16_2_1980_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1989:113
  wire [15:0] _delay_INT16_2_1979_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1988:113
  wire [15:0] _delay_INT16_2_1978_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1987:113
  wire [15:0] _delay_INT16_1_1977_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1986:113
  wire [15:0] _delay_INT16_1_1976_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1985:113
  wire [15:0] _delay_INT16_1_1975_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1984:113
  wire [15:0] _delay_INT16_1_1974_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1983:113
  wire [15:0] _delay_INT16_1_1973_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1982:113
  wire [15:0] _delay_INT16_1_1972_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1981:113
  wire [15:0] _delay_INT16_1_1971_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1980:113
  wire [15:0] _delay_INT16_1_1970_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1979:113
  wire [15:0] _delay_INT16_6_1969_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1978:113
  wire [15:0] _delay_INT16_6_1968_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1977:113
  wire [15:0] _delay_INT16_6_1967_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1976:113
  wire [15:0] _delay_INT16_6_1966_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1975:113
  wire [15:0] _delay_INT16_2_1965_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1974:113
  wire [15:0] _delay_INT16_2_1964_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1973:113
  wire [15:0] _delay_INT16_2_1963_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1972:113
  wire [15:0] _delay_INT16_2_1962_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1971:113
  wire [15:0] _delay_INT16_1_1961_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1970:113
  wire [15:0] _delay_INT16_1_1960_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1969:113
  wire [15:0] _delay_INT16_2_1959_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1968:113
  wire [15:0] _delay_INT16_2_1958_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1967:113
  wire [15:0] _delay_INT16_6_1957_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1966:113
  wire [15:0] _delay_INT16_6_1956_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1965:113
  wire [15:0] _delay_INT16_6_1955_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1964:113
  wire [15:0] _delay_INT16_6_1954_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1963:113
  wire [15:0] _delay_INT16_2_1953_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1962:113
  wire [15:0] _delay_INT16_2_1952_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1961:113
  wire [15:0] _delay_INT16_2_1951_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1960:113
  wire [15:0] _delay_INT16_2_1950_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1959:113
  wire [15:0] _delay_INT16_1_1949_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1958:113
  wire [15:0] _delay_INT16_1_1948_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1957:113
  wire [15:0] _delay_INT16_1_1947_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1956:113
  wire [15:0] _delay_INT16_1_1946_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1955:113
  wire [15:0] _delay_INT16_1_1945_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1954:113
  wire [15:0] _delay_INT16_1_1944_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1953:113
  wire [15:0] _delay_INT16_1_1943_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1952:113
  wire [15:0] _delay_INT16_1_1942_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1951:113
  wire [15:0] _delay_INT16_6_1941_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1950:113
  wire [15:0] _delay_INT16_6_1940_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1949:113
  wire [15:0] _delay_INT16_6_1939_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1948:113
  wire [15:0] _delay_INT16_6_1938_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1947:113
  wire [15:0] _delay_INT16_2_1937_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1946:113
  wire [15:0] _delay_INT16_2_1936_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1945:113
  wire [15:0] _delay_INT16_2_1935_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1944:113
  wire [15:0] _delay_INT16_2_1934_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1943:113
  wire [15:0] _delay_INT16_1_1933_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1942:113
  wire [15:0] _delay_INT16_1_1932_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1941:113
  wire [15:0] _delay_INT16_1_1931_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1940:113
  wire [15:0] _delay_INT16_1_1930_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1939:113
  wire [15:0] _delay_INT16_1_1929_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1938:113
  wire [15:0] _delay_INT16_1_1928_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1937:113
  wire [15:0] _delay_INT16_1_1927_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1936:113
  wire [15:0] _delay_INT16_1_1926_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1935:113
  wire [15:0] _delay_INT16_6_1925_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1934:113
  wire [15:0] _delay_INT16_6_1924_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1933:113
  wire [15:0] _delay_INT16_2_1923_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1932:113
  wire [15:0] _delay_INT16_2_1922_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1931:113
  wire [15:0] _delay_INT16_2_1921_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1930:113
  wire [15:0] _delay_INT16_2_1920_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1929:113
  wire [15:0] _delay_INT16_1_1919_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1928:113
  wire [15:0] _delay_INT16_1_1918_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1927:113
  wire [15:0] _delay_INT16_1_1917_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1926:113
  wire [15:0] _delay_INT16_1_1916_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1925:113
  wire [15:0] _delay_INT16_1_1915_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1924:113
  wire [15:0] _delay_INT16_1_1914_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1923:113
  wire [15:0] _delay_INT16_1_1913_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1922:113
  wire [15:0] _delay_INT16_1_1912_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1921:113
  wire [15:0] _delay_INT16_6_1911_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1920:113
  wire [15:0] _delay_INT16_6_1910_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1919:113
  wire [15:0] _delay_INT16_6_1909_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1918:113
  wire [15:0] _delay_INT16_6_1908_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1917:113
  wire [15:0] _delay_INT16_2_1907_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1916:113
  wire [15:0] _delay_INT16_2_1906_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1915:113
  wire [15:0] _delay_INT16_2_1905_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1914:113
  wire [15:0] _delay_INT16_2_1904_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1913:113
  wire [15:0] _delay_INT16_1_1903_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1912:113
  wire [15:0] _delay_INT16_1_1902_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1911:113
  wire [15:0] _delay_INT16_1_1901_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1910:113
  wire [15:0] _delay_INT16_1_1900_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1909:113
  wire [15:0] _delay_INT16_1_1899_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1908:113
  wire [15:0] _delay_INT16_1_1898_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1907:113
  wire [15:0] _delay_INT16_1_1897_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1906:113
  wire [15:0] _delay_INT16_1_1896_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1905:113
  wire [15:0] _delay_INT16_1_1895_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1904:113
  wire [15:0] _delay_INT16_1_1894_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1903:113
  wire [15:0] _delay_INT16_1_1893_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1902:113
  wire [15:0] _delay_INT16_1_1892_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1901:113
  wire [15:0] _delay_INT16_1_1891_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1900:113
  wire [15:0] _delay_INT16_1_1890_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1899:113
  wire [15:0] _delay_INT16_6_1889_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1898:113
  wire [15:0] _delay_INT16_6_1888_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1897:113
  wire [15:0] _delay_INT16_6_1887_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1896:113
  wire [15:0] _delay_INT16_6_1886_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1895:113
  wire [15:0] _delay_INT16_2_1885_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1894:113
  wire [15:0] _delay_INT16_2_1884_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1893:113
  wire [15:0] _delay_INT16_2_1883_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1892:113
  wire [15:0] _delay_INT16_2_1882_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1891:113
  wire [15:0] _delay_INT16_1_1881_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1890:113
  wire [15:0] _delay_INT16_1_1880_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1889:113
  wire [15:0] _delay_INT16_1_1879_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1888:113
  wire [15:0] _delay_INT16_1_1878_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1887:113
  wire [15:0] _delay_INT16_6_1877_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1886:113
  wire [15:0] _delay_INT16_6_1876_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1885:113
  wire [15:0] _delay_INT16_1_1875_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1884:113
  wire [15:0] _delay_INT16_6_1874_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1883:113
  wire [15:0] _delay_INT16_6_1873_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1882:113
  wire [15:0] _delay_INT16_6_1872_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1881:113
  wire [15:0] _delay_INT16_6_1871_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1880:113
  wire [15:0] _delay_INT16_2_1870_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1879:113
  wire [15:0] _delay_INT16_2_1869_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1878:113
  wire [15:0] _delay_INT16_2_1868_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1877:113
  wire [15:0] _delay_INT16_2_1867_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1876:113
  wire [15:0] _delay_INT16_1_1866_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1875:113
  wire [15:0] _delay_INT16_1_1865_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1874:113
  wire [15:0] _delay_INT16_1_1864_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1873:113
  wire [15:0] _delay_INT16_1_1863_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1872:113
  wire [15:0] _delay_INT16_1_1862_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1871:113
  wire [15:0] _delay_INT16_1_1861_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1870:113
  wire [15:0] _delay_INT16_1_1860_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1869:113
  wire [15:0] _delay_INT16_1_1859_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1868:113
  wire [15:0] _delay_INT16_1_1858_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1867:113
  wire [15:0] _delay_INT16_1_1857_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1866:113
  wire [15:0] _delay_INT16_1_1856_out;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1865:113
  wire [15:0] _col_7_n_val_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1863:90
  wire [15:0] _col_7_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1862:108
  wire [15:0] _col_7_n_val_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1860:90
  wire [15:0] _col_7_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1859:108
  wire [15:0] _col_7_n_val_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1857:90
  wire [15:0] _col_7_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1856:108
  wire [15:0] _col_7_n_val_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1854:90
  wire [15:0] _col_7_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1853:108
  wire [15:0] _col_7_n_val_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1851:90
  wire [15:0] _col_7_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1850:108
  wire [15:0] _col_7_n_val_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1848:90
  wire [15:0] _col_7_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1847:108
  wire [15:0] _col_7_n_val_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1845:90
  wire [15:0] _col_7_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1844:108
  wire [15:0] _col_7_n_val_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1842:90
  wire [15:0] _col_7_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1841:108
  wire [15:0] _col_7_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1840:103
  wire [15:0] _col_7_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1840:103
  wire [15:0] _col_7_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1839:86
  wire [15:0] _col_7_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1838:103
  wire [15:0] _col_7_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1837:103
  wire [15:0] _col_7_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1836:103
  wire [15:0] _col_7_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1835:103
  wire [15:0] _col_7_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1835:103
  wire [15:0] _col_7_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1834:86
  wire [15:0] _col_7_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1833:103
  wire [15:0] _col_7_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1832:103
  wire [15:0] _col_7_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1831:103
  wire [15:0] _col_7_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1830:103
  wire [15:0] _col_7_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1830:103
  wire [15:0] _col_7_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1829:103
  wire [15:0] _col_7_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1828:103
  wire [15:0] _col_7_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1828:103
  wire [15:0] _col_7_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1827:103
  wire [15:0] _col_7_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1826:103
  wire [15:0] _col_7_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1826:103
  wire [15:0] _col_7_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1825:103
  wire [15:0] _col_7_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1824:103
  wire [15:0] _col_7_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1824:103
  wire [15:0] _col_7_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1823:103
  wire [15:0] _col_7_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1822:103
  wire [15:0] _col_7_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1822:103
  wire [15:0] _col_7_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1821:103
  wire [15:0] _col_7_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1820:103
  wire [15:0] _col_7_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1820:103
  wire [15:0] _col_7_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1819:103
  wire [15:0] _col_7_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1818:103
  wire [15:0] _col_7_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1818:103
  wire [15:0] _col_7_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1817:103
  wire [15:0] _col_7_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1816:103
  wire [15:0] _col_7_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1816:103
  wire [15:0] _col_7_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1815:103
  wire [15:0] _col_7_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1814:103
  wire [15:0] _col_7_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1814:103
  wire [15:0] _col_7_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1813:86
  wire [15:0] _col_7_n_v3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1812:103
  wire [15:0] _col_7_n_u3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1811:103
  wire [15:0] _col_7_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1810:103
  wire [15:0] _col_7_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1810:103
  wire [15:0] _col_7_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1809:86
  wire [15:0] _col_7_n_v2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1808:103
  wire [15:0] _col_7_n_u2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1807:103
  wire [15:0] _col_7_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1806:103
  wire [15:0] _col_7_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1806:103
  wire [15:0] _col_7_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1805:103
  wire [15:0] _col_7_n_v1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1804:103
  wire [15:0] _col_7_n_u1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1803:103
  wire [15:0] _col_7_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1802:103
  wire [15:0] _col_7_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1802:103
  wire [15:0] _col_7_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1801:103
  wire [15:0] _col_7_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1800:103
  wire [15:0] _col_7_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1800:103
  wire [15:0] _col_7_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1799:103
  wire [15:0] _col_7_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1798:103
  wire [15:0] _col_7_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1798:103
  wire [15:0] _col_7_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1797:86
  wire [15:0] _col_7_n_v7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1796:103
  wire [15:0] _col_7_n_u7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1795:103
  wire [15:0] _col_7_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1794:103
  wire [15:0] _col_7_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1794:103
  wire [15:0] _col_7_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1793:86
  wire [15:0] _col_7_n_v6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1792:103
  wire [15:0] _col_7_n_u6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1791:103
  wire [15:0] _col_7_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1790:103
  wire [15:0] _col_7_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1790:103
  wire [15:0] _col_7_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1789:103
  wire [15:0] _col_7_n_v8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1788:103
  wire [15:0] _col_7_n_u8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1787:103
  wire [15:0] _col_7_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1786:103
  wire [15:0] _col_7_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1786:103
  wire [15:0] _col_7_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1785:86
  wire [15:0] _col_7_n_v5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1784:103
  wire [15:0] _col_7_n_u5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1783:103
  wire [15:0] _col_7_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1782:103
  wire [15:0] _col_7_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1782:103
  wire [15:0] _col_7_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1781:86
  wire [15:0] _col_7_n_v4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1780:103
  wire [15:0] _col_7_n_u4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1779:103
  wire [15:0] _col_7_d_x8_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1778:103
  wire [15:0] _col_7_d_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1778:103
  wire [15:0] _col_7_n_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1777:103
  wire [15:0] _col_7_n_v8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1776:103
  wire [15:0] _col_7_n_u8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1775:103
  wire [15:0] _col_7_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1774:103
  wire [15:0] _col_7_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1774:103
  wire [15:0] _col_7_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1773:103
  wire [15:0] _col_7_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1773:103
  wire [15:0] _col_7_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1772:103
  wire [15:0] _col_7_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1772:103
  wire [15:0] _col_7_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1771:103
  wire [15:0] _col_7_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1771:103
  wire [15:0] _col_7_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1770:103
  wire [15:0] _col_7_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1770:103
  wire [15:0] _col_7_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1769:103
  wire [15:0] _col_7_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1769:103
  wire [15:0] _col_7_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1768:103
  wire [15:0] _col_7_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1768:103
  wire [15:0] _col_7_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1767:103
  wire [15:0] _col_7_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1767:103
  wire [15:0] _col_7_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1766:103
  wire [15:0] _col_7_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1765:86
  wire [15:0] _col_7_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1764:86
  wire [15:0] _col_7_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1763:88
  wire [15:0] _col_7_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1762:88
  wire [15:0] _col_7_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1761:67
  wire [15:0] _col_7_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1760:88
  wire [15:0] _col_7_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1759:88
  wire [15:0] _col_7_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1758:67
  wire [15:0] _col_7_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1757:88
  wire [15:0] _col_7_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1756:88
  wire [15:0] _col_7_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1755:67
  wire [15:0] _col_7_n_c8192_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1754:76
  wire [15:0] _col_7_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1753:79
  wire [15:0] _col_7_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1752:79
  wire [15:0] _col_7_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1751:79
  wire [15:0] _col_7_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1750:79
  wire [15:0] _col_7_n_c4_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1749:73
  wire [15:0] _col_7_n_c4_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1748:73
  wire [15:0] _col_7_n_c4_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1747:73
  wire [15:0] _col_6_n_val_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1745:90
  wire [15:0] _col_6_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1744:108
  wire [15:0] _col_6_n_val_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1742:90
  wire [15:0] _col_6_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1741:108
  wire [15:0] _col_6_n_val_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1739:90
  wire [15:0] _col_6_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1738:108
  wire [15:0] _col_6_n_val_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1736:90
  wire [15:0] _col_6_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1735:108
  wire [15:0] _col_6_n_val_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1733:90
  wire [15:0] _col_6_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1732:108
  wire [15:0] _col_6_n_val_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1730:90
  wire [15:0] _col_6_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1729:108
  wire [15:0] _col_6_n_val_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1727:90
  wire [15:0] _col_6_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1726:108
  wire [15:0] _col_6_n_val_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1724:90
  wire [15:0] _col_6_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1723:108
  wire [15:0] _col_6_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1722:103
  wire [15:0] _col_6_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1722:103
  wire [15:0] _col_6_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1721:86
  wire [15:0] _col_6_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1720:103
  wire [15:0] _col_6_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1719:103
  wire [15:0] _col_6_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1718:103
  wire [15:0] _col_6_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1717:103
  wire [15:0] _col_6_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1717:103
  wire [15:0] _col_6_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1716:86
  wire [15:0] _col_6_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1715:103
  wire [15:0] _col_6_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1714:103
  wire [15:0] _col_6_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1713:103
  wire [15:0] _col_6_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1712:103
  wire [15:0] _col_6_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1712:103
  wire [15:0] _col_6_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1711:103
  wire [15:0] _col_6_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1710:103
  wire [15:0] _col_6_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1710:103
  wire [15:0] _col_6_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1709:103
  wire [15:0] _col_6_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1708:103
  wire [15:0] _col_6_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1708:103
  wire [15:0] _col_6_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1707:103
  wire [15:0] _col_6_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1706:103
  wire [15:0] _col_6_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1706:103
  wire [15:0] _col_6_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1705:103
  wire [15:0] _col_6_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1704:103
  wire [15:0] _col_6_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1704:103
  wire [15:0] _col_6_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1703:103
  wire [15:0] _col_6_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1702:103
  wire [15:0] _col_6_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1702:103
  wire [15:0] _col_6_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1701:103
  wire [15:0] _col_6_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1700:103
  wire [15:0] _col_6_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1700:103
  wire [15:0] _col_6_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1699:103
  wire [15:0] _col_6_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1698:103
  wire [15:0] _col_6_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1698:103
  wire [15:0] _col_6_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1697:103
  wire [15:0] _col_6_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1696:103
  wire [15:0] _col_6_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1696:103
  wire [15:0] _col_6_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1695:86
  wire [15:0] _col_6_n_v3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1694:103
  wire [15:0] _col_6_n_u3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1693:103
  wire [15:0] _col_6_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1692:103
  wire [15:0] _col_6_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1692:103
  wire [15:0] _col_6_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1691:86
  wire [15:0] _col_6_n_v2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1690:103
  wire [15:0] _col_6_n_u2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1689:103
  wire [15:0] _col_6_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1688:103
  wire [15:0] _col_6_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1688:103
  wire [15:0] _col_6_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1687:103
  wire [15:0] _col_6_n_v1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1686:103
  wire [15:0] _col_6_n_u1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1685:103
  wire [15:0] _col_6_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1684:103
  wire [15:0] _col_6_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1684:103
  wire [15:0] _col_6_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1683:103
  wire [15:0] _col_6_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1682:103
  wire [15:0] _col_6_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1682:103
  wire [15:0] _col_6_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1681:103
  wire [15:0] _col_6_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1680:103
  wire [15:0] _col_6_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1680:103
  wire [15:0] _col_6_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1679:86
  wire [15:0] _col_6_n_v7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1678:103
  wire [15:0] _col_6_n_u7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1677:103
  wire [15:0] _col_6_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1676:103
  wire [15:0] _col_6_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1676:103
  wire [15:0] _col_6_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1675:86
  wire [15:0] _col_6_n_v6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1674:103
  wire [15:0] _col_6_n_u6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1673:103
  wire [15:0] _col_6_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1672:103
  wire [15:0] _col_6_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1672:103
  wire [15:0] _col_6_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1671:103
  wire [15:0] _col_6_n_v8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1670:103
  wire [15:0] _col_6_n_u8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1669:103
  wire [15:0] _col_6_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1668:103
  wire [15:0] _col_6_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1668:103
  wire [15:0] _col_6_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1667:86
  wire [15:0] _col_6_n_v5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1666:103
  wire [15:0] _col_6_n_u5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1665:103
  wire [15:0] _col_6_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1664:103
  wire [15:0] _col_6_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1664:103
  wire [15:0] _col_6_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1663:86
  wire [15:0] _col_6_n_v4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1662:103
  wire [15:0] _col_6_n_u4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1661:103
  wire [15:0] _col_6_d_x8_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1660:103
  wire [15:0] _col_6_d_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1660:103
  wire [15:0] _col_6_n_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1659:103
  wire [15:0] _col_6_n_v8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1658:103
  wire [15:0] _col_6_n_u8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1657:103
  wire [15:0] _col_6_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1656:103
  wire [15:0] _col_6_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1656:103
  wire [15:0] _col_6_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1655:103
  wire [15:0] _col_6_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1655:103
  wire [15:0] _col_6_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1654:103
  wire [15:0] _col_6_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1654:103
  wire [15:0] _col_6_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1653:103
  wire [15:0] _col_6_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1653:103
  wire [15:0] _col_6_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1652:103
  wire [15:0] _col_6_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1652:103
  wire [15:0] _col_6_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1651:103
  wire [15:0] _col_6_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1651:103
  wire [15:0] _col_6_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1650:103
  wire [15:0] _col_6_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1650:103
  wire [15:0] _col_6_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1649:103
  wire [15:0] _col_6_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1649:103
  wire [15:0] _col_6_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1648:103
  wire [15:0] _col_6_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1647:86
  wire [15:0] _col_6_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1646:86
  wire [15:0] _col_6_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1645:88
  wire [15:0] _col_6_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1644:88
  wire [15:0] _col_6_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1643:67
  wire [15:0] _col_6_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1642:88
  wire [15:0] _col_6_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1641:88
  wire [15:0] _col_6_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1640:67
  wire [15:0] _col_6_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1639:88
  wire [15:0] _col_6_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1638:88
  wire [15:0] _col_6_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1637:67
  wire [15:0] _col_6_n_c8192_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1636:76
  wire [15:0] _col_6_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1635:79
  wire [15:0] _col_6_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1634:79
  wire [15:0] _col_6_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1633:79
  wire [15:0] _col_6_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1632:79
  wire [15:0] _col_6_n_c4_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1631:73
  wire [15:0] _col_6_n_c4_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1630:73
  wire [15:0] _col_6_n_c4_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1629:73
  wire [15:0] _col_5_n_val_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1627:90
  wire [15:0] _col_5_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1626:108
  wire [15:0] _col_5_n_val_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1624:90
  wire [15:0] _col_5_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1623:108
  wire [15:0] _col_5_n_val_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1621:90
  wire [15:0] _col_5_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1620:108
  wire [15:0] _col_5_n_val_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1618:90
  wire [15:0] _col_5_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1617:108
  wire [15:0] _col_5_n_val_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1615:90
  wire [15:0] _col_5_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1614:108
  wire [15:0] _col_5_n_val_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1612:90
  wire [15:0] _col_5_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1611:108
  wire [15:0] _col_5_n_val_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1609:90
  wire [15:0] _col_5_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1608:108
  wire [15:0] _col_5_n_val_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1606:90
  wire [15:0] _col_5_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1605:108
  wire [15:0] _col_5_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1604:103
  wire [15:0] _col_5_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1604:103
  wire [15:0] _col_5_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1603:86
  wire [15:0] _col_5_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1602:103
  wire [15:0] _col_5_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1601:103
  wire [15:0] _col_5_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1600:103
  wire [15:0] _col_5_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1599:103
  wire [15:0] _col_5_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1599:103
  wire [15:0] _col_5_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1598:86
  wire [15:0] _col_5_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1597:103
  wire [15:0] _col_5_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1596:103
  wire [15:0] _col_5_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1595:103
  wire [15:0] _col_5_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1594:103
  wire [15:0] _col_5_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1594:103
  wire [15:0] _col_5_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1593:103
  wire [15:0] _col_5_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1592:103
  wire [15:0] _col_5_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1592:103
  wire [15:0] _col_5_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1591:103
  wire [15:0] _col_5_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1590:103
  wire [15:0] _col_5_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1590:103
  wire [15:0] _col_5_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1589:103
  wire [15:0] _col_5_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1588:103
  wire [15:0] _col_5_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1588:103
  wire [15:0] _col_5_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1587:103
  wire [15:0] _col_5_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1586:103
  wire [15:0] _col_5_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1586:103
  wire [15:0] _col_5_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1585:103
  wire [15:0] _col_5_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1584:103
  wire [15:0] _col_5_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1584:103
  wire [15:0] _col_5_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1583:103
  wire [15:0] _col_5_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1582:103
  wire [15:0] _col_5_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1582:103
  wire [15:0] _col_5_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1581:103
  wire [15:0] _col_5_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1580:103
  wire [15:0] _col_5_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1580:103
  wire [15:0] _col_5_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1579:103
  wire [15:0] _col_5_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1578:103
  wire [15:0] _col_5_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1578:103
  wire [15:0] _col_5_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1577:86
  wire [15:0] _col_5_n_v3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1576:103
  wire [15:0] _col_5_n_u3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1575:103
  wire [15:0] _col_5_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1574:103
  wire [15:0] _col_5_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1574:103
  wire [15:0] _col_5_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1573:86
  wire [15:0] _col_5_n_v2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1572:103
  wire [15:0] _col_5_n_u2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1571:103
  wire [15:0] _col_5_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1570:103
  wire [15:0] _col_5_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1570:103
  wire [15:0] _col_5_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1569:103
  wire [15:0] _col_5_n_v1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1568:103
  wire [15:0] _col_5_n_u1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1567:103
  wire [15:0] _col_5_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1566:103
  wire [15:0] _col_5_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1566:103
  wire [15:0] _col_5_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1565:103
  wire [15:0] _col_5_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1564:103
  wire [15:0] _col_5_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1564:103
  wire [15:0] _col_5_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1563:103
  wire [15:0] _col_5_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1562:103
  wire [15:0] _col_5_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1562:103
  wire [15:0] _col_5_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1561:86
  wire [15:0] _col_5_n_v7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1560:103
  wire [15:0] _col_5_n_u7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1559:103
  wire [15:0] _col_5_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1558:103
  wire [15:0] _col_5_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1558:103
  wire [15:0] _col_5_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1557:86
  wire [15:0] _col_5_n_v6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1556:103
  wire [15:0] _col_5_n_u6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1555:103
  wire [15:0] _col_5_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1554:103
  wire [15:0] _col_5_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1554:103
  wire [15:0] _col_5_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1553:103
  wire [15:0] _col_5_n_v8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1552:103
  wire [15:0] _col_5_n_u8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1551:103
  wire [15:0] _col_5_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1550:103
  wire [15:0] _col_5_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1550:103
  wire [15:0] _col_5_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1549:86
  wire [15:0] _col_5_n_v5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1548:103
  wire [15:0] _col_5_n_u5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1547:103
  wire [15:0] _col_5_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1546:103
  wire [15:0] _col_5_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1546:103
  wire [15:0] _col_5_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1545:86
  wire [15:0] _col_5_n_v4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1544:103
  wire [15:0] _col_5_n_u4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1543:103
  wire [15:0] _col_5_d_x8_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1542:103
  wire [15:0] _col_5_d_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1542:103
  wire [15:0] _col_5_n_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1541:103
  wire [15:0] _col_5_n_v8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1540:103
  wire [15:0] _col_5_n_u8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1539:103
  wire [15:0] _col_5_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1538:103
  wire [15:0] _col_5_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1538:103
  wire [15:0] _col_5_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1537:103
  wire [15:0] _col_5_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1537:103
  wire [15:0] _col_5_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1536:103
  wire [15:0] _col_5_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1536:103
  wire [15:0] _col_5_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1535:103
  wire [15:0] _col_5_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1535:103
  wire [15:0] _col_5_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1534:103
  wire [15:0] _col_5_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1534:103
  wire [15:0] _col_5_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1533:103
  wire [15:0] _col_5_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1533:103
  wire [15:0] _col_5_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1532:103
  wire [15:0] _col_5_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1532:103
  wire [15:0] _col_5_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1531:103
  wire [15:0] _col_5_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1531:103
  wire [15:0] _col_5_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1530:103
  wire [15:0] _col_5_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1529:86
  wire [15:0] _col_5_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1528:86
  wire [15:0] _col_5_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1527:88
  wire [15:0] _col_5_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1526:88
  wire [15:0] _col_5_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1525:67
  wire [15:0] _col_5_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1524:88
  wire [15:0] _col_5_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1523:88
  wire [15:0] _col_5_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1522:67
  wire [15:0] _col_5_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1521:88
  wire [15:0] _col_5_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1520:88
  wire [15:0] _col_5_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1519:67
  wire [15:0] _col_5_n_c8192_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1518:76
  wire [15:0] _col_5_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1517:79
  wire [15:0] _col_5_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1516:79
  wire [15:0] _col_5_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1515:79
  wire [15:0] _col_5_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1514:79
  wire [15:0] _col_5_n_c4_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1513:73
  wire [15:0] _col_5_n_c4_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1512:73
  wire [15:0] _col_5_n_c4_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1511:73
  wire [15:0] _col_4_n_val_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1509:90
  wire [15:0] _col_4_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1508:108
  wire [15:0] _col_4_n_val_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1506:90
  wire [15:0] _col_4_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1505:108
  wire [15:0] _col_4_n_val_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1503:90
  wire [15:0] _col_4_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1502:108
  wire [15:0] _col_4_n_val_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1500:90
  wire [15:0] _col_4_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1499:108
  wire [15:0] _col_4_n_val_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1497:90
  wire [15:0] _col_4_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1496:108
  wire [15:0] _col_4_n_val_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1494:90
  wire [15:0] _col_4_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1493:108
  wire [15:0] _col_4_n_val_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1491:90
  wire [15:0] _col_4_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1490:108
  wire [15:0] _col_4_n_val_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1488:90
  wire [15:0] _col_4_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1487:108
  wire [15:0] _col_4_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1486:103
  wire [15:0] _col_4_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1486:103
  wire [15:0] _col_4_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1485:86
  wire [15:0] _col_4_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1484:103
  wire [15:0] _col_4_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1483:103
  wire [15:0] _col_4_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1482:103
  wire [15:0] _col_4_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1481:103
  wire [15:0] _col_4_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1481:103
  wire [15:0] _col_4_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1480:86
  wire [15:0] _col_4_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1479:103
  wire [15:0] _col_4_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1478:103
  wire [15:0] _col_4_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1477:103
  wire [15:0] _col_4_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1476:103
  wire [15:0] _col_4_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1476:103
  wire [15:0] _col_4_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1475:103
  wire [15:0] _col_4_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1474:103
  wire [15:0] _col_4_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1474:103
  wire [15:0] _col_4_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1473:103
  wire [15:0] _col_4_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1472:103
  wire [15:0] _col_4_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1472:103
  wire [15:0] _col_4_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1471:103
  wire [15:0] _col_4_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1470:103
  wire [15:0] _col_4_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1470:103
  wire [15:0] _col_4_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1469:103
  wire [15:0] _col_4_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1468:103
  wire [15:0] _col_4_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1468:103
  wire [15:0] _col_4_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1467:103
  wire [15:0] _col_4_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1466:103
  wire [15:0] _col_4_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1466:103
  wire [15:0] _col_4_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1465:103
  wire [15:0] _col_4_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1464:103
  wire [15:0] _col_4_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1464:103
  wire [15:0] _col_4_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1463:103
  wire [15:0] _col_4_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1462:103
  wire [15:0] _col_4_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1462:103
  wire [15:0] _col_4_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1461:103
  wire [15:0] _col_4_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1460:103
  wire [15:0] _col_4_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1460:103
  wire [15:0] _col_4_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1459:86
  wire [15:0] _col_4_n_v3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1458:103
  wire [15:0] _col_4_n_u3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1457:103
  wire [15:0] _col_4_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1456:103
  wire [15:0] _col_4_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1456:103
  wire [15:0] _col_4_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1455:86
  wire [15:0] _col_4_n_v2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1454:103
  wire [15:0] _col_4_n_u2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1453:103
  wire [15:0] _col_4_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1452:103
  wire [15:0] _col_4_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1452:103
  wire [15:0] _col_4_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1451:103
  wire [15:0] _col_4_n_v1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1450:103
  wire [15:0] _col_4_n_u1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1449:103
  wire [15:0] _col_4_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1448:103
  wire [15:0] _col_4_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1448:103
  wire [15:0] _col_4_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1447:103
  wire [15:0] _col_4_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1446:103
  wire [15:0] _col_4_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1446:103
  wire [15:0] _col_4_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1445:103
  wire [15:0] _col_4_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1444:103
  wire [15:0] _col_4_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1444:103
  wire [15:0] _col_4_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1443:86
  wire [15:0] _col_4_n_v7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1442:103
  wire [15:0] _col_4_n_u7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1441:103
  wire [15:0] _col_4_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1440:103
  wire [15:0] _col_4_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1440:103
  wire [15:0] _col_4_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1439:86
  wire [15:0] _col_4_n_v6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1438:103
  wire [15:0] _col_4_n_u6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1437:103
  wire [15:0] _col_4_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1436:103
  wire [15:0] _col_4_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1436:103
  wire [15:0] _col_4_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1435:103
  wire [15:0] _col_4_n_v8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1434:103
  wire [15:0] _col_4_n_u8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1433:103
  wire [15:0] _col_4_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1432:103
  wire [15:0] _col_4_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1432:103
  wire [15:0] _col_4_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1431:86
  wire [15:0] _col_4_n_v5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1430:103
  wire [15:0] _col_4_n_u5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1429:103
  wire [15:0] _col_4_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1428:103
  wire [15:0] _col_4_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1428:103
  wire [15:0] _col_4_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1427:86
  wire [15:0] _col_4_n_v4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1426:103
  wire [15:0] _col_4_n_u4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1425:103
  wire [15:0] _col_4_d_x8_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1424:103
  wire [15:0] _col_4_d_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1424:103
  wire [15:0] _col_4_n_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1423:103
  wire [15:0] _col_4_n_v8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1422:103
  wire [15:0] _col_4_n_u8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1421:103
  wire [15:0] _col_4_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1420:103
  wire [15:0] _col_4_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1420:103
  wire [15:0] _col_4_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1419:103
  wire [15:0] _col_4_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1419:103
  wire [15:0] _col_4_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1418:103
  wire [15:0] _col_4_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1418:103
  wire [15:0] _col_4_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1417:103
  wire [15:0] _col_4_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1417:103
  wire [15:0] _col_4_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1416:103
  wire [15:0] _col_4_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1416:103
  wire [15:0] _col_4_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1415:103
  wire [15:0] _col_4_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1415:103
  wire [15:0] _col_4_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1414:103
  wire [15:0] _col_4_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1414:103
  wire [15:0] _col_4_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1413:103
  wire [15:0] _col_4_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1413:103
  wire [15:0] _col_4_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1412:103
  wire [15:0] _col_4_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1411:86
  wire [15:0] _col_4_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1410:86
  wire [15:0] _col_4_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1409:88
  wire [15:0] _col_4_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1408:88
  wire [15:0] _col_4_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1407:67
  wire [15:0] _col_4_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1406:88
  wire [15:0] _col_4_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1405:88
  wire [15:0] _col_4_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1404:67
  wire [15:0] _col_4_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1403:88
  wire [15:0] _col_4_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1402:88
  wire [15:0] _col_4_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1401:67
  wire [15:0] _col_4_n_c8192_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1400:76
  wire [15:0] _col_4_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1399:79
  wire [15:0] _col_4_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1398:79
  wire [15:0] _col_4_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1397:79
  wire [15:0] _col_4_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1396:79
  wire [15:0] _col_4_n_c4_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1395:73
  wire [15:0] _col_4_n_c4_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1394:73
  wire [15:0] _col_4_n_c4_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1393:73
  wire [15:0] _col_3_n_val_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1391:90
  wire [15:0] _col_3_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1390:108
  wire [15:0] _col_3_n_val_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1388:90
  wire [15:0] _col_3_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1387:108
  wire [15:0] _col_3_n_val_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1385:90
  wire [15:0] _col_3_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1384:108
  wire [15:0] _col_3_n_val_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1382:90
  wire [15:0] _col_3_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1381:108
  wire [15:0] _col_3_n_val_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1379:90
  wire [15:0] _col_3_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1378:108
  wire [15:0] _col_3_n_val_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1376:90
  wire [15:0] _col_3_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1375:108
  wire [15:0] _col_3_n_val_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1373:90
  wire [15:0] _col_3_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1372:108
  wire [15:0] _col_3_n_val_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1370:90
  wire [15:0] _col_3_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1369:108
  wire [15:0] _col_3_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1368:103
  wire [15:0] _col_3_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1368:103
  wire [15:0] _col_3_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1367:86
  wire [15:0] _col_3_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1366:103
  wire [15:0] _col_3_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1365:103
  wire [15:0] _col_3_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1364:103
  wire [15:0] _col_3_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1363:103
  wire [15:0] _col_3_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1363:103
  wire [15:0] _col_3_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1362:86
  wire [15:0] _col_3_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1361:103
  wire [15:0] _col_3_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1360:103
  wire [15:0] _col_3_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1359:103
  wire [15:0] _col_3_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1358:103
  wire [15:0] _col_3_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1358:103
  wire [15:0] _col_3_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1357:103
  wire [15:0] _col_3_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1356:103
  wire [15:0] _col_3_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1356:103
  wire [15:0] _col_3_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1355:103
  wire [15:0] _col_3_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1354:103
  wire [15:0] _col_3_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1354:103
  wire [15:0] _col_3_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1353:103
  wire [15:0] _col_3_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1352:103
  wire [15:0] _col_3_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1352:103
  wire [15:0] _col_3_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1351:103
  wire [15:0] _col_3_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1350:103
  wire [15:0] _col_3_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1350:103
  wire [15:0] _col_3_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1349:103
  wire [15:0] _col_3_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1348:103
  wire [15:0] _col_3_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1348:103
  wire [15:0] _col_3_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1347:103
  wire [15:0] _col_3_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1346:103
  wire [15:0] _col_3_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1346:103
  wire [15:0] _col_3_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1345:103
  wire [15:0] _col_3_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1344:103
  wire [15:0] _col_3_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1344:103
  wire [15:0] _col_3_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1343:103
  wire [15:0] _col_3_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1342:103
  wire [15:0] _col_3_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1342:103
  wire [15:0] _col_3_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1341:86
  wire [15:0] _col_3_n_v3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1340:103
  wire [15:0] _col_3_n_u3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1339:103
  wire [15:0] _col_3_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1338:103
  wire [15:0] _col_3_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1338:103
  wire [15:0] _col_3_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1337:86
  wire [15:0] _col_3_n_v2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1336:103
  wire [15:0] _col_3_n_u2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1335:103
  wire [15:0] _col_3_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1334:103
  wire [15:0] _col_3_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1334:103
  wire [15:0] _col_3_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1333:103
  wire [15:0] _col_3_n_v1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1332:103
  wire [15:0] _col_3_n_u1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1331:103
  wire [15:0] _col_3_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1330:103
  wire [15:0] _col_3_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1330:103
  wire [15:0] _col_3_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1329:103
  wire [15:0] _col_3_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1328:103
  wire [15:0] _col_3_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1328:103
  wire [15:0] _col_3_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1327:103
  wire [15:0] _col_3_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1326:103
  wire [15:0] _col_3_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1326:103
  wire [15:0] _col_3_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1325:86
  wire [15:0] _col_3_n_v7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1324:103
  wire [15:0] _col_3_n_u7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1323:103
  wire [15:0] _col_3_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1322:103
  wire [15:0] _col_3_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1322:103
  wire [15:0] _col_3_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1321:86
  wire [15:0] _col_3_n_v6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1320:103
  wire [15:0] _col_3_n_u6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1319:103
  wire [15:0] _col_3_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1318:103
  wire [15:0] _col_3_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1318:103
  wire [15:0] _col_3_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1317:103
  wire [15:0] _col_3_n_v8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1316:103
  wire [15:0] _col_3_n_u8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1315:103
  wire [15:0] _col_3_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1314:103
  wire [15:0] _col_3_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1314:103
  wire [15:0] _col_3_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1313:86
  wire [15:0] _col_3_n_v5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1312:103
  wire [15:0] _col_3_n_u5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1311:103
  wire [15:0] _col_3_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1310:103
  wire [15:0] _col_3_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1310:103
  wire [15:0] _col_3_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1309:86
  wire [15:0] _col_3_n_v4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1308:103
  wire [15:0] _col_3_n_u4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1307:103
  wire [15:0] _col_3_d_x8_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1306:103
  wire [15:0] _col_3_d_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1306:103
  wire [15:0] _col_3_n_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1305:103
  wire [15:0] _col_3_n_v8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1304:103
  wire [15:0] _col_3_n_u8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1303:103
  wire [15:0] _col_3_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1302:103
  wire [15:0] _col_3_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1302:103
  wire [15:0] _col_3_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1301:103
  wire [15:0] _col_3_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1301:103
  wire [15:0] _col_3_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1300:103
  wire [15:0] _col_3_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1300:103
  wire [15:0] _col_3_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1299:103
  wire [15:0] _col_3_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1299:103
  wire [15:0] _col_3_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1298:103
  wire [15:0] _col_3_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1298:103
  wire [15:0] _col_3_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1297:103
  wire [15:0] _col_3_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1297:103
  wire [15:0] _col_3_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1296:103
  wire [15:0] _col_3_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1296:103
  wire [15:0] _col_3_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1295:103
  wire [15:0] _col_3_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1295:103
  wire [15:0] _col_3_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1294:103
  wire [15:0] _col_3_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1293:86
  wire [15:0] _col_3_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1292:86
  wire [15:0] _col_3_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1291:88
  wire [15:0] _col_3_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1290:88
  wire [15:0] _col_3_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1289:67
  wire [15:0] _col_3_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1288:88
  wire [15:0] _col_3_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1287:88
  wire [15:0] _col_3_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1286:67
  wire [15:0] _col_3_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1285:88
  wire [15:0] _col_3_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1284:88
  wire [15:0] _col_3_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1283:67
  wire [15:0] _col_3_n_c8192_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1282:76
  wire [15:0] _col_3_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1281:79
  wire [15:0] _col_3_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1280:79
  wire [15:0] _col_3_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1279:79
  wire [15:0] _col_3_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1278:79
  wire [15:0] _col_3_n_c4_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1277:73
  wire [15:0] _col_3_n_c4_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1276:73
  wire [15:0] _col_3_n_c4_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1275:73
  wire [15:0] _col_2_n_val_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1273:90
  wire [15:0] _col_2_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1272:108
  wire [15:0] _col_2_n_val_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1270:90
  wire [15:0] _col_2_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1269:108
  wire [15:0] _col_2_n_val_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1267:90
  wire [15:0] _col_2_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1266:108
  wire [15:0] _col_2_n_val_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1264:90
  wire [15:0] _col_2_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1263:108
  wire [15:0] _col_2_n_val_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1261:90
  wire [15:0] _col_2_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1260:108
  wire [15:0] _col_2_n_val_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1258:90
  wire [15:0] _col_2_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1257:108
  wire [15:0] _col_2_n_val_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1255:90
  wire [15:0] _col_2_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1254:108
  wire [15:0] _col_2_n_val_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1252:90
  wire [15:0] _col_2_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1251:108
  wire [15:0] _col_2_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1250:103
  wire [15:0] _col_2_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1250:103
  wire [15:0] _col_2_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1249:86
  wire [15:0] _col_2_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1248:103
  wire [15:0] _col_2_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1247:103
  wire [15:0] _col_2_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1246:103
  wire [15:0] _col_2_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1245:103
  wire [15:0] _col_2_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1245:103
  wire [15:0] _col_2_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1244:86
  wire [15:0] _col_2_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1243:103
  wire [15:0] _col_2_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1242:103
  wire [15:0] _col_2_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1241:103
  wire [15:0] _col_2_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1240:103
  wire [15:0] _col_2_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1240:103
  wire [15:0] _col_2_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1239:103
  wire [15:0] _col_2_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1238:103
  wire [15:0] _col_2_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1238:103
  wire [15:0] _col_2_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1237:103
  wire [15:0] _col_2_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1236:103
  wire [15:0] _col_2_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1236:103
  wire [15:0] _col_2_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1235:103
  wire [15:0] _col_2_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1234:103
  wire [15:0] _col_2_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1234:103
  wire [15:0] _col_2_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1233:103
  wire [15:0] _col_2_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1232:103
  wire [15:0] _col_2_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1232:103
  wire [15:0] _col_2_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1231:103
  wire [15:0] _col_2_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1230:103
  wire [15:0] _col_2_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1230:103
  wire [15:0] _col_2_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1229:103
  wire [15:0] _col_2_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1228:103
  wire [15:0] _col_2_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1228:103
  wire [15:0] _col_2_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1227:103
  wire [15:0] _col_2_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1226:103
  wire [15:0] _col_2_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1226:103
  wire [15:0] _col_2_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1225:103
  wire [15:0] _col_2_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1224:103
  wire [15:0] _col_2_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1224:103
  wire [15:0] _col_2_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1223:86
  wire [15:0] _col_2_n_v3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1222:103
  wire [15:0] _col_2_n_u3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1221:103
  wire [15:0] _col_2_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1220:103
  wire [15:0] _col_2_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1220:103
  wire [15:0] _col_2_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1219:86
  wire [15:0] _col_2_n_v2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1218:103
  wire [15:0] _col_2_n_u2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1217:103
  wire [15:0] _col_2_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1216:103
  wire [15:0] _col_2_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1216:103
  wire [15:0] _col_2_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1215:103
  wire [15:0] _col_2_n_v1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1214:103
  wire [15:0] _col_2_n_u1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1213:103
  wire [15:0] _col_2_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1212:103
  wire [15:0] _col_2_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1212:103
  wire [15:0] _col_2_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1211:103
  wire [15:0] _col_2_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1210:103
  wire [15:0] _col_2_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1210:103
  wire [15:0] _col_2_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1209:103
  wire [15:0] _col_2_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1208:103
  wire [15:0] _col_2_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1208:103
  wire [15:0] _col_2_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1207:86
  wire [15:0] _col_2_n_v7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1206:103
  wire [15:0] _col_2_n_u7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1205:103
  wire [15:0] _col_2_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1204:103
  wire [15:0] _col_2_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1204:103
  wire [15:0] _col_2_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1203:86
  wire [15:0] _col_2_n_v6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1202:103
  wire [15:0] _col_2_n_u6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1201:103
  wire [15:0] _col_2_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1200:103
  wire [15:0] _col_2_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1200:103
  wire [15:0] _col_2_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1199:103
  wire [15:0] _col_2_n_v8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1198:103
  wire [15:0] _col_2_n_u8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1197:103
  wire [15:0] _col_2_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1196:103
  wire [15:0] _col_2_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1196:103
  wire [15:0] _col_2_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1195:86
  wire [15:0] _col_2_n_v5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1194:103
  wire [15:0] _col_2_n_u5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1193:103
  wire [15:0] _col_2_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1192:103
  wire [15:0] _col_2_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1192:103
  wire [15:0] _col_2_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1191:86
  wire [15:0] _col_2_n_v4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1190:103
  wire [15:0] _col_2_n_u4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1189:103
  wire [15:0] _col_2_d_x8_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1188:103
  wire [15:0] _col_2_d_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1188:103
  wire [15:0] _col_2_n_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1187:103
  wire [15:0] _col_2_n_v8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1186:103
  wire [15:0] _col_2_n_u8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1185:103
  wire [15:0] _col_2_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1184:103
  wire [15:0] _col_2_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1184:103
  wire [15:0] _col_2_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1183:103
  wire [15:0] _col_2_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1183:103
  wire [15:0] _col_2_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1182:103
  wire [15:0] _col_2_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1182:103
  wire [15:0] _col_2_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1181:103
  wire [15:0] _col_2_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1181:103
  wire [15:0] _col_2_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1180:103
  wire [15:0] _col_2_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1180:103
  wire [15:0] _col_2_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1179:103
  wire [15:0] _col_2_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1179:103
  wire [15:0] _col_2_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1178:103
  wire [15:0] _col_2_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1178:103
  wire [15:0] _col_2_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1177:103
  wire [15:0] _col_2_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1177:103
  wire [15:0] _col_2_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1176:103
  wire [15:0] _col_2_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1175:86
  wire [15:0] _col_2_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1174:86
  wire [15:0] _col_2_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1173:88
  wire [15:0] _col_2_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1172:88
  wire [15:0] _col_2_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1171:67
  wire [15:0] _col_2_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1170:88
  wire [15:0] _col_2_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1169:88
  wire [15:0] _col_2_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1168:67
  wire [15:0] _col_2_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1167:88
  wire [15:0] _col_2_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1166:88
  wire [15:0] _col_2_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1165:67
  wire [15:0] _col_2_n_c8192_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1164:76
  wire [15:0] _col_2_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1163:79
  wire [15:0] _col_2_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1162:79
  wire [15:0] _col_2_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1161:79
  wire [15:0] _col_2_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1160:79
  wire [15:0] _col_2_n_c4_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1159:73
  wire [15:0] _col_2_n_c4_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1158:73
  wire [15:0] _col_2_n_c4_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1157:73
  wire [15:0] _col_1_n_val_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1155:90
  wire [15:0] _col_1_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1154:108
  wire [15:0] _col_1_n_val_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1152:90
  wire [15:0] _col_1_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1151:108
  wire [15:0] _col_1_n_val_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1149:90
  wire [15:0] _col_1_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1148:108
  wire [15:0] _col_1_n_val_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1146:90
  wire [15:0] _col_1_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1145:108
  wire [15:0] _col_1_n_val_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1143:90
  wire [15:0] _col_1_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1142:108
  wire [15:0] _col_1_n_val_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1140:90
  wire [15:0] _col_1_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1139:108
  wire [15:0] _col_1_n_val_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1137:90
  wire [15:0] _col_1_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1136:108
  wire [15:0] _col_1_n_val_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1134:90
  wire [15:0] _col_1_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1133:108
  wire [15:0] _col_1_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1132:103
  wire [15:0] _col_1_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1132:103
  wire [15:0] _col_1_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1131:86
  wire [15:0] _col_1_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1130:103
  wire [15:0] _col_1_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1129:103
  wire [15:0] _col_1_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1128:103
  wire [15:0] _col_1_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1127:103
  wire [15:0] _col_1_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1127:103
  wire [15:0] _col_1_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1126:86
  wire [15:0] _col_1_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1125:103
  wire [15:0] _col_1_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1124:103
  wire [15:0] _col_1_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1123:103
  wire [15:0] _col_1_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1122:103
  wire [15:0] _col_1_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1122:103
  wire [15:0] _col_1_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1121:103
  wire [15:0] _col_1_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1120:103
  wire [15:0] _col_1_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1120:103
  wire [15:0] _col_1_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1119:103
  wire [15:0] _col_1_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1118:103
  wire [15:0] _col_1_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1118:103
  wire [15:0] _col_1_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1117:103
  wire [15:0] _col_1_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1116:103
  wire [15:0] _col_1_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1116:103
  wire [15:0] _col_1_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1115:103
  wire [15:0] _col_1_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1114:103
  wire [15:0] _col_1_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1114:103
  wire [15:0] _col_1_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1113:103
  wire [15:0] _col_1_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1112:103
  wire [15:0] _col_1_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1112:103
  wire [15:0] _col_1_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1111:103
  wire [15:0] _col_1_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1110:103
  wire [15:0] _col_1_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1110:103
  wire [15:0] _col_1_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1109:103
  wire [15:0] _col_1_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1108:103
  wire [15:0] _col_1_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1108:103
  wire [15:0] _col_1_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1107:103
  wire [15:0] _col_1_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1106:103
  wire [15:0] _col_1_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1106:103
  wire [15:0] _col_1_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1105:86
  wire [15:0] _col_1_n_v3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1104:103
  wire [15:0] _col_1_n_u3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1103:103
  wire [15:0] _col_1_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1102:103
  wire [15:0] _col_1_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1102:103
  wire [15:0] _col_1_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1101:86
  wire [15:0] _col_1_n_v2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1100:103
  wire [15:0] _col_1_n_u2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1099:103
  wire [15:0] _col_1_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1098:103
  wire [15:0] _col_1_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1098:103
  wire [15:0] _col_1_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1097:103
  wire [15:0] _col_1_n_v1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1096:103
  wire [15:0] _col_1_n_u1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1095:103
  wire [15:0] _col_1_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1094:103
  wire [15:0] _col_1_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1094:103
  wire [15:0] _col_1_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1093:103
  wire [15:0] _col_1_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1092:103
  wire [15:0] _col_1_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1092:103
  wire [15:0] _col_1_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1091:103
  wire [15:0] _col_1_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1090:103
  wire [15:0] _col_1_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1090:103
  wire [15:0] _col_1_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1089:86
  wire [15:0] _col_1_n_v7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1088:103
  wire [15:0] _col_1_n_u7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1087:103
  wire [15:0] _col_1_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1086:103
  wire [15:0] _col_1_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1086:103
  wire [15:0] _col_1_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1085:86
  wire [15:0] _col_1_n_v6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1084:103
  wire [15:0] _col_1_n_u6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1083:103
  wire [15:0] _col_1_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1082:103
  wire [15:0] _col_1_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1082:103
  wire [15:0] _col_1_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1081:103
  wire [15:0] _col_1_n_v8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1080:103
  wire [15:0] _col_1_n_u8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1079:103
  wire [15:0] _col_1_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1078:103
  wire [15:0] _col_1_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1078:103
  wire [15:0] _col_1_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1077:86
  wire [15:0] _col_1_n_v5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1076:103
  wire [15:0] _col_1_n_u5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1075:103
  wire [15:0] _col_1_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1074:103
  wire [15:0] _col_1_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1074:103
  wire [15:0] _col_1_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1073:86
  wire [15:0] _col_1_n_v4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1072:103
  wire [15:0] _col_1_n_u4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1071:103
  wire [15:0] _col_1_d_x8_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1070:103
  wire [15:0] _col_1_d_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1070:103
  wire [15:0] _col_1_n_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1069:103
  wire [15:0] _col_1_n_v8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1068:103
  wire [15:0] _col_1_n_u8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1067:103
  wire [15:0] _col_1_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1066:103
  wire [15:0] _col_1_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1066:103
  wire [15:0] _col_1_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1065:103
  wire [15:0] _col_1_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1065:103
  wire [15:0] _col_1_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1064:103
  wire [15:0] _col_1_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1064:103
  wire [15:0] _col_1_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1063:103
  wire [15:0] _col_1_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1063:103
  wire [15:0] _col_1_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1062:103
  wire [15:0] _col_1_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1062:103
  wire [15:0] _col_1_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1061:103
  wire [15:0] _col_1_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1061:103
  wire [15:0] _col_1_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1060:103
  wire [15:0] _col_1_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1060:103
  wire [15:0] _col_1_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1059:103
  wire [15:0] _col_1_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1059:103
  wire [15:0] _col_1_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1058:103
  wire [15:0] _col_1_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1057:86
  wire [15:0] _col_1_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1056:86
  wire [15:0] _col_1_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1055:88
  wire [15:0] _col_1_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1054:88
  wire [15:0] _col_1_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1053:67
  wire [15:0] _col_1_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1052:88
  wire [15:0] _col_1_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1051:88
  wire [15:0] _col_1_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1050:67
  wire [15:0] _col_1_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1049:88
  wire [15:0] _col_1_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1048:88
  wire [15:0] _col_1_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1047:67
  wire [15:0] _col_1_n_c8192_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1046:76
  wire [15:0] _col_1_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1045:79
  wire [15:0] _col_1_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1044:79
  wire [15:0] _col_1_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1043:79
  wire [15:0] _col_1_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1042:79
  wire [15:0] _col_1_n_c4_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1041:73
  wire [15:0] _col_1_n_c4_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1040:73
  wire [15:0] _col_1_n_c4_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1039:73
  wire [15:0] _col_0_n_val_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1037:90
  wire [15:0] _col_0_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1036:108
  wire [15:0] _col_0_n_val_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1034:90
  wire [15:0] _col_0_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1033:108
  wire [15:0] _col_0_n_val_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1031:90
  wire [15:0] _col_0_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1030:108
  wire [15:0] _col_0_n_val_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1028:90
  wire [15:0] _col_0_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1027:108
  wire [15:0] _col_0_n_val_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1025:90
  wire [15:0] _col_0_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1024:108
  wire [15:0] _col_0_n_val_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1022:90
  wire [15:0] _col_0_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1021:108
  wire [15:0] _col_0_n_val_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1019:90
  wire [15:0] _col_0_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1018:108
  wire [15:0] _col_0_n_val_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1016:90
  wire [15:0] _col_0_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1015:108
  wire [15:0] _col_0_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1014:103
  wire [15:0] _col_0_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1014:103
  wire [15:0] _col_0_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1013:86
  wire [15:0] _col_0_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1012:103
  wire [15:0] _col_0_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1011:103
  wire [15:0] _col_0_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1010:103
  wire [15:0] _col_0_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1009:103
  wire [15:0] _col_0_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1009:103
  wire [15:0] _col_0_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1008:86
  wire [15:0] _col_0_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1007:103
  wire [15:0] _col_0_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1006:103
  wire [15:0] _col_0_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1005:103
  wire [15:0] _col_0_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1004:103
  wire [15:0] _col_0_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1004:103
  wire [15:0] _col_0_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1003:103
  wire [15:0] _col_0_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1002:103
  wire [15:0] _col_0_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1002:103
  wire [15:0] _col_0_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1001:103
  wire [15:0] _col_0_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1000:103
  wire [15:0] _col_0_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1000:103
  wire [15:0] _col_0_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:999:103
  wire [15:0] _col_0_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:998:103
  wire [15:0] _col_0_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:998:103
  wire [15:0] _col_0_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:997:103
  wire [15:0] _col_0_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:996:103
  wire [15:0] _col_0_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:996:103
  wire [15:0] _col_0_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:995:103
  wire [15:0] _col_0_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:994:103
  wire [15:0] _col_0_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:994:103
  wire [15:0] _col_0_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:993:103
  wire [15:0] _col_0_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:992:103
  wire [15:0] _col_0_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:992:103
  wire [15:0] _col_0_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:991:103
  wire [15:0] _col_0_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:990:103
  wire [15:0] _col_0_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:990:103
  wire [15:0] _col_0_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:989:103
  wire [15:0] _col_0_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:988:103
  wire [15:0] _col_0_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:988:103
  wire [15:0] _col_0_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:987:86
  wire [15:0] _col_0_n_v3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:986:103
  wire [15:0] _col_0_n_u3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:985:103
  wire [15:0] _col_0_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:984:103
  wire [15:0] _col_0_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:984:103
  wire [15:0] _col_0_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:983:86
  wire [15:0] _col_0_n_v2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:982:103
  wire [15:0] _col_0_n_u2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:981:103
  wire [15:0] _col_0_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:980:103
  wire [15:0] _col_0_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:980:103
  wire [15:0] _col_0_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:979:103
  wire [15:0] _col_0_n_v1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:978:103
  wire [15:0] _col_0_n_u1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:977:103
  wire [15:0] _col_0_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:976:103
  wire [15:0] _col_0_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:976:103
  wire [15:0] _col_0_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:975:103
  wire [15:0] _col_0_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:974:103
  wire [15:0] _col_0_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:974:103
  wire [15:0] _col_0_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:973:103
  wire [15:0] _col_0_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:972:103
  wire [15:0] _col_0_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:972:103
  wire [15:0] _col_0_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:971:86
  wire [15:0] _col_0_n_v7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:970:103
  wire [15:0] _col_0_n_u7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:969:103
  wire [15:0] _col_0_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:968:103
  wire [15:0] _col_0_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:968:103
  wire [15:0] _col_0_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:967:86
  wire [15:0] _col_0_n_v6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:966:103
  wire [15:0] _col_0_n_u6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:965:103
  wire [15:0] _col_0_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:964:103
  wire [15:0] _col_0_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:964:103
  wire [15:0] _col_0_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:963:103
  wire [15:0] _col_0_n_v8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:962:103
  wire [15:0] _col_0_n_u8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:961:103
  wire [15:0] _col_0_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:960:103
  wire [15:0] _col_0_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:960:103
  wire [15:0] _col_0_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:959:86
  wire [15:0] _col_0_n_v5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:958:103
  wire [15:0] _col_0_n_u5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:957:103
  wire [15:0] _col_0_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:956:103
  wire [15:0] _col_0_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:956:103
  wire [15:0] _col_0_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:955:86
  wire [15:0] _col_0_n_v4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:954:103
  wire [15:0] _col_0_n_u4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:953:103
  wire [15:0] _col_0_d_x8_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:952:103
  wire [15:0] _col_0_d_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:952:103
  wire [15:0] _col_0_n_x8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:951:103
  wire [15:0] _col_0_n_v8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:950:103
  wire [15:0] _col_0_n_u8_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:949:103
  wire [15:0] _col_0_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:948:103
  wire [15:0] _col_0_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:948:103
  wire [15:0] _col_0_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:947:103
  wire [15:0] _col_0_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:947:103
  wire [15:0] _col_0_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:946:103
  wire [15:0] _col_0_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:946:103
  wire [15:0] _col_0_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:945:103
  wire [15:0] _col_0_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:945:103
  wire [15:0] _col_0_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:944:103
  wire [15:0] _col_0_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:944:103
  wire [15:0] _col_0_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:943:103
  wire [15:0] _col_0_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:943:103
  wire [15:0] _col_0_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:942:103
  wire [15:0] _col_0_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:942:103
  wire [15:0] _col_0_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:941:103
  wire [15:0] _col_0_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:941:103
  wire [15:0] _col_0_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:940:103
  wire [15:0] _col_0_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:939:86
  wire [15:0] _col_0_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:938:86
  wire [15:0] _col_0_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:937:88
  wire [15:0] _col_0_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:936:88
  wire [15:0] _col_0_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:935:67
  wire [15:0] _col_0_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:934:88
  wire [15:0] _col_0_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:933:88
  wire [15:0] _col_0_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:932:67
  wire [15:0] _col_0_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:931:88
  wire [15:0] _col_0_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:930:88
  wire [15:0] _col_0_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:929:67
  wire [15:0] _col_0_n_c8192_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:928:76
  wire [15:0] _col_0_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:927:79
  wire [15:0] _col_0_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:926:79
  wire [15:0] _col_0_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:925:79
  wire [15:0] _col_0_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:924:79
  wire [15:0] _col_0_n_c4_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:923:73
  wire [15:0] _col_0_n_c4_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:922:73
  wire [15:0] _col_0_n_c4_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:921:73
  wire [15:0] _row_7_n_shr_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:920:90
  wire [15:0] _row_7_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:919:108
  wire [15:0] _row_7_n_shr_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:918:90
  wire [15:0] _row_7_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:917:108
  wire [15:0] _row_7_n_shr_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:916:90
  wire [15:0] _row_7_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:915:108
  wire [15:0] _row_7_n_shr_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:914:90
  wire [15:0] _row_7_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:913:108
  wire [15:0] _row_7_n_shr_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:912:90
  wire [15:0] _row_7_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:911:108
  wire [15:0] _row_7_n_shr_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:910:90
  wire [15:0] _row_7_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:909:108
  wire [15:0] _row_7_n_shr_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:908:90
  wire [15:0] _row_7_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:907:108
  wire [15:0] _row_7_n_shr_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:906:90
  wire [15:0] _row_7_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:905:108
  wire [15:0] _row_7_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:904:103
  wire [15:0] _row_7_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:904:103
  wire [15:0] _row_7_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:903:86
  wire [15:0] _row_7_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:902:103
  wire [15:0] _row_7_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:901:103
  wire [15:0] _row_7_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:900:103
  wire [15:0] _row_7_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:899:103
  wire [15:0] _row_7_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:899:103
  wire [15:0] _row_7_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:898:86
  wire [15:0] _row_7_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:897:103
  wire [15:0] _row_7_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:896:103
  wire [15:0] _row_7_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:895:103
  wire [15:0] _row_7_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:894:103
  wire [15:0] _row_7_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:894:103
  wire [15:0] _row_7_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:893:103
  wire [15:0] _row_7_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:892:103
  wire [15:0] _row_7_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:892:103
  wire [15:0] _row_7_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:891:103
  wire [15:0] _row_7_d_x8_4_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:890:103
  wire [15:0] _row_7_d_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:890:103
  wire [15:0] _row_7_n_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:889:103
  wire [15:0] _row_7_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:888:103
  wire [15:0] _row_7_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:888:103
  wire [15:0] _row_7_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:887:103
  wire [15:0] _row_7_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:886:103
  wire [15:0] _row_7_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:886:103
  wire [15:0] _row_7_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:885:103
  wire [15:0] _row_7_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:884:103
  wire [15:0] _row_7_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:884:103
  wire [15:0] _row_7_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:883:103
  wire [15:0] _row_7_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:882:103
  wire [15:0] _row_7_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:882:103
  wire [15:0] _row_7_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:881:103
  wire [15:0] _row_7_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:880:103
  wire [15:0] _row_7_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:880:103
  wire [15:0] _row_7_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:879:103
  wire [15:0] _row_7_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:878:103
  wire [15:0] _row_7_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:878:103
  wire [15:0] _row_7_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:877:103
  wire [15:0] _row_7_n_t3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:876:103
  wire [15:0] _row_7_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:875:103
  wire [15:0] _row_7_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:875:103
  wire [15:0] _row_7_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:874:103
  wire [15:0] _row_7_n_t2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:873:103
  wire [15:0] _row_7_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:872:103
  wire [15:0] _row_7_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:872:103
  wire [15:0] _row_7_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:871:103
  wire [15:0] _row_7_n_t1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:870:103
  wire [15:0] _row_7_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:869:103
  wire [15:0] _row_7_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:869:103
  wire [15:0] _row_7_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:868:103
  wire [15:0] _row_7_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:867:103
  wire [15:0] _row_7_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:867:103
  wire [15:0] _row_7_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:866:103
  wire [15:0] _row_7_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:865:103
  wire [15:0] _row_7_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:865:103
  wire [15:0] _row_7_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:864:103
  wire [15:0] _row_7_n_t7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:863:103
  wire [15:0] _row_7_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:862:103
  wire [15:0] _row_7_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:862:103
  wire [15:0] _row_7_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:861:103
  wire [15:0] _row_7_n_t6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:860:103
  wire [15:0] _row_7_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:859:103
  wire [15:0] _row_7_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:859:103
  wire [15:0] _row_7_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:858:103
  wire [15:0] _row_7_n_t8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:857:103
  wire [15:0] _row_7_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:856:103
  wire [15:0] _row_7_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:856:103
  wire [15:0] _row_7_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:855:103
  wire [15:0] _row_7_n_t5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:854:103
  wire [15:0] _row_7_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:853:103
  wire [15:0] _row_7_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:853:103
  wire [15:0] _row_7_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:852:103
  wire [15:0] _row_7_n_t4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:851:103
  wire [15:0] _row_7_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:850:103
  wire [15:0] _row_7_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:850:103
  wire [15:0] _row_7_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:849:103
  wire [15:0] _row_7_n_t8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:848:103
  wire [15:0] _row_7_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:847:103
  wire [15:0] _row_7_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:847:103
  wire [15:0] _row_7_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:846:103
  wire [15:0] _row_7_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:846:103
  wire [15:0] _row_7_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:845:103
  wire [15:0] _row_7_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:845:103
  wire [15:0] _row_7_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:844:103
  wire [15:0] _row_7_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:844:103
  wire [15:0] _row_7_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:843:103
  wire [15:0] _row_7_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:843:103
  wire [15:0] _row_7_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:842:103
  wire [15:0] _row_7_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:842:103
  wire [15:0] _row_7_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:841:103
  wire [15:0] _row_7_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:841:103
  wire [15:0] _row_7_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:840:103
  wire [15:0] _row_7_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:840:103
  wire [15:0] _row_7_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:839:103
  wire [15:0] _row_7_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:838:86
  wire [15:0] _row_7_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:837:86
  wire [15:0] _row_7_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:836:88
  wire [15:0] _row_7_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:835:88
  wire [15:0] _row_7_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:834:67
  wire [15:0] _row_7_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:833:88
  wire [15:0] _row_7_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:832:88
  wire [15:0] _row_7_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:831:67
  wire [15:0] _row_7_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:830:88
  wire [15:0] _row_7_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:829:88
  wire [15:0] _row_7_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:828:67
  wire [15:0] _row_7_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:827:79
  wire [15:0] _row_7_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:826:79
  wire [15:0] _row_7_n_c128_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:825:79
  wire [15:0] _row_7_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:824:79
  wire [15:0] _row_7_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:823:79
  wire [15:0] _row_6_n_shr_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:822:90
  wire [15:0] _row_6_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:821:108
  wire [15:0] _row_6_n_shr_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:820:90
  wire [15:0] _row_6_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:819:108
  wire [15:0] _row_6_n_shr_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:818:90
  wire [15:0] _row_6_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:817:108
  wire [15:0] _row_6_n_shr_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:816:90
  wire [15:0] _row_6_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:815:108
  wire [15:0] _row_6_n_shr_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:814:90
  wire [15:0] _row_6_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:813:108
  wire [15:0] _row_6_n_shr_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:812:90
  wire [15:0] _row_6_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:811:108
  wire [15:0] _row_6_n_shr_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:810:90
  wire [15:0] _row_6_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:809:108
  wire [15:0] _row_6_n_shr_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:808:90
  wire [15:0] _row_6_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:807:108
  wire [15:0] _row_6_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:806:103
  wire [15:0] _row_6_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:806:103
  wire [15:0] _row_6_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:805:86
  wire [15:0] _row_6_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:804:103
  wire [15:0] _row_6_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:803:103
  wire [15:0] _row_6_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:802:103
  wire [15:0] _row_6_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:801:103
  wire [15:0] _row_6_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:801:103
  wire [15:0] _row_6_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:800:86
  wire [15:0] _row_6_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:799:103
  wire [15:0] _row_6_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:798:103
  wire [15:0] _row_6_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:797:103
  wire [15:0] _row_6_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:796:103
  wire [15:0] _row_6_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:796:103
  wire [15:0] _row_6_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:795:103
  wire [15:0] _row_6_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:794:103
  wire [15:0] _row_6_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:794:103
  wire [15:0] _row_6_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:793:103
  wire [15:0] _row_6_d_x8_4_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:792:103
  wire [15:0] _row_6_d_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:792:103
  wire [15:0] _row_6_n_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:791:103
  wire [15:0] _row_6_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:790:103
  wire [15:0] _row_6_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:790:103
  wire [15:0] _row_6_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:789:103
  wire [15:0] _row_6_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:788:103
  wire [15:0] _row_6_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:788:103
  wire [15:0] _row_6_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:787:103
  wire [15:0] _row_6_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:786:103
  wire [15:0] _row_6_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:786:103
  wire [15:0] _row_6_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:785:103
  wire [15:0] _row_6_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:784:103
  wire [15:0] _row_6_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:784:103
  wire [15:0] _row_6_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:783:103
  wire [15:0] _row_6_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:782:103
  wire [15:0] _row_6_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:782:103
  wire [15:0] _row_6_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:781:103
  wire [15:0] _row_6_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:780:103
  wire [15:0] _row_6_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:780:103
  wire [15:0] _row_6_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:779:103
  wire [15:0] _row_6_n_t3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:778:103
  wire [15:0] _row_6_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:777:103
  wire [15:0] _row_6_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:777:103
  wire [15:0] _row_6_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:776:103
  wire [15:0] _row_6_n_t2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:775:103
  wire [15:0] _row_6_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:774:103
  wire [15:0] _row_6_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:774:103
  wire [15:0] _row_6_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:773:103
  wire [15:0] _row_6_n_t1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:772:103
  wire [15:0] _row_6_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:771:103
  wire [15:0] _row_6_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:771:103
  wire [15:0] _row_6_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:770:103
  wire [15:0] _row_6_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:769:103
  wire [15:0] _row_6_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:769:103
  wire [15:0] _row_6_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:768:103
  wire [15:0] _row_6_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:767:103
  wire [15:0] _row_6_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:767:103
  wire [15:0] _row_6_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:766:103
  wire [15:0] _row_6_n_t7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:765:103
  wire [15:0] _row_6_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:764:103
  wire [15:0] _row_6_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:764:103
  wire [15:0] _row_6_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:763:103
  wire [15:0] _row_6_n_t6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:762:103
  wire [15:0] _row_6_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:761:103
  wire [15:0] _row_6_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:761:103
  wire [15:0] _row_6_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:760:103
  wire [15:0] _row_6_n_t8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:759:103
  wire [15:0] _row_6_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:758:103
  wire [15:0] _row_6_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:758:103
  wire [15:0] _row_6_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:757:103
  wire [15:0] _row_6_n_t5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:756:103
  wire [15:0] _row_6_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:755:103
  wire [15:0] _row_6_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:755:103
  wire [15:0] _row_6_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:754:103
  wire [15:0] _row_6_n_t4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:753:103
  wire [15:0] _row_6_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:752:103
  wire [15:0] _row_6_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:752:103
  wire [15:0] _row_6_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:751:103
  wire [15:0] _row_6_n_t8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:750:103
  wire [15:0] _row_6_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:749:103
  wire [15:0] _row_6_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:749:103
  wire [15:0] _row_6_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:748:103
  wire [15:0] _row_6_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:748:103
  wire [15:0] _row_6_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:747:103
  wire [15:0] _row_6_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:747:103
  wire [15:0] _row_6_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:746:103
  wire [15:0] _row_6_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:746:103
  wire [15:0] _row_6_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:745:103
  wire [15:0] _row_6_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:745:103
  wire [15:0] _row_6_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:744:103
  wire [15:0] _row_6_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:744:103
  wire [15:0] _row_6_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:743:103
  wire [15:0] _row_6_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:743:103
  wire [15:0] _row_6_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:742:103
  wire [15:0] _row_6_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:742:103
  wire [15:0] _row_6_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:741:103
  wire [15:0] _row_6_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:740:86
  wire [15:0] _row_6_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:739:86
  wire [15:0] _row_6_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:738:88
  wire [15:0] _row_6_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:737:88
  wire [15:0] _row_6_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:736:67
  wire [15:0] _row_6_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:735:88
  wire [15:0] _row_6_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:734:88
  wire [15:0] _row_6_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:733:67
  wire [15:0] _row_6_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:732:88
  wire [15:0] _row_6_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:731:88
  wire [15:0] _row_6_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:730:67
  wire [15:0] _row_6_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:729:79
  wire [15:0] _row_6_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:728:79
  wire [15:0] _row_6_n_c128_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:727:79
  wire [15:0] _row_6_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:726:79
  wire [15:0] _row_6_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:725:79
  wire [15:0] _row_5_n_shr_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:724:90
  wire [15:0] _row_5_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:723:108
  wire [15:0] _row_5_n_shr_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:722:90
  wire [15:0] _row_5_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:721:108
  wire [15:0] _row_5_n_shr_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:720:90
  wire [15:0] _row_5_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:719:108
  wire [15:0] _row_5_n_shr_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:718:90
  wire [15:0] _row_5_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:717:108
  wire [15:0] _row_5_n_shr_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:716:90
  wire [15:0] _row_5_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:715:108
  wire [15:0] _row_5_n_shr_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:714:90
  wire [15:0] _row_5_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:713:108
  wire [15:0] _row_5_n_shr_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:712:90
  wire [15:0] _row_5_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:711:108
  wire [15:0] _row_5_n_shr_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:710:90
  wire [15:0] _row_5_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:709:108
  wire [15:0] _row_5_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:708:103
  wire [15:0] _row_5_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:708:103
  wire [15:0] _row_5_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:707:86
  wire [15:0] _row_5_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:706:103
  wire [15:0] _row_5_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:705:103
  wire [15:0] _row_5_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:704:103
  wire [15:0] _row_5_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:703:103
  wire [15:0] _row_5_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:703:103
  wire [15:0] _row_5_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:702:86
  wire [15:0] _row_5_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:701:103
  wire [15:0] _row_5_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:700:103
  wire [15:0] _row_5_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:699:103
  wire [15:0] _row_5_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:698:103
  wire [15:0] _row_5_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:698:103
  wire [15:0] _row_5_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:697:103
  wire [15:0] _row_5_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:696:103
  wire [15:0] _row_5_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:696:103
  wire [15:0] _row_5_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:695:103
  wire [15:0] _row_5_d_x8_4_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:694:103
  wire [15:0] _row_5_d_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:694:103
  wire [15:0] _row_5_n_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:693:103
  wire [15:0] _row_5_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:692:103
  wire [15:0] _row_5_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:692:103
  wire [15:0] _row_5_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:691:103
  wire [15:0] _row_5_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:690:103
  wire [15:0] _row_5_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:690:103
  wire [15:0] _row_5_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:689:103
  wire [15:0] _row_5_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:688:103
  wire [15:0] _row_5_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:688:103
  wire [15:0] _row_5_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:687:103
  wire [15:0] _row_5_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:686:103
  wire [15:0] _row_5_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:686:103
  wire [15:0] _row_5_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:685:103
  wire [15:0] _row_5_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:684:103
  wire [15:0] _row_5_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:684:103
  wire [15:0] _row_5_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:683:103
  wire [15:0] _row_5_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:682:103
  wire [15:0] _row_5_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:682:103
  wire [15:0] _row_5_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:681:103
  wire [15:0] _row_5_n_t3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:680:103
  wire [15:0] _row_5_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:679:103
  wire [15:0] _row_5_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:679:103
  wire [15:0] _row_5_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:678:103
  wire [15:0] _row_5_n_t2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:677:103
  wire [15:0] _row_5_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:676:103
  wire [15:0] _row_5_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:676:103
  wire [15:0] _row_5_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:675:103
  wire [15:0] _row_5_n_t1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:674:103
  wire [15:0] _row_5_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:673:103
  wire [15:0] _row_5_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:673:103
  wire [15:0] _row_5_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:672:103
  wire [15:0] _row_5_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:671:103
  wire [15:0] _row_5_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:671:103
  wire [15:0] _row_5_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:670:103
  wire [15:0] _row_5_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:669:103
  wire [15:0] _row_5_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:669:103
  wire [15:0] _row_5_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:668:103
  wire [15:0] _row_5_n_t7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:667:103
  wire [15:0] _row_5_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:666:103
  wire [15:0] _row_5_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:666:103
  wire [15:0] _row_5_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:665:103
  wire [15:0] _row_5_n_t6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:664:103
  wire [15:0] _row_5_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:663:103
  wire [15:0] _row_5_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:663:103
  wire [15:0] _row_5_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:662:103
  wire [15:0] _row_5_n_t8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:661:103
  wire [15:0] _row_5_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:660:103
  wire [15:0] _row_5_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:660:103
  wire [15:0] _row_5_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:659:103
  wire [15:0] _row_5_n_t5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:658:103
  wire [15:0] _row_5_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:657:103
  wire [15:0] _row_5_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:657:103
  wire [15:0] _row_5_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:656:103
  wire [15:0] _row_5_n_t4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:655:103
  wire [15:0] _row_5_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:654:103
  wire [15:0] _row_5_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:654:103
  wire [15:0] _row_5_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:653:103
  wire [15:0] _row_5_n_t8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:652:103
  wire [15:0] _row_5_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:651:103
  wire [15:0] _row_5_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:651:103
  wire [15:0] _row_5_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:650:103
  wire [15:0] _row_5_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:650:103
  wire [15:0] _row_5_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:649:103
  wire [15:0] _row_5_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:649:103
  wire [15:0] _row_5_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:648:103
  wire [15:0] _row_5_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:648:103
  wire [15:0] _row_5_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:647:103
  wire [15:0] _row_5_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:647:103
  wire [15:0] _row_5_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:646:103
  wire [15:0] _row_5_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:646:103
  wire [15:0] _row_5_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:645:103
  wire [15:0] _row_5_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:645:103
  wire [15:0] _row_5_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:644:103
  wire [15:0] _row_5_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:644:103
  wire [15:0] _row_5_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:643:103
  wire [15:0] _row_5_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:642:86
  wire [15:0] _row_5_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:641:86
  wire [15:0] _row_5_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:640:88
  wire [15:0] _row_5_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:639:88
  wire [15:0] _row_5_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:638:67
  wire [15:0] _row_5_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:637:88
  wire [15:0] _row_5_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:636:88
  wire [15:0] _row_5_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:635:67
  wire [15:0] _row_5_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:634:88
  wire [15:0] _row_5_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:633:88
  wire [15:0] _row_5_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:632:67
  wire [15:0] _row_5_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:631:79
  wire [15:0] _row_5_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:630:79
  wire [15:0] _row_5_n_c128_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:629:79
  wire [15:0] _row_5_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:628:79
  wire [15:0] _row_5_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:627:79
  wire [15:0] _row_4_n_shr_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:626:90
  wire [15:0] _row_4_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:625:108
  wire [15:0] _row_4_n_shr_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:624:90
  wire [15:0] _row_4_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:623:108
  wire [15:0] _row_4_n_shr_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:622:90
  wire [15:0] _row_4_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:621:108
  wire [15:0] _row_4_n_shr_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:620:90
  wire [15:0] _row_4_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:619:108
  wire [15:0] _row_4_n_shr_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:618:90
  wire [15:0] _row_4_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:617:108
  wire [15:0] _row_4_n_shr_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:616:90
  wire [15:0] _row_4_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:615:108
  wire [15:0] _row_4_n_shr_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:614:90
  wire [15:0] _row_4_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:613:108
  wire [15:0] _row_4_n_shr_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:612:90
  wire [15:0] _row_4_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:611:108
  wire [15:0] _row_4_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:610:103
  wire [15:0] _row_4_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:610:103
  wire [15:0] _row_4_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:609:86
  wire [15:0] _row_4_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:608:103
  wire [15:0] _row_4_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:607:103
  wire [15:0] _row_4_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:606:103
  wire [15:0] _row_4_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:605:103
  wire [15:0] _row_4_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:605:103
  wire [15:0] _row_4_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:604:86
  wire [15:0] _row_4_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:603:103
  wire [15:0] _row_4_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:602:103
  wire [15:0] _row_4_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:601:103
  wire [15:0] _row_4_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:600:103
  wire [15:0] _row_4_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:600:103
  wire [15:0] _row_4_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:599:103
  wire [15:0] _row_4_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:598:103
  wire [15:0] _row_4_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:598:103
  wire [15:0] _row_4_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:597:103
  wire [15:0] _row_4_d_x8_4_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:596:103
  wire [15:0] _row_4_d_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:596:103
  wire [15:0] _row_4_n_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:595:103
  wire [15:0] _row_4_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:594:103
  wire [15:0] _row_4_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:594:103
  wire [15:0] _row_4_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:593:103
  wire [15:0] _row_4_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:592:103
  wire [15:0] _row_4_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:592:103
  wire [15:0] _row_4_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:591:103
  wire [15:0] _row_4_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:590:103
  wire [15:0] _row_4_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:590:103
  wire [15:0] _row_4_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:589:103
  wire [15:0] _row_4_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:588:103
  wire [15:0] _row_4_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:588:103
  wire [15:0] _row_4_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:587:103
  wire [15:0] _row_4_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:586:103
  wire [15:0] _row_4_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:586:103
  wire [15:0] _row_4_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:585:103
  wire [15:0] _row_4_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:584:103
  wire [15:0] _row_4_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:584:103
  wire [15:0] _row_4_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:583:103
  wire [15:0] _row_4_n_t3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:582:103
  wire [15:0] _row_4_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:581:103
  wire [15:0] _row_4_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:581:103
  wire [15:0] _row_4_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:580:103
  wire [15:0] _row_4_n_t2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:579:103
  wire [15:0] _row_4_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:578:103
  wire [15:0] _row_4_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:578:103
  wire [15:0] _row_4_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:577:103
  wire [15:0] _row_4_n_t1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:576:103
  wire [15:0] _row_4_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:575:103
  wire [15:0] _row_4_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:575:103
  wire [15:0] _row_4_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:574:103
  wire [15:0] _row_4_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:573:103
  wire [15:0] _row_4_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:573:103
  wire [15:0] _row_4_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:572:103
  wire [15:0] _row_4_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:571:103
  wire [15:0] _row_4_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:571:103
  wire [15:0] _row_4_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:570:103
  wire [15:0] _row_4_n_t7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:569:103
  wire [15:0] _row_4_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:568:103
  wire [15:0] _row_4_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:568:103
  wire [15:0] _row_4_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:567:103
  wire [15:0] _row_4_n_t6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:566:103
  wire [15:0] _row_4_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:565:103
  wire [15:0] _row_4_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:565:103
  wire [15:0] _row_4_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:564:103
  wire [15:0] _row_4_n_t8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:563:103
  wire [15:0] _row_4_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:562:103
  wire [15:0] _row_4_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:562:103
  wire [15:0] _row_4_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:561:103
  wire [15:0] _row_4_n_t5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:560:103
  wire [15:0] _row_4_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:559:103
  wire [15:0] _row_4_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:559:103
  wire [15:0] _row_4_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:558:103
  wire [15:0] _row_4_n_t4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:557:103
  wire [15:0] _row_4_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:556:103
  wire [15:0] _row_4_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:556:103
  wire [15:0] _row_4_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:555:103
  wire [15:0] _row_4_n_t8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:554:103
  wire [15:0] _row_4_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:553:103
  wire [15:0] _row_4_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:553:103
  wire [15:0] _row_4_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:552:103
  wire [15:0] _row_4_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:552:103
  wire [15:0] _row_4_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:551:103
  wire [15:0] _row_4_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:551:103
  wire [15:0] _row_4_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:550:103
  wire [15:0] _row_4_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:550:103
  wire [15:0] _row_4_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:549:103
  wire [15:0] _row_4_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:549:103
  wire [15:0] _row_4_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:548:103
  wire [15:0] _row_4_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:548:103
  wire [15:0] _row_4_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:547:103
  wire [15:0] _row_4_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:547:103
  wire [15:0] _row_4_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:546:103
  wire [15:0] _row_4_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:546:103
  wire [15:0] _row_4_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:545:103
  wire [15:0] _row_4_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:544:86
  wire [15:0] _row_4_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:543:86
  wire [15:0] _row_4_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:542:88
  wire [15:0] _row_4_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:541:88
  wire [15:0] _row_4_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:540:67
  wire [15:0] _row_4_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:539:88
  wire [15:0] _row_4_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:538:88
  wire [15:0] _row_4_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:537:67
  wire [15:0] _row_4_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:536:88
  wire [15:0] _row_4_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:535:88
  wire [15:0] _row_4_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:534:67
  wire [15:0] _row_4_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:533:79
  wire [15:0] _row_4_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:532:79
  wire [15:0] _row_4_n_c128_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:531:79
  wire [15:0] _row_4_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:530:79
  wire [15:0] _row_4_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:529:79
  wire [15:0] _row_3_n_shr_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:528:90
  wire [15:0] _row_3_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:527:108
  wire [15:0] _row_3_n_shr_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:526:90
  wire [15:0] _row_3_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:525:108
  wire [15:0] _row_3_n_shr_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:524:90
  wire [15:0] _row_3_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:523:108
  wire [15:0] _row_3_n_shr_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:522:90
  wire [15:0] _row_3_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:521:108
  wire [15:0] _row_3_n_shr_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:520:90
  wire [15:0] _row_3_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:519:108
  wire [15:0] _row_3_n_shr_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:518:90
  wire [15:0] _row_3_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:517:108
  wire [15:0] _row_3_n_shr_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:516:90
  wire [15:0] _row_3_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:515:108
  wire [15:0] _row_3_n_shr_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:514:90
  wire [15:0] _row_3_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:513:108
  wire [15:0] _row_3_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:512:103
  wire [15:0] _row_3_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:512:103
  wire [15:0] _row_3_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:511:86
  wire [15:0] _row_3_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:510:103
  wire [15:0] _row_3_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:509:103
  wire [15:0] _row_3_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:508:103
  wire [15:0] _row_3_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:507:103
  wire [15:0] _row_3_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:507:103
  wire [15:0] _row_3_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:506:86
  wire [15:0] _row_3_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:505:103
  wire [15:0] _row_3_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:504:103
  wire [15:0] _row_3_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:503:103
  wire [15:0] _row_3_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:502:103
  wire [15:0] _row_3_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:502:103
  wire [15:0] _row_3_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:501:103
  wire [15:0] _row_3_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:500:103
  wire [15:0] _row_3_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:500:103
  wire [15:0] _row_3_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:499:103
  wire [15:0] _row_3_d_x8_4_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:498:103
  wire [15:0] _row_3_d_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:498:103
  wire [15:0] _row_3_n_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:497:103
  wire [15:0] _row_3_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:496:103
  wire [15:0] _row_3_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:496:103
  wire [15:0] _row_3_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:495:103
  wire [15:0] _row_3_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:494:103
  wire [15:0] _row_3_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:494:103
  wire [15:0] _row_3_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:493:103
  wire [15:0] _row_3_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:492:103
  wire [15:0] _row_3_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:492:103
  wire [15:0] _row_3_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:491:103
  wire [15:0] _row_3_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:490:103
  wire [15:0] _row_3_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:490:103
  wire [15:0] _row_3_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:489:103
  wire [15:0] _row_3_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:488:103
  wire [15:0] _row_3_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:488:103
  wire [15:0] _row_3_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:487:103
  wire [15:0] _row_3_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:486:103
  wire [15:0] _row_3_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:486:103
  wire [15:0] _row_3_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:485:103
  wire [15:0] _row_3_n_t3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:484:103
  wire [15:0] _row_3_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:483:103
  wire [15:0] _row_3_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:483:103
  wire [15:0] _row_3_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:482:103
  wire [15:0] _row_3_n_t2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:481:103
  wire [15:0] _row_3_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:480:103
  wire [15:0] _row_3_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:480:103
  wire [15:0] _row_3_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:479:103
  wire [15:0] _row_3_n_t1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:478:103
  wire [15:0] _row_3_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:477:103
  wire [15:0] _row_3_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:477:103
  wire [15:0] _row_3_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:476:103
  wire [15:0] _row_3_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:475:103
  wire [15:0] _row_3_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:475:103
  wire [15:0] _row_3_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:474:103
  wire [15:0] _row_3_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:473:103
  wire [15:0] _row_3_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:473:103
  wire [15:0] _row_3_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:472:103
  wire [15:0] _row_3_n_t7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:471:103
  wire [15:0] _row_3_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:470:103
  wire [15:0] _row_3_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:470:103
  wire [15:0] _row_3_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:469:103
  wire [15:0] _row_3_n_t6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:468:103
  wire [15:0] _row_3_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:467:103
  wire [15:0] _row_3_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:467:103
  wire [15:0] _row_3_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:466:103
  wire [15:0] _row_3_n_t8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:465:103
  wire [15:0] _row_3_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:464:103
  wire [15:0] _row_3_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:464:103
  wire [15:0] _row_3_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:463:103
  wire [15:0] _row_3_n_t5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:462:103
  wire [15:0] _row_3_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:461:103
  wire [15:0] _row_3_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:461:103
  wire [15:0] _row_3_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:460:103
  wire [15:0] _row_3_n_t4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:459:103
  wire [15:0] _row_3_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:458:103
  wire [15:0] _row_3_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:458:103
  wire [15:0] _row_3_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:457:103
  wire [15:0] _row_3_n_t8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:456:103
  wire [15:0] _row_3_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:455:103
  wire [15:0] _row_3_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:455:103
  wire [15:0] _row_3_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:454:103
  wire [15:0] _row_3_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:454:103
  wire [15:0] _row_3_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:453:103
  wire [15:0] _row_3_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:453:103
  wire [15:0] _row_3_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:452:103
  wire [15:0] _row_3_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:452:103
  wire [15:0] _row_3_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:451:103
  wire [15:0] _row_3_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:451:103
  wire [15:0] _row_3_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:450:103
  wire [15:0] _row_3_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:450:103
  wire [15:0] _row_3_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:449:103
  wire [15:0] _row_3_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:449:103
  wire [15:0] _row_3_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:448:103
  wire [15:0] _row_3_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:448:103
  wire [15:0] _row_3_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:447:103
  wire [15:0] _row_3_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:446:86
  wire [15:0] _row_3_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:445:86
  wire [15:0] _row_3_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:444:88
  wire [15:0] _row_3_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:443:88
  wire [15:0] _row_3_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:442:67
  wire [15:0] _row_3_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:441:88
  wire [15:0] _row_3_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:440:88
  wire [15:0] _row_3_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:439:67
  wire [15:0] _row_3_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:438:88
  wire [15:0] _row_3_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:437:88
  wire [15:0] _row_3_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:436:67
  wire [15:0] _row_3_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:435:79
  wire [15:0] _row_3_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:434:79
  wire [15:0] _row_3_n_c128_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:433:79
  wire [15:0] _row_3_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:432:79
  wire [15:0] _row_3_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:431:79
  wire [15:0] _row_2_n_shr_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:430:90
  wire [15:0] _row_2_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:429:108
  wire [15:0] _row_2_n_shr_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:428:90
  wire [15:0] _row_2_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:427:108
  wire [15:0] _row_2_n_shr_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:426:90
  wire [15:0] _row_2_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:425:108
  wire [15:0] _row_2_n_shr_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:424:90
  wire [15:0] _row_2_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:423:108
  wire [15:0] _row_2_n_shr_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:422:90
  wire [15:0] _row_2_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:421:108
  wire [15:0] _row_2_n_shr_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:420:90
  wire [15:0] _row_2_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:419:108
  wire [15:0] _row_2_n_shr_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:418:90
  wire [15:0] _row_2_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:417:108
  wire [15:0] _row_2_n_shr_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:416:90
  wire [15:0] _row_2_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:415:108
  wire [15:0] _row_2_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:414:103
  wire [15:0] _row_2_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:414:103
  wire [15:0] _row_2_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:413:86
  wire [15:0] _row_2_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:412:103
  wire [15:0] _row_2_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:411:103
  wire [15:0] _row_2_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:410:103
  wire [15:0] _row_2_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:409:103
  wire [15:0] _row_2_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:409:103
  wire [15:0] _row_2_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:408:86
  wire [15:0] _row_2_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:407:103
  wire [15:0] _row_2_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:406:103
  wire [15:0] _row_2_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:405:103
  wire [15:0] _row_2_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:404:103
  wire [15:0] _row_2_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:404:103
  wire [15:0] _row_2_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:403:103
  wire [15:0] _row_2_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:402:103
  wire [15:0] _row_2_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:402:103
  wire [15:0] _row_2_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:401:103
  wire [15:0] _row_2_d_x8_4_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:400:103
  wire [15:0] _row_2_d_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:400:103
  wire [15:0] _row_2_n_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:399:103
  wire [15:0] _row_2_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:398:103
  wire [15:0] _row_2_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:398:103
  wire [15:0] _row_2_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:397:103
  wire [15:0] _row_2_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:396:103
  wire [15:0] _row_2_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:396:103
  wire [15:0] _row_2_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:395:103
  wire [15:0] _row_2_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:394:103
  wire [15:0] _row_2_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:394:103
  wire [15:0] _row_2_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:393:103
  wire [15:0] _row_2_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:392:103
  wire [15:0] _row_2_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:392:103
  wire [15:0] _row_2_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:391:103
  wire [15:0] _row_2_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:390:103
  wire [15:0] _row_2_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:390:103
  wire [15:0] _row_2_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:389:103
  wire [15:0] _row_2_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:388:103
  wire [15:0] _row_2_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:388:103
  wire [15:0] _row_2_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:387:103
  wire [15:0] _row_2_n_t3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:386:103
  wire [15:0] _row_2_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:385:103
  wire [15:0] _row_2_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:385:103
  wire [15:0] _row_2_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:384:103
  wire [15:0] _row_2_n_t2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:383:103
  wire [15:0] _row_2_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:382:103
  wire [15:0] _row_2_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:382:103
  wire [15:0] _row_2_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:381:103
  wire [15:0] _row_2_n_t1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:380:103
  wire [15:0] _row_2_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:379:103
  wire [15:0] _row_2_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:379:103
  wire [15:0] _row_2_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:378:103
  wire [15:0] _row_2_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:377:103
  wire [15:0] _row_2_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:377:103
  wire [15:0] _row_2_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:376:103
  wire [15:0] _row_2_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:375:103
  wire [15:0] _row_2_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:375:103
  wire [15:0] _row_2_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:374:103
  wire [15:0] _row_2_n_t7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:373:103
  wire [15:0] _row_2_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:372:103
  wire [15:0] _row_2_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:372:103
  wire [15:0] _row_2_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:371:103
  wire [15:0] _row_2_n_t6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:370:103
  wire [15:0] _row_2_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:369:103
  wire [15:0] _row_2_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:369:103
  wire [15:0] _row_2_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:368:103
  wire [15:0] _row_2_n_t8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:367:103
  wire [15:0] _row_2_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:366:103
  wire [15:0] _row_2_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:366:103
  wire [15:0] _row_2_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:365:103
  wire [15:0] _row_2_n_t5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:364:103
  wire [15:0] _row_2_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:363:103
  wire [15:0] _row_2_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:363:103
  wire [15:0] _row_2_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:362:103
  wire [15:0] _row_2_n_t4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:361:103
  wire [15:0] _row_2_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:360:103
  wire [15:0] _row_2_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:360:103
  wire [15:0] _row_2_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:359:103
  wire [15:0] _row_2_n_t8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:358:103
  wire [15:0] _row_2_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:357:103
  wire [15:0] _row_2_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:357:103
  wire [15:0] _row_2_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:356:103
  wire [15:0] _row_2_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:356:103
  wire [15:0] _row_2_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:355:103
  wire [15:0] _row_2_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:355:103
  wire [15:0] _row_2_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:354:103
  wire [15:0] _row_2_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:354:103
  wire [15:0] _row_2_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:353:103
  wire [15:0] _row_2_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:353:103
  wire [15:0] _row_2_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:352:103
  wire [15:0] _row_2_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:352:103
  wire [15:0] _row_2_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:351:103
  wire [15:0] _row_2_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:351:103
  wire [15:0] _row_2_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:350:103
  wire [15:0] _row_2_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:350:103
  wire [15:0] _row_2_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:349:103
  wire [15:0] _row_2_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:348:86
  wire [15:0] _row_2_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:347:86
  wire [15:0] _row_2_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:346:88
  wire [15:0] _row_2_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:345:88
  wire [15:0] _row_2_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:344:67
  wire [15:0] _row_2_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:343:88
  wire [15:0] _row_2_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:342:88
  wire [15:0] _row_2_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:341:67
  wire [15:0] _row_2_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:340:88
  wire [15:0] _row_2_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:339:88
  wire [15:0] _row_2_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:338:67
  wire [15:0] _row_2_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:337:79
  wire [15:0] _row_2_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:336:79
  wire [15:0] _row_2_n_c128_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:335:79
  wire [15:0] _row_2_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:334:79
  wire [15:0] _row_2_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:333:79
  wire [15:0] _row_1_n_shr_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:332:90
  wire [15:0] _row_1_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:331:108
  wire [15:0] _row_1_n_shr_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:330:90
  wire [15:0] _row_1_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:329:108
  wire [15:0] _row_1_n_shr_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:328:90
  wire [15:0] _row_1_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:327:108
  wire [15:0] _row_1_n_shr_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:326:90
  wire [15:0] _row_1_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:325:108
  wire [15:0] _row_1_n_shr_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:324:90
  wire [15:0] _row_1_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:323:108
  wire [15:0] _row_1_n_shr_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:322:90
  wire [15:0] _row_1_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:321:108
  wire [15:0] _row_1_n_shr_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:320:90
  wire [15:0] _row_1_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:319:108
  wire [15:0] _row_1_n_shr_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:318:90
  wire [15:0] _row_1_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:317:108
  wire [15:0] _row_1_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:316:103
  wire [15:0] _row_1_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:316:103
  wire [15:0] _row_1_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:315:86
  wire [15:0] _row_1_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:314:103
  wire [15:0] _row_1_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:313:103
  wire [15:0] _row_1_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:312:103
  wire [15:0] _row_1_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:311:103
  wire [15:0] _row_1_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:311:103
  wire [15:0] _row_1_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:310:86
  wire [15:0] _row_1_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:309:103
  wire [15:0] _row_1_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:308:103
  wire [15:0] _row_1_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:307:103
  wire [15:0] _row_1_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:306:103
  wire [15:0] _row_1_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:306:103
  wire [15:0] _row_1_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:305:103
  wire [15:0] _row_1_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:304:103
  wire [15:0] _row_1_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:304:103
  wire [15:0] _row_1_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:303:103
  wire [15:0] _row_1_d_x8_4_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:302:103
  wire [15:0] _row_1_d_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:302:103
  wire [15:0] _row_1_n_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:301:103
  wire [15:0] _row_1_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:300:103
  wire [15:0] _row_1_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:300:103
  wire [15:0] _row_1_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:299:103
  wire [15:0] _row_1_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:298:103
  wire [15:0] _row_1_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:298:103
  wire [15:0] _row_1_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:297:103
  wire [15:0] _row_1_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:296:103
  wire [15:0] _row_1_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:296:103
  wire [15:0] _row_1_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:295:103
  wire [15:0] _row_1_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:294:103
  wire [15:0] _row_1_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:294:103
  wire [15:0] _row_1_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:293:103
  wire [15:0] _row_1_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:292:103
  wire [15:0] _row_1_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:292:103
  wire [15:0] _row_1_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:291:103
  wire [15:0] _row_1_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:290:103
  wire [15:0] _row_1_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:290:103
  wire [15:0] _row_1_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:289:103
  wire [15:0] _row_1_n_t3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:288:103
  wire [15:0] _row_1_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:287:103
  wire [15:0] _row_1_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:287:103
  wire [15:0] _row_1_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:286:103
  wire [15:0] _row_1_n_t2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:285:103
  wire [15:0] _row_1_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:284:103
  wire [15:0] _row_1_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:284:103
  wire [15:0] _row_1_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:283:103
  wire [15:0] _row_1_n_t1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:282:103
  wire [15:0] _row_1_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:281:103
  wire [15:0] _row_1_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:281:103
  wire [15:0] _row_1_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:280:103
  wire [15:0] _row_1_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:279:103
  wire [15:0] _row_1_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:279:103
  wire [15:0] _row_1_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:278:103
  wire [15:0] _row_1_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:277:103
  wire [15:0] _row_1_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:277:103
  wire [15:0] _row_1_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:276:103
  wire [15:0] _row_1_n_t7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:275:103
  wire [15:0] _row_1_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:274:103
  wire [15:0] _row_1_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:274:103
  wire [15:0] _row_1_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:273:103
  wire [15:0] _row_1_n_t6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:272:103
  wire [15:0] _row_1_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:271:103
  wire [15:0] _row_1_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:271:103
  wire [15:0] _row_1_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:270:103
  wire [15:0] _row_1_n_t8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:269:103
  wire [15:0] _row_1_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:268:103
  wire [15:0] _row_1_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:268:103
  wire [15:0] _row_1_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:267:103
  wire [15:0] _row_1_n_t5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:266:103
  wire [15:0] _row_1_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:265:103
  wire [15:0] _row_1_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:265:103
  wire [15:0] _row_1_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:264:103
  wire [15:0] _row_1_n_t4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:263:103
  wire [15:0] _row_1_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:262:103
  wire [15:0] _row_1_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:262:103
  wire [15:0] _row_1_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:261:103
  wire [15:0] _row_1_n_t8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:260:103
  wire [15:0] _row_1_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:259:103
  wire [15:0] _row_1_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:259:103
  wire [15:0] _row_1_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:258:103
  wire [15:0] _row_1_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:258:103
  wire [15:0] _row_1_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:257:103
  wire [15:0] _row_1_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:257:103
  wire [15:0] _row_1_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:256:103
  wire [15:0] _row_1_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:256:103
  wire [15:0] _row_1_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:255:103
  wire [15:0] _row_1_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:255:103
  wire [15:0] _row_1_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:254:103
  wire [15:0] _row_1_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:254:103
  wire [15:0] _row_1_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:253:103
  wire [15:0] _row_1_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:253:103
  wire [15:0] _row_1_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:252:103
  wire [15:0] _row_1_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:252:103
  wire [15:0] _row_1_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:251:103
  wire [15:0] _row_1_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:250:86
  wire [15:0] _row_1_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:249:86
  wire [15:0] _row_1_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:248:88
  wire [15:0] _row_1_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:247:88
  wire [15:0] _row_1_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:246:67
  wire [15:0] _row_1_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:245:88
  wire [15:0] _row_1_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:244:88
  wire [15:0] _row_1_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:243:67
  wire [15:0] _row_1_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:242:88
  wire [15:0] _row_1_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:241:88
  wire [15:0] _row_1_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:240:67
  wire [15:0] _row_1_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:239:79
  wire [15:0] _row_1_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:238:79
  wire [15:0] _row_1_n_c128_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:237:79
  wire [15:0] _row_1_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:236:79
  wire [15:0] _row_1_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:235:79
  wire [15:0] _row_0_n_shr_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:234:90
  wire [15:0] _row_0_n_tmp_7_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:233:108
  wire [15:0] _row_0_n_shr_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:232:90
  wire [15:0] _row_0_n_tmp_6_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:231:108
  wire [15:0] _row_0_n_shr_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:230:90
  wire [15:0] _row_0_n_tmp_5_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:229:108
  wire [15:0] _row_0_n_shr_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:228:90
  wire [15:0] _row_0_n_tmp_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:227:108
  wire [15:0] _row_0_n_shr_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:226:90
  wire [15:0] _row_0_n_tmp_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:225:108
  wire [15:0] _row_0_n_shr_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:224:90
  wire [15:0] _row_0_n_tmp_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:223:108
  wire [15:0] _row_0_n_shr_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:222:90
  wire [15:0] _row_0_n_tmp_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:221:108
  wire [15:0] _row_0_n_shr_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:220:90
  wire [15:0] _row_0_n_tmp_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:219:108
  wire [15:0] _row_0_d_x4_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:218:103
  wire [15:0] _row_0_d_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:218:103
  wire [15:0] _row_0_n_x4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:217:86
  wire [15:0] _row_0_n_w4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:216:103
  wire [15:0] _row_0_n_v4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:215:103
  wire [15:0] _row_0_n_u4_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:214:103
  wire [15:0] _row_0_d_x2_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:213:103
  wire [15:0] _row_0_d_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:213:103
  wire [15:0] _row_0_n_x2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:212:86
  wire [15:0] _row_0_n_w2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:211:103
  wire [15:0] _row_0_n_v2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:210:103
  wire [15:0] _row_0_n_u2_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:209:103
  wire [15:0] _row_0_d_x0_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:208:103
  wire [15:0] _row_0_d_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:208:103
  wire [15:0] _row_0_n_x0_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:207:103
  wire [15:0] _row_0_d_x3_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:206:103
  wire [15:0] _row_0_d_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:206:103
  wire [15:0] _row_0_n_x3_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:205:103
  wire [15:0] _row_0_d_x8_4_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:204:103
  wire [15:0] _row_0_d_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:204:103
  wire [15:0] _row_0_n_x8_4_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:203:103
  wire [15:0] _row_0_d_x7_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:202:103
  wire [15:0] _row_0_d_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:202:103
  wire [15:0] _row_0_n_x7_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:201:103
  wire [15:0] _row_0_d_x5_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:200:103
  wire [15:0] _row_0_d_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:200:103
  wire [15:0] _row_0_n_x5_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:199:103
  wire [15:0] _row_0_d_x6_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:198:103
  wire [15:0] _row_0_d_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:198:103
  wire [15:0] _row_0_n_x6_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:197:103
  wire [15:0] _row_0_d_x4_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:196:103
  wire [15:0] _row_0_d_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:196:103
  wire [15:0] _row_0_n_x4_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:195:103
  wire [15:0] _row_0_d_x1_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:194:103
  wire [15:0] _row_0_d_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:194:103
  wire [15:0] _row_0_n_x1_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:193:103
  wire [15:0] _row_0_d_x3_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:192:103
  wire [15:0] _row_0_d_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:192:103
  wire [15:0] _row_0_n_x3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:191:103
  wire [15:0] _row_0_n_t3_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:190:103
  wire [15:0] _row_0_d_x2_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:189:103
  wire [15:0] _row_0_d_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:189:103
  wire [15:0] _row_0_n_x2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:188:103
  wire [15:0] _row_0_n_t2_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:187:103
  wire [15:0] _row_0_d_x1_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:186:103
  wire [15:0] _row_0_d_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:186:103
  wire [15:0] _row_0_n_x1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:185:103
  wire [15:0] _row_0_n_t1_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:184:103
  wire [15:0] _row_0_d_x0_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:183:103
  wire [15:0] _row_0_d_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:183:103
  wire [15:0] _row_0_n_x0_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:182:103
  wire [15:0] _row_0_d_x8_3_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:181:103
  wire [15:0] _row_0_d_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:181:103
  wire [15:0] _row_0_n_x8_3_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:180:103
  wire [15:0] _row_0_d_x7_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:179:103
  wire [15:0] _row_0_d_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:179:103
  wire [15:0] _row_0_n_x7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:178:103
  wire [15:0] _row_0_n_t7_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:177:103
  wire [15:0] _row_0_d_x6_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:176:103
  wire [15:0] _row_0_d_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:176:103
  wire [15:0] _row_0_n_x6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:175:103
  wire [15:0] _row_0_n_t6_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:174:103
  wire [15:0] _row_0_d_x8_2_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:173:103
  wire [15:0] _row_0_d_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:173:103
  wire [15:0] _row_0_n_x8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:172:103
  wire [15:0] _row_0_n_t8_2_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:171:103
  wire [15:0] _row_0_d_x5_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:170:103
  wire [15:0] _row_0_d_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:170:103
  wire [15:0] _row_0_n_x5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:169:103
  wire [15:0] _row_0_n_t5_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:168:103
  wire [15:0] _row_0_d_x4_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:167:103
  wire [15:0] _row_0_d_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:167:103
  wire [15:0] _row_0_n_x4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:166:103
  wire [15:0] _row_0_n_t4_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:165:103
  wire [15:0] _row_0_d_x8_1_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:164:103
  wire [15:0] _row_0_d_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:164:103
  wire [15:0] _row_0_n_x8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:163:103
  wire [15:0] _row_0_n_t8_1_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:162:103
  wire [15:0] _row_0_d_x7_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:161:103
  wire [15:0] _row_0_d_x7_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:161:103
  wire [15:0] _row_0_d_x6_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:160:103
  wire [15:0] _row_0_d_x6_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:160:103
  wire [15:0] _row_0_d_x5_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:159:103
  wire [15:0] _row_0_d_x5_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:159:103
  wire [15:0] _row_0_d_x4_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:158:103
  wire [15:0] _row_0_d_x4_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:158:103
  wire [15:0] _row_0_d_x3_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:157:103
  wire [15:0] _row_0_d_x3_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:157:103
  wire [15:0] _row_0_d_x2_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:156:103
  wire [15:0] _row_0_d_x2_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:156:103
  wire [15:0] _row_0_d_x1_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:155:103
  wire [15:0] _row_0_d_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:155:103
  wire [15:0] _row_0_d_x0_0_y;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:154:103
  wire [15:0] _row_0_d_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:154:103
  wire [15:0] _row_0_n_x0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:153:103
  wire [15:0] _row_0_n_t0_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:152:86
  wire [15:0] _row_0_n_x1_0_z;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:151:86
  wire [15:0] _row_0_n_w2_add_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:150:88
  wire [15:0] _row_0_n_w2_sub_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:149:88
  wire [15:0] _row_0_n_w6_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:148:67
  wire [15:0] _row_0_n_w3_add_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:147:88
  wire [15:0] _row_0_n_w3_sub_w5_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:146:88
  wire [15:0] _row_0_n_w3_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:145:67
  wire [15:0] _row_0_n_w1_add_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:144:88
  wire [15:0] _row_0_n_w1_sub_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:143:88
  wire [15:0] _row_0_n_w7_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:142:67
  wire [15:0] _row_0_n_c181_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:141:79
  wire [15:0] _row_0_n_c181_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:140:79
  wire [15:0] _row_0_n_c128_2_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:139:79
  wire [15:0] _row_0_n_c128_1_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:138:79
  wire [15:0] _row_0_n_c128_0_value;	// ./test/data/hil/idct/outputFirrtlIdct.mlir:137:79

  C128 row_0_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:137:79
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_c128_0_value)
  );
  C128 row_0_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:138:79
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_c128_1_value)
  );
  C128 row_0_n_c128_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:139:79
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_c128_2_value)
  );
  C181 row_0_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:140:79
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_c181_0_value)
  );
  C181 row_0_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:141:79
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_c181_1_value)
  );
  W7 row_0_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:142:67
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_w7_value)
  );
  W1_sub_W7 row_0_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:143:88
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_w1_sub_w7_value)
  );
  W1_add_W7 row_0_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:144:88
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_w1_add_w7_value)
  );
  W3 row_0_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:145:67
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_w3_value)
  );
  W3_sub_W5 row_0_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:146:88
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_w3_sub_w5_value)
  );
  W3_add_W5 row_0_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:147:88
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_w3_add_w5_value)
  );
  W6 row_0_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:148:67
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_w6_value)
  );
  W2_sub_W6 row_0_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:149:88
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_w2_sub_w6_value)
  );
  W2_add_W6 row_0_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:150:88
    .clock (clock),
    .reset (reset),
    .value (_row_0_n_w2_add_w6_value)
  );
  SHL_11 row_0_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:151:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_0_4_x),
    .z     (_row_0_n_x1_0_z)
  );
  SHL_11 row_0_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:152:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_0_0_x),
    .z     (_row_0_n_t0_0_z)
  );
  ADD row_0_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:153:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:152:86
    .y     (_row_0_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:137:79
    .z     (_row_0_n_x0_0_z)
  );
  dup_2 row_0_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:154:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:153:103
    .y     (_row_0_d_x0_0_y),
    .z     (_row_0_d_x0_0_z)
  );
  dup_2 row_0_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:155:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:151:86
    .y     (_row_0_d_x1_0_y),
    .z     (_row_0_d_x1_0_z)
  );
  dup_2 row_0_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:156:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_0_6_x),
    .y     (_row_0_d_x2_0_y),
    .z     (_row_0_d_x2_0_z)
  );
  dup_2 row_0_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:157:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_0_2_x),
    .y     (_row_0_d_x3_0_y),
    .z     (_row_0_d_x3_0_z)
  );
  dup_2 row_0_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:158:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_0_1_x),
    .y     (_row_0_d_x4_0_y),
    .z     (_row_0_d_x4_0_z)
  );
  dup_2 row_0_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:159:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_0_7_x),
    .y     (_row_0_d_x5_0_y),
    .z     (_row_0_d_x5_0_z)
  );
  dup_2 row_0_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:160:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_0_5_x),
    .y     (_row_0_d_x6_0_y),
    .z     (_row_0_d_x6_0_z)
  );
  dup_2 row_0_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:161:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_0_3_x),
    .y     (_row_0_d_x7_0_y),
    .z     (_row_0_d_x7_0_z)
  );
  ADD row_0_n_t8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:162:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:158:103
    .y     (_row_0_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:159:103
    .z     (_row_0_n_t8_1_z)
  );
  MUL row_0_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:163:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:142:67
    .y     (_row_0_n_t8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:162:103
    .z     (_row_0_n_x8_1_z)
  );
  dup_2 row_0_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:164:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:163:103
    .y     (_row_0_d_x8_1_y),
    .z     (_row_0_d_x8_1_z)
  );
  MUL row_0_n_t4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:165:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:143:88
    .y     (_row_0_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:158:103
    .z     (_row_0_n_t4_1_z)
  );
  ADD row_0_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:166:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:164:103
    .y     (_delay_INT16_1_1859_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1868:113
    .z     (_row_0_n_x4_1_z)
  );
  dup_2 row_0_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:167:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:166:103
    .y     (_row_0_d_x4_1_y),
    .z     (_row_0_d_x4_1_z)
  );
  MUL row_0_n_t5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:168:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:144:88
    .y     (_row_0_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:159:103
    .z     (_row_0_n_t5_1_z)
  );
  SUB row_0_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:169:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:164:103
    .y     (_delay_INT16_1_1860_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1869:113
    .z     (_row_0_n_x5_1_z)
  );
  dup_2 row_0_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:170:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:169:103
    .y     (_row_0_d_x5_1_y),
    .z     (_row_0_d_x5_1_z)
  );
  ADD row_0_n_t8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:171:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:160:103
    .y     (_row_0_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:161:103
    .z     (_row_0_n_t8_2_z)
  );
  MUL row_0_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:172:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:145:67
    .y     (_row_0_n_t8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:171:103
    .z     (_row_0_n_x8_2_z)
  );
  dup_2 row_0_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:173:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:172:103
    .y     (_row_0_d_x8_2_y),
    .z     (_row_0_d_x8_2_z)
  );
  MUL row_0_n_t6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:174:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:146:88
    .y     (_row_0_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:160:103
    .z     (_row_0_n_t6_1_z)
  );
  SUB row_0_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:175:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:173:103
    .y     (_delay_INT16_1_1861_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1870:113
    .z     (_row_0_n_x6_1_z)
  );
  dup_2 row_0_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:176:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:175:103
    .y     (_row_0_d_x6_1_y),
    .z     (_row_0_d_x6_1_z)
  );
  MUL row_0_n_t7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:177:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:147:88
    .y     (_row_0_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:161:103
    .z     (_row_0_n_t7_1_z)
  );
  SUB row_0_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:178:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:173:103
    .y     (_delay_INT16_1_1862_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1871:113
    .z     (_row_0_n_x7_1_z)
  );
  dup_2 row_0_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:179:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:178:103
    .y     (_row_0_d_x7_1_y),
    .z     (_row_0_d_x7_1_z)
  );
  ADD row_0_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:180:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:154:103
    .y     (_delay_INT16_1_1863_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1872:113
    .z     (_row_0_n_x8_3_z)
  );
  dup_2 row_0_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:181:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:180:103
    .y     (_row_0_d_x8_3_y),
    .z     (_row_0_d_x8_3_z)
  );
  SUB row_0_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:182:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:154:103
    .y     (_delay_INT16_1_1864_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1873:113
    .z     (_row_0_n_x0_1_z)
  );
  dup_2 row_0_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:183:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:182:103
    .y     (_row_0_d_x0_1_y),
    .z     (_row_0_d_x0_1_z)
  );
  ADD row_0_n_t1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:184:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:157:103
    .y     (_row_0_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:156:103
    .z     (_row_0_n_t1_1_z)
  );
  MUL row_0_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:185:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:148:67
    .y     (_row_0_n_t1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:184:103
    .z     (_row_0_n_x1_1_z)
  );
  dup_2 row_0_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:186:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:185:103
    .y     (_row_0_d_x1_1_y),
    .z     (_row_0_d_x1_1_z)
  );
  MUL row_0_n_t2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:187:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:150:88
    .y     (_row_0_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:156:103
    .z     (_row_0_n_t2_1_z)
  );
  SUB row_0_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:188:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:186:103
    .y     (_delay_INT16_1_1865_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1874:113
    .z     (_row_0_n_x2_1_z)
  );
  dup_2 row_0_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:189:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:188:103
    .y     (_row_0_d_x2_1_y),
    .z     (_row_0_d_x2_1_z)
  );
  MUL row_0_n_t3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:190:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:149:88
    .y     (_row_0_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:157:103
    .z     (_row_0_n_t3_1_z)
  );
  ADD row_0_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:191:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:186:103
    .y     (_delay_INT16_1_1866_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1875:113
    .z     (_row_0_n_x3_1_z)
  );
  dup_2 row_0_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:192:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:191:103
    .y     (_row_0_d_x3_1_y),
    .z     (_row_0_d_x3_1_z)
  );
  ADD row_0_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:193:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:167:103
    .y     (_row_0_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:176:103
    .z     (_row_0_n_x1_2_z)
  );
  dup_2 row_0_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:194:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:193:103
    .y     (_row_0_d_x1_2_y),
    .z     (_row_0_d_x1_2_z)
  );
  SUB row_0_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:195:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:167:103
    .y     (_row_0_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:176:103
    .z     (_row_0_n_x4_2_z)
  );
  dup_2 row_0_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:196:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:195:103
    .y     (_row_0_d_x4_2_y),
    .z     (_row_0_d_x4_2_z)
  );
  ADD row_0_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:197:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:170:103
    .y     (_row_0_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:179:103
    .z     (_row_0_n_x6_2_z)
  );
  dup_2 row_0_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:198:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:197:103
    .y     (_row_0_d_x6_2_y),
    .z     (_row_0_d_x6_2_z)
  );
  SUB row_0_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:199:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:170:103
    .y     (_row_0_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:179:103
    .z     (_row_0_n_x5_2_z)
  );
  dup_2 row_0_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:200:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:199:103
    .y     (_row_0_d_x5_2_y),
    .z     (_row_0_d_x5_2_z)
  );
  ADD row_0_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:201:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1867_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1876:113
    .y     (_row_0_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:192:103
    .z     (_row_0_n_x7_2_z)
  );
  dup_2 row_0_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:202:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:201:103
    .y     (_row_0_d_x7_2_y),
    .z     (_row_0_d_x7_2_z)
  );
  SUB row_0_n_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:203:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1868_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1877:113
    .y     (_row_0_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:192:103
    .z     (_row_0_n_x8_4_z)
  );
  dup_2 row_0_d_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:204:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:203:103
    .y     (_row_0_d_x8_4_y),
    .z     (_row_0_d_x8_4_z)
  );
  ADD row_0_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:205:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1869_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1878:113
    .y     (_row_0_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:189:103
    .z     (_row_0_n_x3_2_z)
  );
  dup_2 row_0_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:206:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:205:103
    .y     (_row_0_d_x3_2_y),
    .z     (_row_0_d_x3_2_z)
  );
  SUB row_0_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:207:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1870_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1879:113
    .y     (_row_0_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:189:103
    .z     (_row_0_n_x0_2_z)
  );
  dup_2 row_0_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:208:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:207:103
    .y     (_row_0_d_x0_2_y),
    .z     (_row_0_d_x0_2_z)
  );
  ADD row_0_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:209:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:196:103
    .y     (_row_0_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:200:103
    .z     (_row_0_n_u2_2_z)
  );
  MUL row_0_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:210:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:140:79
    .y     (_row_0_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:209:103
    .z     (_row_0_n_v2_2_z)
  );
  ADD row_0_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:211:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:210:103
    .y     (_row_0_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:138:79
    .z     (_row_0_n_w2_2_z)
  );
  SHR_8 row_0_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:212:86
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:211:103
    .z     (_row_0_n_x2_2_z)
  );
  dup_2 row_0_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:213:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:212:86
    .y     (_row_0_d_x2_2_y),
    .z     (_row_0_d_x2_2_z)
  );
  SUB row_0_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:214:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:196:103
    .y     (_row_0_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:200:103
    .z     (_row_0_n_u4_3_z)
  );
  MUL row_0_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:215:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:141:79
    .y     (_row_0_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:214:103
    .z     (_row_0_n_v4_3_z)
  );
  ADD row_0_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:216:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:215:103
    .y     (_row_0_n_c128_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:139:79
    .z     (_row_0_n_w4_3_z)
  );
  SHR_8 row_0_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:217:86
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:216:103
    .z     (_row_0_n_x4_3_z)
  );
  dup_2 row_0_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:218:103
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:217:86
    .y     (_row_0_d_x4_3_y),
    .z     (_row_0_d_x4_3_z)
  );
  ADD row_0_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:219:108
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:202:103
    .y     (_row_0_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:194:103
    .z     (_row_0_n_tmp_0_z)
  );
  SHR_8 row_0_n_shr_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:220:90
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:219:108
    .z     (_row_0_n_shr_0_z)
  );
  ADD row_0_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:221:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1871_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1880:113
    .y     (_row_0_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:213:103
    .z     (_row_0_n_tmp_1_z)
  );
  SHR_8 row_0_n_shr_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:222:90
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:221:108
    .z     (_row_0_n_shr_1_z)
  );
  ADD row_0_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:223:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1872_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1881:113
    .y     (_row_0_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:218:103
    .z     (_row_0_n_tmp_2_z)
  );
  SHR_8 row_0_n_shr_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:224:90
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:223:108
    .z     (_row_0_n_shr_2_z)
  );
  ADD row_0_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:225:108
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x8_4_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:204:103
    .y     (_row_0_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:198:103
    .z     (_row_0_n_tmp_3_z)
  );
  SHR_8 row_0_n_shr_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:226:90
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:225:108
    .z     (_row_0_n_shr_3_z)
  );
  SUB row_0_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:227:108
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:204:103
    .y     (_row_0_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:198:103
    .z     (_row_0_n_tmp_4_z)
  );
  SHR_8 row_0_n_shr_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:228:90
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:227:108
    .z     (_row_0_n_shr_4_z)
  );
  SUB row_0_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:229:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1873_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1882:113
    .y     (_row_0_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:218:103
    .z     (_row_0_n_tmp_5_z)
  );
  SHR_8 row_0_n_shr_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:230:90
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:229:108
    .z     (_row_0_n_shr_5_z)
  );
  SUB row_0_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:231:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1874_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1883:113
    .y     (_row_0_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:213:103
    .z     (_row_0_n_tmp_6_z)
  );
  SHR_8 row_0_n_shr_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:232:90
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:231:108
    .z     (_row_0_n_shr_6_z)
  );
  SUB row_0_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:233:108
    .clock (clock),
    .reset (reset),
    .x     (_row_0_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:202:103
    .y     (_row_0_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:194:103
    .z     (_row_0_n_tmp_7_z)
  );
  SHR_8 row_0_n_shr_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:234:90
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:233:108
    .z     (_row_0_n_shr_7_z)
  );
  C128 row_1_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:235:79
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_c128_0_value)
  );
  C128 row_1_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:236:79
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_c128_1_value)
  );
  C128 row_1_n_c128_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:237:79
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_c128_2_value)
  );
  C181 row_1_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:238:79
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_c181_0_value)
  );
  C181 row_1_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:239:79
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_c181_1_value)
  );
  W7 row_1_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:240:67
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_w7_value)
  );
  W1_sub_W7 row_1_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:241:88
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_w1_sub_w7_value)
  );
  W1_add_W7 row_1_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:242:88
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_w1_add_w7_value)
  );
  W3 row_1_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:243:67
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_w3_value)
  );
  W3_sub_W5 row_1_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:244:88
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_w3_sub_w5_value)
  );
  W3_add_W5 row_1_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:245:88
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_w3_add_w5_value)
  );
  W6 row_1_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:246:67
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_w6_value)
  );
  W2_sub_W6 row_1_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:247:88
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_w2_sub_w6_value)
  );
  W2_add_W6 row_1_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:248:88
    .clock (clock),
    .reset (reset),
    .value (_row_1_n_w2_add_w6_value)
  );
  SHL_11 row_1_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:249:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_1_4_x),
    .z     (_row_1_n_x1_0_z)
  );
  SHL_11 row_1_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:250:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_1_0_x),
    .z     (_row_1_n_t0_0_z)
  );
  ADD row_1_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:251:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:250:86
    .y     (_row_1_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:235:79
    .z     (_row_1_n_x0_0_z)
  );
  dup_2 row_1_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:252:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:251:103
    .y     (_row_1_d_x0_0_y),
    .z     (_row_1_d_x0_0_z)
  );
  dup_2 row_1_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:253:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:249:86
    .y     (_row_1_d_x1_0_y),
    .z     (_row_1_d_x1_0_z)
  );
  dup_2 row_1_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:254:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_1_6_x),
    .y     (_row_1_d_x2_0_y),
    .z     (_row_1_d_x2_0_z)
  );
  dup_2 row_1_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:255:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_1_2_x),
    .y     (_row_1_d_x3_0_y),
    .z     (_row_1_d_x3_0_z)
  );
  dup_2 row_1_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:256:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_1_1_x),
    .y     (_row_1_d_x4_0_y),
    .z     (_row_1_d_x4_0_z)
  );
  dup_2 row_1_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:257:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_1_7_x),
    .y     (_row_1_d_x5_0_y),
    .z     (_row_1_d_x5_0_z)
  );
  dup_2 row_1_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:258:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_1_5_x),
    .y     (_row_1_d_x6_0_y),
    .z     (_row_1_d_x6_0_z)
  );
  dup_2 row_1_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:259:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_1_3_x),
    .y     (_row_1_d_x7_0_y),
    .z     (_row_1_d_x7_0_z)
  );
  ADD row_1_n_t8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:260:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:256:103
    .y     (_row_1_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:257:103
    .z     (_row_1_n_t8_1_z)
  );
  MUL row_1_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:261:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:240:67
    .y     (_row_1_n_t8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:260:103
    .z     (_row_1_n_x8_1_z)
  );
  dup_2 row_1_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:262:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:261:103
    .y     (_row_1_d_x8_1_y),
    .z     (_row_1_d_x8_1_z)
  );
  MUL row_1_n_t4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:263:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:241:88
    .y     (_row_1_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:256:103
    .z     (_row_1_n_t4_1_z)
  );
  ADD row_1_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:264:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:262:103
    .y     (_delay_INT16_1_1875_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1884:113
    .z     (_row_1_n_x4_1_z)
  );
  dup_2 row_1_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:265:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:264:103
    .y     (_row_1_d_x4_1_y),
    .z     (_row_1_d_x4_1_z)
  );
  MUL row_1_n_t5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:266:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:242:88
    .y     (_row_1_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:257:103
    .z     (_row_1_n_t5_1_z)
  );
  SUB row_1_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:267:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:262:103
    .y     (_delay_INT16_1_1856_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1865:113
    .z     (_row_1_n_x5_1_z)
  );
  dup_2 row_1_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:268:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:267:103
    .y     (_row_1_d_x5_1_y),
    .z     (_row_1_d_x5_1_z)
  );
  ADD row_1_n_t8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:269:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:258:103
    .y     (_row_1_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:259:103
    .z     (_row_1_n_t8_2_z)
  );
  MUL row_1_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:270:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:243:67
    .y     (_row_1_n_t8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:269:103
    .z     (_row_1_n_x8_2_z)
  );
  dup_2 row_1_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:271:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:270:103
    .y     (_row_1_d_x8_2_y),
    .z     (_row_1_d_x8_2_z)
  );
  MUL row_1_n_t6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:272:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:244:88
    .y     (_row_1_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:258:103
    .z     (_row_1_n_t6_1_z)
  );
  SUB row_1_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:273:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:271:103
    .y     (_delay_INT16_1_1857_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1866:113
    .z     (_row_1_n_x6_1_z)
  );
  dup_2 row_1_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:274:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:273:103
    .y     (_row_1_d_x6_1_y),
    .z     (_row_1_d_x6_1_z)
  );
  MUL row_1_n_t7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:275:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:245:88
    .y     (_row_1_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:259:103
    .z     (_row_1_n_t7_1_z)
  );
  SUB row_1_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:276:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:271:103
    .y     (_delay_INT16_1_1858_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1867:113
    .z     (_row_1_n_x7_1_z)
  );
  dup_2 row_1_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:277:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:276:103
    .y     (_row_1_d_x7_1_y),
    .z     (_row_1_d_x7_1_z)
  );
  ADD row_1_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:278:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:252:103
    .y     (_delay_INT16_1_1878_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1887:113
    .z     (_row_1_n_x8_3_z)
  );
  dup_2 row_1_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:279:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:278:103
    .y     (_row_1_d_x8_3_y),
    .z     (_row_1_d_x8_3_z)
  );
  SUB row_1_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:280:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:252:103
    .y     (_delay_INT16_1_1879_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1888:113
    .z     (_row_1_n_x0_1_z)
  );
  dup_2 row_1_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:281:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:280:103
    .y     (_row_1_d_x0_1_y),
    .z     (_row_1_d_x0_1_z)
  );
  ADD row_1_n_t1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:282:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:255:103
    .y     (_row_1_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:254:103
    .z     (_row_1_n_t1_1_z)
  );
  MUL row_1_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:283:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:246:67
    .y     (_row_1_n_t1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:282:103
    .z     (_row_1_n_x1_1_z)
  );
  dup_2 row_1_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:284:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:283:103
    .y     (_row_1_d_x1_1_y),
    .z     (_row_1_d_x1_1_z)
  );
  MUL row_1_n_t2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:285:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:248:88
    .y     (_row_1_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:254:103
    .z     (_row_1_n_t2_1_z)
  );
  SUB row_1_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:286:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:284:103
    .y     (_delay_INT16_1_1880_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1889:113
    .z     (_row_1_n_x2_1_z)
  );
  dup_2 row_1_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:287:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:286:103
    .y     (_row_1_d_x2_1_y),
    .z     (_row_1_d_x2_1_z)
  );
  MUL row_1_n_t3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:288:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:247:88
    .y     (_row_1_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:255:103
    .z     (_row_1_n_t3_1_z)
  );
  ADD row_1_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:289:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:284:103
    .y     (_delay_INT16_1_1881_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1890:113
    .z     (_row_1_n_x3_1_z)
  );
  dup_2 row_1_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:290:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:289:103
    .y     (_row_1_d_x3_1_y),
    .z     (_row_1_d_x3_1_z)
  );
  ADD row_1_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:291:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:265:103
    .y     (_row_1_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:274:103
    .z     (_row_1_n_x1_2_z)
  );
  dup_2 row_1_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:292:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:291:103
    .y     (_row_1_d_x1_2_y),
    .z     (_row_1_d_x1_2_z)
  );
  SUB row_1_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:293:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:265:103
    .y     (_row_1_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:274:103
    .z     (_row_1_n_x4_2_z)
  );
  dup_2 row_1_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:294:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:293:103
    .y     (_row_1_d_x4_2_y),
    .z     (_row_1_d_x4_2_z)
  );
  ADD row_1_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:295:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:268:103
    .y     (_row_1_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:277:103
    .z     (_row_1_n_x6_2_z)
  );
  dup_2 row_1_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:296:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:295:103
    .y     (_row_1_d_x6_2_y),
    .z     (_row_1_d_x6_2_z)
  );
  SUB row_1_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:297:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:268:103
    .y     (_row_1_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:277:103
    .z     (_row_1_n_x5_2_z)
  );
  dup_2 row_1_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:298:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:297:103
    .y     (_row_1_d_x5_2_y),
    .z     (_row_1_d_x5_2_z)
  );
  ADD row_1_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:299:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1882_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1891:113
    .y     (_row_1_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:290:103
    .z     (_row_1_n_x7_2_z)
  );
  dup_2 row_1_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:300:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:299:103
    .y     (_row_1_d_x7_2_y),
    .z     (_row_1_d_x7_2_z)
  );
  SUB row_1_n_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:301:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1883_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1892:113
    .y     (_row_1_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:290:103
    .z     (_row_1_n_x8_4_z)
  );
  dup_2 row_1_d_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:302:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:301:103
    .y     (_row_1_d_x8_4_y),
    .z     (_row_1_d_x8_4_z)
  );
  ADD row_1_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:303:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1884_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1893:113
    .y     (_row_1_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:287:103
    .z     (_row_1_n_x3_2_z)
  );
  dup_2 row_1_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:304:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:303:103
    .y     (_row_1_d_x3_2_y),
    .z     (_row_1_d_x3_2_z)
  );
  SUB row_1_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:305:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1885_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1894:113
    .y     (_row_1_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:287:103
    .z     (_row_1_n_x0_2_z)
  );
  dup_2 row_1_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:306:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:305:103
    .y     (_row_1_d_x0_2_y),
    .z     (_row_1_d_x0_2_z)
  );
  ADD row_1_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:307:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:294:103
    .y     (_row_1_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:298:103
    .z     (_row_1_n_u2_2_z)
  );
  MUL row_1_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:308:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:238:79
    .y     (_row_1_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:307:103
    .z     (_row_1_n_v2_2_z)
  );
  ADD row_1_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:309:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:308:103
    .y     (_row_1_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:236:79
    .z     (_row_1_n_w2_2_z)
  );
  SHR_8 row_1_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:310:86
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:309:103
    .z     (_row_1_n_x2_2_z)
  );
  dup_2 row_1_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:311:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:310:86
    .y     (_row_1_d_x2_2_y),
    .z     (_row_1_d_x2_2_z)
  );
  SUB row_1_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:312:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:294:103
    .y     (_row_1_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:298:103
    .z     (_row_1_n_u4_3_z)
  );
  MUL row_1_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:313:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:239:79
    .y     (_row_1_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:312:103
    .z     (_row_1_n_v4_3_z)
  );
  ADD row_1_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:314:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:313:103
    .y     (_row_1_n_c128_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:237:79
    .z     (_row_1_n_w4_3_z)
  );
  SHR_8 row_1_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:315:86
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:314:103
    .z     (_row_1_n_x4_3_z)
  );
  dup_2 row_1_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:316:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:315:86
    .y     (_row_1_d_x4_3_y),
    .z     (_row_1_d_x4_3_z)
  );
  ADD row_1_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:317:108
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:300:103
    .y     (_row_1_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:292:103
    .z     (_row_1_n_tmp_0_z)
  );
  SHR_8 row_1_n_shr_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:318:90
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:317:108
    .z     (_row_1_n_shr_0_z)
  );
  ADD row_1_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:319:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1886_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1895:113
    .y     (_row_1_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:311:103
    .z     (_row_1_n_tmp_1_z)
  );
  SHR_8 row_1_n_shr_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:320:90
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:319:108
    .z     (_row_1_n_shr_1_z)
  );
  ADD row_1_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:321:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1887_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1896:113
    .y     (_row_1_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:316:103
    .z     (_row_1_n_tmp_2_z)
  );
  SHR_8 row_1_n_shr_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:322:90
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:321:108
    .z     (_row_1_n_shr_2_z)
  );
  ADD row_1_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:323:108
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x8_4_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:302:103
    .y     (_row_1_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:296:103
    .z     (_row_1_n_tmp_3_z)
  );
  SHR_8 row_1_n_shr_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:324:90
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:323:108
    .z     (_row_1_n_shr_3_z)
  );
  SUB row_1_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:325:108
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:302:103
    .y     (_row_1_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:296:103
    .z     (_row_1_n_tmp_4_z)
  );
  SHR_8 row_1_n_shr_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:326:90
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:325:108
    .z     (_row_1_n_shr_4_z)
  );
  SUB row_1_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:327:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1888_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1897:113
    .y     (_row_1_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:316:103
    .z     (_row_1_n_tmp_5_z)
  );
  SHR_8 row_1_n_shr_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:328:90
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:327:108
    .z     (_row_1_n_shr_5_z)
  );
  SUB row_1_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:329:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1889_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1898:113
    .y     (_row_1_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:311:103
    .z     (_row_1_n_tmp_6_z)
  );
  SHR_8 row_1_n_shr_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:330:90
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:329:108
    .z     (_row_1_n_shr_6_z)
  );
  SUB row_1_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:331:108
    .clock (clock),
    .reset (reset),
    .x     (_row_1_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:300:103
    .y     (_row_1_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:292:103
    .z     (_row_1_n_tmp_7_z)
  );
  SHR_8 row_1_n_shr_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:332:90
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:331:108
    .z     (_row_1_n_shr_7_z)
  );
  C128 row_2_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:333:79
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_c128_0_value)
  );
  C128 row_2_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:334:79
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_c128_1_value)
  );
  C128 row_2_n_c128_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:335:79
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_c128_2_value)
  );
  C181 row_2_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:336:79
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_c181_0_value)
  );
  C181 row_2_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:337:79
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_c181_1_value)
  );
  W7 row_2_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:338:67
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_w7_value)
  );
  W1_sub_W7 row_2_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:339:88
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_w1_sub_w7_value)
  );
  W1_add_W7 row_2_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:340:88
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_w1_add_w7_value)
  );
  W3 row_2_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:341:67
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_w3_value)
  );
  W3_sub_W5 row_2_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:342:88
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_w3_sub_w5_value)
  );
  W3_add_W5 row_2_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:343:88
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_w3_add_w5_value)
  );
  W6 row_2_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:344:67
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_w6_value)
  );
  W2_sub_W6 row_2_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:345:88
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_w2_sub_w6_value)
  );
  W2_add_W6 row_2_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:346:88
    .clock (clock),
    .reset (reset),
    .value (_row_2_n_w2_add_w6_value)
  );
  SHL_11 row_2_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:347:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_2_4_x),
    .z     (_row_2_n_x1_0_z)
  );
  SHL_11 row_2_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:348:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_2_0_x),
    .z     (_row_2_n_t0_0_z)
  );
  ADD row_2_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:349:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:348:86
    .y     (_row_2_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:333:79
    .z     (_row_2_n_x0_0_z)
  );
  dup_2 row_2_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:350:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:349:103
    .y     (_row_2_d_x0_0_y),
    .z     (_row_2_d_x0_0_z)
  );
  dup_2 row_2_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:351:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:347:86
    .y     (_row_2_d_x1_0_y),
    .z     (_row_2_d_x1_0_z)
  );
  dup_2 row_2_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:352:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_2_6_x),
    .y     (_row_2_d_x2_0_y),
    .z     (_row_2_d_x2_0_z)
  );
  dup_2 row_2_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:353:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_2_2_x),
    .y     (_row_2_d_x3_0_y),
    .z     (_row_2_d_x3_0_z)
  );
  dup_2 row_2_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:354:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_2_1_x),
    .y     (_row_2_d_x4_0_y),
    .z     (_row_2_d_x4_0_z)
  );
  dup_2 row_2_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:355:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_2_7_x),
    .y     (_row_2_d_x5_0_y),
    .z     (_row_2_d_x5_0_z)
  );
  dup_2 row_2_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:356:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_2_5_x),
    .y     (_row_2_d_x6_0_y),
    .z     (_row_2_d_x6_0_z)
  );
  dup_2 row_2_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:357:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_2_3_x),
    .y     (_row_2_d_x7_0_y),
    .z     (_row_2_d_x7_0_z)
  );
  ADD row_2_n_t8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:358:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:354:103
    .y     (_row_2_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:355:103
    .z     (_row_2_n_t8_1_z)
  );
  MUL row_2_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:359:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:338:67
    .y     (_row_2_n_t8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:358:103
    .z     (_row_2_n_x8_1_z)
  );
  dup_2 row_2_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:360:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:359:103
    .y     (_row_2_d_x8_1_y),
    .z     (_row_2_d_x8_1_z)
  );
  MUL row_2_n_t4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:361:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:339:88
    .y     (_row_2_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:354:103
    .z     (_row_2_n_t4_1_z)
  );
  ADD row_2_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:362:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:360:103
    .y     (_delay_INT16_1_1896_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1905:113
    .z     (_row_2_n_x4_1_z)
  );
  dup_2 row_2_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:363:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:362:103
    .y     (_row_2_d_x4_1_y),
    .z     (_row_2_d_x4_1_z)
  );
  MUL row_2_n_t5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:364:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:340:88
    .y     (_row_2_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:355:103
    .z     (_row_2_n_t5_1_z)
  );
  SUB row_2_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:365:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:360:103
    .y     (_delay_INT16_1_1897_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1906:113
    .z     (_row_2_n_x5_1_z)
  );
  dup_2 row_2_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:366:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:365:103
    .y     (_row_2_d_x5_1_y),
    .z     (_row_2_d_x5_1_z)
  );
  ADD row_2_n_t8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:367:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:356:103
    .y     (_row_2_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:357:103
    .z     (_row_2_n_t8_2_z)
  );
  MUL row_2_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:368:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:341:67
    .y     (_row_2_n_t8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:367:103
    .z     (_row_2_n_x8_2_z)
  );
  dup_2 row_2_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:369:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:368:103
    .y     (_row_2_d_x8_2_y),
    .z     (_row_2_d_x8_2_z)
  );
  MUL row_2_n_t6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:370:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:342:88
    .y     (_row_2_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:356:103
    .z     (_row_2_n_t6_1_z)
  );
  SUB row_2_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:371:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:369:103
    .y     (_delay_INT16_1_1898_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1907:113
    .z     (_row_2_n_x6_1_z)
  );
  dup_2 row_2_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:372:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:371:103
    .y     (_row_2_d_x6_1_y),
    .z     (_row_2_d_x6_1_z)
  );
  MUL row_2_n_t7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:373:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:343:88
    .y     (_row_2_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:357:103
    .z     (_row_2_n_t7_1_z)
  );
  SUB row_2_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:374:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:369:103
    .y     (_delay_INT16_1_1899_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1908:113
    .z     (_row_2_n_x7_1_z)
  );
  dup_2 row_2_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:375:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:374:103
    .y     (_row_2_d_x7_1_y),
    .z     (_row_2_d_x7_1_z)
  );
  ADD row_2_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:376:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:350:103
    .y     (_delay_INT16_1_1900_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1909:113
    .z     (_row_2_n_x8_3_z)
  );
  dup_2 row_2_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:377:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:376:103
    .y     (_row_2_d_x8_3_y),
    .z     (_row_2_d_x8_3_z)
  );
  SUB row_2_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:378:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:350:103
    .y     (_delay_INT16_1_1901_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1910:113
    .z     (_row_2_n_x0_1_z)
  );
  dup_2 row_2_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:379:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:378:103
    .y     (_row_2_d_x0_1_y),
    .z     (_row_2_d_x0_1_z)
  );
  ADD row_2_n_t1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:380:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:353:103
    .y     (_row_2_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:352:103
    .z     (_row_2_n_t1_1_z)
  );
  MUL row_2_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:381:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:344:67
    .y     (_row_2_n_t1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:380:103
    .z     (_row_2_n_x1_1_z)
  );
  dup_2 row_2_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:382:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:381:103
    .y     (_row_2_d_x1_1_y),
    .z     (_row_2_d_x1_1_z)
  );
  MUL row_2_n_t2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:383:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:346:88
    .y     (_row_2_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:352:103
    .z     (_row_2_n_t2_1_z)
  );
  SUB row_2_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:384:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:382:103
    .y     (_delay_INT16_1_1902_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1911:113
    .z     (_row_2_n_x2_1_z)
  );
  dup_2 row_2_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:385:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:384:103
    .y     (_row_2_d_x2_1_y),
    .z     (_row_2_d_x2_1_z)
  );
  MUL row_2_n_t3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:386:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:345:88
    .y     (_row_2_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:353:103
    .z     (_row_2_n_t3_1_z)
  );
  ADD row_2_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:387:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:382:103
    .y     (_delay_INT16_1_1903_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1912:113
    .z     (_row_2_n_x3_1_z)
  );
  dup_2 row_2_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:388:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:387:103
    .y     (_row_2_d_x3_1_y),
    .z     (_row_2_d_x3_1_z)
  );
  ADD row_2_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:389:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:363:103
    .y     (_row_2_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:372:103
    .z     (_row_2_n_x1_2_z)
  );
  dup_2 row_2_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:390:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:389:103
    .y     (_row_2_d_x1_2_y),
    .z     (_row_2_d_x1_2_z)
  );
  SUB row_2_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:391:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:363:103
    .y     (_row_2_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:372:103
    .z     (_row_2_n_x4_2_z)
  );
  dup_2 row_2_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:392:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:391:103
    .y     (_row_2_d_x4_2_y),
    .z     (_row_2_d_x4_2_z)
  );
  ADD row_2_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:393:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:366:103
    .y     (_row_2_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:375:103
    .z     (_row_2_n_x6_2_z)
  );
  dup_2 row_2_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:394:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:393:103
    .y     (_row_2_d_x6_2_y),
    .z     (_row_2_d_x6_2_z)
  );
  SUB row_2_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:395:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:366:103
    .y     (_row_2_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:375:103
    .z     (_row_2_n_x5_2_z)
  );
  dup_2 row_2_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:396:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:395:103
    .y     (_row_2_d_x5_2_y),
    .z     (_row_2_d_x5_2_z)
  );
  ADD row_2_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:397:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1904_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1913:113
    .y     (_row_2_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:388:103
    .z     (_row_2_n_x7_2_z)
  );
  dup_2 row_2_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:398:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:397:103
    .y     (_row_2_d_x7_2_y),
    .z     (_row_2_d_x7_2_z)
  );
  SUB row_2_n_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:399:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1905_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1914:113
    .y     (_row_2_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:388:103
    .z     (_row_2_n_x8_4_z)
  );
  dup_2 row_2_d_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:400:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:399:103
    .y     (_row_2_d_x8_4_y),
    .z     (_row_2_d_x8_4_z)
  );
  ADD row_2_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:401:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1906_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1915:113
    .y     (_row_2_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:385:103
    .z     (_row_2_n_x3_2_z)
  );
  dup_2 row_2_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:402:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:401:103
    .y     (_row_2_d_x3_2_y),
    .z     (_row_2_d_x3_2_z)
  );
  SUB row_2_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:403:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1907_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1916:113
    .y     (_row_2_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:385:103
    .z     (_row_2_n_x0_2_z)
  );
  dup_2 row_2_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:404:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:403:103
    .y     (_row_2_d_x0_2_y),
    .z     (_row_2_d_x0_2_z)
  );
  ADD row_2_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:405:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:392:103
    .y     (_row_2_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:396:103
    .z     (_row_2_n_u2_2_z)
  );
  MUL row_2_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:406:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:336:79
    .y     (_row_2_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:405:103
    .z     (_row_2_n_v2_2_z)
  );
  ADD row_2_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:407:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:406:103
    .y     (_row_2_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:334:79
    .z     (_row_2_n_w2_2_z)
  );
  SHR_8 row_2_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:408:86
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:407:103
    .z     (_row_2_n_x2_2_z)
  );
  dup_2 row_2_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:409:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:408:86
    .y     (_row_2_d_x2_2_y),
    .z     (_row_2_d_x2_2_z)
  );
  SUB row_2_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:410:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:392:103
    .y     (_row_2_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:396:103
    .z     (_row_2_n_u4_3_z)
  );
  MUL row_2_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:411:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:337:79
    .y     (_row_2_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:410:103
    .z     (_row_2_n_v4_3_z)
  );
  ADD row_2_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:412:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:411:103
    .y     (_row_2_n_c128_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:335:79
    .z     (_row_2_n_w4_3_z)
  );
  SHR_8 row_2_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:413:86
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:412:103
    .z     (_row_2_n_x4_3_z)
  );
  dup_2 row_2_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:414:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:413:86
    .y     (_row_2_d_x4_3_y),
    .z     (_row_2_d_x4_3_z)
  );
  ADD row_2_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:415:108
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:398:103
    .y     (_row_2_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:390:103
    .z     (_row_2_n_tmp_0_z)
  );
  SHR_8 row_2_n_shr_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:416:90
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:415:108
    .z     (_row_2_n_shr_0_z)
  );
  ADD row_2_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:417:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1908_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1917:113
    .y     (_row_2_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:409:103
    .z     (_row_2_n_tmp_1_z)
  );
  SHR_8 row_2_n_shr_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:418:90
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:417:108
    .z     (_row_2_n_shr_1_z)
  );
  ADD row_2_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:419:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1909_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1918:113
    .y     (_row_2_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:414:103
    .z     (_row_2_n_tmp_2_z)
  );
  SHR_8 row_2_n_shr_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:420:90
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:419:108
    .z     (_row_2_n_shr_2_z)
  );
  ADD row_2_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:421:108
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x8_4_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:400:103
    .y     (_row_2_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:394:103
    .z     (_row_2_n_tmp_3_z)
  );
  SHR_8 row_2_n_shr_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:422:90
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:421:108
    .z     (_row_2_n_shr_3_z)
  );
  SUB row_2_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:423:108
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:400:103
    .y     (_row_2_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:394:103
    .z     (_row_2_n_tmp_4_z)
  );
  SHR_8 row_2_n_shr_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:424:90
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:423:108
    .z     (_row_2_n_shr_4_z)
  );
  SUB row_2_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:425:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1910_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1919:113
    .y     (_row_2_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:414:103
    .z     (_row_2_n_tmp_5_z)
  );
  SHR_8 row_2_n_shr_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:426:90
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:425:108
    .z     (_row_2_n_shr_5_z)
  );
  SUB row_2_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:427:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1911_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1920:113
    .y     (_row_2_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:409:103
    .z     (_row_2_n_tmp_6_z)
  );
  SHR_8 row_2_n_shr_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:428:90
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:427:108
    .z     (_row_2_n_shr_6_z)
  );
  SUB row_2_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:429:108
    .clock (clock),
    .reset (reset),
    .x     (_row_2_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:398:103
    .y     (_row_2_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:390:103
    .z     (_row_2_n_tmp_7_z)
  );
  SHR_8 row_2_n_shr_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:430:90
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:429:108
    .z     (_row_2_n_shr_7_z)
  );
  C128 row_3_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:431:79
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_c128_0_value)
  );
  C128 row_3_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:432:79
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_c128_1_value)
  );
  C128 row_3_n_c128_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:433:79
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_c128_2_value)
  );
  C181 row_3_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:434:79
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_c181_0_value)
  );
  C181 row_3_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:435:79
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_c181_1_value)
  );
  W7 row_3_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:436:67
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_w7_value)
  );
  W1_sub_W7 row_3_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:437:88
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_w1_sub_w7_value)
  );
  W1_add_W7 row_3_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:438:88
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_w1_add_w7_value)
  );
  W3 row_3_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:439:67
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_w3_value)
  );
  W3_sub_W5 row_3_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:440:88
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_w3_sub_w5_value)
  );
  W3_add_W5 row_3_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:441:88
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_w3_add_w5_value)
  );
  W6 row_3_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:442:67
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_w6_value)
  );
  W2_sub_W6 row_3_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:443:88
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_w2_sub_w6_value)
  );
  W2_add_W6 row_3_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:444:88
    .clock (clock),
    .reset (reset),
    .value (_row_3_n_w2_add_w6_value)
  );
  SHL_11 row_3_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:445:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_3_4_x),
    .z     (_row_3_n_x1_0_z)
  );
  SHL_11 row_3_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:446:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_3_0_x),
    .z     (_row_3_n_t0_0_z)
  );
  ADD row_3_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:447:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:446:86
    .y     (_row_3_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:431:79
    .z     (_row_3_n_x0_0_z)
  );
  dup_2 row_3_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:448:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:447:103
    .y     (_row_3_d_x0_0_y),
    .z     (_row_3_d_x0_0_z)
  );
  dup_2 row_3_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:449:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:445:86
    .y     (_row_3_d_x1_0_y),
    .z     (_row_3_d_x1_0_z)
  );
  dup_2 row_3_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:450:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_3_6_x),
    .y     (_row_3_d_x2_0_y),
    .z     (_row_3_d_x2_0_z)
  );
  dup_2 row_3_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:451:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_3_2_x),
    .y     (_row_3_d_x3_0_y),
    .z     (_row_3_d_x3_0_z)
  );
  dup_2 row_3_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:452:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_3_1_x),
    .y     (_row_3_d_x4_0_y),
    .z     (_row_3_d_x4_0_z)
  );
  dup_2 row_3_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:453:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_3_7_x),
    .y     (_row_3_d_x5_0_y),
    .z     (_row_3_d_x5_0_z)
  );
  dup_2 row_3_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:454:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_3_5_x),
    .y     (_row_3_d_x6_0_y),
    .z     (_row_3_d_x6_0_z)
  );
  dup_2 row_3_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:455:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_3_3_x),
    .y     (_row_3_d_x7_0_y),
    .z     (_row_3_d_x7_0_z)
  );
  ADD row_3_n_t8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:456:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:452:103
    .y     (_row_3_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:453:103
    .z     (_row_3_n_t8_1_z)
  );
  MUL row_3_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:457:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:436:67
    .y     (_row_3_n_t8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:456:103
    .z     (_row_3_n_x8_1_z)
  );
  dup_2 row_3_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:458:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:457:103
    .y     (_row_3_d_x8_1_y),
    .z     (_row_3_d_x8_1_z)
  );
  MUL row_3_n_t4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:459:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:437:88
    .y     (_row_3_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:452:103
    .z     (_row_3_n_t4_1_z)
  );
  ADD row_3_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:460:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:458:103
    .y     (_delay_INT16_1_1912_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1921:113
    .z     (_row_3_n_x4_1_z)
  );
  dup_2 row_3_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:461:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:460:103
    .y     (_row_3_d_x4_1_y),
    .z     (_row_3_d_x4_1_z)
  );
  MUL row_3_n_t5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:462:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:438:88
    .y     (_row_3_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:453:103
    .z     (_row_3_n_t5_1_z)
  );
  SUB row_3_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:463:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:458:103
    .y     (_delay_INT16_1_1913_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1922:113
    .z     (_row_3_n_x5_1_z)
  );
  dup_2 row_3_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:464:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:463:103
    .y     (_row_3_d_x5_1_y),
    .z     (_row_3_d_x5_1_z)
  );
  ADD row_3_n_t8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:465:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:454:103
    .y     (_row_3_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:455:103
    .z     (_row_3_n_t8_2_z)
  );
  MUL row_3_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:466:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:439:67
    .y     (_row_3_n_t8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:465:103
    .z     (_row_3_n_x8_2_z)
  );
  dup_2 row_3_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:467:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:466:103
    .y     (_row_3_d_x8_2_y),
    .z     (_row_3_d_x8_2_z)
  );
  MUL row_3_n_t6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:468:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:440:88
    .y     (_row_3_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:454:103
    .z     (_row_3_n_t6_1_z)
  );
  SUB row_3_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:469:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:467:103
    .y     (_delay_INT16_1_1914_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1923:113
    .z     (_row_3_n_x6_1_z)
  );
  dup_2 row_3_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:470:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:469:103
    .y     (_row_3_d_x6_1_y),
    .z     (_row_3_d_x6_1_z)
  );
  MUL row_3_n_t7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:471:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:441:88
    .y     (_row_3_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:455:103
    .z     (_row_3_n_t7_1_z)
  );
  SUB row_3_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:472:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:467:103
    .y     (_delay_INT16_1_1915_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1924:113
    .z     (_row_3_n_x7_1_z)
  );
  dup_2 row_3_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:473:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:472:103
    .y     (_row_3_d_x7_1_y),
    .z     (_row_3_d_x7_1_z)
  );
  ADD row_3_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:474:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:448:103
    .y     (_delay_INT16_1_1916_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1925:113
    .z     (_row_3_n_x8_3_z)
  );
  dup_2 row_3_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:475:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:474:103
    .y     (_row_3_d_x8_3_y),
    .z     (_row_3_d_x8_3_z)
  );
  SUB row_3_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:476:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:448:103
    .y     (_delay_INT16_1_1917_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1926:113
    .z     (_row_3_n_x0_1_z)
  );
  dup_2 row_3_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:477:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:476:103
    .y     (_row_3_d_x0_1_y),
    .z     (_row_3_d_x0_1_z)
  );
  ADD row_3_n_t1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:478:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:451:103
    .y     (_row_3_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:450:103
    .z     (_row_3_n_t1_1_z)
  );
  MUL row_3_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:479:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:442:67
    .y     (_row_3_n_t1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:478:103
    .z     (_row_3_n_x1_1_z)
  );
  dup_2 row_3_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:480:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:479:103
    .y     (_row_3_d_x1_1_y),
    .z     (_row_3_d_x1_1_z)
  );
  MUL row_3_n_t2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:481:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:444:88
    .y     (_row_3_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:450:103
    .z     (_row_3_n_t2_1_z)
  );
  SUB row_3_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:482:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:480:103
    .y     (_delay_INT16_1_1918_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1927:113
    .z     (_row_3_n_x2_1_z)
  );
  dup_2 row_3_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:483:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:482:103
    .y     (_row_3_d_x2_1_y),
    .z     (_row_3_d_x2_1_z)
  );
  MUL row_3_n_t3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:484:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:443:88
    .y     (_row_3_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:451:103
    .z     (_row_3_n_t3_1_z)
  );
  ADD row_3_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:485:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:480:103
    .y     (_delay_INT16_1_1919_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1928:113
    .z     (_row_3_n_x3_1_z)
  );
  dup_2 row_3_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:486:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:485:103
    .y     (_row_3_d_x3_1_y),
    .z     (_row_3_d_x3_1_z)
  );
  ADD row_3_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:487:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:461:103
    .y     (_row_3_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:470:103
    .z     (_row_3_n_x1_2_z)
  );
  dup_2 row_3_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:488:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:487:103
    .y     (_row_3_d_x1_2_y),
    .z     (_row_3_d_x1_2_z)
  );
  SUB row_3_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:489:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:461:103
    .y     (_row_3_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:470:103
    .z     (_row_3_n_x4_2_z)
  );
  dup_2 row_3_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:490:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:489:103
    .y     (_row_3_d_x4_2_y),
    .z     (_row_3_d_x4_2_z)
  );
  ADD row_3_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:491:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:464:103
    .y     (_row_3_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:473:103
    .z     (_row_3_n_x6_2_z)
  );
  dup_2 row_3_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:492:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:491:103
    .y     (_row_3_d_x6_2_y),
    .z     (_row_3_d_x6_2_z)
  );
  SUB row_3_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:493:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:464:103
    .y     (_row_3_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:473:103
    .z     (_row_3_n_x5_2_z)
  );
  dup_2 row_3_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:494:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:493:103
    .y     (_row_3_d_x5_2_y),
    .z     (_row_3_d_x5_2_z)
  );
  ADD row_3_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:495:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1920_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1929:113
    .y     (_row_3_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:486:103
    .z     (_row_3_n_x7_2_z)
  );
  dup_2 row_3_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:496:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:495:103
    .y     (_row_3_d_x7_2_y),
    .z     (_row_3_d_x7_2_z)
  );
  SUB row_3_n_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:497:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1921_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1930:113
    .y     (_row_3_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:486:103
    .z     (_row_3_n_x8_4_z)
  );
  dup_2 row_3_d_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:498:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:497:103
    .y     (_row_3_d_x8_4_y),
    .z     (_row_3_d_x8_4_z)
  );
  ADD row_3_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:499:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1922_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1931:113
    .y     (_row_3_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:483:103
    .z     (_row_3_n_x3_2_z)
  );
  dup_2 row_3_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:500:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:499:103
    .y     (_row_3_d_x3_2_y),
    .z     (_row_3_d_x3_2_z)
  );
  SUB row_3_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:501:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1923_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1932:113
    .y     (_row_3_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:483:103
    .z     (_row_3_n_x0_2_z)
  );
  dup_2 row_3_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:502:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:501:103
    .y     (_row_3_d_x0_2_y),
    .z     (_row_3_d_x0_2_z)
  );
  ADD row_3_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:503:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:490:103
    .y     (_row_3_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:494:103
    .z     (_row_3_n_u2_2_z)
  );
  MUL row_3_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:504:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:434:79
    .y     (_row_3_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:503:103
    .z     (_row_3_n_v2_2_z)
  );
  ADD row_3_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:505:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:504:103
    .y     (_row_3_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:432:79
    .z     (_row_3_n_w2_2_z)
  );
  SHR_8 row_3_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:506:86
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:505:103
    .z     (_row_3_n_x2_2_z)
  );
  dup_2 row_3_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:507:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:506:86
    .y     (_row_3_d_x2_2_y),
    .z     (_row_3_d_x2_2_z)
  );
  SUB row_3_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:508:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:490:103
    .y     (_row_3_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:494:103
    .z     (_row_3_n_u4_3_z)
  );
  MUL row_3_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:509:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:435:79
    .y     (_row_3_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:508:103
    .z     (_row_3_n_v4_3_z)
  );
  ADD row_3_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:510:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:509:103
    .y     (_row_3_n_c128_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:433:79
    .z     (_row_3_n_w4_3_z)
  );
  SHR_8 row_3_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:511:86
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:510:103
    .z     (_row_3_n_x4_3_z)
  );
  dup_2 row_3_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:512:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:511:86
    .y     (_row_3_d_x4_3_y),
    .z     (_row_3_d_x4_3_z)
  );
  ADD row_3_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:513:108
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:496:103
    .y     (_row_3_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:488:103
    .z     (_row_3_n_tmp_0_z)
  );
  SHR_8 row_3_n_shr_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:514:90
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:513:108
    .z     (_row_3_n_shr_0_z)
  );
  ADD row_3_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:515:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1924_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1933:113
    .y     (_row_3_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:507:103
    .z     (_row_3_n_tmp_1_z)
  );
  SHR_8 row_3_n_shr_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:516:90
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:515:108
    .z     (_row_3_n_shr_1_z)
  );
  ADD row_3_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:517:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1925_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1934:113
    .y     (_row_3_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:512:103
    .z     (_row_3_n_tmp_2_z)
  );
  SHR_8 row_3_n_shr_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:518:90
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:517:108
    .z     (_row_3_n_shr_2_z)
  );
  ADD row_3_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:519:108
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x8_4_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:498:103
    .y     (_row_3_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:492:103
    .z     (_row_3_n_tmp_3_z)
  );
  SHR_8 row_3_n_shr_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:520:90
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:519:108
    .z     (_row_3_n_shr_3_z)
  );
  SUB row_3_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:521:108
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:498:103
    .y     (_row_3_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:492:103
    .z     (_row_3_n_tmp_4_z)
  );
  SHR_8 row_3_n_shr_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:522:90
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:521:108
    .z     (_row_3_n_shr_4_z)
  );
  SUB row_3_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:523:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1876_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1885:113
    .y     (_row_3_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:512:103
    .z     (_row_3_n_tmp_5_z)
  );
  SHR_8 row_3_n_shr_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:524:90
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:523:108
    .z     (_row_3_n_shr_5_z)
  );
  SUB row_3_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:525:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1877_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1886:113
    .y     (_row_3_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:507:103
    .z     (_row_3_n_tmp_6_z)
  );
  SHR_8 row_3_n_shr_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:526:90
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:525:108
    .z     (_row_3_n_shr_6_z)
  );
  SUB row_3_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:527:108
    .clock (clock),
    .reset (reset),
    .x     (_row_3_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:496:103
    .y     (_row_3_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:488:103
    .z     (_row_3_n_tmp_7_z)
  );
  SHR_8 row_3_n_shr_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:528:90
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:527:108
    .z     (_row_3_n_shr_7_z)
  );
  C128 row_4_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:529:79
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_c128_0_value)
  );
  C128 row_4_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:530:79
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_c128_1_value)
  );
  C128 row_4_n_c128_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:531:79
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_c128_2_value)
  );
  C181 row_4_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:532:79
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_c181_0_value)
  );
  C181 row_4_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:533:79
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_c181_1_value)
  );
  W7 row_4_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:534:67
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_w7_value)
  );
  W1_sub_W7 row_4_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:535:88
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_w1_sub_w7_value)
  );
  W1_add_W7 row_4_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:536:88
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_w1_add_w7_value)
  );
  W3 row_4_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:537:67
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_w3_value)
  );
  W3_sub_W5 row_4_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:538:88
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_w3_sub_w5_value)
  );
  W3_add_W5 row_4_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:539:88
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_w3_add_w5_value)
  );
  W6 row_4_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:540:67
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_w6_value)
  );
  W2_sub_W6 row_4_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:541:88
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_w2_sub_w6_value)
  );
  W2_add_W6 row_4_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:542:88
    .clock (clock),
    .reset (reset),
    .value (_row_4_n_w2_add_w6_value)
  );
  SHL_11 row_4_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:543:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_4_4_x),
    .z     (_row_4_n_x1_0_z)
  );
  SHL_11 row_4_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:544:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_4_0_x),
    .z     (_row_4_n_t0_0_z)
  );
  ADD row_4_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:545:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:544:86
    .y     (_row_4_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:529:79
    .z     (_row_4_n_x0_0_z)
  );
  dup_2 row_4_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:546:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:545:103
    .y     (_row_4_d_x0_0_y),
    .z     (_row_4_d_x0_0_z)
  );
  dup_2 row_4_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:547:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:543:86
    .y     (_row_4_d_x1_0_y),
    .z     (_row_4_d_x1_0_z)
  );
  dup_2 row_4_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:548:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_4_6_x),
    .y     (_row_4_d_x2_0_y),
    .z     (_row_4_d_x2_0_z)
  );
  dup_2 row_4_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:549:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_4_2_x),
    .y     (_row_4_d_x3_0_y),
    .z     (_row_4_d_x3_0_z)
  );
  dup_2 row_4_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:550:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_4_1_x),
    .y     (_row_4_d_x4_0_y),
    .z     (_row_4_d_x4_0_z)
  );
  dup_2 row_4_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:551:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_4_7_x),
    .y     (_row_4_d_x5_0_y),
    .z     (_row_4_d_x5_0_z)
  );
  dup_2 row_4_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:552:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_4_5_x),
    .y     (_row_4_d_x6_0_y),
    .z     (_row_4_d_x6_0_z)
  );
  dup_2 row_4_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:553:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_4_3_x),
    .y     (_row_4_d_x7_0_y),
    .z     (_row_4_d_x7_0_z)
  );
  ADD row_4_n_t8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:554:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:550:103
    .y     (_row_4_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:551:103
    .z     (_row_4_n_t8_1_z)
  );
  MUL row_4_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:555:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:534:67
    .y     (_row_4_n_t8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:554:103
    .z     (_row_4_n_x8_1_z)
  );
  dup_2 row_4_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:556:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:555:103
    .y     (_row_4_d_x8_1_y),
    .z     (_row_4_d_x8_1_z)
  );
  MUL row_4_n_t4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:557:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:535:88
    .y     (_row_4_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:550:103
    .z     (_row_4_n_t4_1_z)
  );
  ADD row_4_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:558:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:556:103
    .y     (_delay_INT16_1_1926_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1935:113
    .z     (_row_4_n_x4_1_z)
  );
  dup_2 row_4_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:559:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:558:103
    .y     (_row_4_d_x4_1_y),
    .z     (_row_4_d_x4_1_z)
  );
  MUL row_4_n_t5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:560:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:536:88
    .y     (_row_4_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:551:103
    .z     (_row_4_n_t5_1_z)
  );
  SUB row_4_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:561:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:556:103
    .y     (_delay_INT16_1_1927_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1936:113
    .z     (_row_4_n_x5_1_z)
  );
  dup_2 row_4_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:562:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:561:103
    .y     (_row_4_d_x5_1_y),
    .z     (_row_4_d_x5_1_z)
  );
  ADD row_4_n_t8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:563:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:552:103
    .y     (_row_4_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:553:103
    .z     (_row_4_n_t8_2_z)
  );
  MUL row_4_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:564:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:537:67
    .y     (_row_4_n_t8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:563:103
    .z     (_row_4_n_x8_2_z)
  );
  dup_2 row_4_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:565:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:564:103
    .y     (_row_4_d_x8_2_y),
    .z     (_row_4_d_x8_2_z)
  );
  MUL row_4_n_t6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:566:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:538:88
    .y     (_row_4_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:552:103
    .z     (_row_4_n_t6_1_z)
  );
  SUB row_4_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:567:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:565:103
    .y     (_delay_INT16_1_1928_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1937:113
    .z     (_row_4_n_x6_1_z)
  );
  dup_2 row_4_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:568:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:567:103
    .y     (_row_4_d_x6_1_y),
    .z     (_row_4_d_x6_1_z)
  );
  MUL row_4_n_t7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:569:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:539:88
    .y     (_row_4_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:553:103
    .z     (_row_4_n_t7_1_z)
  );
  SUB row_4_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:570:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:565:103
    .y     (_delay_INT16_1_1929_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1938:113
    .z     (_row_4_n_x7_1_z)
  );
  dup_2 row_4_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:571:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:570:103
    .y     (_row_4_d_x7_1_y),
    .z     (_row_4_d_x7_1_z)
  );
  ADD row_4_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:572:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:546:103
    .y     (_delay_INT16_1_1930_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1939:113
    .z     (_row_4_n_x8_3_z)
  );
  dup_2 row_4_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:573:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:572:103
    .y     (_row_4_d_x8_3_y),
    .z     (_row_4_d_x8_3_z)
  );
  SUB row_4_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:574:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:546:103
    .y     (_delay_INT16_1_1931_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1940:113
    .z     (_row_4_n_x0_1_z)
  );
  dup_2 row_4_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:575:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:574:103
    .y     (_row_4_d_x0_1_y),
    .z     (_row_4_d_x0_1_z)
  );
  ADD row_4_n_t1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:576:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:549:103
    .y     (_row_4_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:548:103
    .z     (_row_4_n_t1_1_z)
  );
  MUL row_4_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:577:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:540:67
    .y     (_row_4_n_t1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:576:103
    .z     (_row_4_n_x1_1_z)
  );
  dup_2 row_4_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:578:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:577:103
    .y     (_row_4_d_x1_1_y),
    .z     (_row_4_d_x1_1_z)
  );
  MUL row_4_n_t2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:579:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:542:88
    .y     (_row_4_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:548:103
    .z     (_row_4_n_t2_1_z)
  );
  SUB row_4_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:580:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:578:103
    .y     (_delay_INT16_1_1932_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1941:113
    .z     (_row_4_n_x2_1_z)
  );
  dup_2 row_4_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:581:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:580:103
    .y     (_row_4_d_x2_1_y),
    .z     (_row_4_d_x2_1_z)
  );
  MUL row_4_n_t3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:582:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:541:88
    .y     (_row_4_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:549:103
    .z     (_row_4_n_t3_1_z)
  );
  ADD row_4_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:583:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:578:103
    .y     (_delay_INT16_1_1933_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1942:113
    .z     (_row_4_n_x3_1_z)
  );
  dup_2 row_4_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:584:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:583:103
    .y     (_row_4_d_x3_1_y),
    .z     (_row_4_d_x3_1_z)
  );
  ADD row_4_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:585:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:559:103
    .y     (_row_4_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:568:103
    .z     (_row_4_n_x1_2_z)
  );
  dup_2 row_4_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:586:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:585:103
    .y     (_row_4_d_x1_2_y),
    .z     (_row_4_d_x1_2_z)
  );
  SUB row_4_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:587:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:559:103
    .y     (_row_4_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:568:103
    .z     (_row_4_n_x4_2_z)
  );
  dup_2 row_4_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:588:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:587:103
    .y     (_row_4_d_x4_2_y),
    .z     (_row_4_d_x4_2_z)
  );
  ADD row_4_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:589:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:562:103
    .y     (_row_4_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:571:103
    .z     (_row_4_n_x6_2_z)
  );
  dup_2 row_4_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:590:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:589:103
    .y     (_row_4_d_x6_2_y),
    .z     (_row_4_d_x6_2_z)
  );
  SUB row_4_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:591:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:562:103
    .y     (_row_4_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:571:103
    .z     (_row_4_n_x5_2_z)
  );
  dup_2 row_4_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:592:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:591:103
    .y     (_row_4_d_x5_2_y),
    .z     (_row_4_d_x5_2_z)
  );
  ADD row_4_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:593:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1934_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1943:113
    .y     (_row_4_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:584:103
    .z     (_row_4_n_x7_2_z)
  );
  dup_2 row_4_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:594:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:593:103
    .y     (_row_4_d_x7_2_y),
    .z     (_row_4_d_x7_2_z)
  );
  SUB row_4_n_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:595:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1935_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1944:113
    .y     (_row_4_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:584:103
    .z     (_row_4_n_x8_4_z)
  );
  dup_2 row_4_d_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:596:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:595:103
    .y     (_row_4_d_x8_4_y),
    .z     (_row_4_d_x8_4_z)
  );
  ADD row_4_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:597:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1936_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1945:113
    .y     (_row_4_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:581:103
    .z     (_row_4_n_x3_2_z)
  );
  dup_2 row_4_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:598:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:597:103
    .y     (_row_4_d_x3_2_y),
    .z     (_row_4_d_x3_2_z)
  );
  SUB row_4_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:599:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1937_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1946:113
    .y     (_row_4_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:581:103
    .z     (_row_4_n_x0_2_z)
  );
  dup_2 row_4_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:600:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:599:103
    .y     (_row_4_d_x0_2_y),
    .z     (_row_4_d_x0_2_z)
  );
  ADD row_4_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:601:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:588:103
    .y     (_row_4_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:592:103
    .z     (_row_4_n_u2_2_z)
  );
  MUL row_4_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:602:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:532:79
    .y     (_row_4_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:601:103
    .z     (_row_4_n_v2_2_z)
  );
  ADD row_4_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:603:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:602:103
    .y     (_row_4_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:530:79
    .z     (_row_4_n_w2_2_z)
  );
  SHR_8 row_4_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:604:86
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:603:103
    .z     (_row_4_n_x2_2_z)
  );
  dup_2 row_4_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:605:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:604:86
    .y     (_row_4_d_x2_2_y),
    .z     (_row_4_d_x2_2_z)
  );
  SUB row_4_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:606:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:588:103
    .y     (_row_4_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:592:103
    .z     (_row_4_n_u4_3_z)
  );
  MUL row_4_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:607:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:533:79
    .y     (_row_4_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:606:103
    .z     (_row_4_n_v4_3_z)
  );
  ADD row_4_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:608:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:607:103
    .y     (_row_4_n_c128_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:531:79
    .z     (_row_4_n_w4_3_z)
  );
  SHR_8 row_4_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:609:86
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:608:103
    .z     (_row_4_n_x4_3_z)
  );
  dup_2 row_4_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:610:103
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:609:86
    .y     (_row_4_d_x4_3_y),
    .z     (_row_4_d_x4_3_z)
  );
  ADD row_4_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:611:108
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:594:103
    .y     (_row_4_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:586:103
    .z     (_row_4_n_tmp_0_z)
  );
  SHR_8 row_4_n_shr_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:612:90
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:611:108
    .z     (_row_4_n_shr_0_z)
  );
  ADD row_4_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:613:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1938_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1947:113
    .y     (_row_4_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:605:103
    .z     (_row_4_n_tmp_1_z)
  );
  SHR_8 row_4_n_shr_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:614:90
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:613:108
    .z     (_row_4_n_shr_1_z)
  );
  ADD row_4_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:615:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1939_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1948:113
    .y     (_row_4_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:610:103
    .z     (_row_4_n_tmp_2_z)
  );
  SHR_8 row_4_n_shr_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:616:90
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:615:108
    .z     (_row_4_n_shr_2_z)
  );
  ADD row_4_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:617:108
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x8_4_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:596:103
    .y     (_row_4_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:590:103
    .z     (_row_4_n_tmp_3_z)
  );
  SHR_8 row_4_n_shr_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:618:90
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:617:108
    .z     (_row_4_n_shr_3_z)
  );
  SUB row_4_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:619:108
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:596:103
    .y     (_row_4_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:590:103
    .z     (_row_4_n_tmp_4_z)
  );
  SHR_8 row_4_n_shr_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:620:90
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:619:108
    .z     (_row_4_n_shr_4_z)
  );
  SUB row_4_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:621:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1940_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1949:113
    .y     (_row_4_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:610:103
    .z     (_row_4_n_tmp_5_z)
  );
  SHR_8 row_4_n_shr_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:622:90
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:621:108
    .z     (_row_4_n_shr_5_z)
  );
  SUB row_4_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:623:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1941_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1950:113
    .y     (_row_4_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:605:103
    .z     (_row_4_n_tmp_6_z)
  );
  SHR_8 row_4_n_shr_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:624:90
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:623:108
    .z     (_row_4_n_shr_6_z)
  );
  SUB row_4_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:625:108
    .clock (clock),
    .reset (reset),
    .x     (_row_4_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:594:103
    .y     (_row_4_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:586:103
    .z     (_row_4_n_tmp_7_z)
  );
  SHR_8 row_4_n_shr_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:626:90
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:625:108
    .z     (_row_4_n_shr_7_z)
  );
  C128 row_5_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:627:79
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_c128_0_value)
  );
  C128 row_5_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:628:79
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_c128_1_value)
  );
  C128 row_5_n_c128_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:629:79
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_c128_2_value)
  );
  C181 row_5_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:630:79
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_c181_0_value)
  );
  C181 row_5_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:631:79
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_c181_1_value)
  );
  W7 row_5_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:632:67
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_w7_value)
  );
  W1_sub_W7 row_5_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:633:88
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_w1_sub_w7_value)
  );
  W1_add_W7 row_5_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:634:88
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_w1_add_w7_value)
  );
  W3 row_5_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:635:67
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_w3_value)
  );
  W3_sub_W5 row_5_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:636:88
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_w3_sub_w5_value)
  );
  W3_add_W5 row_5_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:637:88
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_w3_add_w5_value)
  );
  W6 row_5_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:638:67
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_w6_value)
  );
  W2_sub_W6 row_5_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:639:88
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_w2_sub_w6_value)
  );
  W2_add_W6 row_5_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:640:88
    .clock (clock),
    .reset (reset),
    .value (_row_5_n_w2_add_w6_value)
  );
  SHL_11 row_5_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:641:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_5_4_x),
    .z     (_row_5_n_x1_0_z)
  );
  SHL_11 row_5_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:642:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_5_0_x),
    .z     (_row_5_n_t0_0_z)
  );
  ADD row_5_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:643:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:642:86
    .y     (_row_5_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:627:79
    .z     (_row_5_n_x0_0_z)
  );
  dup_2 row_5_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:644:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:643:103
    .y     (_row_5_d_x0_0_y),
    .z     (_row_5_d_x0_0_z)
  );
  dup_2 row_5_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:645:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:641:86
    .y     (_row_5_d_x1_0_y),
    .z     (_row_5_d_x1_0_z)
  );
  dup_2 row_5_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:646:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_5_6_x),
    .y     (_row_5_d_x2_0_y),
    .z     (_row_5_d_x2_0_z)
  );
  dup_2 row_5_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:647:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_5_2_x),
    .y     (_row_5_d_x3_0_y),
    .z     (_row_5_d_x3_0_z)
  );
  dup_2 row_5_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:648:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_5_1_x),
    .y     (_row_5_d_x4_0_y),
    .z     (_row_5_d_x4_0_z)
  );
  dup_2 row_5_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:649:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_5_7_x),
    .y     (_row_5_d_x5_0_y),
    .z     (_row_5_d_x5_0_z)
  );
  dup_2 row_5_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:650:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_5_5_x),
    .y     (_row_5_d_x6_0_y),
    .z     (_row_5_d_x6_0_z)
  );
  dup_2 row_5_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:651:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_5_3_x),
    .y     (_row_5_d_x7_0_y),
    .z     (_row_5_d_x7_0_z)
  );
  ADD row_5_n_t8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:652:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:648:103
    .y     (_row_5_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:649:103
    .z     (_row_5_n_t8_1_z)
  );
  MUL row_5_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:653:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:632:67
    .y     (_row_5_n_t8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:652:103
    .z     (_row_5_n_x8_1_z)
  );
  dup_2 row_5_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:654:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:653:103
    .y     (_row_5_d_x8_1_y),
    .z     (_row_5_d_x8_1_z)
  );
  MUL row_5_n_t4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:655:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:633:88
    .y     (_row_5_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:648:103
    .z     (_row_5_n_t4_1_z)
  );
  ADD row_5_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:656:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:654:103
    .y     (_delay_INT16_1_1942_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1951:113
    .z     (_row_5_n_x4_1_z)
  );
  dup_2 row_5_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:657:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:656:103
    .y     (_row_5_d_x4_1_y),
    .z     (_row_5_d_x4_1_z)
  );
  MUL row_5_n_t5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:658:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:634:88
    .y     (_row_5_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:649:103
    .z     (_row_5_n_t5_1_z)
  );
  SUB row_5_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:659:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:654:103
    .y     (_delay_INT16_1_1943_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1952:113
    .z     (_row_5_n_x5_1_z)
  );
  dup_2 row_5_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:660:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:659:103
    .y     (_row_5_d_x5_1_y),
    .z     (_row_5_d_x5_1_z)
  );
  ADD row_5_n_t8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:661:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:650:103
    .y     (_row_5_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:651:103
    .z     (_row_5_n_t8_2_z)
  );
  MUL row_5_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:662:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:635:67
    .y     (_row_5_n_t8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:661:103
    .z     (_row_5_n_x8_2_z)
  );
  dup_2 row_5_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:663:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:662:103
    .y     (_row_5_d_x8_2_y),
    .z     (_row_5_d_x8_2_z)
  );
  MUL row_5_n_t6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:664:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:636:88
    .y     (_row_5_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:650:103
    .z     (_row_5_n_t6_1_z)
  );
  SUB row_5_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:665:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:663:103
    .y     (_delay_INT16_1_1944_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1953:113
    .z     (_row_5_n_x6_1_z)
  );
  dup_2 row_5_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:666:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:665:103
    .y     (_row_5_d_x6_1_y),
    .z     (_row_5_d_x6_1_z)
  );
  MUL row_5_n_t7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:667:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:637:88
    .y     (_row_5_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:651:103
    .z     (_row_5_n_t7_1_z)
  );
  SUB row_5_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:668:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:663:103
    .y     (_delay_INT16_1_1945_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1954:113
    .z     (_row_5_n_x7_1_z)
  );
  dup_2 row_5_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:669:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:668:103
    .y     (_row_5_d_x7_1_y),
    .z     (_row_5_d_x7_1_z)
  );
  ADD row_5_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:670:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:644:103
    .y     (_delay_INT16_1_1946_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1955:113
    .z     (_row_5_n_x8_3_z)
  );
  dup_2 row_5_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:671:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:670:103
    .y     (_row_5_d_x8_3_y),
    .z     (_row_5_d_x8_3_z)
  );
  SUB row_5_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:672:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:644:103
    .y     (_delay_INT16_1_1947_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1956:113
    .z     (_row_5_n_x0_1_z)
  );
  dup_2 row_5_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:673:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:672:103
    .y     (_row_5_d_x0_1_y),
    .z     (_row_5_d_x0_1_z)
  );
  ADD row_5_n_t1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:674:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:647:103
    .y     (_row_5_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:646:103
    .z     (_row_5_n_t1_1_z)
  );
  MUL row_5_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:675:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:638:67
    .y     (_row_5_n_t1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:674:103
    .z     (_row_5_n_x1_1_z)
  );
  dup_2 row_5_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:676:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:675:103
    .y     (_row_5_d_x1_1_y),
    .z     (_row_5_d_x1_1_z)
  );
  MUL row_5_n_t2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:677:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:640:88
    .y     (_row_5_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:646:103
    .z     (_row_5_n_t2_1_z)
  );
  SUB row_5_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:678:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:676:103
    .y     (_delay_INT16_1_1948_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1957:113
    .z     (_row_5_n_x2_1_z)
  );
  dup_2 row_5_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:679:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:678:103
    .y     (_row_5_d_x2_1_y),
    .z     (_row_5_d_x2_1_z)
  );
  MUL row_5_n_t3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:680:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:639:88
    .y     (_row_5_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:647:103
    .z     (_row_5_n_t3_1_z)
  );
  ADD row_5_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:681:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:676:103
    .y     (_delay_INT16_1_1949_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1958:113
    .z     (_row_5_n_x3_1_z)
  );
  dup_2 row_5_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:682:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:681:103
    .y     (_row_5_d_x3_1_y),
    .z     (_row_5_d_x3_1_z)
  );
  ADD row_5_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:683:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:657:103
    .y     (_row_5_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:666:103
    .z     (_row_5_n_x1_2_z)
  );
  dup_2 row_5_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:684:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:683:103
    .y     (_row_5_d_x1_2_y),
    .z     (_row_5_d_x1_2_z)
  );
  SUB row_5_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:685:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:657:103
    .y     (_row_5_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:666:103
    .z     (_row_5_n_x4_2_z)
  );
  dup_2 row_5_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:686:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:685:103
    .y     (_row_5_d_x4_2_y),
    .z     (_row_5_d_x4_2_z)
  );
  ADD row_5_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:687:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:660:103
    .y     (_row_5_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:669:103
    .z     (_row_5_n_x6_2_z)
  );
  dup_2 row_5_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:688:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:687:103
    .y     (_row_5_d_x6_2_y),
    .z     (_row_5_d_x6_2_z)
  );
  SUB row_5_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:689:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:660:103
    .y     (_row_5_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:669:103
    .z     (_row_5_n_x5_2_z)
  );
  dup_2 row_5_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:690:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:689:103
    .y     (_row_5_d_x5_2_y),
    .z     (_row_5_d_x5_2_z)
  );
  ADD row_5_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:691:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1950_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1959:113
    .y     (_row_5_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:682:103
    .z     (_row_5_n_x7_2_z)
  );
  dup_2 row_5_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:692:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:691:103
    .y     (_row_5_d_x7_2_y),
    .z     (_row_5_d_x7_2_z)
  );
  SUB row_5_n_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:693:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1951_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1960:113
    .y     (_row_5_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:682:103
    .z     (_row_5_n_x8_4_z)
  );
  dup_2 row_5_d_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:694:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:693:103
    .y     (_row_5_d_x8_4_y),
    .z     (_row_5_d_x8_4_z)
  );
  ADD row_5_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:695:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1952_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1961:113
    .y     (_row_5_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:679:103
    .z     (_row_5_n_x3_2_z)
  );
  dup_2 row_5_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:696:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:695:103
    .y     (_row_5_d_x3_2_y),
    .z     (_row_5_d_x3_2_z)
  );
  SUB row_5_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:697:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1953_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1962:113
    .y     (_row_5_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:679:103
    .z     (_row_5_n_x0_2_z)
  );
  dup_2 row_5_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:698:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:697:103
    .y     (_row_5_d_x0_2_y),
    .z     (_row_5_d_x0_2_z)
  );
  ADD row_5_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:699:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:686:103
    .y     (_row_5_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:690:103
    .z     (_row_5_n_u2_2_z)
  );
  MUL row_5_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:700:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:630:79
    .y     (_row_5_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:699:103
    .z     (_row_5_n_v2_2_z)
  );
  ADD row_5_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:701:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:700:103
    .y     (_row_5_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:628:79
    .z     (_row_5_n_w2_2_z)
  );
  SHR_8 row_5_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:702:86
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:701:103
    .z     (_row_5_n_x2_2_z)
  );
  dup_2 row_5_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:703:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:702:86
    .y     (_row_5_d_x2_2_y),
    .z     (_row_5_d_x2_2_z)
  );
  SUB row_5_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:704:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:686:103
    .y     (_row_5_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:690:103
    .z     (_row_5_n_u4_3_z)
  );
  MUL row_5_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:705:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:631:79
    .y     (_row_5_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:704:103
    .z     (_row_5_n_v4_3_z)
  );
  ADD row_5_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:706:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:705:103
    .y     (_row_5_n_c128_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:629:79
    .z     (_row_5_n_w4_3_z)
  );
  SHR_8 row_5_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:707:86
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:706:103
    .z     (_row_5_n_x4_3_z)
  );
  dup_2 row_5_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:708:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:707:86
    .y     (_row_5_d_x4_3_y),
    .z     (_row_5_d_x4_3_z)
  );
  ADD row_5_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:709:108
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:692:103
    .y     (_row_5_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:684:103
    .z     (_row_5_n_tmp_0_z)
  );
  SHR_8 row_5_n_shr_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:710:90
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:709:108
    .z     (_row_5_n_shr_0_z)
  );
  ADD row_5_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:711:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1954_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1963:113
    .y     (_row_5_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:703:103
    .z     (_row_5_n_tmp_1_z)
  );
  SHR_8 row_5_n_shr_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:712:90
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:711:108
    .z     (_row_5_n_shr_1_z)
  );
  ADD row_5_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:713:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1955_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1964:113
    .y     (_row_5_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:708:103
    .z     (_row_5_n_tmp_2_z)
  );
  SHR_8 row_5_n_shr_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:714:90
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:713:108
    .z     (_row_5_n_shr_2_z)
  );
  ADD row_5_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:715:108
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x8_4_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:694:103
    .y     (_row_5_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:688:103
    .z     (_row_5_n_tmp_3_z)
  );
  SHR_8 row_5_n_shr_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:716:90
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:715:108
    .z     (_row_5_n_shr_3_z)
  );
  SUB row_5_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:717:108
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:694:103
    .y     (_row_5_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:688:103
    .z     (_row_5_n_tmp_4_z)
  );
  SHR_8 row_5_n_shr_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:718:90
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:717:108
    .z     (_row_5_n_shr_4_z)
  );
  SUB row_5_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:719:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1956_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1965:113
    .y     (_row_5_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:708:103
    .z     (_row_5_n_tmp_5_z)
  );
  SHR_8 row_5_n_shr_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:720:90
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:719:108
    .z     (_row_5_n_shr_5_z)
  );
  SUB row_5_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:721:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1957_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1966:113
    .y     (_row_5_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:703:103
    .z     (_row_5_n_tmp_6_z)
  );
  SHR_8 row_5_n_shr_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:722:90
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:721:108
    .z     (_row_5_n_shr_6_z)
  );
  SUB row_5_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:723:108
    .clock (clock),
    .reset (reset),
    .x     (_row_5_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:692:103
    .y     (_row_5_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:684:103
    .z     (_row_5_n_tmp_7_z)
  );
  SHR_8 row_5_n_shr_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:724:90
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:723:108
    .z     (_row_5_n_shr_7_z)
  );
  C128 row_6_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:725:79
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_c128_0_value)
  );
  C128 row_6_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:726:79
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_c128_1_value)
  );
  C128 row_6_n_c128_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:727:79
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_c128_2_value)
  );
  C181 row_6_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:728:79
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_c181_0_value)
  );
  C181 row_6_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:729:79
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_c181_1_value)
  );
  W7 row_6_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:730:67
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_w7_value)
  );
  W1_sub_W7 row_6_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:731:88
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_w1_sub_w7_value)
  );
  W1_add_W7 row_6_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:732:88
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_w1_add_w7_value)
  );
  W3 row_6_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:733:67
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_w3_value)
  );
  W3_sub_W5 row_6_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:734:88
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_w3_sub_w5_value)
  );
  W3_add_W5 row_6_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:735:88
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_w3_add_w5_value)
  );
  W6 row_6_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:736:67
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_w6_value)
  );
  W2_sub_W6 row_6_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:737:88
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_w2_sub_w6_value)
  );
  W2_add_W6 row_6_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:738:88
    .clock (clock),
    .reset (reset),
    .value (_row_6_n_w2_add_w6_value)
  );
  SHL_11 row_6_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:739:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_6_4_x),
    .z     (_row_6_n_x1_0_z)
  );
  SHL_11 row_6_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:740:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_6_0_x),
    .z     (_row_6_n_t0_0_z)
  );
  ADD row_6_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:741:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:740:86
    .y     (_row_6_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:725:79
    .z     (_row_6_n_x0_0_z)
  );
  dup_2 row_6_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:742:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:741:103
    .y     (_row_6_d_x0_0_y),
    .z     (_row_6_d_x0_0_z)
  );
  dup_2 row_6_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:743:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:739:86
    .y     (_row_6_d_x1_0_y),
    .z     (_row_6_d_x1_0_z)
  );
  dup_2 row_6_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:744:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_6_6_x),
    .y     (_row_6_d_x2_0_y),
    .z     (_row_6_d_x2_0_z)
  );
  dup_2 row_6_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:745:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_6_2_x),
    .y     (_row_6_d_x3_0_y),
    .z     (_row_6_d_x3_0_z)
  );
  dup_2 row_6_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:746:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_6_1_x),
    .y     (_row_6_d_x4_0_y),
    .z     (_row_6_d_x4_0_z)
  );
  dup_2 row_6_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:747:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_6_7_x),
    .y     (_row_6_d_x5_0_y),
    .z     (_row_6_d_x5_0_z)
  );
  dup_2 row_6_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:748:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_6_5_x),
    .y     (_row_6_d_x6_0_y),
    .z     (_row_6_d_x6_0_z)
  );
  dup_2 row_6_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:749:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_6_3_x),
    .y     (_row_6_d_x7_0_y),
    .z     (_row_6_d_x7_0_z)
  );
  ADD row_6_n_t8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:750:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:746:103
    .y     (_row_6_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:747:103
    .z     (_row_6_n_t8_1_z)
  );
  MUL row_6_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:751:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:730:67
    .y     (_row_6_n_t8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:750:103
    .z     (_row_6_n_x8_1_z)
  );
  dup_2 row_6_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:752:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:751:103
    .y     (_row_6_d_x8_1_y),
    .z     (_row_6_d_x8_1_z)
  );
  MUL row_6_n_t4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:753:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:731:88
    .y     (_row_6_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:746:103
    .z     (_row_6_n_t4_1_z)
  );
  ADD row_6_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:754:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:752:103
    .y     (_delay_INT16_1_1890_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1899:113
    .z     (_row_6_n_x4_1_z)
  );
  dup_2 row_6_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:755:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:754:103
    .y     (_row_6_d_x4_1_y),
    .z     (_row_6_d_x4_1_z)
  );
  MUL row_6_n_t5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:756:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:732:88
    .y     (_row_6_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:747:103
    .z     (_row_6_n_t5_1_z)
  );
  SUB row_6_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:757:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:752:103
    .y     (_delay_INT16_1_1891_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1900:113
    .z     (_row_6_n_x5_1_z)
  );
  dup_2 row_6_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:758:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:757:103
    .y     (_row_6_d_x5_1_y),
    .z     (_row_6_d_x5_1_z)
  );
  ADD row_6_n_t8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:759:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:748:103
    .y     (_row_6_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:749:103
    .z     (_row_6_n_t8_2_z)
  );
  MUL row_6_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:760:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:733:67
    .y     (_row_6_n_t8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:759:103
    .z     (_row_6_n_x8_2_z)
  );
  dup_2 row_6_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:761:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:760:103
    .y     (_row_6_d_x8_2_y),
    .z     (_row_6_d_x8_2_z)
  );
  MUL row_6_n_t6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:762:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:734:88
    .y     (_row_6_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:748:103
    .z     (_row_6_n_t6_1_z)
  );
  SUB row_6_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:763:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:761:103
    .y     (_delay_INT16_1_1892_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1901:113
    .z     (_row_6_n_x6_1_z)
  );
  dup_2 row_6_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:764:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:763:103
    .y     (_row_6_d_x6_1_y),
    .z     (_row_6_d_x6_1_z)
  );
  MUL row_6_n_t7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:765:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:735:88
    .y     (_row_6_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:749:103
    .z     (_row_6_n_t7_1_z)
  );
  SUB row_6_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:766:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:761:103
    .y     (_delay_INT16_1_1893_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1902:113
    .z     (_row_6_n_x7_1_z)
  );
  dup_2 row_6_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:767:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:766:103
    .y     (_row_6_d_x7_1_y),
    .z     (_row_6_d_x7_1_z)
  );
  ADD row_6_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:768:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:742:103
    .y     (_delay_INT16_1_1894_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1903:113
    .z     (_row_6_n_x8_3_z)
  );
  dup_2 row_6_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:769:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:768:103
    .y     (_row_6_d_x8_3_y),
    .z     (_row_6_d_x8_3_z)
  );
  SUB row_6_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:770:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:742:103
    .y     (_delay_INT16_1_1895_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1904:113
    .z     (_row_6_n_x0_1_z)
  );
  dup_2 row_6_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:771:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:770:103
    .y     (_row_6_d_x0_1_y),
    .z     (_row_6_d_x0_1_z)
  );
  ADD row_6_n_t1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:772:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:745:103
    .y     (_row_6_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:744:103
    .z     (_row_6_n_t1_1_z)
  );
  MUL row_6_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:773:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:736:67
    .y     (_row_6_n_t1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:772:103
    .z     (_row_6_n_x1_1_z)
  );
  dup_2 row_6_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:774:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:773:103
    .y     (_row_6_d_x1_1_y),
    .z     (_row_6_d_x1_1_z)
  );
  MUL row_6_n_t2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:775:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:738:88
    .y     (_row_6_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:744:103
    .z     (_row_6_n_t2_1_z)
  );
  SUB row_6_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:776:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:774:103
    .y     (_delay_INT16_1_1960_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1969:113
    .z     (_row_6_n_x2_1_z)
  );
  dup_2 row_6_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:777:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:776:103
    .y     (_row_6_d_x2_1_y),
    .z     (_row_6_d_x2_1_z)
  );
  MUL row_6_n_t3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:778:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:737:88
    .y     (_row_6_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:745:103
    .z     (_row_6_n_t3_1_z)
  );
  ADD row_6_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:779:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:774:103
    .y     (_delay_INT16_1_1961_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1970:113
    .z     (_row_6_n_x3_1_z)
  );
  dup_2 row_6_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:780:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:779:103
    .y     (_row_6_d_x3_1_y),
    .z     (_row_6_d_x3_1_z)
  );
  ADD row_6_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:781:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:755:103
    .y     (_row_6_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:764:103
    .z     (_row_6_n_x1_2_z)
  );
  dup_2 row_6_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:782:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:781:103
    .y     (_row_6_d_x1_2_y),
    .z     (_row_6_d_x1_2_z)
  );
  SUB row_6_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:783:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:755:103
    .y     (_row_6_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:764:103
    .z     (_row_6_n_x4_2_z)
  );
  dup_2 row_6_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:784:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:783:103
    .y     (_row_6_d_x4_2_y),
    .z     (_row_6_d_x4_2_z)
  );
  ADD row_6_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:785:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:758:103
    .y     (_row_6_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:767:103
    .z     (_row_6_n_x6_2_z)
  );
  dup_2 row_6_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:786:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:785:103
    .y     (_row_6_d_x6_2_y),
    .z     (_row_6_d_x6_2_z)
  );
  SUB row_6_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:787:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:758:103
    .y     (_row_6_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:767:103
    .z     (_row_6_n_x5_2_z)
  );
  dup_2 row_6_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:788:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:787:103
    .y     (_row_6_d_x5_2_y),
    .z     (_row_6_d_x5_2_z)
  );
  ADD row_6_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:789:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1962_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1971:113
    .y     (_row_6_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:780:103
    .z     (_row_6_n_x7_2_z)
  );
  dup_2 row_6_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:790:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:789:103
    .y     (_row_6_d_x7_2_y),
    .z     (_row_6_d_x7_2_z)
  );
  SUB row_6_n_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:791:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1963_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1972:113
    .y     (_row_6_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:780:103
    .z     (_row_6_n_x8_4_z)
  );
  dup_2 row_6_d_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:792:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:791:103
    .y     (_row_6_d_x8_4_y),
    .z     (_row_6_d_x8_4_z)
  );
  ADD row_6_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:793:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1964_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1973:113
    .y     (_row_6_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:777:103
    .z     (_row_6_n_x3_2_z)
  );
  dup_2 row_6_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:794:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:793:103
    .y     (_row_6_d_x3_2_y),
    .z     (_row_6_d_x3_2_z)
  );
  SUB row_6_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:795:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1965_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1974:113
    .y     (_row_6_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:777:103
    .z     (_row_6_n_x0_2_z)
  );
  dup_2 row_6_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:796:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:795:103
    .y     (_row_6_d_x0_2_y),
    .z     (_row_6_d_x0_2_z)
  );
  ADD row_6_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:797:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:784:103
    .y     (_row_6_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:788:103
    .z     (_row_6_n_u2_2_z)
  );
  MUL row_6_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:798:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:728:79
    .y     (_row_6_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:797:103
    .z     (_row_6_n_v2_2_z)
  );
  ADD row_6_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:799:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:798:103
    .y     (_row_6_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:726:79
    .z     (_row_6_n_w2_2_z)
  );
  SHR_8 row_6_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:800:86
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:799:103
    .z     (_row_6_n_x2_2_z)
  );
  dup_2 row_6_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:801:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:800:86
    .y     (_row_6_d_x2_2_y),
    .z     (_row_6_d_x2_2_z)
  );
  SUB row_6_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:802:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:784:103
    .y     (_row_6_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:788:103
    .z     (_row_6_n_u4_3_z)
  );
  MUL row_6_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:803:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:729:79
    .y     (_row_6_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:802:103
    .z     (_row_6_n_v4_3_z)
  );
  ADD row_6_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:804:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:803:103
    .y     (_row_6_n_c128_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:727:79
    .z     (_row_6_n_w4_3_z)
  );
  SHR_8 row_6_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:805:86
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:804:103
    .z     (_row_6_n_x4_3_z)
  );
  dup_2 row_6_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:806:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:805:86
    .y     (_row_6_d_x4_3_y),
    .z     (_row_6_d_x4_3_z)
  );
  ADD row_6_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:807:108
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:790:103
    .y     (_row_6_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:782:103
    .z     (_row_6_n_tmp_0_z)
  );
  SHR_8 row_6_n_shr_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:808:90
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:807:108
    .z     (_row_6_n_shr_0_z)
  );
  ADD row_6_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:809:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1966_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1975:113
    .y     (_row_6_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:801:103
    .z     (_row_6_n_tmp_1_z)
  );
  SHR_8 row_6_n_shr_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:810:90
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:809:108
    .z     (_row_6_n_shr_1_z)
  );
  ADD row_6_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:811:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1967_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1976:113
    .y     (_row_6_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:806:103
    .z     (_row_6_n_tmp_2_z)
  );
  SHR_8 row_6_n_shr_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:812:90
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:811:108
    .z     (_row_6_n_shr_2_z)
  );
  ADD row_6_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:813:108
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x8_4_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:792:103
    .y     (_row_6_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:786:103
    .z     (_row_6_n_tmp_3_z)
  );
  SHR_8 row_6_n_shr_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:814:90
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:813:108
    .z     (_row_6_n_shr_3_z)
  );
  SUB row_6_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:815:108
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:792:103
    .y     (_row_6_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:786:103
    .z     (_row_6_n_tmp_4_z)
  );
  SHR_8 row_6_n_shr_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:816:90
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:815:108
    .z     (_row_6_n_shr_4_z)
  );
  SUB row_6_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:817:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1968_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1977:113
    .y     (_row_6_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:806:103
    .z     (_row_6_n_tmp_5_z)
  );
  SHR_8 row_6_n_shr_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:818:90
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:817:108
    .z     (_row_6_n_shr_5_z)
  );
  SUB row_6_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:819:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1969_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1978:113
    .y     (_row_6_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:801:103
    .z     (_row_6_n_tmp_6_z)
  );
  SHR_8 row_6_n_shr_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:820:90
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:819:108
    .z     (_row_6_n_shr_6_z)
  );
  SUB row_6_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:821:108
    .clock (clock),
    .reset (reset),
    .x     (_row_6_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:790:103
    .y     (_row_6_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:782:103
    .z     (_row_6_n_tmp_7_z)
  );
  SHR_8 row_6_n_shr_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:822:90
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:821:108
    .z     (_row_6_n_shr_7_z)
  );
  C128 row_7_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:823:79
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_c128_0_value)
  );
  C128 row_7_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:824:79
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_c128_1_value)
  );
  C128 row_7_n_c128_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:825:79
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_c128_2_value)
  );
  C181 row_7_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:826:79
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_c181_0_value)
  );
  C181 row_7_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:827:79
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_c181_1_value)
  );
  W7 row_7_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:828:67
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_w7_value)
  );
  W1_sub_W7 row_7_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:829:88
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_w1_sub_w7_value)
  );
  W1_add_W7 row_7_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:830:88
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_w1_add_w7_value)
  );
  W3 row_7_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:831:67
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_w3_value)
  );
  W3_sub_W5 row_7_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:832:88
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_w3_sub_w5_value)
  );
  W3_add_W5 row_7_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:833:88
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_w3_add_w5_value)
  );
  W6 row_7_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:834:67
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_w6_value)
  );
  W2_sub_W6 row_7_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:835:88
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_w2_sub_w6_value)
  );
  W2_add_W6 row_7_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:836:88
    .clock (clock),
    .reset (reset),
    .value (_row_7_n_w2_add_w6_value)
  );
  SHL_11 row_7_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:837:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_7_4_x),
    .z     (_row_7_n_x1_0_z)
  );
  SHL_11 row_7_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:838:86
    .clock (clock),
    .reset (reset),
    .x     (n_in_7_0_x),
    .z     (_row_7_n_t0_0_z)
  );
  ADD row_7_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:839:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:838:86
    .y     (_row_7_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:823:79
    .z     (_row_7_n_x0_0_z)
  );
  dup_2 row_7_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:840:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:839:103
    .y     (_row_7_d_x0_0_y),
    .z     (_row_7_d_x0_0_z)
  );
  dup_2 row_7_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:841:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:837:86
    .y     (_row_7_d_x1_0_y),
    .z     (_row_7_d_x1_0_z)
  );
  dup_2 row_7_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:842:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_7_6_x),
    .y     (_row_7_d_x2_0_y),
    .z     (_row_7_d_x2_0_z)
  );
  dup_2 row_7_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:843:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_7_2_x),
    .y     (_row_7_d_x3_0_y),
    .z     (_row_7_d_x3_0_z)
  );
  dup_2 row_7_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:844:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_7_1_x),
    .y     (_row_7_d_x4_0_y),
    .z     (_row_7_d_x4_0_z)
  );
  dup_2 row_7_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:845:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_7_7_x),
    .y     (_row_7_d_x5_0_y),
    .z     (_row_7_d_x5_0_z)
  );
  dup_2 row_7_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:846:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_7_5_x),
    .y     (_row_7_d_x6_0_y),
    .z     (_row_7_d_x6_0_z)
  );
  dup_2 row_7_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:847:103
    .clock (clock),
    .reset (reset),
    .x     (n_in_7_3_x),
    .y     (_row_7_d_x7_0_y),
    .z     (_row_7_d_x7_0_z)
  );
  ADD row_7_n_t8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:848:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:844:103
    .y     (_row_7_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:845:103
    .z     (_row_7_n_t8_1_z)
  );
  MUL row_7_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:849:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:828:67
    .y     (_row_7_n_t8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:848:103
    .z     (_row_7_n_x8_1_z)
  );
  dup_2 row_7_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:850:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:849:103
    .y     (_row_7_d_x8_1_y),
    .z     (_row_7_d_x8_1_z)
  );
  MUL row_7_n_t4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:851:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:829:88
    .y     (_row_7_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:844:103
    .z     (_row_7_n_t4_1_z)
  );
  ADD row_7_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:852:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:850:103
    .y     (_delay_INT16_1_1970_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1979:113
    .z     (_row_7_n_x4_1_z)
  );
  dup_2 row_7_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:853:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:852:103
    .y     (_row_7_d_x4_1_y),
    .z     (_row_7_d_x4_1_z)
  );
  MUL row_7_n_t5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:854:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:830:88
    .y     (_row_7_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:845:103
    .z     (_row_7_n_t5_1_z)
  );
  SUB row_7_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:855:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:850:103
    .y     (_delay_INT16_1_1971_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1980:113
    .z     (_row_7_n_x5_1_z)
  );
  dup_2 row_7_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:856:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:855:103
    .y     (_row_7_d_x5_1_y),
    .z     (_row_7_d_x5_1_z)
  );
  ADD row_7_n_t8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:857:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:846:103
    .y     (_row_7_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:847:103
    .z     (_row_7_n_t8_2_z)
  );
  MUL row_7_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:858:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:831:67
    .y     (_row_7_n_t8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:857:103
    .z     (_row_7_n_x8_2_z)
  );
  dup_2 row_7_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:859:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:858:103
    .y     (_row_7_d_x8_2_y),
    .z     (_row_7_d_x8_2_z)
  );
  MUL row_7_n_t6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:860:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:832:88
    .y     (_row_7_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:846:103
    .z     (_row_7_n_t6_1_z)
  );
  SUB row_7_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:861:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:859:103
    .y     (_delay_INT16_1_1972_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1981:113
    .z     (_row_7_n_x6_1_z)
  );
  dup_2 row_7_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:862:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:861:103
    .y     (_row_7_d_x6_1_y),
    .z     (_row_7_d_x6_1_z)
  );
  MUL row_7_n_t7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:863:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:833:88
    .y     (_row_7_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:847:103
    .z     (_row_7_n_t7_1_z)
  );
  SUB row_7_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:864:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:859:103
    .y     (_delay_INT16_1_1973_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1982:113
    .z     (_row_7_n_x7_1_z)
  );
  dup_2 row_7_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:865:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:864:103
    .y     (_row_7_d_x7_1_y),
    .z     (_row_7_d_x7_1_z)
  );
  ADD row_7_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:866:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:840:103
    .y     (_delay_INT16_1_1974_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1983:113
    .z     (_row_7_n_x8_3_z)
  );
  dup_2 row_7_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:867:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:866:103
    .y     (_row_7_d_x8_3_y),
    .z     (_row_7_d_x8_3_z)
  );
  SUB row_7_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:868:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:840:103
    .y     (_delay_INT16_1_1975_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1984:113
    .z     (_row_7_n_x0_1_z)
  );
  dup_2 row_7_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:869:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:868:103
    .y     (_row_7_d_x0_1_y),
    .z     (_row_7_d_x0_1_z)
  );
  ADD row_7_n_t1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:870:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:843:103
    .y     (_row_7_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:842:103
    .z     (_row_7_n_t1_1_z)
  );
  MUL row_7_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:871:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:834:67
    .y     (_row_7_n_t1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:870:103
    .z     (_row_7_n_x1_1_z)
  );
  dup_2 row_7_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:872:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:871:103
    .y     (_row_7_d_x1_1_y),
    .z     (_row_7_d_x1_1_z)
  );
  MUL row_7_n_t2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:873:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:836:88
    .y     (_row_7_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:842:103
    .z     (_row_7_n_t2_1_z)
  );
  SUB row_7_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:874:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:872:103
    .y     (_delay_INT16_1_1976_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1985:113
    .z     (_row_7_n_x2_1_z)
  );
  dup_2 row_7_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:875:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:874:103
    .y     (_row_7_d_x2_1_y),
    .z     (_row_7_d_x2_1_z)
  );
  MUL row_7_n_t3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:876:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:835:88
    .y     (_row_7_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:843:103
    .z     (_row_7_n_t3_1_z)
  );
  ADD row_7_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:877:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:872:103
    .y     (_delay_INT16_1_1977_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1986:113
    .z     (_row_7_n_x3_1_z)
  );
  dup_2 row_7_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:878:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:877:103
    .y     (_row_7_d_x3_1_y),
    .z     (_row_7_d_x3_1_z)
  );
  ADD row_7_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:879:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:853:103
    .y     (_row_7_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:862:103
    .z     (_row_7_n_x1_2_z)
  );
  dup_2 row_7_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:880:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:879:103
    .y     (_row_7_d_x1_2_y),
    .z     (_row_7_d_x1_2_z)
  );
  SUB row_7_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:881:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:853:103
    .y     (_row_7_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:862:103
    .z     (_row_7_n_x4_2_z)
  );
  dup_2 row_7_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:882:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:881:103
    .y     (_row_7_d_x4_2_y),
    .z     (_row_7_d_x4_2_z)
  );
  ADD row_7_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:883:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:856:103
    .y     (_row_7_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:865:103
    .z     (_row_7_n_x6_2_z)
  );
  dup_2 row_7_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:884:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:883:103
    .y     (_row_7_d_x6_2_y),
    .z     (_row_7_d_x6_2_z)
  );
  SUB row_7_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:885:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:856:103
    .y     (_row_7_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:865:103
    .z     (_row_7_n_x5_2_z)
  );
  dup_2 row_7_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:886:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:885:103
    .y     (_row_7_d_x5_2_y),
    .z     (_row_7_d_x5_2_z)
  );
  ADD row_7_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:887:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1978_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1987:113
    .y     (_row_7_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:878:103
    .z     (_row_7_n_x7_2_z)
  );
  dup_2 row_7_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:888:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:887:103
    .y     (_row_7_d_x7_2_y),
    .z     (_row_7_d_x7_2_z)
  );
  SUB row_7_n_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:889:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1979_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1988:113
    .y     (_row_7_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:878:103
    .z     (_row_7_n_x8_4_z)
  );
  dup_2 row_7_d_x8_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:890:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:889:103
    .y     (_row_7_d_x8_4_y),
    .z     (_row_7_d_x8_4_z)
  );
  ADD row_7_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:891:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1980_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1989:113
    .y     (_row_7_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:875:103
    .z     (_row_7_n_x3_2_z)
  );
  dup_2 row_7_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:892:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:891:103
    .y     (_row_7_d_x3_2_y),
    .z     (_row_7_d_x3_2_z)
  );
  SUB row_7_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:893:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_2_1981_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1990:113
    .y     (_row_7_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:875:103
    .z     (_row_7_n_x0_2_z)
  );
  dup_2 row_7_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:894:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:893:103
    .y     (_row_7_d_x0_2_y),
    .z     (_row_7_d_x0_2_z)
  );
  ADD row_7_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:895:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:882:103
    .y     (_row_7_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:886:103
    .z     (_row_7_n_u2_2_z)
  );
  MUL row_7_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:896:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:826:79
    .y     (_row_7_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:895:103
    .z     (_row_7_n_v2_2_z)
  );
  ADD row_7_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:897:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:896:103
    .y     (_row_7_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:824:79
    .z     (_row_7_n_w2_2_z)
  );
  SHR_8 row_7_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:898:86
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:897:103
    .z     (_row_7_n_x2_2_z)
  );
  dup_2 row_7_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:899:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:898:86
    .y     (_row_7_d_x2_2_y),
    .z     (_row_7_d_x2_2_z)
  );
  SUB row_7_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:900:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:882:103
    .y     (_row_7_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:886:103
    .z     (_row_7_n_u4_3_z)
  );
  MUL row_7_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:901:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:827:79
    .y     (_row_7_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:900:103
    .z     (_row_7_n_v4_3_z)
  );
  ADD row_7_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:902:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:901:103
    .y     (_row_7_n_c128_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:825:79
    .z     (_row_7_n_w4_3_z)
  );
  SHR_8 row_7_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:903:86
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:902:103
    .z     (_row_7_n_x4_3_z)
  );
  dup_2 row_7_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:904:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:903:86
    .y     (_row_7_d_x4_3_y),
    .z     (_row_7_d_x4_3_z)
  );
  ADD row_7_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:905:108
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:888:103
    .y     (_row_7_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:880:103
    .z     (_row_7_n_tmp_0_z)
  );
  SHR_8 row_7_n_shr_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:906:90
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:905:108
    .z     (_row_7_n_shr_0_z)
  );
  ADD row_7_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:907:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1982_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1991:113
    .y     (_row_7_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:899:103
    .z     (_row_7_n_tmp_1_z)
  );
  SHR_8 row_7_n_shr_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:908:90
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:907:108
    .z     (_row_7_n_shr_1_z)
  );
  ADD row_7_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:909:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1983_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1992:113
    .y     (_row_7_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:904:103
    .z     (_row_7_n_tmp_2_z)
  );
  SHR_8 row_7_n_shr_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:910:90
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:909:108
    .z     (_row_7_n_shr_2_z)
  );
  ADD row_7_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:911:108
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x8_4_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:890:103
    .y     (_row_7_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:884:103
    .z     (_row_7_n_tmp_3_z)
  );
  SHR_8 row_7_n_shr_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:912:90
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:911:108
    .z     (_row_7_n_shr_3_z)
  );
  SUB row_7_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:913:108
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x8_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:890:103
    .y     (_row_7_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:884:103
    .z     (_row_7_n_tmp_4_z)
  );
  SHR_8 row_7_n_shr_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:914:90
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:913:108
    .z     (_row_7_n_shr_4_z)
  );
  SUB row_7_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:915:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1984_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1993:113
    .y     (_row_7_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:904:103
    .z     (_row_7_n_tmp_5_z)
  );
  SHR_8 row_7_n_shr_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:916:90
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:915:108
    .z     (_row_7_n_shr_5_z)
  );
  SUB row_7_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:917:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1985_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1994:113
    .y     (_row_7_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:899:103
    .z     (_row_7_n_tmp_6_z)
  );
  SHR_8 row_7_n_shr_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:918:90
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:917:108
    .z     (_row_7_n_shr_6_z)
  );
  SUB row_7_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:919:108
    .clock (clock),
    .reset (reset),
    .x     (_row_7_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:888:103
    .y     (_row_7_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:880:103
    .z     (_row_7_n_tmp_7_z)
  );
  SHR_8 row_7_n_shr_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:920:90
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:919:108
    .z     (_row_7_n_shr_7_z)
  );
  C4 col_0_n_c4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:921:73
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_c4_0_value)
  );
  C4 col_0_n_c4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:922:73
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_c4_1_value)
  );
  C4 col_0_n_c4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:923:73
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_c4_2_value)
  );
  C128 col_0_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:924:79
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_c128_0_value)
  );
  C128 col_0_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:925:79
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_c128_1_value)
  );
  C181 col_0_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:926:79
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_c181_0_value)
  );
  C181 col_0_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:927:79
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_c181_1_value)
  );
  C8192 col_0_n_c8192 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:928:76
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_c8192_value)
  );
  W7 col_0_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:929:67
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_w7_value)
  );
  W1_sub_W7 col_0_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:930:88
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_w1_sub_w7_value)
  );
  W1_add_W7 col_0_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:931:88
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_w1_add_w7_value)
  );
  W3 col_0_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:932:67
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_w3_value)
  );
  W3_sub_W5 col_0_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:933:88
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_w3_sub_w5_value)
  );
  W3_add_W5 col_0_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:934:88
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_w3_add_w5_value)
  );
  W6 col_0_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:935:67
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_w6_value)
  );
  W2_sub_W6 col_0_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:936:88
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_w2_sub_w6_value)
  );
  W2_add_W6 col_0_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:937:88
    .clock (clock),
    .reset (reset),
    .value (_col_0_n_w2_add_w6_value)
  );
  SHL_8 col_0_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:938:86
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_shr_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:612:90
    .z     (_col_0_n_x1_0_z)
  );
  SHL_8 col_0_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:939:86
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_shr_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:220:90
    .z     (_col_0_n_t0_0_z)
  );
  ADD col_0_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:940:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:939:86
    .y     (_col_0_n_c8192_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:928:76
    .z     (_col_0_n_x0_0_z)
  );
  dup_2 col_0_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:941:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:940:103
    .y     (_col_0_d_x0_0_y),
    .z     (_col_0_d_x0_0_z)
  );
  dup_2 col_0_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:942:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:938:86
    .y     (_col_0_d_x1_0_y),
    .z     (_col_0_d_x1_0_z)
  );
  dup_2 col_0_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:943:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_shr_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:808:90
    .y     (_col_0_d_x2_0_y),
    .z     (_col_0_d_x2_0_z)
  );
  dup_2 col_0_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:944:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_shr_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:416:90
    .y     (_col_0_d_x3_0_y),
    .z     (_col_0_d_x3_0_z)
  );
  dup_2 col_0_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:945:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_shr_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:318:90
    .y     (_col_0_d_x4_0_y),
    .z     (_col_0_d_x4_0_z)
  );
  dup_2 col_0_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:946:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_shr_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:906:90
    .y     (_col_0_d_x5_0_y),
    .z     (_col_0_d_x5_0_z)
  );
  dup_2 col_0_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:947:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_shr_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:710:90
    .y     (_col_0_d_x6_0_y),
    .z     (_col_0_d_x6_0_z)
  );
  dup_2 col_0_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:948:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_shr_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:514:90
    .y     (_col_0_d_x7_0_y),
    .z     (_col_0_d_x7_0_z)
  );
  ADD col_0_n_u8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:949:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:945:103
    .y     (_col_0_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:946:103
    .z     (_col_0_n_u8_0_z)
  );
  MUL col_0_n_v8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:950:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:929:67
    .y     (_col_0_n_u8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:949:103
    .z     (_col_0_n_v8_0_z)
  );
  ADD col_0_n_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:951:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:950:103
    .y     (_col_0_n_c4_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:921:73
    .z     (_col_0_n_x8_0_z)
  );
  dup_2 col_0_d_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:952:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:951:103
    .y     (_col_0_d_x8_0_y),
    .z     (_col_0_d_x8_0_z)
  );
  MUL col_0_n_u4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:953:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:930:88
    .y     (_col_0_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:945:103
    .z     (_col_0_n_u4_1_z)
  );
  ADD col_0_n_v4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:954:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x8_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:952:103
    .y     (_delay_INT16_2_1986_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1995:113
    .z     (_col_0_n_v4_1_z)
  );
  SHR_3 col_0_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:955:86
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:954:103
    .z     (_col_0_n_x4_1_z)
  );
  dup_2 col_0_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:956:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:955:86
    .y     (_col_0_d_x4_1_y),
    .z     (_col_0_d_x4_1_z)
  );
  MUL col_0_n_u5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:957:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:931:88
    .y     (_col_0_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:946:103
    .z     (_col_0_n_u5_1_z)
  );
  SUB col_0_n_v5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:958:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:952:103
    .y     (_delay_INT16_2_1987_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1996:113
    .z     (_col_0_n_v5_1_z)
  );
  SHR_3 col_0_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:959:86
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:958:103
    .z     (_col_0_n_x5_1_z)
  );
  dup_2 col_0_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:960:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:959:86
    .y     (_col_0_d_x5_1_y),
    .z     (_col_0_d_x5_1_z)
  );
  ADD col_0_n_u8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:961:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:947:103
    .y     (_col_0_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:948:103
    .z     (_col_0_n_u8_1_z)
  );
  MUL col_0_n_v8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:962:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:932:67
    .y     (_col_0_n_u8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:961:103
    .z     (_col_0_n_v8_1_z)
  );
  ADD col_0_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:963:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:962:103
    .y     (_col_0_n_c4_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:922:73
    .z     (_col_0_n_x8_1_z)
  );
  dup_2 col_0_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:964:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:963:103
    .y     (_col_0_d_x8_1_y),
    .z     (_col_0_d_x8_1_z)
  );
  MUL col_0_n_u6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:965:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:933:88
    .y     (_col_0_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:947:103
    .z     (_col_0_n_u6_1_z)
  );
  SUB col_0_n_v6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:966:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:964:103
    .y     (_delay_INT16_2_1988_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1997:113
    .z     (_col_0_n_v6_1_z)
  );
  SHR_3 col_0_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:967:86
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:966:103
    .z     (_col_0_n_x6_1_z)
  );
  dup_2 col_0_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:968:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:967:86
    .y     (_col_0_d_x6_1_y),
    .z     (_col_0_d_x6_1_z)
  );
  MUL col_0_n_u7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:969:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:934:88
    .y     (_col_0_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:948:103
    .z     (_col_0_n_u7_1_z)
  );
  SUB col_0_n_v7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:970:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:964:103
    .y     (_delay_INT16_2_1989_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1998:113
    .z     (_col_0_n_v7_1_z)
  );
  SHR_3 col_0_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:971:86
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:970:103
    .z     (_col_0_n_x7_1_z)
  );
  dup_2 col_0_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:972:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:971:86
    .y     (_col_0_d_x7_1_y),
    .z     (_col_0_d_x7_1_z)
  );
  ADD col_0_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:973:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:941:103
    .y     (_delay_INT16_1_1990_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1999:113
    .z     (_col_0_n_x8_2_z)
  );
  dup_2 col_0_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:974:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:973:103
    .y     (_col_0_d_x8_2_y),
    .z     (_col_0_d_x8_2_z)
  );
  SUB col_0_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:975:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:941:103
    .y     (_delay_INT16_1_1991_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2000:113
    .z     (_col_0_n_x0_1_z)
  );
  dup_2 col_0_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:976:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:975:103
    .y     (_col_0_d_x0_1_y),
    .z     (_col_0_d_x0_1_z)
  );
  ADD col_0_n_u1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:977:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:944:103
    .y     (_col_0_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:943:103
    .z     (_col_0_n_u1_1_z)
  );
  MUL col_0_n_v1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:978:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:935:67
    .y     (_col_0_n_u1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:977:103
    .z     (_col_0_n_v1_1_z)
  );
  ADD col_0_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:979:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:978:103
    .y     (_col_0_n_c4_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:923:73
    .z     (_col_0_n_x1_1_z)
  );
  dup_2 col_0_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:980:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:979:103
    .y     (_col_0_d_x1_1_y),
    .z     (_col_0_d_x1_1_z)
  );
  MUL col_0_n_u2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:981:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:937:88
    .y     (_col_0_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:943:103
    .z     (_col_0_n_u2_1_z)
  );
  SUB col_0_n_v2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:982:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:980:103
    .y     (_delay_INT16_2_1992_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2001:113
    .z     (_col_0_n_v2_1_z)
  );
  SHR_3 col_0_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:983:86
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:982:103
    .z     (_col_0_n_x2_1_z)
  );
  dup_2 col_0_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:984:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:983:86
    .y     (_col_0_d_x2_1_y),
    .z     (_col_0_d_x2_1_z)
  );
  MUL col_0_n_u3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:985:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:936:88
    .y     (_col_0_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:944:103
    .z     (_col_0_n_u3_1_z)
  );
  ADD col_0_n_v3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:986:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:980:103
    .y     (_delay_INT16_2_1993_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2002:113
    .z     (_col_0_n_v3_1_z)
  );
  SHR_3 col_0_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:987:86
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:986:103
    .z     (_col_0_n_x3_1_z)
  );
  dup_2 col_0_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:988:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:987:86
    .y     (_col_0_d_x3_1_y),
    .z     (_col_0_d_x3_1_z)
  );
  ADD col_0_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:989:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:956:103
    .y     (_col_0_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:968:103
    .z     (_col_0_n_x1_2_z)
  );
  dup_2 col_0_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:990:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:989:103
    .y     (_col_0_d_x1_2_y),
    .z     (_col_0_d_x1_2_z)
  );
  SUB col_0_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:991:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:956:103
    .y     (_col_0_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:968:103
    .z     (_col_0_n_x4_2_z)
  );
  dup_2 col_0_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:992:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:991:103
    .y     (_col_0_d_x4_2_y),
    .z     (_col_0_d_x4_2_z)
  );
  ADD col_0_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:993:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:960:103
    .y     (_col_0_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:972:103
    .z     (_col_0_n_x6_2_z)
  );
  dup_2 col_0_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:994:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:993:103
    .y     (_col_0_d_x6_2_y),
    .z     (_col_0_d_x6_2_z)
  );
  SUB col_0_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:995:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:960:103
    .y     (_col_0_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:972:103
    .z     (_col_0_n_x5_2_z)
  );
  dup_2 col_0_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:996:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:995:103
    .y     (_col_0_d_x5_2_y),
    .z     (_col_0_d_x5_2_z)
  );
  ADD col_0_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:997:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_1994_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2003:113
    .y     (_col_0_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:988:103
    .z     (_col_0_n_x7_2_z)
  );
  dup_2 col_0_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:998:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:997:103
    .y     (_col_0_d_x7_2_y),
    .z     (_col_0_d_x7_2_z)
  );
  SUB col_0_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:999:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_1995_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2004:113
    .y     (_col_0_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:988:103
    .z     (_col_0_n_x8_3_z)
  );
  dup_2 col_0_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1000:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:999:103
    .y     (_col_0_d_x8_3_y),
    .z     (_col_0_d_x8_3_z)
  );
  ADD col_0_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1001:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_1996_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2005:113
    .y     (_col_0_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:984:103
    .z     (_col_0_n_x3_2_z)
  );
  dup_2 col_0_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1002:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1001:103
    .y     (_col_0_d_x3_2_y),
    .z     (_col_0_d_x3_2_z)
  );
  SUB col_0_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1003:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_1997_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2006:113
    .y     (_col_0_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:984:103
    .z     (_col_0_n_x0_2_z)
  );
  dup_2 col_0_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1004:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1003:103
    .y     (_col_0_d_x0_2_y),
    .z     (_col_0_d_x0_2_z)
  );
  ADD col_0_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1005:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:992:103
    .y     (_col_0_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:996:103
    .z     (_col_0_n_u2_2_z)
  );
  MUL col_0_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1006:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:926:79
    .y     (_col_0_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1005:103
    .z     (_col_0_n_v2_2_z)
  );
  ADD col_0_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1007:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1006:103
    .y     (_col_0_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:924:79
    .z     (_col_0_n_w2_2_z)
  );
  SHR_8 col_0_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1008:86
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1007:103
    .z     (_col_0_n_x2_2_z)
  );
  dup_2 col_0_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1009:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1008:86
    .y     (_col_0_d_x2_2_y),
    .z     (_col_0_d_x2_2_z)
  );
  SUB col_0_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1010:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:992:103
    .y     (_col_0_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:996:103
    .z     (_col_0_n_u4_3_z)
  );
  MUL col_0_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1011:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:927:79
    .y     (_col_0_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1010:103
    .z     (_col_0_n_v4_3_z)
  );
  ADD col_0_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1012:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1011:103
    .y     (_col_0_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:925:79
    .z     (_col_0_n_w4_3_z)
  );
  SHR_8 col_0_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1013:86
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1012:103
    .z     (_col_0_n_x4_3_z)
  );
  dup_2 col_0_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1014:103
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1013:86
    .y     (_col_0_d_x4_3_y),
    .z     (_col_0_d_x4_3_z)
  );
  ADD col_0_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1015:108
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:998:103
    .y     (_col_0_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:990:103
    .z     (_col_0_n_tmp_0_z)
  );
  SHR_14 col_0_n_val_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1016:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1015:108
    .z     (_col_0_n_val_0_z)
  );
  CLIP col_0_n_clp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1017:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_val_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1016:90
    .z     (n_out_0_0_x)
  );
  ADD col_0_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1018:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1998_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2007:113
    .y     (_col_0_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1009:103
    .z     (_col_0_n_tmp_1_z)
  );
  SHR_14 col_0_n_val_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1019:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1018:108
    .z     (_col_0_n_val_1_z)
  );
  CLIP col_0_n_clp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1020:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_val_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1019:90
    .z     (n_out_1_0_x)
  );
  ADD col_0_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1021:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_1999_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2008:113
    .y     (_col_0_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1014:103
    .z     (_col_0_n_tmp_2_z)
  );
  SHR_14 col_0_n_val_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1022:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1021:108
    .z     (_col_0_n_val_2_z)
  );
  CLIP col_0_n_clp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1023:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_val_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1022:90
    .z     (n_out_2_0_x)
  );
  ADD col_0_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1024:108
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1000:103
    .y     (_col_0_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:994:103
    .z     (_col_0_n_tmp_3_z)
  );
  SHR_14 col_0_n_val_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1025:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1024:108
    .z     (_col_0_n_val_3_z)
  );
  CLIP col_0_n_clp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1026:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_val_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1025:90
    .z     (n_out_3_0_x)
  );
  SUB col_0_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1027:108
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1000:103
    .y     (_col_0_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:994:103
    .z     (_col_0_n_tmp_4_z)
  );
  SHR_14 col_0_n_val_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1028:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1027:108
    .z     (_col_0_n_val_4_z)
  );
  CLIP col_0_n_clp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1029:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_val_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1028:90
    .z     (n_out_4_0_x)
  );
  SUB col_0_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1030:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2000_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2009:113
    .y     (_col_0_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1014:103
    .z     (_col_0_n_tmp_5_z)
  );
  SHR_14 col_0_n_val_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1031:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1030:108
    .z     (_col_0_n_val_5_z)
  );
  CLIP col_0_n_clp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1032:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_val_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1031:90
    .z     (n_out_5_0_x)
  );
  SUB col_0_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1033:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2001_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2010:113
    .y     (_col_0_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1009:103
    .z     (_col_0_n_tmp_6_z)
  );
  SHR_14 col_0_n_val_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1034:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1033:108
    .z     (_col_0_n_val_6_z)
  );
  CLIP col_0_n_clp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1035:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_val_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1034:90
    .z     (n_out_6_0_x)
  );
  SUB col_0_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1036:108
    .clock (clock),
    .reset (reset),
    .x     (_col_0_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:998:103
    .y     (_col_0_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:990:103
    .z     (_col_0_n_tmp_7_z)
  );
  SHR_14 col_0_n_val_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1037:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1036:108
    .z     (_col_0_n_val_7_z)
  );
  CLIP col_0_n_clp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1038:90
    .clock (clock),
    .reset (reset),
    .x     (_col_0_n_val_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1037:90
    .z     (n_out_7_0_x)
  );
  C4 col_1_n_c4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1039:73
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_c4_0_value)
  );
  C4 col_1_n_c4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1040:73
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_c4_1_value)
  );
  C4 col_1_n_c4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1041:73
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_c4_2_value)
  );
  C128 col_1_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1042:79
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_c128_0_value)
  );
  C128 col_1_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1043:79
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_c128_1_value)
  );
  C181 col_1_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1044:79
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_c181_0_value)
  );
  C181 col_1_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1045:79
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_c181_1_value)
  );
  C8192 col_1_n_c8192 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1046:76
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_c8192_value)
  );
  W7 col_1_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1047:67
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_w7_value)
  );
  W1_sub_W7 col_1_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1048:88
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_w1_sub_w7_value)
  );
  W1_add_W7 col_1_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1049:88
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_w1_add_w7_value)
  );
  W3 col_1_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1050:67
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_w3_value)
  );
  W3_sub_W5 col_1_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1051:88
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_w3_sub_w5_value)
  );
  W3_add_W5 col_1_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1052:88
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_w3_add_w5_value)
  );
  W6 col_1_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1053:67
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_w6_value)
  );
  W2_sub_W6 col_1_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1054:88
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_w2_sub_w6_value)
  );
  W2_add_W6 col_1_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1055:88
    .clock (clock),
    .reset (reset),
    .value (_col_1_n_w2_add_w6_value)
  );
  SHL_8 col_1_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1056:86
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_shr_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:614:90
    .z     (_col_1_n_x1_0_z)
  );
  SHL_8 col_1_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1057:86
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_shr_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:222:90
    .z     (_col_1_n_t0_0_z)
  );
  ADD col_1_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1058:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1057:86
    .y     (_col_1_n_c8192_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1046:76
    .z     (_col_1_n_x0_0_z)
  );
  dup_2 col_1_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1059:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1058:103
    .y     (_col_1_d_x0_0_y),
    .z     (_col_1_d_x0_0_z)
  );
  dup_2 col_1_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1060:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1056:86
    .y     (_col_1_d_x1_0_y),
    .z     (_col_1_d_x1_0_z)
  );
  dup_2 col_1_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1061:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_shr_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:810:90
    .y     (_col_1_d_x2_0_y),
    .z     (_col_1_d_x2_0_z)
  );
  dup_2 col_1_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1062:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_shr_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:418:90
    .y     (_col_1_d_x3_0_y),
    .z     (_col_1_d_x3_0_z)
  );
  dup_2 col_1_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1063:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_shr_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:320:90
    .y     (_col_1_d_x4_0_y),
    .z     (_col_1_d_x4_0_z)
  );
  dup_2 col_1_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1064:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_shr_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:908:90
    .y     (_col_1_d_x5_0_y),
    .z     (_col_1_d_x5_0_z)
  );
  dup_2 col_1_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1065:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_shr_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:712:90
    .y     (_col_1_d_x6_0_y),
    .z     (_col_1_d_x6_0_z)
  );
  dup_2 col_1_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1066:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_shr_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:516:90
    .y     (_col_1_d_x7_0_y),
    .z     (_col_1_d_x7_0_z)
  );
  ADD col_1_n_u8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1067:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1063:103
    .y     (_col_1_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1064:103
    .z     (_col_1_n_u8_0_z)
  );
  MUL col_1_n_v8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1068:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1047:67
    .y     (_col_1_n_u8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1067:103
    .z     (_col_1_n_v8_0_z)
  );
  ADD col_1_n_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1069:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1068:103
    .y     (_col_1_n_c4_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1039:73
    .z     (_col_1_n_x8_0_z)
  );
  dup_2 col_1_d_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1070:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1069:103
    .y     (_col_1_d_x8_0_y),
    .z     (_col_1_d_x8_0_z)
  );
  MUL col_1_n_u4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1071:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1048:88
    .y     (_col_1_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1063:103
    .z     (_col_1_n_u4_1_z)
  );
  ADD col_1_n_v4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1072:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x8_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1070:103
    .y     (_delay_INT16_2_2002_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2011:113
    .z     (_col_1_n_v4_1_z)
  );
  SHR_3 col_1_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1073:86
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1072:103
    .z     (_col_1_n_x4_1_z)
  );
  dup_2 col_1_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1074:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1073:86
    .y     (_col_1_d_x4_1_y),
    .z     (_col_1_d_x4_1_z)
  );
  MUL col_1_n_u5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1075:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1049:88
    .y     (_col_1_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1064:103
    .z     (_col_1_n_u5_1_z)
  );
  SUB col_1_n_v5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1076:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1070:103
    .y     (_delay_INT16_2_2003_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2012:113
    .z     (_col_1_n_v5_1_z)
  );
  SHR_3 col_1_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1077:86
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1076:103
    .z     (_col_1_n_x5_1_z)
  );
  dup_2 col_1_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1078:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1077:86
    .y     (_col_1_d_x5_1_y),
    .z     (_col_1_d_x5_1_z)
  );
  ADD col_1_n_u8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1079:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1065:103
    .y     (_col_1_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1066:103
    .z     (_col_1_n_u8_1_z)
  );
  MUL col_1_n_v8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1080:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1050:67
    .y     (_col_1_n_u8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1079:103
    .z     (_col_1_n_v8_1_z)
  );
  ADD col_1_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1081:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1080:103
    .y     (_col_1_n_c4_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1040:73
    .z     (_col_1_n_x8_1_z)
  );
  dup_2 col_1_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1082:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1081:103
    .y     (_col_1_d_x8_1_y),
    .z     (_col_1_d_x8_1_z)
  );
  MUL col_1_n_u6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1083:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1051:88
    .y     (_col_1_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1065:103
    .z     (_col_1_n_u6_1_z)
  );
  SUB col_1_n_v6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1084:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1082:103
    .y     (_delay_INT16_2_2004_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2013:113
    .z     (_col_1_n_v6_1_z)
  );
  SHR_3 col_1_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1085:86
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1084:103
    .z     (_col_1_n_x6_1_z)
  );
  dup_2 col_1_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1086:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1085:86
    .y     (_col_1_d_x6_1_y),
    .z     (_col_1_d_x6_1_z)
  );
  MUL col_1_n_u7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1087:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1052:88
    .y     (_col_1_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1066:103
    .z     (_col_1_n_u7_1_z)
  );
  SUB col_1_n_v7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1088:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1082:103
    .y     (_delay_INT16_2_2005_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2014:113
    .z     (_col_1_n_v7_1_z)
  );
  SHR_3 col_1_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1089:86
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1088:103
    .z     (_col_1_n_x7_1_z)
  );
  dup_2 col_1_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1090:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1089:86
    .y     (_col_1_d_x7_1_y),
    .z     (_col_1_d_x7_1_z)
  );
  ADD col_1_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1091:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1059:103
    .y     (_delay_INT16_1_2006_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2015:113
    .z     (_col_1_n_x8_2_z)
  );
  dup_2 col_1_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1092:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1091:103
    .y     (_col_1_d_x8_2_y),
    .z     (_col_1_d_x8_2_z)
  );
  SUB col_1_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1093:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1059:103
    .y     (_delay_INT16_1_2007_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2016:113
    .z     (_col_1_n_x0_1_z)
  );
  dup_2 col_1_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1094:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1093:103
    .y     (_col_1_d_x0_1_y),
    .z     (_col_1_d_x0_1_z)
  );
  ADD col_1_n_u1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1095:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1062:103
    .y     (_col_1_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1061:103
    .z     (_col_1_n_u1_1_z)
  );
  MUL col_1_n_v1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1096:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1053:67
    .y     (_col_1_n_u1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1095:103
    .z     (_col_1_n_v1_1_z)
  );
  ADD col_1_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1097:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1096:103
    .y     (_col_1_n_c4_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1041:73
    .z     (_col_1_n_x1_1_z)
  );
  dup_2 col_1_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1098:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1097:103
    .y     (_col_1_d_x1_1_y),
    .z     (_col_1_d_x1_1_z)
  );
  MUL col_1_n_u2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1099:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1055:88
    .y     (_col_1_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1061:103
    .z     (_col_1_n_u2_1_z)
  );
  SUB col_1_n_v2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1100:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1098:103
    .y     (_delay_INT16_2_2008_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2017:113
    .z     (_col_1_n_v2_1_z)
  );
  SHR_3 col_1_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1101:86
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1100:103
    .z     (_col_1_n_x2_1_z)
  );
  dup_2 col_1_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1102:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1101:86
    .y     (_col_1_d_x2_1_y),
    .z     (_col_1_d_x2_1_z)
  );
  MUL col_1_n_u3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1103:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1054:88
    .y     (_col_1_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1062:103
    .z     (_col_1_n_u3_1_z)
  );
  ADD col_1_n_v3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1104:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1098:103
    .y     (_delay_INT16_2_2009_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2018:113
    .z     (_col_1_n_v3_1_z)
  );
  SHR_3 col_1_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1105:86
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1104:103
    .z     (_col_1_n_x3_1_z)
  );
  dup_2 col_1_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1106:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1105:86
    .y     (_col_1_d_x3_1_y),
    .z     (_col_1_d_x3_1_z)
  );
  ADD col_1_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1107:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1074:103
    .y     (_col_1_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1086:103
    .z     (_col_1_n_x1_2_z)
  );
  dup_2 col_1_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1108:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1107:103
    .y     (_col_1_d_x1_2_y),
    .z     (_col_1_d_x1_2_z)
  );
  SUB col_1_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1109:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1074:103
    .y     (_col_1_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1086:103
    .z     (_col_1_n_x4_2_z)
  );
  dup_2 col_1_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1110:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1109:103
    .y     (_col_1_d_x4_2_y),
    .z     (_col_1_d_x4_2_z)
  );
  ADD col_1_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1111:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1078:103
    .y     (_col_1_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1090:103
    .z     (_col_1_n_x6_2_z)
  );
  dup_2 col_1_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1112:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1111:103
    .y     (_col_1_d_x6_2_y),
    .z     (_col_1_d_x6_2_z)
  );
  SUB col_1_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1113:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1078:103
    .y     (_col_1_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1090:103
    .z     (_col_1_n_x5_2_z)
  );
  dup_2 col_1_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1114:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1113:103
    .y     (_col_1_d_x5_2_y),
    .z     (_col_1_d_x5_2_z)
  );
  ADD col_1_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1115:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2010_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2019:113
    .y     (_col_1_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1106:103
    .z     (_col_1_n_x7_2_z)
  );
  dup_2 col_1_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1116:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1115:103
    .y     (_col_1_d_x7_2_y),
    .z     (_col_1_d_x7_2_z)
  );
  SUB col_1_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1117:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2011_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2020:113
    .y     (_col_1_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1106:103
    .z     (_col_1_n_x8_3_z)
  );
  dup_2 col_1_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1118:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1117:103
    .y     (_col_1_d_x8_3_y),
    .z     (_col_1_d_x8_3_z)
  );
  ADD col_1_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1119:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2012_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2021:113
    .y     (_col_1_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1102:103
    .z     (_col_1_n_x3_2_z)
  );
  dup_2 col_1_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1120:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1119:103
    .y     (_col_1_d_x3_2_y),
    .z     (_col_1_d_x3_2_z)
  );
  SUB col_1_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1121:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2013_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2022:113
    .y     (_col_1_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1102:103
    .z     (_col_1_n_x0_2_z)
  );
  dup_2 col_1_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1122:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1121:103
    .y     (_col_1_d_x0_2_y),
    .z     (_col_1_d_x0_2_z)
  );
  ADD col_1_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1123:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1110:103
    .y     (_col_1_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1114:103
    .z     (_col_1_n_u2_2_z)
  );
  MUL col_1_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1124:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1044:79
    .y     (_col_1_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1123:103
    .z     (_col_1_n_v2_2_z)
  );
  ADD col_1_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1125:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1124:103
    .y     (_col_1_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1042:79
    .z     (_col_1_n_w2_2_z)
  );
  SHR_8 col_1_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1126:86
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1125:103
    .z     (_col_1_n_x2_2_z)
  );
  dup_2 col_1_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1127:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1126:86
    .y     (_col_1_d_x2_2_y),
    .z     (_col_1_d_x2_2_z)
  );
  SUB col_1_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1128:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1110:103
    .y     (_col_1_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1114:103
    .z     (_col_1_n_u4_3_z)
  );
  MUL col_1_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1129:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1045:79
    .y     (_col_1_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1128:103
    .z     (_col_1_n_v4_3_z)
  );
  ADD col_1_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1130:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1129:103
    .y     (_col_1_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1043:79
    .z     (_col_1_n_w4_3_z)
  );
  SHR_8 col_1_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1131:86
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1130:103
    .z     (_col_1_n_x4_3_z)
  );
  dup_2 col_1_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1132:103
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1131:86
    .y     (_col_1_d_x4_3_y),
    .z     (_col_1_d_x4_3_z)
  );
  ADD col_1_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1133:108
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1116:103
    .y     (_col_1_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1108:103
    .z     (_col_1_n_tmp_0_z)
  );
  SHR_14 col_1_n_val_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1134:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1133:108
    .z     (_col_1_n_val_0_z)
  );
  CLIP col_1_n_clp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1135:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_val_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1134:90
    .z     (n_out_0_1_x)
  );
  ADD col_1_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1136:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2014_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2023:113
    .y     (_col_1_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1127:103
    .z     (_col_1_n_tmp_1_z)
  );
  SHR_14 col_1_n_val_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1137:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1136:108
    .z     (_col_1_n_val_1_z)
  );
  CLIP col_1_n_clp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1138:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_val_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1137:90
    .z     (n_out_1_1_x)
  );
  ADD col_1_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1139:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2015_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2024:113
    .y     (_col_1_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1132:103
    .z     (_col_1_n_tmp_2_z)
  );
  SHR_14 col_1_n_val_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1140:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1139:108
    .z     (_col_1_n_val_2_z)
  );
  CLIP col_1_n_clp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1141:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_val_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1140:90
    .z     (n_out_2_1_x)
  );
  ADD col_1_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1142:108
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1118:103
    .y     (_col_1_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1112:103
    .z     (_col_1_n_tmp_3_z)
  );
  SHR_14 col_1_n_val_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1143:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1142:108
    .z     (_col_1_n_val_3_z)
  );
  CLIP col_1_n_clp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1144:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_val_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1143:90
    .z     (n_out_3_1_x)
  );
  SUB col_1_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1145:108
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1118:103
    .y     (_col_1_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1112:103
    .z     (_col_1_n_tmp_4_z)
  );
  SHR_14 col_1_n_val_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1146:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1145:108
    .z     (_col_1_n_val_4_z)
  );
  CLIP col_1_n_clp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1147:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_val_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1146:90
    .z     (n_out_4_1_x)
  );
  SUB col_1_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1148:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2016_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2025:113
    .y     (_col_1_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1132:103
    .z     (_col_1_n_tmp_5_z)
  );
  SHR_14 col_1_n_val_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1149:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1148:108
    .z     (_col_1_n_val_5_z)
  );
  CLIP col_1_n_clp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1150:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_val_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1149:90
    .z     (n_out_5_1_x)
  );
  SUB col_1_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1151:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2017_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2026:113
    .y     (_col_1_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1127:103
    .z     (_col_1_n_tmp_6_z)
  );
  SHR_14 col_1_n_val_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1152:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1151:108
    .z     (_col_1_n_val_6_z)
  );
  CLIP col_1_n_clp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1153:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_val_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1152:90
    .z     (n_out_6_1_x)
  );
  SUB col_1_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1154:108
    .clock (clock),
    .reset (reset),
    .x     (_col_1_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1116:103
    .y     (_col_1_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1108:103
    .z     (_col_1_n_tmp_7_z)
  );
  SHR_14 col_1_n_val_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1155:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1154:108
    .z     (_col_1_n_val_7_z)
  );
  CLIP col_1_n_clp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1156:90
    .clock (clock),
    .reset (reset),
    .x     (_col_1_n_val_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1155:90
    .z     (n_out_7_1_x)
  );
  C4 col_2_n_c4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1157:73
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_c4_0_value)
  );
  C4 col_2_n_c4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1158:73
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_c4_1_value)
  );
  C4 col_2_n_c4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1159:73
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_c4_2_value)
  );
  C128 col_2_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1160:79
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_c128_0_value)
  );
  C128 col_2_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1161:79
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_c128_1_value)
  );
  C181 col_2_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1162:79
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_c181_0_value)
  );
  C181 col_2_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1163:79
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_c181_1_value)
  );
  C8192 col_2_n_c8192 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1164:76
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_c8192_value)
  );
  W7 col_2_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1165:67
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_w7_value)
  );
  W1_sub_W7 col_2_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1166:88
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_w1_sub_w7_value)
  );
  W1_add_W7 col_2_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1167:88
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_w1_add_w7_value)
  );
  W3 col_2_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1168:67
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_w3_value)
  );
  W3_sub_W5 col_2_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1169:88
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_w3_sub_w5_value)
  );
  W3_add_W5 col_2_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1170:88
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_w3_add_w5_value)
  );
  W6 col_2_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1171:67
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_w6_value)
  );
  W2_sub_W6 col_2_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1172:88
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_w2_sub_w6_value)
  );
  W2_add_W6 col_2_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1173:88
    .clock (clock),
    .reset (reset),
    .value (_col_2_n_w2_add_w6_value)
  );
  SHL_8 col_2_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1174:86
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_shr_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:616:90
    .z     (_col_2_n_x1_0_z)
  );
  SHL_8 col_2_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1175:86
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_shr_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:224:90
    .z     (_col_2_n_t0_0_z)
  );
  ADD col_2_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1176:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1175:86
    .y     (_col_2_n_c8192_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1164:76
    .z     (_col_2_n_x0_0_z)
  );
  dup_2 col_2_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1177:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1176:103
    .y     (_col_2_d_x0_0_y),
    .z     (_col_2_d_x0_0_z)
  );
  dup_2 col_2_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1178:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1174:86
    .y     (_col_2_d_x1_0_y),
    .z     (_col_2_d_x1_0_z)
  );
  dup_2 col_2_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1179:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_shr_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:812:90
    .y     (_col_2_d_x2_0_y),
    .z     (_col_2_d_x2_0_z)
  );
  dup_2 col_2_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1180:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_shr_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:420:90
    .y     (_col_2_d_x3_0_y),
    .z     (_col_2_d_x3_0_z)
  );
  dup_2 col_2_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1181:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_shr_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:322:90
    .y     (_col_2_d_x4_0_y),
    .z     (_col_2_d_x4_0_z)
  );
  dup_2 col_2_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1182:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_shr_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:910:90
    .y     (_col_2_d_x5_0_y),
    .z     (_col_2_d_x5_0_z)
  );
  dup_2 col_2_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1183:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_shr_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:714:90
    .y     (_col_2_d_x6_0_y),
    .z     (_col_2_d_x6_0_z)
  );
  dup_2 col_2_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1184:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_shr_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:518:90
    .y     (_col_2_d_x7_0_y),
    .z     (_col_2_d_x7_0_z)
  );
  ADD col_2_n_u8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1185:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1181:103
    .y     (_col_2_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1182:103
    .z     (_col_2_n_u8_0_z)
  );
  MUL col_2_n_v8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1186:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1165:67
    .y     (_col_2_n_u8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1185:103
    .z     (_col_2_n_v8_0_z)
  );
  ADD col_2_n_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1187:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1186:103
    .y     (_col_2_n_c4_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1157:73
    .z     (_col_2_n_x8_0_z)
  );
  dup_2 col_2_d_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1188:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1187:103
    .y     (_col_2_d_x8_0_y),
    .z     (_col_2_d_x8_0_z)
  );
  MUL col_2_n_u4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1189:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1166:88
    .y     (_col_2_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1181:103
    .z     (_col_2_n_u4_1_z)
  );
  ADD col_2_n_v4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1190:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x8_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1188:103
    .y     (_delay_INT16_2_2018_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2027:113
    .z     (_col_2_n_v4_1_z)
  );
  SHR_3 col_2_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1191:86
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1190:103
    .z     (_col_2_n_x4_1_z)
  );
  dup_2 col_2_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1192:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1191:86
    .y     (_col_2_d_x4_1_y),
    .z     (_col_2_d_x4_1_z)
  );
  MUL col_2_n_u5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1193:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1167:88
    .y     (_col_2_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1182:103
    .z     (_col_2_n_u5_1_z)
  );
  SUB col_2_n_v5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1194:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1188:103
    .y     (_delay_INT16_2_2019_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2028:113
    .z     (_col_2_n_v5_1_z)
  );
  SHR_3 col_2_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1195:86
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1194:103
    .z     (_col_2_n_x5_1_z)
  );
  dup_2 col_2_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1196:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1195:86
    .y     (_col_2_d_x5_1_y),
    .z     (_col_2_d_x5_1_z)
  );
  ADD col_2_n_u8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1197:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1183:103
    .y     (_col_2_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1184:103
    .z     (_col_2_n_u8_1_z)
  );
  MUL col_2_n_v8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1198:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1168:67
    .y     (_col_2_n_u8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1197:103
    .z     (_col_2_n_v8_1_z)
  );
  ADD col_2_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1199:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1198:103
    .y     (_col_2_n_c4_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1158:73
    .z     (_col_2_n_x8_1_z)
  );
  dup_2 col_2_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1200:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1199:103
    .y     (_col_2_d_x8_1_y),
    .z     (_col_2_d_x8_1_z)
  );
  MUL col_2_n_u6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1201:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1169:88
    .y     (_col_2_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1183:103
    .z     (_col_2_n_u6_1_z)
  );
  SUB col_2_n_v6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1202:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1200:103
    .y     (_delay_INT16_2_2020_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2029:113
    .z     (_col_2_n_v6_1_z)
  );
  SHR_3 col_2_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1203:86
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1202:103
    .z     (_col_2_n_x6_1_z)
  );
  dup_2 col_2_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1204:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1203:86
    .y     (_col_2_d_x6_1_y),
    .z     (_col_2_d_x6_1_z)
  );
  MUL col_2_n_u7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1205:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1170:88
    .y     (_col_2_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1184:103
    .z     (_col_2_n_u7_1_z)
  );
  SUB col_2_n_v7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1206:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1200:103
    .y     (_delay_INT16_2_2021_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2030:113
    .z     (_col_2_n_v7_1_z)
  );
  SHR_3 col_2_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1207:86
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1206:103
    .z     (_col_2_n_x7_1_z)
  );
  dup_2 col_2_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1208:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1207:86
    .y     (_col_2_d_x7_1_y),
    .z     (_col_2_d_x7_1_z)
  );
  ADD col_2_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1209:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1177:103
    .y     (_delay_INT16_1_2022_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2031:113
    .z     (_col_2_n_x8_2_z)
  );
  dup_2 col_2_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1210:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1209:103
    .y     (_col_2_d_x8_2_y),
    .z     (_col_2_d_x8_2_z)
  );
  SUB col_2_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1211:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1177:103
    .y     (_delay_INT16_1_2023_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2032:113
    .z     (_col_2_n_x0_1_z)
  );
  dup_2 col_2_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1212:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1211:103
    .y     (_col_2_d_x0_1_y),
    .z     (_col_2_d_x0_1_z)
  );
  ADD col_2_n_u1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1213:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1180:103
    .y     (_col_2_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1179:103
    .z     (_col_2_n_u1_1_z)
  );
  MUL col_2_n_v1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1214:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1171:67
    .y     (_col_2_n_u1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1213:103
    .z     (_col_2_n_v1_1_z)
  );
  ADD col_2_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1215:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1214:103
    .y     (_col_2_n_c4_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1159:73
    .z     (_col_2_n_x1_1_z)
  );
  dup_2 col_2_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1216:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1215:103
    .y     (_col_2_d_x1_1_y),
    .z     (_col_2_d_x1_1_z)
  );
  MUL col_2_n_u2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1217:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1173:88
    .y     (_col_2_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1179:103
    .z     (_col_2_n_u2_1_z)
  );
  SUB col_2_n_v2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1218:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1216:103
    .y     (_delay_INT16_2_2024_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2033:113
    .z     (_col_2_n_v2_1_z)
  );
  SHR_3 col_2_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1219:86
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1218:103
    .z     (_col_2_n_x2_1_z)
  );
  dup_2 col_2_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1220:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1219:86
    .y     (_col_2_d_x2_1_y),
    .z     (_col_2_d_x2_1_z)
  );
  MUL col_2_n_u3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1221:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1172:88
    .y     (_col_2_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1180:103
    .z     (_col_2_n_u3_1_z)
  );
  ADD col_2_n_v3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1222:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1216:103
    .y     (_delay_INT16_2_2025_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2034:113
    .z     (_col_2_n_v3_1_z)
  );
  SHR_3 col_2_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1223:86
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1222:103
    .z     (_col_2_n_x3_1_z)
  );
  dup_2 col_2_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1224:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1223:86
    .y     (_col_2_d_x3_1_y),
    .z     (_col_2_d_x3_1_z)
  );
  ADD col_2_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1225:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1192:103
    .y     (_col_2_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1204:103
    .z     (_col_2_n_x1_2_z)
  );
  dup_2 col_2_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1226:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1225:103
    .y     (_col_2_d_x1_2_y),
    .z     (_col_2_d_x1_2_z)
  );
  SUB col_2_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1227:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1192:103
    .y     (_col_2_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1204:103
    .z     (_col_2_n_x4_2_z)
  );
  dup_2 col_2_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1228:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1227:103
    .y     (_col_2_d_x4_2_y),
    .z     (_col_2_d_x4_2_z)
  );
  ADD col_2_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1229:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1196:103
    .y     (_col_2_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1208:103
    .z     (_col_2_n_x6_2_z)
  );
  dup_2 col_2_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1230:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1229:103
    .y     (_col_2_d_x6_2_y),
    .z     (_col_2_d_x6_2_z)
  );
  SUB col_2_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1231:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1196:103
    .y     (_col_2_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1208:103
    .z     (_col_2_n_x5_2_z)
  );
  dup_2 col_2_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1232:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1231:103
    .y     (_col_2_d_x5_2_y),
    .z     (_col_2_d_x5_2_z)
  );
  ADD col_2_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1233:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2026_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2035:113
    .y     (_col_2_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1224:103
    .z     (_col_2_n_x7_2_z)
  );
  dup_2 col_2_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1234:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1233:103
    .y     (_col_2_d_x7_2_y),
    .z     (_col_2_d_x7_2_z)
  );
  SUB col_2_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1235:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2027_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2036:113
    .y     (_col_2_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1224:103
    .z     (_col_2_n_x8_3_z)
  );
  dup_2 col_2_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1236:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1235:103
    .y     (_col_2_d_x8_3_y),
    .z     (_col_2_d_x8_3_z)
  );
  ADD col_2_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1237:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2028_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2037:113
    .y     (_col_2_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1220:103
    .z     (_col_2_n_x3_2_z)
  );
  dup_2 col_2_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1238:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1237:103
    .y     (_col_2_d_x3_2_y),
    .z     (_col_2_d_x3_2_z)
  );
  SUB col_2_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1239:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2029_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2038:113
    .y     (_col_2_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1220:103
    .z     (_col_2_n_x0_2_z)
  );
  dup_2 col_2_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1240:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1239:103
    .y     (_col_2_d_x0_2_y),
    .z     (_col_2_d_x0_2_z)
  );
  ADD col_2_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1241:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1228:103
    .y     (_col_2_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1232:103
    .z     (_col_2_n_u2_2_z)
  );
  MUL col_2_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1242:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1162:79
    .y     (_col_2_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1241:103
    .z     (_col_2_n_v2_2_z)
  );
  ADD col_2_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1243:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1242:103
    .y     (_col_2_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1160:79
    .z     (_col_2_n_w2_2_z)
  );
  SHR_8 col_2_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1244:86
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1243:103
    .z     (_col_2_n_x2_2_z)
  );
  dup_2 col_2_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1245:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1244:86
    .y     (_col_2_d_x2_2_y),
    .z     (_col_2_d_x2_2_z)
  );
  SUB col_2_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1246:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1228:103
    .y     (_col_2_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1232:103
    .z     (_col_2_n_u4_3_z)
  );
  MUL col_2_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1247:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1163:79
    .y     (_col_2_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1246:103
    .z     (_col_2_n_v4_3_z)
  );
  ADD col_2_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1248:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1247:103
    .y     (_col_2_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1161:79
    .z     (_col_2_n_w4_3_z)
  );
  SHR_8 col_2_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1249:86
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1248:103
    .z     (_col_2_n_x4_3_z)
  );
  dup_2 col_2_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1250:103
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1249:86
    .y     (_col_2_d_x4_3_y),
    .z     (_col_2_d_x4_3_z)
  );
  ADD col_2_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1251:108
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1234:103
    .y     (_col_2_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1226:103
    .z     (_col_2_n_tmp_0_z)
  );
  SHR_14 col_2_n_val_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1252:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1251:108
    .z     (_col_2_n_val_0_z)
  );
  CLIP col_2_n_clp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1253:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_val_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1252:90
    .z     (n_out_0_2_x)
  );
  ADD col_2_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1254:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2030_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2039:113
    .y     (_col_2_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1245:103
    .z     (_col_2_n_tmp_1_z)
  );
  SHR_14 col_2_n_val_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1255:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1254:108
    .z     (_col_2_n_val_1_z)
  );
  CLIP col_2_n_clp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1256:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_val_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1255:90
    .z     (n_out_1_2_x)
  );
  ADD col_2_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1257:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2031_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2040:113
    .y     (_col_2_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1250:103
    .z     (_col_2_n_tmp_2_z)
  );
  SHR_14 col_2_n_val_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1258:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1257:108
    .z     (_col_2_n_val_2_z)
  );
  CLIP col_2_n_clp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1259:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_val_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1258:90
    .z     (n_out_2_2_x)
  );
  ADD col_2_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1260:108
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1236:103
    .y     (_col_2_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1230:103
    .z     (_col_2_n_tmp_3_z)
  );
  SHR_14 col_2_n_val_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1261:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1260:108
    .z     (_col_2_n_val_3_z)
  );
  CLIP col_2_n_clp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1262:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_val_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1261:90
    .z     (n_out_3_2_x)
  );
  SUB col_2_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1263:108
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1236:103
    .y     (_col_2_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1230:103
    .z     (_col_2_n_tmp_4_z)
  );
  SHR_14 col_2_n_val_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1264:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1263:108
    .z     (_col_2_n_val_4_z)
  );
  CLIP col_2_n_clp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1265:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_val_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1264:90
    .z     (n_out_4_2_x)
  );
  SUB col_2_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1266:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2032_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2041:113
    .y     (_col_2_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1250:103
    .z     (_col_2_n_tmp_5_z)
  );
  SHR_14 col_2_n_val_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1267:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1266:108
    .z     (_col_2_n_val_5_z)
  );
  CLIP col_2_n_clp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1268:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_val_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1267:90
    .z     (n_out_5_2_x)
  );
  SUB col_2_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1269:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2033_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2042:113
    .y     (_col_2_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1245:103
    .z     (_col_2_n_tmp_6_z)
  );
  SHR_14 col_2_n_val_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1270:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1269:108
    .z     (_col_2_n_val_6_z)
  );
  CLIP col_2_n_clp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1271:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_val_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1270:90
    .z     (n_out_6_2_x)
  );
  SUB col_2_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1272:108
    .clock (clock),
    .reset (reset),
    .x     (_col_2_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1234:103
    .y     (_col_2_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1226:103
    .z     (_col_2_n_tmp_7_z)
  );
  SHR_14 col_2_n_val_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1273:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1272:108
    .z     (_col_2_n_val_7_z)
  );
  CLIP col_2_n_clp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1274:90
    .clock (clock),
    .reset (reset),
    .x     (_col_2_n_val_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1273:90
    .z     (n_out_7_2_x)
  );
  C4 col_3_n_c4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1275:73
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_c4_0_value)
  );
  C4 col_3_n_c4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1276:73
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_c4_1_value)
  );
  C4 col_3_n_c4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1277:73
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_c4_2_value)
  );
  C128 col_3_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1278:79
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_c128_0_value)
  );
  C128 col_3_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1279:79
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_c128_1_value)
  );
  C181 col_3_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1280:79
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_c181_0_value)
  );
  C181 col_3_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1281:79
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_c181_1_value)
  );
  C8192 col_3_n_c8192 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1282:76
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_c8192_value)
  );
  W7 col_3_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1283:67
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_w7_value)
  );
  W1_sub_W7 col_3_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1284:88
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_w1_sub_w7_value)
  );
  W1_add_W7 col_3_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1285:88
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_w1_add_w7_value)
  );
  W3 col_3_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1286:67
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_w3_value)
  );
  W3_sub_W5 col_3_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1287:88
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_w3_sub_w5_value)
  );
  W3_add_W5 col_3_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1288:88
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_w3_add_w5_value)
  );
  W6 col_3_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1289:67
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_w6_value)
  );
  W2_sub_W6 col_3_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1290:88
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_w2_sub_w6_value)
  );
  W2_add_W6 col_3_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1291:88
    .clock (clock),
    .reset (reset),
    .value (_col_3_n_w2_add_w6_value)
  );
  SHL_8 col_3_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1292:86
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_shr_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:618:90
    .z     (_col_3_n_x1_0_z)
  );
  SHL_8 col_3_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1293:86
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_shr_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:226:90
    .z     (_col_3_n_t0_0_z)
  );
  ADD col_3_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1294:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1293:86
    .y     (_col_3_n_c8192_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1282:76
    .z     (_col_3_n_x0_0_z)
  );
  dup_2 col_3_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1295:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1294:103
    .y     (_col_3_d_x0_0_y),
    .z     (_col_3_d_x0_0_z)
  );
  dup_2 col_3_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1296:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1292:86
    .y     (_col_3_d_x1_0_y),
    .z     (_col_3_d_x1_0_z)
  );
  dup_2 col_3_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1297:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_shr_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:814:90
    .y     (_col_3_d_x2_0_y),
    .z     (_col_3_d_x2_0_z)
  );
  dup_2 col_3_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1298:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_shr_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:422:90
    .y     (_col_3_d_x3_0_y),
    .z     (_col_3_d_x3_0_z)
  );
  dup_2 col_3_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1299:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_shr_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:324:90
    .y     (_col_3_d_x4_0_y),
    .z     (_col_3_d_x4_0_z)
  );
  dup_2 col_3_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1300:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_shr_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:912:90
    .y     (_col_3_d_x5_0_y),
    .z     (_col_3_d_x5_0_z)
  );
  dup_2 col_3_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1301:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_shr_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:716:90
    .y     (_col_3_d_x6_0_y),
    .z     (_col_3_d_x6_0_z)
  );
  dup_2 col_3_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1302:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_shr_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:520:90
    .y     (_col_3_d_x7_0_y),
    .z     (_col_3_d_x7_0_z)
  );
  ADD col_3_n_u8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1303:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1299:103
    .y     (_col_3_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1300:103
    .z     (_col_3_n_u8_0_z)
  );
  MUL col_3_n_v8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1304:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1283:67
    .y     (_col_3_n_u8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1303:103
    .z     (_col_3_n_v8_0_z)
  );
  ADD col_3_n_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1305:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1304:103
    .y     (_col_3_n_c4_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1275:73
    .z     (_col_3_n_x8_0_z)
  );
  dup_2 col_3_d_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1306:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1305:103
    .y     (_col_3_d_x8_0_y),
    .z     (_col_3_d_x8_0_z)
  );
  MUL col_3_n_u4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1307:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1284:88
    .y     (_col_3_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1299:103
    .z     (_col_3_n_u4_1_z)
  );
  ADD col_3_n_v4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1308:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x8_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1306:103
    .y     (_delay_INT16_2_2034_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2043:113
    .z     (_col_3_n_v4_1_z)
  );
  SHR_3 col_3_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1309:86
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1308:103
    .z     (_col_3_n_x4_1_z)
  );
  dup_2 col_3_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1310:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1309:86
    .y     (_col_3_d_x4_1_y),
    .z     (_col_3_d_x4_1_z)
  );
  MUL col_3_n_u5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1311:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1285:88
    .y     (_col_3_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1300:103
    .z     (_col_3_n_u5_1_z)
  );
  SUB col_3_n_v5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1312:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1306:103
    .y     (_delay_INT16_2_2035_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2044:113
    .z     (_col_3_n_v5_1_z)
  );
  SHR_3 col_3_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1313:86
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1312:103
    .z     (_col_3_n_x5_1_z)
  );
  dup_2 col_3_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1314:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1313:86
    .y     (_col_3_d_x5_1_y),
    .z     (_col_3_d_x5_1_z)
  );
  ADD col_3_n_u8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1315:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1301:103
    .y     (_col_3_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1302:103
    .z     (_col_3_n_u8_1_z)
  );
  MUL col_3_n_v8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1316:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1286:67
    .y     (_col_3_n_u8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1315:103
    .z     (_col_3_n_v8_1_z)
  );
  ADD col_3_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1317:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1316:103
    .y     (_col_3_n_c4_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1276:73
    .z     (_col_3_n_x8_1_z)
  );
  dup_2 col_3_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1318:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1317:103
    .y     (_col_3_d_x8_1_y),
    .z     (_col_3_d_x8_1_z)
  );
  MUL col_3_n_u6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1319:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1287:88
    .y     (_col_3_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1301:103
    .z     (_col_3_n_u6_1_z)
  );
  SUB col_3_n_v6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1320:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1318:103
    .y     (_delay_INT16_2_2036_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2045:113
    .z     (_col_3_n_v6_1_z)
  );
  SHR_3 col_3_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1321:86
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1320:103
    .z     (_col_3_n_x6_1_z)
  );
  dup_2 col_3_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1322:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1321:86
    .y     (_col_3_d_x6_1_y),
    .z     (_col_3_d_x6_1_z)
  );
  MUL col_3_n_u7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1323:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1288:88
    .y     (_col_3_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1302:103
    .z     (_col_3_n_u7_1_z)
  );
  SUB col_3_n_v7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1324:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1318:103
    .y     (_delay_INT16_2_2037_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2046:113
    .z     (_col_3_n_v7_1_z)
  );
  SHR_3 col_3_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1325:86
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1324:103
    .z     (_col_3_n_x7_1_z)
  );
  dup_2 col_3_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1326:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1325:86
    .y     (_col_3_d_x7_1_y),
    .z     (_col_3_d_x7_1_z)
  );
  ADD col_3_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1327:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1295:103
    .y     (_delay_INT16_1_2038_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2047:113
    .z     (_col_3_n_x8_2_z)
  );
  dup_2 col_3_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1328:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1327:103
    .y     (_col_3_d_x8_2_y),
    .z     (_col_3_d_x8_2_z)
  );
  SUB col_3_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1329:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1295:103
    .y     (_delay_INT16_1_2039_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2048:113
    .z     (_col_3_n_x0_1_z)
  );
  dup_2 col_3_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1330:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1329:103
    .y     (_col_3_d_x0_1_y),
    .z     (_col_3_d_x0_1_z)
  );
  ADD col_3_n_u1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1331:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1298:103
    .y     (_col_3_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1297:103
    .z     (_col_3_n_u1_1_z)
  );
  MUL col_3_n_v1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1332:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1289:67
    .y     (_col_3_n_u1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1331:103
    .z     (_col_3_n_v1_1_z)
  );
  ADD col_3_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1333:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1332:103
    .y     (_col_3_n_c4_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1277:73
    .z     (_col_3_n_x1_1_z)
  );
  dup_2 col_3_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1334:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1333:103
    .y     (_col_3_d_x1_1_y),
    .z     (_col_3_d_x1_1_z)
  );
  MUL col_3_n_u2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1335:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1291:88
    .y     (_col_3_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1297:103
    .z     (_col_3_n_u2_1_z)
  );
  SUB col_3_n_v2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1336:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1334:103
    .y     (_delay_INT16_2_2040_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2049:113
    .z     (_col_3_n_v2_1_z)
  );
  SHR_3 col_3_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1337:86
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1336:103
    .z     (_col_3_n_x2_1_z)
  );
  dup_2 col_3_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1338:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1337:86
    .y     (_col_3_d_x2_1_y),
    .z     (_col_3_d_x2_1_z)
  );
  MUL col_3_n_u3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1339:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1290:88
    .y     (_col_3_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1298:103
    .z     (_col_3_n_u3_1_z)
  );
  ADD col_3_n_v3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1340:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1334:103
    .y     (_delay_INT16_2_2041_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2050:113
    .z     (_col_3_n_v3_1_z)
  );
  SHR_3 col_3_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1341:86
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1340:103
    .z     (_col_3_n_x3_1_z)
  );
  dup_2 col_3_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1342:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1341:86
    .y     (_col_3_d_x3_1_y),
    .z     (_col_3_d_x3_1_z)
  );
  ADD col_3_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1343:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1310:103
    .y     (_col_3_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1322:103
    .z     (_col_3_n_x1_2_z)
  );
  dup_2 col_3_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1344:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1343:103
    .y     (_col_3_d_x1_2_y),
    .z     (_col_3_d_x1_2_z)
  );
  SUB col_3_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1345:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1310:103
    .y     (_col_3_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1322:103
    .z     (_col_3_n_x4_2_z)
  );
  dup_2 col_3_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1346:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1345:103
    .y     (_col_3_d_x4_2_y),
    .z     (_col_3_d_x4_2_z)
  );
  ADD col_3_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1347:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1314:103
    .y     (_col_3_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1326:103
    .z     (_col_3_n_x6_2_z)
  );
  dup_2 col_3_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1348:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1347:103
    .y     (_col_3_d_x6_2_y),
    .z     (_col_3_d_x6_2_z)
  );
  SUB col_3_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1349:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1314:103
    .y     (_col_3_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1326:103
    .z     (_col_3_n_x5_2_z)
  );
  dup_2 col_3_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1350:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1349:103
    .y     (_col_3_d_x5_2_y),
    .z     (_col_3_d_x5_2_z)
  );
  ADD col_3_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1351:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2042_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2051:113
    .y     (_col_3_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1342:103
    .z     (_col_3_n_x7_2_z)
  );
  dup_2 col_3_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1352:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1351:103
    .y     (_col_3_d_x7_2_y),
    .z     (_col_3_d_x7_2_z)
  );
  SUB col_3_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1353:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2043_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2052:113
    .y     (_col_3_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1342:103
    .z     (_col_3_n_x8_3_z)
  );
  dup_2 col_3_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1354:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1353:103
    .y     (_col_3_d_x8_3_y),
    .z     (_col_3_d_x8_3_z)
  );
  ADD col_3_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1355:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2044_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2053:113
    .y     (_col_3_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1338:103
    .z     (_col_3_n_x3_2_z)
  );
  dup_2 col_3_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1356:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1355:103
    .y     (_col_3_d_x3_2_y),
    .z     (_col_3_d_x3_2_z)
  );
  SUB col_3_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1357:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2045_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2054:113
    .y     (_col_3_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1338:103
    .z     (_col_3_n_x0_2_z)
  );
  dup_2 col_3_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1358:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1357:103
    .y     (_col_3_d_x0_2_y),
    .z     (_col_3_d_x0_2_z)
  );
  ADD col_3_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1359:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1346:103
    .y     (_col_3_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1350:103
    .z     (_col_3_n_u2_2_z)
  );
  MUL col_3_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1360:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1280:79
    .y     (_col_3_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1359:103
    .z     (_col_3_n_v2_2_z)
  );
  ADD col_3_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1361:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1360:103
    .y     (_col_3_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1278:79
    .z     (_col_3_n_w2_2_z)
  );
  SHR_8 col_3_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1362:86
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1361:103
    .z     (_col_3_n_x2_2_z)
  );
  dup_2 col_3_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1363:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1362:86
    .y     (_col_3_d_x2_2_y),
    .z     (_col_3_d_x2_2_z)
  );
  SUB col_3_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1364:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1346:103
    .y     (_col_3_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1350:103
    .z     (_col_3_n_u4_3_z)
  );
  MUL col_3_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1365:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1281:79
    .y     (_col_3_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1364:103
    .z     (_col_3_n_v4_3_z)
  );
  ADD col_3_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1366:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1365:103
    .y     (_col_3_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1279:79
    .z     (_col_3_n_w4_3_z)
  );
  SHR_8 col_3_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1367:86
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1366:103
    .z     (_col_3_n_x4_3_z)
  );
  dup_2 col_3_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1368:103
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1367:86
    .y     (_col_3_d_x4_3_y),
    .z     (_col_3_d_x4_3_z)
  );
  ADD col_3_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1369:108
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1352:103
    .y     (_col_3_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1344:103
    .z     (_col_3_n_tmp_0_z)
  );
  SHR_14 col_3_n_val_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1370:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1369:108
    .z     (_col_3_n_val_0_z)
  );
  CLIP col_3_n_clp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1371:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_val_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1370:90
    .z     (n_out_0_3_x)
  );
  ADD col_3_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1372:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2046_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2055:113
    .y     (_col_3_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1363:103
    .z     (_col_3_n_tmp_1_z)
  );
  SHR_14 col_3_n_val_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1373:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1372:108
    .z     (_col_3_n_val_1_z)
  );
  CLIP col_3_n_clp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1374:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_val_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1373:90
    .z     (n_out_1_3_x)
  );
  ADD col_3_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1375:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2047_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2056:113
    .y     (_col_3_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1368:103
    .z     (_col_3_n_tmp_2_z)
  );
  SHR_14 col_3_n_val_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1376:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1375:108
    .z     (_col_3_n_val_2_z)
  );
  CLIP col_3_n_clp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1377:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_val_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1376:90
    .z     (n_out_2_3_x)
  );
  ADD col_3_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1378:108
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1354:103
    .y     (_col_3_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1348:103
    .z     (_col_3_n_tmp_3_z)
  );
  SHR_14 col_3_n_val_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1379:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1378:108
    .z     (_col_3_n_val_3_z)
  );
  CLIP col_3_n_clp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1380:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_val_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1379:90
    .z     (n_out_3_3_x)
  );
  SUB col_3_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1381:108
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1354:103
    .y     (_col_3_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1348:103
    .z     (_col_3_n_tmp_4_z)
  );
  SHR_14 col_3_n_val_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1382:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1381:108
    .z     (_col_3_n_val_4_z)
  );
  CLIP col_3_n_clp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1383:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_val_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1382:90
    .z     (n_out_4_3_x)
  );
  SUB col_3_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1384:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2048_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2057:113
    .y     (_col_3_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1368:103
    .z     (_col_3_n_tmp_5_z)
  );
  SHR_14 col_3_n_val_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1385:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1384:108
    .z     (_col_3_n_val_5_z)
  );
  CLIP col_3_n_clp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1386:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_val_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1385:90
    .z     (n_out_5_3_x)
  );
  SUB col_3_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1387:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2049_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2058:113
    .y     (_col_3_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1363:103
    .z     (_col_3_n_tmp_6_z)
  );
  SHR_14 col_3_n_val_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1388:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1387:108
    .z     (_col_3_n_val_6_z)
  );
  CLIP col_3_n_clp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1389:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_val_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1388:90
    .z     (n_out_6_3_x)
  );
  SUB col_3_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1390:108
    .clock (clock),
    .reset (reset),
    .x     (_col_3_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1352:103
    .y     (_col_3_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1344:103
    .z     (_col_3_n_tmp_7_z)
  );
  SHR_14 col_3_n_val_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1391:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1390:108
    .z     (_col_3_n_val_7_z)
  );
  CLIP col_3_n_clp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1392:90
    .clock (clock),
    .reset (reset),
    .x     (_col_3_n_val_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1391:90
    .z     (n_out_7_3_x)
  );
  C4 col_4_n_c4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1393:73
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_c4_0_value)
  );
  C4 col_4_n_c4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1394:73
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_c4_1_value)
  );
  C4 col_4_n_c4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1395:73
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_c4_2_value)
  );
  C128 col_4_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1396:79
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_c128_0_value)
  );
  C128 col_4_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1397:79
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_c128_1_value)
  );
  C181 col_4_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1398:79
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_c181_0_value)
  );
  C181 col_4_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1399:79
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_c181_1_value)
  );
  C8192 col_4_n_c8192 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1400:76
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_c8192_value)
  );
  W7 col_4_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1401:67
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_w7_value)
  );
  W1_sub_W7 col_4_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1402:88
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_w1_sub_w7_value)
  );
  W1_add_W7 col_4_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1403:88
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_w1_add_w7_value)
  );
  W3 col_4_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1404:67
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_w3_value)
  );
  W3_sub_W5 col_4_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1405:88
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_w3_sub_w5_value)
  );
  W3_add_W5 col_4_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1406:88
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_w3_add_w5_value)
  );
  W6 col_4_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1407:67
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_w6_value)
  );
  W2_sub_W6 col_4_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1408:88
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_w2_sub_w6_value)
  );
  W2_add_W6 col_4_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1409:88
    .clock (clock),
    .reset (reset),
    .value (_col_4_n_w2_add_w6_value)
  );
  SHL_8 col_4_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1410:86
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_shr_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:620:90
    .z     (_col_4_n_x1_0_z)
  );
  SHL_8 col_4_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1411:86
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_shr_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:228:90
    .z     (_col_4_n_t0_0_z)
  );
  ADD col_4_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1412:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1411:86
    .y     (_col_4_n_c8192_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1400:76
    .z     (_col_4_n_x0_0_z)
  );
  dup_2 col_4_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1413:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1412:103
    .y     (_col_4_d_x0_0_y),
    .z     (_col_4_d_x0_0_z)
  );
  dup_2 col_4_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1414:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1410:86
    .y     (_col_4_d_x1_0_y),
    .z     (_col_4_d_x1_0_z)
  );
  dup_2 col_4_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1415:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_shr_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:816:90
    .y     (_col_4_d_x2_0_y),
    .z     (_col_4_d_x2_0_z)
  );
  dup_2 col_4_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1416:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_shr_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:424:90
    .y     (_col_4_d_x3_0_y),
    .z     (_col_4_d_x3_0_z)
  );
  dup_2 col_4_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1417:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_shr_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:326:90
    .y     (_col_4_d_x4_0_y),
    .z     (_col_4_d_x4_0_z)
  );
  dup_2 col_4_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1418:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_shr_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:914:90
    .y     (_col_4_d_x5_0_y),
    .z     (_col_4_d_x5_0_z)
  );
  dup_2 col_4_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1419:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_shr_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:718:90
    .y     (_col_4_d_x6_0_y),
    .z     (_col_4_d_x6_0_z)
  );
  dup_2 col_4_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1420:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_shr_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:522:90
    .y     (_col_4_d_x7_0_y),
    .z     (_col_4_d_x7_0_z)
  );
  ADD col_4_n_u8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1421:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1417:103
    .y     (_col_4_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1418:103
    .z     (_col_4_n_u8_0_z)
  );
  MUL col_4_n_v8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1422:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1401:67
    .y     (_col_4_n_u8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1421:103
    .z     (_col_4_n_v8_0_z)
  );
  ADD col_4_n_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1423:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1422:103
    .y     (_col_4_n_c4_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1393:73
    .z     (_col_4_n_x8_0_z)
  );
  dup_2 col_4_d_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1424:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1423:103
    .y     (_col_4_d_x8_0_y),
    .z     (_col_4_d_x8_0_z)
  );
  MUL col_4_n_u4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1425:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1402:88
    .y     (_col_4_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1417:103
    .z     (_col_4_n_u4_1_z)
  );
  ADD col_4_n_v4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1426:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x8_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1424:103
    .y     (_delay_INT16_2_2050_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2059:113
    .z     (_col_4_n_v4_1_z)
  );
  SHR_3 col_4_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1427:86
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1426:103
    .z     (_col_4_n_x4_1_z)
  );
  dup_2 col_4_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1428:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1427:86
    .y     (_col_4_d_x4_1_y),
    .z     (_col_4_d_x4_1_z)
  );
  MUL col_4_n_u5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1429:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1403:88
    .y     (_col_4_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1418:103
    .z     (_col_4_n_u5_1_z)
  );
  SUB col_4_n_v5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1430:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1424:103
    .y     (_delay_INT16_2_2051_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2060:113
    .z     (_col_4_n_v5_1_z)
  );
  SHR_3 col_4_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1431:86
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1430:103
    .z     (_col_4_n_x5_1_z)
  );
  dup_2 col_4_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1432:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1431:86
    .y     (_col_4_d_x5_1_y),
    .z     (_col_4_d_x5_1_z)
  );
  ADD col_4_n_u8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1433:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1419:103
    .y     (_col_4_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1420:103
    .z     (_col_4_n_u8_1_z)
  );
  MUL col_4_n_v8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1434:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1404:67
    .y     (_col_4_n_u8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1433:103
    .z     (_col_4_n_v8_1_z)
  );
  ADD col_4_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1435:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1434:103
    .y     (_col_4_n_c4_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1394:73
    .z     (_col_4_n_x8_1_z)
  );
  dup_2 col_4_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1436:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1435:103
    .y     (_col_4_d_x8_1_y),
    .z     (_col_4_d_x8_1_z)
  );
  MUL col_4_n_u6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1437:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1405:88
    .y     (_col_4_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1419:103
    .z     (_col_4_n_u6_1_z)
  );
  SUB col_4_n_v6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1438:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1436:103
    .y     (_delay_INT16_2_2052_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2061:113
    .z     (_col_4_n_v6_1_z)
  );
  SHR_3 col_4_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1439:86
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1438:103
    .z     (_col_4_n_x6_1_z)
  );
  dup_2 col_4_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1440:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1439:86
    .y     (_col_4_d_x6_1_y),
    .z     (_col_4_d_x6_1_z)
  );
  MUL col_4_n_u7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1441:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1406:88
    .y     (_col_4_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1420:103
    .z     (_col_4_n_u7_1_z)
  );
  SUB col_4_n_v7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1442:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1436:103
    .y     (_delay_INT16_2_2053_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2062:113
    .z     (_col_4_n_v7_1_z)
  );
  SHR_3 col_4_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1443:86
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1442:103
    .z     (_col_4_n_x7_1_z)
  );
  dup_2 col_4_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1444:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1443:86
    .y     (_col_4_d_x7_1_y),
    .z     (_col_4_d_x7_1_z)
  );
  ADD col_4_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1445:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1413:103
    .y     (_delay_INT16_1_2054_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2063:113
    .z     (_col_4_n_x8_2_z)
  );
  dup_2 col_4_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1446:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1445:103
    .y     (_col_4_d_x8_2_y),
    .z     (_col_4_d_x8_2_z)
  );
  SUB col_4_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1447:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1413:103
    .y     (_delay_INT16_1_2055_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2064:113
    .z     (_col_4_n_x0_1_z)
  );
  dup_2 col_4_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1448:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1447:103
    .y     (_col_4_d_x0_1_y),
    .z     (_col_4_d_x0_1_z)
  );
  ADD col_4_n_u1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1449:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1416:103
    .y     (_col_4_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1415:103
    .z     (_col_4_n_u1_1_z)
  );
  MUL col_4_n_v1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1450:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1407:67
    .y     (_col_4_n_u1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1449:103
    .z     (_col_4_n_v1_1_z)
  );
  ADD col_4_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1451:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1450:103
    .y     (_col_4_n_c4_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1395:73
    .z     (_col_4_n_x1_1_z)
  );
  dup_2 col_4_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1452:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1451:103
    .y     (_col_4_d_x1_1_y),
    .z     (_col_4_d_x1_1_z)
  );
  MUL col_4_n_u2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1453:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1409:88
    .y     (_col_4_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1415:103
    .z     (_col_4_n_u2_1_z)
  );
  SUB col_4_n_v2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1454:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1452:103
    .y     (_delay_INT16_2_2056_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2065:113
    .z     (_col_4_n_v2_1_z)
  );
  SHR_3 col_4_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1455:86
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1454:103
    .z     (_col_4_n_x2_1_z)
  );
  dup_2 col_4_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1456:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1455:86
    .y     (_col_4_d_x2_1_y),
    .z     (_col_4_d_x2_1_z)
  );
  MUL col_4_n_u3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1457:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1408:88
    .y     (_col_4_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1416:103
    .z     (_col_4_n_u3_1_z)
  );
  ADD col_4_n_v3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1458:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1452:103
    .y     (_delay_INT16_2_2057_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2066:113
    .z     (_col_4_n_v3_1_z)
  );
  SHR_3 col_4_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1459:86
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1458:103
    .z     (_col_4_n_x3_1_z)
  );
  dup_2 col_4_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1460:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1459:86
    .y     (_col_4_d_x3_1_y),
    .z     (_col_4_d_x3_1_z)
  );
  ADD col_4_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1461:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1428:103
    .y     (_col_4_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1440:103
    .z     (_col_4_n_x1_2_z)
  );
  dup_2 col_4_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1462:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1461:103
    .y     (_col_4_d_x1_2_y),
    .z     (_col_4_d_x1_2_z)
  );
  SUB col_4_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1463:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1428:103
    .y     (_col_4_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1440:103
    .z     (_col_4_n_x4_2_z)
  );
  dup_2 col_4_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1464:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1463:103
    .y     (_col_4_d_x4_2_y),
    .z     (_col_4_d_x4_2_z)
  );
  ADD col_4_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1465:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1432:103
    .y     (_col_4_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1444:103
    .z     (_col_4_n_x6_2_z)
  );
  dup_2 col_4_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1466:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1465:103
    .y     (_col_4_d_x6_2_y),
    .z     (_col_4_d_x6_2_z)
  );
  SUB col_4_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1467:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1432:103
    .y     (_col_4_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1444:103
    .z     (_col_4_n_x5_2_z)
  );
  dup_2 col_4_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1468:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1467:103
    .y     (_col_4_d_x5_2_y),
    .z     (_col_4_d_x5_2_z)
  );
  ADD col_4_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1469:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2058_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2067:113
    .y     (_col_4_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1460:103
    .z     (_col_4_n_x7_2_z)
  );
  dup_2 col_4_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1470:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1469:103
    .y     (_col_4_d_x7_2_y),
    .z     (_col_4_d_x7_2_z)
  );
  SUB col_4_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1471:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2059_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2068:113
    .y     (_col_4_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1460:103
    .z     (_col_4_n_x8_3_z)
  );
  dup_2 col_4_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1472:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1471:103
    .y     (_col_4_d_x8_3_y),
    .z     (_col_4_d_x8_3_z)
  );
  ADD col_4_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1473:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2060_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2069:113
    .y     (_col_4_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1456:103
    .z     (_col_4_n_x3_2_z)
  );
  dup_2 col_4_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1474:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1473:103
    .y     (_col_4_d_x3_2_y),
    .z     (_col_4_d_x3_2_z)
  );
  SUB col_4_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1475:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2061_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2070:113
    .y     (_col_4_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1456:103
    .z     (_col_4_n_x0_2_z)
  );
  dup_2 col_4_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1476:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1475:103
    .y     (_col_4_d_x0_2_y),
    .z     (_col_4_d_x0_2_z)
  );
  ADD col_4_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1477:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1464:103
    .y     (_col_4_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1468:103
    .z     (_col_4_n_u2_2_z)
  );
  MUL col_4_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1478:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1398:79
    .y     (_col_4_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1477:103
    .z     (_col_4_n_v2_2_z)
  );
  ADD col_4_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1479:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1478:103
    .y     (_col_4_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1396:79
    .z     (_col_4_n_w2_2_z)
  );
  SHR_8 col_4_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1480:86
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1479:103
    .z     (_col_4_n_x2_2_z)
  );
  dup_2 col_4_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1481:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1480:86
    .y     (_col_4_d_x2_2_y),
    .z     (_col_4_d_x2_2_z)
  );
  SUB col_4_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1482:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1464:103
    .y     (_col_4_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1468:103
    .z     (_col_4_n_u4_3_z)
  );
  MUL col_4_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1483:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1399:79
    .y     (_col_4_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1482:103
    .z     (_col_4_n_v4_3_z)
  );
  ADD col_4_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1484:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1483:103
    .y     (_col_4_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1397:79
    .z     (_col_4_n_w4_3_z)
  );
  SHR_8 col_4_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1485:86
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1484:103
    .z     (_col_4_n_x4_3_z)
  );
  dup_2 col_4_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1486:103
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1485:86
    .y     (_col_4_d_x4_3_y),
    .z     (_col_4_d_x4_3_z)
  );
  ADD col_4_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1487:108
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1470:103
    .y     (_col_4_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1462:103
    .z     (_col_4_n_tmp_0_z)
  );
  SHR_14 col_4_n_val_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1488:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1487:108
    .z     (_col_4_n_val_0_z)
  );
  CLIP col_4_n_clp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1489:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_val_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1488:90
    .z     (n_out_0_4_x)
  );
  ADD col_4_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1490:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2062_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2071:113
    .y     (_col_4_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1481:103
    .z     (_col_4_n_tmp_1_z)
  );
  SHR_14 col_4_n_val_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1491:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1490:108
    .z     (_col_4_n_val_1_z)
  );
  CLIP col_4_n_clp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1492:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_val_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1491:90
    .z     (n_out_1_4_x)
  );
  ADD col_4_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1493:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2063_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2072:113
    .y     (_col_4_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1486:103
    .z     (_col_4_n_tmp_2_z)
  );
  SHR_14 col_4_n_val_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1494:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1493:108
    .z     (_col_4_n_val_2_z)
  );
  CLIP col_4_n_clp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1495:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_val_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1494:90
    .z     (n_out_2_4_x)
  );
  ADD col_4_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1496:108
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1472:103
    .y     (_col_4_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1466:103
    .z     (_col_4_n_tmp_3_z)
  );
  SHR_14 col_4_n_val_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1497:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1496:108
    .z     (_col_4_n_val_3_z)
  );
  CLIP col_4_n_clp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1498:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_val_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1497:90
    .z     (n_out_3_4_x)
  );
  SUB col_4_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1499:108
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1472:103
    .y     (_col_4_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1466:103
    .z     (_col_4_n_tmp_4_z)
  );
  SHR_14 col_4_n_val_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1500:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1499:108
    .z     (_col_4_n_val_4_z)
  );
  CLIP col_4_n_clp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1501:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_val_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1500:90
    .z     (n_out_4_4_x)
  );
  SUB col_4_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1502:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2064_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2073:113
    .y     (_col_4_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1486:103
    .z     (_col_4_n_tmp_5_z)
  );
  SHR_14 col_4_n_val_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1503:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1502:108
    .z     (_col_4_n_val_5_z)
  );
  CLIP col_4_n_clp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1504:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_val_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1503:90
    .z     (n_out_5_4_x)
  );
  SUB col_4_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1505:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2065_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2074:113
    .y     (_col_4_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1481:103
    .z     (_col_4_n_tmp_6_z)
  );
  SHR_14 col_4_n_val_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1506:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1505:108
    .z     (_col_4_n_val_6_z)
  );
  CLIP col_4_n_clp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1507:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_val_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1506:90
    .z     (n_out_6_4_x)
  );
  SUB col_4_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1508:108
    .clock (clock),
    .reset (reset),
    .x     (_col_4_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1470:103
    .y     (_col_4_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1462:103
    .z     (_col_4_n_tmp_7_z)
  );
  SHR_14 col_4_n_val_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1509:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1508:108
    .z     (_col_4_n_val_7_z)
  );
  CLIP col_4_n_clp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1510:90
    .clock (clock),
    .reset (reset),
    .x     (_col_4_n_val_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1509:90
    .z     (n_out_7_4_x)
  );
  C4 col_5_n_c4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1511:73
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_c4_0_value)
  );
  C4 col_5_n_c4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1512:73
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_c4_1_value)
  );
  C4 col_5_n_c4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1513:73
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_c4_2_value)
  );
  C128 col_5_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1514:79
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_c128_0_value)
  );
  C128 col_5_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1515:79
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_c128_1_value)
  );
  C181 col_5_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1516:79
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_c181_0_value)
  );
  C181 col_5_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1517:79
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_c181_1_value)
  );
  C8192 col_5_n_c8192 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1518:76
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_c8192_value)
  );
  W7 col_5_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1519:67
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_w7_value)
  );
  W1_sub_W7 col_5_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1520:88
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_w1_sub_w7_value)
  );
  W1_add_W7 col_5_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1521:88
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_w1_add_w7_value)
  );
  W3 col_5_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1522:67
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_w3_value)
  );
  W3_sub_W5 col_5_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1523:88
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_w3_sub_w5_value)
  );
  W3_add_W5 col_5_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1524:88
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_w3_add_w5_value)
  );
  W6 col_5_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1525:67
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_w6_value)
  );
  W2_sub_W6 col_5_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1526:88
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_w2_sub_w6_value)
  );
  W2_add_W6 col_5_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1527:88
    .clock (clock),
    .reset (reset),
    .value (_col_5_n_w2_add_w6_value)
  );
  SHL_8 col_5_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1528:86
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_shr_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:622:90
    .z     (_col_5_n_x1_0_z)
  );
  SHL_8 col_5_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1529:86
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_shr_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:230:90
    .z     (_col_5_n_t0_0_z)
  );
  ADD col_5_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1530:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1529:86
    .y     (_col_5_n_c8192_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1518:76
    .z     (_col_5_n_x0_0_z)
  );
  dup_2 col_5_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1531:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1530:103
    .y     (_col_5_d_x0_0_y),
    .z     (_col_5_d_x0_0_z)
  );
  dup_2 col_5_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1532:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1528:86
    .y     (_col_5_d_x1_0_y),
    .z     (_col_5_d_x1_0_z)
  );
  dup_2 col_5_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1533:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_shr_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:818:90
    .y     (_col_5_d_x2_0_y),
    .z     (_col_5_d_x2_0_z)
  );
  dup_2 col_5_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1534:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_shr_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:426:90
    .y     (_col_5_d_x3_0_y),
    .z     (_col_5_d_x3_0_z)
  );
  dup_2 col_5_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1535:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_shr_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:328:90
    .y     (_col_5_d_x4_0_y),
    .z     (_col_5_d_x4_0_z)
  );
  dup_2 col_5_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1536:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_shr_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:916:90
    .y     (_col_5_d_x5_0_y),
    .z     (_col_5_d_x5_0_z)
  );
  dup_2 col_5_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1537:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_shr_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:720:90
    .y     (_col_5_d_x6_0_y),
    .z     (_col_5_d_x6_0_z)
  );
  dup_2 col_5_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1538:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_shr_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:524:90
    .y     (_col_5_d_x7_0_y),
    .z     (_col_5_d_x7_0_z)
  );
  ADD col_5_n_u8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1539:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1535:103
    .y     (_col_5_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1536:103
    .z     (_col_5_n_u8_0_z)
  );
  MUL col_5_n_v8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1540:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1519:67
    .y     (_col_5_n_u8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1539:103
    .z     (_col_5_n_v8_0_z)
  );
  ADD col_5_n_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1541:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1540:103
    .y     (_col_5_n_c4_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1511:73
    .z     (_col_5_n_x8_0_z)
  );
  dup_2 col_5_d_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1542:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1541:103
    .y     (_col_5_d_x8_0_y),
    .z     (_col_5_d_x8_0_z)
  );
  MUL col_5_n_u4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1543:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1520:88
    .y     (_col_5_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1535:103
    .z     (_col_5_n_u4_1_z)
  );
  ADD col_5_n_v4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1544:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x8_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1542:103
    .y     (_delay_INT16_2_2066_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2075:113
    .z     (_col_5_n_v4_1_z)
  );
  SHR_3 col_5_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1545:86
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1544:103
    .z     (_col_5_n_x4_1_z)
  );
  dup_2 col_5_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1546:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1545:86
    .y     (_col_5_d_x4_1_y),
    .z     (_col_5_d_x4_1_z)
  );
  MUL col_5_n_u5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1547:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1521:88
    .y     (_col_5_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1536:103
    .z     (_col_5_n_u5_1_z)
  );
  SUB col_5_n_v5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1548:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1542:103
    .y     (_delay_INT16_2_2067_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2076:113
    .z     (_col_5_n_v5_1_z)
  );
  SHR_3 col_5_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1549:86
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1548:103
    .z     (_col_5_n_x5_1_z)
  );
  dup_2 col_5_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1550:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1549:86
    .y     (_col_5_d_x5_1_y),
    .z     (_col_5_d_x5_1_z)
  );
  ADD col_5_n_u8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1551:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1537:103
    .y     (_col_5_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1538:103
    .z     (_col_5_n_u8_1_z)
  );
  MUL col_5_n_v8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1552:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1522:67
    .y     (_col_5_n_u8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1551:103
    .z     (_col_5_n_v8_1_z)
  );
  ADD col_5_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1553:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1552:103
    .y     (_col_5_n_c4_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1512:73
    .z     (_col_5_n_x8_1_z)
  );
  dup_2 col_5_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1554:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1553:103
    .y     (_col_5_d_x8_1_y),
    .z     (_col_5_d_x8_1_z)
  );
  MUL col_5_n_u6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1555:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1523:88
    .y     (_col_5_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1537:103
    .z     (_col_5_n_u6_1_z)
  );
  SUB col_5_n_v6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1556:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1554:103
    .y     (_delay_INT16_2_2068_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2077:113
    .z     (_col_5_n_v6_1_z)
  );
  SHR_3 col_5_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1557:86
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1556:103
    .z     (_col_5_n_x6_1_z)
  );
  dup_2 col_5_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1558:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1557:86
    .y     (_col_5_d_x6_1_y),
    .z     (_col_5_d_x6_1_z)
  );
  MUL col_5_n_u7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1559:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1524:88
    .y     (_col_5_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1538:103
    .z     (_col_5_n_u7_1_z)
  );
  SUB col_5_n_v7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1560:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1554:103
    .y     (_delay_INT16_2_2069_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2078:113
    .z     (_col_5_n_v7_1_z)
  );
  SHR_3 col_5_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1561:86
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1560:103
    .z     (_col_5_n_x7_1_z)
  );
  dup_2 col_5_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1562:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1561:86
    .y     (_col_5_d_x7_1_y),
    .z     (_col_5_d_x7_1_z)
  );
  ADD col_5_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1563:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1531:103
    .y     (_delay_INT16_1_2070_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2079:113
    .z     (_col_5_n_x8_2_z)
  );
  dup_2 col_5_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1564:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1563:103
    .y     (_col_5_d_x8_2_y),
    .z     (_col_5_d_x8_2_z)
  );
  SUB col_5_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1565:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1531:103
    .y     (_delay_INT16_1_2071_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2080:113
    .z     (_col_5_n_x0_1_z)
  );
  dup_2 col_5_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1566:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1565:103
    .y     (_col_5_d_x0_1_y),
    .z     (_col_5_d_x0_1_z)
  );
  ADD col_5_n_u1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1567:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1534:103
    .y     (_col_5_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1533:103
    .z     (_col_5_n_u1_1_z)
  );
  MUL col_5_n_v1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1568:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1525:67
    .y     (_col_5_n_u1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1567:103
    .z     (_col_5_n_v1_1_z)
  );
  ADD col_5_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1569:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1568:103
    .y     (_col_5_n_c4_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1513:73
    .z     (_col_5_n_x1_1_z)
  );
  dup_2 col_5_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1570:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1569:103
    .y     (_col_5_d_x1_1_y),
    .z     (_col_5_d_x1_1_z)
  );
  MUL col_5_n_u2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1571:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1527:88
    .y     (_col_5_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1533:103
    .z     (_col_5_n_u2_1_z)
  );
  SUB col_5_n_v2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1572:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1570:103
    .y     (_delay_INT16_2_2072_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2081:113
    .z     (_col_5_n_v2_1_z)
  );
  SHR_3 col_5_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1573:86
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1572:103
    .z     (_col_5_n_x2_1_z)
  );
  dup_2 col_5_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1574:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1573:86
    .y     (_col_5_d_x2_1_y),
    .z     (_col_5_d_x2_1_z)
  );
  MUL col_5_n_u3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1575:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1526:88
    .y     (_col_5_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1534:103
    .z     (_col_5_n_u3_1_z)
  );
  ADD col_5_n_v3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1576:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1570:103
    .y     (_delay_INT16_2_2073_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2082:113
    .z     (_col_5_n_v3_1_z)
  );
  SHR_3 col_5_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1577:86
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1576:103
    .z     (_col_5_n_x3_1_z)
  );
  dup_2 col_5_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1578:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1577:86
    .y     (_col_5_d_x3_1_y),
    .z     (_col_5_d_x3_1_z)
  );
  ADD col_5_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1579:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1546:103
    .y     (_col_5_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1558:103
    .z     (_col_5_n_x1_2_z)
  );
  dup_2 col_5_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1580:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1579:103
    .y     (_col_5_d_x1_2_y),
    .z     (_col_5_d_x1_2_z)
  );
  SUB col_5_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1581:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1546:103
    .y     (_col_5_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1558:103
    .z     (_col_5_n_x4_2_z)
  );
  dup_2 col_5_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1582:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1581:103
    .y     (_col_5_d_x4_2_y),
    .z     (_col_5_d_x4_2_z)
  );
  ADD col_5_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1583:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1550:103
    .y     (_col_5_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1562:103
    .z     (_col_5_n_x6_2_z)
  );
  dup_2 col_5_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1584:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1583:103
    .y     (_col_5_d_x6_2_y),
    .z     (_col_5_d_x6_2_z)
  );
  SUB col_5_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1585:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1550:103
    .y     (_col_5_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1562:103
    .z     (_col_5_n_x5_2_z)
  );
  dup_2 col_5_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1586:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1585:103
    .y     (_col_5_d_x5_2_y),
    .z     (_col_5_d_x5_2_z)
  );
  ADD col_5_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1587:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2074_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2083:113
    .y     (_col_5_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1578:103
    .z     (_col_5_n_x7_2_z)
  );
  dup_2 col_5_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1588:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1587:103
    .y     (_col_5_d_x7_2_y),
    .z     (_col_5_d_x7_2_z)
  );
  SUB col_5_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1589:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2075_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2084:113
    .y     (_col_5_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1578:103
    .z     (_col_5_n_x8_3_z)
  );
  dup_2 col_5_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1590:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1589:103
    .y     (_col_5_d_x8_3_y),
    .z     (_col_5_d_x8_3_z)
  );
  ADD col_5_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1591:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2076_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2085:113
    .y     (_col_5_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1574:103
    .z     (_col_5_n_x3_2_z)
  );
  dup_2 col_5_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1592:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1591:103
    .y     (_col_5_d_x3_2_y),
    .z     (_col_5_d_x3_2_z)
  );
  SUB col_5_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1593:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2077_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2086:113
    .y     (_col_5_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1574:103
    .z     (_col_5_n_x0_2_z)
  );
  dup_2 col_5_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1594:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1593:103
    .y     (_col_5_d_x0_2_y),
    .z     (_col_5_d_x0_2_z)
  );
  ADD col_5_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1595:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1582:103
    .y     (_col_5_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1586:103
    .z     (_col_5_n_u2_2_z)
  );
  MUL col_5_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1596:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1516:79
    .y     (_col_5_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1595:103
    .z     (_col_5_n_v2_2_z)
  );
  ADD col_5_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1597:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1596:103
    .y     (_col_5_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1514:79
    .z     (_col_5_n_w2_2_z)
  );
  SHR_8 col_5_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1598:86
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1597:103
    .z     (_col_5_n_x2_2_z)
  );
  dup_2 col_5_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1599:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1598:86
    .y     (_col_5_d_x2_2_y),
    .z     (_col_5_d_x2_2_z)
  );
  SUB col_5_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1600:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1582:103
    .y     (_col_5_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1586:103
    .z     (_col_5_n_u4_3_z)
  );
  MUL col_5_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1601:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1517:79
    .y     (_col_5_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1600:103
    .z     (_col_5_n_v4_3_z)
  );
  ADD col_5_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1602:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1601:103
    .y     (_col_5_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1515:79
    .z     (_col_5_n_w4_3_z)
  );
  SHR_8 col_5_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1603:86
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1602:103
    .z     (_col_5_n_x4_3_z)
  );
  dup_2 col_5_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1604:103
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1603:86
    .y     (_col_5_d_x4_3_y),
    .z     (_col_5_d_x4_3_z)
  );
  ADD col_5_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1605:108
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1588:103
    .y     (_col_5_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1580:103
    .z     (_col_5_n_tmp_0_z)
  );
  SHR_14 col_5_n_val_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1606:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1605:108
    .z     (_col_5_n_val_0_z)
  );
  CLIP col_5_n_clp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1607:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_val_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1606:90
    .z     (n_out_0_5_x)
  );
  ADD col_5_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1608:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2078_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2087:113
    .y     (_col_5_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1599:103
    .z     (_col_5_n_tmp_1_z)
  );
  SHR_14 col_5_n_val_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1609:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1608:108
    .z     (_col_5_n_val_1_z)
  );
  CLIP col_5_n_clp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1610:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_val_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1609:90
    .z     (n_out_1_5_x)
  );
  ADD col_5_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1611:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2079_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2088:113
    .y     (_col_5_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1604:103
    .z     (_col_5_n_tmp_2_z)
  );
  SHR_14 col_5_n_val_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1612:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1611:108
    .z     (_col_5_n_val_2_z)
  );
  CLIP col_5_n_clp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1613:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_val_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1612:90
    .z     (n_out_2_5_x)
  );
  ADD col_5_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1614:108
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1590:103
    .y     (_col_5_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1584:103
    .z     (_col_5_n_tmp_3_z)
  );
  SHR_14 col_5_n_val_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1615:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1614:108
    .z     (_col_5_n_val_3_z)
  );
  CLIP col_5_n_clp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1616:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_val_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1615:90
    .z     (n_out_3_5_x)
  );
  SUB col_5_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1617:108
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1590:103
    .y     (_col_5_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1584:103
    .z     (_col_5_n_tmp_4_z)
  );
  SHR_14 col_5_n_val_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1618:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1617:108
    .z     (_col_5_n_val_4_z)
  );
  CLIP col_5_n_clp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1619:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_val_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1618:90
    .z     (n_out_4_5_x)
  );
  SUB col_5_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1620:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2080_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2089:113
    .y     (_col_5_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1604:103
    .z     (_col_5_n_tmp_5_z)
  );
  SHR_14 col_5_n_val_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1621:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1620:108
    .z     (_col_5_n_val_5_z)
  );
  CLIP col_5_n_clp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1622:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_val_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1621:90
    .z     (n_out_5_5_x)
  );
  SUB col_5_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1623:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2081_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2090:113
    .y     (_col_5_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1599:103
    .z     (_col_5_n_tmp_6_z)
  );
  SHR_14 col_5_n_val_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1624:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1623:108
    .z     (_col_5_n_val_6_z)
  );
  CLIP col_5_n_clp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1625:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_val_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1624:90
    .z     (n_out_6_5_x)
  );
  SUB col_5_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1626:108
    .clock (clock),
    .reset (reset),
    .x     (_col_5_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1588:103
    .y     (_col_5_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1580:103
    .z     (_col_5_n_tmp_7_z)
  );
  SHR_14 col_5_n_val_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1627:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1626:108
    .z     (_col_5_n_val_7_z)
  );
  CLIP col_5_n_clp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1628:90
    .clock (clock),
    .reset (reset),
    .x     (_col_5_n_val_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1627:90
    .z     (n_out_7_5_x)
  );
  C4 col_6_n_c4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1629:73
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_c4_0_value)
  );
  C4 col_6_n_c4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1630:73
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_c4_1_value)
  );
  C4 col_6_n_c4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1631:73
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_c4_2_value)
  );
  C128 col_6_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1632:79
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_c128_0_value)
  );
  C128 col_6_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1633:79
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_c128_1_value)
  );
  C181 col_6_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1634:79
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_c181_0_value)
  );
  C181 col_6_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1635:79
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_c181_1_value)
  );
  C8192 col_6_n_c8192 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1636:76
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_c8192_value)
  );
  W7 col_6_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1637:67
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_w7_value)
  );
  W1_sub_W7 col_6_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1638:88
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_w1_sub_w7_value)
  );
  W1_add_W7 col_6_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1639:88
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_w1_add_w7_value)
  );
  W3 col_6_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1640:67
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_w3_value)
  );
  W3_sub_W5 col_6_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1641:88
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_w3_sub_w5_value)
  );
  W3_add_W5 col_6_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1642:88
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_w3_add_w5_value)
  );
  W6 col_6_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1643:67
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_w6_value)
  );
  W2_sub_W6 col_6_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1644:88
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_w2_sub_w6_value)
  );
  W2_add_W6 col_6_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1645:88
    .clock (clock),
    .reset (reset),
    .value (_col_6_n_w2_add_w6_value)
  );
  SHL_8 col_6_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1646:86
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_shr_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:624:90
    .z     (_col_6_n_x1_0_z)
  );
  SHL_8 col_6_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1647:86
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_shr_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:232:90
    .z     (_col_6_n_t0_0_z)
  );
  ADD col_6_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1648:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1647:86
    .y     (_col_6_n_c8192_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1636:76
    .z     (_col_6_n_x0_0_z)
  );
  dup_2 col_6_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1649:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1648:103
    .y     (_col_6_d_x0_0_y),
    .z     (_col_6_d_x0_0_z)
  );
  dup_2 col_6_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1650:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1646:86
    .y     (_col_6_d_x1_0_y),
    .z     (_col_6_d_x1_0_z)
  );
  dup_2 col_6_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1651:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_shr_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:820:90
    .y     (_col_6_d_x2_0_y),
    .z     (_col_6_d_x2_0_z)
  );
  dup_2 col_6_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1652:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_shr_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:428:90
    .y     (_col_6_d_x3_0_y),
    .z     (_col_6_d_x3_0_z)
  );
  dup_2 col_6_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1653:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_shr_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:330:90
    .y     (_col_6_d_x4_0_y),
    .z     (_col_6_d_x4_0_z)
  );
  dup_2 col_6_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1654:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_shr_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:918:90
    .y     (_col_6_d_x5_0_y),
    .z     (_col_6_d_x5_0_z)
  );
  dup_2 col_6_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1655:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_shr_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:722:90
    .y     (_col_6_d_x6_0_y),
    .z     (_col_6_d_x6_0_z)
  );
  dup_2 col_6_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1656:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_shr_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:526:90
    .y     (_col_6_d_x7_0_y),
    .z     (_col_6_d_x7_0_z)
  );
  ADD col_6_n_u8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1657:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1653:103
    .y     (_col_6_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1654:103
    .z     (_col_6_n_u8_0_z)
  );
  MUL col_6_n_v8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1658:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1637:67
    .y     (_col_6_n_u8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1657:103
    .z     (_col_6_n_v8_0_z)
  );
  ADD col_6_n_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1659:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1658:103
    .y     (_col_6_n_c4_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1629:73
    .z     (_col_6_n_x8_0_z)
  );
  dup_2 col_6_d_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1660:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1659:103
    .y     (_col_6_d_x8_0_y),
    .z     (_col_6_d_x8_0_z)
  );
  MUL col_6_n_u4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1661:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1638:88
    .y     (_col_6_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1653:103
    .z     (_col_6_n_u4_1_z)
  );
  ADD col_6_n_v4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1662:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x8_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1660:103
    .y     (_delay_INT16_2_1958_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1967:113
    .z     (_col_6_n_v4_1_z)
  );
  SHR_3 col_6_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1663:86
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1662:103
    .z     (_col_6_n_x4_1_z)
  );
  dup_2 col_6_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1664:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1663:86
    .y     (_col_6_d_x4_1_y),
    .z     (_col_6_d_x4_1_z)
  );
  MUL col_6_n_u5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1665:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1639:88
    .y     (_col_6_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1654:103
    .z     (_col_6_n_u5_1_z)
  );
  SUB col_6_n_v5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1666:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1660:103
    .y     (_delay_INT16_2_1959_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1968:113
    .z     (_col_6_n_v5_1_z)
  );
  SHR_3 col_6_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1667:86
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1666:103
    .z     (_col_6_n_x5_1_z)
  );
  dup_2 col_6_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1668:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1667:86
    .y     (_col_6_d_x5_1_y),
    .z     (_col_6_d_x5_1_z)
  );
  ADD col_6_n_u8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1669:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1655:103
    .y     (_col_6_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1656:103
    .z     (_col_6_n_u8_1_z)
  );
  MUL col_6_n_v8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1670:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1640:67
    .y     (_col_6_n_u8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1669:103
    .z     (_col_6_n_v8_1_z)
  );
  ADD col_6_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1671:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1670:103
    .y     (_col_6_n_c4_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1630:73
    .z     (_col_6_n_x8_1_z)
  );
  dup_2 col_6_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1672:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1671:103
    .y     (_col_6_d_x8_1_y),
    .z     (_col_6_d_x8_1_z)
  );
  MUL col_6_n_u6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1673:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1641:88
    .y     (_col_6_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1655:103
    .z     (_col_6_n_u6_1_z)
  );
  SUB col_6_n_v6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1674:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1672:103
    .y     (_delay_INT16_2_2082_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2091:113
    .z     (_col_6_n_v6_1_z)
  );
  SHR_3 col_6_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1675:86
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1674:103
    .z     (_col_6_n_x6_1_z)
  );
  dup_2 col_6_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1676:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1675:86
    .y     (_col_6_d_x6_1_y),
    .z     (_col_6_d_x6_1_z)
  );
  MUL col_6_n_u7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1677:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1642:88
    .y     (_col_6_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1656:103
    .z     (_col_6_n_u7_1_z)
  );
  SUB col_6_n_v7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1678:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1672:103
    .y     (_delay_INT16_2_2083_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2092:113
    .z     (_col_6_n_v7_1_z)
  );
  SHR_3 col_6_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1679:86
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1678:103
    .z     (_col_6_n_x7_1_z)
  );
  dup_2 col_6_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1680:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1679:86
    .y     (_col_6_d_x7_1_y),
    .z     (_col_6_d_x7_1_z)
  );
  ADD col_6_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1681:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1649:103
    .y     (_delay_INT16_1_2084_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2093:113
    .z     (_col_6_n_x8_2_z)
  );
  dup_2 col_6_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1682:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1681:103
    .y     (_col_6_d_x8_2_y),
    .z     (_col_6_d_x8_2_z)
  );
  SUB col_6_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1683:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1649:103
    .y     (_delay_INT16_1_2085_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2094:113
    .z     (_col_6_n_x0_1_z)
  );
  dup_2 col_6_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1684:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1683:103
    .y     (_col_6_d_x0_1_y),
    .z     (_col_6_d_x0_1_z)
  );
  ADD col_6_n_u1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1685:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1652:103
    .y     (_col_6_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1651:103
    .z     (_col_6_n_u1_1_z)
  );
  MUL col_6_n_v1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1686:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1643:67
    .y     (_col_6_n_u1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1685:103
    .z     (_col_6_n_v1_1_z)
  );
  ADD col_6_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1687:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1686:103
    .y     (_col_6_n_c4_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1631:73
    .z     (_col_6_n_x1_1_z)
  );
  dup_2 col_6_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1688:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1687:103
    .y     (_col_6_d_x1_1_y),
    .z     (_col_6_d_x1_1_z)
  );
  MUL col_6_n_u2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1689:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1645:88
    .y     (_col_6_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1651:103
    .z     (_col_6_n_u2_1_z)
  );
  SUB col_6_n_v2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1690:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1688:103
    .y     (_delay_INT16_2_2086_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2095:113
    .z     (_col_6_n_v2_1_z)
  );
  SHR_3 col_6_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1691:86
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1690:103
    .z     (_col_6_n_x2_1_z)
  );
  dup_2 col_6_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1692:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1691:86
    .y     (_col_6_d_x2_1_y),
    .z     (_col_6_d_x2_1_z)
  );
  MUL col_6_n_u3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1693:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1644:88
    .y     (_col_6_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1652:103
    .z     (_col_6_n_u3_1_z)
  );
  ADD col_6_n_v3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1694:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1688:103
    .y     (_delay_INT16_2_2087_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2096:113
    .z     (_col_6_n_v3_1_z)
  );
  SHR_3 col_6_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1695:86
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1694:103
    .z     (_col_6_n_x3_1_z)
  );
  dup_2 col_6_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1696:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1695:86
    .y     (_col_6_d_x3_1_y),
    .z     (_col_6_d_x3_1_z)
  );
  ADD col_6_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1697:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1664:103
    .y     (_col_6_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1676:103
    .z     (_col_6_n_x1_2_z)
  );
  dup_2 col_6_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1698:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1697:103
    .y     (_col_6_d_x1_2_y),
    .z     (_col_6_d_x1_2_z)
  );
  SUB col_6_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1699:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1664:103
    .y     (_col_6_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1676:103
    .z     (_col_6_n_x4_2_z)
  );
  dup_2 col_6_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1700:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1699:103
    .y     (_col_6_d_x4_2_y),
    .z     (_col_6_d_x4_2_z)
  );
  ADD col_6_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1701:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1668:103
    .y     (_col_6_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1680:103
    .z     (_col_6_n_x6_2_z)
  );
  dup_2 col_6_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1702:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1701:103
    .y     (_col_6_d_x6_2_y),
    .z     (_col_6_d_x6_2_z)
  );
  SUB col_6_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1703:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1668:103
    .y     (_col_6_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1680:103
    .z     (_col_6_n_x5_2_z)
  );
  dup_2 col_6_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1704:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1703:103
    .y     (_col_6_d_x5_2_y),
    .z     (_col_6_d_x5_2_z)
  );
  ADD col_6_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1705:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2088_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2097:113
    .y     (_col_6_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1696:103
    .z     (_col_6_n_x7_2_z)
  );
  dup_2 col_6_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1706:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1705:103
    .y     (_col_6_d_x7_2_y),
    .z     (_col_6_d_x7_2_z)
  );
  SUB col_6_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1707:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2089_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2098:113
    .y     (_col_6_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1696:103
    .z     (_col_6_n_x8_3_z)
  );
  dup_2 col_6_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1708:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1707:103
    .y     (_col_6_d_x8_3_y),
    .z     (_col_6_d_x8_3_z)
  );
  ADD col_6_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1709:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2090_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2099:113
    .y     (_col_6_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1692:103
    .z     (_col_6_n_x3_2_z)
  );
  dup_2 col_6_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1710:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1709:103
    .y     (_col_6_d_x3_2_y),
    .z     (_col_6_d_x3_2_z)
  );
  SUB col_6_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1711:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2091_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2100:113
    .y     (_col_6_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1692:103
    .z     (_col_6_n_x0_2_z)
  );
  dup_2 col_6_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1712:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1711:103
    .y     (_col_6_d_x0_2_y),
    .z     (_col_6_d_x0_2_z)
  );
  ADD col_6_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1713:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1700:103
    .y     (_col_6_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1704:103
    .z     (_col_6_n_u2_2_z)
  );
  MUL col_6_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1714:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1634:79
    .y     (_col_6_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1713:103
    .z     (_col_6_n_v2_2_z)
  );
  ADD col_6_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1715:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1714:103
    .y     (_col_6_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1632:79
    .z     (_col_6_n_w2_2_z)
  );
  SHR_8 col_6_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1716:86
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1715:103
    .z     (_col_6_n_x2_2_z)
  );
  dup_2 col_6_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1717:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1716:86
    .y     (_col_6_d_x2_2_y),
    .z     (_col_6_d_x2_2_z)
  );
  SUB col_6_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1718:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1700:103
    .y     (_col_6_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1704:103
    .z     (_col_6_n_u4_3_z)
  );
  MUL col_6_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1719:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1635:79
    .y     (_col_6_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1718:103
    .z     (_col_6_n_v4_3_z)
  );
  ADD col_6_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1720:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1719:103
    .y     (_col_6_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1633:79
    .z     (_col_6_n_w4_3_z)
  );
  SHR_8 col_6_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1721:86
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1720:103
    .z     (_col_6_n_x4_3_z)
  );
  dup_2 col_6_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1722:103
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1721:86
    .y     (_col_6_d_x4_3_y),
    .z     (_col_6_d_x4_3_z)
  );
  ADD col_6_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1723:108
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1706:103
    .y     (_col_6_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1698:103
    .z     (_col_6_n_tmp_0_z)
  );
  SHR_14 col_6_n_val_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1724:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1723:108
    .z     (_col_6_n_val_0_z)
  );
  CLIP col_6_n_clp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1725:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_val_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1724:90
    .z     (n_out_0_6_x)
  );
  ADD col_6_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1726:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2092_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2101:113
    .y     (_col_6_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1717:103
    .z     (_col_6_n_tmp_1_z)
  );
  SHR_14 col_6_n_val_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1727:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1726:108
    .z     (_col_6_n_val_1_z)
  );
  CLIP col_6_n_clp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1728:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_val_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1727:90
    .z     (n_out_1_6_x)
  );
  ADD col_6_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1729:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2093_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2102:113
    .y     (_col_6_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1722:103
    .z     (_col_6_n_tmp_2_z)
  );
  SHR_14 col_6_n_val_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1730:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1729:108
    .z     (_col_6_n_val_2_z)
  );
  CLIP col_6_n_clp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1731:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_val_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1730:90
    .z     (n_out_2_6_x)
  );
  ADD col_6_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1732:108
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1708:103
    .y     (_col_6_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1702:103
    .z     (_col_6_n_tmp_3_z)
  );
  SHR_14 col_6_n_val_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1733:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1732:108
    .z     (_col_6_n_val_3_z)
  );
  CLIP col_6_n_clp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1734:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_val_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1733:90
    .z     (n_out_3_6_x)
  );
  SUB col_6_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1735:108
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1708:103
    .y     (_col_6_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1702:103
    .z     (_col_6_n_tmp_4_z)
  );
  SHR_14 col_6_n_val_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1736:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1735:108
    .z     (_col_6_n_val_4_z)
  );
  CLIP col_6_n_clp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1737:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_val_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1736:90
    .z     (n_out_4_6_x)
  );
  SUB col_6_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1738:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2094_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2103:113
    .y     (_col_6_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1722:103
    .z     (_col_6_n_tmp_5_z)
  );
  SHR_14 col_6_n_val_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1739:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1738:108
    .z     (_col_6_n_val_5_z)
  );
  CLIP col_6_n_clp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1740:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_val_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1739:90
    .z     (n_out_5_6_x)
  );
  SUB col_6_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1741:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2095_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2104:113
    .y     (_col_6_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1717:103
    .z     (_col_6_n_tmp_6_z)
  );
  SHR_14 col_6_n_val_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1742:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1741:108
    .z     (_col_6_n_val_6_z)
  );
  CLIP col_6_n_clp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1743:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_val_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1742:90
    .z     (n_out_6_6_x)
  );
  SUB col_6_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1744:108
    .clock (clock),
    .reset (reset),
    .x     (_col_6_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1706:103
    .y     (_col_6_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1698:103
    .z     (_col_6_n_tmp_7_z)
  );
  SHR_14 col_6_n_val_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1745:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1744:108
    .z     (_col_6_n_val_7_z)
  );
  CLIP col_6_n_clp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1746:90
    .clock (clock),
    .reset (reset),
    .x     (_col_6_n_val_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1745:90
    .z     (n_out_7_6_x)
  );
  C4 col_7_n_c4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1747:73
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_c4_0_value)
  );
  C4 col_7_n_c4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1748:73
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_c4_1_value)
  );
  C4 col_7_n_c4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1749:73
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_c4_2_value)
  );
  C128 col_7_n_c128_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1750:79
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_c128_0_value)
  );
  C128 col_7_n_c128_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1751:79
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_c128_1_value)
  );
  C181 col_7_n_c181_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1752:79
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_c181_0_value)
  );
  C181 col_7_n_c181_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1753:79
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_c181_1_value)
  );
  C8192 col_7_n_c8192 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1754:76
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_c8192_value)
  );
  W7 col_7_n_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1755:67
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_w7_value)
  );
  W1_sub_W7 col_7_n_w1_sub_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1756:88
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_w1_sub_w7_value)
  );
  W1_add_W7 col_7_n_w1_add_w7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1757:88
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_w1_add_w7_value)
  );
  W3 col_7_n_w3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1758:67
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_w3_value)
  );
  W3_sub_W5 col_7_n_w3_sub_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1759:88
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_w3_sub_w5_value)
  );
  W3_add_W5 col_7_n_w3_add_w5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1760:88
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_w3_add_w5_value)
  );
  W6 col_7_n_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1761:67
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_w6_value)
  );
  W2_sub_W6 col_7_n_w2_sub_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1762:88
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_w2_sub_w6_value)
  );
  W2_add_W6 col_7_n_w2_add_w6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1763:88
    .clock (clock),
    .reset (reset),
    .value (_col_7_n_w2_add_w6_value)
  );
  SHL_8 col_7_n_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1764:86
    .clock (clock),
    .reset (reset),
    .x     (_row_4_n_shr_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:626:90
    .z     (_col_7_n_x1_0_z)
  );
  SHL_8 col_7_n_t0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1765:86
    .clock (clock),
    .reset (reset),
    .x     (_row_0_n_shr_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:234:90
    .z     (_col_7_n_t0_0_z)
  );
  ADD col_7_n_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1766:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_t0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1765:86
    .y     (_col_7_n_c8192_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1754:76
    .z     (_col_7_n_x0_0_z)
  );
  dup_2 col_7_d_x0_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1767:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1766:103
    .y     (_col_7_d_x0_0_y),
    .z     (_col_7_d_x0_0_z)
  );
  dup_2 col_7_d_x1_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1768:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1764:86
    .y     (_col_7_d_x1_0_y),
    .z     (_col_7_d_x1_0_z)
  );
  dup_2 col_7_d_x2_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1769:103
    .clock (clock),
    .reset (reset),
    .x     (_row_6_n_shr_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:822:90
    .y     (_col_7_d_x2_0_y),
    .z     (_col_7_d_x2_0_z)
  );
  dup_2 col_7_d_x3_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1770:103
    .clock (clock),
    .reset (reset),
    .x     (_row_2_n_shr_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:430:90
    .y     (_col_7_d_x3_0_y),
    .z     (_col_7_d_x3_0_z)
  );
  dup_2 col_7_d_x4_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1771:103
    .clock (clock),
    .reset (reset),
    .x     (_row_1_n_shr_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:332:90
    .y     (_col_7_d_x4_0_y),
    .z     (_col_7_d_x4_0_z)
  );
  dup_2 col_7_d_x5_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1772:103
    .clock (clock),
    .reset (reset),
    .x     (_row_7_n_shr_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:920:90
    .y     (_col_7_d_x5_0_y),
    .z     (_col_7_d_x5_0_z)
  );
  dup_2 col_7_d_x6_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1773:103
    .clock (clock),
    .reset (reset),
    .x     (_row_5_n_shr_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:724:90
    .y     (_col_7_d_x6_0_y),
    .z     (_col_7_d_x6_0_z)
  );
  dup_2 col_7_d_x7_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1774:103
    .clock (clock),
    .reset (reset),
    .x     (_row_3_n_shr_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:528:90
    .y     (_col_7_d_x7_0_y),
    .z     (_col_7_d_x7_0_z)
  );
  ADD col_7_n_u8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1775:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x4_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1771:103
    .y     (_col_7_d_x5_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1772:103
    .z     (_col_7_n_u8_0_z)
  );
  MUL col_7_n_v8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1776:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1755:67
    .y     (_col_7_n_u8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1775:103
    .z     (_col_7_n_v8_0_z)
  );
  ADD col_7_n_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1777:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1776:103
    .y     (_col_7_n_c4_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1747:73
    .z     (_col_7_n_x8_0_z)
  );
  dup_2 col_7_d_x8_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1778:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1777:103
    .y     (_col_7_d_x8_0_y),
    .z     (_col_7_d_x8_0_z)
  );
  MUL col_7_n_u4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1779:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w1_sub_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1756:88
    .y     (_col_7_d_x4_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1771:103
    .z     (_col_7_n_u4_1_z)
  );
  ADD col_7_n_v4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1780:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x8_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1778:103
    .y     (_delay_INT16_2_2096_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2105:113
    .z     (_col_7_n_v4_1_z)
  );
  SHR_3 col_7_n_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1781:86
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1780:103
    .z     (_col_7_n_x4_1_z)
  );
  dup_2 col_7_d_x4_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1782:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1781:86
    .y     (_col_7_d_x4_1_y),
    .z     (_col_7_d_x4_1_z)
  );
  MUL col_7_n_u5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1783:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w1_add_w7_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1757:88
    .y     (_col_7_d_x5_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1772:103
    .z     (_col_7_n_u5_1_z)
  );
  SUB col_7_n_v5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1784:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x8_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1778:103
    .y     (_delay_INT16_2_2097_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2106:113
    .z     (_col_7_n_v5_1_z)
  );
  SHR_3 col_7_n_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1785:86
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1784:103
    .z     (_col_7_n_x5_1_z)
  );
  dup_2 col_7_d_x5_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1786:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1785:86
    .y     (_col_7_d_x5_1_y),
    .z     (_col_7_d_x5_1_z)
  );
  ADD col_7_n_u8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1787:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x6_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1773:103
    .y     (_col_7_d_x7_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1774:103
    .z     (_col_7_n_u8_1_z)
  );
  MUL col_7_n_v8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1788:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w3_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1758:67
    .y     (_col_7_n_u8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1787:103
    .z     (_col_7_n_v8_1_z)
  );
  ADD col_7_n_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1789:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1788:103
    .y     (_col_7_n_c4_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1748:73
    .z     (_col_7_n_x8_1_z)
  );
  dup_2 col_7_d_x8_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1790:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1789:103
    .y     (_col_7_d_x8_1_y),
    .z     (_col_7_d_x8_1_z)
  );
  MUL col_7_n_u6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1791:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w3_sub_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1759:88
    .y     (_col_7_d_x6_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1773:103
    .z     (_col_7_n_u6_1_z)
  );
  SUB col_7_n_v6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1792:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x8_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1790:103
    .y     (_delay_INT16_2_2098_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2107:113
    .z     (_col_7_n_v6_1_z)
  );
  SHR_3 col_7_n_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1793:86
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1792:103
    .z     (_col_7_n_x6_1_z)
  );
  dup_2 col_7_d_x6_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1794:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1793:86
    .y     (_col_7_d_x6_1_y),
    .z     (_col_7_d_x6_1_z)
  );
  MUL col_7_n_u7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1795:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w3_add_w5_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1760:88
    .y     (_col_7_d_x7_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1774:103
    .z     (_col_7_n_u7_1_z)
  );
  SUB col_7_n_v7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1796:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x8_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1790:103
    .y     (_delay_INT16_2_2099_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2108:113
    .z     (_col_7_n_v7_1_z)
  );
  SHR_3 col_7_n_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1797:86
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1796:103
    .z     (_col_7_n_x7_1_z)
  );
  dup_2 col_7_d_x7_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1798:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1797:86
    .y     (_col_7_d_x7_1_y),
    .z     (_col_7_d_x7_1_z)
  );
  ADD col_7_n_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1799:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x0_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1767:103
    .y     (_delay_INT16_1_2100_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2109:113
    .z     (_col_7_n_x8_2_z)
  );
  dup_2 col_7_d_x8_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1800:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1799:103
    .y     (_col_7_d_x8_2_y),
    .z     (_col_7_d_x8_2_z)
  );
  SUB col_7_n_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1801:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x0_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1767:103
    .y     (_delay_INT16_1_2101_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2110:113
    .z     (_col_7_n_x0_1_z)
  );
  dup_2 col_7_d_x0_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1802:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1801:103
    .y     (_col_7_d_x0_1_y),
    .z     (_col_7_d_x0_1_z)
  );
  ADD col_7_n_u1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1803:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x3_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1770:103
    .y     (_col_7_d_x2_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1769:103
    .z     (_col_7_n_u1_1_z)
  );
  MUL col_7_n_v1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1804:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1761:67
    .y     (_col_7_n_u1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1803:103
    .z     (_col_7_n_v1_1_z)
  );
  ADD col_7_n_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1805:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1804:103
    .y     (_col_7_n_c4_2_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1749:73
    .z     (_col_7_n_x1_1_z)
  );
  dup_2 col_7_d_x1_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1806:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1805:103
    .y     (_col_7_d_x1_1_y),
    .z     (_col_7_d_x1_1_z)
  );
  MUL col_7_n_u2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1807:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w2_add_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1763:88
    .y     (_col_7_d_x2_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1769:103
    .z     (_col_7_n_u2_1_z)
  );
  SUB col_7_n_v2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1808:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x1_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1806:103
    .y     (_delay_INT16_2_2102_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2111:113
    .z     (_col_7_n_v2_1_z)
  );
  SHR_3 col_7_n_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1809:86
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1808:103
    .z     (_col_7_n_x2_1_z)
  );
  dup_2 col_7_d_x2_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1810:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1809:86
    .y     (_col_7_d_x2_1_y),
    .z     (_col_7_d_x2_1_z)
  );
  MUL col_7_n_u3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1811:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w2_sub_w6_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1762:88
    .y     (_col_7_d_x3_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1770:103
    .z     (_col_7_n_u3_1_z)
  );
  ADD col_7_n_v3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1812:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x1_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1806:103
    .y     (_delay_INT16_2_2103_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2112:113
    .z     (_col_7_n_v3_1_z)
  );
  SHR_3 col_7_n_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1813:86
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1812:103
    .z     (_col_7_n_x3_1_z)
  );
  dup_2 col_7_d_x3_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1814:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1813:86
    .y     (_col_7_d_x3_1_y),
    .z     (_col_7_d_x3_1_z)
  );
  ADD col_7_n_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1815:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x4_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1782:103
    .y     (_col_7_d_x6_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1794:103
    .z     (_col_7_n_x1_2_z)
  );
  dup_2 col_7_d_x1_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1816:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1815:103
    .y     (_col_7_d_x1_2_y),
    .z     (_col_7_d_x1_2_z)
  );
  SUB col_7_n_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1817:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1782:103
    .y     (_col_7_d_x6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1794:103
    .z     (_col_7_n_x4_2_z)
  );
  dup_2 col_7_d_x4_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1818:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1817:103
    .y     (_col_7_d_x4_2_y),
    .z     (_col_7_d_x4_2_z)
  );
  ADD col_7_n_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1819:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x5_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1786:103
    .y     (_col_7_d_x7_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1798:103
    .z     (_col_7_n_x6_2_z)
  );
  dup_2 col_7_d_x6_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1820:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1819:103
    .y     (_col_7_d_x6_2_y),
    .z     (_col_7_d_x6_2_z)
  );
  SUB col_7_n_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1821:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1786:103
    .y     (_col_7_d_x7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1798:103
    .z     (_col_7_n_x5_2_z)
  );
  dup_2 col_7_d_x5_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1822:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1821:103
    .y     (_col_7_d_x5_2_y),
    .z     (_col_7_d_x5_2_z)
  );
  ADD col_7_n_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1823:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2104_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2113:113
    .y     (_col_7_d_x3_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1814:103
    .z     (_col_7_n_x7_2_z)
  );
  dup_2 col_7_d_x7_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1824:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1823:103
    .y     (_col_7_d_x7_2_y),
    .z     (_col_7_d_x7_2_z)
  );
  SUB col_7_n_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1825:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2105_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2114:113
    .y     (_col_7_d_x3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1814:103
    .z     (_col_7_n_x8_3_z)
  );
  dup_2 col_7_d_x8_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1826:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1825:103
    .y     (_col_7_d_x8_3_y),
    .z     (_col_7_d_x8_3_z)
  );
  ADD col_7_n_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1827:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2106_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2115:113
    .y     (_col_7_d_x2_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1810:103
    .z     (_col_7_n_x3_2_z)
  );
  dup_2 col_7_d_x3_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1828:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1827:103
    .y     (_col_7_d_x3_2_y),
    .z     (_col_7_d_x3_2_z)
  );
  SUB col_7_n_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1829:103
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_4_2107_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2116:113
    .y     (_col_7_d_x2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1810:103
    .z     (_col_7_n_x0_2_z)
  );
  dup_2 col_7_d_x0_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1830:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1829:103
    .y     (_col_7_d_x0_2_y),
    .z     (_col_7_d_x0_2_z)
  );
  ADD col_7_n_u2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1831:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x4_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1818:103
    .y     (_col_7_d_x5_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1822:103
    .z     (_col_7_n_u2_2_z)
  );
  MUL col_7_n_v2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1832:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_c181_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1752:79
    .y     (_col_7_n_u2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1831:103
    .z     (_col_7_n_v2_2_z)
  );
  ADD col_7_n_w2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1833:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1832:103
    .y     (_col_7_n_c128_0_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1750:79
    .z     (_col_7_n_w2_2_z)
  );
  SHR_8 col_7_n_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1834:86
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1833:103
    .z     (_col_7_n_x2_2_z)
  );
  dup_2 col_7_d_x2_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1835:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1834:86
    .y     (_col_7_d_x2_2_y),
    .z     (_col_7_d_x2_2_z)
  );
  SUB col_7_n_u4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1836:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x4_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1818:103
    .y     (_col_7_d_x5_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1822:103
    .z     (_col_7_n_u4_3_z)
  );
  MUL col_7_n_v4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1837:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_c181_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1753:79
    .y     (_col_7_n_u4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1836:103
    .z     (_col_7_n_v4_3_z)
  );
  ADD col_7_n_w4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1838:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_v4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1837:103
    .y     (_col_7_n_c128_1_value),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1751:79
    .z     (_col_7_n_w4_3_z)
  );
  SHR_8 col_7_n_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1839:86
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_w4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1838:103
    .z     (_col_7_n_x4_3_z)
  );
  dup_2 col_7_d_x4_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1840:103
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1839:86
    .y     (_col_7_d_x4_3_y),
    .z     (_col_7_d_x4_3_z)
  );
  ADD col_7_n_tmp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1841:108
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x7_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1824:103
    .y     (_col_7_d_x1_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1816:103
    .z     (_col_7_n_tmp_0_z)
  );
  SHR_14 col_7_n_val_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1842:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_tmp_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1841:108
    .z     (_col_7_n_val_0_z)
  );
  CLIP col_7_n_clp_0 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1843:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_val_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1842:90
    .z     (n_out_0_7_x)
  );
  ADD col_7_n_tmp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1844:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2108_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2117:113
    .y     (_col_7_d_x2_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1835:103
    .z     (_col_7_n_tmp_1_z)
  );
  SHR_14 col_7_n_val_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1845:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_tmp_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1844:108
    .z     (_col_7_n_val_1_z)
  );
  CLIP col_7_n_clp_1 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1846:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_val_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1845:90
    .z     (n_out_1_7_x)
  );
  ADD col_7_n_tmp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1847:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2109_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2118:113
    .y     (_col_7_d_x4_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1840:103
    .z     (_col_7_n_tmp_2_z)
  );
  SHR_14 col_7_n_val_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1848:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_tmp_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1847:108
    .z     (_col_7_n_val_2_z)
  );
  CLIP col_7_n_clp_2 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1849:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_val_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1848:90
    .z     (n_out_2_7_x)
  );
  ADD col_7_n_tmp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1850:108
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1826:103
    .y     (_col_7_d_x6_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1820:103
    .z     (_col_7_n_tmp_3_z)
  );
  SHR_14 col_7_n_val_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1851:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_tmp_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1850:108
    .z     (_col_7_n_val_3_z)
  );
  CLIP col_7_n_clp_3 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1852:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_val_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1851:90
    .z     (n_out_3_7_x)
  );
  SUB col_7_n_tmp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1853:108
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1826:103
    .y     (_col_7_d_x6_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1820:103
    .z     (_col_7_n_tmp_4_z)
  );
  SHR_14 col_7_n_val_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1854:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_tmp_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1853:108
    .z     (_col_7_n_val_4_z)
  );
  CLIP col_7_n_clp_4 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1855:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_val_4_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1854:90
    .z     (n_out_4_7_x)
  );
  SUB col_7_n_tmp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1856:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2110_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2119:113
    .y     (_col_7_d_x4_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1840:103
    .z     (_col_7_n_tmp_5_z)
  );
  SHR_14 col_7_n_val_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1857:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_tmp_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1856:108
    .z     (_col_7_n_val_5_z)
  );
  CLIP col_7_n_clp_5 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1858:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_val_5_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1857:90
    .z     (n_out_5_7_x)
  );
  SUB col_7_n_tmp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1859:108
    .clock (clock),
    .reset (reset),
    .x     (_delay_INT16_6_2111_out),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2120:113
    .y     (_col_7_d_x2_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1835:103
    .z     (_col_7_n_tmp_6_z)
  );
  SHR_14 col_7_n_val_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1860:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_tmp_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1859:108
    .z     (_col_7_n_val_6_z)
  );
  CLIP col_7_n_clp_6 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1861:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_val_6_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1860:90
    .z     (n_out_6_7_x)
  );
  SUB col_7_n_tmp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1862:108
    .clock (clock),
    .reset (reset),
    .x     (_col_7_d_x7_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1824:103
    .y     (_col_7_d_x1_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1816:103
    .z     (_col_7_n_tmp_7_z)
  );
  SHR_14 col_7_n_val_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1863:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_tmp_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1862:108
    .z     (_col_7_n_val_7_z)
  );
  CLIP col_7_n_clp_7 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1864:90
    .clock (clock),
    .reset (reset),
    .x     (_col_7_n_val_7_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1863:90
    .z     (n_out_7_7_x)
  );
  delay_INT16_1 delay_INT16_1_1856 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1865:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_n_t5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:266:103
    .out   (_delay_INT16_1_1856_out)
  );
  delay_INT16_1 delay_INT16_1_1857 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1866:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_n_t6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:272:103
    .out   (_delay_INT16_1_1857_out)
  );
  delay_INT16_1 delay_INT16_1_1858 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1867:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_n_t7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:275:103
    .out   (_delay_INT16_1_1858_out)
  );
  delay_INT16_1 delay_INT16_1_1859 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1868:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_n_t4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:165:103
    .out   (_delay_INT16_1_1859_out)
  );
  delay_INT16_1 delay_INT16_1_1860 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1869:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_n_t5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:168:103
    .out   (_delay_INT16_1_1860_out)
  );
  delay_INT16_1 delay_INT16_1_1861 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1870:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_n_t6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:174:103
    .out   (_delay_INT16_1_1861_out)
  );
  delay_INT16_1 delay_INT16_1_1862 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1871:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_n_t7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:177:103
    .out   (_delay_INT16_1_1862_out)
  );
  delay_INT16_1 delay_INT16_1_1863 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1872:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:155:103
    .out   (_delay_INT16_1_1863_out)
  );
  delay_INT16_1 delay_INT16_1_1864 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1873:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:155:103
    .out   (_delay_INT16_1_1864_out)
  );
  delay_INT16_1 delay_INT16_1_1865 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1874:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_n_t2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:187:103
    .out   (_delay_INT16_1_1865_out)
  );
  delay_INT16_1 delay_INT16_1_1866 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1875:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_n_t3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:190:103
    .out   (_delay_INT16_1_1866_out)
  );
  delay_INT16_2 delay_INT16_2_1867 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1876:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:181:103
    .out   (_delay_INT16_2_1867_out)
  );
  delay_INT16_2 delay_INT16_2_1868 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1877:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:181:103
    .out   (_delay_INT16_2_1868_out)
  );
  delay_INT16_2 delay_INT16_2_1869 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1878:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:183:103
    .out   (_delay_INT16_2_1869_out)
  );
  delay_INT16_2 delay_INT16_2_1870 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1879:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:183:103
    .out   (_delay_INT16_2_1870_out)
  );
  delay_INT16_6 delay_INT16_6_1871 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1880:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:206:103
    .out   (_delay_INT16_6_1871_out)
  );
  delay_INT16_6 delay_INT16_6_1872 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1881:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:208:103
    .out   (_delay_INT16_6_1872_out)
  );
  delay_INT16_6 delay_INT16_6_1873 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1882:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:208:103
    .out   (_delay_INT16_6_1873_out)
  );
  delay_INT16_6 delay_INT16_6_1874 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1883:113
    .clock (clock),
    .reset (reset),
    .in    (_row_0_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:206:103
    .out   (_delay_INT16_6_1874_out)
  );
  delay_INT16_1 delay_INT16_1_1875 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1884:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_n_t4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:263:103
    .out   (_delay_INT16_1_1875_out)
  );
  delay_INT16_6 delay_INT16_6_1876 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1885:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:502:103
    .out   (_delay_INT16_6_1876_out)
  );
  delay_INT16_6 delay_INT16_6_1877 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1886:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:500:103
    .out   (_delay_INT16_6_1877_out)
  );
  delay_INT16_1 delay_INT16_1_1878 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1887:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:253:103
    .out   (_delay_INT16_1_1878_out)
  );
  delay_INT16_1 delay_INT16_1_1879 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1888:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:253:103
    .out   (_delay_INT16_1_1879_out)
  );
  delay_INT16_1 delay_INT16_1_1880 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1889:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_n_t2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:285:103
    .out   (_delay_INT16_1_1880_out)
  );
  delay_INT16_1 delay_INT16_1_1881 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1890:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_n_t3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:288:103
    .out   (_delay_INT16_1_1881_out)
  );
  delay_INT16_2 delay_INT16_2_1882 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1891:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:279:103
    .out   (_delay_INT16_2_1882_out)
  );
  delay_INT16_2 delay_INT16_2_1883 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1892:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:279:103
    .out   (_delay_INT16_2_1883_out)
  );
  delay_INT16_2 delay_INT16_2_1884 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1893:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:281:103
    .out   (_delay_INT16_2_1884_out)
  );
  delay_INT16_2 delay_INT16_2_1885 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1894:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:281:103
    .out   (_delay_INT16_2_1885_out)
  );
  delay_INT16_6 delay_INT16_6_1886 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1895:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:304:103
    .out   (_delay_INT16_6_1886_out)
  );
  delay_INT16_6 delay_INT16_6_1887 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1896:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:306:103
    .out   (_delay_INT16_6_1887_out)
  );
  delay_INT16_6 delay_INT16_6_1888 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1897:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:306:103
    .out   (_delay_INT16_6_1888_out)
  );
  delay_INT16_6 delay_INT16_6_1889 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1898:113
    .clock (clock),
    .reset (reset),
    .in    (_row_1_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:304:103
    .out   (_delay_INT16_6_1889_out)
  );
  delay_INT16_1 delay_INT16_1_1890 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1899:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_n_t4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:753:103
    .out   (_delay_INT16_1_1890_out)
  );
  delay_INT16_1 delay_INT16_1_1891 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1900:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_n_t5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:756:103
    .out   (_delay_INT16_1_1891_out)
  );
  delay_INT16_1 delay_INT16_1_1892 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1901:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_n_t6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:762:103
    .out   (_delay_INT16_1_1892_out)
  );
  delay_INT16_1 delay_INT16_1_1893 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1902:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_n_t7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:765:103
    .out   (_delay_INT16_1_1893_out)
  );
  delay_INT16_1 delay_INT16_1_1894 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1903:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:743:103
    .out   (_delay_INT16_1_1894_out)
  );
  delay_INT16_1 delay_INT16_1_1895 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1904:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:743:103
    .out   (_delay_INT16_1_1895_out)
  );
  delay_INT16_1 delay_INT16_1_1896 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1905:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_n_t4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:361:103
    .out   (_delay_INT16_1_1896_out)
  );
  delay_INT16_1 delay_INT16_1_1897 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1906:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_n_t5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:364:103
    .out   (_delay_INT16_1_1897_out)
  );
  delay_INT16_1 delay_INT16_1_1898 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1907:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_n_t6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:370:103
    .out   (_delay_INT16_1_1898_out)
  );
  delay_INT16_1 delay_INT16_1_1899 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1908:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_n_t7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:373:103
    .out   (_delay_INT16_1_1899_out)
  );
  delay_INT16_1 delay_INT16_1_1900 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1909:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:351:103
    .out   (_delay_INT16_1_1900_out)
  );
  delay_INT16_1 delay_INT16_1_1901 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1910:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:351:103
    .out   (_delay_INT16_1_1901_out)
  );
  delay_INT16_1 delay_INT16_1_1902 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1911:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_n_t2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:383:103
    .out   (_delay_INT16_1_1902_out)
  );
  delay_INT16_1 delay_INT16_1_1903 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1912:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_n_t3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:386:103
    .out   (_delay_INT16_1_1903_out)
  );
  delay_INT16_2 delay_INT16_2_1904 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1913:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:377:103
    .out   (_delay_INT16_2_1904_out)
  );
  delay_INT16_2 delay_INT16_2_1905 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1914:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:377:103
    .out   (_delay_INT16_2_1905_out)
  );
  delay_INT16_2 delay_INT16_2_1906 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1915:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:379:103
    .out   (_delay_INT16_2_1906_out)
  );
  delay_INT16_2 delay_INT16_2_1907 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1916:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:379:103
    .out   (_delay_INT16_2_1907_out)
  );
  delay_INT16_6 delay_INT16_6_1908 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1917:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:402:103
    .out   (_delay_INT16_6_1908_out)
  );
  delay_INT16_6 delay_INT16_6_1909 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1918:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:404:103
    .out   (_delay_INT16_6_1909_out)
  );
  delay_INT16_6 delay_INT16_6_1910 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1919:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:404:103
    .out   (_delay_INT16_6_1910_out)
  );
  delay_INT16_6 delay_INT16_6_1911 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1920:113
    .clock (clock),
    .reset (reset),
    .in    (_row_2_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:402:103
    .out   (_delay_INT16_6_1911_out)
  );
  delay_INT16_1 delay_INT16_1_1912 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1921:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_n_t4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:459:103
    .out   (_delay_INT16_1_1912_out)
  );
  delay_INT16_1 delay_INT16_1_1913 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1922:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_n_t5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:462:103
    .out   (_delay_INT16_1_1913_out)
  );
  delay_INT16_1 delay_INT16_1_1914 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1923:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_n_t6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:468:103
    .out   (_delay_INT16_1_1914_out)
  );
  delay_INT16_1 delay_INT16_1_1915 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1924:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_n_t7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:471:103
    .out   (_delay_INT16_1_1915_out)
  );
  delay_INT16_1 delay_INT16_1_1916 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1925:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:449:103
    .out   (_delay_INT16_1_1916_out)
  );
  delay_INT16_1 delay_INT16_1_1917 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1926:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:449:103
    .out   (_delay_INT16_1_1917_out)
  );
  delay_INT16_1 delay_INT16_1_1918 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1927:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_n_t2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:481:103
    .out   (_delay_INT16_1_1918_out)
  );
  delay_INT16_1 delay_INT16_1_1919 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1928:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_n_t3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:484:103
    .out   (_delay_INT16_1_1919_out)
  );
  delay_INT16_2 delay_INT16_2_1920 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1929:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:475:103
    .out   (_delay_INT16_2_1920_out)
  );
  delay_INT16_2 delay_INT16_2_1921 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1930:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:475:103
    .out   (_delay_INT16_2_1921_out)
  );
  delay_INT16_2 delay_INT16_2_1922 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1931:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:477:103
    .out   (_delay_INT16_2_1922_out)
  );
  delay_INT16_2 delay_INT16_2_1923 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1932:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:477:103
    .out   (_delay_INT16_2_1923_out)
  );
  delay_INT16_6 delay_INT16_6_1924 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1933:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:500:103
    .out   (_delay_INT16_6_1924_out)
  );
  delay_INT16_6 delay_INT16_6_1925 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1934:113
    .clock (clock),
    .reset (reset),
    .in    (_row_3_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:502:103
    .out   (_delay_INT16_6_1925_out)
  );
  delay_INT16_1 delay_INT16_1_1926 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1935:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_n_t4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:557:103
    .out   (_delay_INT16_1_1926_out)
  );
  delay_INT16_1 delay_INT16_1_1927 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1936:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_n_t5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:560:103
    .out   (_delay_INT16_1_1927_out)
  );
  delay_INT16_1 delay_INT16_1_1928 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1937:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_n_t6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:566:103
    .out   (_delay_INT16_1_1928_out)
  );
  delay_INT16_1 delay_INT16_1_1929 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1938:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_n_t7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:569:103
    .out   (_delay_INT16_1_1929_out)
  );
  delay_INT16_1 delay_INT16_1_1930 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1939:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:547:103
    .out   (_delay_INT16_1_1930_out)
  );
  delay_INT16_1 delay_INT16_1_1931 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1940:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:547:103
    .out   (_delay_INT16_1_1931_out)
  );
  delay_INT16_1 delay_INT16_1_1932 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1941:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_n_t2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:579:103
    .out   (_delay_INT16_1_1932_out)
  );
  delay_INT16_1 delay_INT16_1_1933 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1942:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_n_t3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:582:103
    .out   (_delay_INT16_1_1933_out)
  );
  delay_INT16_2 delay_INT16_2_1934 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1943:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:573:103
    .out   (_delay_INT16_2_1934_out)
  );
  delay_INT16_2 delay_INT16_2_1935 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1944:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:573:103
    .out   (_delay_INT16_2_1935_out)
  );
  delay_INT16_2 delay_INT16_2_1936 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1945:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:575:103
    .out   (_delay_INT16_2_1936_out)
  );
  delay_INT16_2 delay_INT16_2_1937 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1946:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:575:103
    .out   (_delay_INT16_2_1937_out)
  );
  delay_INT16_6 delay_INT16_6_1938 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1947:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:598:103
    .out   (_delay_INT16_6_1938_out)
  );
  delay_INT16_6 delay_INT16_6_1939 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1948:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:600:103
    .out   (_delay_INT16_6_1939_out)
  );
  delay_INT16_6 delay_INT16_6_1940 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1949:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:600:103
    .out   (_delay_INT16_6_1940_out)
  );
  delay_INT16_6 delay_INT16_6_1941 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1950:113
    .clock (clock),
    .reset (reset),
    .in    (_row_4_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:598:103
    .out   (_delay_INT16_6_1941_out)
  );
  delay_INT16_1 delay_INT16_1_1942 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1951:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_n_t4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:655:103
    .out   (_delay_INT16_1_1942_out)
  );
  delay_INT16_1 delay_INT16_1_1943 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1952:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_n_t5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:658:103
    .out   (_delay_INT16_1_1943_out)
  );
  delay_INT16_1 delay_INT16_1_1944 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1953:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_n_t6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:664:103
    .out   (_delay_INT16_1_1944_out)
  );
  delay_INT16_1 delay_INT16_1_1945 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1954:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_n_t7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:667:103
    .out   (_delay_INT16_1_1945_out)
  );
  delay_INT16_1 delay_INT16_1_1946 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1955:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:645:103
    .out   (_delay_INT16_1_1946_out)
  );
  delay_INT16_1 delay_INT16_1_1947 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1956:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:645:103
    .out   (_delay_INT16_1_1947_out)
  );
  delay_INT16_1 delay_INT16_1_1948 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1957:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_n_t2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:677:103
    .out   (_delay_INT16_1_1948_out)
  );
  delay_INT16_1 delay_INT16_1_1949 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1958:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_n_t3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:680:103
    .out   (_delay_INT16_1_1949_out)
  );
  delay_INT16_2 delay_INT16_2_1950 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1959:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:671:103
    .out   (_delay_INT16_2_1950_out)
  );
  delay_INT16_2 delay_INT16_2_1951 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1960:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:671:103
    .out   (_delay_INT16_2_1951_out)
  );
  delay_INT16_2 delay_INT16_2_1952 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1961:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:673:103
    .out   (_delay_INT16_2_1952_out)
  );
  delay_INT16_2 delay_INT16_2_1953 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1962:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:673:103
    .out   (_delay_INT16_2_1953_out)
  );
  delay_INT16_6 delay_INT16_6_1954 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1963:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:696:103
    .out   (_delay_INT16_6_1954_out)
  );
  delay_INT16_6 delay_INT16_6_1955 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1964:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:698:103
    .out   (_delay_INT16_6_1955_out)
  );
  delay_INT16_6 delay_INT16_6_1956 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1965:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:698:103
    .out   (_delay_INT16_6_1956_out)
  );
  delay_INT16_6 delay_INT16_6_1957 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1966:113
    .clock (clock),
    .reset (reset),
    .in    (_row_5_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:696:103
    .out   (_delay_INT16_6_1957_out)
  );
  delay_INT16_2 delay_INT16_2_1958 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1967:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_n_u4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1661:103
    .out   (_delay_INT16_2_1958_out)
  );
  delay_INT16_2 delay_INT16_2_1959 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1968:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_n_u5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1665:103
    .out   (_delay_INT16_2_1959_out)
  );
  delay_INT16_1 delay_INT16_1_1960 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1969:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_n_t2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:775:103
    .out   (_delay_INT16_1_1960_out)
  );
  delay_INT16_1 delay_INT16_1_1961 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1970:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_n_t3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:778:103
    .out   (_delay_INT16_1_1961_out)
  );
  delay_INT16_2 delay_INT16_2_1962 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1971:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:769:103
    .out   (_delay_INT16_2_1962_out)
  );
  delay_INT16_2 delay_INT16_2_1963 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1972:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:769:103
    .out   (_delay_INT16_2_1963_out)
  );
  delay_INT16_2 delay_INT16_2_1964 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1973:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:771:103
    .out   (_delay_INT16_2_1964_out)
  );
  delay_INT16_2 delay_INT16_2_1965 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1974:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:771:103
    .out   (_delay_INT16_2_1965_out)
  );
  delay_INT16_6 delay_INT16_6_1966 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1975:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:794:103
    .out   (_delay_INT16_6_1966_out)
  );
  delay_INT16_6 delay_INT16_6_1967 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1976:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:796:103
    .out   (_delay_INT16_6_1967_out)
  );
  delay_INT16_6 delay_INT16_6_1968 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1977:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:796:103
    .out   (_delay_INT16_6_1968_out)
  );
  delay_INT16_6 delay_INT16_6_1969 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1978:113
    .clock (clock),
    .reset (reset),
    .in    (_row_6_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:794:103
    .out   (_delay_INT16_6_1969_out)
  );
  delay_INT16_1 delay_INT16_1_1970 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1979:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_n_t4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:851:103
    .out   (_delay_INT16_1_1970_out)
  );
  delay_INT16_1 delay_INT16_1_1971 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1980:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_n_t5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:854:103
    .out   (_delay_INT16_1_1971_out)
  );
  delay_INT16_1 delay_INT16_1_1972 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1981:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_n_t6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:860:103
    .out   (_delay_INT16_1_1972_out)
  );
  delay_INT16_1 delay_INT16_1_1973 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1982:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_n_t7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:863:103
    .out   (_delay_INT16_1_1973_out)
  );
  delay_INT16_1 delay_INT16_1_1974 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1983:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:841:103
    .out   (_delay_INT16_1_1974_out)
  );
  delay_INT16_1 delay_INT16_1_1975 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1984:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:841:103
    .out   (_delay_INT16_1_1975_out)
  );
  delay_INT16_1 delay_INT16_1_1976 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1985:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_n_t2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:873:103
    .out   (_delay_INT16_1_1976_out)
  );
  delay_INT16_1 delay_INT16_1_1977 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1986:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_n_t3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:876:103
    .out   (_delay_INT16_1_1977_out)
  );
  delay_INT16_2 delay_INT16_2_1978 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1987:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_d_x8_3_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:867:103
    .out   (_delay_INT16_2_1978_out)
  );
  delay_INT16_2 delay_INT16_2_1979 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1988:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_d_x8_3_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:867:103
    .out   (_delay_INT16_2_1979_out)
  );
  delay_INT16_2 delay_INT16_2_1980 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1989:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:869:103
    .out   (_delay_INT16_2_1980_out)
  );
  delay_INT16_2 delay_INT16_2_1981 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1990:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:869:103
    .out   (_delay_INT16_2_1981_out)
  );
  delay_INT16_6 delay_INT16_6_1982 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1991:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:892:103
    .out   (_delay_INT16_6_1982_out)
  );
  delay_INT16_6 delay_INT16_6_1983 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1992:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:894:103
    .out   (_delay_INT16_6_1983_out)
  );
  delay_INT16_6 delay_INT16_6_1984 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1993:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:894:103
    .out   (_delay_INT16_6_1984_out)
  );
  delay_INT16_6 delay_INT16_6_1985 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1994:113
    .clock (clock),
    .reset (reset),
    .in    (_row_7_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:892:103
    .out   (_delay_INT16_6_1985_out)
  );
  delay_INT16_2 delay_INT16_2_1986 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1995:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_n_u4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:953:103
    .out   (_delay_INT16_2_1986_out)
  );
  delay_INT16_2 delay_INT16_2_1987 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1996:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_n_u5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:957:103
    .out   (_delay_INT16_2_1987_out)
  );
  delay_INT16_2 delay_INT16_2_1988 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1997:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_n_u6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:965:103
    .out   (_delay_INT16_2_1988_out)
  );
  delay_INT16_2 delay_INT16_2_1989 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1998:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_n_u7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:969:103
    .out   (_delay_INT16_2_1989_out)
  );
  delay_INT16_1 delay_INT16_1_1990 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1999:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:942:103
    .out   (_delay_INT16_1_1990_out)
  );
  delay_INT16_1 delay_INT16_1_1991 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2000:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:942:103
    .out   (_delay_INT16_1_1991_out)
  );
  delay_INT16_2 delay_INT16_2_1992 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2001:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_n_u2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:981:103
    .out   (_delay_INT16_2_1992_out)
  );
  delay_INT16_2 delay_INT16_2_1993 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2002:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_n_u3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:985:103
    .out   (_delay_INT16_2_1993_out)
  );
  delay_INT16_4 delay_INT16_4_1994 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2003:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:974:103
    .out   (_delay_INT16_4_1994_out)
  );
  delay_INT16_4 delay_INT16_4_1995 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2004:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:974:103
    .out   (_delay_INT16_4_1995_out)
  );
  delay_INT16_4 delay_INT16_4_1996 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2005:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:976:103
    .out   (_delay_INT16_4_1996_out)
  );
  delay_INT16_4 delay_INT16_4_1997 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2006:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:976:103
    .out   (_delay_INT16_4_1997_out)
  );
  delay_INT16_6 delay_INT16_6_1998 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2007:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1002:103
    .out   (_delay_INT16_6_1998_out)
  );
  delay_INT16_6 delay_INT16_6_1999 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2008:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1004:103
    .out   (_delay_INT16_6_1999_out)
  );
  delay_INT16_6 delay_INT16_6_2000 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2009:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1004:103
    .out   (_delay_INT16_6_2000_out)
  );
  delay_INT16_6 delay_INT16_6_2001 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2010:113
    .clock (clock),
    .reset (reset),
    .in    (_col_0_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1002:103
    .out   (_delay_INT16_6_2001_out)
  );
  delay_INT16_2 delay_INT16_2_2002 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2011:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_n_u4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1071:103
    .out   (_delay_INT16_2_2002_out)
  );
  delay_INT16_2 delay_INT16_2_2003 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2012:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_n_u5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1075:103
    .out   (_delay_INT16_2_2003_out)
  );
  delay_INT16_2 delay_INT16_2_2004 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2013:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_n_u6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1083:103
    .out   (_delay_INT16_2_2004_out)
  );
  delay_INT16_2 delay_INT16_2_2005 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2014:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_n_u7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1087:103
    .out   (_delay_INT16_2_2005_out)
  );
  delay_INT16_1 delay_INT16_1_2006 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2015:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1060:103
    .out   (_delay_INT16_1_2006_out)
  );
  delay_INT16_1 delay_INT16_1_2007 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2016:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1060:103
    .out   (_delay_INT16_1_2007_out)
  );
  delay_INT16_2 delay_INT16_2_2008 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2017:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_n_u2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1099:103
    .out   (_delay_INT16_2_2008_out)
  );
  delay_INT16_2 delay_INT16_2_2009 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2018:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_n_u3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1103:103
    .out   (_delay_INT16_2_2009_out)
  );
  delay_INT16_4 delay_INT16_4_2010 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2019:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1092:103
    .out   (_delay_INT16_4_2010_out)
  );
  delay_INT16_4 delay_INT16_4_2011 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2020:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1092:103
    .out   (_delay_INT16_4_2011_out)
  );
  delay_INT16_4 delay_INT16_4_2012 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2021:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1094:103
    .out   (_delay_INT16_4_2012_out)
  );
  delay_INT16_4 delay_INT16_4_2013 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2022:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1094:103
    .out   (_delay_INT16_4_2013_out)
  );
  delay_INT16_6 delay_INT16_6_2014 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2023:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1120:103
    .out   (_delay_INT16_6_2014_out)
  );
  delay_INT16_6 delay_INT16_6_2015 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2024:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1122:103
    .out   (_delay_INT16_6_2015_out)
  );
  delay_INT16_6 delay_INT16_6_2016 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2025:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1122:103
    .out   (_delay_INT16_6_2016_out)
  );
  delay_INT16_6 delay_INT16_6_2017 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2026:113
    .clock (clock),
    .reset (reset),
    .in    (_col_1_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1120:103
    .out   (_delay_INT16_6_2017_out)
  );
  delay_INT16_2 delay_INT16_2_2018 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2027:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_n_u4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1189:103
    .out   (_delay_INT16_2_2018_out)
  );
  delay_INT16_2 delay_INT16_2_2019 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2028:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_n_u5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1193:103
    .out   (_delay_INT16_2_2019_out)
  );
  delay_INT16_2 delay_INT16_2_2020 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2029:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_n_u6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1201:103
    .out   (_delay_INT16_2_2020_out)
  );
  delay_INT16_2 delay_INT16_2_2021 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2030:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_n_u7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1205:103
    .out   (_delay_INT16_2_2021_out)
  );
  delay_INT16_1 delay_INT16_1_2022 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2031:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1178:103
    .out   (_delay_INT16_1_2022_out)
  );
  delay_INT16_1 delay_INT16_1_2023 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2032:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1178:103
    .out   (_delay_INT16_1_2023_out)
  );
  delay_INT16_2 delay_INT16_2_2024 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2033:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_n_u2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1217:103
    .out   (_delay_INT16_2_2024_out)
  );
  delay_INT16_2 delay_INT16_2_2025 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2034:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_n_u3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1221:103
    .out   (_delay_INT16_2_2025_out)
  );
  delay_INT16_4 delay_INT16_4_2026 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2035:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1210:103
    .out   (_delay_INT16_4_2026_out)
  );
  delay_INT16_4 delay_INT16_4_2027 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2036:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1210:103
    .out   (_delay_INT16_4_2027_out)
  );
  delay_INT16_4 delay_INT16_4_2028 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2037:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1212:103
    .out   (_delay_INT16_4_2028_out)
  );
  delay_INT16_4 delay_INT16_4_2029 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2038:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1212:103
    .out   (_delay_INT16_4_2029_out)
  );
  delay_INT16_6 delay_INT16_6_2030 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2039:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1238:103
    .out   (_delay_INT16_6_2030_out)
  );
  delay_INT16_6 delay_INT16_6_2031 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2040:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1240:103
    .out   (_delay_INT16_6_2031_out)
  );
  delay_INT16_6 delay_INT16_6_2032 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2041:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1240:103
    .out   (_delay_INT16_6_2032_out)
  );
  delay_INT16_6 delay_INT16_6_2033 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2042:113
    .clock (clock),
    .reset (reset),
    .in    (_col_2_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1238:103
    .out   (_delay_INT16_6_2033_out)
  );
  delay_INT16_2 delay_INT16_2_2034 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2043:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_n_u4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1307:103
    .out   (_delay_INT16_2_2034_out)
  );
  delay_INT16_2 delay_INT16_2_2035 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2044:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_n_u5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1311:103
    .out   (_delay_INT16_2_2035_out)
  );
  delay_INT16_2 delay_INT16_2_2036 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2045:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_n_u6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1319:103
    .out   (_delay_INT16_2_2036_out)
  );
  delay_INT16_2 delay_INT16_2_2037 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2046:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_n_u7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1323:103
    .out   (_delay_INT16_2_2037_out)
  );
  delay_INT16_1 delay_INT16_1_2038 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2047:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1296:103
    .out   (_delay_INT16_1_2038_out)
  );
  delay_INT16_1 delay_INT16_1_2039 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2048:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1296:103
    .out   (_delay_INT16_1_2039_out)
  );
  delay_INT16_2 delay_INT16_2_2040 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2049:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_n_u2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1335:103
    .out   (_delay_INT16_2_2040_out)
  );
  delay_INT16_2 delay_INT16_2_2041 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2050:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_n_u3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1339:103
    .out   (_delay_INT16_2_2041_out)
  );
  delay_INT16_4 delay_INT16_4_2042 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2051:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1328:103
    .out   (_delay_INT16_4_2042_out)
  );
  delay_INT16_4 delay_INT16_4_2043 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2052:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1328:103
    .out   (_delay_INT16_4_2043_out)
  );
  delay_INT16_4 delay_INT16_4_2044 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2053:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1330:103
    .out   (_delay_INT16_4_2044_out)
  );
  delay_INT16_4 delay_INT16_4_2045 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2054:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1330:103
    .out   (_delay_INT16_4_2045_out)
  );
  delay_INT16_6 delay_INT16_6_2046 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2055:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1356:103
    .out   (_delay_INT16_6_2046_out)
  );
  delay_INT16_6 delay_INT16_6_2047 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2056:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1358:103
    .out   (_delay_INT16_6_2047_out)
  );
  delay_INT16_6 delay_INT16_6_2048 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2057:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1358:103
    .out   (_delay_INT16_6_2048_out)
  );
  delay_INT16_6 delay_INT16_6_2049 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2058:113
    .clock (clock),
    .reset (reset),
    .in    (_col_3_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1356:103
    .out   (_delay_INT16_6_2049_out)
  );
  delay_INT16_2 delay_INT16_2_2050 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2059:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_n_u4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1425:103
    .out   (_delay_INT16_2_2050_out)
  );
  delay_INT16_2 delay_INT16_2_2051 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2060:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_n_u5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1429:103
    .out   (_delay_INT16_2_2051_out)
  );
  delay_INT16_2 delay_INT16_2_2052 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2061:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_n_u6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1437:103
    .out   (_delay_INT16_2_2052_out)
  );
  delay_INT16_2 delay_INT16_2_2053 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2062:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_n_u7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1441:103
    .out   (_delay_INT16_2_2053_out)
  );
  delay_INT16_1 delay_INT16_1_2054 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2063:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1414:103
    .out   (_delay_INT16_1_2054_out)
  );
  delay_INT16_1 delay_INT16_1_2055 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2064:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1414:103
    .out   (_delay_INT16_1_2055_out)
  );
  delay_INT16_2 delay_INT16_2_2056 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2065:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_n_u2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1453:103
    .out   (_delay_INT16_2_2056_out)
  );
  delay_INT16_2 delay_INT16_2_2057 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2066:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_n_u3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1457:103
    .out   (_delay_INT16_2_2057_out)
  );
  delay_INT16_4 delay_INT16_4_2058 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2067:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1446:103
    .out   (_delay_INT16_4_2058_out)
  );
  delay_INT16_4 delay_INT16_4_2059 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2068:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1446:103
    .out   (_delay_INT16_4_2059_out)
  );
  delay_INT16_4 delay_INT16_4_2060 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2069:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1448:103
    .out   (_delay_INT16_4_2060_out)
  );
  delay_INT16_4 delay_INT16_4_2061 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2070:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1448:103
    .out   (_delay_INT16_4_2061_out)
  );
  delay_INT16_6 delay_INT16_6_2062 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2071:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1474:103
    .out   (_delay_INT16_6_2062_out)
  );
  delay_INT16_6 delay_INT16_6_2063 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2072:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1476:103
    .out   (_delay_INT16_6_2063_out)
  );
  delay_INT16_6 delay_INT16_6_2064 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2073:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1476:103
    .out   (_delay_INT16_6_2064_out)
  );
  delay_INT16_6 delay_INT16_6_2065 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2074:113
    .clock (clock),
    .reset (reset),
    .in    (_col_4_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1474:103
    .out   (_delay_INT16_6_2065_out)
  );
  delay_INT16_2 delay_INT16_2_2066 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2075:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_n_u4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1543:103
    .out   (_delay_INT16_2_2066_out)
  );
  delay_INT16_2 delay_INT16_2_2067 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2076:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_n_u5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1547:103
    .out   (_delay_INT16_2_2067_out)
  );
  delay_INT16_2 delay_INT16_2_2068 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2077:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_n_u6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1555:103
    .out   (_delay_INT16_2_2068_out)
  );
  delay_INT16_2 delay_INT16_2_2069 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2078:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_n_u7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1559:103
    .out   (_delay_INT16_2_2069_out)
  );
  delay_INT16_1 delay_INT16_1_2070 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2079:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1532:103
    .out   (_delay_INT16_1_2070_out)
  );
  delay_INT16_1 delay_INT16_1_2071 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2080:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1532:103
    .out   (_delay_INT16_1_2071_out)
  );
  delay_INT16_2 delay_INT16_2_2072 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2081:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_n_u2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1571:103
    .out   (_delay_INT16_2_2072_out)
  );
  delay_INT16_2 delay_INT16_2_2073 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2082:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_n_u3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1575:103
    .out   (_delay_INT16_2_2073_out)
  );
  delay_INT16_4 delay_INT16_4_2074 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2083:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1564:103
    .out   (_delay_INT16_4_2074_out)
  );
  delay_INT16_4 delay_INT16_4_2075 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2084:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1564:103
    .out   (_delay_INT16_4_2075_out)
  );
  delay_INT16_4 delay_INT16_4_2076 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2085:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1566:103
    .out   (_delay_INT16_4_2076_out)
  );
  delay_INT16_4 delay_INT16_4_2077 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2086:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1566:103
    .out   (_delay_INT16_4_2077_out)
  );
  delay_INT16_6 delay_INT16_6_2078 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2087:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1592:103
    .out   (_delay_INT16_6_2078_out)
  );
  delay_INT16_6 delay_INT16_6_2079 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2088:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1594:103
    .out   (_delay_INT16_6_2079_out)
  );
  delay_INT16_6 delay_INT16_6_2080 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2089:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1594:103
    .out   (_delay_INT16_6_2080_out)
  );
  delay_INT16_6 delay_INT16_6_2081 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2090:113
    .clock (clock),
    .reset (reset),
    .in    (_col_5_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1592:103
    .out   (_delay_INT16_6_2081_out)
  );
  delay_INT16_2 delay_INT16_2_2082 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2091:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_n_u6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1673:103
    .out   (_delay_INT16_2_2082_out)
  );
  delay_INT16_2 delay_INT16_2_2083 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2092:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_n_u7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1677:103
    .out   (_delay_INT16_2_2083_out)
  );
  delay_INT16_1 delay_INT16_1_2084 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2093:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1650:103
    .out   (_delay_INT16_1_2084_out)
  );
  delay_INT16_1 delay_INT16_1_2085 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2094:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1650:103
    .out   (_delay_INT16_1_2085_out)
  );
  delay_INT16_2 delay_INT16_2_2086 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2095:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_n_u2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1689:103
    .out   (_delay_INT16_2_2086_out)
  );
  delay_INT16_2 delay_INT16_2_2087 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2096:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_n_u3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1693:103
    .out   (_delay_INT16_2_2087_out)
  );
  delay_INT16_4 delay_INT16_4_2088 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2097:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1682:103
    .out   (_delay_INT16_4_2088_out)
  );
  delay_INT16_4 delay_INT16_4_2089 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2098:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1682:103
    .out   (_delay_INT16_4_2089_out)
  );
  delay_INT16_4 delay_INT16_4_2090 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2099:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1684:103
    .out   (_delay_INT16_4_2090_out)
  );
  delay_INT16_4 delay_INT16_4_2091 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2100:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1684:103
    .out   (_delay_INT16_4_2091_out)
  );
  delay_INT16_6 delay_INT16_6_2092 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2101:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1710:103
    .out   (_delay_INT16_6_2092_out)
  );
  delay_INT16_6 delay_INT16_6_2093 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2102:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1712:103
    .out   (_delay_INT16_6_2093_out)
  );
  delay_INT16_6 delay_INT16_6_2094 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2103:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1712:103
    .out   (_delay_INT16_6_2094_out)
  );
  delay_INT16_6 delay_INT16_6_2095 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2104:113
    .clock (clock),
    .reset (reset),
    .in    (_col_6_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1710:103
    .out   (_delay_INT16_6_2095_out)
  );
  delay_INT16_2 delay_INT16_2_2096 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2105:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_n_u4_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1779:103
    .out   (_delay_INT16_2_2096_out)
  );
  delay_INT16_2 delay_INT16_2_2097 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2106:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_n_u5_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1783:103
    .out   (_delay_INT16_2_2097_out)
  );
  delay_INT16_2 delay_INT16_2_2098 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2107:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_n_u6_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1791:103
    .out   (_delay_INT16_2_2098_out)
  );
  delay_INT16_2 delay_INT16_2_2099 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2108:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_n_u7_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1795:103
    .out   (_delay_INT16_2_2099_out)
  );
  delay_INT16_1 delay_INT16_1_2100 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2109:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_d_x1_0_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1768:103
    .out   (_delay_INT16_1_2100_out)
  );
  delay_INT16_1 delay_INT16_1_2101 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2110:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_d_x1_0_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1768:103
    .out   (_delay_INT16_1_2101_out)
  );
  delay_INT16_2 delay_INT16_2_2102 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2111:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_n_u2_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1807:103
    .out   (_delay_INT16_2_2102_out)
  );
  delay_INT16_2 delay_INT16_2_2103 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2112:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_n_u3_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1811:103
    .out   (_delay_INT16_2_2103_out)
  );
  delay_INT16_4 delay_INT16_4_2104 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2113:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_d_x8_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1800:103
    .out   (_delay_INT16_4_2104_out)
  );
  delay_INT16_4 delay_INT16_4_2105 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2114:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_d_x8_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1800:103
    .out   (_delay_INT16_4_2105_out)
  );
  delay_INT16_4 delay_INT16_4_2106 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2115:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_d_x0_1_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1802:103
    .out   (_delay_INT16_4_2106_out)
  );
  delay_INT16_4 delay_INT16_4_2107 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2116:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_d_x0_1_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1802:103
    .out   (_delay_INT16_4_2107_out)
  );
  delay_INT16_6 delay_INT16_6_2108 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2117:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_d_x3_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1828:103
    .out   (_delay_INT16_6_2108_out)
  );
  delay_INT16_6 delay_INT16_6_2109 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2118:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_d_x0_2_y),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1830:103
    .out   (_delay_INT16_6_2109_out)
  );
  delay_INT16_6 delay_INT16_6_2110 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2119:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_d_x0_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1830:103
    .out   (_delay_INT16_6_2110_out)
  );
  delay_INT16_6 delay_INT16_6_2111 (	// ./test/data/hil/idct/outputFirrtlIdct.mlir:2120:113
    .clock (clock),
    .reset (reset),
    .in    (_col_7_d_x3_2_z),	// ./test/data/hil/idct/outputFirrtlIdct.mlir:1828:103
    .out   (_delay_INT16_6_2111_out)
  );
endmodule

