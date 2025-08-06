/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.jni;

import java.net.URI;
import java.net.URISyntaxException;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Table;

public class ParseURITest {
  void testProtocol(String[] testData) {
    String[] expectedProtocolStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String scheme = null;
      try {
        URI uri = new URI(testData[i]);
        scheme = uri.getScheme();
      } catch (URISyntaxException ex) {
        // leave the scheme null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the scheme null if URI is null
      }
      expectedProtocolStrings[i] = scheme;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector expectedProtocol = ColumnVector.fromStrings(expectedProtocolStrings);
      ColumnVector protocolResult = ParseURI.parseURIProtocol(v0)) {
      AssertUtils.assertColumnsAreEqual(expectedProtocol, protocolResult);
    }
  }

  void testHost(String[] testData) {
    String[] expectedHostStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String host = null;
      try {
        URI uri = new URI(testData[i]);
        host = uri.getHost();
      } catch (URISyntaxException ex) {
        // leave the host null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the host null if URI is null
      }

      expectedHostStrings[i] = host;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector expectedHost = ColumnVector.fromStrings(expectedHostStrings);
      ColumnVector hostResult = ParseURI.parseURIHost(v0)) {
      AssertUtils.assertColumnsAreEqual(expectedHost, hostResult);
    }
  }

  void testQuery(String[] testData) {
    String[] expectedQueryStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String query = null;
      try {
        URI uri = new URI(testData[i]);
        query = uri.getRawQuery();
      } catch (URISyntaxException ex) {
        // leave the query null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the query null if URI is null
      }

      expectedQueryStrings[i] = query;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector expectedQuery = ColumnVector.fromStrings(expectedQueryStrings);
      ColumnVector queryResult = ParseURI.parseURIQuery(v0)) {
      AssertUtils.assertColumnsAreEqual(expectedQuery, queryResult);
    }
  }

  void testQuery(String[] testData, String param) {
    String[] expectedQueryStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String query = null;
      try {
        URI uri = new URI(testData[i]);
        query = uri.getRawQuery();
      } catch (URISyntaxException ex) {
        // leave the query null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the query null if URI is null
      }

      String subquery = null;

      if (query != null) {
        String[] pairs = query.split("&");
        for (String pair : pairs) {
          int idx = pair.indexOf("=");
          if (idx > 0 && pair.substring(0, idx).equals(param)) {
            subquery = pair.substring(idx + 1);
            break;
          }
        }
      }
      expectedQueryStrings[i] = subquery;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector expectedQuery = ColumnVector.fromStrings(expectedQueryStrings);
      ColumnVector queryResult = ParseURI.parseURIQueryWithLiteral(v0, param)) {
      AssertUtils.assertColumnsAreEqual(expectedQuery, queryResult);
    }
  }

  void testQuery(String[] testData, String[] params) {
    String[] expectedQueryStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String query = null;
      try {
        URI uri = new URI(testData[i]);
        query = uri.getRawQuery();
      } catch (URISyntaxException ex) {
        // leave the query null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the query null if URI is null
      }

      String subquery = null;

      if (query != null) {
        String[] pairs = query.split("&");
        for (String pair : pairs) {
          int idx = pair.indexOf("=");
          if (idx >= 0 && pair.substring(0, idx).equals(params[i])) {
            subquery = pair.substring(idx + 1);
            break;
          }
        }
      }
      expectedQueryStrings[i] = subquery;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector p0 = ColumnVector.fromStrings(params);
      ColumnVector expectedQuery = ColumnVector.fromStrings(expectedQueryStrings);
      ColumnVector queryResult = ParseURI.parseURIQueryWithColumn(v0, p0)) {
      AssertUtils.assertColumnsAreEqual(expectedQuery, queryResult);
    }
  }

  void testPath(String[] testData) {
    String[] expectedPathStrings = new String[testData.length];
    for (int i=0; i<testData.length; i++) {
      String path = null;
      try {
        URI uri = new URI(testData[i]);
        path = uri.getRawPath();
      } catch (URISyntaxException ex) {
        // leave the path null if URI is invalid
      } catch (NullPointerException ex) {
        // leave the path null if URI is null
      }
      expectedPathStrings[i] = path;
    }
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
      ColumnVector expectedPath = ColumnVector.fromStrings(expectedPathStrings);
      ColumnVector pathResult = ParseURI.parseURIPath(v0)) {
      AssertUtils.assertColumnsAreEqual(expectedPath, pathResult);
    }
  }

  @Test
  void parseURIAnsiModeTest() {
    String[] validTestData = {
        "https://www.nvidia.com:443/path?query=value#fragment",
        "https://valid.com/path"
    };
    
    String[] invalidTestData = {
        "https://www.nvidia.com:443/path?query=value#fragment",
        "invalid://[bad:IPv6]",
        "https://valid.com/path",
        null
    };
    
    // Test protocol parsing with ANSI mode on valid data
    try (ColumnVector validData = ColumnVector.fromStrings(validTestData);
         ColumnVector protocolResult = ParseURI.parseURIProtocol(validData, true)) {
      try (ColumnVector expectedProtocol = ColumnVector.fromStrings(new String[]{"https", "https"})) {
        AssertUtils.assertColumnsAreEqual(expectedProtocol, protocolResult);
      }
    }
    
    // Test that non-ANSI mode works as expected
    try (ColumnVector invalidData = ColumnVector.fromStrings(invalidTestData);
         ColumnVector protocolResult = ParseURI.parseURIProtocol(invalidData, false)) {
      // Should work fine, just return nulls for invalid URLs
      try (ColumnVector expectedProtocol = ColumnVector.fromStrings(new String[]{"https", null, "https", null})) {
        AssertUtils.assertColumnsAreEqual(expectedProtocol, protocolResult);
      }
    }
    
    // Test the Table-based API and validity checking functionality
    try (ColumnVector invalidData = ColumnVector.fromStrings(invalidTestData);
         Table protocolResultWithAnsi = ParseURI.parseURIProtocolAnsi(invalidData, true);
         Table protocolResultWithoutAnsi = ParseURI.parseURIProtocolAnsi(invalidData, false)) {
      
      // With ANSI mode, should get 2 columns: parsed data and validity
      assert protocolResultWithAnsi.getNumberOfColumns() == 2;
      // Without ANSI mode, should get 1 column: parsed data only
      assert protocolResultWithoutAnsi.getNumberOfColumns() == 1;
      
      // Test validity checking functionality
      assert ParseURI.hasInvalidUrls(protocolResultWithAnsi) : "Should detect invalid URLs";
      assert !ParseURI.hasInvalidUrls(protocolResultWithoutAnsi) : "Non-ANSI mode should not have validity info";
      
      // Verify the validity column in ANSI mode
      try (ColumnVector expectedValidity = ColumnVector.fromBoxedBooleans(new Boolean[]{true, false, true, true});
           ColumnVector actualValidity = protocolResultWithAnsi.getColumn(1).copyToColumnVector()) {
        AssertUtils.assertColumnsAreEqual(expectedValidity, actualValidity);
      }
      
      // Test extracting parsed data
      try (ColumnVector parsedData = ParseURI.extractParsedData(protocolResultWithAnsi);
           ColumnVector expectedParsed = ColumnVector.fromStrings(new String[]{"https", null, "https", null})) {
        AssertUtils.assertColumnsAreEqual(expectedParsed, parsedData);
      }
    }
  }

  @Test
  void parseURIResourceManagementTest() {
    // Test specifically to reproduce the "Close called too many times" issue
    String[] testData = {
        "https://www.nvidia.com:443/path?query=value#fragment",
        "invalid://[bad:IPv6]",
        "https://valid.com/path"
    };
    
    try (ColumnVector inputData = ColumnVector.fromStrings(testData)) {
      
      // Test the pattern that should be used in the plugin
      ColumnVector result1 = null;
      try (Table table = ParseURI.parseURIProtocolAnsi(inputData, true)) {
        // Check validity
        boolean hasInvalid = ParseURI.hasInvalidUrls(table);
        assert hasInvalid : "Should detect invalid URLs";
        
        // Extract data before table is closed
        result1 = ParseURI.extractParsedData(table);
      }
      
      // Verify the result is still valid after table was closed
      try (ColumnVector expectedResult = ColumnVector.fromStrings(new String[]{"https", null, "https"})) {
        AssertUtils.assertColumnsAreEqual(expectedResult, result1);
      }
      
      // Clean up
      if (result1 != null) {
        result1.close();
      }
    }
  }

  @Test
  void parseURIMultipleExtractionsTest() {
    // Test multiple extractions from the same table to ensure no double-close issues
    String[] testData = {
        "https://www.nvidia.com:443/path?query=value#fragment",
        "https://valid.com/path"
    };
    
    try (ColumnVector inputData = ColumnVector.fromStrings(testData)) {
      try (Table table = ParseURI.parseURIProtocolAnsi(inputData, true)) {
        
        // Extract data multiple times - create separate extractions
        ColumnVector result1 = null;
        ColumnVector result2 = null;
        try {
          result1 = ParseURI.extractParsedData(table);
          result2 = ParseURI.extractParsedData(table);
          
          // Both results should be identical
          AssertUtils.assertColumnsAreEqual(result1, result2);
          
          // Verify the content
          try (ColumnVector expected = ColumnVector.fromStrings(new String[]{"https", "https"})) {
            AssertUtils.assertColumnsAreEqual(expected, result1);
            AssertUtils.assertColumnsAreEqual(expected, result2);
          }
        } finally {
          if (result1 != null) result1.close();
          if (result2 != null) result2.close();
        }
        
        // Table should still be usable for validity checking
        assert !ParseURI.hasInvalidUrls(table) : "Should not detect invalid URLs in this test";
      }
    }
  }

  @Test
  void parseURIAnsiBasicTest() {
    // Simple test to check if ANSI functions work at all
    String[] testData = {"https://www.nvidia.com/path"};
    
    try (ColumnVector inputData = ColumnVector.fromStrings(testData)) {
      try (Table table = ParseURI.parseURIProtocolAnsi(inputData, true)) {
        // Basic checks
        assert table != null : "Table should not be null";
        assert table.getNumberOfColumns() == 2 : "Should have 2 columns, got: " + table.getNumberOfColumns();
        assert table.getRowCount() == 1 : "Should have 1 row, got: " + table.getRowCount();
        
        // Check if columns are accessible without using try-with-resources to avoid the close issue
        ColumnVector col0 = table.getColumn(0);
        ColumnVector col1 = table.getColumn(1);
        assert col0 != null : "First column should not be null";
        assert col1 != null : "Second column should not be null";
        
        System.out.println("Column 0 type: " + col0.getType());
        System.out.println("Column 1 type: " + col1.getType());
        
        // Don't close col0 and col1 as they are owned by the table
      }
    }
  }

  @Test
  void parseURIMissingQueryParamTest() {
    // Test the specific case where query parameter is missing
    String[] testData = {
        "https://secure.payment.com/process?amount=100&currency=USD",  // has 'amount'
        "http://analytics.site.com/track?event=click&user=456",        // no 'amount'
        "https://cdn.images.com/photos/image.jpg?size=large",          // no 'amount'
        "http://api.example.com/users?id=123&format=json",             // no 'amount'
        "ftp://backup.server.com/files/data.csv"                       // no query at all
    };
    
    // Test with non-ANSI mode - should return nulls for missing params
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
         ColumnVector result = ParseURI.parseURIQueryWithLiteral(v0, "amount", false)) {
      String[] expected = {"100", null, null, null, null};
      try (ColumnVector expectedCV = ColumnVector.fromStrings(expected)) {
        AssertUtils.assertColumnsAreEqual(expectedCV, result);
      }
    }
    
    // Test with different missing parameter to make sure it also returns nulls
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
         ColumnVector result = ParseURI.parseURIQueryWithLiteral(v0, "nonexistent", false)) {
      String[] expected = {null, null, null, null, null};
      try (ColumnVector expectedCV = ColumnVector.fromStrings(expected)) {
        AssertUtils.assertColumnsAreEqual(expectedCV, result);
      }
    }
    
    // Test with ANSI mode - should ALSO return nulls for missing params, not throw exceptions
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
         ColumnVector result = ParseURI.parseURIQueryWithLiteral(v0, "amount", true)) {
      String[] expected = {"100", null, null, null, null};
      try (ColumnVector expectedCV = ColumnVector.fromStrings(expected)) {
        AssertUtils.assertColumnsAreEqual(expectedCV, result);
      }
    }
    
    // Test that ANSI mode still throws for truly invalid URLs
    String[] invalidData = {"://completely-malformed"};
    try (ColumnVector v0 = ColumnVector.fromStrings(invalidData)) {
      try {
        ColumnVector result = ParseURI.parseURIQueryWithLiteral(v0, "param", true);
        result.close(); // Should not reach here
        throw new AssertionError("Expected exception for invalid URL in ANSI mode");
      } catch (RuntimeException e) {
        // Expected - invalid URL should throw exception in ANSI mode
        // assert (e.getMessage() != null && e.getMessage().contains("Invalid")) || 
        //        e instanceof ai.rapids.cudf.CudfException ||
        //        e instanceof com.nvidia.spark.rapids.jni.ExceptionWithRowIndex;
        assert e instanceof com.nvidia.spark.rapids.jni.ExceptionWithRowIndex;
      }
    }
  }

  @Test
  void parseURINullInputTest() {
    // Test that NULL inputs return NULL outputs, not exceptions, even in ANSI mode
    String[] testData = {
        "http://www.abc.com",
        null
    };
    
    // Test HOST parsing with NULL input - non-ANSI mode
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
         ColumnVector result = ParseURI.parseURIHost(v0, false)) {
      String[] expected = {"www.abc.com", null};
      try (ColumnVector expectedCV = ColumnVector.fromStrings(expected)) {
        AssertUtils.assertColumnsAreEqual(expectedCV, result);
      }
    }
    
    // Test HOST parsing with NULL input - ANSI mode (should not throw!)
    try (ColumnVector v0 = ColumnVector.fromStrings(testData);
         ColumnVector result = ParseURI.parseURIHost(v0, true)) {
      String[] expected = {"www.abc.com", null};
      try (ColumnVector expectedCV = ColumnVector.fromStrings(expected)) {
        AssertUtils.assertColumnsAreEqual(expectedCV, result);
      }
    }
  }

  @Test
  void parseURISparkTest() {
    String[] testData = {
        "http://localhost",
        "http://localhost:8080/simple/path",
        "https://mydomain.com/a/complicated/path?some=query_param&param2=val%20with%20spaces",
        "ftp://something.com:2121/files?myfilename.csv",
        "myprotocol://example.com:5432/something",
        null,
        "",
        "  "
    };
    testProtocol(testData);
    testHost(testData);
    testQuery(testData);
    testQuery(testData, "query");
    testPath(testData);
  }

  @Test
  void parseURIUTF8Test() {
    String[] testData = {
      "https:// /path/to/file",
      "https://nvidia.com/%4EV%49%44%49%41",
      "http://%77%77%77.%4EV%49%44%49%41.com",
      "http://✪↩d⁚f„⁈.ws/123"};

    testProtocol(testData);
    testHost(testData);
    testQuery(testData);
    testQuery(testData, "query");
    testPath(testData);
  }

  @Test
  void parseURIIP4Test() {
    String[] testData = {
      "https://192.168.1.100/",
      "https://192.168.1.100:8443/",
      "https://192.168.1.100.5/",
      "https://192.168.1/",
      "https://280.100.1.1/",
      "https://182.168..100/path/to/file"};

    testProtocol(testData);
    testHost(testData);
    testQuery(testData);
    testQuery(testData, "query");
    testPath(testData);
  }

  @Test
  void parseURIIP6Test() {
    String[] testData = {
      "https://[fe80::]",
      "https://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]",
      "https://[2001:0DB8:85A3:0000:0000:8A2E:0370:7334]",
      "https://[2001:db8::1:0]",
      "http://[2001:db8::2:1]",
      "https://[::1]",
      "https://[2001:db8:85a3:8d3:1319:8a2e:370:7348]:443",
      "https://[2001:db8:3333:4444:5555:6666:1.2.3.4]/path/to/file",
      "https://[2001:db8:3333:4444:5555:6666:7777:8888:1.2.3.4]/path/to/file",
      "https://[::db8:3333:4444:5555:6666:1.2.3.4]/path/to/file]",
      "https://[2001:db8:85a3:8d3:1319:8a2e:370:7348]:443",
      "https://[2001:]db8:85a3:8d3:1319:8a2e:370:7348/",
      "https://[][][][]nvidia.com/",
      "https://[2001:db8:85a3:8d3:1319:8a2e:370:7348:2001:db8:85a3]/path",
      "http://[1:2:3:4:5:6:7::]",
      "http://[::2:3:4:5:6:7:8]",
      "http://[fe80::7:8%eth0]",
      "http://[fe80::7:8%1]",
    };
    
    testProtocol(testData);
    testHost(testData);
    testQuery(testData);
    testQuery(testData, "query");
    testPath(testData);
  }
}
