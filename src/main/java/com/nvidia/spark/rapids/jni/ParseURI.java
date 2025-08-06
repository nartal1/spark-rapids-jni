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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;

public class ParseURI {
  static {
    NativeDepsLoader.loadNativeDeps();
  }



  /**
   * Parse protocol for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @return A string column with protocol data extracted.
   */
  public static ColumnVector parseURIProtocol(ColumnView uriColumn) {
    return parseURIProtocol(uriColumn, false);
  }

  /**
   * Parse protocol for each URI from the incoming column with ANSI mode support.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, use ANSI-aware parsing (validity checking should be done by caller).
   * @return A string column with protocol data extracted.
   */
  public static ColumnVector parseURIProtocol(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    if (failOnError) {
      return new ColumnVector(parseProtocolAnsi(uriColumn.getNativeView(), true));
    } else {
      return new ColumnVector(parseProtocol(uriColumn.getNativeView()));
    }
  }

  /**
   * Parse host for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @return A string column with host data extracted.
   */
  public static ColumnVector parseURIHost(ColumnView uriColumn) {
    return parseURIHost(uriColumn, false);
  }

  /**
   * Parse host for each URI from the incoming column with ANSI mode support.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, use ANSI-aware parsing (validity checking should be done by caller).
   * @return A string column with host data extracted.
   */
  public static ColumnVector parseURIHost(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    if (failOnError) {
      return new ColumnVector(parseHostAnsi(uriColumn.getNativeView(), true));
    } else {
      return new ColumnVector(parseHost(uriColumn.getNativeView()));
    }
  }

  /**
   * Parse query for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQuery(ColumnView uriColumn) {
    return parseURIQuery(uriColumn, false);
  }

  /**
   * Parse query for each URI from the incoming column with ANSI mode support.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, throw exception on invalid URLs (ANSI mode).
   * @return A string column with query data extracted.
   * @throws IllegalArgumentException if failOnError is true and invalid URLs are found
   */
  public static ColumnVector parseURIQuery(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    if (failOnError) {
      return new ColumnVector(parseQueryAnsi(uriColumn.getNativeView(), true));
    } else {
      return new ColumnVector(parseQuery(uriColumn.getNativeView()));
    }
  }

  /**
   * Parse query and return a specific parameter for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @param String The parameter to extract from the query
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQueryWithLiteral(ColumnView uriColumn, String query) {
    return parseURIQueryWithLiteral(uriColumn, query, false);
  }

  /**
   * Parse query and return a specific parameter for each URI from the incoming column with ANSI mode support.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param query The parameter to extract from the query.
   * @param failOnError If true, use ANSI-aware parsing (validity checking should be done by caller).
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQueryWithLiteral(ColumnView uriColumn, String query, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    if (failOnError) {
      return new ColumnVector(parseQueryWithLiteralAnsi(uriColumn.getNativeView(), query, true));
    } else {
      return new ColumnVector(parseQueryWithLiteral(uriColumn.getNativeView(), query));
    }
  }

    /**
   * Parse query and return a specific parameter for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @param String The parameter to extract from the query
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQueryWithColumn(ColumnView uriColumn, ColumnView queryColumn) {
    return parseURIQueryWithColumn(uriColumn, queryColumn, false);
  }

  /**
   * Parse query and return a specific parameter for each URI from the incoming column with ANSI mode support.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param queryColumn The parameter to extract from the query.
   * @param failOnError If true, use ANSI-aware parsing (validity checking should be done by caller).
   * @return A string column with query data extracted.
   */
  public static ColumnVector parseURIQueryWithColumn(ColumnView uriColumn, ColumnView queryColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    assert queryColumn.getType().equals(DType.STRING) : "Query type must be String";
    if (failOnError) {
      return new ColumnVector(parseQueryWithColumnAnsi(uriColumn.getNativeView(), queryColumn.getNativeView(), true));
    } else {
      return new ColumnVector(parseQueryWithColumn(uriColumn.getNativeView(), queryColumn.getNativeView()));
    }
  }

  /**
   * Parse path for each URI from the incoming column.
   *
   * @param URIColumn The input strings column in which each row contains a URI.
   * @return A string column with the URI path extracted.
   */
  public static ColumnVector parseURIPath(ColumnView uriColumn) {
    return parseURIPath(uriColumn, false);
  }

  /**
   * Parse path for each URI from the incoming column with ANSI mode support.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param failOnError If true, use ANSI-aware parsing (validity checking should be done by caller).
   * @return A string column with the URI path extracted.
   */
  public static ColumnVector parseURIPath(ColumnView uriColumn, boolean failOnError) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    if (failOnError) {
      return new ColumnVector(parsePathAnsi(uriColumn.getNativeView(), true));
    } else {
      return new ColumnVector(parsePath(uriColumn.getNativeView()));
    }
  }

  // ANSI-aware APIs



  private static native long parseProtocol(long inputColumnHandle);
  private static native long parseHost(long inputColumnHandle);
  private static native long parseQuery(long inputColumnHandle);
  private static native long parseQueryWithLiteral(long inputColumnHandle, String query);
  private static native long parseQueryWithColumn(long inputColumnHandle, long queryColumnHandle);
  private static native long parsePath(long inputColumnHandle);

  // ANSI-aware native methods
  private static native long parseProtocolAnsi(long inputColumnHandle, boolean ansiMode);
  private static native long parseHostAnsi(long inputColumnHandle, boolean ansiMode);
  private static native long parseQueryAnsi(long inputColumnHandle, boolean ansiMode);
  private static native long parseQueryWithLiteralAnsi(long inputColumnHandle, String query, boolean ansiMode);
  private static native long parseQueryWithColumnAnsi(long inputColumnHandle, long queryColumnHandle, boolean ansiMode);
  private static native long parsePathAnsi(long inputColumnHandle, boolean ansiMode);

  /**
   * Parse protocol for each URI from the incoming column with Table-based ANSI mode support.
   *
   * @param uriColumn The input strings column in which each row contains a URI.
   * @param ansiMode If true, return a table with validity information; if false, return table with just parsed data.
   * @return A table with parsed protocol data and optionally validity information.
   */
  public static Table parseURIProtocolAnsi(ColumnView uriColumn, boolean ansiMode) {
    assert uriColumn.getType().equals(DType.STRING) : "Input type must be String";
    return new Table(parseProtocolAnsiTable(uriColumn.getNativeView(), ansiMode));
  }

  /**
   * Check if a table returned by an ANSI parse function contains invalid URLs.
   *
   * @param table The table returned by an ANSI parse function.
   * @return true if the table has a validity column and contains invalid URLs, false otherwise.
   */
  public static boolean hasInvalidUrls(Table table) {
    // If the table has only 1 column, it's non-ANSI mode (no validity info)
    if (table.getNumberOfColumns() <= 1) {
      return false;
    }
    
    // Check if the validity column (column 1) contains any false values
    try (ColumnVector validityColumn = table.getColumn(1).copyToColumnVector()) {
      // Use reduction to check if any value is false
      // This is equivalent to checking if NOT ALL are true
      try (Scalar allResult = validityColumn.all()) {
        return !allResult.getBoolean();
      }
    }
  }

  /**
   * Extract the parsed data from a table returned by an ANSI parse function.
   *
   * @param table The table returned by an ANSI parse function.
   * @return A column vector containing the parsed data (first column of the table).
   */
  public static ColumnVector extractParsedData(Table table) {
    return table.getColumn(0).copyToColumnVector();
  }

  // New native method for table-based ANSI protocol parsing
  private static native long[] parseProtocolAnsiTable(long inputColumnHandle, boolean ansiMode);
}
