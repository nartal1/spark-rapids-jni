/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ai.rapids.cudf.Table;

import java.util.Objects;

/** JSON document-per-row parsing into the Parquet VARIANT representation. */
public final class VariantParser {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  // Values match cudf::io::parquet::experimental::variant_operation_status.
  public static final byte SUCCESS = 0;
  public static final byte ROW_NULL = 1;
  public static final byte INVALID_JSON = 8;
  public static final byte DUPLICATE_KEY = 9;
  public static final byte SIZE_LIMIT = 10;
  public static final byte UNSUPPORTED_INPUT = 11;

  private VariantParser() {}

  /**
   * Parse strings with the experimental cuDF Spark 4.0 encoding profile. The input must contain
   * valid UTF-8. A SQL null input produces ROW_NULL; JSON null produces a valid VARIANT value.
   * Every rejected row has a null VARIANT and an explicit status. In particular, UNSUPPORTED_INPUT
   * is not a JSON validation failure and must be evaluated on CPU when a result is required.
   * This low-level API does not apply Spark error handling or the compatibility preflight below.
   *
   * @param input string column or view, retained by the caller
   * @param allowDuplicateKeys keep the last occurrence of duplicate keys when true
   * @return owning result; callers must close it
   */
  public static ParsedVariant parseJsonToVariant(ColumnView input, boolean allowDuplicateKeys) {
    checkStrings(input);
    Table table = new Table(parseJsonToVariant(input.getNativeView(), allowDuplicateKeys));
    return Arms.closeIfException(table, ParsedVariant::new);
  }

  /**
   * Conservative preflight for the current Spark 4.0-compatible GPU fast path. Returns true when
   * any non-null row contains non-ASCII bytes, a Unicode escape, or at least 16 MiB of input
   * (2 MiB when SPARK_TESTING is set), or the input batch has at least 256 MiB of character data.
   * Such batches need Spark CPU evaluation until Unicode/key-order compatibility and large-row
   * limits are supported end to end. Checks raw bytes on device, including malformed UTF-8;
   * does not parse JSON or transfer the input strings to host. A false result does not imply that
   * the input is valid JSON. The caller must still inspect the parser status.
   */
  public static boolean requiresLegacySparkFallback(ColumnView input) {
    checkStrings(input);
    return requiresLegacySparkFallback(input.getNativeView());
  }

  /**
   * Check for metadata that cannot safely use the current native extractor. Spark can encode an
   * unpaired-surrogate key as '?' after sorting fields by the original UTF-16 key. That can break
   * native binary lookup even for ASCII paths. Conservatively require CPU extraction whenever a
   * dictionary key contains '?', the metadata has unsupported header flags, or its offset table
   * is malformed. Valid Unicode
   * keys alone do not trigger fallback for ASCII paths. Checks dictionary text only, not binary
   * headers or offsets, and accepts metadata represented as STRING or LIST&lt;UINT8&gt;.
   */
  public static boolean requiresLegacyMetadataFallback(ColumnView metadata) {
    return requiresLegacyMetadataFallback(metadata, null);
  }

  /**
   * Check metadata while respecting the containing VARIANT's SQL-null rows. A null parent row
   * may have an empty, non-null metadata child and must not cause fallback. Empty or null metadata
   * on a non-null parent is malformed and does cause fallback. Parent and metadata views must have
   * the same row count; sliced views are supported. A null parent argument checks metadata alone.
   */
  public static boolean requiresLegacyMetadataFallback(
      ColumnView metadata, ColumnView variantParent) {
    Objects.requireNonNull(metadata, "metadata");
    if (!DType.STRING.equals(metadata.getType()) && !DType.LIST.equals(metadata.getType())) {
      throw new IllegalArgumentException("metadata must be STRING or LIST<UINT8>");
    }
    if (variantParent != null) {
      if (!DType.STRUCT.equals(variantParent.getType())) {
        throw new IllegalArgumentException("VARIANT parent must be STRUCT");
      }
      if (metadata.getRowCount() != variantParent.getRowCount()) {
        throw new IllegalArgumentException("metadata and VARIANT parent row counts must match");
      }
    }
    return requiresLegacyMetadataFallback(metadata.getNativeView(),
        variantParent == null ? 0 : variantParent.getNativeView());
  }

  private static void checkStrings(ColumnView input) {
    Objects.requireNonNull(input, "input");
    if (!DType.STRING.equals(input.getType())) {
      throw new IllegalArgumentException("input must be a STRING column");
    }
  }

  /** Owns the parsed VARIANT column and its non-nullable UINT8 status column. */
  public static final class ParsedVariant implements AutoCloseable {
    private Table table;

    private ParsedVariant(Table table) {
      this.table = table;
    }

    /**
     * Borrowed STRUCT(metadata LIST&lt;UINT8&gt;, value LIST&lt;UINT8&gt;) column. Increment its
     * reference count to retain it after closing this result. Child order is not Spark's order.
     */
    public ColumnVector getVariant() {
      return getTable().getColumn(0);
    }

    /** Borrowed per-row status column. Increment its reference count to retain it separately. */
    public ColumnVector getStatus() {
      return getTable().getColumn(1);
    }

    private Table getTable() {
      if (table == null) {
        throw new IllegalStateException("parsed VARIANT result is closed");
      }
      return table;
    }

    @Override
    public void close() {
      Table current = table;
      table = null;
      if (current != null) {
        current.close();
      }
    }
  }

  private static native long[] parseJsonToVariant(long input, boolean allowDuplicateKeys);

  private static native boolean requiresLegacySparkFallback(long input);

  private static native boolean requiresLegacyMetadataFallback(long metadata, long variantParent);
}
