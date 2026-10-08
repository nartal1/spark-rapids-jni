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
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.VariantUtils;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VariantParserTest {
  private static final HostColumnVector.ListType BINARY_TYPE = new HostColumnVector.ListType(
      true, new HostColumnVector.BasicType(false, DType.UINT8));

  private static List<Byte> bytes(int... values) {
    List<Byte> result = new ArrayList<>(values.length);
    for (int value : values) {
      result.add((byte) value);
    }
    return result;
  }

  private static void assertStatuses(VariantParser.ParsedVariant result, byte... expected) {
    assertEquals(DType.UINT8, result.getStatus().getType());
    assertEquals(0, result.getStatus().getNullCount());
    try (HostColumnVector statuses = result.getStatus().copyToHost()) {
      assertEquals(expected.length, statuses.getRowCount());
      for (int i = 0; i < expected.length; ++i) {
        assertEquals(expected[i], statuses.getByte(i), "status at row " + i);
      }
    }
  }

  @Test
  void scalarGoldensAndSqlNullAreDistinct() {
    try (ColumnVector input = ColumnVector.fromStrings(null, "null", "1", "true", "false", "\"h\"");
         VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, false);
         HostColumnVector variant = result.getVariant().copyToHost()) {
      assertStatuses(result, VariantParser.ROW_NULL, VariantParser.SUCCESS, VariantParser.SUCCESS,
          VariantParser.SUCCESS, VariantParser.SUCCESS, VariantParser.SUCCESS);
      assertEquals(DType.STRUCT, result.getVariant().getType());
      assertTrue(variant.isNull(0));
      assertFalse(variant.isNull(1), "JSON null is a non-null VARIANT");
      byte[][] values = {{0}, {12, 1}, {4}, {8}, {5, 104}};
      for (int i = 1; i < 6; ++i) {
        assertArrayEquals(new byte[]{1, 0, 0},
            variant.getChildColumnView(0).getBytesFromList(i));
        assertArrayEquals(values[i - 1], variant.getChildColumnView(1).getBytesFromList(i));
      }
    }
  }

  @Test
  void invalidRowsAndDuplicateKeysHaveSeparateStatuses() {
    try (ColumnVector input = ColumnVector.fromStrings(
             "{", "", "{\"a\":1,\"a\":2}", "{\"a\":1}", "{\"a\":1} junk");
         VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, false);
         HostColumnVector variant = result.getVariant().copyToHost()) {
      assertStatuses(result, VariantParser.INVALID_JSON, VariantParser.INVALID_JSON,
          VariantParser.DUPLICATE_KEY, VariantParser.SUCCESS, VariantParser.SUCCESS);
      assertTrue(variant.isNull(0));
      assertTrue(variant.isNull(1));
      assertTrue(variant.isNull(2));
      assertFalse(variant.isNull(3));
      assertFalse(variant.isNull(4));
      assertArrayEquals(variant.getChildColumnView(1).getBytesFromList(3),
          variant.getChildColumnView(1).getBytesFromList(4));
    }
  }

  @Test
  void allowedDuplicatesKeepLastValue() {
    try (ColumnVector input = ColumnVector.fromStrings("{\"a\":1,\"a\":2}");
         VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, true);
         ColumnVector extracted = VariantUtils.extractVariantField(
             result.getVariant(), "$.a", DType.INT8);
         ColumnVector expected = ColumnVector.fromBytes((byte) 2)) {
      assertStatuses(result, VariantParser.SUCCESS);
      assertColumnsAreEqual(expected, extracted);
    }
  }

  @Test
  void nestedAndArrayOutputCanBeExtracted() {
    try (ColumnVector input = ColumnVector.fromStrings(
             "{\"a\":{\"b\":[1,2]},\"x\":true}", "{\"a\":{\"b\":[3]}}");
         VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, false);
         ColumnVector extracted = VariantUtils.extractVariantField(
             result.getVariant(), "$.a.b[0]", DType.INT8);
         ColumnVector expected = ColumnVector.fromBytes((byte) 1, (byte) 3)) {
      assertStatuses(result, VariantParser.SUCCESS, VariantParser.SUCCESS);
      assertColumnsAreEqual(expected, extracted);
    }
  }

  @Test
  void unsupportedDepthIsNotMalformedJson() {
    char[] opening = new char[128];
    char[] closing = new char[128];
    Arrays.fill(opening, '[');
    Arrays.fill(closing, ']');
    String deep = new String(opening) + "0" + new String(closing);
    try (ColumnVector input = ColumnVector.fromStrings(deep, "[0]", "{");
         VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, false)) {
      assertStatuses(result, VariantParser.UNSUPPORTED_INPUT, VariantParser.SUCCESS,
          VariantParser.INVALID_JSON);
    }
  }

  @Test
  void emptyInputAndInputViewsAreSupported() {
    try (ColumnVector input = ColumnVector.fromStrings();
         VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, false)) {
      assertEquals(0, result.getVariant().getRowCount());
      assertStatuses(result);
      assertFalse(VariantParser.requiresLegacySparkFallback(input));
    }
    try (ColumnVector input = ColumnVector.fromStrings("{", "1", "2", "é")) {
      ColumnView[] views = input.splitAsViews(1, 3);
      try (ColumnView first = views[0];
           ColumnView middle = views[1];
           ColumnView last = views[2];
           VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(middle, false)) {
        assertStatuses(result, VariantParser.SUCCESS, VariantParser.SUCCESS);
        assertFalse(VariantParser.requiresLegacySparkFallback(middle));
        assertTrue(VariantParser.requiresLegacySparkFallback(last));
      }
    }
  }

  @Test
  void legacySparkPreflightIsConservativeAndDoesNotValidateJson() {
    try (ColumnVector ascii = ColumnVector.fromStrings(null, "{", "{\"a\":1}", "\"a\\nb\"");
         ColumnVector unicode = ColumnVector.fromStrings("{\"é\":1}");
         ColumnVector escape = ColumnVector.fromStrings("{\"\\u0061\":1}")) {
      assertFalse(VariantParser.requiresLegacySparkFallback(ascii));
      assertTrue(VariantParser.requiresLegacySparkFallback(unicode));
      assertTrue(VariantParser.requiresLegacySparkFallback(escape));
    }
  }

  @Test
  void heterogeneousObjectsAndEmptyKeysArePreserved() {
    try (ColumnVector input = ColumnVector.fromStrings(
             "{\"\":1,\"a.b\":2}", "{\"other\":true}");
         VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, false);
         HostColumnVector variant = result.getVariant().copyToHost()) {
      assertStatuses(result, VariantParser.SUCCESS, VariantParser.SUCCESS);
      assertArrayEquals(new byte[]{1, 2, 0, 0, 3, 97, 46, 98},
          variant.getChildColumnView(0).getBytesFromList(0));
      assertArrayEquals(new byte[]{1, 1, 0, 5, 111, 116, 104, 101, 114},
          variant.getChildColumnView(0).getBytesFromList(1));
    }
  }

  @Test
  void wideAsciiObjectsStayEligibleAndExtractCorrectly() {
    StringBuilder wide = new StringBuilder("{");
    for (int i = 0; i < 180; ++i) {
      if (i != 0) {
        wide.append(',');
      }
      wide.append('"').append("ascii_long_field_").append(i).append("\":1");
    }
    wide.append('}');
    try (ColumnVector input = ColumnVector.fromStrings(wide.toString());
         VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, false);
         ColumnVector extracted = VariantUtils.extractVariantField(
             result.getVariant(), "$.ascii_long_field_179", DType.INT8);
         ColumnVector expected = ColumnVector.fromBytes((byte) 1);
         ColumnView metadata = result.getVariant().getChildColumnView(0)) {
      assertFalse(VariantParser.requiresLegacySparkFallback(input));
      assertFalse(VariantParser.requiresLegacyMetadataFallback(metadata));
      assertStatuses(result, VariantParser.SUCCESS);
      assertColumnsAreEqual(expected, extracted);
    }
  }

  @Test
  void metadataCompatibilityGuardChecksOnlyDictionaryKeys() {
    char[] chars = new char[63];
    Arrays.fill(chars, 'a');
    String offsetQuestionMark = "{\"" + new String(chars) + "\":1}";
    try (ColumnVector input = ColumnVector.fromStrings(
             "{\"?\":1,\"a\":2}", "{\"é\":1,\"a\":2}", offsetQuestionMark, "{}");
         VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, false);
         ColumnView metadata = result.getVariant().getChildColumnView(0)) {
      ColumnView[] views = metadata.splitAsViews(1);
      try (ColumnView questionKey = views[0]; ColumnView otherKeys = views[1]) {
        assertTrue(VariantParser.requiresLegacyMetadataFallback(questionKey));
        assertFalse(VariantParser.requiresLegacyMetadataFallback(otherKeys),
            "Unicode alone, '?' in binary offsets, and empty dictionaries stay eligible");
      }
    }
    // Spark encodes an unpaired-surrogate key as '?' in its dictionary. Its object field IDs
    // remain ordered by the original UTF-16 keys, so native binary lookup may miss ASCII 'a'.
    try (ColumnVector surrogateMetadata = ColumnVector.fromLists(
             BINARY_TYPE, bytes(1, 2, 0, 1, 2, 63, 97));
         ColumnVector stringMetadata = ColumnVector.fromStrings(
             "\u0001\u0001\u0000\u0001?", "\u0001\u0000\u0000")) {
      assertTrue(VariantParser.requiresLegacyMetadataFallback(surrogateMetadata));
      assertTrue(VariantParser.requiresLegacyMetadataFallback(stringMetadata));
      ColumnView[] views = stringMetadata.splitAsViews(1);
      try (ColumnView questionKey = views[0]; ColumnView empty = views[1]) {
        assertTrue(VariantParser.requiresLegacyMetadataFallback(questionKey));
        assertFalse(VariantParser.requiresLegacyMetadataFallback(empty));
      }
    }
  }

  @Test
  void metadataCompatibilityGuardHandlesNullsEmptyInputAndMalformedOffsets() {
    try (ColumnVector empty = ColumnVector.fromLists(BINARY_TYPE);
         ColumnVector nulls = ColumnVector.fromLists(BINARY_TYPE, (List<Byte>) null);
         ColumnVector wideEmpty = ColumnVector.fromLists(BINARY_TYPE,
             bytes(129, 0, 0, 0, 0, 0, 0), bytes(193, 0, 0, 0, 0, 0, 0, 0, 0))) {
      assertFalse(VariantParser.requiresLegacyMetadataFallback(empty));
      assertFalse(VariantParser.requiresLegacyMetadataFallback(nulls));
      assertFalse(VariantParser.requiresLegacyMetadataFallback(wideEmpty));
    }
    List<List<Byte>> malformed = Arrays.asList(
        Collections.emptyList(), bytes(2, 0, 0), bytes(1, 255, 0), bytes(193, 0, 0),
        bytes(1, 0, 1), bytes(1, 1, 0, 5, 97), bytes(1, 2, 0, 2, 1, 97, 98));
    for (List<Byte> row : malformed) {
      try (ColumnVector metadata = ColumnVector.fromLists(BINARY_TYPE, row)) {
        assertTrue(VariantParser.requiresLegacyMetadataFallback(metadata), row.toString());
      }
    }
    HostColumnVector.ListType ints = new HostColumnVector.ListType(
        true, new HostColumnVector.BasicType(false, DType.INT32));
    try (ColumnVector wrongChild = ColumnVector.fromLists(ints, Arrays.asList(1, 0, 0))) {
      assertThrows(IllegalArgumentException.class,
          () -> VariantParser.requiresLegacyMetadataFallback(wrongChild));
    }
  }

  @Test
  void metadataCompatibilityGuardRespectsParentNullsAndSlicedParentMasks() {
    try (ColumnVector input = ColumnVector.fromStrings("1", null, "2");
         VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, false);
         ColumnView metadata = result.getVariant().getChildColumnView(0)) {
      assertFalse(VariantParser.requiresLegacyMetadataFallback(metadata, result.getVariant()),
          "empty metadata on SQL-null parents must not force CPU extraction");
      ColumnView[] parents = result.getVariant().splitAsViews(1, 2);
      ColumnView[] children = metadata.splitAsViews(1, 2);
      try (ColumnView firstParent = parents[0]; ColumnView nullParent = parents[1];
           ColumnView lastParent = parents[2]; ColumnView firstChild = children[0];
           ColumnView nullChild = children[1]; ColumnView lastChild = children[2]) {
        assertFalse(VariantParser.requiresLegacyMetadataFallback(firstChild, firstParent));
        assertFalse(VariantParser.requiresLegacyMetadataFallback(nullChild, nullParent));
        assertFalse(VariantParser.requiresLegacyMetadataFallback(lastChild, lastParent));
      }
      try (ColumnVector oneRow = ColumnVector.fromLists(BINARY_TYPE, bytes(1, 0, 0))) {
        assertThrows(IllegalArgumentException.class,
            () -> VariantParser.requiresLegacyMetadataFallback(oneRow, result.getVariant()));
      }
    }
    HostColumnVector.StructType variantType = new HostColumnVector.StructType(
        true, Arrays.asList(BINARY_TYPE, BINARY_TYPE));
    try (ColumnVector validParentWithEmptyMetadata = ColumnVector.fromStructs(variantType,
             new HostColumnVector.StructData(Collections.emptyList(), Collections.emptyList()));
         ColumnView metadata = validParentWithEmptyMetadata.getChildColumnView(0)) {
      assertTrue(VariantParser.requiresLegacyMetadataFallback(
          metadata, validParentWithEmptyMetadata),
          "a non-null parent with empty metadata remains malformed");
    }
    try (ColumnVector validParentWithNullMetadata = ColumnVector.fromStructs(variantType,
             new HostColumnVector.StructData(null, Collections.emptyList()));
         ColumnView metadata = validParentWithNullMetadata.getChildColumnView(0)) {
      assertTrue(VariantParser.requiresLegacyMetadataFallback(metadata, validParentWithNullMetadata),
          "a non-null parent with null metadata remains malformed");
    }
    try (ColumnVector metadata = ColumnVector.fromLists(BINARY_TYPE, bytes(1, 0, 0));
         ColumnVector wrongParent = ColumnVector.fromInts(1)) {
      assertThrows(IllegalArgumentException.class,
          () -> VariantParser.requiresLegacyMetadataFallback(metadata, wrongParent));
    }
  }

  @Test
  void largeRowsRequireLegacySparkFallback() {
    char[] chars = new char[16 * 1024 * 1024];
    Arrays.fill(chars, 'a');
    try (ColumnVector input = ColumnVector.fromStrings(new String(chars))) {
      assertTrue(VariantParser.requiresLegacySparkFallback(input));
    }
  }

  @Test
  void resultCanRetainColumnAndClosesIdempotently() {
    try (ColumnVector input = ColumnVector.fromStrings("1")) {
      VariantParser.ParsedVariant result = VariantParser.parseJsonToVariant(input, false);
      try (ColumnVector retained = result.getVariant().incRefCount()) {
        result.close();
        result.close();
        assertEquals(1, retained.getRowCount());
        assertThrows(IllegalStateException.class, result::getVariant);
        assertThrows(IllegalStateException.class, result::getStatus);
      } finally {
        result.close();
      }
    }
  }

  @Test
  void invalidTypesAndNullArgumentsAreRejected() {
    assertThrows(NullPointerException.class, () -> VariantParser.parseJsonToVariant(null, false));
    assertThrows(NullPointerException.class, () -> VariantParser.requiresLegacySparkFallback(null));
    assertThrows(NullPointerException.class, () -> VariantParser.requiresLegacyMetadataFallback(null));
    try (ColumnVector integers = ColumnVector.fromInts(1)) {
      assertThrows(IllegalArgumentException.class,
          () -> VariantParser.parseJsonToVariant(integers, false));
      assertThrows(IllegalArgumentException.class,
          () -> VariantParser.requiresLegacySparkFallback(integers));
      assertThrows(IllegalArgumentException.class,
          () -> VariantParser.requiresLegacyMetadataFallback(integers));
    }
  }
}
