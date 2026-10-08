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

#include "cudf_jni_apis.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <bit>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {
using status = cudf::io::parquet::experimental::variant_operation_status;
static_assert(static_cast<uint8_t>(status::SUCCESS) == 0);
static_assert(static_cast<uint8_t>(status::ROW_NULL) == 1);
static_assert(static_cast<uint8_t>(status::INVALID_JSON) == 8);
static_assert(static_cast<uint8_t>(status::DUPLICATE_KEY) == 9);
static_assert(static_cast<uint8_t>(status::SIZE_LIMIT) == 10);
static_assert(static_cast<uint8_t>(status::UNSUPPORTED_INPUT) == 11);
}  // namespace

extern "C" {
JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_VariantParser_parseJsonToVariant(
  JNIEnv* env, jclass, jlong input, jboolean allow_duplicate_keys)
{
  JNI_NULL_CHECK(env, input, "input column is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const view = std::bit_cast<cudf::column_view const*>(input);
    auto statuses = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::UINT8}, view->size(), cudf::mask_state::UNALLOCATED);
    auto variant = cudf::io::parquet::experimental::parse_json_to_variant(
      cudf::strings_column_view{*view}, allow_duplicate_keys, statuses->mutable_view());
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(variant));
    columns.push_back(std::move(statuses));
    return cudf::jni::convert_table_for_return(env, std::make_unique<cudf::table>(std::move(columns)));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jboolean JNICALL
Java_com_nvidia_spark_rapids_jni_VariantParser_requiresLegacySparkFallback(
  JNIEnv* env, jclass, jlong input)
{
  JNI_NULL_CHECK(env, input, "input column is null", false);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const view = std::bit_cast<cudf::column_view const*>(input);
    CUDF_EXPECTS(view->type().id() == cudf::type_id::STRING, "input must be STRING");
    auto const stream = cudf::get_default_stream();
    if (cudf::strings_column_view{*view}.chars_size(stream) >= 256 * 1024 * 1024) {
      return true;
    }
    auto const max_row_bytes = (std::getenv("SPARK_TESTING") != nullptr ? 2 : 16) * 1024 * 1024;
    auto const device_view = cudf::column_device_view::create(*view, stream);
    auto const rows = thrust::make_counting_iterator<cudf::size_type>(0);
    return thrust::any_of(
      rmm::exec_policy(stream), rows, rows + view->size(),
      [strings = *device_view, max_row_bytes] __device__(cudf::size_type row) {
        if (strings.is_null(row)) { return false; }
        auto const str = strings.element<cudf::string_view>(row);
        auto const length = str.size_bytes();
        if (length >= max_row_bytes) { return true; }
        for (cudf::size_type i = 0; i < length; ++i) {
          auto const byte = static_cast<uint8_t>(str.data()[i]);
          if (byte >= 0x80 || (byte == 0x5c && i + 1 < length && str.data()[i + 1] == 'u')) {
            return true;
          }
        }
        return false;
      });
  }
  JNI_CATCH(env, false);
}

JNIEXPORT jboolean JNICALL
Java_com_nvidia_spark_rapids_jni_VariantParser_requiresLegacyMetadataFallback(
  JNIEnv* env, jclass, jlong metadata, jlong variant_parent)
{
  JNI_NULL_CHECK(env, metadata, "metadata column is null", false);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const view = std::bit_cast<cudf::column_view const*>(metadata);
    auto const parent = std::bit_cast<cudf::column_view const*>(variant_parent);
    if (parent != nullptr) {
      JNI_ARG_CHECK(env, parent->type().id() == cudf::type_id::STRUCT,
                    "VARIANT parent must be STRUCT", false);
      JNI_ARG_CHECK(env, parent->size() == view->size(),
                    "metadata and VARIANT parent row counts must match", false);
    }
    auto const parent_mask = parent == nullptr ? nullptr : parent->null_mask();
    auto const parent_offset = parent == nullptr ? cudf::size_type{0} : parent->offset();
    auto const has_parent = parent != nullptr;
    auto const strings = view->type().id() == cudf::type_id::STRING;
    CUDF_EXPECTS(strings || view->type().id() == cudf::type_id::LIST,
                 "metadata must be STRING or LIST<UINT8>");
    if (!strings) {
      JNI_ARG_CHECK(env,
                    cudf::lists_column_view{*view}.child().type().id() == cudf::type_id::UINT8,
                    "metadata LIST child must be UINT8",
                    false);
    }
    auto const stream = cudf::get_default_stream();
    auto const device_view = cudf::column_device_view::create(*view, stream);
    auto const rows = thrust::make_counting_iterator<cudf::size_type>(0);
    return thrust::any_of(
      rmm::exec_policy(stream), rows, rows + view->size(),
      [column = *device_view, strings, parent_mask, parent_offset, has_parent]
      __device__(cudf::size_type row) {
        if (parent_mask != nullptr && !cudf::bit_is_set(parent_mask, parent_offset + row)) {
          return false;
        }
        if (column.is_null(row)) { return has_parent; }
        uint8_t const* bytes;
        cudf::size_type length;
        if (strings) {
          auto const str = column.element<cudf::string_view>(row);
          bytes = reinterpret_cast<uint8_t const*>(str.data());
          length = str.size_bytes();
        } else {
          cudf::lists_column_device_view const lists{column};
          auto const begin = lists.offset_at(row);
          auto const end = lists.offset_at(row + 1);
          if (begin < 0 || end < begin || end > lists.child().size()) { return true; }
          if (end == begin) { return true; }
          bytes = lists.child().data<uint8_t>() + begin;
          length = end - begin;
        }
        if (length < 3 || (bytes[0] & 0x3f) != 1) { return true; }
        // Spark 4.0 uses the same width for dictionary count and key offsets. Header bits 6..7
        // encode width-1; bits 0..3 encode version 1. Offsets are relative to the key-text region.
        auto const width = static_cast<uint32_t>(((bytes[0] >> 6) & 3) + 1);
        if (length < 1 + 2 * width) { return true; }
        uint64_t count = 0;
        for (uint32_t i = 0; i < width; ++i) {
          count |= static_cast<uint64_t>(bytes[1 + i]) << (8 * i);
        }
        auto const offsets_begin = static_cast<uint64_t>(1 + width);
        auto const keys_begin = offsets_begin + (count + 1) * width;
        if (keys_begin > static_cast<uint64_t>(length)) { return true; }
        auto const extent = static_cast<uint64_t>(length) - keys_begin;
        uint64_t previous = 0;
        for (uint64_t entry = 0; entry <= count; ++entry) {
          uint64_t offset = 0;
          for (uint32_t i = 0; i < width; ++i) {
            offset |= static_cast<uint64_t>(bytes[offsets_begin + entry * width + i]) << (8 * i);
          }
          if ((entry == 0 && offset != 0) || offset < previous || offset > extent) { return true; }
          previous = offset;
        }
        // Do not scan binary header/count/offset bytes: wide ASCII dictionaries commonly have
        // high-bit bytes and '?' (0x3f) in their offsets. The terminal offset bounds the text.
        for (uint64_t i = keys_begin; i < keys_begin + previous; ++i) {
          if (bytes[i] == 0x3f) { return true; }
        }
        return false;
      });
  }
  JNI_CATCH(env, false);
}

}  // extern "C"
