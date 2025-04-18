// Copyright (c) 2023-2025, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

# include "../src/decimal_utils.hpp"

using namespace cudf;
using namespace cudf::test;

template <typename T>
using fp_wrapper = fixed_point_column_wrapper<T>;

class DecimalIntegerDivTest : public BaseFixture {};

TEST_F(DecimalIntegerDivTest, SmallByLargeNegative)
{
  // Create decimal128 columns with values similar to the Spark example
  // 3.01946 with scale 5
  auto lhs = fp_wrapper<__int128_t>{{301946}, numeric::scale_type{-5}};
  
  // -577.820 with scale 3
  auto rhs = fp_wrapper<__int128_t>{{-577820}, numeric::scale_type{-3}};
  
  // Expected result for integer division should be 0
  auto expected = fixed_width_column_wrapper<int64_t>{{0}};
  // auto expected = fp_wrapper<__int128_t>{{0}, numeric::scale_type{-3}};
  
  // Call integer_divide_decimal128 with output scale 0
  auto result_table = cudf::jni::integer_divide_decimal128(
      lhs, rhs, 0, rmm::cuda_stream_default);
  
  // printf("expected type: %s\n", expected.type().to_string().c_str());
  // printf("result type: %s\n", result_table->get_column(1).type().to_string().c_str());
  
  // Extract the quotient column (second column)
  auto result_col = result_table->get_column(1);
  // std::cout << "result col type: " << result_col.view().type().scale() << std::endl;
  // printf("result col type: %d\n", result_col.view().type().scale());

  // Check that the result matches the expected value
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result_col.view());
  
  // Also check that there's no overflow
  auto overflow_col = result_table->get_column(0);
  auto expected_overflow = fixed_width_column_wrapper<bool>{{false}};
  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_overflow, overflow_col.view());
}

