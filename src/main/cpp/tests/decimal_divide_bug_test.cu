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

#include "../src/decimal_utils.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>

using namespace cudf;
using namespace cudf::test;

template <typename T>
using fp_wrapper = fixed_point_column_wrapper<T>;

class DecimalDivideBugTest : public BaseFixture {};

std::string to_string(__int128_t value)
{
  if (value == 0) return "0";
  
  bool is_negative = value < 0;
  __int128_t abs_value = is_negative ? -value : value;
  
  std::string result;
  while (abs_value > 0) {
    result = char('0' + abs_value % 10) + result;
    abs_value /= 10;
  }
  
  if (is_negative) result = "-" + result;
  return result;
}

TEST_F(DecimalDivideBugTest, TestActualDecimalDivisionCase81)
{
    // Recreate the exact scenario from Spark Rapids that triggers the bug
    // This should use the real compiled decimal_utils.cu functions and trigger pow_ten(16)
    
    std::cout << "=== Testing Real Decimal Division That Triggers pow_ten(16) ===" << std::endl;
    
    // Create decimal128 columns that would trigger the problematic pow_ten(16) call
    // Based on your testing: n=81, and operations that result in pow_ten(16)
    auto lhs = fp_wrapper<__int128_t>{{81}, numeric::scale_type{0}};
    // auto rhs = fp_wrapper<__int128_t>{{1}, numeric::scale_type{-16}};  // This should trigger pow_ten(16)
    // 65300.73829 with scale 5
    auto rhs = fp_wrapper<__int128_t>{{6530073829}, numeric::scale_type{-5}};
    // auto rhs = fp_wrapper<__int128_t>{{6530073829}, numeric::scale_type{-4}};
    
    std::cout << "LHS: 81 (scale 0)" << std::endl;
    std::cout << "RHS: 65300.73829 (scale -5, effectively 65300.73829*10^-5)" << std::endl;
    // std::cout << "RHS: 653007.3829 (scale -4, effectively 65300.73829*10^-4)" << std::endl;
    // std::cout << "Expected result: 81 / (1*10^-16) = 81 * 10^16 = 810000000000000000" << std::endl;
    
    // Call the real divide_decimal128 function from decimal_utils.cu
    auto result_table = cudf::jni::divide_decimal128(
        lhs, rhs, -11, rmm::cuda_stream_default);
    
    std::cout << "Result table has " << result_table->num_columns() << " columns and " 
              << result_table->num_rows() << " rows" << std::endl;
    
    // Extract and examine the results
    const auto& overflow_col = result_table->get_column(0);  // First column is overflow flag
    const auto& result_col = result_table->get_column(1);   // Second column is the result
    
    auto overflow_view = overflow_col.view();
    auto result_view = result_col.view();
    
    auto h_overflow_pair = cudf::test::to_host<bool>(overflow_view);
    auto h_result_pair = cudf::test::to_host<__int128_t>(result_view);
    
    const auto& h_overflow = h_overflow_pair.first;
    const auto& h_result = h_result_pair.first;
    
    std::cout << "Overflow: " << (h_overflow[0] ? "true" : "false") << std::endl;
    std::cout << "Result: " << to_string(h_result[0]) << std::endl;
    
    // The bug would manifest as an incorrect result here
    // 81/65300.73829 = 0.00124041477 (actual expected result=124041477)
    // Expected: 81 / (1 * 10^-16) = 81 * 10^16 = 810000000000000000
    __int128_t expected = 124041477LL;

    // 81/653007.3829 = 0.00012404148 (actual expected result=12404148)
    // __int128_t expected = 12404148LL;
    
    if (h_result[0] != expected) {
        std::cout << std::endl << "*** COMPILER BUG DETECTED IN REAL DECIMAL DIVISION! ***" << std::endl;
        std::cout << "Expected: " << to_string(expected) << std::endl;
        std::cout << "Actual:   " << to_string(h_result[0]) << std::endl;
        std::cout << "This confirms the pow_ten(16) compiler bug in the real context!" << std::endl;
    } else {
        std::cout << std::endl << "Decimal division result is correct" << std::endl;
    }
    
    // Don't fail the test, just report findings
    EXPECT_EQ(h_result[0], expected) << "Decimal division should be correct";
}

