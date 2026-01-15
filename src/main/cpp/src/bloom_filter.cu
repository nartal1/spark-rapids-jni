/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "bloom_filter.hpp"
#include "hash/murmur_hash.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/logical.h>

#include <byteswap.h>

namespace spark_rapids_jni {

namespace {

using bloom_hash_type = spark_rapids_jni::murmur_hash_value_type;

// V1 bit indexing - has int32 truncation issue for large bloom filters
__device__ inline std::pair<cudf::size_type, cudf::bitmask_type> gpu_get_hash_mask_v1(
  bloom_hash_type h, cudf::size_type bloom_filter_bits)
{
  // https://github.com/apache/spark/blob/7bfbeb62cb1dc58d81243d22888faa688bad8064/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L94
  auto const index = (h < 0 ? ~h : h) % static_cast<bloom_hash_type>(bloom_filter_bits);

  // spark expects serialized bloom filters to be big endian (64 bit longs),
  // so we will produce a big endian buffer. if spark CPU ends up consuming it, it can do so
  // directly. the gpu bloom filter implementation will always be handed the same serialized buffer.
  auto const word_index = cudf::word_index(index) ^ 0x1;  // word-swizzle within 64 bit long
  auto const bit_index =
    cudf::intra_word_index(index) ^ 0x18;                 // byte swizzle within the 32 bit word

  return {word_index, (1 << bit_index)};
}

// V2 bit indexing - SPARK-47547
// V2 uses a completely different algorithm than V1:
// - combinedHash starts at h1 * Integer.MAX_VALUE (0x7FFFFFFF)
// - Each iteration adds h2 to combinedHash (accumulative)
// This function takes the pre-computed 64-bit combinedHash
__device__ inline std::pair<cudf::size_type, cudf::bitmask_type> gpu_get_hash_mask_v2(
  int64_t combined_hash, int64_t bloom_filter_bits)
{
  // Flip all bits if negative (guaranteed positive number)
  int64_t index = (combined_hash < 0) ? ~combined_hash : combined_hash;
  index = index % bloom_filter_bits;

  // spark expects serialized bloom filters to be big endian (64 bit longs),
  // so we will produce a big endian buffer.
  auto const word_index = cudf::word_index(index) ^ 0x1;  // word-swizzle within 64 bit long
  auto const bit_index =
    cudf::intra_word_index(index) ^ 0x18;                 // byte swizzle within the 32 bit word

  return {word_index, (1 << bit_index)};
}

// Integer.MAX_VALUE from Java
constexpr int64_t INTEGER_MAX_VALUE = 0x7FFFFFFF;

template <bool nullable, int bloom_version>
CUDF_KERNEL void gpu_bloom_filter_put(cudf::bitmask_type* const bloom_filter,
                                      int64_t bloom_filter_bits,
                                      cudf::column_device_view input,
                                      cudf::size_type num_hashes)
{
  size_t const tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= input.size()) { return; }

  if constexpr (nullable) {
    if (!input.is_valid(tid)) { return; }
  }

  auto const el            = input.element<int64_t>(tid);
  bloom_hash_type const h1 = MurmurHash3_32<int64_t>(0)(el);
  bloom_hash_type const h2 = MurmurHash3_32<int64_t>(h1)(el);

  if constexpr (bloom_version == 1) {
    // V1: https://github.com/apache/spark/blob/7bfbeb62cb1dc58d81243d22888faa688bad8064/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L87
    // combinedHash = h1 + (i * h2) for i = 1 to numHashes
    for (auto idx = 1; idx <= num_hashes; idx++) {
      bloom_hash_type combined_hash = h1 + (idx * h2);
      auto const [word_index, mask] = gpu_get_hash_mask_v1(combined_hash, 
                                                           static_cast<cudf::size_type>(bloom_filter_bits));
      atomicOr(bloom_filter + word_index, mask);
    }
  } else {
    // V2: https://github.com/apache/spark/blob/3ad4ecaffff22e58bc2df35bb5d3319e4bb4ba09/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImplV2.java
    // combinedHash = h1 * Integer.MAX_VALUE, then add h2 each iteration
    int64_t combined_hash = static_cast<int64_t>(h1) * INTEGER_MAX_VALUE;
    for (auto idx = 0; idx < num_hashes; idx++) {
      combined_hash += h2;
      auto const [word_index, mask] = gpu_get_hash_mask_v2(combined_hash, bloom_filter_bits);
      atomicOr(bloom_filter + word_index, mask);
    }
  }
}

template <int bloom_version>
struct bloom_probe_functor {
  cudf::bitmask_type const* const bloom_filter;
  int64_t const bloom_filter_bits;
  cudf::size_type const num_hashes;

  __device__ bool operator()(int64_t input) const
  {
    // this code could be combined with the very similar code in gpu_bloom_filter_put. i've
    // left it this way since the expectation is that we will early out fairly often, whereas
    // in the build case we never early out so doing the additional if() return check is pointless.
    bloom_hash_type const h1 = MurmurHash3_32<int64_t>(0)(input);
    bloom_hash_type const h2 = MurmurHash3_32<int64_t>(h1)(input);

    if constexpr (bloom_version == 1) {
      // V1: https://github.com/apache/spark/blob/7bfbeb62cb1dc58d81243d22888faa688bad8064/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImpl.java#L110
      // combinedHash = h1 + (i * h2) for i = 1 to numHashes
      for (auto idx = 1; idx <= num_hashes; idx++) {
        bloom_hash_type combined_hash = h1 + (idx * h2);
        auto const [word_index, mask] = gpu_get_hash_mask_v1(combined_hash,
                                                             static_cast<cudf::size_type>(bloom_filter_bits));
        if (!(bloom_filter[word_index] & mask)) { return false; }
      }
    } else {
      // V2: https://github.com/apache/spark/blob/3ad4ecaffff22e58bc2df35bb5d3319e4bb4ba09/common/sketch/src/main/java/org/apache/spark/util/sketch/BloomFilterImplV2.java
      // combinedHash = h1 * Integer.MAX_VALUE, then add h2 each iteration
      int64_t combined_hash = static_cast<int64_t>(h1) * INTEGER_MAX_VALUE;
      for (auto idx = 0; idx < num_hashes; idx++) {
        combined_hash += h2;
        auto const [word_index, mask] = gpu_get_hash_mask_v2(combined_hash, bloom_filter_bits);
        if (!(bloom_filter[word_index] & mask)) { return false; }
      }
    }
    return true;
  }
};

constexpr int spark_bloom_filter_version_v1 = 1;
constexpr int spark_bloom_filter_version_v2 = 2;

// Byte swap for V1 header (12 bytes)
bloom_filter_header_v1 byte_swap_header_v1(bloom_filter_header_v1 const& header)
{
  return {static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header.version))),
          static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header.num_hashes))),
          static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header.num_longs)))};
}

// Byte swap for V2 header (16 bytes)
bloom_filter_header_v2 byte_swap_header_v2(bloom_filter_header_v2 const& header)
{
  return {static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header.version))),
          static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header.num_hashes))),
          static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header.seed))),
          static_cast<int32_t>(bswap_32(static_cast<uint32_t>(header.num_longs)))};
}

/*
  Pack a bloom_filter_header (passed as little endian) into a bloom filter buffer.
  Supports both V1 (12 bytes) and V2 (16 bytes) headers.
*/
void pack_bloom_filter_header(cudf::device_span<uint8_t> buf,
                              bloom_filter_header const& header,
                              int version,
                              rmm::cuda_stream_view stream)
{
  if (version == 2) {
    // V2: Pack full 16-byte header
    bloom_filter_header_v2 header_v2{header.version, header.num_hashes, header.seed, header.num_longs};
    bloom_filter_header_v2 header_swizzled = byte_swap_header_v2(header_v2);
    cudaMemcpyAsync(
      buf.data(), &header_swizzled, bloom_filter_header_v2_size, cudaMemcpyHostToDevice, stream);
  } else {
    // V1: Pack 12-byte header
    bloom_filter_header_v1 header_v1{header.version, header.num_hashes, header.num_longs};
    bloom_filter_header_v1 header_swizzled = byte_swap_header_v1(header_v1);
    cudaMemcpyAsync(
      buf.data(), &header_swizzled, bloom_filter_header_v1_size, cudaMemcpyHostToDevice, stream);
  }
}

/*
  Get the header size based on version number.
*/
int get_header_size(int version)
{
  return (version == 2) ? bloom_filter_header_v2_size : bloom_filter_header_v1_size;
}

/*
  Unpack bloom filter information from a bloom filter buffer. returns the header, a span
  representing the bloom filter bits and the number of bloom filter bits.
  Supports both V1 (12-byte header) and V2 (16-byte header) formats.
*/
std::tuple<bloom_filter_header, cudf::device_span<cudf::bitmask_type>, int64_t> unpack_bloom_filter(
  cudf::device_span<uint8_t> bloom_filter, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(bloom_filter.size() >= bloom_filter_header_v1_size,
               "Encountered truncated bloom filter");

  // First, read just the version to determine header size
  int32_t version_swizzled;
  cudaMemcpyAsync(&version_swizzled,
                  bloom_filter.data(),
                  sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream);
  stream.synchronize();
  
  int version = static_cast<int>(bswap_32(static_cast<uint32_t>(version_swizzled)));
  CUDF_EXPECTS(version == 1 || version == 2, "Unexpected bloom filter version");

  bloom_filter_header header;
  int header_size;
  
  if (version == 2) {
    // V2: Read 16-byte header
    CUDF_EXPECTS(bloom_filter.size() >= bloom_filter_header_v2_size,
                 "Encountered truncated bloom filter (V2)");
    
    bloom_filter_header_v2 header_v2_swizzled;
    cudaMemcpyAsync(&header_v2_swizzled,
                    bloom_filter.data(),
                    bloom_filter_header_v2_size,
                    cudaMemcpyDeviceToHost,
                    stream);
    stream.synchronize();
    
    bloom_filter_header_v2 header_v2 = byte_swap_header_v2(header_v2_swizzled);
    header = {header_v2.version, header_v2.num_hashes, header_v2.seed, header_v2.num_longs};
    header_size = bloom_filter_header_v2_size;
  } else {
    // V1: Read 12-byte header
    bloom_filter_header_v1 header_v1_swizzled;
    cudaMemcpyAsync(&header_v1_swizzled,
                    bloom_filter.data(),
                    bloom_filter_header_v1_size,
                    cudaMemcpyDeviceToHost,
                    stream);
    stream.synchronize();
    
    bloom_filter_header_v1 header_v1 = byte_swap_header_v1(header_v1_swizzled);
    header = {header_v1.version, header_v1.num_hashes, 0, header_v1.num_longs};
    header_size = bloom_filter_header_v1_size;
  }

  int64_t const bloom_filter_bits = static_cast<int64_t>(header.num_longs) * 64;
  auto const num_bitmask_words = static_cast<size_t>(header.num_longs) * 2;

  CUDF_EXPECTS(bloom_filter_bits > 0, "Invalid empty bloom filter size");
  CUDF_EXPECTS(num_bitmask_words == cudf::num_bitmask_words(bloom_filter_bits),
               "Bloom filter bit/length mismatch");

  return {header,
          {reinterpret_cast<cudf::bitmask_type*>(bloom_filter.data() + header_size),
           num_bitmask_words},
          bloom_filter_bits};
}

/*
  Unpack bloom filter information a from column_view that wraps a single bloom filter buffer.
  returns the header, a span representing the bloom filter bits and the number of bloom filter bits.
*/
std::tuple<bloom_filter_header, cudf::device_span<cudf::bitmask_type>, int64_t> unpack_bloom_filter(
  cudf::column_view const& bloom_filter, rmm::cuda_stream_view stream)
{
  // the const_cast is necessary because list_scalar does not provide a mutable_view() function.
  return unpack_bloom_filter(
    cudf::device_span<uint8_t>{const_cast<uint8_t*>(bloom_filter.data<uint8_t>()),
                               static_cast<size_t>(bloom_filter.size())},
    stream);
}

// V1 header comparison (12 bytes) - used in merge validation
struct bloom_filter_same_v1 {
  bloom_filter_header_v1 header;
  cudf::detail::lists_column_device_view ldv;
  cudf::size_type stride;

  bool __device__ operator()(cudf::size_type i)
  {
    bloom_filter_header_v1 const* a =
      reinterpret_cast<bloom_filter_header_v1 const*>(ldv.child().data<uint8_t>() + stride * i);
    return (a->version == header.version) && (a->num_hashes == header.num_hashes) &&
           (a->num_longs == header.num_longs);
  }
};

// V2 header comparison (16 bytes) - used in merge validation
struct bloom_filter_same_v2 {
  bloom_filter_header_v2 header;
  cudf::detail::lists_column_device_view ldv;
  cudf::size_type stride;

  bool __device__ operator()(cudf::size_type i)
  {
    bloom_filter_header_v2 const* a =
      reinterpret_cast<bloom_filter_header_v2 const*>(ldv.child().data<uint8_t>() + stride * i);
    return (a->version == header.version) && (a->num_hashes == header.num_hashes) &&
           (a->seed == header.seed) && (a->num_longs == header.num_longs);
  }
};

/*
  Returns a pair indicating:
  - size of the bloom filter bits
  - total size of the bloom filter buffer (header + bits)
*/
std::pair<int, int> get_bloom_filter_stride(int bloom_filter_longs, int version)
{
  auto const bloom_filter_size = (bloom_filter_longs * sizeof(int64_t));
  auto const header_size = (version == 2) ? bloom_filter_header_v2_size : bloom_filter_header_v1_size;
  auto const buf_size = header_size + bloom_filter_size;
  return {bloom_filter_size, buf_size};
}

}  // anonymous namespace

/*
  Creates a new bloom filter.  The bloom filter is stored using a cudf list_scalar with a specific
  structure.
  - The data type is int8, representing a generic buffer
  - The first 12 bytes (V1) or 16 bytes (V2) of the buffer are a bloom_filter_header
  - The remaining bytes are the bloom filter buffer itself. The length of the remaining bytes must
  be bloom_filter_header.num_longs * 8
  - All of the data in the buffer is stored in big-endian format.  unpack_bloom_filter() unpacks
  this into a usable form, and pack_bloom_filter_header packs new data into the output (big-endian)
  form.
*/
std::unique_ptr<cudf::list_scalar> bloom_filter_create(int num_hashes,
                                                       int bloom_filter_longs,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  // Default to V1 for backwards compatibility
  return bloom_filter_create(num_hashes, bloom_filter_longs, 1, stream, mr);
}

/*
  Creates a new bloom filter with specified version.
*/
std::unique_ptr<cudf::list_scalar> bloom_filter_create(int num_hashes,
                                                       int bloom_filter_longs,
                                                       int version,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(version == 1 || version == 2, "Bloom filter version must be 1 or 2");
  
  auto [bloom_filter_size, buf_size] = get_bloom_filter_stride(bloom_filter_longs, version);
  auto const header_size = (version == 2) ? bloom_filter_header_v2_size : bloom_filter_header_v1_size;

  // build the packed bloom filter buffer ------------------
  rmm::device_buffer buf{static_cast<size_t>(buf_size), stream, mr};

  // pack the header
  int bloom_version = (version == 2) ? spark_bloom_filter_version_v2 : spark_bloom_filter_version_v1;
  bloom_filter_header header{bloom_version, num_hashes, 0, bloom_filter_longs};
  pack_bloom_filter_header(
    {reinterpret_cast<uint8_t*>(buf.data()), static_cast<size_t>(buf_size)}, header, version, stream);
  // memset the bloom filter bits to 0.

  CUDF_CUDA_TRY(cudaMemsetAsync(reinterpret_cast<uint8_t*>(buf.data()) + header_size,
                                0,
                                bloom_filter_size,
                                stream));

  // create the 1-row list column and move it into a scalar.
  return std::make_unique<cudf::list_scalar>(
    cudf::column(
      cudf::data_type{cudf::type_id::UINT8}, buf_size, std::move(buf), rmm::device_buffer{}, 0),
    true,
    stream,
    mr);
}

void bloom_filter_put(cudf::list_scalar& bloom_filter,
                      cudf::column_view const& input,
                      rmm::cuda_stream_view stream)
{
  // unpack the bloom filter
  auto [header, buffer, bloom_filter_bits] = unpack_bloom_filter(bloom_filter.view(), stream);
  auto const header_size = get_header_size(header.version);
  CUDF_EXPECTS(bloom_filter.view().size() == (buffer.size() * 4) + header_size,
               "Encountered invalid/mismatched bloom filter buffer data");

  constexpr int block_size = 256;
  auto grid                = cudf::detail::grid_1d{input.size(), block_size, 1};
  auto d_input             = cudf::column_device_view::create(input);

  if (header.version == 2) {
    // Use V2 kernel
    if (input.has_nulls()) {
      gpu_bloom_filter_put<true, 2><<<grid.num_blocks, block_size, 0, stream.value()>>>(
        buffer.data(), bloom_filter_bits, *d_input, header.num_hashes);
    } else {
      gpu_bloom_filter_put<false, 2><<<grid.num_blocks, block_size, 0, stream.value()>>>(
        buffer.data(), bloom_filter_bits, *d_input, header.num_hashes);
    }
  } else {
    // Use V1 kernel
    if (input.has_nulls()) {
      gpu_bloom_filter_put<true, 1><<<grid.num_blocks, block_size, 0, stream.value()>>>(
        buffer.data(), bloom_filter_bits, *d_input, header.num_hashes);
    } else {
      gpu_bloom_filter_put<false, 1><<<grid.num_blocks, block_size, 0, stream.value()>>>(
        buffer.data(), bloom_filter_bits, *d_input, header.num_hashes);
    }
  }
}

std::unique_ptr<cudf::list_scalar> bloom_filter_merge(cudf::column_view const& bloom_filters,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  // unpack the bloom filter
  cudf::lists_column_view lcv(bloom_filters);

  // since the list child column is just a bunch of packed bloom filter buffers one after another,
  // we can just pass the base data pointer to unpack the first one.
  auto [header, buffer, bloom_filter_bits] = unpack_bloom_filter(lcv.child(), stream);
  auto const header_size = get_header_size(header.version);
  
  // NOTE: since this is a column containing multiple bloom filters, the expected total size is the
  // size for one bloom filter times the number of rows (bloom_filters.size())
  CUDF_EXPECTS(
    lcv.child().size() == ((buffer.size() * 4) + header_size) * bloom_filters.size(),
    "Encountered invalid/mismatched bloom filter buffer data");

  auto [bloom_filter_size, buf_size] = get_bloom_filter_stride(header.num_longs, header.version);

  // validate all the bloom filters are the same
  auto dv = cudf::column_device_view::create(bloom_filters);
  
  if (header.version == 2) {
    bloom_filter_header_v2 header_v2{header.version, header.num_hashes, header.seed, header.num_longs};
    bloom_filter_header_v2 header_v2_swizzled = byte_swap_header_v2(header_v2);
    CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(cudf::get_default_stream()),
                                thrust::make_counting_iterator(1),
                                thrust::make_counting_iterator(bloom_filters.size()),
                                bloom_filter_same_v2{header_v2_swizzled, *dv, buf_size}),
                 "Mismatch of bloom filter parameters");
  } else {
    bloom_filter_header_v1 header_v1{header.version, header.num_hashes, header.num_longs};
    bloom_filter_header_v1 header_v1_swizzled = byte_swap_header_v1(header_v1);
    CUDF_EXPECTS(thrust::all_of(rmm::exec_policy(cudf::get_default_stream()),
                                thrust::make_counting_iterator(1),
                                thrust::make_counting_iterator(bloom_filters.size()),
                                bloom_filter_same_v1{header_v1_swizzled, *dv, buf_size}),
                 "Mismatch of bloom filter parameters");
  }

  // build the packed bloom filter buffer ------------------
  rmm::device_buffer buf{static_cast<size_t>(buf_size), stream, mr};
  pack_bloom_filter_header(
    {reinterpret_cast<uint8_t*>(buf.data()), static_cast<size_t>(buf_size)}, header, header.version, stream);

  auto src = lcv.child().data<uint8_t>() + header_size;
  auto dst = reinterpret_cast<cudf::bitmask_type*>(reinterpret_cast<uint8_t*>(buf.data()) +
                                                   header_size);

  // bitwise-or all the bloom filters together
  cudf::size_type num_words = header.num_longs * 2;
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + num_words,
    dst,
    cuda::proclaim_return_type<cudf::bitmask_type>(
      [src, num_buffers = bloom_filters.size(), stride = buf_size] __device__(
        cudf::size_type word_index) {
        cudf::bitmask_type out = (reinterpret_cast<cudf::bitmask_type const*>(src))[word_index];
        for (auto idx = 1; idx < num_buffers; idx++) {
          out |= (reinterpret_cast<cudf::bitmask_type const*>(src + idx * stride))[word_index];
        }
        return out;
      }));

  // create the 1-row list column and move it into a scalar.
  return std::make_unique<cudf::list_scalar>(
    cudf::column(
      cudf::data_type{cudf::type_id::UINT8}, buf_size, std::move(buf), rmm::device_buffer{}, 0),
    true,
    stream,
    mr);
}

std::unique_ptr<cudf::column> bloom_filter_probe(cudf::column_view const& input,
                                                 cudf::device_span<uint8_t const> bloom_filter,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  // unpack the bloom filter (need non-const span for unpack)
  auto [header, buffer, bloom_filter_bits] = unpack_bloom_filter(
    cudf::device_span<uint8_t>{const_cast<uint8_t*>(bloom_filter.data()), bloom_filter.size()}, 
    stream);
  auto const header_size = get_header_size(header.version);
  CUDF_EXPECTS(bloom_filter.size() == (buffer.size() * 4) + header_size,
               "Encountered invalid/mismatched bloom filter buffer data");

  // duplicate input mask
  auto out = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                           input.size(),
                                           cudf::copy_bitmask(input),
                                           input.null_count(),
                                           stream,
                                           mr);

  if (header.version == 2) {
    thrust::transform(
      rmm::exec_policy(stream),
      input.begin<int64_t>(),
      input.end<int64_t>(),
      out->mutable_view().begin<bool>(),
      bloom_probe_functor<2>{buffer.data(), bloom_filter_bits, header.num_hashes});
  } else {
    thrust::transform(
      rmm::exec_policy(stream),
      input.begin<int64_t>(),
      input.end<int64_t>(),
      out->mutable_view().begin<bool>(),
      bloom_probe_functor<1>{buffer.data(), bloom_filter_bits, header.num_hashes});
  }

  return out;
}

std::unique_ptr<cudf::column> bloom_filter_probe(cudf::column_view const& input,
                                                 cudf::list_scalar& bloom_filter,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto view = bloom_filter.view();
  return bloom_filter_probe(input, 
    cudf::device_span<uint8_t const>{view.data<uint8_t>(), static_cast<size_t>(view.size())},
    stream, mr);
}

}  // namespace spark_rapids_jni
