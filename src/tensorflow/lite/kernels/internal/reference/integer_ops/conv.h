/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include "perf.h"
#include "cfu.h"
#include <stdio.h>

#include <cstdio>
#include <cstring>
#include <cstdint>


#include <stdint.h>
#include <string.h>
#include "playground_util/print_params.h"

namespace tflite {
namespace reference_integer_ops {

///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef CONV_PROFILE
#define CONV_PROFILE 0
#endif

#if CONV_PROFILE
#define CONV_CNT_ADD(var, val) ((var) += (val))
#else
#define CONV_CNT_ADD(var, val) ((void)0)
#endif

// ------------------------------------------------------------
// Hardware Opcode Mapping
// ------------------------------------------------------------
#define OP_SET_OFFSET    1
#define OP_SET_K_N       2
#define OP_WRITE_A0      3
#define OP_WRITE_B_CAST  4
#define OP_READ_C0       5
#define OP_SET_M         7
#define OP_START_RUN     10
#define OP_WRITE_A1      13
#define OP_READ_C1       15
#define OP_WRITE_A2      20
#define OP_READ_C2       22
#define OP_WRITE_A3      25
#define OP_READ_C3       27

// ------------------------------------------------------------
// Fast Packing Helper
// ------------------------------------------------------------
static inline uint32_t pack4_u8(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3) {
  return ((uint32_t)a0 << 24) | ((uint32_t)a1 << 16) |
         ((uint32_t)a2 << 8)  | (uint32_t)a3;
}

// ------------------------------------------------------------
// Static Weight Caching & Helpers
// ------------------------------------------------------------
namespace {

#define MAX_WEIGHT_WORDS (4000 * 1000)
// [Modified] Removed MAX_K_BUFFER_SIZE and g_input_tile_buffer for Direct Write optimization
#define MAX_K_BLOCK_SCRATCH_WORDS (1 << 18)  // 262144 int32 slots (~1MB)

static uint32_t g_weight_cache[MAX_WEIGHT_WORDS];
static const int8_t* g_cached_filter_ptr = nullptr;
static int g_cached_count = 0;
static int g_conv_call_counter = 0;
static int32_t g_k_block_scratch[MAX_K_BLOCK_SCRATCH_WORDS];

inline int CeilDiv(int a, int b) { return (a + b - 1) / b; }

inline int CalculateWordsPerGroup(int FH, int FW, int IC) {
  int words = 0;
  int ic_temp = 0;
  for (; ic_temp <= IC - 4; ic_temp += 4) words += 4;
  for (; ic_temp < IC; ++ic_temp) words += 1;
  return words * (FH * FW);
}

inline void PackGroupLogic(const int8_t* filter_data, int FH, int FW, int IC,
                           int out_ch, int filter_ch_stride,
                           uint32_t* target_buffer) {
  const int8_t* w0 = filter_data + out_ch * filter_ch_stride;
  const int8_t* w1 = w0 + filter_ch_stride;
  const int8_t* w2 = w0 + 2 * filter_ch_stride;
  const int8_t* w3 = w0 + 3 * filter_ch_stride;
  int k_idx_cfu = 0;

  for (int fy = 0; fy < FH; ++fy) {
    for (int fx = 0; fx < FW; ++fx) {
      int ic = 0;
      for (; ic <= IC - 4; ic += 4) {
        uint32_t v0 = pack4_u8(w0[0], w1[0], w2[0], w3[0]);
        uint32_t v1 = pack4_u8(w0[1], w1[1], w2[1], w3[1]);
        uint32_t v2 = pack4_u8(w0[2], w1[2], w2[2], w3[2]);
        uint32_t v3 = pack4_u8(w0[3], w1[3], w2[3], w3[3]);
        if (target_buffer) {
          *target_buffer++ = v0; *target_buffer++ = v1;
          *target_buffer++ = v2; *target_buffer++ = v3;
        } else {
          cfu_op0(OP_WRITE_B_CAST, k_idx_cfu++, v0);
          cfu_op0(OP_WRITE_B_CAST, k_idx_cfu++, v1);
          cfu_op0(OP_WRITE_B_CAST, k_idx_cfu++, v2);
          cfu_op0(OP_WRITE_B_CAST, k_idx_cfu++, v3);
        }
        w0 += 4; w1 += 4; w2 += 4; w3 += 4;
      }
      for (; ic < IC; ++ic) {
        uint32_t v = pack4_u8(w0[0], w1[0], w2[0], w3[0]);
        if (target_buffer) *target_buffer++ = v;
        else cfu_op0(OP_WRITE_B_CAST, k_idx_cfu++, v);
        w0++; w1++; w2++; w3++;
      }
    }
  }
}

inline void PackWeightsIntoStaticCache(const int8_t* filter_data, int FH, int FW, int IC, int OC) {
  g_cached_filter_ptr = filter_data;
  int filter_ch_stride = FH * FW * IC;
  uint32_t* p_cache = g_weight_cache;
  for (int out_ch = 0; out_ch < OC; out_ch += 4) {
    PackGroupLogic(filter_data, FH, FW, IC, out_ch, filter_ch_stride, p_cache);
    p_cache += CalculateWordsPerGroup(FH, FW, IC);
  }
  g_cached_count = (int)(p_cache - g_weight_cache);
}

inline void PackGroupRangeLogic(const int8_t* filter_data, int FH, int FW, int IC,
                                int out_ch, int filter_ch_stride,
                                int k_start, int k_count) {
  // Stream a partial [k_start, k_start + k_count) window of weights directly to CFU.
  const int8_t* w0 = filter_data + out_ch * filter_ch_stride;
  const int8_t* w1 = w0 + filter_ch_stride;
  const int8_t* w2 = w0 + 2 * filter_ch_stride;
  const int8_t* w3 = w0 + 3 * filter_ch_stride;

  const int start_spatial = k_start / IC;
  const int start_ic = k_start % IC;
  const int end_idx = k_start + k_count - 1;
  const int end_spatial = end_idx / IC;
  const int end_ic = end_idx % IC;

  int k_idx_cfu = 0;
  for (int spatial = start_spatial; spatial <= end_spatial; ++spatial) {
    const int fy = spatial / FW;
    const int fx = spatial % FW;
    const int ic_begin = (spatial == start_spatial) ? start_ic : 0;
    const int ic_end_exclusive = (spatial == end_spatial) ? (end_ic + 1) : IC;

    const int base_offset = (fy * FW + fx) * IC;
    const int8_t* ptr0 = w0 + base_offset + ic_begin;
    const int8_t* ptr1 = w1 + base_offset + ic_begin;
    const int8_t* ptr2 = w2 + base_offset + ic_begin;
    const int8_t* ptr3 = w3 + base_offset + ic_begin;

    int remaining = ic_end_exclusive - ic_begin;
    while (remaining >= 4) {
      cfu_op0(OP_WRITE_B_CAST, k_idx_cfu++, pack4_u8(ptr0[0], ptr1[0], ptr2[0], ptr3[0]));
      cfu_op0(OP_WRITE_B_CAST, k_idx_cfu++, pack4_u8(ptr0[1], ptr1[1], ptr2[1], ptr3[1]));
      cfu_op0(OP_WRITE_B_CAST, k_idx_cfu++, pack4_u8(ptr0[2], ptr1[2], ptr2[2], ptr3[2]));
      cfu_op0(OP_WRITE_B_CAST, k_idx_cfu++, pack4_u8(ptr0[3], ptr1[3], ptr2[3], ptr3[3]));
      ptr0 += 4; ptr1 += 4; ptr2 += 4; ptr3 += 4;
      remaining -= 4;
    }
    while (remaining > 0) {
      cfu_op0(OP_WRITE_B_CAST, k_idx_cfu++, pack4_u8(ptr0[0], ptr1[0], ptr2[0], ptr3[0]));
      ptr0++; ptr1++; ptr2++; ptr3++;
      remaining--;
    }
  }
}

} // namespace

// ------------------------------------------------------------
// Main Conv Function
// ------------------------------------------------------------
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {


#if CONV_PROFILE
  perf_enable_counter(6);
  // Counters
  uint64_t cnt_pack_A = 0;
  uint64_t cnt_pack_B = 0;
  uint64_t cnt_run    = 0;
  uint64_t cnt_read   = 0;
#endif

  // Params
  const int32_t input_offset  = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int stride_w   = params.stride_width;
  const int stride_h   = params.stride_height;
  const int pad_w      = params.padding_values.width;
  const int pad_h      = params.padding_values.height;
  const int dilation_w = params.dilation_width_factor;
  const int dilation_h = params.dilation_height_factor;
  const int32_t act_min = params.quantized_activation_min;
  const int32_t act_max = params.quantized_activation_max;

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int IH = input_shape.Dims(1);
  const int IW = input_shape.Dims(2);
  const int IC = input_shape.Dims(3);
  const int FH = filter_shape.Dims(1);
  const int FW = filter_shape.Dims(2);
  const int OH = output_shape.Dims(1);
  const int OW = output_shape.Dims(2);
  const int OC = MatchingDim(filter_shape, 0, output_shape, 3);
  const bool has_bias = (bias_data != nullptr);

  const int K = FH * FW * IC;
  const int M_total_pixels = OH * OW;
  const int in_row_stride   = IW * IC;
  const int out_row_stride  = OW * OC;

  // HW Config
  const int HW_BUFFER_DEPTH  = 8192;
  const int OUT_BUFFER_DEPTH = 1024;

  const int base_tiles_per_core = std::max(1, HW_BUFFER_DEPTH / K);
  const int base_max_rows_per_core =
      std::min(base_tiles_per_core * 4, OUT_BUFFER_DEPTH);
  const int base_quad_core_step = base_max_rows_per_core * 4;
  const int spatial_size = FH * FW;
  const int oc_groups = (OC + 3) / 4;

  int best_block_spatial = spatial_size;
  int best_block_size = K;
  int best_block_count = 1;
  int best_max_rows_per_core = base_max_rows_per_core;
  int best_quad_core_step = base_quad_core_step;
  const uint64_t base_cost =
      (uint64_t)CeilDiv(M_total_pixels, base_quad_core_step) *
      (uint64_t)oc_groups;
  uint64_t best_cost = base_cost;

  if (K > 0 && spatial_size > 0) {
    const int candidate_blocks[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
    const int k_block_tie_min = 3072;  // only take equal-cost ties for large K
    for (int i = 0; i < (int)(sizeof(candidate_blocks) / sizeof(int)); ++i) {
      int block_spatial = candidate_blocks[i];
      if (block_spatial >= spatial_size) continue;
      int block_size = block_spatial * IC;
      if (block_size <= 0 || block_size >= K) continue;

      int cand_tiles = std::max(1, HW_BUFFER_DEPTH / block_size);
      int cand_rows = std::min(cand_tiles * 4, OUT_BUFFER_DEPTH);
      int cand_quad = cand_rows * 4;
      int block_count = CeilDiv(spatial_size, block_spatial);
      uint64_t cand_cost =
          (uint64_t)CeilDiv(M_total_pixels, cand_quad) *
          (uint64_t)block_count * (uint64_t)oc_groups;
      bool better_cost = cand_cost < best_cost;
      bool better_tie = (cand_cost == best_cost) &&
                        (cand_rows > best_max_rows_per_core) &&
                        (K >= k_block_tie_min);
      if (better_cost || better_tie) {
        best_cost = cand_cost;
        best_block_spatial = block_spatial;
        best_block_size = block_size;
        best_block_count = block_count;
        best_max_rows_per_core = cand_rows;
        best_quad_core_step = cand_quad;
      }
    }
  }

  bool tile_too_skinny = base_max_rows_per_core <= 16;
  bool cost_better = best_cost + best_block_count < base_cost + 1;
  bool use_k_blocking = (best_block_size < K) && (tile_too_skinny || cost_better);

  if (!use_k_blocking) {
    best_block_spatial = spatial_size;
    best_block_size = K;
    best_block_count = 1;
    best_max_rows_per_core = base_max_rows_per_core;
    best_quad_core_step = base_quad_core_step;
    best_cost = base_cost;
  }

  // 強制對 K=8000 做更細的 blocking：切成約 8 份 (每份 ~1000)
  if (K == 8000) {
    use_k_blocking = true;
    best_block_size = 1000;
    best_block_spatial = std::max(1, best_block_size / IC);
    best_block_count = CeilDiv(spatial_size, best_block_spatial);
    int cand_tiles = std::max(1, HW_BUFFER_DEPTH / best_block_size);
    best_max_rows_per_core = std::min(cand_tiles * 4, OUT_BUFFER_DEPTH);
    best_quad_core_step = best_max_rows_per_core * 4;
  }

  const size_t worst_case_scratch =
      (size_t)best_quad_core_step * (size_t)OC;
  if (use_k_blocking && worst_case_scratch > MAX_K_BLOCK_SCRATCH_WORDS) {
    use_k_blocking = false;
    best_block_spatial = spatial_size;
    best_block_size = K;
    best_block_count = 1;
    best_max_rows_per_core = base_max_rows_per_core;
    best_quad_core_step = base_quad_core_step;
  }

  int max_rows_per_core = best_max_rows_per_core;
  const int QUAD_CORE_STEP = best_quad_core_step;

  int conv_call_id = ++g_conv_call_counter;
  printf("[CONV] call=%d use_k_blocking=%d K=%d best_block=%d base_rows=%d quad=%d M_total=%d OC=%d IH=%d IW=%d FH=%d FW=%d stride=(%d,%d) pad=(%d,%d) act=(%ld,%ld) in_off=%ld out_off=%ld\n",
         conv_call_id, (int)use_k_blocking, K, best_block_size, base_max_rows_per_core,
         QUAD_CORE_STEP, M_total_pixels, OC, IH, IW, FH, FW, stride_h, stride_w, pad_h, pad_w,
         (long)act_min, (long)act_max, (long)input_offset, (long)output_offset);

  // Cache Logic
  int words_per_group = CalculateWordsPerGroup(FH, FW, IC);
  const int words_per_spatial = (spatial_size > 0) ? (words_per_group / spatial_size) : 0;
  int num_groups = (OC + 3) / 4;
  int total_needed_words = num_groups * words_per_group;
  bool use_cache = (total_needed_words <= MAX_WEIGHT_WORDS);

  if (use_cache && filter_data != g_cached_filter_ptr) {
    PackWeightsIntoStaticCache(filter_data, FH, FW, IC, OC);
  } else if (!use_cache) {
    g_cached_filter_ptr = nullptr;
  }
  const uint32_t* packed_weights_ptr = g_weight_cache;
  int filter_ch_stride = FH * FW * IC;

  // Buffer Init
  const int MAX_PAD_BUFFER = 4096;
  static int8_t dummy_pad_buffer[MAX_PAD_BUFFER];
  static int pad_fill_cached = 0x7fffffff;
  const int8_t pad_fill = static_cast<int8_t>(-input_offset);
  if (pad_fill_cached != pad_fill) {
    memset(dummy_pad_buffer, pad_fill, MAX_PAD_BUFFER);
    pad_fill_cached = pad_fill;
  }

  cfu_op0(OP_SET_OFFSET, input_offset, 0);
  const int block_count = CeilDiv(spatial_size, best_block_spatial);

  // ------------------------------------------------------------
  // Main Loops
  // ------------------------------------------------------------
  for (int batch = 0; batch < batches; ++batch) {
    const int8_t* in_batch_base = input_data + batch * IH * in_row_stride;
    int8_t* out_batch_base = output_data + batch * OH * out_row_stride;

    for (int slide_base = 0; slide_base < M_total_pixels; slide_base += QUAD_CORE_STEP) {

      int M_remain = M_total_pixels - slide_base;
      int M_curr_step = (QUAD_CORE_STEP < M_remain) ? QUAD_CORE_STEP : M_remain;

      int m_temp = M_curr_step;
      int M_c0 = (max_rows_per_core < m_temp) ? max_rows_per_core : m_temp; m_temp -= M_c0;
      int M_c1 = (max_rows_per_core < m_temp) ? max_rows_per_core : m_temp; m_temp -= M_c1;
      int M_c2 = (max_rows_per_core < m_temp) ? max_rows_per_core : m_temp; m_temp -= M_c2;
      int M_c3 = m_temp;

      // =================================================================
      // Step A: Input Packing (OPTIMIZED with Code B Strategy & Direct Write)
      // =================================================================
      auto pack_core_input = [&](int core_pixel_offset, int core_m_count,
                                 int op_write_addr, int k_range_start,
                                 int k_range_size) {
        if (core_m_count <= 0 || k_range_size <= 0) return;

        int tiles_count = (core_m_count + 3) >> 2;
        int start_m_abs = slide_base + core_pixel_offset;
        const int start_spatial = k_range_start / IC;
        const int start_ic = k_range_start % IC;
        const int end_idx = k_range_start + k_range_size - 1;
        const int end_spatial = end_idx / IC;
        const int end_ic = end_idx % IC;

        for (int t = 0; t < tiles_count; ++t) {
          int tile_start_m = start_m_abs + t * 4;
          int iy[4], ix[4];
          bool any_padding_needed = false;

          int cur_oy = tile_start_m / OW;
          int cur_ox = tile_start_m % OW;

          for (int s = 0; s < 4; ++s) {
            if (t * 4 + s < core_m_count) {
              iy[s] = cur_oy * stride_h - pad_h;
              ix[s] = cur_ox * stride_w - pad_w;
              int y_end = iy[s] + (FH - 1) * dilation_h;
              int x_end = ix[s] + (FW - 1) * dilation_w;
              if (iy[s] < 0 || ix[s] < 0 || y_end >= IH || x_end >= IW) {
                any_padding_needed = true;
              }
              cur_ox++;
              if (cur_ox == OW) { cur_ox = 0; cur_oy++; }
            } else {
              iy[s] = -9999;
              ix[s] = -9999;
              any_padding_needed = true;
            }
          }

          int k_addr = t * k_range_size;
          auto write_to_cfu = [&](uint32_t val) {
            if      (op_write_addr == OP_WRITE_A0) { cfu_op0(OP_WRITE_A0, k_addr++, val); }
            else if (op_write_addr == OP_WRITE_A1) { cfu_op0(OP_WRITE_A1, k_addr++, val); }
            else if (op_write_addr == OP_WRITE_A2) { cfu_op0(OP_WRITE_A2, k_addr++, val); }
            else                                   { cfu_op0(OP_WRITE_A3, k_addr++, val); }
          };

          for (int spatial = start_spatial; spatial <= end_spatial; ++spatial) {
            const int fy = spatial / FW;
            const int fx = spatial % FW;
            const int ic_begin = (spatial == start_spatial) ? start_ic : 0;
            const int ic_end_exclusive = (spatial == end_spatial) ? (end_ic + 1) : IC;

            const int8_t* ptrs[4];
            for (int s = 0; s < 4; ++s) {
              if (iy[s] == -9999) {
                ptrs[s] = dummy_pad_buffer;
                continue;
              }
              int y = iy[s] + fy * dilation_h;
              int x = ix[s] + fx * dilation_w;
              if (!any_padding_needed) {
                ptrs[s] = in_batch_base + y * in_row_stride + x * IC;
              } else {
                if (y < 0 || y >= IH || x < 0 || x >= IW) ptrs[s] = dummy_pad_buffer;
                else ptrs[s] = in_batch_base + y * in_row_stride + x * IC;
              }
            }

            const int8_t* p0 = ptrs[0] + ic_begin;
            const int8_t* p1 = ptrs[1] + ic_begin;
            const int8_t* p2 = ptrs[2] + ic_begin;
            const int8_t* p3 = ptrs[3] + ic_begin;
            int remaining = ic_end_exclusive - ic_begin;
            while (remaining >= 4) {
              write_to_cfu(pack4_u8(p0[0], p1[0], p2[0], p3[0]));
              write_to_cfu(pack4_u8(p0[1], p1[1], p2[1], p3[1]));
              write_to_cfu(pack4_u8(p0[2], p1[2], p2[2], p3[2]));
              write_to_cfu(pack4_u8(p0[3], p1[3], p2[3], p3[3]));
              p0 += 4; p1 += 4; p2 += 4; p3 += 4;
              remaining -= 4;
            }
            while (remaining > 0) {
              write_to_cfu(pack4_u8(p0[0], p1[0], p2[0], p3[0]));
              p0++; p1++; p2++; p3++;
              remaining--;
            }
          }

          CONV_CNT_ADD(cnt_pack_A, (uint64_t)k_range_size);
        }
      };

      size_t scratch_elems = 0;
      bool use_k_blocking_this_slide = use_k_blocking;
      if (use_k_blocking_this_slide) {
        // Accumulate per-tile partial sums across K-blocks before finalizing.
        scratch_elems = (size_t)M_curr_step * (size_t)OC;
        if (scratch_elems > MAX_K_BLOCK_SCRATCH_WORDS) {
          use_k_blocking_this_slide = false;
        } else {
          std::fill(g_k_block_scratch, g_k_block_scratch + scratch_elems, 0);
        }
      }

      int block_count_loop = block_count;
      int block_spatial_loop = best_block_spatial;
      if (!use_k_blocking_this_slide) {
        block_count_loop = 1;
        block_spatial_loop = spatial_size;
      }

      // =================================================================
      // Step B: Loop Output Channels with optional K-blocking
      // =================================================================
      for (int block_idx = 0; block_idx < block_count_loop; ++block_idx) {
        const int spatial_start = block_idx * block_spatial_loop;
        int spatial_this_block = block_spatial_loop;
        if (spatial_start + spatial_this_block > spatial_size) {
          spatial_this_block = spatial_size - spatial_start;
        }
        const int k_block_size = spatial_this_block * IC;
        const int k_block_start = spatial_start * IC;

        pack_core_input(0,                  M_c0, OP_WRITE_A0, k_block_start, k_block_size);
        pack_core_input(M_c0,               M_c1, OP_WRITE_A1, k_block_start, k_block_size);
        pack_core_input(M_c0 + M_c1,        M_c2, OP_WRITE_A2, k_block_start, k_block_size);
        pack_core_input(M_c0 + M_c1 + M_c2, M_c3, OP_WRITE_A3, k_block_start, k_block_size);

        for (int out_ch = 0; out_ch < OC; out_ch += 4) {
          cfu_op0(OP_SET_K_N, k_block_size, 4);

          const int block_word_start = (k_block_start / IC) * words_per_spatial;
          const int block_word_size = spatial_this_block * words_per_spatial;

          bool use_block_cache = use_cache;
          if (use_block_cache) {
            int tile_idx = out_ch / 4;
            const uint32_t* tile_w_ptr =
                packed_weights_ptr + (tile_idx * words_per_group) + block_word_start;
            int w_remain = block_word_size;
            int i = 0;
            while (w_remain >= 4) {
              cfu_op0(OP_WRITE_B_CAST, i++, *tile_w_ptr++);
              cfu_op0(OP_WRITE_B_CAST, i++, *tile_w_ptr++);
              cfu_op0(OP_WRITE_B_CAST, i++, *tile_w_ptr++);
              cfu_op0(OP_WRITE_B_CAST, i++, *tile_w_ptr++);
              w_remain -= 4;
              CONV_CNT_ADD(cnt_pack_B, 4);
            }
            while (w_remain > 0) {
              cfu_op0(OP_WRITE_B_CAST, i++, *tile_w_ptr++);
              w_remain--;
              CONV_CNT_ADD(cnt_pack_B, 1);
            }
          } else {
            PackGroupRangeLogic(filter_data, FH, FW, IC, out_ch, filter_ch_stride,
                                k_block_start, k_block_size);
            CONV_CNT_ADD(cnt_pack_B, (uint64_t)block_word_size);
          }

          int run_length = std::max(std::max(M_c0, M_c1), std::max(M_c2, M_c3));
          cfu_op0(OP_SET_M, run_length, 0);
          cfu_op0(OP_START_RUN, 0, 0);
          cfu_op0(OP_SET_K_N, act_min, act_max);
          CONV_CNT_ADD(cnt_run, 1);

          auto read_core_output = [&](int core_pixel_offset, int core_m_count,
                                      int op_read) {
            if (core_m_count <= 0) return;
            const int local_start = core_pixel_offset;
            const int m_start = slide_base + core_pixel_offset;
            const int channels_this_block = std::min(4, OC - out_ch);

            if (!use_k_blocking_this_slide) {
              int8_t* dst_base = out_batch_base + (m_start * OC) + out_ch;
              for (int n = 0; n < channels_this_block; ++n) {
                const int ch_idx = out_ch + n;
                const int32_t bias_val = has_bias ? bias_data[ch_idx] : 0;
                cfu_op0(8, output_multiplier[ch_idx], output_shift[ch_idx]);

                int8_t* dst_ptr = dst_base + n;
                for (int m_local = 0; m_local < core_m_count; ++m_local) {
                  int32_t acc;
                  if      (op_read == OP_READ_C0) { acc = cfu_op0(OP_READ_C0, m_local, 3 - n); }
                  else if (op_read == OP_READ_C1) { acc = cfu_op0(OP_READ_C1, m_local, 3 - n); }
                  else if (op_read == OP_READ_C2) { acc = cfu_op0(OP_READ_C2, m_local, 3 - n); }
                  else                            { acc = cfu_op0(OP_READ_C3, m_local, 3 - n); }
                  CONV_CNT_ADD(cnt_read, 1);
                  acc += bias_val;
                  acc = cfu_op0(9, acc, output_offset);
                  *dst_ptr = (int8_t)acc;
                  dst_ptr += OC;
                }
              }
            } else {
              for (int n = 0; n < channels_this_block; ++n) {
                int ch_idx = out_ch + n;
                for (int m_local = 0; m_local < core_m_count; ++m_local) {
                  int32_t acc;
                  if      (op_read == OP_READ_C0) { acc = cfu_op0(OP_READ_C0, m_local, 3 - n); }
                  else if (op_read == OP_READ_C1) { acc = cfu_op0(OP_READ_C1, m_local, 3 - n); }
                  else if (op_read == OP_READ_C2) { acc = cfu_op0(OP_READ_C2, m_local, 3 - n); }
                  else                            { acc = cfu_op0(OP_READ_C3, m_local, 3 - n); }
                  CONV_CNT_ADD(cnt_read, 1);
                  size_t dst_index = ((size_t)local_start + (size_t)m_local) * (size_t)OC + (size_t)ch_idx;
                  g_k_block_scratch[dst_index] += acc;
                }
              }
            }
          };

          read_core_output(0,                  M_c0, OP_READ_C0);
          read_core_output(M_c0,               M_c1, OP_READ_C1);
          read_core_output(M_c0 + M_c1,        M_c2, OP_READ_C2);
          read_core_output(M_c0 + M_c1 + M_c2, M_c3, OP_READ_C3);
        }
      }

      if (use_k_blocking_this_slide) {
        for (int out_ch = 0; out_ch < OC; out_ch += 4) {
          const int channels_this_block = std::min(4, OC - out_ch);
          int8_t* dst_base = out_batch_base + (slide_base * OC) + out_ch;
          for (int n = 0; n < channels_this_block; ++n) {
            const int ch_idx = out_ch + n;
            const int32_t bias_val = has_bias ? bias_data[ch_idx] : 0;
            cfu_op0(8, output_multiplier[ch_idx], output_shift[ch_idx]);
            for (int m_local = 0; m_local < M_curr_step; ++m_local) {
              size_t idx = (size_t)m_local * (size_t)OC + (size_t)ch_idx;
              int32_t acc = g_k_block_scratch[idx] + bias_val;
              acc = cfu_op0(9, acc, output_offset);
              if (acc < act_min) acc = act_min;
              if (acc > act_max) acc = act_max;
              dst_base[m_local * OC + n] = (int8_t)acc;
            }
          }
        }
      }
    }
  }

#if CONV_PROFILE
  uint64_t total_io = cnt_pack_A + cnt_pack_B + cnt_read;
  printf("\n===== [Conv 4-Core Optimized Summary] =====\n");
  printf("Pack A  : %llu\n", (unsigned long long)cnt_pack_A);
  printf("Pack B  : %llu\n", (unsigned long long)cnt_pack_B);
  printf("TPU Run : %llu\n", (unsigned long long)cnt_run);
  printf("Read    : %llu\n", (unsigned long long)cnt_read);
  if (total_io > 0) {
    printf("Ratio A:B:R = %.1f%% : %.1f%% : %.1f%%\n",
           100.0 * (double)cnt_pack_A / (double)total_io,
           100.0 * (double)cnt_pack_B / (double)total_io,
           100.0 * (double)cnt_read   / (double)total_io);
  }
  printf("===========================================\n");
  perf_disable_counter(6);
#endif
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_





