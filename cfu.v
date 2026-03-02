`include "TPU.v"
`include "global_buffer_bram.v"

module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

  // ==============================================================
  // Decode
  // ==============================================================
  wire [6:0] op    = cmd_payload_function_id[9:3];
  wire [1:0] c_sel = cmd_payload_inputs_1[1:0];

  localparam OP_SET_OFFSET = 7'd1;
  localparam OP_SET_K_N    = 7'd2;
  localparam OP_WRITE_A0   = 7'd3;  
  localparam OP_WRITE_B    = 7'd4;
  localparam OP_READ_C0    = 7'd5;  
  localparam OP_SET_MS2    = 7'd6;
  localparam OP_SET_M      = 7'd7;
  localparam OP_SET_MS     = 7'd8;
  localparam OP_QTIZE_ACC  = 7'd9;
  localparam OP_START      = 7'd10;

  localparam OP_RELU  = 7'd11;
  localparam OP_BIAS_OFFSET = 7'd12;

  localparam OP_WRITE_A1   = 7'd13; 
  localparam OP_READ_C1    = 7'd15; 

  localparam OP_WRITE_A2   = 7'd20; 
  localparam OP_READ_C2    = 7'd22; 
  
  localparam OP_WRITE_A3   = 7'd25; 
  localparam OP_READ_C3    = 7'd27; 

// ==============================================================
  // parameters setting
  // ==============================================================
  reg [31:0] K, M, N;
  reg [31:0] input_offset;
  reg signed [31:0] multiplier, shift;
  reg signed [31:0] m2, s2;

  wire in_valid = (cmd_valid && op == OP_START);
  wire rst_n = (~reset) & (~(cmd_valid && op == OP_SET_OFFSET)); 

  always @(posedge clk) begin
    if (reset) begin
      K <= 32'd0;
      M <= 32'd0;
      N <= 32'd0;
      input_offset <= 32'd0;
      multiplier <= 32'd0;
      shift <= 32'd0;
      m2 <= 32'd0;
      s2 <= 32'd0;
    end else if (cmd_valid) begin
      case (op)
        OP_SET_OFFSET: input_offset <= cmd_payload_inputs_0;
        OP_SET_K_N: begin
          K <= cmd_payload_inputs_0;
          N <= cmd_payload_inputs_1;
        end
        OP_SET_M: begin
          M <= cmd_payload_inputs_0;
        end
        OP_SET_MS: begin
          multiplier <= cmd_payload_inputs_0;
          shift <= cmd_payload_inputs_1;
        end
        OP_SET_MS2: begin
          m2 <= cmd_payload_inputs_0;
          s2 <= cmd_payload_inputs_1;
        end
      endcase
    end
  end

  // ==============================================================
  // quantize
  // ==============================================================
  wire signed [31:0] x_in = cmd_payload_inputs_0; 

  wire [5:0] left_shift  = (shift > 0) ? shift[5:0] : 6'd0;
  wire [5:0] right_shift = (shift > 0) ? 6'd0 : (-shift[5:0]);

  wire signed [63:0] x_64 = $signed(x_in) <<< left_shift;

  wire signed [63:0] prod = x_64 * $signed(multiplier);

  wire signed [63:0] nudge1 = (prod >= 0) ? 64'sh0000000040000000 : 64'shFFFFFFFFC0000001;
  wire signed [63:0] high_mul_tmp = (prod + nudge1) >>> 31;

  wire overflow = (x_64 == 64'shFFFFFFFF80000000) && (multiplier == 32'sh80000000);
  wire signed [31:0] high_mul_res = overflow ? 32'h7FFF_FFFF : high_mul_tmp[31:0];

  wire [5:0] r_shift_minus_1 = (right_shift > 0) ? (right_shift - 6'd1) : 6'd0;
  wire signed [63:0] nudge2 = (right_shift > 0) ? (64'sh1 << r_shift_minus_1) : 64'sh0;

  wire signed [63:0] res_before_shift = $signed({{32{high_mul_res[31]}}, high_mul_res}) + nudge2;
  wire signed [63:0] final_quantized_64 = res_before_shift >>> right_shift;
  wire [31:0] final_quantized_res = final_quantized_64[31:0] + $signed(cmd_payload_inputs_1);

  wire [31:0] final_quantized_res2 = ($signed(final_quantized_res) < $signed(K)) ? K : (($signed(final_quantized_res) > $signed(N)) ? N : final_quantized_res);
  
  wire [31:0] RELU_max = ($signed(K) < $signed(final_quantized_res)) ? final_quantized_res : K;
  wire [31:0] RELU_result = ($signed(RELU_max) < $signed(N)) ? RELU_max : N;

  // ==============================================================
  // quantize-alpha
  // ==============================================================

  wire [5:0] left_shift2  = (s2 > 0) ? s2[5:0] : 6'd0;
  wire [5:0] right_shift2 = (s2 > 0) ? 6'd0 : (-s2[5:0]);

  wire signed [63:0] x_64_m2 = $signed(x_in) <<< left_shift2;

  wire signed [63:0] prod_m2 = x_64_m2 * $signed(m2);

  wire signed [63:0] nudge1_m2 = (prod_m2 >= 0) ? 64'sh0000000040000000 : 64'shFFFFFFFFC0000001;
  wire signed [63:0] high_mul_tmp_m2 = (prod_m2 + nudge1_m2) >>> 31;

  wire overflow_m2 = (x_64_m2 == 64'shFFFFFFFF80000000) && (m2 == 32'sh80000000);
  wire signed [31:0] high_mul_res_m2 = overflow_m2 ? 32'h7FFF_FFFF : high_mul_tmp_m2[31:0];

  wire [5:0] r_shift_minus_1_m2 = (right_shift2 > 0) ? (right_shift2 - 6'd1) : 6'd0;
  wire signed [63:0] nudge2_m2 = (right_shift2 > 0) ? (64'sh1 << r_shift_minus_1_m2) : 64'sh0;

  wire signed [63:0] res_before_shift_m2 = $signed({{32{high_mul_res_m2[31]}}, high_mul_res_m2}) + nudge2_m2;
  wire signed [63:0] final_quantized_64_m2 = res_before_shift_m2 >>> right_shift2;
  wire [31:0] final_quantized_res_m2 = final_quantized_64_m2[31:0] + $signed(cmd_payload_inputs_1);

  wire [31:0] RELU_max_m2 = ($signed(K) < $signed(final_quantized_res_m2)) ? final_quantized_res_m2 : K;
  wire [31:0] RELU_result_m2 = ($signed(RELU_max_m2) < $signed(N)) ? RELU_max_m2 : N;

  // ==============================================================
  // TPU control
  // ==============================================================
  wire busy0, busy1, busy2, busy3;
  wire busy = busy0 | busy1 | busy2 | busy3;

  // TPU0
  wire        A_wr_en0, B_wr_en0, C_wr_en0;
  wire [12:0] A_index0, B_index0, C_index0;
  wire [31:0] A_data_in0, B_data_in0;
  wire [127:0] C_data_in0;
  
  // TPU1
  wire        A_wr_en1, B_wr_en1, C_wr_en1;
  wire [12:0] A_index1, B_index1, C_index1;
  wire [31:0] A_data_in1, B_data_in1;
  wire [127:0] C_data_in1;

  // TPU2
  wire        A_wr_en2, B_wr_en2, C_wr_en2;
  wire [12:0] A_index2, B_index2, C_index2;
  wire [31:0] A_data_in2, B_data_in2;
  wire [127:0] C_data_in2;

  // TPU3
  wire        A_wr_en3, B_wr_en3, C_wr_en3;
  wire [12:0] A_index3, B_index3, C_index3;
  wire [31:0] A_data_in3, B_data_in3;
  wire [127:0] C_data_in3;

  // Buffer Outputs
  wire [31:0]  A0_data_out, B0_data_out;
  wire [31:0]  A1_data_out, B1_data_out;
  wire [31:0]  A2_data_out, B2_data_out;
  wire [31:0]  A3_data_out, B3_data_out;
  
  wire [127:0] C0_data_out, C1_data_out, C2_data_out, C3_data_out;

  // ==============================================================
  // A Buffer
  // ==============================================================
  
  // A0
  reg A0_wr_en_mux;
  reg [12:0] A0_index_mux;
  reg [31:0] A0_data_in_mux;
  always @(*) begin
    if (busy) begin 
      A0_wr_en_mux = A_wr_en0;
      A0_index_mux = A_index0;
      A0_data_in_mux = A_data_in0;
    end 
    else begin 
      A0_wr_en_mux = (cmd_valid && op == OP_WRITE_A0);
      A0_index_mux = cmd_payload_inputs_0[12:0];
      A0_data_in_mux = cmd_payload_inputs_1;
    end
  end

  // A1
  reg A1_wr_en_mux;
  reg [12:0] A1_index_mux;
  reg [31:0] A1_data_in_mux;
  always @(*) begin
    if (busy) begin 
      A1_wr_en_mux = A_wr_en1;
      A1_index_mux = A_index1;
      A1_data_in_mux = A_data_in1;
    end 
    else begin 
      A1_wr_en_mux = (cmd_valid && op == OP_WRITE_A1);
      A1_index_mux = cmd_payload_inputs_0[12:0];
      A1_data_in_mux = cmd_payload_inputs_1;
    end
  end

  // A2
  reg A2_wr_en_mux;
  reg [12:0] A2_index_mux;
  reg [31:0] A2_data_in_mux;
  always @(*) begin
    if (busy) begin 
      A2_wr_en_mux = A_wr_en2;
      A2_index_mux = A_index2;
      A2_data_in_mux = A_data_in2;
    end 
    else begin 
      A2_wr_en_mux = (cmd_valid && op == OP_WRITE_A2);
      A2_index_mux = cmd_payload_inputs_0[12:0];
      A2_data_in_mux = cmd_payload_inputs_1;
    end
  end

  // A3
  reg A3_wr_en_mux;
  reg [12:0] A3_index_mux;
  reg [31:0] A3_data_in_mux;
  always @(*) begin
    if (busy) begin 
      A3_wr_en_mux = A_wr_en3;
      A3_index_mux = A_index3;
      A3_data_in_mux = A_data_in3;
    end 
    else begin 
      A3_wr_en_mux = (cmd_valid && op == OP_WRITE_A3);
      A3_index_mux = cmd_payload_inputs_0[12:0];
      A3_data_in_mux = cmd_payload_inputs_1;
    end
  end

  // ==============================================================
  // B Buffer
  // ==============================================================
  reg B_wr_en_mux;
  reg [12:0] B_index_mux;
  reg [31:0] B_data_in_mux;

  always @(*) begin
    if (busy) begin 
      B_wr_en_mux = B_wr_en0;
      B_index_mux = B_index0;
      B_data_in_mux = B_data_in0;
    end 
    else begin 
      B_wr_en_mux = (cmd_valid && op == OP_WRITE_B);
      B_index_mux = cmd_payload_inputs_0[12:0];
      B_data_in_mux = cmd_payload_inputs_1;
    end
  end

  // ==============================================================
  // C Buffer
  // ==============================================================
  
  // C0
  reg C0_wr_en_mux;
  reg [12:0] C0_index_mux;
  reg [127:0] C0_data_in_mux;
  always @(*) begin
    if (busy) begin 
      C0_wr_en_mux = C_wr_en0;
      C0_index_mux = C_index0;
      C0_data_in_mux = C_data_in0;
    end 
    else begin 
      C0_wr_en_mux = 1'b0;
      C0_index_mux = cmd_payload_inputs_0[12:0];
      C0_data_in_mux = 128'b0;
    end
  end

  // C1
  reg C1_wr_en_mux;
  reg [12:0] C1_index_mux;
  reg [127:0] C1_data_in_mux;
  always @(*) begin
    if (busy) begin 
      C1_wr_en_mux = C_wr_en1;
      C1_index_mux = C_index1;
      C1_data_in_mux = C_data_in1;
    end 
    else begin 
      C1_wr_en_mux = 1'b0;
      C1_index_mux = cmd_payload_inputs_0[12:0];
      C1_data_in_mux = 128'b0;
    end
  end

  // C2
  reg C2_wr_en_mux;
  reg [12:0] C2_index_mux;
  reg [127:0] C2_data_in_mux;
  always @(*) begin
    if (busy) begin 
      C2_wr_en_mux = C_wr_en2;
      C2_index_mux = C_index2;
      C2_data_in_mux = C_data_in2;
    end 
    else begin 
      C2_wr_en_mux = 1'b0;
      C2_index_mux = cmd_payload_inputs_0[12:0];
      C2_data_in_mux = 128'b0;
    end
  end

  // C3
  reg C3_wr_en_mux;
  reg [12:0] C3_index_mux;
  reg [127:0] C3_data_in_mux;
  always @(*) begin
    if (busy) begin 
      C3_wr_en_mux = C_wr_en3;
      C3_index_mux = C_index3;
      C3_data_in_mux = C_data_in3;
    end 
    else begin 
      C3_wr_en_mux = 1'b0;
      C3_index_mux = cmd_payload_inputs_0[12:0];
      C3_data_in_mux = 128'b0;
    end
  end

  // ==============================================================
  // Handshake
  // ==============================================================
  reg busy_d;
  always @(posedge clk) begin
    if (reset) busy_d <= 1'b0;
    else busy_d <= busy;
  end

  wire tpu_done = (busy_d && !busy);
  assign cmd_ready = ~rsp_valid;
  always @(posedge clk) begin
    if (reset) begin
      rsp_valid <= 1'b0;
    end
    else if (rsp_valid) begin
      rsp_valid <= ~rsp_ready;
    end
    else if (tpu_done) begin
      rsp_valid <= 1'b1;
    end
    else if (cmd_valid && op == OP_START) begin
      rsp_valid <= 1'b0;
    end
    else if (cmd_valid) begin
      rsp_valid <= 1'b1;
    end
  end

  // ==============================================================
  // Readback
  // ==============================================================
  reg [31:0] rsp_payload_outputs_0_next;

  always @(*) begin
    rsp_payload_outputs_0_next = 32'd0; 
    case (op)
      OP_READ_C0: begin
        case (c_sel)
          2'd0: rsp_payload_outputs_0_next = C0_data_out[31:0];
          2'd1: rsp_payload_outputs_0_next = C0_data_out[63:32];
          2'd2: rsp_payload_outputs_0_next = C0_data_out[95:64];
          2'd3: rsp_payload_outputs_0_next = C0_data_out[127:96];
        endcase
      end
      OP_READ_C1: begin
        case (c_sel)
          2'd0: rsp_payload_outputs_0_next = C1_data_out[31:0];
          2'd1: rsp_payload_outputs_0_next = C1_data_out[63:32];
          2'd2: rsp_payload_outputs_0_next = C1_data_out[95:64];
          2'd3: rsp_payload_outputs_0_next = C1_data_out[127:96];
        endcase
      end
      OP_READ_C2: begin
        case (c_sel)
          2'd0: rsp_payload_outputs_0_next = C2_data_out[31:0];
          2'd1: rsp_payload_outputs_0_next = C2_data_out[63:32];
          2'd2: rsp_payload_outputs_0_next = C2_data_out[95:64];
          2'd3: rsp_payload_outputs_0_next = C2_data_out[127:96];
        endcase
      end
      OP_READ_C3: begin
        case (c_sel)
          2'd0: rsp_payload_outputs_0_next = C3_data_out[31:0];
          2'd1: rsp_payload_outputs_0_next = C3_data_out[63:32];
          2'd2: rsp_payload_outputs_0_next = C3_data_out[95:64];
          2'd3: rsp_payload_outputs_0_next = C3_data_out[127:96];
        endcase
      end
      OP_QTIZE_ACC: begin
        rsp_payload_outputs_0_next = final_quantized_res2;
      end
      OP_RELU: begin
        rsp_payload_outputs_0_next = (cmd_payload_inputs_0[31]) ? RELU_result_m2 : RELU_result;
      end
      default: begin
        rsp_payload_outputs_0_next = 32'd0;
      end
    endcase
  end

  always @(posedge clk) begin
    if (reset) begin
      rsp_payload_outputs_0 <= 32'd0;
    end else if (cmd_valid) begin
      rsp_payload_outputs_0 <= rsp_payload_outputs_0_next;
    end
  end

  // ==============================================================
  // Buffer
  // ==============================================================
  
  // A Buffers
  global_buffer_bram #(.ADDR_BITS(13), .DATA_BITS(32)) gbuff_A0( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(A0_wr_en_mux), .index(A0_index_mux), .data_in(A0_data_in_mux), .data_out(A0_data_out) );
  global_buffer_bram #(.ADDR_BITS(13), .DATA_BITS(32)) gbuff_A1( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(A1_wr_en_mux), .index(A1_index_mux), .data_in(A1_data_in_mux), .data_out(A1_data_out) );
  global_buffer_bram #(.ADDR_BITS(13), .DATA_BITS(32)) gbuff_A2( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(A2_wr_en_mux), .index(A2_index_mux), .data_in(A2_data_in_mux), .data_out(A2_data_out) );
  global_buffer_bram #(.ADDR_BITS(13), .DATA_BITS(32)) gbuff_A3( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(A3_wr_en_mux), .index(A3_index_mux), .data_in(A3_data_in_mux), .data_out(A3_data_out) );

  // B Buffers
  global_buffer_bram #(.ADDR_BITS(13), .DATA_BITS(32)) gbuff_B0( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(B_wr_en_mux), .index(B_index_mux), .data_in(B_data_in_mux), .data_out(B0_data_out) );
  global_buffer_bram #(.ADDR_BITS(13), .DATA_BITS(32)) gbuff_B1( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(B_wr_en_mux), .index(B_index_mux), .data_in(B_data_in_mux), .data_out(B1_data_out) );
  global_buffer_bram #(.ADDR_BITS(13), .DATA_BITS(32)) gbuff_B2( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(B_wr_en_mux), .index(B_index_mux), .data_in(B_data_in_mux), .data_out(B2_data_out) );
  global_buffer_bram #(.ADDR_BITS(13), .DATA_BITS(32)) gbuff_B3( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(B_wr_en_mux), .index(B_index_mux), .data_in(B_data_in_mux), .data_out(B3_data_out) );

  // C Buffers
  global_buffer_bram #(.ADDR_BITS(10), .DATA_BITS(128)) gbuff_C0( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(C0_wr_en_mux), .index(C0_index_mux[9:0]), .data_in(C0_data_in_mux), .data_out(C0_data_out) );
  global_buffer_bram #(.ADDR_BITS(10), .DATA_BITS(128)) gbuff_C1( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(C1_wr_en_mux), .index(C1_index_mux[9:0]), .data_in(C1_data_in_mux), .data_out(C1_data_out) );
  global_buffer_bram #(.ADDR_BITS(10), .DATA_BITS(128)) gbuff_C2( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(C2_wr_en_mux), .index(C2_index_mux[9:0]), .data_in(C2_data_in_mux), .data_out(C2_data_out) );
  global_buffer_bram #(.ADDR_BITS(10), .DATA_BITS(128)) gbuff_C3( .clk(clk), .rst_n(reset), .ram_en(1'b1), .wr_en(C3_wr_en_mux), .index(C3_index_mux[9:0]), .data_in(C3_data_in_mux), .data_out(C3_data_out) );

  // ==============================================================
  // TPU
  // ==============================================================
  TPU tpu0( .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .K(K), .M(M), .N(N), .busy(busy0), .input_offset(input_offset),
    .A_wr_en(A_wr_en0), .A_index(A_index0), .A_data_in(A_data_in0), .A_data_out(A0_data_out),
    .B_wr_en(B_wr_en0), .B_index(B_index0), .B_data_in(B_data_in0), .B_data_out(B0_data_out),
    .C_wr_en(C_wr_en0), .C_index(C_index0), .C_data_in(C_data_in0), .C_data_out(C0_data_out) );

  TPU tpu1( .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .K(K), .M(M), .N(N), .busy(busy1), .input_offset(input_offset),
    .A_wr_en(A_wr_en1), .A_index(A_index1), .A_data_in(A_data_in1), .A_data_out(A1_data_out),
    .B_wr_en(B_wr_en1), .B_index(B_index1), .B_data_in(B_data_in1), .B_data_out(B1_data_out),
    .C_wr_en(C_wr_en1), .C_index(C_index1), .C_data_in(C_data_in1), .C_data_out(C1_data_out) );

  TPU tpu2( .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .K(K), .M(M), .N(N), .busy(busy2), .input_offset(input_offset),
    .A_wr_en(A_wr_en2), .A_index(A_index2), .A_data_in(A_data_in2), .A_data_out(A2_data_out),
    .B_wr_en(B_wr_en2), .B_index(B_index2), .B_data_in(B_data_in2), .B_data_out(B2_data_out),
    .C_wr_en(C_wr_en2), .C_index(C_index2), .C_data_in(C_data_in2), .C_data_out(C2_data_out) );

  TPU tpu3( .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .K(K), .M(M), .N(N), .busy(busy3), .input_offset(input_offset),
    .A_wr_en(A_wr_en3), .A_index(A_index3), .A_data_in(A_data_in3), .A_data_out(A3_data_out),
    .B_wr_en(B_wr_en3), .B_index(B_index3), .B_data_in(B_data_in3), .B_data_out(B3_data_out),
    .C_wr_en(C_wr_en3), .C_index(C_index3), .C_data_in(C_data_in3), .C_data_out(C3_data_out) );

endmodule