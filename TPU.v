
module TPU(
    clk,
    rst_n,

    in_valid,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out,
    input_offset
);


input clk;
input rst_n;
input            in_valid;
input [31:0]      K;
input [31:0]      M;
input [31:0]      N;
output  reg      busy;

output           A_wr_en;
output reg [15:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output reg [15:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output           C_wr_en;
output reg [15:0]    C_index;
output [127:0]   C_data_in;
input  [127:0]   C_data_out;

input [31:0] input_offset;


//* Implement your design here
reg[2:0] state, next;
parameter S_IDLE = 3'd0;
parameter S_CAL = 3'd1;
parameter S_WRITE = 3'd2;
parameter S_DONE = 3'd3;

reg[31:0] M_reg, M_reg_next;
reg[31:0] K_reg, K_reg_next;
reg[31:0] N_reg, N_reg_next;

assign A_wr_en = 0;
assign B_wr_en = 0;

///FSM
always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        state <= S_IDLE;
    end
    else begin
        state <= next;
    end
end

always@(*)begin
    next = state;
    case(state)
        S_IDLE:begin
            if(in_valid)begin
                next = S_CAL;
            end
            else begin
                next = S_IDLE;
            end
        end
        S_CAL:begin
            if(K_cnt == K_reg + 7)begin
                next = S_WRITE;
            end
            else begin
                next = S_CAL;
            end
        end
        S_WRITE:begin
            if(outcnt == 3)begin
                if(Acnt == Agroups && Bcnt == Bgroups)begin
                    next = S_DONE;
                end
                else begin
                    next = S_CAL;
                end
            end
            else begin
                next = S_WRITE;
            end
        end
        S_DONE:begin
            next = S_IDLE;
        end
    endcase
end

////MKN
always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        M_reg <= 0;
        K_reg <= 0;
        N_reg <= 0;
    end 
    else begin
        M_reg <= M_reg_next;
        K_reg <= K_reg_next;
        N_reg <= N_reg_next;
    end
end

always@(*)begin
    M_reg_next = M_reg;
    K_reg_next = K_reg;
    N_reg_next = N_reg;
    if(in_valid)begin
        M_reg_next = M;
        K_reg_next = K;
        N_reg_next = N;
    end
end
/////////groups
wire [31:0] Agroups;
wire [2:0] Arem;
assign Agroups = ((M_reg + 8'd3) >> 2) - 1 ;
assign Arem = M_reg % 4;

wire [31:0] Bgroups;
wire [2:0] Brem;
assign Bgroups = ((N_reg + 8'd3) >> 2) - 1;
assign Brem = N_reg % 4;

reg [31:0] Acnt, Acnt_next;
reg [31:0] Bcnt, Bcnt_next;
always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        Acnt <= 0;
        Bcnt <= 0;
    end
    else begin
        Acnt <= Acnt_next;
        Bcnt <= Bcnt_next;
    end
end

always@(*)begin
    Acnt_next = Acnt;
    Bcnt_next = Bcnt;
    if(state == S_WRITE && outcnt == 3)begin
        if(Acnt == Agroups)begin
            Bcnt_next = Bcnt + 1;
            Acnt_next = 0;
        end
        else begin
            Acnt_next = Acnt + 1;
        end
    end
    else if(state == S_DONE)begin
        Acnt_next = 0;
        Bcnt_next = 0;
    end        
end
            



////busy
reg busy_next;
always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        busy <= 0;
    end
    else begin
        busy <= busy_next;
    end
end

always@(*)begin
    busy_next = busy;
    if(in_valid)begin
        busy_next = 1;
    end
    else if(next == S_DONE)begin
        busy_next = 0;
    end
end



/////////K counter
reg [32:0] K_cnt, K_cnt_next;
always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        K_cnt <= 0;
    end
    else begin
        K_cnt <= K_cnt_next; 
    end
end

always@(*)begin
    K_cnt_next = K_cnt;
    if(state == S_CAL && K_cnt < K_reg + 7)begin
        K_cnt_next = K_cnt + 1;
    end
    else begin
        K_cnt_next = 0;
    end
end

//////////buffer A
reg[15:0] A_index_next;
always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        A_index <= 0;
    end
    else begin
        A_index <= A_index_next;
    end
end

always@(*)begin
    A_index_next = A_index;
    if(state == S_CAL)begin
        A_index_next = A_index + 1;
    end
    else if(state == S_WRITE && outcnt == 3)begin
        if(Acnt == Agroups)begin
            A_index_next = 0;
        end
        else begin
            A_index_next = K_reg * (Acnt + 1);
        end
    end
    else if(state == S_DONE)begin
        A_index_next = 0;
    end
end

reg signed[7:0] A_stall_0, A_stall_0_next;
reg signed[15:0] A_stall_1, A_stall_1_next;
reg signed[23:0] A_stall_2, A_stall_2_next;
reg signed[31:0] A_stall_3, A_stall_3_next;

always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        A_stall_0 <= 0;
        A_stall_1 <= 0;
        A_stall_2 <= 0;
        A_stall_3 <= 0;
    end
    else begin
        A_stall_0 <= A_stall_0_next;
        A_stall_1 <= A_stall_1_next;
        A_stall_2 <= A_stall_2_next;
        A_stall_3 <= A_stall_3_next;
    end
end

always@(*)begin
    A_stall_0_next = A_stall_0;
    A_stall_1_next = A_stall_1;
    A_stall_2_next = A_stall_2;
    A_stall_3_next = A_stall_3;
    if(state == S_CAL && K_cnt < K_reg)begin
        A_stall_1_next = A_stall_1 >> 8;
        A_stall_2_next = A_stall_2 >> 8;
        A_stall_3_next = A_stall_3 >> 8;
        A_stall_0_next = A_data_out[31:24];
        A_stall_1_next[15:8] = A_data_out[23:16];
        A_stall_2_next[23:16] = A_data_out[15:8];
        A_stall_3_next[31:24] = A_data_out[7:0];
    end
    else if(state == S_CAL && (K_cnt == K_reg))begin
        A_stall_0_next = 0;
        A_stall_1_next = A_stall_1 >> 8;
        A_stall_2_next = A_stall_2 >> 8;
        A_stall_3_next = A_stall_3 >> 8;
    end
    else if(state == S_CAL && (K_cnt == K_reg + 1))begin
        A_stall_0_next = 0;
        A_stall_1_next = 0;
        A_stall_2_next = A_stall_2 >> 8;
        A_stall_3_next = A_stall_3 >> 8;
    end 
    else if(state == S_CAL && (K_cnt == K_reg + 2))begin
        A_stall_0_next = 0;
        A_stall_1_next = 0;
        A_stall_2_next = 0;
        A_stall_3_next = A_stall_3 >> 8;
    end 
    else if(state == S_CAL && (K_cnt == K_reg + 3))begin
        A_stall_0_next = 0;
        A_stall_1_next = 0;
        A_stall_2_next = 0;
        A_stall_3_next = 0;
    end 
end

/////////buffer B
reg[15:0] B_index_next;
always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        B_index <= 0;
    end
    else begin
        B_index <= B_index_next;
    end
end

always@(*)begin
    B_index_next = B_index;
    if(state == S_CAL)begin
        B_index_next = B_index + 1;
    end
    else if(state == S_WRITE)begin
        if(Acnt == Agroups)begin
            B_index_next = K_reg * (Bcnt + 1);
        end
        else begin
            B_index_next = K_reg * Bcnt;
        end
    end
    else if(state == S_DONE)begin
        B_index_next = 0;
    end
end
reg signed[7:0]  B_stall_0, B_stall_0_next;
reg signed[15:0] B_stall_1, B_stall_1_next;
reg signed[23:0] B_stall_2, B_stall_2_next;
reg signed[31:0] B_stall_3, B_stall_3_next;

always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        B_stall_0 <= 0;
        B_stall_1 <= 0;
        B_stall_2 <= 0;
        B_stall_3 <= 0;
    end
    else begin
        B_stall_0 <= B_stall_0_next;
        B_stall_1 <= B_stall_1_next;
        B_stall_2 <= B_stall_2_next;
        B_stall_3 <= B_stall_3_next;
    end
end

always@(*)begin
    B_stall_0_next = B_stall_0;
    B_stall_1_next = B_stall_1;
    B_stall_2_next = B_stall_2;
    B_stall_3_next = B_stall_3;
    if(state == S_CAL && K_cnt < K_reg )begin
        B_stall_1_next = B_stall_1 >> 8;
        B_stall_2_next = B_stall_2 >> 8;
        B_stall_3_next = B_stall_3 >> 8;
        B_stall_0_next = B_data_out[31:24];
        B_stall_1_next[15:8] = B_data_out[23:16];
        B_stall_2_next[23:16] = B_data_out[15:8];
        B_stall_3_next[31:24] = B_data_out[7:0];
    end
    else if(state == S_CAL && K_cnt == K_reg)begin
        B_stall_0_next = 0;
        B_stall_1_next = B_stall_1 >> 8;
        B_stall_2_next = B_stall_2 >> 8;
        B_stall_3_next = B_stall_3 >> 8;        
    end
    else if(state == S_CAL && K_cnt == K_reg + 1)begin
        B_stall_0_next = 0;
        B_stall_1_next = 0;
        B_stall_2_next = B_stall_2 >> 8;
        B_stall_3_next = B_stall_3 >> 8;        
    end
    else if(state == S_CAL && K_cnt == K_reg + 2)begin
        B_stall_0_next = 0;
        B_stall_1_next = 0;
        B_stall_2_next = 0;
        B_stall_3_next = B_stall_3 >> 8;        
    end
    else if(state == S_CAL && K_cnt == K_reg + 3)begin
        B_stall_0_next = 0;
        B_stall_1_next = 0;
        B_stall_2_next = 0;
        B_stall_3_next = 0;        
    end

end

/////////c buffer
reg [1:0] outcnt, outcnt_next;
always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        outcnt <= 0;
    end
    else begin
        outcnt <= outcnt_next;
    end
end

always@(*)begin
    outcnt_next = outcnt;
    if(state == S_WRITE)begin
        outcnt_next = outcnt + 1;
    end
end

reg [15:0] C_index_next;
always@(posedge clk or negedge rst_n)begin
    if(!rst_n)begin
        C_index <= 0;
    end
    else begin
        C_index <= C_index_next;
    end
end

always@(*)begin
    C_index_next = C_index;
    if(state == S_WRITE)begin
        if(outcnt == 3 && Acnt == Agroups)begin
            if(Arem != 0)begin
                C_index_next = C_index - (4 - Arem) + 1;
            end
            else begin
               C_index_next = C_index + 1;
            end 
        end
        else begin
            C_index_next = C_index + 1;
        end
    end
    else if(state == S_DONE)begin
        C_index_next = 0;
    end
end

wire signed[127:0] psum_1, psum_2, psum_3, psum_0;
assign C_wr_en = (state == S_WRITE) ? 1 : 0;
assign C_data_in = (!rst_n) ? 0 :
                   (outcnt == 2'd0) ? psum_0 :
                   (outcnt == 2'd1) ? psum_1 :
                   (outcnt == 2'd2) ? psum_2 :
                   (outcnt == 2'd3) ? psum_3 : 0;


// //////////systolic

systolic_array sa(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .SA_A0 (A_stall_0),
    .SA_A1 (A_stall_1[7:0]),
    .SA_A2 (A_stall_2[7:0]),
    .SA_A3 (A_stall_3[7:0]),
    .SA_B0 (B_stall_0),
    .SA_B1 (B_stall_1[7:0]), 
    .SA_B2 (B_stall_2[7:0]),
    .SA_B3 (B_stall_3[7:0]),


    .SAout0 ( psum_0),
    .SAout1 ( psum_1),
    .SAout2 ( psum_2),
    .SAout3 ( psum_3),
    .input_offset(input_offset)

);


wire clear;
assign clear = ((state == S_DONE) || (state == S_WRITE && next == S_CAL)) ? 1 : 0;

endmodule


module systolic_array(
    input clk,
    input rst_n,
    input clear,

    input signed[7:0] SA_A0,
    input signed[7:0] SA_A1,
    input signed[7:0] SA_A2,
    input signed[7:0] SA_A3,
    input signed[7:0] SA_B0,
    input signed[7:0] SA_B1,
    input signed[7:0] SA_B2,
    input signed[7:0] SA_B3,
    

    output signed[127:0] SAout0,
    output signed[127:0] SAout1,
    output signed[127:0] SAout2,
    output signed[127:0] SAout3,
    input [31:0] input_offset


);
wire signed[7:0] horizon[0:11];
wire signed[7:0] vertical[0:11];

wire signed[7:0]  A_in [0:3];
wire signed[7:0]  B_in [0:3];
assign A_in[0] = SA_A0;
assign A_in[1] = SA_A1;
assign A_in[2] = SA_A2;
assign A_in[3] = SA_A3;

assign B_in[0] = SA_B0;
assign B_in[1] = SA_B1;
assign B_in[2] = SA_B2;
assign B_in[3] = SA_B3;

wire signed[31:0] C_acc[0:3][0:3];

assign SAout0 = {C_acc[0][0], C_acc[0][1], C_acc[0][2], C_acc[0][3]};
assign SAout1 = {C_acc[1][0], C_acc[1][1], C_acc[1][2], C_acc[1][3]};
assign SAout2 = {C_acc[2][0], C_acc[2][1], C_acc[2][2], C_acc[2][3]};
assign SAout3 = {C_acc[3][0], C_acc[3][1], C_acc[3][2], C_acc[3][3]};

/////first row
PE pe0_0(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(B_in[0]),
    .left(A_in[0]),
    .right(horizon[0]),
    .down(vertical[0]),
    .acc_out(C_acc[0][0]),
    .input_offset(input_offset)
);

PE pe0_1(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(B_in[1]),
    .left(horizon[0]),
    .right(horizon[1]),
    .down(vertical[1]),
    .acc_out(C_acc[0][1]),
    .input_offset(input_offset)
);

PE pe0_2(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(B_in[2]),
    .left(horizon[1]),
    .right(horizon[2]),
    .down(vertical[2]),
    .acc_out(C_acc[0][2]),
    .input_offset(input_offset)
);

PE pe0_3(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(B_in[3]),
    .left(horizon[2]),
    .right(),
    .down(vertical[3]),
    .acc_out(C_acc[0][3]),
    .input_offset(input_offset)
);

//////second row

PE pe1_0(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[0]),
    .left(A_in[1]),
    .right(horizon[3]),
    .down(vertical[4]),
    .acc_out(C_acc[1][0]),
    .input_offset(input_offset)
);

PE pe1_1(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[1]),
    .left(horizon[3]),
    .right(horizon[4]),
    .down(vertical[5]),
    .acc_out(C_acc[1][1]),
    .input_offset(input_offset)
);

PE pe1_2(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[2]),
    .left(horizon[4]),
    .right(horizon[5]),
    .down(vertical[6]),
    .acc_out(C_acc[1][2]),
    .input_offset(input_offset)
);

PE pe1_3(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[3]),
    .left(horizon[5]),
    .right(),
    .down(vertical[7]),
    .acc_out(C_acc[1][3]),
    .input_offset(input_offset)
);

////third row
PE pe2_0(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[4]),
    .left(A_in[2]),
    .right(horizon[6]),
    .down(vertical[8]),
    .acc_out(C_acc[2][0]),
    .input_offset(input_offset)
);

PE pe2_1(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[5]),
    .left(horizon[6]),
    .right(horizon[7]),
    .down(vertical[9]),
    .acc_out(C_acc[2][1]),
    .input_offset(input_offset)
);

PE pe2_2(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[6]),
    .left(horizon[7]),
    .right(horizon[8]),
    .down(vertical[10]),
    .acc_out(C_acc[2][2]),
    .input_offset(input_offset)
);

PE pe2_3(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[7]),
    .left(horizon[8]),
    .right(),
    .down(vertical[11]),
    .acc_out(C_acc[2][3]),
    .input_offset(input_offset)
);

////fourth row
PE pe3_0(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[8]),
    .left(A_in[3]),
    .right(horizon[9]),
    .down(),
    .acc_out(C_acc[3][0]),
    .input_offset(input_offset)
);

PE pe3_1(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[9]),
    .left(horizon[9]),
    .right(horizon[10]),
    .down(),
    .acc_out(C_acc[3][1]),
    .input_offset(input_offset)
);

PE pe3_2(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[10]),
    .left(horizon[10]),
    .right(horizon[11]),
    .down(),
    .acc_out(C_acc[3][2]),
    .input_offset(input_offset)
);

PE pe3_3(
    .clk (clk),
    .rst_n (rst_n),
    .clear (clear),

    .up(vertical[11]),
    .left(horizon[11]),
    .right(),
    .down(),
    .acc_out(C_acc[3][3]),
    .input_offset(input_offset)
);


endmodule


module PE(
    input clk,
    input rst_n,
    input clear,

    input signed[7:0] up,
    input signed[7:0] left,
    output reg signed[7:0] right,
    output reg signed[7:0] down,
    output reg signed[31:0] acc_out,
    input[31:0] input_offset
);
    wire signed[8:0] left_offset = left + $signed(input_offset[8:0]);

    /////acc
    reg signed[31:0] acc_out_next;


    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            acc_out <= 0;
        end
        else begin 
            acc_out <= acc_out_next;
        end
    end

    always@(*)begin
        acc_out_next = acc_out;
        if(clear)begin
            acc_out_next = 0;
        end
        else begin
            acc_out_next = up * left_offset + acc_out;
        end     
    end



    /////input output
    reg signed[7:0] right_next, down_next;
    always@(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            right <= 0;
            down <= 0;
        end
        else begin
            right <= right_next;
            down <= down_next;
        end
    end

    always@(*)begin
        right_next = right;
        down_next = down;
        right_next = left;
        down_next = up;
    end
    


endmodule
// module TPU(
//     input                clk,
//     input                rst_n,
//     input                in_valid,
//     input       [12:0]   K,          // ==== changed: 12 -> 13 bits ====
//     input       [12:0]   M,          // ==== changed ====
//     input       [12:0]   N,          // ==== changed ====
//     output reg           busy,

//     output reg           A_wr_en,
//     output reg  [12:0]   A_index,    // ==== changed: 12 -> 13 bits ====
//     output      [31:0]   A_data_in,
//     input       [31:0]   A_data_out,

//     output reg           B_wr_en,
//     output reg  [12:0]   B_index,    // ==== changed ====
//     output      [31:0]   B_data_in,
//     input       [31:0]   B_data_out,

//     output reg           C_wr_en,
//     output reg  [12:0]   C_index,    // ==== changed ====
//     output reg  [127:0]  C_data_in,
//     input       [127:0]  C_data_out,

//     input       [31:0]   input_offset
// );

//     parameter IDLE = 2'd0;
//     parameter CALC = 2'd1;
//     parameter WRITE = 2'd2;
//     parameter DONE = 2'd3;

//     // ==== changed: 12 -> 13 bits ====
//     reg [12:0] K_reg;
//     reg [12:0] M_reg;
//     reg [12:0] N_reg;

//     reg [1:0] current_state, next_state;

//     //-------------------------
//     // FSM
//     //-------------------------
//     always@(posedge clk or negedge rst_n) begin
//         if(!rst_n)
//             current_state <= IDLE;
//         else
//             current_state <= next_state;
//     end

//     always @(*) begin
//         case(current_state)
//             IDLE:   next_state = (in_valid ? CALC : IDLE);

//             CALC:   next_state =
//                         (counter < (K_reg + 6) && busy) ? CALC : WRITE;

//             WRITE: begin
//                 if (output_num < rem_out)
//                     next_state = WRITE;
//                 else
//                     next_state = (outer_cnt == outer_groups ? DONE : CALC);
//             end

//             DONE:   next_state = IDLE;

//             default: next_state = IDLE;
//         endcase
//     end

//     //-------------------------
//     // Busy signal
//     //-------------------------
//     always @(posedge clk or negedge rst_n) begin
//         if(!rst_n)
//             busy <= 0;
//         else if(in_valid)
//             busy <= 1;
//         else if(next_state == DONE)
//             busy <= 0;
//     end

//     //-------------------------
//     // Write enable
//     //-------------------------
//     always @(*) begin
//         if(next_state == WRITE) begin
//             A_wr_en = 0;
//             B_wr_en = 0;
//             C_wr_en = 1;
//         end else begin
//             A_wr_en = 0;
//             B_wr_en = 0;
//             C_wr_en = 0;
//         end
//     end

//     //-------------------------
//     // latch K/M/N
//     //-------------------------
//     always @(posedge clk) begin
//         if (in_valid) begin
//             K_reg <= K;
//             M_reg <= M;
//             N_reg <= N;
//         end
//     end

//     //-------------------------
//     // group count (12->13)
//     //-------------------------
//     wire [12:0] outer_groups, inner_groups;
//     assign outer_groups = (N_reg + 13'd3) >> 2;
//     assign inner_groups = (M_reg + 13'd3) >> 2;

//     //-------------------------
//     // remainder
//     //-------------------------
//     reg [2:0] rem_out;
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n)
//             rem_out <= 3'd0;
//         else if (current_state == CALC) begin
//             if (inner_cnt == (inner_groups - 1) && M_reg[1:0] != 2'b00)
//                 rem_out <= M_reg[1:0];
//             else
//                 rem_out <= 4;
//         end
//     end

//     //-------------------------
//     // global counter (unchanged)
//     //-------------------------
//     reg [31:0] counter;
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n)
//             counter <= 0;
//         else if (current_state == CALC)
//             counter <= counter + 1;
//         else
//             counter <= 0;
//     end

//     //-------------------------
//     // inner_cnt (12->13)
//     //-------------------------
//     reg [12:0] inner_cnt;
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n)
//             inner_cnt <= 0;
//         else if (current_state == DONE)
//             inner_cnt <= 0;
//         else if (current_state == WRITE) begin
//             if (inner_groups == 1)
//                 inner_cnt <= 1;
//             else if (output_num == (rem_out - 1))
//                 inner_cnt <= (inner_cnt < inner_groups ? inner_cnt + 1 : 1);
//         end
//     end

//     //-------------------------
//     // outer_cnt (12->13)
//     //-------------------------
//     reg [12:0] outer_cnt, next_outer_cnt;

//     always @(*) begin
//         next_outer_cnt = outer_cnt;

//         if (current_state == WRITE && next_state == CALC && busy) begin
//             if ((inner_groups == 1) ||
//                 (inner_groups != 1 && inner_cnt == inner_groups - 1))
//                 next_outer_cnt = outer_cnt + 1;
//         end else if (current_state == DONE)
//             next_outer_cnt = 0;
//     end

//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n)
//             outer_cnt <= 0;
//         else
//             outer_cnt <= next_outer_cnt;
//     end

//     //-------------------------
//     // output_num
//     //-------------------------
//     reg [31:0] output_num;
//     always @(posedge clk or negedge rst_n) begin
//         if(!rst_n)
//             output_num <= 0;
//         else if(current_state == WRITE)
//             output_num <= output_num + 1;
//         else
//             output_num <= 0;
//     end

//     //-------------------------
//     // A_index (12->13 bits)
//     //-------------------------
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n)
//             A_index <= 13'd0;
//         else begin
//             case (current_state)
//                 WRITE: begin
//                     if (K_reg == 1)
//                         A_index <= 13'd1;
//                     else if (next_state == CALC) begin
//                         if (inner_cnt == inner_groups)
//                             A_index <= 13'd0;
//                         else
//                             A_index <= A_index + 13'd1;
//                     end
//                 end
//                 CALC: begin
//                     if (counter < (K_reg - 1))
//                         A_index <= A_index + 13'd1;
//                 end
//                 DONE:
//                     A_index <= 13'd0;
//             endcase
//         end
//     end

//     //-------------------------
//     // B_index (12->13 bits)
//     //-------------------------
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n)
//             B_index <= 13'd0;
//         else begin
//             case (current_state)
//                 WRITE: begin
//                     if (busy) begin
//                         if (inner_groups == 1)
//                             B_index <= K_reg * next_outer_cnt;
//                         else
//                             B_index <= K_reg * outer_cnt;
//                     end
//                 end
//                 CALC: begin
//                     if (counter < K_reg)
//                         B_index <= B_index + 13'd1;
//                 end
//                 DONE:
//                     B_index <= 13'd0;
//             endcase
//         end
//     end

//     //-------------------------
//     // C_index (12->13 bits)
//     //-------------------------
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n)
//             C_index <= 13'd0;
//         else begin
//             case (current_state)
//                 DONE: C_index <= 13'd0;

//                 WRITE: begin
//                     if (next_state == WRITE)
//                         C_index <= C_index + 13'd1;
//                 end
//             endcase
//         end
//     end

//     reg signed [7:0] A_in_reg1;
//     reg signed [7:0] A_in_reg2 [0:1];
//     reg signed [7:0] A_in_reg3 [0:2];
//     reg signed [7:0] A_in_reg4 [0:3];

//     reg signed [7:0] B_in_reg1;
//     reg signed [7:0] B_in_reg2 [0:1];
//     reg signed [7:0] B_in_reg3 [0:2];
//     reg signed [7:0] B_in_reg4 [0:3];

//     integer i;
//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             A_in_reg1 <= 8'd0;
//             B_in_reg1 <= 8'd0;

//             for (i = 0; i < 2; i = i + 1) begin
//                 A_in_reg2[i] <= 8'd0;
//                 B_in_reg2[i] <= 8'd0;
//             end
//             for (i = 0; i < 3; i = i + 1) begin
//                 A_in_reg3[i] <= 8'd0;
//                 B_in_reg3[i] <= 8'd0;
//             end
//             for (i = 0; i < 4; i = i + 1) begin
//                 A_in_reg4[i] <= 8'd0;
//                 B_in_reg4[i] <= 8'd0;
//             end
//         end 
//         else if (current_state == CALC) begin
//             A_in_reg2[0] <= A_in_reg2[1];
//             A_in_reg3[0] <= A_in_reg3[1];
//             A_in_reg3[1] <= A_in_reg3[2];
//             A_in_reg4[0] <= A_in_reg4[1];
//             A_in_reg4[1] <= A_in_reg4[2];
//             A_in_reg4[2] <= A_in_reg4[3];

//             B_in_reg2[0] <= B_in_reg2[1];
//             B_in_reg3[0] <= B_in_reg3[1];
//             B_in_reg3[1] <= B_in_reg3[2];
//             B_in_reg4[0] <= B_in_reg4[1];
//             B_in_reg4[1] <= B_in_reg4[2];
//             B_in_reg4[2] <= B_in_reg4[3];

//             if (counter < K_reg) begin
//                 A_in_reg1    <= $signed(A_data_out[31:24]);
//                 A_in_reg2[1] <= $signed(A_data_out[23:16]);
//                 A_in_reg3[2] <= $signed(A_data_out[15:8]);
//                 A_in_reg4[3] <= $signed(A_data_out[7:0]);

//                 B_in_reg1    <= $signed(B_data_out[31:24]);
//                 B_in_reg2[1] <= $signed(B_data_out[23:16]);
//                 B_in_reg3[2] <= $signed(B_data_out[15:8]);
//                 B_in_reg4[3] <= $signed(B_data_out[7:0]);
//             end
//             else begin
//                 A_in_reg1    <= 8'd0;
//                 A_in_reg2[1] <= 8'd0;
//                 A_in_reg3[2] <= 8'd0;
//                 A_in_reg4[3] <= 8'd0;

//                 B_in_reg1    <= 8'd0;
//                 B_in_reg2[1] <= 8'd0;
//                 B_in_reg3[2] <= 8'd0;
//                 B_in_reg4[3] <= 8'd0;
//             end
//         end
//     end

//     // output mux
//     wire [127:0] C_row1, C_row2, C_row3, C_row4;

//     always @(*) begin
//         case (output_num)
//             0: C_data_in = C_row1;
//             1: C_data_in = C_row2;
//             2: C_data_in = C_row3;
//             3: C_data_in = C_row4;
//             default: C_data_in = 0;
//         endcase
//     end

//     wire signed [7:0] A_in1 = A_in_reg1;
//     wire signed [7:0] A_in2 = A_in_reg2[0];
//     wire signed [7:0] A_in3 = A_in_reg3[0];
//     wire signed [7:0] A_in4 = A_in_reg4[0];

//     wire signed [7:0] B_in1 = B_in_reg1;
//     wire signed [7:0] B_in2 = B_in_reg2[0];
//     wire signed [7:0] B_in3 = B_in_reg3[0];
//     wire signed [7:0] B_in4 = B_in_reg4[0];

//     SystolicArray4x4 systolic1 (
//         .clk(clk),
//         .rst_n(rst_n),
//         .pe_reset(current_state == WRITE && next_state == CALC || current_state == DONE && next_state == IDLE),
//         .A_in1(A_in1),
//         .A_in2(A_in2),
//         .A_in3(A_in3),
//         .A_in4(A_in4),
//         .B_in1(B_in1),
//         .B_in2(B_in2),
//         .B_in3(B_in3),
//         .B_in4(B_in4),
//         .C_row1(C_row1),
//         .C_row2(C_row2),
//         .C_row3(C_row3),
//         .C_row4(C_row4),
//         .input_offset(input_offset)
//     );

// endmodule

// //=====================================================================
// // SYSTOLIC ARRAY
// //=====================================================================
// module SystolicArray4x4 (
//     input  clk,
//     input  rst_n,
//     input  pe_reset,
//     input  signed [7:0] A_in1,
//     input  signed [7:0] A_in2,
//     input  signed [7:0] A_in3,
//     input  signed [7:0] A_in4,
//     input  signed [7:0] B_in1,
//     input  signed [7:0] B_in2,
//     input  signed [7:0] B_in3,
//     input  signed [7:0] B_in4,
//     output wire [127:0] C_row1,
//     output wire [127:0] C_row2,
//     output wire [127:0] C_row3,
//     output wire [127:0] C_row4,

//     input [31:0] input_offset
// );

//     wire signed [7:0] right_wire [0:3][0:3];
//     wire signed [7:0] down_wire  [0:3][0:3];
//     wire signed [31:0] C_out     [0:3][0:3];

//     wire signed [7:0] A_in [0:3];
//     wire signed [7:0] B_in [0:3];

//     assign A_in[0] = A_in1;
//     assign A_in[1] = A_in2;
//     assign A_in[2] = A_in3;
//     assign A_in[3] = A_in4;

//     assign B_in[0] = B_in1;
//     assign B_in[1] = B_in2;
//     assign B_in[2] = B_in3;
//     assign B_in[3] = B_in4;

//     genvar i, j;
//     generate
//         for (i = 0; i < 4; i = i + 1) begin : ROW
//             for (j = 0; j < 4; j = j + 1) begin : COL
//                 if (i == 0 && j == 0) begin
//                     PE u_pe (
//                         .clk(clk), .rst_n(rst_n), .pe_reset(pe_reset),
//                         .in_up(B_in[j]), .in_left(A_in[i]),
//                         .out_right(right_wire[i][j]),
//                         .out_down(down_wire[i][j]),
//                         .result(C_out[i][j]),
//                         .input_offset(input_offset)
//                     );
//                 end
//                 else if (i == 0) begin
//                     PE u_pe (
//                         .clk(clk), .rst_n(rst_n), .pe_reset(pe_reset),
//                         .in_up(B_in[j]), .in_left(right_wire[i][j-1]),
//                         .out_right(right_wire[i][j]),
//                         .out_down(down_wire[i][j]),
//                         .result(C_out[i][j]),
//                         .input_offset(input_offset)
//                     );
//                 end
//                 else if (j == 0) begin
//                     PE u_pe (
//                         .clk(clk), .rst_n(rst_n), .pe_reset(pe_reset),
//                         .in_up(down_wire[i-1][j]), .in_left(A_in[i]),
//                         .out_right(right_wire[i][j]),
//                         .out_down(down_wire[i][j]),
//                         .result(C_out[i][j]),
//                         .input_offset(input_offset)
//                     );
//                 end
//                 else begin
//                     PE u_pe (
//                         .clk(clk), .rst_n(rst_n), .pe_reset(pe_reset),
//                         .in_up(down_wire[i-1][j]), .in_left(right_wire[i][j-1]),
//                         .out_right(right_wire[i][j]),
//                         .out_down(down_wire[i][j]),
//                         .result(C_out[i][j]),
//                         .input_offset(input_offset)
//                     );
//                 end
//             end
//         end
//     endgenerate

//     assign C_row1 = {C_out[0][0], C_out[0][1], C_out[0][2], C_out[0][3]};
//     assign C_row2 = {C_out[1][0], C_out[1][1], C_out[1][2], C_out[1][3]};
//     assign C_row3 = {C_out[2][0], C_out[2][1], C_out[2][2], C_out[2][3]};
//     assign C_row4 = {C_out[3][0], C_out[3][1], C_out[3][2], C_out[3][3]};

// endmodule


// //=====================================================================
// // PROCESSING ELEMENT
// //=====================================================================
// module PE(
//     input clk, 
//     input rst_n,
//     input pe_reset,
//     input  signed [7:0] in_up, 
//     input  signed [7:0] in_left, 
//     output reg signed [7:0] out_right, 
//     output reg signed [7:0] out_down, 
//     output reg signed [31:0] result,
//     input [31:0] input_offset
// );
//     wire signed [31:0] product;

//     wire signed [8:0] in_left_offset = in_left + $signed(input_offset[8:0]); 
//     wire signed [8:0] in_up_ext = in_up;

//     assign product = in_up_ext * in_left_offset;

//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             result    <= 0;
//             out_right <= 0;
//             out_down  <= 0;
//         end
//         else if (pe_reset) begin
//             result    <= 0;
//             out_right <= 0;
//             out_down  <= 0;
//         end
//         else begin
//             result    <= result + product;
//             out_right <= in_left;
//             out_down  <= in_up;
//         end
//     end
// endmodule
