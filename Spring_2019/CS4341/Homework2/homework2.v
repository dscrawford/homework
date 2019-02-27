/*
module Add_half (input a, b,  output c_out, sum);
   xor G1(sum, a, b);       // Gate instance names are optional
   and G2(c_out, a, b);
endmodule // Add_half

module Add_full (input a, b, c_in, output c_out, sum);       // See Fig. 4.8
   wire w1, w2, w3;                   // w1 is c_out; w2 is sum
   Add_half M1 (a, b, w1, w2);
   Add_half M0 (w2, c_in, w3, sum);
   or (c_out, w1, w3);

endmodule // Add_full

module Add_rca_4 (input [3:0] a, b, input c_in, output c_out, output[3:0] sum);
   wire c_in1, c_in2, c_in3, c_in4;
               // Intermediate carries
   Add_full M0 (a[0], b[0], c_in,   c_in1, sum[0]);
   Add_full M1 (a[1], b[1], c_in1, c_in2, sum[1]);
   Add_full M2 (a[2], b[2], c_in2, c_in3, sum[2]);
   Add_full M3 (a[3], b[3], c_in3, c_out, sum[3]);

endmodule // Add_rca_4

module Add_rca_8 (input [7:0] a, b, input c_in, output c_out, output [7:0] sum);
  wire c_in4;
   Add_rca_4 M0 (a[3:0], b[3:0], c_in, c_in4, sum[3:0]);
   Add_rca_4 M1 (a[7:4], b[7:4], c_in4, c_out, sum[7:4]);

endmodule // Add_rca_8
*/

module Add_half (input a, b, output c_out, sum);
   xor G1(sum, a, b);    // Gate instance names are optional
   and G2(c_out, a, b);
endmodule

module Add_full (input a, b, m, c_in, output c_out, sum);    // See Fig. 4.8
   wire w1, w2, w3;                // w1 is c_out; w2 is sum
   Add_half M1 (a, b ^ m, w1, w2);
   Add_half M0 (w2, c_in, w3, sum);
   or (c_out, w1, w3);
endmodule // Add_full

module Operate_4bit(input[3:0] a, b, input m, output c_out, output [3:0] res);
   wire c_in1, c_in2, c_in3;

   Add_full M0 (a[0], b[0], m, m,  c_in1, res[0]);
   Add_full M1 (a[1], b[1], m, c_in1, c_in2, res[1]);
   Add_full M2 (a[2], b[2], m, c_in2, c_in3, res[2]);
   Add_full M3 (a[3], b[3], m, c_in3, c_out, res[3]);

endmodule // Operate_4bit

module test_bench ;
   reg [0:3] A;
   reg [0:3] B;
   reg 	     M;
   wire      c_out;
   wire [3:0] res;
   integer    i;
   
   Operate_4bit M0(A,B,M,c_out,res);

   initial begin
      A[0]=0; A[1] = 1; A[2] = 1; A[3] = 0;
      B[0]=1; B[1] = 0; B[2] = 0; B[3] = 0;
      M = 0;

      for (i = 0; i < 4; ++i) begin
	 $write("%b", A[i]);
      end
      $display("");
      for (i = 0; i < 4; ++i) begin
	 $write("%b", B[i]);
      end
      $display("");
      #1
      $display("carry:%b", c_out);
      for (i = 3; i >= 0; --i)
	 $write("%b",res[i]);
      $display("");
   end
/*
   reg [7:0] num1;
   reg [7:0] num2;
   reg 	     c_in;
   wire      c_out;
   wire [7:0] res;
   integer    i;

   Add_rca_8 Add_rca_8(num1,num2,c_in,cout,res);

   initial begin
      for (i = 0; i < 8; ++i) begin
	 num1[i] = 0;
	 num2[i] = 1;
      end
      num1[4] = 1;
      num2[4] = 1;

      c_in = 0;
      
      #4
	for (i = 7; i >= 0; --i) begin
	   $write("%b", num1[i]);
	end
      $display("");
      	for (i = 7; i >= 0; --i) begin
	   $write("%b", num2[i]);
	end
      $display("");
      	for (i = 7; i >= 0; --i) begin
	   $write("%b", res[i]);
	end
   end

   initial begin
      #10 $finish;
   end */
endmodule // test_bech
