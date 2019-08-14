//Made for iverilog on 02/27/18 by Team Dan and 3 others
//To change the output puts, go to testbench and locate arrays A and B and Mode M. Change their values to your desire.

//Following two modules made by Dr.Becker
module Add_half (input a, b, output c_out, sum);
   xor G1(sum, a, b);    // Gate instance names are optional
   and G2(c_out, a, b);
endmodule

module Add_full (input a, b, c_in, output c_out, sum);    // See Fig. 4.8
   wire w1, w2, w3;                // w1 is c_out; w2 is sum
   Add_half M1 (a, b, w1, w2);
   Add_half M0 (w2, c_in, w3, sum);
   or (c_out, w1, w3);
endmodule // Add_full

module Operate_4bit(input[3:0] a, b, input m, output c_out, output [3:0] res);
   wire c_in1, c_in2, c_in3;

   Add_full M0 (a[0], b[0] ^ m, m,  c_in1, res[0]);
   Add_full M1 (a[1], b[1] ^ m, c_in1, c_in2, res[1]);
   Add_full M2 (a[2], b[2] ^ m, c_in2, c_in3, res[2]);
   Add_full M3 (a[3], b[3] ^ m, c_in3, c_out, res[3]);

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
      A[0]=1; A[1] = 0; A[2] = 0; A[3] = 0;
      B[0]=0; B[1] = 0; B[2] = 0; B[3] = 1;
      M = 1;

      $write("A:      " );
      for (i = 0; i < 4; ++i) begin
	 $write("%b", A[i]);
      end
      $display("");
      
      $write("B:      ");
      for (i = 0; i < 4; ++i) begin
	 $write("%b", B[i]);
      end
      $display("");
      #1
	
      $display("Carry:  %b", c_out);
      $write("Result: ");
      for (i = 3; i >= 0; --i)
	 $write("%b",res[i]);
      $display("");
      
   end
   
endmodule // test_bech
