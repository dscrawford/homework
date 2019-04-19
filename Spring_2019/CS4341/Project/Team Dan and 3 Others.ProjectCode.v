//Compiled using Iverilog
//Made by Team Dan and 3 Others

//Fulladder simply returns result of two bits + carry sum
module FullAdder_1bit(input A, B, cin, 
		      output S, cout);
   //S = (A xor B) xor cin
   xor M0(S, A ^ B, cin);
   //cout = (A & B) | (cin & (A xor B))
   xor M1(cout, A && B, cin && (A ^ B));
endmodule // FullAdder_1

//Using out FullAdder 4 times
//Like other FullAdder, but 4 bits instead.
module FullAdder_4bit(input [3:0] A, B, input cin,
		      output [3:0] sum, output cout);
   wire 			   cin1,cin2,cin3;
   FullAdder_1bit M0(A[0], B[0], cin, sum[0], cin1);
   FullAdder_1bit M1(A[1], B[1], cin1,sum[1], cin2);
   FullAdder_1bit M2(A[2], B[2], cin2,sum[2], cin3);
   FullAdder_1bit M3(A[3], B[3], cin3,sum[3], cout);
endmodule // FullAdder_4bit

//Using our FullAdder 8 times
//8 bit FullAdder
module FullAdder_8bit(input [7:0] A, B, input cin,
		      output [7:0] sum, output cout);
   wire 			   cin1;
   FullAdder_4bit M0(A[3:0], B[3:0], cin, sum[3:0], cin1);
   FullAdder_4bit M1(A[7:4], B[7:4], cin1,sum[7:4], cout);
endmodule // FullAdder_8bit

module FullAdder_16bit(input [15:0] A, B, input cin,
		       output [15:0] sum, output cout);
   wire 			     cin1;
   FullAdder_8bit M0(A[7:0], B[7:0], cin, sum[7:0], cin1);
   FullAdder_8bit M1(A[15:8], B[15:8], cin1, sum[15:8], cout);
endmodule // FullAdder_16bit


module Mux_16to1(input [3:0] op, input [15:0][15:0] a,
		 output[15:0] res);
   reg[15:0] res;
   always @(*)
     begin
	//0000
	if (~op[3] && ~op[2] && ~op[1] && ~op[0])
	  res <= a[0];
	//0001
	else if (~op[3] && ~op[2] && ~op[1] && op[0])
	  res <= a[1];
	//0010
	else if (~op[3] && ~op[2] && op[1] && ~op[0])
	  res <= a[2];
	//0011
	else if (~op[3] && ~op[2] && op[1] && op[0])
	  res <= a[3];
	//0100
	else if (~op[3] && op[2] && ~op[1] && ~op[0])
	  res <= a[4];
	//0101
	else if (~op[3] && op[2] && ~op[1] && op[0])
	  res <= a[5];
	//0110
	else if (~op[3] && op[2] && op[1] && ~op[0])
	  res <= a[6];
	//0111
	else if (~op[3] && op[2] && op[1] && op[0])
	  res <= a[7];
	//1000
	else if (op[3] && ~op[2] && ~op[1] && ~op[0])
	  res <= a[8];
	//1001
	else if (op[3] && ~op[2] && ~op[1] && op[0])
	  res <= a[9];
	//1010
	else if (op[3] && ~op[2] && op[1] && ~op[0])
	  res <= a[10];
	//1011
	else if (op[3] && ~op[2] && op[1] && op[0])
	  res <= a[11];
	//1100
	else if (op[3] && op[2] && ~op[1] && ~op[0])
	  res <= a[12];
	//1101
	else if (op[3] && op[2] && ~op[1] && op[0])
	  res <= a[13];
	//1110
	else if (op[3] && op[2] && op[1] && ~op[0])
	  res <= a[14];
	//1111
	else if (op[3] && op[2] && op[1] && op[0])
	  res <= a[15];	  
     end
endmodule // Opcode_Mux

module And_4bit(output [3:0] S, input [3:0] A, B);
   and(S[0],A[0],B[0]);
   and(S[1],A[1],B[1]);
   and(S[2],A[2],B[2]);
   and(S[3],A[3],B[3]);
endmodule // And_4bit

module And_16bit(output [15:0] S, input [15:0] A, B);
   And_4bit M0(S[3:0],A[3:0],B[3:0]);
   And_4bit M1(S[7:4],A[7:4],B[7:4]);
   And_4bit M2(S[11:8], A[11:8], B[11:8]);
   And_4bit M3(S[15:12], A[15:12], B[15:12]);
endmodule // And_16bit
       
module Or_4bit(output [3:0] S, input [3:0] A, B);   
   or(S[0],A[0],B[0]);
   or(S[1],A[1],B[1]);
   or(S[2],A[2],B[2]);
   or(S[3],A[3],B[3]);
endmodule // Or_4bit

module Or_16bit(output [15:0] S, input [15:0] A, B);
   Or_4bit M0(S[3:0],A[3:0],B[3:0]);
   Or_4bit M1(S[7:4],A[7:4],B[7:4]);
   Or_4bit M2(S[11:8], A[11:8], B[11:8]);
   Or_4bit M3(S[15:12], A[15:12], B[15:12]);
endmodule // Or_16bit
 
module Xor_4bit(output [3:0] S, input [3:0] A, B);
   xor(S[0],A[0],B[0]);
   xor(S[1],A[1],B[1]);
   xor(S[2],A[2],B[2]);
   xor(S[3],A[3],B[3]);
endmodule // Xor_4bit

module Xor_16bit(output [15:0] S, input [15:0] A, B);
   Xor_4bit M0(S[3:0],A[3:0],B[3:0]);
   Xor_4bit M1(S[7:4],A[7:4],B[7:4]);
   Xor_4bit M2(S[11:8], A[11:8], B[11:8]);
   Xor_4bit M3(S[15:12], A[15:12], B[15:12]);
endmodule // Xor_16bit

module Nand_4bit(output [3:0] S, input [3:0] A, B);
   nand(S[0],A[0],B[0]);
   nand(S[1],A[1],B[1]);
   nand(S[2],A[2],B[2]);
   nand(S[3],A[3],B[3]);
endmodule // Xor_4bit

module Nand_16bit(output [15:0] S, input [15:0] A, B);
   Nand_4bit M0(S[3:0],A[3:0],B[3:0]);
   Nand_4bit M1(S[7:4],A[7:4],B[7:4]);
   Nand_4bit M2(S[11:8], A[11:8], B[11:8]);
   Nand_4bit M3(S[15:12], A[15:12], B[15:12]);
endmodule // Xor_16bit

module Nor_4bit(output [3:0] S, input [3:0] A, B);   
   nor(S[0],A[0],B[0]);
   nor(S[1],A[1],B[1]);
   nor(S[2],A[2],B[2]);
   nor(S[3],A[3],B[3]);
endmodule // Or_4bit

module Nor_16bit(output [15:0] S, input [15:0] A, B);
   Nor_4bit M0(S[3:0],A[3:0],B[3:0]);
   Nor_4bit M1(S[7:4],A[7:4],B[7:4]);
   Nor_4bit M2(S[11:8], A[11:8], B[11:8]);
   Nor_4bit M3(S[15:12], A[15:12], B[15:12]);
endmodule // Or_16bit

module D_FlipFlop(input d, clk, output q);
   reg q, res;

   always @(*)
     begin
	res = ~d;
	if (res == 1) //If not at reset
	  q = 0;
	else
	  q = d;
     end
   
endmodule // D_FlipFlop

module Shift_Left(output [15:0] S, input [15:0] A);
   D_FlipFlop M0(1'b0,1'b1, S[0]);
   D_FlipFlop M1(A[0],1'b1, S[1]);
   D_FlipFlop M2(A[1],1'b1, S[2]);
   D_FlipFlop M3(A[2],1'b1, S[3]);
   D_FlipFlop M4(A[3],1'b1, S[4]);
   D_FlipFlop M5(A[4],1'b1, S[5]);
   D_FlipFlop M6(A[5],1'b1, S[6]);
   D_FlipFlop M7(A[6],1'b1, S[7]);
   D_FlipFlop M8(A[7],1'b1, S[8]);
   D_FlipFlop M9(A[8],1'b1, S[9]);
   D_FlipFlop M10(A[9],1'b1, S[10]);
   D_FlipFlop M11(A[10],1'b1,S[11]);
   D_FlipFlop M12(A[11],1'b1,S[12]);
   D_FlipFlop M13(A[12],1'b1,S[13]);
   D_FlipFlop M14(A[13],1'b1,S[14]);
   D_FlipFlop M15(A[14],1'b1,S[15]);
endmodule // Shift_Right


module Shift_Right(output [15:0] S, input [15:0] A);
   D_FlipFlop M0(1'b0,1'b1, S[15]);
   D_FlipFlop M1(A[15],1'b1, S[14]);
   D_FlipFlop M2(A[14],1'b1, S[13]);
   D_FlipFlop M3(A[13],1'b1, S[12]);
   D_FlipFlop M4(A[12],1'b1, S[11]);
   D_FlipFlop M5(A[11],1'b1, S[10]);
   D_FlipFlop M6(A[10],1'b1, S[9]);
   D_FlipFlop M7(A[9],1'b1, S[8]);
   D_FlipFlop M8(A[8],1'b1, S[7]);
   D_FlipFlop M9(A[7],1'b1, S[6]);
   D_FlipFlop M10(A[6],1'b1, S[5]);
   D_FlipFlop M11(A[5],1'b1,S[4]);
   D_FlipFlop M12(A[4],1'b1,S[3]);
   D_FlipFlop M13(A[3],1'b1,S[2]);
   D_FlipFlop M14(A[2],1'b1,S[1]);
   D_FlipFlop M15(A[1],1'b1,S[0]);
endmodule // Shift_Left

module ALU(input [15:0] A, B, input [3:0] op, output cout,
	   output [15:0] S);
   wire [15:0][15:0] C;
   //cout is an empty variable, its not used.
   wire 	    cout;

   //0: Add the two numbers A and B
   FullAdder_16bit Adder(A, B, 1'b0, C[0], cout);
   //1: Subtract the number B from A. (ignore cout)
   FullAdder_16bit Subtractor(A, ~B, 1'b1, C[1], );
   //2: SHIFT LEFT
   Shift_Left LeftShifter(C[2],A);
   //3: SHIFT RIGHT
   Shift_Right RightShifter(C[3],A);
   //4: AND TOGETHER
   And_16bit Ander(C[4],A,B);
   //5: OR TOGETHER
   Or_16bit Orer(C[5],A,B);
   //6: XOR TOGETHER
   Xor_16bit Xorer(C[6],A,B);
   //7: NOT A
   assign C[7] = ~A;
   //8: NAND TOGETHER
   Nand_16bit Nander(C[8],A,B);
   //9: NOR TOGETHER
   Nor_16bit Norer(C[9],A,B);
   
   Mux_16to1 Mux(op, C, S);   
endmodule // ALU


module test_bench ;
   reg [15:0]  A,B;
   reg [3:0]   op;
   wire	       cout;
   wire [15:0] S;
      
   ALU M0(A,B,op,cout,S);
   
   initial begin
      $display("output: ({carry}) {binary no.}");

      op = 4'b0000;
      $display();

      A  <= 16'b1111111111111111;
      B  <= 16'b0101010101010101;
      #1;
      $display("A + B: %b + %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b0010001000011100;
      B  <= 16'b0010100100010100;
      #1;
      $display("A + B: %b + %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b1000011000010011;
      B  <= 16'b0010100100100100;
      #1;
      $display("A + B: %b + %b", A, B);
      $display("(%b)%b",cout, S);
      
      //Now displaying Subtraction
      op = 4'b0001;
      $display();

      A  <= 16'b1000000000010011;
      B  <= 16'b0010100100100100;
      #1;
      $display("A - B: %b - %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b1000000000000000;
      B  <= 16'b1000000000000000;
      #1;
      $display("A - B: %b - %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b0000000000000001;
      B  <= 16'b0000000000000100;
      #1;
      $display("A - B: %b - %b", A, B);
      $display("(%b)%b",cout, S);

      //Now displaying Shift left
      op = 4'b010;
      $display();
      
      A  <= 16'b0010101010101010;
      #1;      
      $display("A: %b -> shift left", A);
      $display("(%b)%b",cout, S);
      
      //Now displaying Shift right
      op = 4'b011;
      $display();
      
      A  <= 16'b1010101010101010;
      #1;      
      $display("A: %b -> shift right", A);
      $display("(%b)%b",cout, S);

      
      //Now displaying And
      op = 4'b0100;
      $display();
      
      A  <= 16'b1010101010101010;
      B  <= 16'b0101010101010101;
      #1;      
      $display("A & B: %b & %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b1010101010101010;
      B  <= 16'b1001111111111111;      
      #1;      
      $display("A & B: %b & %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b1111111111111111;
      B  <= 16'b0000000000000001;      
      #1;      
      $display("A & B: %b & %b", A, B);
      $display("(%b)%b",cout, S);


      op = 4'b0101;
      //Now displaying or
      $display();
      
      A  <= 16'b0101010101010101;
      B  <= 16'b1010101010101010;
      #1;      
      $display("A | B: %b | %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b1111000001111111;
      B  <= 16'b0011001111100001;
      #1;      
      $display("A | B: %b | %b", A, B);
      $display("(%b)%b",cout, S);
      
      A  <= 16'b0000000000000000;
      B  <= 16'b0010000000010001;
      #1;      
      $display("A | B: %b | %b", A, B);
      $display("(%b)%b",cout, S);

      
      op = 4'b0110;
      //Now displaying xor
      $display();
      
      A  <= 16'b0101010101010101;
      B  <= 16'b1010101010101010;
      #1;      
      $display("A xor B: %b xor %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b1111111111111111;
      B  <= 16'b0000000000000000;
      #1;      
      $display("A xor B: %b xor %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b0111000011110000;
      B  <= 16'b0111001111000011;
      #1;      
      $display("A xor B: %b xor %b", A, B);
      $display("(%b)%b",cout, S);


      op = 4'b0111;
      //Now displaying Not A
      $display();
      
      A  <= 16'b1111111111111111;      
      #1;
      $display("A: %b --> Not", A);
      $display("(%b)%b",cout, S);

      A  <= 16'b0000000001111111;      
      #1;
      $display("A: %b --> Not", A);
      $display("(%b)%b",cout, S);

      A  <= 16'b1010101010101010;      
      #1;
      $display("A: %b --> Not", A);
      $display("(%b)%b",cout, S);
      
      op = 4'b1000;
      //Now displaying NAND
      $display();

      A  <= 16'b1111111111111111;
      B  <= 16'b0000011001100000;
      #1;      
      $display("A nand B: %b nand %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b0101010101010010;
      B  <= 16'b1100110011001100;
      #1;      
      $display("A nand B: %b nand %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b0000000000000000;
      B  <= 16'b0000000000000111;
      #1;      
      $display("A nand B: %b nand %b", A, B);
      $display("(%b)%b",cout, S);

      op = 4'b1001;
      //Now displaying NOR
      $display();
      
      A  <= 16'b0000000000000000;
      B  <= 16'b0000000000000000;
      #1;      
      $display("A nor B: %b nor %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b0000000001111111;
      B  <= 16'b1001000001000001;
      #1;      
      $display("A nor B: %b nor %b", A, B);
      $display("(%b)%b",cout, S);

      A  <= 16'b1111111111111111;
      B  <= 16'b1010101010101010;
      #1;      
      $display("A nor B: %b nor %b", A, B);
      $display("(%b)%b",cout, S);
      
   end
endmodule
