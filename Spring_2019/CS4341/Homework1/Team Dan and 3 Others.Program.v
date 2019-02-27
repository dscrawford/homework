//Using iverilog from http://iverilog.icarus.com/
//Made by Daniel for Team Dan and 3 Others on Feb 7, 2019

module Out_1(A,B,C,D,out1);
   input A;
   input B;
   input C;
   input D;
   output out1;
   reg 	  out1;
   

   always @(*) begin
      out1 = (A || (!B)) && (!C) && (C || D);
   end
endmodule // Out_1

module Out_2(A,B,C,D,out2);
   input A;
   input B;
   input C;
   input D;
   output out2;
   reg 	  out2;
   

   always @(*) begin
      out2 = ( ( (!C) && D) || (B && C && D) || ( C && (!D))) && ( (!A) || B );
   end
endmodule // Out_2

module Out_3(A,B,C,D,out3);
   input A;
   input B;
   input C;
   input D;
   output out3;
   reg 	  out3;
   

   always @(*) begin
      out3 = (A && B || C) && D || (!B) && C;      
   end
endmodule // Out_3



module Test_bench ;
   reg A;
   reg B;
   reg C;
   reg D;
   wire out1;
   wire out2;
   wire out3;
   integer count;

   Out_1 output1(A,B,C,D,out1);
   Out_2 output2(A,B,C,D,out2);
   Out_3 output3(A,B,C,D,out3);

   initial begin
      $display("          #|A|B|C|D||1|2|3|");
      $display("===========+=+=+=+=++=+=+=+");

      count = 0;
      #1 $display("%d|%b|%b|%b|%b||%b|%b|%b|",count, A, B, C, D, out1, out2, out3);
      forever begin
	 count = count + 1;
	 #2 $display("%d|%b|%b|%b|%b||%b|%b|%b|",count, A, B, C, D, out1, out2, out3);
      end

   end

   initial begin
      A = 0;
      #16 A = 1;
   end

   initial begin
      B = 0;
      forever begin
	 #8 B = 1;
	 #8 B = 0;
      end
   end

   initial begin
      C = 0;
      forever begin
	 #4 C = 1;
	 #4 C = 0;
      end
   end

   initial begin;
      D = 0;
      forever begin
	 #2 D = 1;
	 #2 D = 0;
      end
   end

   initial begin;
      #32 $finish;   
   end
   
endmodule
