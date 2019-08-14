/*Made by Daniel Crawford on 3/17/2018(dsc160130@utdallas.edu)
 *Section 3377.501
 */
%define parse.error verbose

%union {
  char* string;
  int ival;
}
   
%{
  #include "program4.h"
  #include <stdlib.h>
  #include <string.h>

  //set yydebug to a nonzero value to be given debug information
  int yydebug = 0;
  extern int yylex();
  extern char* yytext;
  extern void yyerror(const char*);
  extern YYSTYPE yylval;
%}

%token <string> NAMETOKEN IDENTIFIERTOKEN NAME_INITIAL_TOKEN ROMANTOKEN SRTOKEN
           JRTOKEN EOLTOKEN COMMATOKEN DASHTOKEN HASHTOKEN

%token <ival> INTTOKEN

%type <string> postal_addresses address_block name_part personal_part last_name suffix_part
          street_address street_number street_name location_part town_name state_code 


//Start program at RUN
%start RUN

%%
 /* Run through the grammars and print the XML format of the tokens metainformation once
    it determines what information it represents..
  */
RUN              : postal_addresses
                 ;

postal_addresses : address_block EOLTOKEN postal_addresses       
                 | address_block                                  { printf("\n"); }
                 ; 

address_block    : name_part street_address location_part         { fprintf(stderr,"\n"); }
                 ;
		 
//If unsuccessful, skip to newline and notify user where it failed6
name_part        : personal_part last_name suffix_part EOLTOKEN 
                 | personal_part last_name EOLTOKEN             
                 | error EOLTOKEN                                 { printf("Bad name_part ... skipping to newline\n"); yyerrok; }
                 ;

personal_part    : NAMETOKEN                                      { fprintf(stderr,"<FirstName>%s</FirstName>\n",$1); }
                 | NAME_INITIAL_TOKEN                             { fprintf(stderr,"<FirstName>%s</FirstName>\n",$1); }
                 ;

last_name        : NAMETOKEN                                      { fprintf(stderr,"<LastName>%s</LastName>\n",$1); }
                 ;

suffix_part      : SRTOKEN                                        { fprintf(stderr,"<Suffix>%s</Suffix>\n",$1); } 
                 | JRTOKEN                                        { fprintf(stderr,"<Suffix>%s</Suffix>\n",$1); }
                 | ROMANTOKEN                                     { fprintf(stderr,"<Suffix>%s</Suffix>\n",$1); }
                 ;
		 
//If unsuccessful, skip to newline and notify user where it failed
street_address   : street_number street_name INTTOKEN EOLTOKEN    { fprintf(stderr,"<AptNum>%d</AptNum>\n",$3); }
                 | street_number street_name HASHTOKEN INTTOKEN EOLTOKEN
		                                                  { fprintf(stderr,"<AptNum>%d</AptNum\n",$4); }
                 | street_number street_name EOLTOKEN
                 | error EOLTOKEN                                 { printf("Bad street_address ... skipping to newline\n"); yyerrok; }
                 ;

street_number    : INTTOKEN                                       { fprintf(stderr,"<HouseNumber>%d</HouseNumber>\n",$1); }
                 | IDENTIFIERTOKEN                                { fprintf(stderr,"<HouseNumber>%s</HouseNumber>\n",$1); }
                 ;

street_name      : NAMETOKEN                                      { fprintf(stderr,"<StreetName>%s</StreetName>\n",$1); }
                 ;
		 
//If unsuccessful, skip to newline and notify user where it failed
location_part    : town_name COMMATOKEN state_code zip_code EOLTOKEN
                 | error EOLTOKEN                                 { printf("Bad location_part ... skipping to newline\n"); yyerrok; }
                 ;

town_name        : NAMETOKEN                                      { fprintf(stderr,"<City>%s</City>\n",$1); }
                 ;

state_code       : NAMETOKEN                                      { fprintf(stderr,"<State>%s</State>\n",$1); }
                 ;

zip_code         : INTTOKEN DASHTOKEN INTTOKEN                    { fprintf(stderr,"<Zip5>%d</Zip5>\n<Zip4>%d</Zip4>\n",$1,$3); }
                 | INTTOKEN                                       { fprintf(stderr,"<Zip5>%d</Zip5>\n",$1); }
                 ;


%%
//Just have this here so bison doesn't flip when an error occurs
void yyerror(const char* s) {}
