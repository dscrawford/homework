/*Made by Daniel Crawford on 3/17/2018(dsc160130@utdallas.edu)
 *Section 3377.501
 */
%option noyywrap noinput nounput

%{
  #include "program4.h"
  #include "parse.tab.h"

  extern YYSTYPE yylval;
%}

%%


[ \t\r]+                                                  //ignore whitespace
\n                                                        {return EOLTOKEN;}
","                                                       {return COMMATOKEN;}
"-"                                                       {return DASHTOKEN;}
"#"                                                       {return HASHTOKEN;}
"Sr."                                                     {yylval.string = strdup(yytext); return SRTOKEN;}
"Jr."                                                     {yylval.string = strdup(yytext); return JRTOKEN;}
M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})    {yylval.string = strdup(yytext); return ROMANTOKEN;}
[a-zA-Z]{2}[a-zA-Z]*                                      {yylval.string= strdup(yytext); return NAMETOKEN;}
[0-9]+                                                    {yylval.ival = atoi(yytext); return INTTOKEN;}
([a-zA-Z])|([a-zA-Z]\.)                                   {yylval.string = strdup(yytext); return NAME_INITIAL_TOKEN;}
^[a-zA-Z0-9][a-zA-Z0-9]+                                  {yylval.string = strdup(yytext); return IDENTIFIERTOKEN;}
.                                                         {return *yytext; }

%%

//leave empty, code will be used in other files
