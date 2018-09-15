//Made by Daniel Crawford on 3/17/2018(dsc160130@utdallas.edu)
//Section 3377.501
#ifndef PROGRAM4_H
#define PROGRAM4_H

#include <stdio.h>
#include "parse.tab.h"
#include <string.h>

int yylex(void);
extern char* yytext;
extern int yyparse(void);
extern int yylex(void);

#endif /* PROGRAM4_H */
