//Made by Daniel Crawford on 3/17/2018(dsc160130)
//Section 3377.501
#include "program4.h"
#include <stdio.h>

int main(int argc, char* argv[]) {

  if (strcmp(argv[0],"./parser") == 0) {
    //If yyparse somehow does not return 0, report it
    printf("Operating in parse mode\n\n");

    int returnval;

    returnval = yyparse();

    if (returnval == 0)
      printf("Parsing Successful!\n");
    else if (returnval == 1)
      printf("Parsing failure: invalid input\n");
    else if (returnval == 2)
      printf("Parsing failure: memory exhaustion\n");
    else
      printf("Unknown return value from yyparse()");
    
  }
  else if (strcmp(argv[0],"./scanner") == 0) {
    printf("Operating in scan mode\n\n");
    
    int token;
    
    //loop will display the type of token just displayed, then its value    
    while ( (token = yylex()) != 0 ) {
      
      //tokenstr holds metainformation from the token
      char* tokenstr;
      
      switch(token) {
	
      case NAMETOKEN:
	tokenstr = "NAMETOKEN";
	break;
      case IDENTIFIERTOKEN:
	tokenstr = "IDENTIFIERTOKEN";
	break;
      case NAME_INITIAL_TOKEN:
	tokenstr = "NAME_INITIAL_TOKEN";
	break;
      case ROMANTOKEN:
	tokenstr = "ROMANTOKEN";
	break;
      case SRTOKEN:
	tokenstr = "SRTOKEN";
	break;
      case JRTOKEN:
	tokenstr = "JRTOKEN";
	break;
      case EOLTOKEN:
	tokenstr = "EOLTOKEN";
	break;
      case INTTOKEN:
	tokenstr = "INTTOKEN";
	break;
      case COMMATOKEN:
	tokenstr = "COMMATOKEN";
	break;
      case DASHTOKEN:
	tokenstr = "DASHTOKEN";
	break;
      case HASHTOKEN:
	tokenstr = "HASHTOKEN";
	break;
      default:
	tokenstr = "SYNTAX ERROR";
	break;
	
      }
      
      //Display the name if the token matches a name token, otherwise print the value of the token
      if (token == NAMETOKEN || token == NAME_INITIAL_TOKEN)
	printf("yylex returned %s token (%s)\n",tokenstr,yytext);
      else
	printf("yylex returned %s token (%d)\n",tokenstr, token);
    }
    
  }
  else {
    printf("error: Invalid argument \"%s\", expected ./scanner or ./parser\n", argv[0]);
  }
  
  return 0;
}
