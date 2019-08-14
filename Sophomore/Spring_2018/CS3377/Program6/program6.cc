//Made by Daniel Crawford on 4/19/2018(dsc160130@utdallas.edu)
//CS 3377.501
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <sstream>
#include <iomanip>
#include <string>
#include "cdk.h"


#define MATRIX_WIDTH 3
#define MATRIX_HEIGHT 5
#define BOX_WIDTH 20
#define MATRIX_NAME_STRING "Binary File Contents"

using namespace std;

class BinaryHeaderFile{
public:
  uint32_t magicNumber;
  uint32_t versionNumber;
  uint64_t numRecords;
};

const int maxRecordStringLength = 25;

class BinaryFileRecord {
public:
  uint8_t strLength;
  char    stringBuffer[maxRecordStringLength];
};

void findData(CDKMATRIX *);


int main()
{

  WINDOW	*window;
  CDKSCREEN	*cdkscreen;
  CDKMATRIX     *myMatrix;           // CDK Screen Matrix

  const char   *rowTitles[MATRIX_HEIGHT+1] = {"", "a", "b", "c","d","e"};
  const char   *columnTitles[MATRIX_WIDTH+1] = {"", "a", "b", "c"};
  int		boxWidths[MATRIX_WIDTH+1] = {BOX_WIDTH, BOX_WIDTH, BOX_WIDTH,
					     BOX_WIDTH};
  int		boxTypes[MATRIX_WIDTH+1] = {vMIXED, vMIXED, vMIXED, vMIXED};

  /*
   * Initialize the Cdk screen.
   *
   * Make sure the putty terminal is large enough
   */
  window = initscr();
  cdkscreen = initCDKScreen(window);

  /* Start CDK Colors */

  initCDKColor();

  /*
   * Create the matrix.  Need to manually cast (const char**) to (char **)
  */
  myMatrix = newCDKMatrix(cdkscreen, CENTER, CENTER, MATRIX_HEIGHT, MATRIX_WIDTH,
			  MATRIX_HEIGHT, MATRIX_WIDTH, MATRIX_NAME_STRING, 
			  (char **) rowTitles, (char **) columnTitles, boxWidths,
			  boxTypes, 1, 1, ' ', ROW, true, true, false);

  if (myMatrix ==NULL)
    {
      printf("Error creating Matrix\n");
      _exit(1);
    }

  /* Display the Matrix */
  drawCDKMatrix(myMatrix, true);
  
  findData(myMatrix);

  drawCDKMatrix(myMatrix, true);    /* required  */

  /* so we can see results */
  std::cin.get();
  // Cleanup screen
  endCDK();
}

void findData(CDKMATRIX *myMatrix) {
  //Open the file cs3377.bin
  std::ifstream file;
  file.open("cs3377.bin", std::ios::out | std::ios::binary);
  //If file doesnt open, exit
  if (!file) {
    printf("Error! File cs3377.bin failed to open.");
    _exit(1);
  }

  //Read header data from binary file
  BinaryHeaderFile* headerdata = new BinaryHeaderFile();
  file.seekg(0, file.beg);
  file.read((char *) headerdata, sizeof(BinaryHeaderFile) );

  std::stringstream ss;
  //Print hex value and fill any leftover space with 0's
  ss << setfill('0') << std::hex << std::uppercase << headerdata->magicNumber;
  std::string str = ss.str();
  
  //Set matrix 1x1 to tell the magic number
  setCDKMatrixCell(myMatrix, 1, 1, (const char *) ("Magic: 0x" + str).c_str() );

  //Set stream to display decimals now
  ss << std::dec;
  ss.clear();
  ss.str(std::string());
  ss << headerdata->versionNumber;
  str = ss.str();

  setCDKMatrixCell(myMatrix, 1, 2, (const char *) ("Version: " + str).c_str() );

  ss.clear();
  ss.str(std::string());
  ss << headerdata->numRecords;
  str = ss.str();
  setCDKMatrixCell(myMatrix, 1, 3, (const char *) ("NumRecords: " + str).c_str());


  //Find size(have to iterate to end of file)
  file.seekg(0,file.end);
  int size = file.tellg();

  //Set currentpos to position after byte of header file data.
  int currentpos = sizeof(BinaryHeaderFile);
  //Start displaying a row 2
  int i = 2;
  //While the current byte position is lesser than the total size of the file
  //And while current row <= max rows
  while (currentpos < size && i <= MATRIX_HEIGHT) {
    BinaryFileRecord *recordData = new BinaryFileRecord();
    //Go to next position and read data
    file.seekg(currentpos);
    file.read( (char *) recordData, sizeof(BinaryFileRecord) );
    
    //Clear stream and read uint8_t
    ss.clear();
    ss.str(std::string());
    ss << (int)recordData->strLength;

    //Display strlen on first column of ith row
    setCDKMatrixCell(myMatrix, i, 1,
		     (const char *) ("strlen: " + ss.str()).c_str());
    //Display buffer on second column of ith row
    setCDKMatrixCell(myMatrix, i, 2, 
		     (const char *) recordData->stringBuffer);

    currentpos += sizeof(BinaryFileRecord);
    //Go to next row
    i++;
  }
  file.close();
}
