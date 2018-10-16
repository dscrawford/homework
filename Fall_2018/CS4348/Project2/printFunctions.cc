#include <iostream>
void CheckIn(int cust, int bags) {
  printf("Guest %d enters hotel with %d bags\n", cust, bags);
}
void GiveRoom(int frontdesk, int cust, int room) {
  printf("Front desk employee %d registers guest %d and assigns room %d\n",
	 frontdesk, cust, room);
}
void GetRoom(int frontdesk, int cust, int room) {
  printf("Guest %d receives key for room %d from front desk employee %d\n",
	 cust, room, frontdesk);
}
void GetBellHop(int cust) {
  printf("Guest %d requests help with bags\n", cust);
}
void EnterRoom(int cust, int room) {
  printf("Guest %d enters room %d\n", cust, room);
}
void GetBags(int bellhop, int cust) {
  printf("Bellhop %d receives bags from guest %d\n", bellhop, cust);
}
void GiveBags(int bellhop, int cust) {
  printf("Bellhop %d delivers bags to guest %d\n", bellhop, cust);
}
void GetBackBags(int bellhop, int cust) {
  printf("Guest %d receives bags from bellhop %d and gives tip\n",
	 cust, bellhop);
}
void Retire(int cust) {
  printf("Guest %d retires for the evening\n", cust);
}
