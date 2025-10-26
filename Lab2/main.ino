#include <SoftwareSerial.h>
SoftwareSerial BTSerial(8, 9);
int input_pin = A0;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  BTSerial.begin(9600);
}

void loop() {
  byte value = map(analogRead(input_pin), 0, 1023, 0, 255);
  if(BTSerial.available()){
    char c = BTSerial.read();
    if(c == 'C'){
      BTSerial.write(value);
    }
  }
  Serial.println(value);
  //Serial.write(value);
  delay(1);
}