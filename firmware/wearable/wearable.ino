#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "secrets.h"

const int UDP_PORT = 4210;

Adafruit_MPU6050 mpu;
WiFiUDP udp;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  if (!mpu.begin()) {
    Serial.println("MPU6050 not found - check wiring");
    while (1) delay(10);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("MPU6050 ready");

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! Board IP: ");
  Serial.println(WiFi.localIP());

  udp.begin(UDP_PORT);
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  String packet = "{";
  packet += "\"t\":" + String(millis()) + ",";
  packet += "\"ax\":" + String(a.acceleration.x, 3) + ",";
  packet += "\"ay\":" + String(a.acceleration.y, 3) + ",";
  packet += "\"az\":" + String(a.acceleration.z, 3) + ",";
  packet += "\"gx\":" + String(g.gyro.x, 3) + ",";
  packet += "\"gy\":" + String(g.gyro.y, 3) + ",";
  packet += "\"gz\":" + String(g.gyro.z, 3) + ",";
  packet += "\"batt\":0";
  packet += "}";

  udp.beginPacket(LAPTOP_IP, UDP_PORT);
  udp.print(packet);
  udp.endPacket();

  delay(20);
}