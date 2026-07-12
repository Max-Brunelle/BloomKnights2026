#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>


#define USE_MPU6050 true

Adafruit_MPU6050 mpu;
WiFiUDP udp;
bool mpu_ok = false;

void setup() {
  Serial.begin(115200);
  delay(1000);

  #if USE_MPU6050
    if (!mpu.begin()) {
      Serial.println("MPU6050 not found - check wiring");
      // Do NOT hang here anymore - continue so WiFi can still be tested
    } else {
      mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
      mpu.setGyroRange(MPU6050_RANGE_500_DEG);
      mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
      mpu_ok = true;
      Serial.println("MPU6050 ready");
    }
  #else
    Serial.println("Sensor init skipped (USE_MPU6050 = false) - testing WiFi/UDP only");
  #endif

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  int wifi_attempts = 0;
  while (WiFi.status() != WL_CONNECTED && wifi_attempts < 40) {
    delay(500);
    Serial.print(".");
    wifi_attempts++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Connected! Board IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi FAILED to connect after 20 seconds - check SSID/password/band");
  }

  udp.begin(UDP_PORT);
}

void loop() {
  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;

  if (mpu_ok) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    ax = a.acceleration.x;
    ay = a.acceleration.y;
    az = a.acceleration.z;
    gx = g.gyro.x;
    gy = g.gyro.y;
    gz = g.gyro.z;
  }

  // Build a compact JSON packet matching the agreed schema
  String packet = "{";
  packet += "\"t\":" + String(millis()) + ",";
  packet += "\"ax\":" + String(ax, 3) + ",";
  packet += "\"ay\":" + String(ay, 3) + ",";
  packet += "\"az\":" + String(az, 3) + ",";
  packet += "\"gx\":" + String(gx, 3) + ",";
  packet += "\"gy\":" + String(gy, 3) + ",";
  packet += "\"gz\":" + String(gz, 3) + ",";
  packet += "\"batt\":0";
  packet += "}";

  udp.beginPacket(LAPTOP_IP, UDP_PORT);
  udp.print(packet);
  udp.endPacket();

  delay(20); // ~50Hz
}