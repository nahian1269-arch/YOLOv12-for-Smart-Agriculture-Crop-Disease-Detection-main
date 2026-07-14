#include <DHT.h>
#include <BH1750.h>
#include <Wire.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// ====== Change these ======
uint8_t GATEWAY_MAC[] = {0xE8, 0x6B, 0xEA, 0xD4, 0xD8, 0x34};
const int ESPNOW_CHANNEL = 6;

// Your listed pins. These are treated as digital module outputs.
// For real analog values, move AO wires to ADC pins such as GPIO32, 33, 34, 35, 36, 39.
const bool DIGITAL_SENSOR_MODE = true;
const bool GAS_ACTIVE_LOW = true;

const int PIN_MQ5 = 23;
const int PIN_MQ2 = 22;
const int PIN_MQ7 = 21;
const int PIN_MQ135 = 19;
const int PIN_SOIL = 18;
const int PIN_MOTION = 4;
const int PIN_DHT11 = 2;
const int PIN_DHT22 = 15;
const int PIN_RELAY = 13;
const int PIN_RAIN = 12;
const int PIN_BH1750_SDA = 16;
const int PIN_BH1750_SCL = 17;

// Optional analog pins if you rewire.
const int PIN_MQ3_ANALOG = 34;
const int PIN_PH_ANALOG = 35;
const int PIN_LDR_ANALOG = 32;

DHT dht11(PIN_DHT11, DHT11);
DHT dht22(PIN_DHT22, DHT22);
BH1750 lightMeter;
bool bh1750Ready = false;

struct SensorPacket {
  char deviceId[24];
  float dht11Temp;
  float dht11Humidity;
  float dht22Temp;
  float dht22Humidity;
  float mq2;
  float mq3;
  float mq5;
  float mq7;
  float mq135;
  float soilMoisture;
  float rainDrop;
  float ph;
  float ldr;
  float lux;
  int motion;
  int rssi;
};

struct CommandPacket {
  char device[20];
  char value[16];
};

unsigned long lastSendMs = 0;
bool lastSendOk = false;

float digitalAlertValue(int pin) {
  int raw = digitalRead(pin);
  bool active = GAS_ACTIVE_LOW ? raw == LOW : raw == HIGH;
  return active ? 1000.0 : 0.0;
}

float percentFromDigital(int pin, bool activeLow = true) {
  int raw = digitalRead(pin);
  bool active = activeLow ? raw == LOW : raw == HIGH;
  return active ? 100.0 : 0.0;
}

float percentFromAnalog(int pin, bool invert = false) {
  int raw = analogRead(pin);
  float value = (raw / 4095.0) * 100.0;
  return invert ? 100.0 - value : value;
}

float phFromAnalog(int pin) {
  int raw = analogRead(pin);
  float voltage = raw * (3.3 / 4095.0);
  // Calibrate this formula with pH 4.0, 7.0, and 10.0 buffer solution.
  return 7.0 + ((2.5 - voltage) / 0.18);
}

float safeDhtRead(float value) {
  return isnan(value) ? -999.0 : value;
}

const char* gasStatus(float value) {
  return value > 0 ? "ALERT" : "clear";
}

const char* percentStatus(float value) {
  return value > 0 ? "DETECTED" : "clear";
}

SensorPacket readSensors() {
  SensorPacket packet = {};
  strlcpy(packet.deviceId, "esp32-sensor-node", sizeof(packet.deviceId));

  packet.dht11Temp = safeDhtRead(dht11.readTemperature());
  packet.dht11Humidity = safeDhtRead(dht11.readHumidity());
  packet.dht22Temp = safeDhtRead(dht22.readTemperature());
  packet.dht22Humidity = safeDhtRead(dht22.readHumidity());
  packet.mq2 = digitalAlertValue(PIN_MQ2);
  packet.mq3 = percentFromAnalog(PIN_MQ3_ANALOG);
  packet.mq5 = digitalAlertValue(PIN_MQ5);
  packet.mq7 = digitalAlertValue(PIN_MQ7);
  packet.mq135 = digitalAlertValue(PIN_MQ135);
  packet.soilMoisture = percentFromDigital(PIN_SOIL);
  packet.rainDrop = percentFromDigital(PIN_RAIN);
  packet.ph = phFromAnalog(PIN_PH_ANALOG);
  packet.lux = bh1750Ready ? lightMeter.readLightLevel() : -1;
  if (packet.lux < 0) packet.lux = 0;
  packet.ldr = min(100.0f, packet.lux / 180.0f);
  packet.motion = digitalRead(PIN_MOTION) == HIGH ? 1 : 0;
  packet.rssi = 0;
  return packet;
}

void printDhtValue(const char* label, float temp, float humidity) {
  Serial.print(label);
  Serial.print(": ");
  if (temp <= -900 || humidity <= -900) {
    Serial.println("NOT READING - check VCC/GND/DATA pin");
    return;
  }
  Serial.print(temp, 1);
  Serial.print(" C, ");
  Serial.print(humidity, 1);
  Serial.println(" %");
}

void printSensorReport(const SensorPacket& packet, esp_err_t sendResult) {
  Serial.println();
  Serial.println("========== SENSOR NODE TELEMETRY ==========");
  Serial.print("Device ID: ");
  Serial.println(packet.deviceId);
  Serial.print("Gateway MAC: E8:6B:EA:D4:D8:34 | ESP-NOW channel: ");
  Serial.println(ESPNOW_CHANNEL);

  printDhtValue("DHT11", packet.dht11Temp, packet.dht11Humidity);
  printDhtValue("DHT22", packet.dht22Temp, packet.dht22Humidity);

  Serial.print("BH1750 light: ");
  Serial.print(packet.lux, 1);
  Serial.print(" lux | module: ");
  Serial.println(bh1750Ready ? "ready" : "not detected");

  Serial.print("Soil moisture digital: ");
  Serial.print(packet.soilMoisture, 0);
  Serial.print("% | ");
  Serial.println(percentStatus(packet.soilMoisture));

  Serial.print("Rain digital: ");
  Serial.print(packet.rainDrop, 0);
  Serial.print("% | ");
  Serial.println(percentStatus(packet.rainDrop));

  Serial.print("Motion: ");
  Serial.println(packet.motion ? "DETECTED" : "clear");

  Serial.print("MQ2: ");
  Serial.print(gasStatus(packet.mq2));
  Serial.print(" | MQ5: ");
  Serial.print(gasStatus(packet.mq5));
  Serial.print(" | MQ7: ");
  Serial.print(gasStatus(packet.mq7));
  Serial.print(" | MQ135: ");
  Serial.println(gasStatus(packet.mq135));

  Serial.print("MQ3 analog: ");
  Serial.print(packet.mq3, 1);
  Serial.println(" %");

  Serial.print("pH analog estimate: ");
  Serial.println(packet.ph, 2);

  Serial.print("Relay output GPIO13: ");
  Serial.println(digitalRead(PIN_RELAY) == HIGH ? "ON" : "OFF");

  Serial.print("ESP-NOW send: ");
  Serial.println(sendResult == ESP_OK ? "OK - packet queued to gateway" : "FAILED - check gateway MAC/channel/power");
  Serial.println("===========================================");
}

void onCommandReceive(const uint8_t* mac, const uint8_t* data, int len) {
  if (len != sizeof(CommandPacket)) return;
  CommandPacket command;
  memcpy(&command, data, sizeof(command));
  Serial.printf("Command: %s=%s\n", command.device, command.value);

  if (strcmp(command.device, "relay") == 0 || strcmp(command.device, "pump_1") == 0 || strcmp(command.device, "pump_2") == 0) {
    bool on = strcmp(command.value, "on") == 0 || strcmp(command.value, "active") == 0 || strcmp(command.value, "auto_on") == 0;
    digitalWrite(PIN_RELAY, on ? HIGH : LOW);
  }
}

void setupEspNow() {
  WiFi.mode(WIFI_STA);
  esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    ESP.restart();
  }
  esp_now_register_recv_cb(onCommandReceive);
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, GATEWAY_MAC, 6);
  peerInfo.channel = ESPNOW_CHANNEL;
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println("Starting NuroAgro sensor node...");
  pinMode(PIN_MQ5, INPUT);
  pinMode(PIN_MQ2, INPUT);
  pinMode(PIN_MQ7, INPUT);
  pinMode(PIN_MQ135, INPUT);
  pinMode(PIN_SOIL, INPUT);
  pinMode(PIN_MOTION, INPUT);
  pinMode(PIN_RAIN, INPUT);
  pinMode(PIN_RELAY, OUTPUT);
  digitalWrite(PIN_RELAY, LOW);
  dht11.begin();
  dht22.begin();
  Wire.begin(PIN_BH1750_SDA, PIN_BH1750_SCL);
  bh1750Ready = lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, 0x23, &Wire);
  Serial.printf("BH1750 light sensor: %s\n", bh1750Ready ? "ready" : "not detected");
  setupEspNow();
  Serial.print("Sensor node MAC: ");
  Serial.println(WiFi.macAddress());
  Serial.printf("Gateway MAC: E8:6B:EA:D4:D8:34\n");
  Serial.printf("ESP-NOW channel: %d\n", ESPNOW_CHANNEL);
}

void loop() {
  if (millis() - lastSendMs < 5000) return;
  lastSendMs = millis();

  SensorPacket packet = readSensors();
  esp_err_t result = esp_now_send(GATEWAY_MAC, (uint8_t*)&packet, sizeof(packet));
  lastSendOk = result == ESP_OK;
  printSensorReport(packet, result);
}
