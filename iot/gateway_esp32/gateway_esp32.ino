#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

// ====== Change these ======
const char* WIFI_SSID = "Nahin";
const char* WIFI_PASSWORD = "01914371557";
// Use your computer's WiFi IPv4 address. Do not use 127.0.0.1 on ESP32.
// Current PC WiFi IP detected by ipconfig: 192.168.0.160
const char* APP_BASE_URL = "http://192.168.0.160:5000";
const char* PROJECT_ID = "PRJ-001";

// Learned automatically when the sensor node sends its first ESP-NOW packet.
uint8_t SENSOR_NODE_MAC[] = {0, 0, 0, 0, 0, 0};
bool hasSensorNodeMac = false;

// The sensor and camera ESP-NOW nodes must use this WiFi channel.
// After gateway connects, Serial prints the real channel. Copy it into the node sketches.
int ESPNOW_CHANNEL = 1;

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

struct HeartbeatPacket {
  char deviceId[24];
  char role[16];
  int rssi;
};

struct CommandPacket {
  char device[20];
  char value[16];
};

unsigned long lastHeartbeatMs = 0;
unsigned long lastCommandPollMs = 0;
volatile bool hasSensorPacket = false;
SensorPacket latestSensorPacket;

void printMac(const uint8_t* mac) {
  Serial.printf("%02X:%02X:%02X:%02X:%02X:%02X", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void addPeer(const uint8_t* mac) {
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, mac, 6);
  peerInfo.channel = ESPNOW_CHANNEL;
  peerInfo.encrypt = false;
  if (!esp_now_is_peer_exist(mac)) {
    esp_now_add_peer(&peerInfo);
  }
}

void onEspNowReceive(const uint8_t* mac, const uint8_t* data, int len) {
  if (len == sizeof(SensorPacket)) {
    memcpy(SENSOR_NODE_MAC, mac, 6);
    hasSensorNodeMac = true;
    addPeer(SENSOR_NODE_MAC);
    memcpy(&latestSensorPacket, data, sizeof(latestSensorPacket));
    latestSensorPacket.rssi = WiFi.RSSI();
    hasSensorPacket = true;
    Serial.print("Sensor packet received from ");
    printMac(mac);
    Serial.printf(" | lux %.1f | soil %.1f\n", latestSensorPacket.lux, latestSensorPacket.soilMoisture);
  } else if (len == sizeof(HeartbeatPacket)) {
    HeartbeatPacket hb;
    memcpy(&hb, data, sizeof(hb));
    if (hb.rssi == 0) {
      hb.rssi = WiFi.RSSI();
    }
    postHeartbeat(hb.deviceId, hb.role, hb.rssi);
  }
}

bool postJson(const String& path, const String& body) {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  http.begin(String(APP_BASE_URL) + path);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  String response = http.getString();
  http.end();
  Serial.printf("POST %s -> %d %s\n", path.c_str(), code, response.c_str());
  return code >= 200 && code < 300;
}

void postHeartbeat(const char* deviceId, const char* role, int rssi) {
  StaticJsonDocument<256> doc;
  doc["device_id"] = deviceId;
  doc["role"] = role;
  doc["rssi"] = rssi;
  doc["gateway_mac"] = WiFi.macAddress();
  String body;
  serializeJson(doc, body);
  postJson("/api/iot/heartbeat", body);
}

void postSensorPacket(const SensorPacket& p) {
  StaticJsonDocument<768> doc;
  doc["device_id"] = p.deviceId;
  doc["role"] = "sensor";
  doc["project_id"] = PROJECT_ID;
  doc["rssi"] = p.rssi;
  doc["dht11_temp"] = p.dht11Temp;
  doc["dht11_humidity"] = p.dht11Humidity;
  doc["dht22_temp"] = p.dht22Temp;
  doc["dht22_humidity"] = p.dht22Humidity;
  doc["mq2"] = p.mq2;
  doc["mq3"] = p.mq3;
  doc["mq5"] = p.mq5;
  doc["mq7"] = p.mq7;
  doc["mq135"] = p.mq135;
  doc["soil_moisture"] = p.soilMoisture;
  doc["rain_drop"] = p.rainDrop;
  doc["ph"] = p.ph;
  doc["ldr"] = p.ldr;
  doc["lux"] = p.lux;
  doc["motion"] = p.motion;
  String body;
  serializeJson(doc, body);
  postJson("/api/sensors", body);
}

void pollCommands() {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  http.begin(String(APP_BASE_URL) + "/api/device-commands");
  int code = http.GET();
  String response = http.getString();
  http.end();
  if (code != 200) {
    Serial.printf("Command poll failed: %d %s\n", code, response.c_str());
    return;
  }

  StaticJsonDocument<1024> doc;
  if (deserializeJson(doc, response)) return;
  JsonArray commands = doc["commands"].as<JsonArray>();
  for (JsonObject command : commands) {
    if (!hasSensorNodeMac) {
      Serial.println("Command skipped: sensor node MAC not learned yet");
      continue;
    }
    CommandPacket packet = {};
    strlcpy(packet.device, command["device"] | "", sizeof(packet.device));
    strlcpy(packet.value, command["value"] | "", sizeof(packet.value));
    esp_now_send(SENSOR_NODE_MAC, (uint8_t*)&packet, sizeof(packet));
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.printf("Gateway IP: %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("Gateway MAC: %s\n", WiFi.macAddress().c_str());
  ESPNOW_CHANNEL = WiFi.channel();
  Serial.printf("Use ESP-NOW channel on nodes: %d\n", ESPNOW_CHANNEL);
  esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    ESP.restart();
  }
  esp_now_register_recv_cb(onEspNowReceive);
  postHeartbeat("esp32-gateway", "gateway", WiFi.RSSI());
}

void loop() {
  if (hasSensorPacket) {
    noInterrupts();
    SensorPacket packet = latestSensorPacket;
    hasSensorPacket = false;
    interrupts();
    postSensorPacket(packet);
  }

  if (millis() - lastHeartbeatMs > 30000) {
    lastHeartbeatMs = millis();
    postHeartbeat("esp32-gateway", "gateway", WiFi.RSSI());
  }

  if (millis() - lastCommandPollMs > 5000) {
    lastCommandPollMs = millis();
    pollCommands();
  }
}
