#include <HTTPClient.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_now.h>

// ====== Change these ======
const char* WIFI_SSID = "Nahin";
const char* WIFI_PASSWORD = "01914371557";
// Use your computer's WiFi IPv4 address. Do not use 127.0.0.1 on ESP32.
const char* APP_BASE_URL = "http://192.168.0.160:5000";
const char* PROJECT_ID = "PRJ-001";
// Gateway MAC printed by your gateway Serial Monitor: 2C:BC:BB:0D:B6:E4
uint8_t GATEWAY_MAC[] = {0x2C, 0xBC, 0xBB, 0x0D, 0xB6, 0xE4};

// AI Thinker ESP32-CAM pins
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

struct HeartbeatPacket {
  char deviceId[24];
  char role[16];
  int rssi;
};

unsigned long lastHeartbeatMs = 0;
unsigned long lastUploadMs = 0;

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  return esp_camera_init(&config) == ESP_OK;
}

void sendHeartbeat() {
  HeartbeatPacket hb = {};
  strlcpy(hb.deviceId, "esp32-cam-node", sizeof(hb.deviceId));
  strlcpy(hb.role, "camera", sizeof(hb.role));
  hb.rssi = WiFi.RSSI();
  esp_err_t result = esp_now_send(GATEWAY_MAC, (uint8_t*)&hb, sizeof(hb));
  Serial.printf("Gateway heartbeat -> %s\n", result == ESP_OK ? "sent" : "failed");
}

bool uploadImage() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return false;
  }

  String boundary = "----NuroAgroBoundary";
  String head =
    "--" + boundary + "\r\n"
    "Content-Disposition: form-data; name=\"device_id\"\r\n\r\nesp32-cam-node\r\n"
    "--" + boundary + "\r\n"
    "Content-Disposition: form-data; name=\"role\"\r\n\r\ncamera\r\n"
    "--" + boundary + "\r\n"
    "Content-Disposition: form-data; name=\"project_id\"\r\n\r\n" + String(PROJECT_ID) + "\r\n"
    "--" + boundary + "\r\n"
    "Content-Disposition: form-data; name=\"image\"; filename=\"esp32cam.jpg\"\r\n"
    "Content-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";

  HTTPClient http;
  http.begin(String(APP_BASE_URL) + "/api/camera/upload");
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  int totalLen = head.length() + fb->len + tail.length();
  uint8_t* body = (uint8_t*)malloc(totalLen);
  if (!body) {
    esp_camera_fb_return(fb);
    return false;
  }
  memcpy(body, head.c_str(), head.length());
  memcpy(body + head.length(), fb->buf, fb->len);
  memcpy(body + head.length() + fb->len, tail.c_str(), tail.length());

  int code = http.POST(body, totalLen);
  String response = http.getString();
  Serial.printf("Camera upload -> %d %s\n", code, response.c_str());

  free(body);
  http.end();
  esp_camera_fb_return(fb);
  return code >= 200 && code < 300;
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
  Serial.printf("Camera IP: %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("Camera MAC: %s\n", WiFi.macAddress().c_str());
  Serial.printf("WiFi channel: %d\n", WiFi.channel());
  Serial.printf("App URL: %s\n", APP_BASE_URL);

  if (!initCamera()) {
    Serial.println("Camera init failed");
    ESP.restart();
  }
  if (esp_now_init() == ESP_OK) {
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, GATEWAY_MAC, 6);
    peerInfo.channel = WiFi.channel();
    peerInfo.encrypt = false;
    if (esp_now_add_peer(&peerInfo) == ESP_OK) {
      Serial.println("Gateway ESP-NOW peer added");
    } else {
      Serial.println("Gateway ESP-NOW peer add failed");
    }
  } else {
    Serial.println("ESP-NOW init failed");
  }
  sendHeartbeat();
  uploadImage();
}

void loop() {
  if (millis() - lastHeartbeatMs > 30000) {
    lastHeartbeatMs = millis();
    sendHeartbeat();
  }

  // Upload an image every 60 seconds. You can trigger this from PIR/motion later.
  if (millis() - lastUploadMs > 60000) {
    lastUploadMs = millis();
    uploadImage();
  }
}
