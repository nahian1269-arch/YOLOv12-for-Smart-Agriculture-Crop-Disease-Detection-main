# NuroAgro IoT Setup

## Architecture

Use three ESP boards:

1. Gateway ESP-WROOM-32D
   - Connects to your WiFi router.
   - Sends data to Flask app at `http://YOUR_PC_IP:5000`.
   - Receives ESP-NOW sensor packets.
   - Polls `/api/device-commands` and sends relay/pump commands to the sensor node.

2. Sensor ESP-WROOM-32
   - Reads DHT11, DHT22, MQ sensors, soil, rain, pH, LDR, motion.
   - Sends compact ESP-NOW packets to the gateway.
   - Receives relay/pump commands from gateway.

3. ESP32-CAM
   - Sends heartbeat to gateway by ESP-NOW.
   - Uploads JPEG to `/api/camera/upload` over WiFi so the Flask YOLO detection pipeline can process it.

## Dashboard Connection Names

The Sensors page in the application always shows these three expected ESP boards:

| Dashboard Name | Device ID in Code | Online When |
| --- | --- | --- |
| Main Gateway ESP-WROOM-32D | `esp32-gateway` | Gateway posts heartbeat to `/api/iot/heartbeat` |
| Sensor ESP-WROOM-32 | `esp32-sensor-node` | Gateway forwards sensor packet to `/api/sensors` |
| ESP32-CAM | `esp32-cam-node` | Camera sends heartbeat/upload to the app |

The app marks a board offline if no heartbeat or packet is received for 90 seconds. The Duration column shows how long it has been online or offline.

## Important Pin Warning

Your listed pins are mostly digital-only GPIO pins:

| Sensor | Your Pin | GPIO | Good For | Analog AO? |
| --- | ---: | ---: | --- | --- |
| MQ-05 | D23 | GPIO23 | Digital DO | No |
| MQ-02 | D22 | GPIO22 | Digital DO | No |
| MQ-07 | D21 | GPIO21 | Digital DO | No |
| MQ-135 | D19 | GPIO19 | Digital DO | No |
| Soil | D18 | GPIO18 | Digital DO | No |
| Motion | D4 | GPIO4 | Digital | Yes, but use digital PIR output |
| DHT11 | D2 | GPIO2 | Digital | Not analog |
| DHT22 | D15 | GPIO15 | Digital | Not analog |
| Relay | D13 | GPIO13 | Digital output | No |
| Rain | D12 | GPIO12 | Digital DO | ADC2 only |

If your MQ/soil/rain/pH/LDR modules use AO analog pins and you want real values, rewire analog outputs to ADC pins:

| Analog Sensor | Recommended ESP32 ADC Pin |
| --- | --- |
| MQ-5 AO | GPIO34 |
| MQ-2 AO | GPIO35 |
| MQ-7 AO | GPIO32 |
| MQ-135 AO | GPIO33 |
| Soil AO | GPIO36 |
| pH AO | GPIO39 |
| LDR AO | GPIO34 or GPIO32 if free |

The included `sensor_node_esp32.ino` keeps your listed pins working as digital alert pins and uses optional ADC pins for MQ-3, pH, and LDR.

## BH1750 Light Sensor Wiring

The sensor node now uses BH1750 for real light intensity in lux. Because your GPIO21 and GPIO22 are already assigned to MQ-7 and MQ-2, the code uses custom I2C pins:

| BH1750 Pin | ESP-WROOM-32 Sensor Node |
| --- | --- |
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO16 |
| SCL | GPIO17 |
| ADDR | GND or not connected |

If `ADDR` is connected to GND or left open, the BH1750 address is `0x23`, which is what the sketch uses. Install the `BH1750` library by Christopher Laws from Arduino Library Manager.

## Flashing Order

1. Start your Flask app:
   - `npm run dev`
   - Open `http://127.0.0.1:5000/healthz`.

2. Find your computer LAN IP:
   - Windows: `ipconfig`
   - Use the IPv4 address, for example `192.168.1.100`.

3. Edit all sketches:
   - Replace `YOUR_WIFI_NAME`.
   - Replace `YOUR_WIFI_PASSWORD`.
   - Replace `APP_BASE_URL` with `http://YOUR_PC_IP:5000`.
   - Keep `PROJECT_ID` as `PRJ-001` unless you create another project in the app.

4. Flash `gateway_esp32.ino`.
   - Open Serial Monitor at `115200`.
   - Copy the printed `Gateway MAC`.
   - Copy the printed `Use ESP-NOW channel on nodes`.

5. Edit sensor and camera sketches:
   - Put the gateway MAC into `GATEWAY_MAC`.
   - Put the printed channel into `ESPNOW_CHANNEL` in the sensor sketch.

6. Flash `sensor_node_esp32.ino`.
   - Serial should print `Send sensor packet: ok`.

7. Flash `camera_node_esp32cam.ino`.
   - Board: `AI Thinker ESP32-CAM`.
   - Serial should print `Camera upload -> 200 ...`.

## ESP32-CAM Hardware Wiring

Most ESP32-CAM boards do not have a built-in USB port. Use an FTDI/USB-to-TTL programmer.

### Upload Mode Wiring

| FTDI / USB-TTL | ESP32-CAM |
| --- | --- |
| 5V | 5V |
| GND | GND |
| TX | U0R / RX0 |
| RX | U0T / TX0 |
| GND | GPIO0 |

Steps:

1. Connect `GPIO0` to `GND`.
2. Plug in the FTDI programmer.
3. Select board `AI Thinker ESP32-CAM`.
4. Upload `camera_node_esp32cam.ino`.
5. After upload, disconnect `GPIO0` from `GND`.
6. Press the ESP32-CAM reset button.

### Normal Run Wiring

| Power Supply | ESP32-CAM |
| --- | --- |
| 5V regulated, 1A or higher | 5V |
| GND | GND |

Do not keep `GPIO0` connected to `GND` during normal run, or the camera will stay in flash/upload mode.

### Wireless Connection

The ESP32-CAM uses two wireless paths:

1. WiFi to application:
   - Uploads images to `http://192.168.0.160:5000/api/camera/upload`
   - This runs disease detection in the Flask app.

2. ESP-NOW to gateway:
   - Sends heartbeat to gateway MAC `2C:BC:BB:0D:B6:E4`
   - This lets the dashboard show `ESP32-CAM` online/offline.

The ESP32-CAM and gateway must be on the same WiFi channel. Your gateway currently reports channel `6`, so keep the camera on the same WiFi network.

## Arduino Libraries

Install these from Arduino Library Manager:

- ArduinoJson by Benoit Blanchon
- DHT sensor library by Adafruit
- Adafruit Unified Sensor

ESP32 board package is required:

- Boards Manager URL: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`

## Check Connection

Open these in your browser:

- App health: `http://127.0.0.1:5000/healthz`
- IoT status: `http://127.0.0.1:5000/api/iot/status`
- Latest sensor data: `http://127.0.0.1:5000/api/status`

If ESP is connected, `/api/iot/status` shows devices like:

```json
{
  "ok": true,
  "devices": [
    {
      "id": "esp32-gateway",
      "role": "gateway",
      "status": "online",
      "online": true
    },
    {
      "id": "esp32-sensor-node",
      "role": "sensor",
      "status": "online",
      "online": true
    }
  ]
}
```

## Test Without ESP

PowerShell heartbeat test:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/iot/heartbeat -ContentType application/json -Body '{"device_id":"test-esp","role":"gateway","rssi":-45}'
```

PowerShell sensor test:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/sensors -ContentType application/json -Body '{"device_id":"test-sensor","role":"sensor","project_id":"PRJ-001","dht11_temp":27,"dht11_humidity":65,"dht22_temp":27,"dht22_humidity":64,"mq2":0,"mq3":0,"mq5":0,"mq7":0,"mq135":0,"soil_moisture":55,"rain_drop":0,"ph":6.4,"ldr":60,"motion":0}'
```

## Power Notes

- MQ sensors and relays should use a separate stable 5V supply if needed.
- All module grounds must connect to ESP32 GND.
- ESP32 GPIO is 3.3V only. If any module outputs 5V logic, use a divider or level shifter.
- Relay modules controlling pumps/lights must use proper isolation and external power.
