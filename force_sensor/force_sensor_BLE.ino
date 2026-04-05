#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

// ── Pines ────────────────────────────────────────────────
const int sensorPin  = 32;
const int sensorPin2 = 33;
const int chipSelect = 5;

// ── Timing ───────────────────────────────────────────────
unsigned long lastReadTime = 0;
const unsigned long interval = 5000;

// ── SD ───────────────────────────────────────────────────
bool sdOK = false;
unsigned long writeErrorCount = 0;

// ── BLE ──────────────────────────────────────────────────
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

BLECharacteristic *pCharacteristic = nullptr;
bool bleClientConnected = false;

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) override {
    bleClientConnected = true;
    Serial.println("BLE client connected");
  }
  void onDisconnect(BLEServer *pServer) override {
    bleClientConnected = false;
    Serial.println("BLE client disconnected");
    pServer->startAdvertising(); // reanudar advertising al desconectar
  }
};

// ── SD helpers ───────────────────────────────────────────
bool initSD() {
  if (!SD.begin(chipSelect)) {
    Serial.println("SD init failed!");
    return false;
  }
  Serial.println("SD initialized.");
  return true;
}

bool writeHeader() {
  File f = SD.open("/FSR_log.csv", FILE_APPEND);
  if (!f) return false;
  if (f.size() == 0) {
    f.println("Time_s,SensorValue1,SensorValue2");
    f.flush();
  }
  f.close();
  return true;
}

bool writeToSD(float timeSec, int v1, int v2) {
  File f = SD.open("/FSR_log.csv", FILE_APPEND);
  if (!f) return false;
  f.print(timeSec, 2);
  f.print(",");
  f.print(v1);
  f.print(",");
  f.println(v2);
  f.flush();
  f.close();
  return true;
}

// ── Setup ────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  // LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();

  // SD
  sdOK = initSD();
  if (sdOK) sdOK = writeHeader();
  lcd.setCursor(0, 0);
  lcd.print(sdOK ? "SD OK           " : "SD FAIL         ");
  delay(1500);
  lcd.clear();

  // BLE
  BLEDevice::init("FSR_Sensor");
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristic->addDescriptor(new BLE2902());

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  BLEDevice::startAdvertising();

  Serial.println("BLE advertising started — device name: FSR_Sensor");

  lcd.setCursor(0, 0);
  lcd.print("BLE ready       ");
  delay(1000);
  lcd.clear();
}

// ── Loop ─────────────────────────────────────────────────
void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastReadTime >= interval) {
    lastReadTime = currentTime;

    // -- Promediar lecturas --
    const int numSamples = 10;
    long total = 0, total2 = 0;
    for (int i = 0; i < numSamples; i++) {
      total  += analogRead(sensorPin);
      total2 += analogRead(sensorPin2);
      delay(5);
    }

    int sensorValue  = total  / numSamples;
    int sensorValue2 = total2 / numSamples;

    int y  = map(sensorValue,  0, 4095, 4095, 0);
    int y2 = map(sensorValue2, 0, 4095, 4095, 0);

    float timeSec = currentTime / 1000.0;

    // -- Serial --
    Serial.printf("Time: %.2f s | S1: %d | S2: %d\n", timeSec, y, y2);

    // -- LCD --
    lcd.setCursor(0, 0);
    lcd.printf("%.0fs F1=%-5d  ", timeSec, y);
    lcd.setCursor(0, 1);
    lcd.printf("F2=%-5d Err:%-3lu", y2, writeErrorCount);

    // -- SD --
    if (!sdOK) {
      sdOK = initSD();
      if (sdOK) writeHeader();
    }
    if (sdOK) {
      bool ok = writeToSD(timeSec, sensorValue, sensorValue2);
      if (!ok) {
        writeErrorCount++;
        sdOK = false;
        Serial.printf("Error escritura SD (#%lu)\n", writeErrorCount);
      }
    }

    // -- BLE notify --
    if (bleClientConnected && pCharacteristic != nullptr) {
      // Formato CSV: "tiempo,fuerza1,fuerza2"
      char bleMsg[32];
      snprintf(bleMsg, sizeof(bleMsg), "%.2f,%d,%d", timeSec, y, y2);
      pCharacteristic->setValue((uint8_t*)bleMsg, strlen(bleMsg));
      pCharacteristic->notify();
      Serial.printf("BLE sent: %s\n", bleMsg);
    }
  }
}