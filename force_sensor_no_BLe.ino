#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

const int sensorPin  = 32;
const int sensorPin2 = 33;
const int chipSelect = 5;

unsigned long lastReadTime = 0;
const unsigned long interval = 5000;

bool sdOK = false;
unsigned long writeErrorCount = 0;

// ── helpers ──────────────────────────────────────────────
bool initSD() {
  if (!SD.begin(chipSelect)) {
    Serial.println("SD init failed!");
    return false;
  }
  Serial.println("SD initialized.");
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
  f.flush();   // <-- fuerza escritura física antes de cerrar
  f.close();
  return true;
}

bool writeHeader() {
  // Solo escribe header si el archivo no existe o está vacío
  File f = SD.open("/FSR_log.csv", FILE_APPEND);
  if (!f) return false;
  if (f.size() == 0) {
    f.println("Time_s,SensorValue1,SensorValue2");
    f.flush();
  }
  f.close();
  return true;
}

// ── setup ─────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  lcd.init();
  lcd.backlight();
  lcd.clear();

  sdOK = initSD();

  if (sdOK) {
    sdOK = writeHeader();
  }

  lcd.setCursor(0, 0);
  lcd.print(sdOK ? "SD OK           " : "SD FAIL         ");
  delay(1500);
  lcd.clear();
}

// ── loop ──────────────────────────────────────────────────
void loop() {
  unsigned long currentTime = millis();

  // Manejo seguro de overflow de millis()
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

    // Invertir escala (sensor físico: más presión = valor más bajo)
    int y  = map(sensorValue,  0, 4095, 4095, 0);
    int y2 = map(sensorValue2, 0, 4095, 4095, 0);

    float timeSec = currentTime / 1000.0;

    // -- Serial --
    Serial.printf("Time: %.2f s | S1: %d | S2: %d\n", timeSec, y, y2);

    // -- LCD (sin clear() para evitar parpadeo) --
    lcd.setCursor(0, 0);
    lcd.printf("%.0fs F1=%-5d  ", timeSec, y);
    lcd.setCursor(0, 1);
    lcd.printf("F2=%-5d  Err:%-3lu", y2, writeErrorCount);

    // -- SD con reintento automático --
    if (!sdOK) {
      Serial.println("Reintentando SD...");
      sdOK = initSD();
      if (sdOK) writeHeader();
    }

    if (sdOK) {
      bool ok = writeToSD(timeSec, sensorValue, sensorValue2);
      if (!ok) {
        writeErrorCount++;
        sdOK = false;   // forzará reintento en el próximo ciclo
        Serial.printf("Error escritura SD (#%lu)\n", writeErrorCount);
      }
    }
  }
}