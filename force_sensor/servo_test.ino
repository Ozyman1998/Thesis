// ── Servo SG90 — Test de movimiento ──────────────────────────────────────────
// Sube de 0° a 90° en pasos de 5° y vuelve a 0°.
// Usado para verificar el rango mecánico antes de la calibración del FSR.
//
// Conexión SG90:
//   Marrón → GND
//   Rojo   → 5V
//   Naranja → pin 3 (PWM)

#include <Servo.h>

Servo servo;

const int SERVO_PIN  = 3;
const int PASO       = 5;
const int ANGULO_MAX = 90;
const int RETARDO    = 500;  // ms entre pasos

void setup() {
  Serial.begin(9600);
  servo.attach(SERVO_PIN);
  servo.write(0);
  delay(1000);
  Serial.println("Servo listo. Iniciando barrido 0-90 deg...");
}

void loop() {
  for (int angulo = 0; angulo <= ANGULO_MAX; angulo += PASO) {
    servo.write(angulo);
    delay(RETARDO);
    Serial.print("Angulo: ");
    Serial.print(angulo);
    Serial.println(" deg");
  }

  delay(1000);

  for (int angulo = ANGULO_MAX; angulo >= 0; angulo -= PASO) {
    servo.write(angulo);
    delay(RETARDO);
    Serial.print("Angulo: ");
    Serial.print(angulo);
    Serial.println(" deg");
  }

  delay(2000);
}
