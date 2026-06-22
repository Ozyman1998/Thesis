// ── Servo SG90 — Posicionamiento manual por monitor serie ────────────────────
// Mueve el servo al ángulo que escribas (0–180) en el monitor serie.
// Útil para encontrar el ángulo exacto de contacto con el FSR.
//
// Conexión SG90:
//   Marrón → GND
//   Rojo   → 5V
//   Naranja → pin 3 (PWM)
//
// Abre el monitor serie a 9600 baudios, escribe un ángulo y pulsa Enter.

#include <Servo.h>

Servo servo;

const int SERVO_PIN = 3;

void setup() {
  Serial.begin(9600);
  servo.attach(SERVO_PIN);
  servo.write(90);            // posición inicial — brazo vertical
  delay(500);
  Serial.println("Servo en 90 deg (posicion inicial).");
  Serial.println("Escribe un angulo (0-180) y pulsa Enter:");
}

void loop() {
  if (Serial.available() > 0) {
    int angulo = Serial.parseInt();
    if (angulo >= 0 && angulo <= 180) {
      servo.write(angulo);
      Serial.print("Moviendo a: ");
      Serial.print(angulo);
      Serial.println(" deg");
    } else {
      Serial.println("Angulo fuera de rango. Introduce un valor entre 0 y 180.");
    }
  }
}
