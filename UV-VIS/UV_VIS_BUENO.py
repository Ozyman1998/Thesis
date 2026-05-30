#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 13:38:47 2026

@author: nacho
"""

# -*- coding: utf-8 -*-
"""
Análisis completo de transparencia UV-VIS para alineadores dentales
Versión completa con todos los parámetros y diagnóstico
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Ruta al archivo de datos
archivo = "/home/nacho/Escritorio/THESIS_USB/UV_VIS/th/c_L2d.txt"

# Parámetros de suavizado (ajustar según ruido de tus datos)
SMOOTH_WINDOW = 11  # Debe ser impar
SMOOTH_POLYORDER = 3

# ============================================================================
# LECTURA DE DATOS
# ============================================================================

print("="*80)
print("ANÁLISIS DE TRANSPARENCIA UV-VIS PARA ALINEADORES DENTALES")
print("="*80)

# Leer archivo
with open(archivo, 'r+') as ff:
    wl = []
    for line in ff:
        wl.append(line.split())

# Extraer información
l = len(wl)
sample = str(wl[0][0])
headers = wl[1]

# Extraer datos numéricos
wn = []
T = []
for i in range(2, l-2):
    try:
        wn.append(float(wl[i][0]))
        T.append(float(wl[i][1]))
    except:
        continue

# Convertir a arrays
wn = np.array(wn)
T = np.array(T)

print(f"\n📄 Archivo leído: {archivo}")
print(f"📊 Muestra: {sample}")
print(f"📏 Puntos de datos: {len(wn)}")
print(f"📐 Rango espectral: {wn.min():.1f} - {wn.max():.1f} nm")

# ============================================================================
# SUAVIZADO DE DATOS
# ============================================================================

T_smooth = savgol_filter(T, window_length=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)

print(f"\n✓ Datos suavizados (Savitzky-Golay: ventana={SMOOTH_WINDOW}, orden={SMOOTH_POLYORDER})")

# ============================================================================
# ANÁLISIS UV
# ============================================================================

print("\n" + "="*80)
print("📊 PROTECCIÓN UV (prevención de degradación)")
print("="*80)

# Máscaras espectrales
mask_uva = (wn >= 315) & (wn <= 400)
mask_uvb = (wn >= 280) & (wn <= 315)
mask_uv_total = (wn >= 280) & (wn <= 400)

# Transmitancia promedio en cada rango
T_uva_avg = np.mean(T_smooth[mask_uva])
T_uvb_avg = np.mean(T_smooth[mask_uvb])
T_uv_total_avg = np.mean(T_smooth[mask_uv_total])

# Bloqueo = 100% - Transmitancia
blocking_uva = 100 - T_uva_avg
blocking_uvb = 100 - T_uvb_avg
blocking_uv_total = 100 - T_uv_total_avg

# Evaluación
def evaluar_uva(bloqueo):
    if bloqueo > 75:
        return "✓✓✓ EXCELENTE (puede afectar T visible)"
    elif bloqueo > 60:
        return "✓✓ MUY BUENO"
    elif bloqueo > 45:
        return "✓ ACEPTABLE"
    elif bloqueo > 30:
        return "⚠ LÍMITE (mejorable)"
    else:
        return "❌ INSUFICIENTE"

def evaluar_uvb(bloqueo):
    if bloqueo > 95:
        return "✓✓✓ EXCELENTE"
    elif bloqueo > 85:
        return "✓✓ BUENO"
    elif bloqueo > 70:
        return "✓ ACEPTABLE"
    else:
        return "⚠ BAJO"

print(f"  • Bloqueo UVB (280-315 nm): {blocking_uvb:.1f}% → {evaluar_uvb(blocking_uvb)}")
print(f"  • Bloqueo UVA (315-400 nm): {blocking_uva:.1f}% → {evaluar_uva(blocking_uva)}")
print(f"  • Bloqueo UV total (280-400 nm): {blocking_uv_total:.1f}%")

# ============================================================================
# LONGITUD DE ONDA DE CORTE
# ============================================================================

# λ donde T = 50%
idx_50 = np.argmin(np.abs(T_smooth - 50))
lambda_cutoff_50 = wn[idx_50]

# λ donde T = 90%
idx_90 = np.argmin(np.abs(T_smooth - 90))
lambda_90 = wn[idx_90]

# λ donde T = 10%
idx_10 = np.argmin(np.abs(T_smooth - 10))
lambda_cutoff_10 = wn[idx_10]

print(f"\n📐 LONGITUDES DE ONDA DE CORTE:")
print(f"  • λ (T=10%): {lambda_cutoff_10:.1f} nm")
print(f"  • λ (T=50%): {lambda_cutoff_50:.1f} nm")
print(f"  • λ (T=90%): {lambda_90:.1f} nm")

# ============================================================================
# TRANSPARENCIA VISIBLE
# ============================================================================

print("\n" + "="*80)
print("👁️  TRANSPARENCIA VISIBLE (estética/invisibilidad)")
print("="*80)

# Rango visible
mask_vis = (wn >= 400) & (wn <= 700)
T_vis_avg = np.mean(T_smooth[mask_vis])

# Bandas de color específicas
mask_blue = (wn >= 450) & (wn <= 495)
mask_green = (wn >= 495) & (wn <= 570)
mask_yellow = (wn >= 570) & (wn <= 590)
mask_red = (wn >= 620) & (wn <= 700)

T_blue = np.mean(T_smooth[mask_blue])
T_green = np.mean(T_smooth[mask_green])
T_yellow = np.mean(T_smooth[mask_yellow])
T_red = np.mean(T_smooth[mask_red])

# T a 550 nm (crítico)
idx_550 = np.argmin(np.abs(wn - 550))
T_550 = T_smooth[idx_550]

# Evaluación
def evaluar_transparencia(T):
    if T > 92:
        return "✓✓✓ EXCELENTE - Prácticamente invisible"
    elif T > 88:
        return "✓✓ MUY BUENO - Alta calidad"
    elif T > 85:
        return "✓ ACEPTABLE - Discreto"
    elif T > 80:
        return "⚠ LÍMITE - Notablemente visible"
    else:
        return "❌ NO APTO - Claramente visible"

print(f"  • T visible media (400-700 nm): {T_vis_avg:.1f}%")
print(f"    {evaluar_transparencia(T_vis_avg)}")
print(f"\n  • T a 550 nm (CRÍTICO): {T_550:.1f}%")
print(f"    {evaluar_transparencia(T_550)}")
print(f"\n  Transmitancia por bandas de color:")
print(f"    - Azul (450-495 nm):    {T_blue:.1f}%")
print(f"    - Verde (495-570 nm):   {T_green:.1f}%")
print(f"    - Amarillo (570-590 nm): {T_yellow:.1f}%")
print(f"    - Rojo (620-700 nm):    {T_red:.1f}%")

# ============================================================================
# BALANCE DE COLOR E ÍNDICE DE AMARILLEAMIENTO
# ============================================================================

print("\n" + "="*80)
print("🎨 BALANCE DE COLOR")
print("="*80)

# Diferencia Rojo-Azul
diff_rojo_azul = T_red - T_blue

# Índice de amarilleamiento (aproximado)
YI_approx = (T_red - T_blue) * 100 / T_vis_avg if T_vis_avg > 0 else 0

# Desviación estándar en visible (uniformidad)
std_vis = np.std(T_smooth[mask_vis])

# Evaluaciones
def evaluar_diff_rb(diff):
    if -1 <= diff <= 3:
        return "✓✓✓ EXCELENTE - Balance neutro"
    elif -3 <= diff <= 5:
        return "✓✓ BUENO"
    elif -5 <= diff <= 8 or diff < -3:
        return "⚠ ACEPTABLE"
    else:
        return "❌ PROBLEMÁTICO"

def evaluar_yi(yi):
    if abs(yi) < 3:
        return "✓✓✓ EXCELENTE - Sin color perceptible"
    elif abs(yi) < 5:
        return "✓✓ BUENO"
    elif abs(yi) < 8:
        return "✓ ACEPTABLE"
    else:
        return "⚠ LÍMITE"

print(f"  • Diferencia Rojo-Azul: {diff_rojo_azul:+.1f}%")
print(f"    {evaluar_diff_rb(diff_rojo_azul)}")
print(f"\n  • Índice de Amarilleamiento (YI): {YI_approx:+.2f}")
print(f"    {evaluar_yi(YI_approx)}")
print(f"\n  • Uniformidad espectral (desv. std): {std_vis:.2f}%")
if std_vis < 3:
    print(f"    ✓ Alta uniformidad - Material homogéneo")
else:
    print(f"    ⚠ Variación notable")

# Interpretación de valores negativos
if diff_rojo_azul < -0.5 or YI_approx < -0.5:
    print(f"\n  ℹ️  NOTA: Valores negativos indican ligero sesgo AZULADO")
    print(f"      • Típico en materiales con OBAs (blanqueadores ópticos)")
    print(f"      • Puede ser BENEFICIOSO: compensa amarilleamiento futuro")
    print(f"      • NO típico en TPU puro (TPU suele tener YI positivo +2 a +5)")

# ============================================================================
# TRANSICIÓN UV-VISIBLE
# ============================================================================

print("\n" + "="*80)
print("📐 TRANSICIÓN UV-VISIBLE")
print("="*80)

# Pendiente en zona de transición
mask_transition = (wn >= lambda_cutoff_50 - 20) & (wn <= lambda_cutoff_50 + 20)
if np.sum(mask_transition) > 1:
    slope = np.gradient(T_smooth[mask_transition], wn[mask_transition])
    max_slope = np.max(slope)
else:
    max_slope = 0

print(f"  • λ (T=90%): {lambda_90:.1f} nm")
print(f"  • Pendiente máxima: {max_slope:.2f} %/nm")
if max_slope > 2:
    print(f"    ✓ Transición abrupta (buen corte UV)")
else:
    print(f"    → Transición gradual")

# ============================================================================
# EVALUACIÓN GENERAL
# ============================================================================

print("\n" + "="*80)
print("✅ EVALUACIÓN GENERAL PARA ALINEADORES DENTALES")
print("="*80)

# Criterios de aceptación
criterios_cumplidos = 0
total_criterios = 5

print("\nCriterios críticos:")

# 1. Transparencia visible
if T_vis_avg > 88:
    print("  ✓ T visible > 88%: CUMPLE")
    criterios_cumplidos += 1
elif T_vis_avg > 85:
    print("  ⚠ T visible > 85%: LÍMITE (aceptable para mercado medio)")
    criterios_cumplidos += 0.5
else:
    print(f"  ❌ T visible = {T_vis_avg:.1f}%: NO CUMPLE (requiere >85%, ideal >88%)")

# 2. T a 550 nm
if T_550 > 88:
    print("  ✓ T₅₅₀ > 88%: CUMPLE")
    criterios_cumplidos += 1
elif T_550 > 85:
    print("  ⚠ T₅₅₀ > 85%: LÍMITE")
    criterios_cumplidos += 0.5
else:
    print(f"  ❌ T₅₅₀ = {T_550:.1f}%: NO CUMPLE")

# 3. Bloqueo UVA
if 45 <= blocking_uva <= 70:
    print(f"  ✓ Bloqueo UVA = {blocking_uva:.1f}%: CUMPLE (45-70%)")
    criterios_cumplidos += 1
elif 40 <= blocking_uva < 45 or 70 < blocking_uva <= 75:
    print(f"  ⚠ Bloqueo UVA = {blocking_uva:.1f}%: LÍMITE")
    criterios_cumplidos += 0.5
else:
    print(f"  ❌ Bloqueo UVA = {blocking_uva:.1f}%: Fuera de rango óptimo")

# 4. Balance de color
if -1 <= diff_rojo_azul <= 3:
    print(f"  ✓ Balance color (ΔR-B = {diff_rojo_azul:+.1f}%): CUMPLE")
    criterios_cumplidos += 1
elif -3 <= diff_rojo_azul <= 5:
    print(f"  ⚠ Balance color: ACEPTABLE")
    criterios_cumplidos += 0.5
else:
    print(f"  ❌ Balance color: Fuera de rango")

# 5. Amarilleamiento
if abs(YI_approx) < 5:
    print(f"  ✓ YI = {YI_approx:+.2f}: CUMPLE")
    criterios_cumplidos += 1
elif abs(YI_approx) < 8:
    print(f"  ⚠ YI: ACEPTABLE")
    criterios_cumplidos += 0.5
else:
    print(f"  ❌ YI: Fuera de rango")

# Veredicto final
print(f"\n{'='*80}")
print(f"PUNTUACIÓN: {criterios_cumplidos:.1f}/{total_criterios}")
print(f"{'='*80}")

if criterios_cumplidos >= 4.5:
    print("✅ MATERIAL APTO - Alta calidad para alineadores dentales")
elif criterios_cumplidos >= 3.5:
    print("✓ MATERIAL ACEPTABLE - Calidad media, comercializable")
elif criterios_cumplidos >= 2.5:
    print("⚠ MATERIAL EN LÍMITE - Requiere mejoras para ser competitivo")
else:
    print("❌ MATERIAL NO APTO - Reformulación necesaria")

# Recomendaciones específicas
print("\n" + "="*80)
print("💡 RECOMENDACIONES PRIORITARIAS")
print("="*80)

recomendaciones = []

if T_vis_avg < 85:
    delta_T = 88 - T_vis_avg
    recomendaciones.append(f"🔴 CRÍTICO: Aumentar T visible de {T_vis_avg:.1f}% a >85% (Δ = +{delta_T:.1f}%)")
    recomendaciones.append(f"   → Reducir absorbedor UV en ~{100*(1-T_vis_avg/88):.0f}%")
    recomendaciones.append(f"   → O cambiar a absorbedor con λmax <335 nm")

if blocking_uva < 45:
    delta_uva = 50 - blocking_uva
    recomendaciones.append(f"🟡 IMPORTANTE: Aumentar bloqueo UVA de {blocking_uva:.1f}% a 50-55%")
    recomendaciones.append(f"   → Añadir {delta_uva:.1f}% más de protección UVA")

if abs(YI_approx) < 1 and YI_approx < 0:
    if "TPU" in sample.upper() or "POLIURETANO" in sample.upper():
        recomendaciones.append(f"ℹ️  INFORMACIÓN: YI negativo es ANÓMALO para TPU")
        recomendaciones.append(f"   → Verificar si material contiene OBAs (blanqueadores ópticos)")
        recomendaciones.append(f"   → Prueba: iluminar con UV 365nm (debe fluorescer azul si tiene OBAs)")

if lambda_cutoff_50 > 360:
    recomendaciones.append(f"⚠ ATENCIÓN: λ corte = {lambda_cutoff_50:.0f} nm está cerca del visible")
    recomendaciones.append(f"   → Optimizar para λ₅₀ < 350 nm")

if not recomendaciones:
    print("✓ Material cumple todas las especificaciones")
    print("  Continuar con validación mecánica y biocompatibilidad")
else:
    for i, rec in enumerate(recomendaciones, 1):
        print(f"{i}. {rec}")

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

print("\n" + "="*80)
print("📊 GENERANDO GRÁFICOS...")
print("="*80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# ===== PLOT 1: Espectro completo =====
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(wn, T_smooth, 'b-', linewidth=2.5, label='Transmitancia')

# Zonas sombreadas
ax1.axvspan(280, 315, alpha=0.2, color='red', label='UVB')
ax1.axvspan(315, 400, alpha=0.2, color='orange', label='UVA')
ax1.axvspan(400, 700, alpha=0.15, color='yellow', label='Visible')

# Líneas de referencia
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(y=85, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target T=85%')
ax1.axhline(y=T_vis_avg, color='red', linestyle='--', linewidth=2, 
            label=f'T visible = {T_vis_avg:.1f}%')
ax1.axvline(x=lambda_cutoff_50, color='purple', linestyle='--', 
            label=f'λ₅₀ = {lambda_cutoff_50:.1f} nm')
ax1.axvline(x=550, color='darkgreen', linestyle=':', linewidth=2, 
            label=f'550 nm (T={T_550:.1f}%)')

ax1.set_xlabel('Longitud de onda (nm)', fontsize=12)
ax1.set_ylabel('Transmitancia (%)', fontsize=12)
ax1.set_title(f'Espectro UV-VIS completo: {sample}', fontsize=14, fontweight='bold')
ax1.set_xlim(280, 800)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=10, ncol=2)

# ===== PLOT 2: Zoom transición UV-Visible =====
ax2 = fig.add_subplot(gs[1, 0])
mask_zoom = (wn >= 300) & (wn <= 450)
ax2.plot(wn[mask_zoom], T_smooth[mask_zoom], 'b-', linewidth=3)
ax2.axvline(x=400, color='purple', linestyle='--', linewidth=2, label='Inicio visible (400 nm)')
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax2.axhline(y=90, color='green', linestyle=':', alpha=0.5)
ax2.fill_between(wn[mask_zoom], 0, T_smooth[mask_zoom], 
                  where=(wn[mask_zoom] < 400), alpha=0.2, color='orange')
ax2.set_xlabel('Longitud de onda (nm)', fontsize=11)
ax2.set_ylabel('Transmitancia (%)', fontsize=11)
ax2.set_title('Zona de transición UV-Visible', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim(300, 450)

# ===== PLOT 3: Transmitancia por bandas =====
ax3 = fig.add_subplot(gs[1, 1])
bands = ['Azul\n450-495', 'Verde\n495-570', 'Amarillo\n570-590', 'Rojo\n620-700', 'Media\nvisible']
values = [T_blue, T_green, T_yellow, T_red, T_vis_avg]
colors = ['#4169E1', '#32CD32', '#FFD700', '#DC143C', '#808080']
bars = ax3.bar(bands, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.axhline(y=T_vis_avg, color='black', linestyle='--', linewidth=2)
ax3.axhline(y=85, color='green', linestyle=':', linewidth=2, label='Mínimo: 85%')
ax3.axhline(y=88, color='darkgreen', linestyle='--', linewidth=2, label='Óptimo: 88%')
ax3.set_ylabel('Transmitancia (%)', fontsize=11)
ax3.set_title('Transmitancia por bandas espectrales', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 100)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2., val + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ===== PLOT 4: Absorbancia en visible =====
ax4 = fig.add_subplot(gs[2, 0])
A_vis = -np.log10(T_smooth[mask_vis]/100)
ax4.plot(wn[mask_vis], A_vis, 'orange', linewidth=2.5)
ax4.axhline(y=0.05, color='green', linestyle='--', linewidth=2, label='A ideal <0.05')
ax4.axhline(y=0.15, color='red', linestyle=':', linewidth=2, label='A problemática >0.15')
ax4.fill_between(wn[mask_vis], 0.05, A_vis, where=(A_vis > 0.05),
                  alpha=0.2, color='red')
ax4.set_xlabel('Longitud de onda (nm)', fontsize=11)
ax4.set_ylabel('Absorbancia', fontsize=11)
ax4.set_title('Absorbancia en región visible', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.set_xlim(400, 700)

# ===== PLOT 5: Resumen de métricas =====
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

# Determinar estado global
if criterios_cumplidos >= 4:
    estado_color = 'lightgreen'
    estado_text = 'APTO'
elif criterios_cumplidos >= 3:
    estado_color = 'lightyellow'
    estado_text = 'ACEPTABLE'
elif criterios_cumplidos >= 2:
    estado_color = 'orange'
    estado_text = 'LÍMITE'
else:
    estado_color = 'lightcoral'
    estado_text = 'NO APTO'

summary_text = f"""
╔═══════════════════════════════════════════════╗
║  RESUMEN DE PROPIEDADES                       ║
╚═══════════════════════════════════════════════╝

Muestra: {sample}

PROTECCIÓN UV:
  • Bloqueo UVB:  {blocking_uvb:.1f}%
  • Bloqueo UVA:  {blocking_uva:.1f}%
  • λ corte (50%): {lambda_cutoff_50:.1f} nm

TRANSPARENCIA:
  • T visible:    {T_vis_avg:.1f}%
  • T a 550 nm:   {T_550:.1f}%

BALANCE DE COLOR:
  • Δ(Rojo-Azul): {diff_rojo_azul:+.1f}%
  • YI:           {YI_approx:+.2f}
  • Uniformidad:  {std_vis:.2f}%

PUNTUACIÓN: {criterios_cumplidos:.1f}/5

ESTADO: {estado_text}
"""

ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor=estado_color, alpha=0.5, 
                   edgecolor='black', linewidth=2))

plt.suptitle(f'Análisis completo de transparencia: {sample}', 
             fontsize=16, fontweight='bold', y=0.995)

# Guardar figura
output_file = f'{sample}_analisis_completo.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Gráfico guardado: {output_file}")

plt.show()

# ============================================================================
# EXPORTAR DATOS
# ============================================================================

# Guardar datos procesados
output_csv = f'{sample}_datos_procesados.csv'
header_csv = 'Wavelength_nm,Transmittance_raw_%,Transmittance_smooth_%,Absorbance'
data_export = np.column_stack([
    wn, 
    T, 
    T_smooth, 
    -np.log10(T_smooth/100)
])
np.savetxt(output_csv, data_export, delimiter=',', header=header_csv, comments='')
print(f"✓ Datos exportados: {output_csv}")

# Guardar resumen de resultados
output_summary = f'{sample}_resumen.txt'
with open(output_summary, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RESUMEN DE ANÁLISIS UV-VIS PARA ALINEADORES DENTALES\n")
    f.write("="*80 + "\n\n")
    f.write(f"Muestra: {sample}\n")
    f.write(f"Archivo: {archivo}\n")
    f.write(f"Puntos de datos: {len(wn)}\n")
    f.write(f"Rango espectral: {wn.min():.1f} - {wn.max():.1f} nm\n\n")
    
    f.write("-"*80 + "\n")
    f.write("PROTECCIÓN UV\n")
    f.write("-"*80 + "\n")
    f.write(f"Bloqueo UVB (280-315 nm): {blocking_uvb:.1f}%\n")
    f.write(f"Bloqueo UVA (315-400 nm): {blocking_uva:.1f}%\n")
    f.write(f"Bloqueo UV total: {blocking_uv_total:.1f}%\n")
    f.write(f"λ corte (T=50%): {lambda_cutoff_50:.1f} nm\n")
    f.write(f"λ (T=90%): {lambda_90:.1f} nm\n\n")
    
    f.write("-"*80 + "\n")
    f.write("TRANSPARENCIA VISIBLE\n")
    f.write("-"*80 + "\n")
    f.write(f"T visible media (400-700 nm): {T_vis_avg:.1f}%\n")
    f.write(f"T a 550 nm (CRÍTICO): {T_550:.1f}%\n")
    f.write(f"T azul (450-495 nm): {T_blue:.1f}%\n")
    f.write(f"T verde (495-570 nm): {T_green:.1f}%\n")
    f.write(f"T amarillo (570-590 nm): {T_yellow:.1f}%\n")
    f.write(f"T rojo (620-700 nm): {T_red:.1f}%\n\n")
    
    f.write("-"*80 + "\n")
    f.write("BALANCE DE COLOR\n")
    f.write("-"*80 + "\n")
    f.write(f"Diferencia Rojo-Azul: {diff_rojo_azul:+.1f}%\n")
    f.write(f"Índice Amarilleamiento (YI): {YI_approx:+.2f}\n")
    f.write(f"Uniformidad espectral (σ): {std_vis:.2f}%\n\n")
    
    f.write("-"*80 + "\n")
    f.write("EVALUACIÓN FINAL\n")
    f.write("-"*80 + "\n")
    f.write(f"Puntuación: {criterios_cumplidos:.1f}/5\n")
    f.write(f"Estado: {estado_text}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("RECOMENDACIONES\n")
    f.write("-"*80 + "\n")
    if recomendaciones:
        for i, rec in enumerate(recomendaciones, 1):
            f.write(f"{i}. {rec}\n")
    else:
        f.write("Material cumple especificaciones\n")

print(f"✓ Resumen guardado: {output_summary}")

print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
print(f"\nArchivos generados:")
print(f"  1. {output_file} - Gráficos completos")
print(f"  2. {output_csv} - Datos procesados")
print(f"  3. {output_summary} - Resumen de resultados")