import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONSTANTES GEOMÉTRICAS Y MASAS
# ==========================================
L1 = 0.150      # Longitud de la barra 1 (m)
L2 = 0.085      # Longitud de la barra 2 (m)
L3 = 0.053      # Distancia horizontal al pivote A (m)
L4 = 0.13239    # Distancia vertical al pivote A (m)
Rp = 0.030      # Radio del pin excéntrico (m)

# Masas (convertidas a kg para cálculos de fuerza y torque)
m_barra1 = 14.76 / 1000   # 0.01476 kg
m_barra2 = 7.68 / 1000    # 0.00768 kg
m_sellador = 4.58 / 1000  # 0.00458 kg 

g = 9.81  # Gravedad (m/s²)

# Momentos de Inercia de las barras
I_A = (1/3) * m_barra1 * (L1**2)    # Barra 1 respecto al pivote fijo A
I_G2 = (1/12) * m_barra2 * (L2**2)  # Barra 2 respecto a su centro de masa

# Propiedades de sellado
k_goma = 80000     # Rigidez de la goma (N/m)
y_goma = -0.093    # Altura de inicio de compresión de la goma (m)

# Velocidad angular de entrada (15 RPM)
Wr_rpm = 15
Wr = Wr_rpm * (2 * np.pi / 60) 

# ==========================================
# 2. BUCLE ÚNICO: CINEMÁTICA + CINÉTICA
# ==========================================
theta1_deg = np.linspace(0, 360, 360)
theta1_rad = np.radians(theta1_deg)
datos = [] 

for t1 in theta1_rad:
    # --- CINEMÁTICA: POSICIÓN BARRA 1 ---
    xc, yc = Rp * np.cos(t1), Rp * np.sin(t1)
    xA, yA = L3, L4
    D1 = np.sqrt((xc - xA)**2 + (yc - yA)**2)
    t2 = np.arctan2((yc - yA), (xc - xA))
    
    # --- CINEMÁTICA: VELOCIDAD BARRA 1 ---
    Vc_mag = Wr * Rp
    Vcx, Vcy = -Vc_mag * np.sin(t1), Vc_mag * np.cos(t1)
    A_vel = np.array([[-D1 * np.sin(t2), np.cos(t2)], [D1 * np.cos(t2), np.sin(t2)]])
    X_vel = np.linalg.solve(A_vel, np.array([Vcx, Vcy]))
    Wab, V_c_AB = X_vel[0], X_vel[1]
    
    # --- CINEMÁTICA: ACELERACIÓN BARRA 1 ---
    Ac_n = (Wr**2) * Rp
    Ac_x, Ac_y = -Ac_n * np.cos(t1), -Ac_n * np.sin(t1)
    Ac_prim_n = (Wab**2) * D1
    A_coriolis = 2 * Wab * V_c_AB
    Bx_acc = Ac_x - Ac_prim_n * np.cos(t2) - A_coriolis * np.cos(t2 + np.pi/2)
    By_acc = Ac_y - Ac_prim_n * np.sin(t2) - A_coriolis * np.sin(t2 + np.pi/2)
    X_acc = np.linalg.solve(A_vel, np.array([Bx_acc, By_acc]))
    alpha_AB, a_c_AB = X_acc[0], X_acc[1]
    
    Ab_n = (Wab**2) * L1
    Ab_t = alpha_AB * L1

    # --- CINEMÁTICA: BARRA 2 Y SELLADOR D ---
    xB, yB = xA + L1 * np.cos(t2), yA + L1 * np.sin(t2)
    Xd = 0.0 
    
    if abs(Xd - xB) <= L2:
        yD = yB - np.sqrt(L2**2 - (Xd - xB)**2) 
        t3 = np.arctan2((yD - yB), (Xd - xB))
        
        Vbx, Vby = -Wab * L1 * np.sin(t2), Wab * L1 * np.cos(t2)
        W_BD = Vbx / (L2 * np.sin(t3))
        V_Dy = Vby + W_BD * L2 * np.cos(t3)
        
        Abx = -Ab_t * np.sin(t2) - Ab_n * np.cos(t2)
        Aby = Ab_t * np.cos(t2) - Ab_n * np.sin(t2)
        Alpha_BD = (Abx - (W_BD**2) * L2 * np.cos(t3)) / (L2 * np.sin(t3))
        A_Dy = Aby + Alpha_BD * L2 * np.cos(t3) - (W_BD**2) * L2 * np.sin(t3)
        
        # --- CINÉTICA: FUERZAS Y TORQUE ---
        if yD <= y_goma:
            F_sellado = k_goma * (y_goma - yD)
        else:
            F_sellado = 0.0
            
        V_G1y = Wab * (L1/2) * np.cos(t2)
        V_G2x = Vbx / 2
        V_G2y = (Vby + V_Dy) / 2
        a_G2x = Abx / 2
        a_G2y = (Aby + A_Dy) / 2
        
        # Potencias Virtuales
        P_grav = - (m_barra1 * g * V_G1y) - (m_barra2 * g * V_G2y) - (m_sellador * g * V_Dy)
        P_iner_1 = - I_A * alpha_AB * Wab
        P_iner_2 = - (m_barra2 * a_G2x * V_G2x + m_barra2 * a_G2y * V_G2y) - I_G2 * Alpha_BD * W_BD
        P_iner_D = - m_sellador * A_Dy * V_Dy  
        P_ext = - F_sellado * V_Dy  
        
        # Torque del Motor final
        T_motor = - (P_grav + P_iner_1 + P_iner_2 + P_iner_D + P_ext) / Wr
        
    else:
        t3, W_BD, V_Dy, Alpha_BD, A_Dy, F_sellado, T_motor = [np.nan]*7

    datos.append({
        'Theta1_deg': np.degrees(t1),
        'D1_m': D1,
        'Theta2_deg': np.degrees(t2),
        'Theta3_deg': np.degrees(t3) if not np.isnan(t3) else np.nan,
        'W_AB_rad_s': Wab,
        'W_BD_rad_s': W_BD,
        'V_deslizamiento_c_AB_m_s': V_c_AB,
        'Velocidad_Sellador_D_m_s': V_Dy,
        'Alpha_AB_rad_s2': alpha_AB,
        'Alpha_BD_rad_s2': Alpha_BD,
        'A_deslizamiento_c_AB_m_s2': a_c_AB,
        'A_Coriolis_m_s2': A_coriolis,
        'Aceleracion_Sellador_D_m_s2': A_Dy,
        'Fuerza_Sellado_N': F_sellado,
        'Torque_Motor_Nm': T_motor
    })

# ==========================================
# 3. EXPORTACIÓN A EXCEL
# ==========================================
df = pd.DataFrame(datos)
nombre_archivo = 'Analisis_Mecanico_Total.xlsx'
df.to_excel(nombre_archivo, index=False)
print(f"✅ ¡Éxito! Todas las variables guardadas en: {nombre_archivo}")

# ==========================================
# 4. GRÁFICAS (NUEVO PANEL 2x2)
# ==========================================
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análisis Mecánico Unificado a 15 RPM', fontsize=18, fontweight='bold')

# [0,0] VELOCIDADES ANGULARES
axs[0, 0].plot(df['Theta1_deg'], df['W_AB_rad_s'], label='W_AB (Barra 1)', color='blue')
axs[0, 0].plot(df['Theta1_deg'], df['W_BD_rad_s'], label='W_BD (Barra 2)', color='red', linestyle='--')
axs[0, 0].set_title('Velocidades Angulares')
axs[0, 0].set_ylabel('rad/s')
axs[0, 0].set_xlim(0, 360)
axs[0, 0].legend()

# [0,1] ACELERACIONES ANGULARES
axs[0, 1].plot(df['Theta1_deg'], df['Alpha_AB_rad_s2'], label='Alpha_AB (Barra 1)', color='red')
axs[0, 1].plot(df['Theta1_deg'], df['Alpha_BD_rad_s2'], label='Alpha_BD (Barra 2)', color='brown', linestyle='--')
axs[0, 1].set_title('Aceleraciones Angulares')
axs[0, 1].set_ylabel('rad/s²')
axs[0, 1].set_xlim(0, 360)
axs[0, 1].legend()

# [1,0] ACELERACIONES LINEALES
axs[1, 0].plot(df['Theta1_deg'], df['Aceleracion_Sellador_D_m_s2'], label='Aceleración Sellador', color='magenta')
axs[1, 0].plot(df['Theta1_deg'], df['A_Coriolis_m_s2'], label='A Coriolis', color='orange')
axs[1, 0].plot(df['Theta1_deg'], df['A_deslizamiento_c_AB_m_s2'], label='A deslizamiento', color='purple')
axs[1, 0].set_title('Aceleraciones Lineales')
axs[1, 0].set_xlabel('Ángulo de entrada Theta 1 (Grados)')
axs[1, 0].set_ylabel('m/s²')
axs[1, 0].set_xlim(0, 360)
axs[1, 0].legend()

# [1,1] TORQUE DEL MOTOR (🛠️ ÚNICA SECCIÓN MODIFICADA CON ESCALA SIMÉTRICA LOGARÍTMICA)
axs[1, 1].plot(df['Theta1_deg'], df['Torque_Motor_Nm'], label='Torque Motor', color='purple', linewidth=2)
axs[1, 1].set_yscale('symlog', linthresh=0.005)  # <- Muestra el comportamiento fino (0.002 Nm) y el pico de sellado juntos
axs[1, 1].set_title('Torque Requerido (Cinética)')
axs[1, 1].set_xlabel('Ángulo de entrada Theta 1 (Grados)')
axs[1, 1].set_ylabel('N·m (Escala Symlog)')
axs[1, 1].set_xlim(0, 360)
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
