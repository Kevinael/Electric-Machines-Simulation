# -*- coding: utf-8 -*-
"""Agrupado - Simulação de Máquina de Indução Trifásica
Contém todos os experimentos: Partida, Carga e Parâmetros da Máquina
"""


# ==============================================================================
# EXPERIMENTOS DE PARTIDA
# ==============================================================================

# @title Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from IPython.display import display

"""---

# O Colab aqui apresentado tem o objetivo de simular situações específicas no funcionamento de uma máquina de indução trifásica.

# 1. Partida Direta (DOL - Direct On Line)

**Objetivo:** Analisar a corrente e o torque de partida diretos.

**Configuração:**

- **Modo de Operação:** Motor  
- **Tensão RMS de entrada** ($V_l$): 220.0 V  
- **Frequência** ($f$): 60.0 Hz  
- **Torque mecânico inicial** ($T_{l\_initial}$): 0.0 Nm  
- **Torque mecânico final** ($T_{l\_final}$): 80.0 Nm (valor nominal sugerido)  
- **Tempo de aplicação do torque mecânico** ($t_{carga}$): 0.1 s (aplicar torque logo no início)  
- **Referência para a transformada de Park:** Síncrona ($\omega_e$)

**Análise Esperada:** Observar picos elevados de corrente e torque na partida.

## 1.1. Alterações a serem feitas:
Na partida direta, a máquina começa com uma rotação nula e uma alimentação nominal entre os terminais.
$$ wr_{initial} = 0 RPM$$
$$V_l = 220 V$$
"""

# @title Parâmetros de simulação para partida direta
print('Simulação de máquina de indução gaiola de esquilo trifásica')
print('Partida direta (DOL - Direct On Line)')

## Simulation Parameters (using @param for interactive input)
print('\nParâmetros de simulação:')
tmax = 2  # @param {type:"slider", min:0.1, max:10.0, step:0.1, default:2.0}
print(f"tmax (Tempo de simulação) = {tmax} s")
h = 0.001  # @param {type:"slider", min:0.00001, max:0.001, step:0.00001, default:0.0001}
t_carga = 0.1  # @param {type:"slider", min:0.1, max:5.0, step:0.1, default:1.0}
print(f"t_2 (Tempo de aplicação do torque mecânico) = {t_carga} s")

## Motor Operation Mode (using @param for interactive input)
print('\nModo de operação da máquina de indução:')
mop_choice = "Motor"  # @param ["Motor", "Gerador conectado ao barramento infinito"]
mop = 1 if mop_choice == "Motor" else 2

## Three-Phase Induction Machine Simulation Parameters (using @param for interactive input)
print('\nParâmetros de dados da máquina de indução trifásica:')
Vl = 220  # @param {type:"slider", min:10.0, max:500.0, step:10.0, default:220.0}
print(f"Vl (Tensão RMS de linha) = {Vl} V")
f = 60.0  # @param {type:"slider", min:10.0, max:120.0, step:1.0, default:60.0}
print(f"f (Frequência) = {f} Hz")
Rs = 0.435  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.435}
print(f"Rs (Resistência do estator) = {Rs} Ω")
Rr = 0.816  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.816}
print(f"Rr (Resistência do rotor) = {Rr} Ω")
Xm = 26.13  # @param {type:"slider", min:1.0, max:100.0, step:0.1, default:26.13}
print(f"Xm (Reatância de magnetização) = {Xm} Ω")
Xls = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xls (Reatância de dispersão do estator) = {Xls} Ω")
Xlr = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xlr (Reatância de dispersão do rotor) = {Xlr} Ω")
p = 4  # @param {type:"slider", min:2, max:8, step:2, default:4}
print(f"p (Número de polos) = {p}")
J = 0.089  # @param {type:"slider", min:0.01, max:1.0, step:0.001, default:0.089}
print(f"J (Inércia do rotor) = {J} kg·m²")
Tl_initial = 0.0  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:0.0}
print(f"Tl_initial (Torque inicial) = {Tl_initial} Nm")
Tl_final = 80  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:80.0}
print(f"Tl_final (Torque final) = {Tl_final} Nm")
# Torque final nulo pois consideraremos a máquina apenas em vazio.
B = 0.0  # @param {type:"slider", min:0.0, max:1.0, step:0.001, default:0.0}
print(f"B (Campo magnético inicial) = {B} Nm·s/rad")


## Park Transform Reference Option (using @param for interactive input)
print('\nEscolha a velocidade angular de referência para a transformada de Park:')
ref_choice = "referência síncrona (we)"  # @param ["referência síncrona (we)", "referência rotórica (wr)", "referência estacionária (0)"]
if ref_choice == "referência síncrona (we)":
    ref = 1
elif ref_choice == "referência rotórica (wr)":
    ref = 2
else:
    ref = 3

if mop == 1:
    print("\nModo de operação: motor")
    Tl = Tl_initial
    Tl_2 = Tl_final
else:
    print("\nModo de operação: gerador conectado ao barramento infinito")
    Tl = -Tl_initial
    Tl_2 = -Tl_final

if ref == 1:
    print('Referência: síncrona (we)')
elif ref == 2:
    print('Referência: rotórica (wr)')
else:
    print('Referência: estacionária (0)')


# Definição das variáveis para iteração
PSIqs_initial = 0
PSIds_initial = 0
PSIqr_initial = 0
PSIdr_initial = 0
wr_initial = 0 # @param {"type":"slider","min":0,"max":3600,"step":50}
# Na partida, a rotação do eixo é nula.
initial_states = [PSIqs_initial, PSIds_initial, PSIqr_initial, PSIdr_initial, wr_initial]

# @title Simulador

# Cálculo de parâmetros
Xml = 1 / ((1 / Xm) + (1 / Xls) + (1 / Xlr))
wb = 2 * np.pi * f  # Base angular velocity
Vqr_squirrel_cage = 0
Vdr_squirrel_cage = 0 # Squirrel cage rotor

## State-Space Representation and ODEs
def induction_motor_equations(states, t, Vqs_t, Vds_t, current_Tl, w_ref_val):
    PSIqs, PSIds, PSIqr, PSIdr, wr = states

    # Dependent variables calculation
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    ids_val = (1 / Xls) * (PSIds - PSImd)
    iqs_val = (1 / Xls) * (PSIqs - PSImq)

    # Differential equations
    dPSIqs_dt = wb * (Vqs_t - (w_ref_val / wb) * PSIds + (Rs / Xls) * (PSImq - PSIqs))
    dPSIds_dt = wb * (Vds_t + (w_ref_val / wb) * PSIqs + (Rs / Xls) * (PSImd - PSIds))
    dPSIqr_dt = wb * (Vqr_squirrel_cage - ((w_ref_val - wr) / wb) * PSIdr + (Rr / Xlr) * (PSImq - PSIqr))
    dPSIdr_dt = wb * (Vdr_squirrel_cage + ((w_ref_val - wr) / wb) * PSIqr + (Rr / Xlr) * (PSImd - PSIdr))

    # Electromagnetic torque
    Tem_val = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs_val - PSIqs * ids_val)
    dwr_dt = (p / (2 * J)) * (Tem_val - current_Tl) - (B / J) * wr

    return [dPSIqs_dt, dPSIds_dt, dPSIqr_dt, dPSIdr_dt, dwr_dt]

## Simulation Loop (using odeint for integration)
t_values = np.arange(0, tmax, h)

# Store results
wr_results = []
Te_results = []
n_results = []
ids_results = []
iqs_results = []
idr_results = []
iqr_results = []
ias_results = []
ibs_results = []
ics_results = []
iar_results = []
ibr_results = []
icr_results = []
Va_results = []
Vb_results = []
Vc_results = []
Vds_results = []
Vqs_results = []

# To store the last `wr` value for `w_ref_val` if `ref == 2`
last_wr = wr_initial

for i, t_val in enumerate(t_values):
    # Determine mechanical torque
    if t_val >= t_carga:
        current_Tl = Tl_2
    else:
        current_Tl = Tl

    # Stator electrical components
    we_val = 2 * np.pi * f
    tetae = we_val * t_val

    # Define arbitrary reference for the transform based on input choice
    if ref == 1:
        w_ref_val = we_val
    elif ref == 2:
        w_ref_val = last_wr  # Use the last calculated wr
    else:
        w_ref_val = 0

    # Input voltage (infinite bus)
    Va_val = np.sqrt(2) * Vl * np.sin(tetae)
    Vb_val = np.sqrt(2) * Vl * np.sin(tetae - ((2 * np.pi) / 3))
    Vc_val = np.sqrt(2) * Vl * np.sin(tetae + ((2 * np.pi) / 3))

    Vaf_val = np.sqrt(3 / 2) * (Va_val + (-1 / 2) * Vb_val + (-1 / 2) * Vc_val)
    Vbt_val = np.sqrt(3 / 2) * (0 * Va_val + (np.sqrt(3) / 2) * Vb_val + (-np.sqrt(3) / 2) * Vc_val)

    Vds_val = np.cos(tetae) * Vaf_val + np.sin(tetae) * Vbt_val
    Vqs_val = -np.sin(tetae) * Vaf_val + np.cos(tetae) * Vbt_val

    # Integrate one step using odeint
    sol = odeint(induction_motor_equations, initial_states, [t_val, t_val + h], args=(Vqs_val, Vds_val, current_Tl, w_ref_val))
    PSIqs, PSIds, PSIqr, PSIdr, wr = sol[1] # Take the results at t_val + h
    initial_states = [PSIqs, PSIds, PSIqr, PSIdr, wr] # Update initial states for next iteration
    last_wr = wr # Update last_wr for next iteration if ref == 2

    # Store mechanical angular velocity
    wr_results.append(wr)

    # Calculate and store speed in RPM
    n = (120 / p) * (wr / (2 * np.pi))
    n_results.append(n)

    # Calculate currents and store
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)

    ids = (1 / Xls) * (PSIds - PSImd)
    iqs = (1 / Xls) * (PSIqs - PSImq)
    idr = (1 / Xlr) * (PSIdr - PSImd)
    iqr = (1 / Xlr) * (PSIqr - PSImq)

    ids_results.append(ids)
    iqs_results.append(iqs)
    idr_results.append(idr)
    iqr_results.append(iqr)

    # Calculate and store electromagnetic torque
    Te = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs - PSIqs * ids)
    Te_results.append(Te)

    # Stator and Rotor currents (alpha-beta and abc)
    tetar = wr * t_val # Using t_val for tetar calculation here, consistent with Scilab code

    iafs = np.cos(tetae) * ids - np.sin(tetae) * iqs
    ibts = np.sin(tetae) * ids + np.cos(tetae) * iqs

    iafr = np.cos(tetar) * idr - np.sin(tetar) * iqr
    ibtr = np.sin(tetar) * idr + np.cos(tetar) * iqr # Corrected to tetar, assuming original was a typo

    ias = np.sqrt(3/2) * (1 * iafs + 0 * ibts)
    ibs = np.sqrt(3/2) * ((-1/2) * iafs + (np.sqrt(3)/2) * ibts)
    ics = np.sqrt(3/2) * ((-1/2) * iafs + (-np.sqrt(3)/2) * ibts)

    iar = np.sqrt(3/2) * (1 * iafr + 0 * ibtr)
    ibr = np.sqrt(3/2) * ((-1/2) * iafr + (np.sqrt(3)/2) * ibtr)
    icr = np.sqrt(3/2) * ((-1/2) * iafr + (-np.sqrt(3)/2) * ibtr)

    ias_results.append(ias)
    ibs_results.append(ibs)
    ics_results.append(ics)
    iar_results.append(iar)
    ibr_results.append(ibr)
    icr_results.append(icr)

    Va_results.append(Va_val)
    Vb_results.append(Vb_val)
    Vc_results.append(Vc_val)
    Vds_results.append(Vds_val)
    Vqs_results.append(Vqs_val)


# Convert lists to numpy arrays for easier plotting
wr_results = np.array(wr_results)
Te_results = np.array(Te_results)
n_results = np.array(n_results)
ids_results = np.array(ids_results)
iqs_results = np.array(iqs_results)
idr_results = np.array(idr_results)
iqr_results = np.array(iqr_results)
ias_results = np.array(ias_results)
ibs_results = np.array(ibs_results)
ics_results = np.array(ics_results)
iar_results = np.array(iar_results)
ibr_results = np.array(ibr_results)
icr_results = np.array(icr_results)
Va_results = np.array(Va_results)
Vb_results = np.array(Vb_results)
Vc_results = np.array(Vc_results)
Vds_results = np.array(Vds_results)
Vqs_results = np.array(Vqs_results)


## Steady-State Values
#print("\nValores (finais) em regime permanente das variáveis:")
#print("s f_r ids iqs n")
# Assuming the last values are steady-state
s_ss = (wb - wr_results[-1]) / wb
fr_ss = wr_results[-1] / (2 * np.pi)
ids_ss = ids_results[-1]
iqs_ss = iqs_results[-1]
n_ss = n_results[-1]
#print(f"{s_ss:.4f} {fr_ss:.4f} {ids_ss:.4f} {iqs_ss:.4f} {n_ss:.4f}")

## Plotting
# Torque
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Te_results, label='$T_e$')
plt.title('$T_e$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Torque (Nm)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda fora
plt.grid(True)

# Velocidade
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 2)
plt.plot(t_values, n_results, label='n (RPM)')
plt.title('n (RPM)', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Velocidade (RPM)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda fora
plt.grid(True)
plt.show()

# Stator Voltages
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Va_results, label='$V_a$')
plt.plot(t_values, Vb_results, label='$V_b$')
plt.plot(t_values, Vc_results, label='$V_c$')
plt.title('$V_a$ x $V_b$ x $V_c$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Tensão (V)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)

# Stator Currents
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 2)
plt.plot(t_values, ias_results, label='$i_{as}$')
plt.plot(t_values, ibs_results, label='$i_{bs}$')
plt.plot(t_values, ics_results, label='$i_{cs}$')
plt.title('$I_{as}$ x $I_{bs}$ x $I_{cs}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# Rotor Currents
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, iar_results, label='$i_{ar}$')
plt.plot(t_values, ibr_results, label='$i_{br}$')
plt.plot(t_values, icr_results, label='$i_{cr}$')
plt.title('$I_{ar}$ x $I_{br}$ x $I_{cr}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

"""## Resultados:
Para a partida direta, percebemos uma alta corrente, se comparadas com o valor final, para o Rotor e para o Estator. Além disso, há um elevado torque de partida, mesmo que ainda não haja carga no eixo.
"""

# Obter os valores máximos:
print('Valores máximos para partida Direta - Direct on Line')
valores_max_DOL = {
  'Corrente máxima na fase A do Estator (A)': max(ias_results),
  'Corrente máxima na fase B do Estator (A)': max(ibs_results),
  'Corrente máxima na fase C do Estator (A)': max(ics_results),
    'Corrente máxima na fase A do Rotor (A)': max(iar_results),
    'Corrente máxima na fase B do Rotor (A)': max(ibr_results),
    'Corrente máxima na fase C do Rotor (A)': max(icr_results),
  'Torque máximo desenvolvido no eixo (N.m)': max(Te_results),
}

# Resultados máximos:
for item, valor in valores_max_DOL.items():
    print(f'{item}: {valor:.3f}')

"""## 2. Partida Y−Δ (Estrela-Triângulo)

**Objetivo:** Demonstrar a redução da corrente de partida e do torque de partida em comparação com a partida direta.

**Considerações:**  
No modelo $0dq$ simplificado, a partida Y−Δ é simulada reduzindo a tensão de entrada durante o período inicial de partida (equivalente a $\frac{1}{\sqrt{3}}$ da tensão nominal).

**Configuração:**

- **Modo de Operação:** Motor  
- **Tensão RMS de entrada** ($V_L$): Iniciar com aproximadamente 127.0 V ($\frac{220}{\sqrt{3}}$) por um tempo (ex: 0.5 s), depois alterar para 220.0 V  
- **Frequência** ($f$): 60.0 Hz  
- **Torque mecânico inicial** ($T_{l\_initial}$): 0.0 Nm  
- **Torque mecânico final** ($T_{l\_final}$): 80.0 Nm  
- **Tempo de aplicação do torque mecânico** ($t_{carga}$): 0.1 s  
- **Referência para a transformada de Park:** Síncrona ($\omega_e$)

**Análise Esperada:**  
Comparar as curvas de corrente e torque com a partida direta, observando a redução dos picos.

## 2.1 Alterações a serem feitas:
O Valor inicial da tensão nos parâmetros deve ser de $\frac{220}{\sqrt{3}}$ V. Após t_2 segundos, o valor deve ser o nominal que é de $220$ V.
"""

# @title Parâmetros de simulação para partida Y−Δ (Estrela-Triângulo)
print('Simulação de máquina de indução gaiola de esquilo trifásica')
print('Partida direta (DOL - Direct On Line)')

## Simulation Parameters (using @param for interactive input)
print('\nParâmetros de simulação:')
tmax = 2  # @param {type:"slider", min:0.1, max:10.0, step:0.1, default:2.0}
print(f"tmax (Tempo de simulação) = {tmax} s")
h = 0.001  # @param {type:"slider", min:0.00001, max:0.001, step:0.00001, default:0.0001}
t_2 = 0.5  # @param {type:"slider", min:0.1, max:5.0, step:0.1, default:1.0}
t_carga = 0.1 # s (Momento em que a carga é aplicada)
print(f"t_carga (Tempo de aplicação da carga) = {t_carga} s")
print(f"t_2 (Tempo de troca para Delta) = {t_2} s")

## Motor Operation Mode (using @param for interactive input)
print('\nModo de operação da máquina de indução:')
mop_choice = "Motor"  # @param ["Motor", "Gerador conectado ao barramento infinito"]
mop = 1 if mop_choice == "Motor" else 2

## Three-Phase Induction Machine Simulation Parameters (using @param for interactive input)
print('\nParâmetros de dados da máquina de indução trifásica:')
Vl = 220  # @param {type:"slider", min:10.0, max:500.0, step:10.0, default:220.0}
print(f"Vl (Tensão RMS de linha) = {Vl} V")

Vl_fase_Y = Vl / np.sqrt(3)
print(f"Vl_fase_Y (Tensão RMS de linha na fase Y) = {Vl_fase_Y:.3f} V")

f = 60.0  # @param {type:"slider", min:10.0, max:120.0, step:1.0, default:60.0}
print(f"f (Frequência) = {f} Hz")
Rs = 0.435  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.435}
print(f"Rs (Resistência do estator) = {Rs} Ω")
Rr = 0.816  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.816}
print(f"Rr (Resistência do rotor) = {Rr} Ω")
Xm = 26.13  # @param {type:"slider", min:1.0, max:100.0, step:0.1, default:26.13}
print(f"Xm (Reatância de magnetização) = {Xm} Ω")
Xls = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xls (Reatância de dispersão do estator) = {Xls} Ω")
Xlr = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xlr (Reatância de dispersão do rotor) = {Xlr} Ω")
p = 4  # @param {type:"slider", min:2, max:8, step:2, default:4}
print(f"p (Número de polos) = {p}")
J = 0.089  # @param {type:"slider", min:0.01, max:1.0, step:0.001, default:0.089}
print(f"J (Inércia do rotor) = {J} kg·m²")
Tl_initial = 0.0  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:0.0}
print(f"Tl_initial (Torque inicial) = {Tl_initial} Nm")
Tl_final = 80  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:80.0}
print(f"Tl_final (Torque final) = {Tl_final} Nm")
# Torque final nulo pois consideraremos a máquina apenas em vazio.
B = 0.0  # @param {type:"slider", min:0.0, max:1.0, step:0.001, default:0.0}
print(f"B (Campo magnético inicial) = {B} Nm·s/rad")


## Park Transform Reference Option (using @param for interactive input)
print('\nEscolha a velocidade angular de referência para a transformada de Park:')
ref_choice = "referência síncrona (we)"  # @param ["referência síncrona (we)", "referência rotórica (wr)", "referência estacionária (0)"]
if ref_choice == "referência síncrona (we)":
    ref = 1
elif ref_choice == "referência rotórica (wr)":
    ref = 2
else:
    ref = 3

if mop == 1:
    print("\nModo de operação: motor")
    Tl = Tl_initial
    Tl_2 = Tl_final
else:
    print("\nModo de operação: gerador conectado ao barramento infinito")
    Tl = -Tl_initial
    Tl_2 = -Tl_final

if ref == 1:
    print('Referência: síncrona (we)')
elif ref == 2:
    print('Referência: rotórica (wr)')
else:
    print('Referência: estacionária (0)')


# Definição das variáveis para iteração
PSIqs_initial = 0
PSIds_initial = 0
PSIqr_initial = 0
PSIdr_initial = 0
wr_initial = 0 # @param {"type":"slider","min":0,"max":3600,"step":50}
# Na partida, a rotação do eixo é nula.
initial_states = [PSIqs_initial, PSIds_initial, PSIqr_initial, PSIdr_initial, wr_initial]

# @title Simulador Y−Δ (Estrela-Triângulo)

# Cálculo de parâmetros
Xml = 1 / ((1 / Xm) + (1 / Xls) + (1 / Xlr))
wb = 2 * np.pi * f  # Base angular velocity
Vqr_squirrel_cage = 0
Vdr_squirrel_cage = 0 # Squirrel cage rotor

## State-Space Representation and ODEs
def induction_motor_equations(states, t, Vqs_t, Vds_t, current_Tl, w_ref_val):
    PSIqs, PSIds, PSIqr, PSIdr, wr = states

    # Dependent variables calculation
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    ids_val = (1 / Xls) * (PSIds - PSImd)
    iqs_val = (1 / Xls) * (PSIqs - PSImq)

    # Differential equations
    dPSIqs_dt = wb * (Vqs_t - (w_ref_val / wb) * PSIds + (Rs / Xls) * (PSImq - PSIqs))
    dPSIds_dt = wb * (Vds_t + (w_ref_val / wb) * PSIqs + (Rs / Xls) * (PSImd - PSIds))
    dPSIqr_dt = wb * (Vqr_squirrel_cage - ((w_ref_val - wr) / wb) * PSIdr + (Rr / Xlr) * (PSImq - PSIqr))
    dPSIdr_dt = wb * (Vdr_squirrel_cage + ((w_ref_val - wr) / wb) * PSIqr + (Rr / Xlr) * (PSImd - PSIdr))

    # Electromagnetic torque
    Tem_val = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs_val - PSIqs * ids_val)
    dwr_dt = (p / (2 * J)) * (Tem_val - current_Tl) - (B / J) * wr

    return [dPSIqs_dt, dPSIds_dt, dPSIqr_dt, dPSIdr_dt, dwr_dt]

## Simulation Loop (using odeint for integration)
t_values = np.arange(0, tmax, h)

# Store results
wr_results = []
Te_results = []
n_results = []
ids_results = []
iqs_results = []
idr_results = []
iqr_results = []
ias_results = []
ibs_results = []
ics_results = []
iar_results = []
ibr_results = []
icr_results = []
Va_results = []
Vb_results = []
Vc_results = []
Vds_results = []
Vqs_results = []

# To store the last `wr` value for `w_ref_val` if `ref == 2`
last_wr = wr_initial

for i, t_val in enumerate(t_values):
    # Determine mechanical torque
    if t_val >= t_2:
        V_aplicada = Vl
    else:
        current_Tl = Tl
        V_aplicada = Vl_fase_Y

    if t_val >= t_carga:
        current_Tl = Tl_2


    # Stator electrical components
    we_val = 2 * np.pi * f
    tetae = we_val * t_val

    # Define arbitrary reference for the transform based on input choice
    if ref == 1:
        w_ref_val = we_val
    elif ref == 2:
        w_ref_val = last_wr  # Use the last calculated wr
    else:
        w_ref_val = 0

    # Input voltage (infinite bus)
    Va_val = np.sqrt(2) * V_aplicada * np.sin(tetae)
    Vb_val = np.sqrt(2) * V_aplicada * np.sin(tetae - ((2 * np.pi) / 3))
    Vc_val = np.sqrt(2) * V_aplicada * np.sin(tetae + ((2 * np.pi) / 3))

    Vaf_val = np.sqrt(3 / 2) * (Va_val + (-1 / 2) * Vb_val + (-1 / 2) * Vc_val)
    Vbt_val = np.sqrt(3 / 2) * (0 * Va_val + (np.sqrt(3) / 2) * Vb_val + (-np.sqrt(3) / 2) * Vc_val)

    Vds_val = np.cos(tetae) * Vaf_val + np.sin(tetae) * Vbt_val
    Vqs_val = -np.sin(tetae) * Vaf_val + np.cos(tetae) * Vbt_val

    # Integrate one step using odeint
    sol = odeint(induction_motor_equations, initial_states, [t_val, t_val + h], args=(Vqs_val, Vds_val, current_Tl, w_ref_val))
    PSIqs, PSIds, PSIqr, PSIdr, wr = sol[1] # Take the results at t_val + h
    initial_states = [PSIqs, PSIds, PSIqr, PSIdr, wr] # Update initial states for next iteration
    last_wr = wr # Update last_wr for next iteration if ref == 2

    # Store mechanical angular velocity
    wr_results.append(wr)

    # Calculate and store speed in RPM
    n = (120 / p) * (wr / (2 * np.pi))
    n_results.append(n)

    # Calculate currents and store
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)

    ids = (1 / Xls) * (PSIds - PSImd)
    iqs = (1 / Xls) * (PSIqs - PSImq)
    idr = (1 / Xlr) * (PSIdr - PSImd)
    iqr = (1 / Xlr) * (PSIqr - PSImq)

    ids_results.append(ids)
    iqs_results.append(iqs)
    idr_results.append(idr)
    iqr_results.append(iqr)

    # Calculate and store electromagnetic torque
    Te = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs - PSIqs * ids)
    Te_results.append(Te)

    # Stator and Rotor currents (alpha-beta and abc)
    tetar = wr * t_val # Using t_val for tetar calculation here, consistent with Scilab code

    iafs = np.cos(tetae) * ids - np.sin(tetae) * iqs
    ibts = np.sin(tetae) * ids + np.cos(tetae) * iqs

    iafr = np.cos(tetar) * idr - np.sin(tetar) * iqr
    ibtr = np.sin(tetar) * idr + np.cos(tetar) * iqr # Corrected to tetar, assuming original was a typo

    ias = np.sqrt(3/2) * (1 * iafs + 0 * ibts)
    ibs = np.sqrt(3/2) * ((-1/2) * iafs + (np.sqrt(3)/2) * ibts)
    ics = np.sqrt(3/2) * ((-1/2) * iafs + (-np.sqrt(3)/2) * ibts)

    iar = np.sqrt(3/2) * (1 * iafr + 0 * ibtr)
    ibr = np.sqrt(3/2) * ((-1/2) * iafr + (np.sqrt(3)/2) * ibtr)
    icr = np.sqrt(3/2) * ((-1/2) * iafr + (-np.sqrt(3)/2) * ibtr)

    ias_results.append(ias)
    ibs_results.append(ibs)
    ics_results.append(ics)
    iar_results.append(iar)
    ibr_results.append(ibr)
    icr_results.append(icr)

    Va_results.append(Va_val)
    Vb_results.append(Vb_val)
    Vc_results.append(Vc_val)
    Vds_results.append(Vds_val)
    Vqs_results.append(Vqs_val)


# Convert lists to numpy arrays for easier plotting
wr_results = np.array(wr_results)
Te_results = np.array(Te_results)
n_results = np.array(n_results)
ids_results = np.array(ids_results)
iqs_results = np.array(iqs_results)
idr_results = np.array(idr_results)
iqr_results = np.array(iqr_results)
ias_results = np.array(ias_results)
ibs_results = np.array(ibs_results)
ics_results = np.array(ics_results)
iar_results = np.array(iar_results)
ibr_results = np.array(ibr_results)
icr_results = np.array(icr_results)
Va_results = np.array(Va_results)
Vb_results = np.array(Vb_results)
Vc_results = np.array(Vc_results)
Vds_results = np.array(Vds_results)
Vqs_results = np.array(Vqs_results)


## Steady-State Values
print("\nValores (finais) em regime permanente das variáveis:")
print("s f_r ids iqs n")
# Assuming the last values are steady-state
s_ss = (wb - wr_results[-1]) / wb
fr_ss = wr_results[-1] / (2 * np.pi)
ids_ss = ids_results[-1]
iqs_ss = iqs_results[-1]
n_ss = n_results[-1]
print(f"{s_ss:.4f} {fr_ss:.4f} {ids_ss:.4f} {iqs_ss:.4f} {n_ss:.4f}")

# Torque
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Te_results, label='$T_e$')
plt.title('$T_e$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Torque (Nm)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda fora
plt.grid(True)

# Velocidade
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 2)
plt.plot(t_values, n_results, label='n (RPM)')
plt.title('n (RPM)', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Velocidade (RPM)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda fora
plt.grid(True)
plt.show()

# Stator Voltages
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Va_results, label='$V_a$')
plt.plot(t_values, Vb_results, label='$V_b$')
plt.plot(t_values, Vc_results, label='$V_c$')
plt.title('$V_a$ x $V_b$ x $V_c$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Tensão (V)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)

# Stator Currents
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 2)
plt.plot(t_values, ias_results, label='$i_{as}$')
plt.plot(t_values, ibs_results, label='$i_{bs}$')
plt.plot(t_values, ics_results, label='$i_{cs}$')
plt.title('$I_{as}$ x $I_{bs}$ x $I_{cs}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# Rotor Currents
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, iar_results, label='$i_{ar}$')
plt.plot(t_values, ibr_results, label='$i_{br}$')
plt.plot(t_values, icr_results, label='$i_{cr}$')
plt.title('$I_{ar}$ x $I_{br}$ x $I_{cr}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# @title Resultados e comparação
print('Valores máximos para partida Y−Δ (Estrela-Triângulo)')

# Valores máximos
valores_max_YD = {
    'Corrente máxima na fase A do Estator (A)': max(ias_results),
    'Corrente máxima na fase B do Estator (A)': max(ibs_results),
    'Corrente máxima na fase C do Estator (A)': max(ics_results),
    'Corrente máxima na fase A do Rotor (A)'  : max(iar_results),
    'Corrente máxima na fase B do Rotor (A)'  : max(ibr_results),
    'Corrente máxima na fase C do Rotor (A)'  : max(icr_results),
    'Torque máximo desenvolvido no eixo (N.m)': max(Te_results),
}

# Comparação e impressão
for item, valor_YD in valores_max_YD.items():
    valor_DOL = valores_max_DOL[item]  # pega o valor correspondente da partida direta
    relacao = valor_YD / valor_DOL * 100
    #diferenca_percentual = (1 - relacao) * 100
    print(f'{item}: {valor_YD:.3f} ({relacao:.1f}% do valor máximo em Partida Direta)')

"""## 3. Partida com Autotransformador (Compensadora)

**Objetivo:** Simular a partida com um autotransformador para reduzir a tensão de partida.

**Considerações:**  
Similar à Y−Δ, a partida compensadora é simulada reduzindo a tensão de entrada para um percentual (ex: 50% ou 70%) por um tempo, e depois aumentando para a tensão nominal.

**Configuração:**

- **Modo de Operação:** Motor  
- **Tensão RMS de entrada** ($V_L$): Iniciar com 110.0 V ($220 \times 0.5$) por um tempo (ex: 0.5 s), depois alterar para 220.0 V  
- **Frequência** ($f$): 60.0 Hz  
- **Torque mecânico inicial** ($T_{l\_initial}$): 0.0 Nm  
- **Torque mecânico final** ($T_{l\_final}$): 80.0 Nm  
- **Tempo de aplicação do torque mecânico** ($t_2$): 0.1 s  
- **Referência para a transformada de Park:** Síncrona ($\omega_e$)

**Análise Esperada:**  
Comparar os perfis de corrente e torque com as partidas direta e Y−Δ, notando a redução dependendo do percentual de tensão aplicado.
## 3.1. Alterações a serem feitas:
Similar ao Y-D. O valor inicial da tensão deve ser menor que o nominal

"""

# @title Parâmetros de simulação para partida Compensadora
print('Simulação de máquina de indução gaiola de esquilo trifásica')
print('Partida Compensadora')

## Simulation Parameters (using @param for interactive input)
print('\nParâmetros de simulação:')
tmax = 2  # @param {type:"slider", min:0.1, max:10.0, step:0.1, default:2.0}
print(f"tmax (Tempo de simulação) = {tmax} s")
h = 0.001  # @param {type:"slider", min:0.00001, max:0.001, step:0.00001, default:0.0001}
t_2 = 0.5  # @param {type:"slider", min:0.1, max:5.0, step:0.1, default:1.0}
t_carga = 0.1 # s (Momento em que a carga é aplicada)
print(f"t_carga (Tempo de aplicação da carga) = {t_carga} s")
print(f"t_2 (Tempo de para ação da compensação) = {t_2} s")

## Motor Operation Mode (using @param for interactive input)
print('\nModo de operação da máquina de indução:')
mop_choice = "Motor"  # @param ["Motor", "Gerador conectado ao barramento infinito"]
mop = 1 if mop_choice == "Motor" else 2

## Three-Phase Induction Machine Simulation Parameters (using @param for interactive input)
print('\nParâmetros de dados da máquina de indução trifásica:')
Vl = 220  # @param {type:"slider", min:10.0, max:500.0, step:10.0, default:220.0}
print(f"Vl (Tensão RMS de linha) = {Vl} V")
Regulação_tensão = 0.5 # @param {type:"slider", min:0.1, max:1, step:0.01, default:0.5}
Vl_aplicada_inicialmente = Vl * Regulação_tensão
print(f"Vl_aplicada_inicialmente (Tensão RMS aplicada incialmente) = {Vl_fase_Y:.3f} V")

f = 60.0  # @param {type:"slider", min:10.0, max:120.0, step:1.0, default:60.0}
print(f"f (Frequência) = {f} Hz")
Rs = 0.435  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.435}
print(f"Rs (Resistência do estator) = {Rs} Ω")
Rr = 0.816  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.816}
print(f"Rr (Resistência do rotor) = {Rr} Ω")
Xm = 26.13  # @param {type:"slider", min:1.0, max:100.0, step:0.1, default:26.13}
print(f"Xm (Reatância de magnetização) = {Xm} Ω")
Xls = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xls (Reatância de dispersão do estator) = {Xls} Ω")
Xlr = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xlr (Reatância de dispersão do rotor) = {Xlr} Ω")
p = 4  # @param {type:"slider", min:2, max:8, step:2, default:4}
print(f"p (Número de polos) = {p}")
J = 0.089  # @param {type:"slider", min:0.01, max:1.0, step:0.001, default:0.089}
print(f"J (Inércia do rotor) = {J} kg·m²")
Tl_initial = 0.0  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:0.0}
print(f"Tl_initial (Torque inicial) = {Tl_initial} Nm")
Tl_final = 80  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:80.0}
print(f"Tl_final (Torque final) = {Tl_final} Nm")
# Torque final nulo pois consideraremos a máquina apenas em vazio.
B = 0.0  # @param {type:"slider", min:0.0, max:1.0, step:0.001, default:0.0}
print(f"B (Campo magnético inicial) = {B} Nm·s/rad")


## Park Transform Reference Option (using @param for interactive input)
print('\nEscolha a velocidade angular de referência para a transformada de Park:')
ref_choice = "referência síncrona (we)"  # @param ["referência síncrona (we)", "referência rotórica (wr)", "referência estacionária (0)"]
if ref_choice == "referência síncrona (we)":
    ref = 1
elif ref_choice == "referência rotórica (wr)":
    ref = 2
else:
    ref = 3

if mop == 1:
    print("\nModo de operação: motor")
    Tl = Tl_initial
    Tl_2 = Tl_final
else:
    print("\nModo de operação: gerador conectado ao barramento infinito")
    Tl = -Tl_initial
    Tl_2 = -Tl_final

if ref == 1:
    print('Referência: síncrona (we)')
elif ref == 2:
    print('Referência: rotórica (wr)')
else:
    print('Referência: estacionária (0)')


# Definição das variáveis para iteração
PSIqs_initial = 0
PSIds_initial = 0
PSIqr_initial = 0
PSIdr_initial = 0
wr_initial = 0 # @param {"type":"slider","min":0,"max":3600,"step":50}
# Na partida, a rotação do eixo é nula.
initial_states = [PSIqs_initial, PSIds_initial, PSIqr_initial, PSIdr_initial, wr_initial]

# @title Simulador

# Cálculo de parâmetros
Xml = 1 / ((1 / Xm) + (1 / Xls) + (1 / Xlr))
wb = 2 * np.pi * f  # Base angular velocity
Vqr_squirrel_cage = 0
Vdr_squirrel_cage = 0 # Squirrel cage rotor

## State-Space Representation and ODEs
def induction_motor_equations(states, t, Vqs_t, Vds_t, current_Tl, w_ref_val):
    PSIqs, PSIds, PSIqr, PSIdr, wr = states

    # Dependent variables calculation
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    ids_val = (1 / Xls) * (PSIds - PSImd)
    iqs_val = (1 / Xls) * (PSIqs - PSImq)

    # Differential equations
    dPSIqs_dt = wb * (Vqs_t - (w_ref_val / wb) * PSIds + (Rs / Xls) * (PSImq - PSIqs))
    dPSIds_dt = wb * (Vds_t + (w_ref_val / wb) * PSIqs + (Rs / Xls) * (PSImd - PSIds))
    dPSIqr_dt = wb * (Vqr_squirrel_cage - ((w_ref_val - wr) / wb) * PSIdr + (Rr / Xlr) * (PSImq - PSIqr))
    dPSIdr_dt = wb * (Vdr_squirrel_cage + ((w_ref_val - wr) / wb) * PSIqr + (Rr / Xlr) * (PSImd - PSIdr))

    # Electromagnetic torque
    Tem_val = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs_val - PSIqs * ids_val)
    dwr_dt = (p / (2 * J)) * (Tem_val - current_Tl) - (B / J) * wr

    return [dPSIqs_dt, dPSIds_dt, dPSIqr_dt, dPSIdr_dt, dwr_dt]

## Simulation Loop (using odeint for integration)
t_values = np.arange(0, tmax, h)

# Store results
wr_results = []
Te_results = []
n_results = []
ids_results = []
iqs_results = []
idr_results = []
iqr_results = []
ias_results = []
ibs_results = []
ics_results = []
iar_results = []
ibr_results = []
icr_results = []
Va_results = []
Vb_results = []
Vc_results = []
Vds_results = []
Vqs_results = []

# To store the last `wr` value for `w_ref_val` if `ref == 2`
last_wr = wr_initial

for i, t_val in enumerate(t_values):
    # Determine mechanical torque
    if t_val >= t_2:
        V_aplicada = Vl
    else:
        current_Tl = Tl
        V_aplicada = Vl_aplicada_inicialmente

    if t_val >= t_carga:
        current_Tl = Tl_2


    # Stator electrical components
    we_val = 2 * np.pi * f
    tetae = we_val * t_val

    # Define arbitrary reference for the transform based on input choice
    if ref == 1:
        w_ref_val = we_val
    elif ref == 2:
        w_ref_val = last_wr  # Use the last calculated wr
    else:
        w_ref_val = 0

    # Input voltage (infinite bus)
    Va_val = np.sqrt(2) * V_aplicada * np.sin(tetae)
    Vb_val = np.sqrt(2) * V_aplicada * np.sin(tetae - ((2 * np.pi) / 3))
    Vc_val = np.sqrt(2) * V_aplicada * np.sin(tetae + ((2 * np.pi) / 3))

    Vaf_val = np.sqrt(3 / 2) * (Va_val + (-1 / 2) * Vb_val + (-1 / 2) * Vc_val)
    Vbt_val = np.sqrt(3 / 2) * (0 * Va_val + (np.sqrt(3) / 2) * Vb_val + (-np.sqrt(3) / 2) * Vc_val)

    Vds_val = np.cos(tetae) * Vaf_val + np.sin(tetae) * Vbt_val
    Vqs_val = -np.sin(tetae) * Vaf_val + np.cos(tetae) * Vbt_val

    # Integrate one step using odeint
    sol = odeint(induction_motor_equations, initial_states, [t_val, t_val + h], args=(Vqs_val, Vds_val, current_Tl, w_ref_val))
    PSIqs, PSIds, PSIqr, PSIdr, wr = sol[1] # Take the results at t_val + h
    initial_states = [PSIqs, PSIds, PSIqr, PSIdr, wr] # Update initial states for next iteration
    last_wr = wr # Update last_wr for next iteration if ref == 2

    # Store mechanical angular velocity
    wr_results.append(wr)

    # Calculate and store speed in RPM
    n = (120 / p) * (wr / (2 * np.pi))
    n_results.append(n)

    # Calculate currents and store
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)

    ids = (1 / Xls) * (PSIds - PSImd)
    iqs = (1 / Xls) * (PSIqs - PSImq)
    idr = (1 / Xlr) * (PSIdr - PSImd)
    iqr = (1 / Xlr) * (PSIqr - PSImq)

    ids_results.append(ids)
    iqs_results.append(iqs)
    idr_results.append(idr)
    iqr_results.append(iqr)

    # Calculate and store electromagnetic torque
    Te = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs - PSIqs * ids)
    Te_results.append(Te)

    # Stator and Rotor currents (alpha-beta and abc)
    tetar = wr * t_val # Using t_val for tetar calculation here, consistent with Scilab code

    iafs = np.cos(tetae) * ids - np.sin(tetae) * iqs
    ibts = np.sin(tetae) * ids + np.cos(tetae) * iqs

    iafr = np.cos(tetar) * idr - np.sin(tetar) * iqr
    ibtr = np.sin(tetar) * idr + np.cos(tetar) * iqr # Corrected to tetar, assuming original was a typo

    ias = np.sqrt(3/2) * (1 * iafs + 0 * ibts)
    ibs = np.sqrt(3/2) * ((-1/2) * iafs + (np.sqrt(3)/2) * ibts)
    ics = np.sqrt(3/2) * ((-1/2) * iafs + (-np.sqrt(3)/2) * ibts)

    iar = np.sqrt(3/2) * (1 * iafr + 0 * ibtr)
    ibr = np.sqrt(3/2) * ((-1/2) * iafr + (np.sqrt(3)/2) * ibtr)
    icr = np.sqrt(3/2) * ((-1/2) * iafr + (-np.sqrt(3)/2) * ibtr)

    ias_results.append(ias)
    ibs_results.append(ibs)
    ics_results.append(ics)
    iar_results.append(iar)
    ibr_results.append(ibr)
    icr_results.append(icr)

    Va_results.append(Va_val)
    Vb_results.append(Vb_val)
    Vc_results.append(Vc_val)
    Vds_results.append(Vds_val)
    Vqs_results.append(Vqs_val)


# Convert lists to numpy arrays for easier plotting
wr_results = np.array(wr_results)
Te_results = np.array(Te_results)
n_results = np.array(n_results)
ids_results = np.array(ids_results)
iqs_results = np.array(iqs_results)
idr_results = np.array(idr_results)
iqr_results = np.array(iqr_results)
ias_results = np.array(ias_results)
ibs_results = np.array(ibs_results)
ics_results = np.array(ics_results)
iar_results = np.array(iar_results)
ibr_results = np.array(ibr_results)
icr_results = np.array(icr_results)
Va_results = np.array(Va_results)
Vb_results = np.array(Vb_results)
Vc_results = np.array(Vc_results)
Vds_results = np.array(Vds_results)
Vqs_results = np.array(Vqs_results)


## Steady-State Values
print("\nValores (finais) em regime permanente das variáveis:")
print("s f_r ids iqs n")
# Assuming the last values are steady-state
s_ss = (wb - wr_results[-1]) / wb
fr_ss = wr_results[-1] / (2 * np.pi)
ids_ss = ids_results[-1]
iqs_ss = iqs_results[-1]
n_ss = n_results[-1]
print(f"{s_ss:.4f} {fr_ss:.4f} {ids_ss:.4f} {iqs_ss:.4f} {n_ss:.4f}")

# Torque
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Te_results, label='$T_e$')
plt.title('$T_e$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Torque (Nm)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda fora
plt.grid(True)

# Velocidade
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 2)
plt.plot(t_values, n_results, label='n (RPM)')
plt.title('n (RPM)', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Velocidade (RPM)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda fora
plt.grid(True)
plt.show()

# Stator Voltages
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Va_results, label='$V_a$')
plt.plot(t_values, Vb_results, label='$V_b$')
plt.plot(t_values, Vc_results, label='$V_c$')
plt.title('$V_a$ x $V_b$ x $V_c$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Tensão (V)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)

# Stator Currents
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 2)
plt.plot(t_values, ias_results, label='$i_{as}$')
plt.plot(t_values, ibs_results, label='$i_{bs}$')
plt.plot(t_values, ics_results, label='$i_{cs}$')
plt.title('$I_{as}$ x $I_{bs}$ x $I_{cs}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# Rotor Currents
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, iar_results, label='$i_{ar}$')
plt.plot(t_values, ibr_results, label='$i_{br}$')
plt.plot(t_values, icr_results, label='$i_{cr}$')
plt.title('$I_{ar}$ x $I_{br}$ x $I_{cr}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# @title Resultados e comparação
print('Valores máximos para partida com Autotransformador')

# Valores máximos
valores_max_AT = {
    'Corrente máxima na fase A do Estator (A)': max(ias_results),
    'Corrente máxima na fase B do Estator (A)': max(ibs_results),
    'Corrente máxima na fase C do Estator (A)': max(ics_results),
    'Corrente máxima na fase A do Rotor (A)'  : max(iar_results),
    'Corrente máxima na fase B do Rotor (A)'  : max(ibr_results),
    'Corrente máxima na fase C do Rotor (A)'  : max(icr_results),
    'Torque máximo desenvolvido no eixo (N.m)': max(Te_results),
}

# Comparação com DOL e impressão
for item, valor_AT in valores_max_AT.items():
    valor_DOL = valores_max_DOL[item]  # refere-se ao valor correspondente da partida direta
    valor_YD = valores_max_YD[item]  # refere-se ao valor correspondente da partida Y−Δ
    relacao_DOL = valor_AT / valor_DOL * 100
    relacao_YD = valor_AT / valor_YD * 100
    #diferenca_percentual_DOL = (1 - relacao) * 100
    print(f'{item}: {valor_AT:.3f} ({relacao_DOL:.1f}% do valor máximo em Partida Direta e {relacao_YD:.1f}% do valor máximo em Partida Y−Δ)')

"""## 3. Partida Soft-Starter (Mais ou menos)

**Objetivo:** Simular a partida com um autotransformador para reduzir a tensão de partida.

**Considerações:**  
Similar à Y−Δ, a partida compensadora é simulada reduzindo a tensão de entrada para um percentual (ex: 50% ou 70%) por um tempo, e depois aumentando para a tensão nominal.

**Configuração:**

- **Modo de Operação:** Motor  
- **Tensão RMS de entrada** ($V_L$): Iniciar com 110.0 V ($220 \times 0.5$) por um tempo (ex: 0.5 s), depois alterar para 220.0 V  
- **Frequência** ($f$): 60.0 Hz  
- **Torque mecânico inicial** ($T_{l\_initial}$): 0.0 Nm  
- **Torque mecânico final** ($T_{l\_final}$): 80.0 Nm  
- **Tempo de aplicação do torque mecânico** ($t_2$): 0.1 s  
- **Referência para a transformada de Park:** Síncrona ($\omega_e$)

**Análise Esperada:**  
Comparar os perfis de corrente e torque com as partidas direta e Y−Δ, notando a redução dependendo do percentual de tensão aplicado.

"""

# @title Parâmetros de simulação para Soft-Starter
print('Simulação de máquina de indução gaiola de esquilo trifásica')
print('Partida Soft-Starter')

# Simulation Parameters (using @param for interactive input)
print('\nParâmetros de simulação:')
tmax = 10  # @param {type:"slider", min:0.1, max:10.0, step:0.1, default:2.0}
h = 0.001  # @param {type:"slider", min:0.00001, max:0.001, step:0.00001, default:0.0001}
t_2 = 0.9  # @param {type:"slider", min:0.1, max:5.0, step:0.1, default:1.0}
t_pico = 5  # @param {type:"slider", min:1, max:10, step:0.1, default:1.0}
# Tempo em que a tensão aplicada deve ser igual a nominal
t_carga = 0.1 # @param {type:"slider", min:0, max:10, step:0.1, default:1.0}
# s (Momento em que a carga é aplicada)
print(f"tmax (Tempo de simulação) = {tmax} s")
print(f"t_carga (Tempo de aplicação da carga) = {t_carga} s")
print(f"t_2 (Tempo de alteração da tensão) = {t_2} s")
print(f"t_pico (Tempo de pico) = {t_pico} s")

## Motor Operation Mode (using @param for interactive input)
print('\nModo de operação da máquina de indução:')
mop_choice = "Motor"  # @param ["Motor", "Gerador conectado ao barramento infinito"]
mop = 1 if mop_choice == "Motor" else 2

## Three-Phase Induction Machine Simulation Parameters (using @param for interactive input)
print('\nParâmetros de dados da máquina de indução trifásica:')
Regulação_tensão = 0.5  # @param {type:"slider", min:0, max:1, step:0.01, default:0.5}
Vl = 220  # @param {type:"slider", min:10.0, max:500.0, step:10.0, default:220.0}
Vl_aplicada_inicialmente = Vl * Regulação_tensão
#Valor inicial da tensão
print(f"Valor inicial da tensão (Tensão RMS de linha) = {Vl_aplicada_inicialmente} V")
print(f"Vl (Tensão Nominal RMS de linha) = {Vl} V")
f = 60  # @param {type:"slider", min:10.0, max:120.0, step:1.0, default:60.0}
print(f"f (Frequência) = {f} Hz")
Rs = 0.435  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.435}
print(f"Rs (Resistência do estator) = {Rs} Ω")
Rr = 0.816  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.816}
print(f"Rr (Resistência do rotor) = {Rr} Ω")
Xm = 26.13  # @param {type:"slider", min:1.0, max:100.0, step:0.1, default:26.13}
print(f"Xm (Reatância de magnetização) = {Xm} Ω")
Xls = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xls (Reatância de dispersão do estator) = {Xls} Ω")
Xlr = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xlr (Reatância de dispersão do rotor) = {Xlr} Ω")
p = 4  # @param {type:"slider", min:2, max:8, step:2, default:4}
print(f"p (Número de polos) = {p}")
J = 0.089  # @param {type:"slider", min:0.01, max:1.0, step:0.001, default:0.089}
print(f"J (Inércia do rotor) = {J} kg·m²")
Tl_initial = 0.0  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:0.0}
print(f"Tl_initial (Torque inicial) = {Tl_initial} Nm")
Tl_final = 80  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:80.0}
print(f"Tl_final (Torque final) = {Tl_final} Nm")
# Torque final nulo pois consideraremos a máquina apenas em vazio.
B = 0.0  # @param {type:"slider", min:0.0, max:1.0, step:0.001, default:0.0}
print(f"B (Campo magnético inicial) = {B} Nm·s/rad")


## Park Transform Reference Option (using @param for interactive input)
print('\nEscolha a velocidade angular de referência para a transformada de Park:')
ref_choice = "referência síncrona (we)"  # @param ["referência síncrona (we)", "referência rotórica (wr)", "referência estacionária (0)"]
if ref_choice == "referência síncrona (we)":
    ref = 1
elif ref_choice == "referência rotórica (wr)":
    ref = 2
else:
    ref = 3

if mop == 1:
    print("\nModo de operação: motor")
    Tl = Tl_initial
    Tl_2 = Tl_final
else:
    print("\nModo de operação: gerador conectado ao barramento infinito")
    Tl = -Tl_initial
    Tl_2 = -Tl_final

if ref == 1:
    print('Referência: síncrona (we)')
elif ref == 2:
    print('Referência: rotórica (wr)')
else:
    print('Referência: estacionária (0)')


# Definição das variáveis para iteração
PSIqs_initial = 0
PSIds_initial = 0
PSIqr_initial = 0
PSIdr_initial = 0
wr_initial    = 0 # @param {"type":"slider","min":0,"max":3600,"step":50}
# Na partida, a rotação do eixo é nula.
initial_states = [PSIqs_initial, PSIds_initial, PSIqr_initial, PSIdr_initial, wr_initial]

# @title Simulador de partida Soft-Starter

# Cálculo de parâmetros
Xml = 1 / ((1 / Xm) + (1 / Xls) + (1 / Xlr))
wb = 2 * np.pi * f  # Base angular velocity
Vqr_squirrel_cage = 0
Vdr_squirrel_cage = 0 # Squirrel cage rotor

## State-Space Representation and ODEs
def induction_motor_equations(states, t, Vqs_t, Vds_t, current_Tl, w_ref_val):
    PSIqs, PSIds, PSIqr, PSIdr, wr = states

    # Dependent variables calculation
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    ids_val = (1 / Xls) * (PSIds - PSImd)
    iqs_val = (1 / Xls) * (PSIqs - PSImq)

    # Differential equations
    dPSIqs_dt = wb * (Vqs_t - (w_ref_val / wb) * PSIds + (Rs / Xls) * (PSImq - PSIqs))
    dPSIds_dt = wb * (Vds_t + (w_ref_val / wb) * PSIqs + (Rs / Xls) * (PSImd - PSIds))
    dPSIqr_dt = wb * (Vqr_squirrel_cage - ((w_ref_val - wr) / wb) * PSIdr + (Rr / Xlr) * (PSImq - PSIqr))
    dPSIdr_dt = wb * (Vdr_squirrel_cage + ((w_ref_val - wr) / wb) * PSIqr + (Rr / Xlr) * (PSImd - PSIdr))

    # Electromagnetic torque
    Tem_val = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs_val - PSIqs * ids_val)
    dwr_dt = (p / (2 * J)) * (Tem_val - current_Tl) - (B / J) * wr

    return [dPSIqs_dt, dPSIds_dt, dPSIqr_dt, dPSIdr_dt, dwr_dt]

## Simulation Loop (using odeint for integration)
t_values = np.arange(0, tmax, h)

# Store results
wr_results  = []
Te_results  = []
n_results   = []
ids_results = []
iqs_results = []
idr_results = []
iqr_results = []
ias_results = []
ibs_results = []
ics_results = []
iar_results = []
ibr_results = []
icr_results = []
Va_results  = []
Vb_results  = []
Vc_results  = []
Vds_results = []
Vqs_results = []

# To store the last `wr` value for `w_ref_val` if `ref == 2`
last_wr = wr_initial

for i, t_val in enumerate(t_values):
    # Determine mechanical torque
    if t_val < t_2:
       V_apli = Vl_aplicada_inicialmente
    elif t_val >= t_2 and t_val < t_pico:
       V_apli = Vl_aplicada_inicialmente + (Vl - Vl_aplicada_inicialmente) * (t_val - t_2) / (t_pico - t_2) # A tensão aplicada cresce linearmente com relação ao tempo.
    else:
       V_apli = Vl

    if t_val >= t_carga:
        current_Tl = Tl_2
    else:
      current_Tl = Tl


    # Stator electrical components
    we_val = 2 * np.pi * f
    tetae = we_val * t_val

    # Define arbitrary reference for the transform based on input choice
    if ref == 1:
        w_ref_val = we_val
    elif ref == 2:
        w_ref_val = last_wr  # Use the last calculated wr
    else:
        w_ref_val = 0

    # Input voltage (infinite bus)
    Va_val = np.sqrt(2) * V_apli * np.sin(tetae)
    Vb_val = np.sqrt(2) * V_apli * np.sin(tetae - ((2 * np.pi) / 3))
    Vc_val = np.sqrt(2) * V_apli * np.sin(tetae + ((2 * np.pi) / 3))

    Vaf_val = np.sqrt(3 / 2) * (Va_val + (-1 / 2) * Vb_val + (-1 / 2) * Vc_val)
    Vbt_val = np.sqrt(3 / 2) * (0 * Va_val + (np.sqrt(3) / 2) * Vb_val + (-np.sqrt(3) / 2) * Vc_val)

    Vds_val = np.cos(tetae) * Vaf_val + np.sin(tetae) * Vbt_val
    Vqs_val = -np.sin(tetae) * Vaf_val + np.cos(tetae) * Vbt_val

    # Integrate one step using odeint
    sol = odeint(induction_motor_equations, initial_states, [t_val, t_val + h], args=(Vqs_val, Vds_val, current_Tl, w_ref_val))
    PSIqs, PSIds, PSIqr, PSIdr, wr = sol[1] # Take the results at t_val + h
    initial_states = [PSIqs, PSIds, PSIqr, PSIdr, wr] # Update initial states for next iteration
    last_wr = wr # Update last_wr for next iteration if ref == 2

    # Store mechanical angular velocity
    wr_results.append(wr)

    # Calculate and store speed in RPM
    n = (120 / p) * (wr / (2 * np.pi))
    n_results.append(n)

    # Calculate currents and store
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)

    ids = (1 / Xls) * (PSIds - PSImd)
    iqs = (1 / Xls) * (PSIqs - PSImq)
    idr = (1 / Xlr) * (PSIdr - PSImd)
    iqr = (1 / Xlr) * (PSIqr - PSImq)

    ids_results.append(ids)
    iqs_results.append(iqs)
    idr_results.append(idr)
    iqr_results.append(iqr)

    # Calculate and store electromagnetic torque
    Te = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs - PSIqs * ids)
    Te_results.append(Te)

    # Stator and Rotor currents (alpha-beta and abc)
    tetar = wr * t_val # Using t_val for tetar calculation here, consistent with Scilab code

    iafs = np.cos(tetae) * ids - np.sin(tetae) * iqs
    ibts = np.sin(tetae) * ids + np.cos(tetae) * iqs

    iafr = np.cos(tetar) * idr - np.sin(tetar) * iqr
    ibtr = np.sin(tetar) * idr + np.cos(tetar) * iqr # Corrected to tetar, assuming original was a typo

    ias = np.sqrt(3/2) * (1 * iafs + 0 * ibts)
    ibs = np.sqrt(3/2) * ((-1/2) * iafs + (np.sqrt(3)/2) * ibts)
    ics = np.sqrt(3/2) * ((-1/2) * iafs + (-np.sqrt(3)/2) * ibts)

    iar = np.sqrt(3/2) * (1 * iafr + 0 * ibtr)
    ibr = np.sqrt(3/2) * ((-1/2) * iafr + (np.sqrt(3)/2) * ibtr)
    icr = np.sqrt(3/2) * ((-1/2) * iafr + (-np.sqrt(3)/2) * ibtr)

    ias_results.append(ias)
    ibs_results.append(ibs)
    ics_results.append(ics)
    iar_results.append(iar)
    ibr_results.append(ibr)
    icr_results.append(icr)

    Va_results.append(Va_val)
    Vb_results.append(Vb_val)
    Vc_results.append(Vc_val)
    Vds_results.append(Vds_val)
    Vqs_results.append(Vqs_val)


# Convert lists to numpy arrays for easier plotting
wr_results  = np.array(wr_results)
Te_results  = np.array(Te_results)
n_results   = np.array(n_results)
ids_results = np.array(ids_results)
iqs_results = np.array(iqs_results)
idr_results = np.array(idr_results)
iqr_results = np.array(iqr_results)
ias_results = np.array(ias_results)
ibs_results = np.array(ibs_results)
ics_results = np.array(ics_results)
iar_results = np.array(iar_results)
ibr_results = np.array(ibr_results)
icr_results = np.array(icr_results)
Va_results  = np.array(Va_results)
Vb_results  = np.array(Vb_results)
Vc_results  = np.array(Vc_results)
Vds_results = np.array(Vds_results)
Vqs_results = np.array(Vqs_results)


## Steady-State Values
print("\nValores (finais) em regime permanente das variáveis:")
print("s f_r ids iqs n")
# Assuming the last values are steady-state
s_ss = (wb - wr_results[-1]) / wb
fr_ss = wr_results[-1] / (2 * np.pi)
ids_ss = ids_results[-1]
iqs_ss = iqs_results[-1]
n_ss = n_results[-1]
print(f"{s_ss:.4f} {fr_ss:.4f} {ids_ss:.4f} {iqs_ss:.4f} {n_ss:.4f}")

# Torque
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Te_results, label='$T_e$')
plt.title('$T_e$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Torque (Nm)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda fora
plt.grid(True)

# Velocidade
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 2)
plt.plot(t_values, n_results, label='n (RPM)')
plt.title('n (RPM)', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Velocidade (RPM)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda fora
plt.grid(True)
plt.show()

# Stator Voltages
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Va_results, label='$V_a$')
plt.plot(t_values, Vb_results, label='$V_b$')
plt.plot(t_values, Vc_results, label='$V_c$')
plt.title('$V_a$ x $V_b$ x $V_c$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Tensão (V)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)

# Stator Currents
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 2)
plt.plot(t_values, ias_results, label='$i_{as}$')
plt.plot(t_values, ibs_results, label='$i_{bs}$')
plt.plot(t_values, ics_results, label='$i_{cs}$')
plt.title('$I_{as}$ x $I_{bs}$ x $I_{cs}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# Rotor Currents
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, iar_results, label='$i_{ar}$')
plt.plot(t_values, ibr_results, label='$i_{br}$')
plt.plot(t_values, icr_results, label='$i_{cr}$')
plt.title('$I_{ar}$ x $I_{br}$ x $I_{cr}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

print('Valores máximos para partida Soft-Starter')

# Valores máximos
valores_max_SF = {
    'Corrente máxima na fase A do Estator (A)': max(ias_results),
    'Corrente máxima na fase B do Estator (A)': max(ibs_results),
    'Corrente máxima na fase C do Estator (A)': max(ics_results),
    'Corrente máxima na fase A do Rotor (A)'  : max(iar_results),
    'Corrente máxima na fase B do Rotor (A)'  : max(ibr_results),
    'Corrente máxima na fase C do Rotor (A)'  : max(icr_results),
    'Torque máximo desenvolvido no eixo (N.m)': max(Te_results),
}

for item, valor_SF in valores_max_AT.items():
    valor_DOL = valores_max_DOL[item]  # refere-se ao valor correspondente da partida direta
    valor_YD = valores_max_YD[item]  # refere-se ao valor correspondente da partida Y−Δ
    relacao_DOL = valor_SF / valor_DOL * 100
    relacao_YD = valor_SF / valor_YD * 100
    print(f'{item}: {valor_SF:.3f} ({relacao_DOL:.1f}% do valor máximo em Partida Direta e {relacao_YD:.1f}% do valor máximo em Partida Y−Δ)')

# ==============================================================================
# EXPERIMENTOS DE CARGA
# ==============================================================================

# @title Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from IPython.display import display

"""---

# O Colab aqui apresentado tem o objetivo de simular situações específicas no funcionamento de uma máquina de indução trifásica.

# 4. Partida em Vazio e Aplicação de Carga Nominal

* Objetivo: Observar a aceleração da máquina sem carga e a resposta dinâmica à aplicação repentina de torque nominal.

* Configuração: M

<ol type = "i">

  <li> Modo de Operação: Motor</li>

  <li> Torque mecânico inicial ($T_{l\_initial}$): 0.0 Nm </li>

  <li> Torque mecânico final ($T_{l\_final}$): 80.0 Nm </li>

  <li> Tempo de aplicação de torque mecânico ($t_{2}$): 1.0 s </li>

 </ol>

Análise Esperada: Verificar o tempo de aceleração em vazio e o transitório na velocidade e corrente ao aplicar a carga.

Já que não foi especificada, consideramemos uma partida direta.
"""

# @title Parâmetros de simulação para partida em vazio e aplicação de carga nominal.
print('Simulação de máquina de indução gaiola de esquilo trifásica')
print('Partida em Vazio e Aplicação de Carga Nominal')

# Simulation Parameters (using @param for interactive input)
print('\nParâmetros de simulação:')
tmax = 2  # @param {type:"slider", min:0.1, max:10.0, step:0.1, default:2.0}
h = 0.001  # @param {type:"slider", min:0.00001, max:0.001, step:0.00001, default:0.0001}
t_2 = 0.9  # @param {type:"slider", min:0.1, max:5.0, step:0.1, default:1.0}
t_pico = 5  # @param {type:"slider", min:1, max:10, step:0.1, default:1.0}
# Tempo em que a tensão aplicada deve ser igual a nominal
t_carga = 1 # @param {type:"slider", min:0, max:10, step:0.1, default:1.0}
# s (Momento em que a carga é aplicada)
print(f"tmax (Tempo de simulação) = {tmax} s")
print(f"t_carga (Tempo de aplicação da carga) = {t_carga} s")
print(f"t_2 (Tempo de alteração da tensão) = {t_2} s")
print(f"t_pico (Tempo de pico) = {t_pico} s")

## Motor Operation Mode (using @param for interactive input)
print('\nModo de operação da máquina de indução:')
mop_choice = "Motor"  # @param ["Motor", "Gerador conectado ao barramento infinito"]
mop = 1 if mop_choice == "Motor" else 2

## Three-Phase Induction Machine Simulation Parameters (using @param for interactive input)
print('\nParâmetros de dados da máquina de indução trifásica:')
Regulação_tensão = 0.5  # @param {type:"slider", min:0, max:1, step:0.01, default:0.5}
Vl = 220  # @param {type:"slider", min:10.0, max:500.0, step:10.0, default:220.0}
Vl_aplicada_inicialmente = Vl * Regulação_tensão
#Valor inicial da tensão
print(f"Valor inicial da tensão (Tensão RMS de linha) = {Vl_aplicada_inicialmente} V")
print(f"Vl (Tensão Nominal RMS de linha) = {Vl} V")
f = 60  # @param {type:"slider", min:10.0, max:120.0, step:1.0, default:60.0}
print(f"f (Frequência) = {f} Hz")
Rs = 0.435  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.435}
print(f"Rs (Resistência do estator) = {Rs} Ω")
Rr = 0.816  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.816}
print(f"Rr (Resistência do rotor) = {Rr} Ω")
Xm = 26.13  # @param {type:"slider", min:1.0, max:100.0, step:0.1, default:26.13}
print(f"Xm (Reatância de magnetização) = {Xm} Ω")
Xls = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xls (Reatância de dispersão do estator) = {Xls} Ω")
Xlr = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xlr (Reatância de dispersão do rotor) = {Xlr} Ω")
p = 4  # @param {type:"slider", min:2, max:8, step:2, default:4}
print(f"p (Número de polos) = {p}")
J = 0.089  # @param {type:"slider", min:0.01, max:1.0, step:0.001, default:0.089}
print(f"J (Inércia do rotor) = {J} kg·m²")
Tl_initial = 0.0  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:0.0}
print(f"Tl_initial (Torque inicial) = {Tl_initial} Nm")
Tl_final = 80  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:80.0}
print(f"Tl_final (Torque final) = {Tl_final} Nm")
# Torque final nulo pois consideraremos a máquina apenas em vazio.
B = 0.0  # @param {type:"slider", min:0.0, max:1.0, step:0.001, default:0.0}
print(f"B (Campo magnético inicial) = {B} Nm·s/rad")


## Park Transform Reference Option (using @param for interactive input)
print('\nEscolha a velocidade angular de referência para a transformada de Park:')
ref_choice = "referência síncrona (we)"  # @param ["referência síncrona (we)", "referência rotórica (wr)", "referência estacionária (0)"]
if ref_choice == "referência síncrona (we)":
    ref = 1
elif ref_choice == "referência rotórica (wr)":
    ref = 2
else:
    ref = 3

if mop == 1:
    print("\nModo de operação: motor")
    Tl = Tl_initial
    Tl_2 = Tl_final
else:
    print("\nModo de operação: gerador conectado ao barramento infinito")
    Tl = -Tl_initial
    Tl_2 = -Tl_final

if ref == 1:
    print('Referência: síncrona (we)')
elif ref == 2:
    print('Referência: rotórica (wr)')
else:
    print('Referência: estacionária (0)')


# Definição das variáveis para iteração
PSIqs_initial = 0
PSIds_initial = 0
PSIqr_initial = 0
PSIdr_initial = 0
wr_initial    = 0 # @param {"type":"slider","min":0,"max":3600,"step":50}
# Na partida, a rotação do eixo é nula.
initial_states = [PSIqs_initial, PSIds_initial, PSIqr_initial, PSIdr_initial, wr_initial]

# @title Simulador de partida em vazio e aplicação da carga nominal

# Cálculo de parâmetros
Xml = 1 / ((1 / Xm) + (1 / Xls) + (1 / Xlr))
wb = 2 * np.pi * f  # Base angular velocity
Vqr_squirrel_cage = 0
Vdr_squirrel_cage = 0 # Squirrel cage rotor

## State-Space Representation and ODEs
def induction_motor_equations(states, t, Vqs_t, Vds_t, current_Tl, w_ref_val):
    PSIqs, PSIds, PSIqr, PSIdr, wr = states

    # Dependent variables calculation
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    ids_val = (1 / Xls) * (PSIds - PSImd)
    iqs_val = (1 / Xls) * (PSIqs - PSImq)

    # Differential equations
    dPSIqs_dt = wb * (Vqs_t - (w_ref_val / wb) * PSIds + (Rs / Xls) * (PSImq - PSIqs))
    dPSIds_dt = wb * (Vds_t + (w_ref_val / wb) * PSIqs + (Rs / Xls) * (PSImd - PSIds))
    dPSIqr_dt = wb * (Vqr_squirrel_cage - ((w_ref_val - wr) / wb) * PSIdr + (Rr / Xlr) * (PSImq - PSIqr))
    dPSIdr_dt = wb * (Vdr_squirrel_cage + ((w_ref_val - wr) / wb) * PSIqr + (Rr / Xlr) * (PSImd - PSIdr))

    # Electromagnetic torque
    Tem_val = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs_val - PSIqs * ids_val)
    dwr_dt = (p / (2 * J)) * (Tem_val - current_Tl) - (B / J) * wr

    return [dPSIqs_dt, dPSIds_dt, dPSIqr_dt, dPSIdr_dt, dwr_dt]

## Simulation Loop (using odeint for integration)
t_values = np.arange(0, tmax, h)

# Store results
wr_results  = []
Te_results  = []
n_results   = []
ids_results = []
iqs_results = []
idr_results = []
iqr_results = []
ias_results = []
ibs_results = []
ics_results = []
iar_results = []
ibr_results = []
icr_results = []
Va_results  = []
Vb_results  = []
Vc_results  = []
Vds_results = []
Vqs_results = []

# To store the last `wr` value for `w_ref_val` if `ref == 2`
last_wr = wr_initial

for i, t_val in enumerate(t_values):
    # Determine mechanical torque
    V_apli = Vl

    if t_val >= t_carga:
        current_Tl = Tl_2
    else:
      current_Tl = Tl


    # Stator electrical components
    we_val = 2 * np.pi * f
    tetae = we_val * t_val

    # Define arbitrary reference for the transform based on input choice
    if ref == 1:
        w_ref_val = we_val
    elif ref == 2:
        w_ref_val = last_wr  # Use the last calculated wr
    else:
        w_ref_val = 0

    # Input voltage (infinite bus)
    Va_val = np.sqrt(2) * V_apli * np.sin(tetae)
    Vb_val = np.sqrt(2) * V_apli * np.sin(tetae - ((2 * np.pi) / 3))
    Vc_val = np.sqrt(2) * V_apli * np.sin(tetae + ((2 * np.pi) / 3))

    Vaf_val = np.sqrt(3 / 2) * (Va_val + (-1 / 2) * Vb_val + (-1 / 2) * Vc_val)
    Vbt_val = np.sqrt(3 / 2) * (0 * Va_val + (np.sqrt(3) / 2) * Vb_val + (-np.sqrt(3) / 2) * Vc_val)

    Vds_val = np.cos(tetae) * Vaf_val + np.sin(tetae) * Vbt_val
    Vqs_val = -np.sin(tetae) * Vaf_val + np.cos(tetae) * Vbt_val

    # Integrate one step using odeint
    sol = odeint(induction_motor_equations, initial_states, [t_val, t_val + h], args=(Vqs_val, Vds_val, current_Tl, w_ref_val))
    PSIqs, PSIds, PSIqr, PSIdr, wr = sol[1] # Take the results at t_val + h
    initial_states = [PSIqs, PSIds, PSIqr, PSIdr, wr] # Update initial states for next iteration
    last_wr = wr # Update last_wr for next iteration if ref == 2

    # Store mechanical angular velocity
    wr_results.append(wr)

    # Calculate and store speed in RPM
    n = (120 / p) * (wr / (2 * np.pi))
    n_results.append(n)

    # Calculate currents and store
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)

    ids = (1 / Xls) * (PSIds - PSImd)
    iqs = (1 / Xls) * (PSIqs - PSImq)
    idr = (1 / Xlr) * (PSIdr - PSImd)
    iqr = (1 / Xlr) * (PSIqr - PSImq)

    ids_results.append(ids)
    iqs_results.append(iqs)
    idr_results.append(idr)
    iqr_results.append(iqr)

    # Calculate and store electromagnetic torque
    Te = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs - PSIqs * ids)
    Te_results.append(Te)

    # Stator and Rotor currents (alpha-beta and abc)
    tetar = wr * t_val # Using t_val for tetar calculation here, consistent with Scilab code

    iafs = np.cos(tetae) * ids - np.sin(tetae) * iqs
    ibts = np.sin(tetae) * ids + np.cos(tetae) * iqs

    iafr = np.cos(tetar) * idr - np.sin(tetar) * iqr
    ibtr = np.sin(tetar) * idr + np.cos(tetar) * iqr # Corrected to tetar, assuming original was a typo

    ias = np.sqrt(3/2) * (1 * iafs + 0 * ibts)
    ibs = np.sqrt(3/2) * ((-1/2) * iafs + (np.sqrt(3)/2) * ibts)
    ics = np.sqrt(3/2) * ((-1/2) * iafs + (-np.sqrt(3)/2) * ibts)

    iar = np.sqrt(3/2) * (1 * iafr + 0 * ibtr)
    ibr = np.sqrt(3/2) * ((-1/2) * iafr + (np.sqrt(3)/2) * ibtr)
    icr = np.sqrt(3/2) * ((-1/2) * iafr + (-np.sqrt(3)/2) * ibtr)

    ias_results.append(ias)
    ibs_results.append(ibs)
    ics_results.append(ics)
    iar_results.append(iar)
    ibr_results.append(ibr)
    icr_results.append(icr)

    Va_results.append(Va_val)
    Vb_results.append(Vb_val)
    Vc_results.append(Vc_val)
    Vds_results.append(Vds_val)
    Vqs_results.append(Vqs_val)


# Convert lists to numpy arrays for easier plotting
wr_results  = np.array(wr_results)
Te_results  = np.array(Te_results)
n_results   = np.array(n_results)
ids_results = np.array(ids_results)
iqs_results = np.array(iqs_results)
idr_results = np.array(idr_results)
iqr_results = np.array(iqr_results)
ias_results = np.array(ias_results)
ibs_results = np.array(ibs_results)
ics_results = np.array(ics_results)
iar_results = np.array(iar_results)
ibr_results = np.array(ibr_results)
icr_results = np.array(icr_results)
Va_results  = np.array(Va_results)
Vb_results  = np.array(Vb_results)
Vc_results  = np.array(Vc_results)
Vds_results = np.array(Vds_results)
Vqs_results = np.array(Vqs_results)


## Steady-State Values
print("\nValores (finais) em regime permanente das variáveis:")
print("s f_r ids iqs n")
# Assuming the last values are steady-state
s_ss = (wb - wr_results[-1]) / wb
fr_ss = wr_results[-1] / (2 * np.pi)
ids_ss = ids_results[-1]
iqs_ss = iqs_results[-1]
n_ss = n_results[-1]
print(f"{s_ss:.4f} {fr_ss:.4f} {ids_ss:.4f} {iqs_ss:.4f} {n_ss:.4f}")

# Torque
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Te_results, label='$T_e$')
plt.title('$T_e$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Torque (Nm)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda fora
plt.grid(True)

# Velocidade
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 2)
plt.plot(t_values, n_results, label='n (RPM)')
plt.title('n (RPM)', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Velocidade (RPM)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))  # legenda fora
plt.grid(True)
plt.show()

# Stator Voltages
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Va_results, label='$V_a$')
plt.plot(t_values, Vb_results, label='$V_b$')
plt.plot(t_values, Vc_results, label='$V_c$')
plt.title('$V_a$ x $V_b$ x $V_c$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Tensão (V)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)

# Stator Currents
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 2)
plt.plot(t_values, ias_results, label='$i_{as}$')
plt.plot(t_values, ibs_results, label='$i_{bs}$')
plt.plot(t_values, ics_results, label='$i_{cs}$')
plt.title('$I_{as}$ x $I_{bs}$ x $I_{cs}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# Rotor Currents
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, iar_results, label='$i_{ar}$')
plt.plot(t_values, ibr_results, label='$i_{br}$')
plt.plot(t_values, icr_results, label='$i_{cr}$')
plt.title('$I_{ar}$ x $I_{br}$ x $I_{cr}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# @title Resultados
# Obter os valores máximos:
print('Valores máximos para partida em vazio e aplicação da carga nominal')
valores_max_PVN = {
  'Corrente máxima na fase A do Estator (A)': max(ias_results),
  'Corrente máxima na fase B do Estator (A)': max(ibs_results),
  'Corrente máxima na fase C do Estator (A)': max(ics_results),
    'Corrente máxima na fase A do Rotor (A)': max(iar_results),
    'Corrente máxima na fase B do Rotor (A)': max(ibr_results),
    'Corrente máxima na fase C do Rotor (A)': max(icr_results),
  'Torque máximo desenvolvido no eixo (N.m)': max(Te_results),
}

# Resultados máximos:
for item, valor in valores_max_PVN.items():
    print(f'{item}: {valor:.3f}')

"""# 5. Variação da Carga (50% do Nominal)

*   Objetivo: Analisar o comportamento da máquina sob carga parcial.


# Configuração:
1. Modo de Operação: Motor

2. Torque mecânico inicial ( Tlinitial
 ): 0.0 Nm
3. Torque mecânico final ( Tlfinal
 ): 40.0 Nm (50% de 80 Nm)
4. Tempo de aplicação de torque mecânico ( t2
 ): 1.0 s
# Análise Esperada: Comparar a velocidade de regime, corrente e escorregamento com a carga nominal.
"""

# @title Parâmetros de simulação para partida em vazio e aplicação de carga nominal.
print('Simulação de máquina de indução gaiola de esquilo trifásica')
print('Partida em Vazio e Aplicação de Carga Nominal')

# Simulation Parameters (using @param for interactive input)
print('\nParâmetros de simulação:')
tmax = 2  # @param {type:"slider", min:0.1, max:10.0, step:0.1, default:2.0}
h = 0.001  # @param {type:"slider", min:0.00001, max:0.001, step:0.00001, default:0.0001}
t_2 = 1  # @param {type:"slider", min:0.1, max:5.0, step:0.1, default:1.0}
t_pico = 5  # @param {type:"slider", min:1, max:10, step:0.1, default:1.0}
# Tempo em que a tensão aplicada deve ser igual a nominal
t_carga = 1 # @param {type:"slider", min:0, max:10, step:0.1, default:1.0}
# s (Momento em que a carga é aplicada)
print(f"tmax (Tempo de simulação) = {tmax} s")
print(f"t_carga (Tempo de aplicação da carga) = {t_carga} s")
print(f"t_2 (Tempo de alteração da tensão) = {t_2} s")
print(f"t_pico (Tempo de pico) = {t_pico} s")

## Motor Operation Mode (using @param for interactive input)
print('\nModo de operação da máquina de indução:')
mop_choice = "Motor"  # @param ["Motor", "Gerador conectado ao barramento infinito"]
mop = 1 if mop_choice == "Motor" else 2

## Three-Phase Induction Machine Simulation Parameters (using @param for interactive input)
print('\nParâmetros de dados da máquina de indução trifásica:')
Regulação_tensão = 0.5  # @param {type:"slider", min:0, max:1, step:0.01, default:0.5}
Vl = 220  # @param {type:"slider", min:10.0, max:500.0, step:10.0, default:220.0}
Vl_aplicada_inicialmente = Vl * Regulação_tensão
#Valor inicial da tensão
print(f"Valor inicial da tensão (Tensão RMS de linha) = {Vl_aplicada_inicialmente} V")
print(f"Vl (Tensão Nominal RMS de linha) = {Vl} V")
f = 60  # @param {type:"slider", min:10.0, max:120.0, step:1.0, default:60.0}
print(f"f (Frequência) = {f} Hz")
Rs = 0.435  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.435}
print(f"Rs (Resistência do estator) = {Rs} Ω")
Rr = 0.816  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.816}
print(f"Rr (Resistência do rotor) = {Rr} Ω")
Xm = 26.13  # @param {type:"slider", min:1.0, max:100.0, step:0.1, default:26.13}
print(f"Xm (Reatância de magnetização) = {Xm} Ω")
Xls = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xls (Reatância de dispersão do estator) = {Xls} Ω")
Xlr = 0.754  # @param {type:"slider", min:0.1, max:5.0, step:0.01, default:0.754}
print(f"Xlr (Reatância de dispersão do rotor) = {Xlr} Ω")
p = 4  # @param {type:"slider", min:2, max:8, step:2, default:4}
print(f"p (Número de polos) = {p}")
J = 0.089  # @param {type:"slider", min:0.01, max:1.0, step:0.001, default:0.089}
print(f"J (Inércia do rotor) = {J} kg·m²")
Tl_initial = 0.0  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:0.0}
print(f"Tl_initial (Torque inicial) = {Tl_initial} Nm")
Tl_final = 80  # @param {type:"slider", min:0.0, max:200.0, step:1.0, default:80.0}
print(f"Tl_final (Torque final) = {Tl_final} Nm")
# Torque final nulo pois consideraremos a máquina apenas em vazio.
B = 0.0  # @param {type:"slider", min:0.0, max:1.0, step:0.001, default:0.0}
print(f"B (Campo magnético inicial) = {B} Nm·s/rad")


## Park Transform Reference Option (using @param for interactive input)
print('\nEscolha a velocidade angular de referência para a transformada de Park:')
ref_choice = "referência síncrona (we)"  # @param ["referência síncrona (we)", "referência rotórica (wr)", "referência estacionária (0)"]
if ref_choice == "referência síncrona (we)":
    ref = 1
elif ref_choice == "referência rotórica (wr)":
    ref = 2
else:
    ref = 3

if mop == 1:
    print("\nModo de operação: motor")
    Tl = Tl_initial
    Tl_2 = Tl_final
else:
    print("\nModo de operação: gerador conectado ao barramento infinito")
    Tl = -Tl_initial
    Tl_2 = -Tl_final

if ref == 1:
    print('Referência: síncrona (we)')
elif ref == 2:
    print('Referência: rotórica (wr)')
else:
    print('Referência: estacionária (0)')


# Definição das variáveis para iteração
PSIqs_initial = 0
PSIds_initial = 0
PSIqr_initial = 0
PSIdr_initial = 0
wr_initial    = 0 # @param {"type":"slider","min":0,"max":3600,"step":50}
# Na partida, a rotação do eixo é nula.
initial_states = [PSIqs_initial, PSIds_initial, PSIqr_initial, PSIdr_initial, wr_initial]

# @title Simulador de variação de carga (50% do nominal) após partida em vazio


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Cálculo de parâmetros auxiliares
Xml = 1 / ((1 / Xm) + (1 / Xls) + (1 / Xlr))
wb = 2 * np.pi * f  # velocidade angular base
Vqr_squirrel_cage = 0
Vdr_squirrel_cage = 0

def induction_motor_equations(states, t, Vqs_t, Vds_t, current_Tl, w_ref_val):
    PSIqs, PSIds, PSIqr, PSIdr, wr = states

    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    ids_val = (1 / Xls) * (PSIds - PSImd)
    iqs_val = (1 / Xls) * (PSIqs - PSImq)

    dPSIqs_dt = wb * (Vqs_t - (w_ref_val / wb) * PSIds + (Rs / Xls) * (PSImq - PSIqs))
    dPSIds_dt = wb * (Vds_t + (w_ref_val / wb) * PSIqs + (Rs / Xls) * (PSImd - PSIds))
    dPSIqr_dt = wb * (Vqr_squirrel_cage - ((w_ref_val - wr) / wb) * PSIdr + (Rr / Xlr) * (PSImq - PSIqr))
    dPSIdr_dt = wb * (Vdr_squirrel_cage + ((w_ref_val - wr) / wb) * PSIqr + (Rr / Xlr) * (PSImd - PSIdr))

    Tem_val = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs_val - PSIqs * ids_val)
    dwr_dt = (p / (2 * J)) * (Tem_val - current_Tl) - (B / J) * wr

    return [dPSIqs_dt, dPSIds_dt, dPSIqr_dt, dPSIdr_dt, dwr_dt]

# Tempo da simulação
t_values = np.arange(0, tmax, h)

# Inicialização dos resultados
wr_results  = []
Te_results  = []
n_results   = []
ids_results = []
iqs_results = []
idr_results = []
iqr_results = []
ias_results = []
ibs_results = []
ics_results = []
iar_results = []
ibr_results = []
icr_results = []
Va_results  = []
Vb_results  = []
Vc_results  = []
Vds_results = []
Vqs_results = []

last_wr = wr_initial

for i, t_val in enumerate(t_values):
    V_apli = Vl

    # Aplica torque mecânico inicial ou final dependendo do tempo
    if t_val >= t_carga:
        current_Tl = Tl_final/2  # 50% da carga nominal (ex: 40 Nm)
    else:
        current_Tl = Tl_initial  # 0 Nm

    we_val = 2 * np.pi * f
    tetae = we_val * t_val

    if ref == 1:
        w_ref_val = we_val
    elif ref == 2:
        w_ref_val = last_wr
    else:
        w_ref_val = 0

    Va_val = np.sqrt(2) * V_apli * np.sin(tetae)
    Vb_val = np.sqrt(2) * V_apli * np.sin(tetae - 2 * np.pi / 3)
    Vc_val = np.sqrt(2) * V_apli * np.sin(tetae + 2 * np.pi / 3)

    Vaf_val = np.sqrt(3 / 2) * (Va_val + (-1/2) * Vb_val + (-1/2) * Vc_val)
    Vbt_val = np.sqrt(3 / 2) * (0 * Va_val + (np.sqrt(3)/2) * Vb_val + (-np.sqrt(3)/2) * Vc_val)

    Vds_val = np.cos(tetae) * Vaf_val + np.sin(tetae) * Vbt_val
    Vqs_val = -np.sin(tetae) * Vaf_val + np.cos(tetae) * Vbt_val

    sol = odeint(induction_motor_equations, initial_states, [t_val, t_val + h], args=(Vqs_val, Vds_val, current_Tl, w_ref_val))
    PSIqs, PSIds, PSIqr, PSIdr, wr = sol[1]
    initial_states = [PSIqs, PSIds, PSIqr, PSIdr, wr]
    last_wr = wr

    wr_results.append(wr)
    n = (120 / p) * (wr / (2 * np.pi))
    n_results.append(n)

    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)

    ids = (1 / Xls) * (PSIds - PSImd)
    iqs = (1 / Xls) * (PSIqs - PSImq)
    idr = (1 / Xlr) * (PSIdr - PSImd)
    iqr = (1 / Xlr) * (PSIqr - PSImq)

    ids_results.append(ids)
    iqs_results.append(iqs)
    idr_results.append(idr)
    iqr_results.append(iqr)

    Te = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs - PSIqs * ids)
    Te_results.append(Te)

    tetar = wr * t_val

    iafs = np.cos(tetae) * ids - np.sin(tetae) * iqs
    ibts = np.sin(tetae) * ids + np.cos(tetae) * iqs

    iafr = np.cos(tetar) * idr - np.sin(tetar) * iqr
    ibtr = np.sin(tetar) * idr + np.cos(tetar) * iqr

    ias = np.sqrt(3/2) * (1 * iafs + 0 * ibts)
    ibs = np.sqrt(3/2) * ((-1/2) * iafs + (np.sqrt(3)/2) * ibts)
    ics = np.sqrt(3/2) * ((-1/2) * iafs + (-np.sqrt(3)/2) * ibts)

    iar = np.sqrt(3/2) * (1 * iafr + 0 * ibtr)
    ibr = np.sqrt(3/2) * ((-1/2) * iafr + (np.sqrt(3)/2) * ibtr)
    icr = np.sqrt(3/2) * ((-1/2) * iafr + (-np.sqrt(3)/2) * ibtr)

    ias_results.append(ias)
    ibs_results.append(ibs)
    ics_results.append(ics)
    iar_results.append(iar)
    ibr_results.append(ibr)
    icr_results.append(icr)

    Va_results.append(Va_val)
    Vb_results.append(Vb_val)
    Vc_results.append(Vc_val)
    Vds_results.append(Vds_val)
    Vqs_results.append(Vqs_val)


# Convertendo para arrays numpy
wr_results  = np.array(wr_results)
Te_results  = np.array(Te_results)
n_results   = np.array(n_results)
ids_results = np.array(ids_results)
iqs_results = np.array(iqs_results)
idr_results = np.array(idr_results)
iqr_results = np.array(iqr_results)
ias_results = np.array(ias_results)
ibs_results = np.array(ibs_results)
ics_results = np.array(ics_results)
iar_results = np.array(iar_results)
ibr_results = np.array(ibr_results)
icr_results = np.array(icr_results)
Va_results  = np.array(Va_results)
Vb_results  = np.array(Vb_results)
Vc_results  = np.array(Vc_results)
Vds_results = np.array(Vds_results)
Vqs_results = np.array(Vqs_results)

print("\nValores (finais) em regime permanente das variáveis:")
print("s f_r ids iqs n")
s_ss = (wb - wr_results[-1]) / wb
fr_ss = wr_results[-1] / (2 * np.pi)
ids_ss = ids_results[-1]
iqs_ss = iqs_results[-1]
n_ss = n_results[-1]
print(f"{s_ss:.4f} {fr_ss:.4f} {ids_ss:.4f} {iqs_ss:.4f} {n_ss:.4f}")

# Gráficos

plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Te_results, label='$T_e$')
plt.title('$T_e$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Torque (Nm)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_values, n_results, label='n (RPM)')
plt.title('n (RPM)', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Velocidade (RPM)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Va_results, label='$V_a$')
plt.plot(t_values, Vb_results, label='$V_b$')
plt.plot(t_values, Vc_results, label='$V_c$')
plt.title('$V_a$ x $V_b$ x $V_c$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Tensão (V)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_values, ias_results, label='$i_{as}$')
plt.plot(t_values, ibs_results, label='$i_{bs}$')
plt.plot(t_values, ics_results, label='$i_{cs}$')
plt.title('$I_{as}$ x $I_{bs}$ x $I_{cs}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, iar_results, label='$i_{ar}$')
plt.plot(t_values, ibr_results, label='$i_{br}$')
plt.plot(t_values, icr_results, label='$i_{cr}$')
plt.title('$I_{ar}$ x $I_{br}$ x $I_{cr}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

"""# 6. Sobrecarga Temporária (120% do Nominal)

*   Objetivo: Testar a capacidade da máquina de lidar com uma sobrecarga temporária.

# Configuração:
1. Modo de Operação: Motor

2. Torque mecânico inicial ( Tlinitial
 ): 0.0 Nm
3. Torque mecânico final ( Tlfinal
 ): 96.0 Nm (120% de 80 Nm)
4. Tempo de aplicação de torque mecânico ( t2
 ): 1.0 s
# Análise Esperada: Observar a resposta da velocidade, torque eletromagnético e correntes sob sobrecarga.
"""

# @title Simulador de Sobrecarga (120% do nominal) após partida em vazio


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Cálculo de parâmetros auxiliares
Xml = 1 / ((1 / Xm) + (1 / Xls) + (1 / Xlr))
wb = 2 * np.pi * f  # velocidade angular base
Vqr_squirrel_cage = 0
Vdr_squirrel_cage = 0

def induction_motor_equations(states, t, Vqs_t, Vds_t, current_Tl, w_ref_val):
    PSIqs, PSIds, PSIqr, PSIdr, wr = states

    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    ids_val = (1 / Xls) * (PSIds - PSImd)
    iqs_val = (1 / Xls) * (PSIqs - PSImq)

    dPSIqs_dt = wb * (Vqs_t - (w_ref_val / wb) * PSIds + (Rs / Xls) * (PSImq - PSIqs))
    dPSIds_dt = wb * (Vds_t + (w_ref_val / wb) * PSIqs + (Rs / Xls) * (PSImd - PSIds))
    dPSIqr_dt = wb * (Vqr_squirrel_cage - ((w_ref_val - wr) / wb) * PSIdr + (Rr / Xlr) * (PSImq - PSIqr))
    dPSIdr_dt = wb * (Vdr_squirrel_cage + ((w_ref_val - wr) / wb) * PSIqr + (Rr / Xlr) * (PSImd - PSIdr))

    Tem_val = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs_val - PSIqs * ids_val)
    dwr_dt = (p / (2 * J)) * (Tem_val - current_Tl) - (B / J) * wr

    return [dPSIqs_dt, dPSIds_dt, dPSIqr_dt, dPSIdr_dt, dwr_dt]

# Tempo da simulação
t_values = np.arange(0, tmax, h)

# Inicialização dos resultados
wr_results  = []
Te_results  = []
n_results   = []
ids_results = []
iqs_results = []
idr_results = []
iqr_results = []
ias_results = []
ibs_results = []
ics_results = []
iar_results = []
ibr_results = []
icr_results = []
Va_results  = []
Vb_results  = []
Vc_results  = []
Vds_results = []
Vqs_results = []

last_wr = wr_initial

for i, t_val in enumerate(t_values):
    V_apli = Vl

    # Aplica torque mecânico inicial ou final dependendo do tempo
    if t_val >= t_carga:
        current_Tl = Tl_final * 1.2  # 120% da carga nominal (ex: 96 Nm)
    else:
        current_Tl = Tl_initial  # 0 Nm

    we_val = 2 * np.pi * f
    tetae = we_val * t_val

    if ref == 1:
        w_ref_val = we_val
    elif ref == 2:
        w_ref_val = last_wr
    else:
        w_ref_val = 0

    Va_val = np.sqrt(2) * V_apli * np.sin(tetae)
    Vb_val = np.sqrt(2) * V_apli * np.sin(tetae - 2 * np.pi / 3)
    Vc_val = np.sqrt(2) * V_apli * np.sin(tetae + 2 * np.pi / 3)

    Vaf_val = np.sqrt(3 / 2) * (Va_val + (-1/2) * Vb_val + (-1/2) * Vc_val)
    Vbt_val = np.sqrt(3 / 2) * (0 * Va_val + (np.sqrt(3)/2) * Vb_val + (-np.sqrt(3)/2) * Vc_val)

    Vds_val = np.cos(tetae) * Vaf_val + np.sin(tetae) * Vbt_val
    Vqs_val = -np.sin(tetae) * Vaf_val + np.cos(tetae) * Vbt_val

    sol = odeint(induction_motor_equations, initial_states, [t_val, t_val + h], args=(Vqs_val, Vds_val, current_Tl, w_ref_val))
    PSIqs, PSIds, PSIqr, PSIdr, wr = sol[1]
    initial_states = [PSIqs, PSIds, PSIqr, PSIdr, wr]
    last_wr = wr

    wr_results.append(wr)
    n = (120 / p) * (wr / (2 * np.pi))
    n_results.append(n)

    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)

    ids = (1 / Xls) * (PSIds - PSImd)
    iqs = (1 / Xls) * (PSIqs - PSImq)
    idr = (1 / Xlr) * (PSIdr - PSImd)
    iqr = (1 / Xlr) * (PSIqr - PSImq)

    ids_results.append(ids)
    iqs_results.append(iqs)
    idr_results.append(idr)
    iqr_results.append(iqr)

    Te = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs - PSIqs * ids)
    Te_results.append(Te)

    tetar = wr * t_val

    iafs = np.cos(tetae) * ids - np.sin(tetae) * iqs
    ibts = np.sin(tetae) * ids + np.cos(tetae) * iqs

    iafr = np.cos(tetar) * idr - np.sin(tetar) * iqr
    ibtr = np.sin(tetar) * idr + np.cos(tetar) * iqr

    ias = np.sqrt(3/2) * (1 * iafs + 0 * ibts)
    ibs = np.sqrt(3/2) * ((-1/2) * iafs + (np.sqrt(3)/2) * ibts)
    ics = np.sqrt(3/2) * ((-1/2) * iafs + (-np.sqrt(3)/2) * ibts)

    iar = np.sqrt(3/2) * (1 * iafr + 0 * ibtr)
    ibr = np.sqrt(3/2) * ((-1/2) * iafr + (np.sqrt(3)/2) * ibtr)
    icr = np.sqrt(3/2) * ((-1/2) * iafr + (-np.sqrt(3)/2) * ibtr)

    ias_results.append(ias)
    ibs_results.append(ibs)
    ics_results.append(ics)
    iar_results.append(iar)
    ibr_results.append(ibr)
    icr_results.append(icr)

    Va_results.append(Va_val)
    Vb_results.append(Vb_val)
    Vc_results.append(Vc_val)
    Vds_results.append(Vds_val)
    Vqs_results.append(Vqs_val)


# Convertendo para arrays numpy
wr_results  = np.array(wr_results)
Te_results  = np.array(Te_results)
n_results   = np.array(n_results)
ids_results = np.array(ids_results)
iqs_results = np.array(iqs_results)
idr_results = np.array(idr_results)
iqr_results = np.array(iqr_results)
ias_results = np.array(ias_results)
ibs_results = np.array(ibs_results)
ics_results = np.array(ics_results)
iar_results = np.array(iar_results)
ibr_results = np.array(ibr_results)
icr_results = np.array(icr_results)
Va_results  = np.array(Va_results)
Vb_results  = np.array(Vb_results)
Vc_results  = np.array(Vc_results)
Vds_results = np.array(Vds_results)
Vqs_results = np.array(Vqs_results)

print("\nValores (finais) em regime permanente das variáveis:")
print("s f_r ids iqs n")
s_ss = (wb - wr_results[-1]) / wb
fr_ss = wr_results[-1] / (2 * np.pi)
ids_ss = ids_results[-1]
iqs_ss = iqs_results[-1]
n_ss = n_results[-1]
print(f"{s_ss:.4f} {fr_ss:.4f} {ids_ss:.4f} {iqs_ss:.4f} {n_ss:.4f}")

# Gráficos

plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Te_results, label='$T_e$')
plt.title('$T_e$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Torque (Nm)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_values, n_results, label='n (RPM)')
plt.title('n (RPM)', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Velocidade (RPM)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, Va_results, label='$V_a$')
plt.plot(t_values, Vb_results, label='$V_b$')
plt.plot(t_values, Vc_results, label='$V_c$')
plt.title('$V_a$ x $V_b$ x $V_c$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Tensão (V)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t_values, ias_results, label='$i_{as}$')
plt.plot(t_values, ibs_results, label='$i_{bs}$')
plt.plot(t_values, ics_results, label='$i_{cs}$')
plt.title('$I_{as}$ x $I_{bs}$ x $I_{cs}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t_values, iar_results, label='$i_{ar}$')
plt.plot(t_values, ibr_results, label='$i_{br}$')
plt.plot(t_values, icr_results, label='$i_{cr}$')
plt.title('$I_{ar}$ x $I_{br}$ x $I_{cr}$', fontsize=14)
plt.xlabel('tempo (segundos)', fontsize=12)
plt.ylabel('Corrente (A)', fontsize=12)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================================================================
# EXPERIMENTOS DE PARÂMETROS DA MÁQUINA
# ==============================================================================

# @title Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from IPython.display import display

"""---

# O Colab aqui apresentado tem o objetivo de simular situações específicas no funcionamento de uma máquina de indução trifásica.

# **7. Impacto da Resistência Rotórica ( Rr) no Torque de Partida e Escorregamento**
***Objetivo: Entender como a variação de Rr afeta o desempenho da partida e o escorregamento em regime.***

Configuração:

* Modo de Operação: Motor Rr: Variações ( 0.3 Ohm, 1.0 Ohm e 1.2 Ohm)

* Torque mecânico final (Tlfinal): 80.0 Nm

Análise Esperada: Comparar o torque máximo na partida e a velocidade (ou escorregamento) em regime para diferentes valores de  Rr.
"""

# Q7 — Impacto da Resistência Rotórica (Rr) no Torque de Partida e Escorregamento

# PARÂMETROS DA MÁQUINA

p = 4             # nº de polos
f = 60.0          # Hz
V_ll = 220.0      # tensão linha-linha RMS (V)
conn = "Y"        # "Y" (estrela) ou "D" (triângulo)
Rs = 0.435        # ohm
Ls = 0.002        # H (indutância TOTAL de fase do estator)
Lr = 0.002        # H (indutância TOTAL de fase do rotor referido)
Lm = 0.0693       # H (magnetização)
T_load = 80.0     # N·m (torque da carga p/ ponto de operação)

# Valores de Rr:
Rr_list = [0.30, 1.00, 1.20]  # ohm


# IMPLEMENTAÇÃO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, sqrt

def _v_phase(V_ll, conn):
    return V_ll / sqrt(3.0) if str(conn).upper() == "Y" else V_ll

def _split_leak(Ls, Lr, Lm):
    Lls = max(Ls - Lm, 0.0)   # dispersão do estator
    Llr = max(Lr - Lm, 0.0)   # dispersão do rotor
    return Lls, Llr

def _torque_curve_regime(Rr_value, *, p, f, V_ll, conn, Rs, Ls, Lr, Lm, T_load, n_pts=4000):
    # Velocidades síncronas
    w_sync = 4 * pi * f / p       # mecânica (rad/s)
    w_e = 2 * pi * f              # elétrica (rad/s)

    # Tensões/reatâncias
    Vph = _v_phase(V_ll, conn)
    Lls, Llr = _split_leak(Ls, Lr, Lm)
    X1 = w_e * Lls
    X2 = w_e * Llr
    Xm = w_e * Lm

    # Impedâncias do estator e magnetização
    Z1 = complex(Rs, X1)
    Zm = complex(0.0, Xm)

    # Thevenin visto pelo rotor
    Vth = Vph * (Zm / (Z1 + Zm))
    Zth = (Zm * Z1) / (Z1 + Zm)
    Rth, Xth = Zth.real, Zth.imag

    # Varredura de escorregamento (evita s=0)
    s = np.linspace(1e-4, 1.0, n_pts)[::-1]

    # Torque eletromagnético em regime (forma clássica com Thevenin)
    R2 = Rr_value
    num = 3.0 * (abs(Vth)**2) * (R2 / s)
    den = w_sync * ((Rth + (R2 / s))**2 + (Xth + X2)**2)
    Te = num / den  # N·m

    # rpm
    ns_rpm = 120.0 * f / p
    n_rpm = (1 - s) * ns_rpm

    # Métricas
    Te_start = (3.0 * (abs(Vth)**2) * (R2 / 1.0)) / (w_sync * ((Rth + (R2 / 1.0))**2 + (Xth + X2)**2))
    s_Tmax = R2 / np.sqrt(Rth**2 + (Xth + X2)**2)
    s_Tmax = np.clip(s_Tmax, 1e-4, 1.0)
    Te_max  = (3.0 * (abs(Vth)**2) * (R2 / s_Tmax)) / (w_sync * ((Rth + (R2 / s_Tmax))**2 + (Xth + X2)**2))
    n_Tmax_rpm = (1 - s_Tmax) * ns_rpm
    idx = int(np.argmin(np.abs(Te - T_load)))
    s_oper = float(s[idx])
    n_oper_rpm = float(n_rpm[idx])

    return {
        "s": s, "Te": Te, "n_rpm": n_rpm,
        "Te_start": float(Te_start),
        "s_Tmax": float(s_Tmax),
        "Te_max": float(Te_max),
        "n_Tmax_rpm": float(n_Tmax_rpm),
        "s_oper": float(s_oper),
        "n_oper_rpm": float(n_oper_rpm),
        "ns_rpm": float(ns_rpm),
    }

# Validação rápida
if p <= 0 or f <= 0:
    raise ValueError("Parâmetros inválidos: 'p' e 'f' devem ser positivos.")
for name, val in dict(Rs=Rs, Ls=Ls, Lr=Lr, Lm=Lm, V_ll=V_ll, T_load=T_load).items():
    if val < 0:
        raise ValueError(f"Parâmetro inválido: {name} < 0.")

# ===== Execução =====
results = {}
for Rr in Rr_list:
    if Rr <= 0:
        raise ValueError(f"Rr deve ser positivo. Valor recebido: {Rr}")
    results[Rr] = _torque_curve_regime(
        Rr, p=p, f=f, V_ll=V_ll, conn=conn, Rs=Rs, Ls=Ls, Lr=Lr, Lm=Lm, T_load=T_load
    )

# ===== Tabela =====
rows = []
for Rr, r in results.items():
    rows.append({
        "Rr (Ohm)": Rr,
        "T_partida (N.m)": round(r["Te_start"], 2),
        "s_Tmax (-)": round(r["s_Tmax"], 4),
        "Tmax (N.m)": round(r["Te_max"], 2),
        "n_Tmax (rpm)": round(r["n_Tmax_rpm"], 1),
        f"s_oper (T={T_load:.0f} N.m)": round(r["s_oper"], 4),
        "n_oper (rpm)": round(r["n_oper_rpm"], 1),
    })
df = pd.DataFrame(rows).sort_values("Rr (Ohm)").reset_index(drop=True)

# Exibir na UI
try:
    from google.colab import data_table
    data_table.DataTable(df)     # Colab
except Exception:
    try:
        display(df)              # Jupyter
    except Exception:
        print(df.to_string(index=False))

# CSV opcional
try:
    csv_path = "Q7_resultados_Rr_singlecell.csv"
    df.to_csv(csv_path, index=False)
    print("\nArquivo salvo:", csv_path)
except Exception as e:
    print("Aviso: não foi possível salvar CSV ->", e)

# ===== Gráficos (um por figura) =====
# 1) Torque x Velocidade
plt.figure()
for Rr, r in results.items():
    plt.plot(r["n_rpm"], r["Te"], label=f"Rr={Rr:.2f} Ω")
plt.axhline(T_load, linestyle="--", label="Torque da carga")
plt.xlabel("Velocidade (rpm)")
plt.ylabel("Torque eletromagnético (N·m)")
plt.title("Torque x Velocidade para diferentes Rr")
plt.legend(); plt.grid(True); plt.show()

# 2) Torque de partida vs Rr
plt.figure()
Rr_sorted = sorted(results.keys())
Tstart = [results[Rr]["Te_start"] for Rr in Rr_sorted]
plt.plot(Rr_sorted, Tstart, marker="o")
plt.xlabel("Rr (Ohm)")
plt.ylabel("Torque de partida (N·m)")
plt.title("Variação do torque de partida com Rr")
plt.grid(True); plt.show()

# 3) Escorregamento (regime, T=constante) vs Rr
plt.figure()
s_oper = [results[Rr]["s_oper"] for Rr in Rr_sorted]
plt.plot(Rr_sorted, s_oper, marker="o")
plt.xlabel("Rr (Ohm)")
plt.ylabel("Escorregamento em regime (–)")
plt.title("Escorregamento em regime para T_load fixo")
plt.grid(True); plt.show()

"""# **8. Impacto do Momento de Inércia (J) no Tempo de Aceleração**

***Objetivo: Observar como J influencia a dinâmica de aceleração.***

***Configuração:***

- Modo de Operação: Motor
- J: Variações (ex: 0.04 kg·m², 0.15 kg·m²)
- Torque mecânico final ( Tlfinal): 80.0 Nm

Análise Esperada: Medir o tempo para a máquina atingir a velocidade de regime para diferentes valores de J.
"""

import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from ipywidgets import FloatSlider, VBox, interactive_output
from IPython.display import display

# Parâmetros do motor e simulação

p = 4                 # Número de polos do motor
f = 60.0              # Frequência da rede (Hz)
wb = 2 * np.pi * f    # Velocidade angular base (rad/s)

Vl = 220.0    # Tensão de linha (V)
Rs = 0.435    # Resistência do estator (Ohm)
Rr = 0.816    # Resistência do rotor (Ohm)
Xm = 26.13    # Reatância magnetizante (Ohm)
Xls = 0.754   # Reatância de dispersão do estator (Ohm)
Xlr = 0.754   # Reatância de dispersão do rotor (Ohm)
B = 0.0       # Coeficiente de atrito viscoso (Nm.s/rad)

# Reatância magnética equivalente (paralelo de Xm, Xls e Xlr)
Xml = 1 / (1/Xm + 1/Xls + 1/Xlr)

# Cálculo da velocidade síncrona (rad/s e rpm)
w_sync = wb * 2 / p
n_sync_rpm = 120 * f / p

# Faixas e valores padrão para sliders interativos
J_min, J_max, J_default = 0.01, 0.4, 0.04      # Momento de inércia (kg·m²)
Tl_min, Tl_max, Tl_default = 0.0, 150.0, 80.0  # Torque mecânico final (Nm)
tempo_subida_t2 = 2.0  # Tempo para o torque aumentar da carga inicial até final (segundos)

# Torque inicial fixo (partida sem carga)
Tl_initial = 0.0

def torque_carga(t, Tl_initial, Tl_final, t2):
    """Torque de carga variando linearmente até t2 segundos."""
    if t < t2:
        return Tl_initial + (Tl_final - Tl_initial) * (t / t2)  # Rampa linear
    else:
        return Tl_final  # Torque constante após t2 segundos

def motor_0dq(states, t, Vqs_t, Vds_t, Tl_inst, w_ref, J):
    # Desempacota variáveis de estado: fluxos e velocidade rotor
    PSIqs, PSIds, PSIqr, PSIdr, wr = states

    # Fluxo magnético no eixo q e d (equivalente magnético)
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)

    # Correntes de eixo d e q no estator
    ids = (1 / Xls) * (PSIds - PSImd)
    iqs = (1 / Xls) * (PSIqs - PSImq)

    # Equações diferenciais do fluxo do estator e rotor
    dPSIqs_dt = wb * (Vqs_t - (w_ref / wb) * PSIds + (Rs / Xls) * (PSImq - PSIqs))
    dPSIds_dt = wb * (Vds_t + (w_ref / wb) * PSIqs + (Rs / Xls) * (PSImd - PSIds))
    dPSIqr_dt = wb * (0 - ((w_ref - wr) / wb) * PSIdr + (Rr / Xlr) * (PSImq - PSIqr))
    dPSIdr_dt = wb * (0 + ((w_ref - wr) / wb) * PSIqr + (Rr / Xlr) * (PSImd - PSIdr))

    # Torque eletromagnético instantâneo (Nm)
    Tem = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs - PSIqs * ids)

    # Derivada da velocidade angular do rotor (aceleração)
    dwr_dt = (p / (2 * J)) * (Tem - Tl_inst) - (B / J) * wr

    # Retorna derivadas para o solver ODE
    return [dPSIqs_dt, dPSIds_dt, dPSIqr_dt, dPSIdr_dt, dwr_dt]

def simular_motor(J, Tl_initial, Tl_final, t2, tmax=5.0, h=0.0005):
    """Simula o motor no intervalo de tempo dado, retorna dados para análise."""
    t_values = np.arange(0, tmax, h)  # Vetor de tempos
    states = [0, 0, 0, 0, 0]           # Condições iniciais: fluxos e velocidade zero
    w_ref = wb                       # Velocidade síncrona como referência

    # Listas para armazenar resultados
    wr_res = []
    Te_res = []
    n_res = []

    for t in t_values:
        # Torque mecânico instantâneo na rampa
        Tl_inst = torque_carga(t, Tl_initial, Tl_final, t2)

        # Tensões trifásicas no tempo t (senoidal, base RMS para pico)
        thetae = wb * t
        Va = np.sqrt(2) * Vl * np.sin(thetae)
        Vb = np.sqrt(2) * Vl * np.sin(thetae - 2*np.pi/3)
        Vc = np.sqrt(2) * Vl * np.sin(thetae + 2*np.pi/3)

        # Transformação de Clarke para referência estacionária
        Vaf = np.sqrt(3/2) * (Va - 0.5*Vb - 0.5*Vc)
        Vbt = np.sqrt(3/2) * (0*Va + (np.sqrt(3)/2)*Vb - (np.sqrt(3)/2)*Vc)

        # Transformação Park para referência síncrona dq
        Vds = np.cos(thetae)*Vaf + np.sin(thetae)*Vbt
        Vqs = -np.sin(thetae)*Vaf + np.cos(thetae)*Vbt

        # Integra as equações do motor de t até t+h
        sol = odeint(motor_0dq, states, [t, t+h], args=(Vqs, Vds, Tl_inst, w_ref, J))
        states = sol[1]  # Atualiza estados para próxima iteração
        PSIqs, PSIds, PSIqr, PSIdr, wr = states

        # Armazena velocidade em rpm
        wr_res.append(wr)
        n = (120 / p) * (wr / (2 * np.pi))
        n_res.append(n)

        # Calcula torque eletromagnético para análise
        PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
        PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
        ids = (1 / Xls) * (PSIds - PSImd)
        iqs = (1 / Xls) * (PSIqs - PSImq)
        Te = (3 / 2) * (p / 2) * (1 / wb) * (PSIds * iqs - PSIqs * ids)
        Te_res.append(Te)

    # Converte resultados para arrays NumPy
    wr_res = np.array(wr_res)
    n_res = np.array(n_res)
    Te_res = np.array(Te_res)
    t_values = np.array(t_values)

    # Calcula o tempo que o motor leva para atingir 95% da velocidade síncrona
    n_target = 0.95 * n_sync_rpm
    t95 = np.nan  # Caso não alcance
    idxs = np.where(n_res >= n_target)[0]
    if len(idxs) > 0:
        t95 = t_values[idxs[0]]

    return t_values, n_res, Te_res, t95

from google.colab import output
output.enable_custom_widget_manager()  # Ativa suporte a widgets no Colab

def atualizar_grafico(J_value, Tl_final_value):
    # Executa a simulação para os parâmetros atuais dos sliders
    t_values, n_res, Te_res, t95 = simular_motor(J_value, Tl_initial, Tl_final_value, tempo_subida_t2)

    # Cria figura com Plotly
    fig = go.Figure()
    # Adiciona curva da velocidade em rpm
    fig.add_trace(go.Scatter(x=t_values, y=n_res, mode='lines', name='Velocidade (RPM)',
                             line=dict(color='royalblue', width=3)))
    # Adiciona curva do torque eletromagnético
    fig.add_trace(go.Scatter(x=t_values, y=Te_res, mode='lines', name='Torque Eletromagnético (Nm)',
                             line=dict(color='firebrick', width=3, dash='dash')))

    # Linha horizontal marcando 95% da velocidade síncrona
    fig.add_shape(type="line", x0=t_values[0], y0=0.95 * n_sync_rpm,
                  x1=t_values[-1], y1=0.95 * n_sync_rpm,
                  line=dict(color="green", width=2, dash="dot"))
    # Anotação explicativa da linha de referência
    fig.add_annotation(x=t_values[int(len(t_values)*0.6)], y=0.95 * n_sync_rpm,
                       text="95% Vel. Síncrona", showarrow=False,
                       font=dict(color="green", size=12), bgcolor="white", opacity=0.8)

    # Configura layout do gráfico
    fig.update_layout(
        title=f"Simulação Motor Indução - J={J_value:.3f} kg·m², Torque Final={Tl_final_value:.1f} Nm",
        xaxis_title="Tempo (s)",
        yaxis_title="Velocidade (RPM) / Torque (Nm)",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    # Exibe gráfico
    fig.show()

    # Imprime tempo para atingir 95% da velocidade síncrona
    print(f"Tempo para atingir 95% da velocidade síncrona: {t95:.3f} s")

# Configuração dos sliders para parâmetros interativos
slider_J = FloatSlider(value=J_default, min=J_min, max=J_max, step=0.005,
                       description='Momento de Inércia J (kg·m²)', continuous_update=False)
slider_Tl = FloatSlider(value=Tl_default, min=Tl_min, max=80.0, step=1.0,
                       description='Torque final (Nm)', continuous_update=False)

# Liga sliders à função que atualiza o gráfico
out = interactive_output(atualizar_grafico, {'J_value': slider_J, 'Tl_final_value': slider_Tl})

# Exibe sliders e gráfico na vertical
display(VBox([slider_J, slider_Tl, out]))

"""Percebe-se que a alteração do valor de J influencia no tempo em que a máquina leva para chegar a velocidade síncrona. Valore menores permitem uma aceleração maior mas deixam o sistema mais oscilante. Valores maiores diminuem a aceleração mas fazem com que o motor tenha uma caracterítica mais estável.

#9. Variação da Frequência da Rede
* Objetivo: Analisar o impacto da frequência nominal no comportamento do motor.

Configuração:

* Modo de Operação: Motor
* Frequência (f): Variações (ex: 50.0 Hz, 40.0 Hz)
* Tensão RMS de entrada ( Vl): 220.0 V (ou ajustar para V/f constante, ex: 183V para 50Hz)
* Torque mecânico final ( Tlfinal ): 80.0 Nm

Análise Esperada: Observar a velocidade síncrona, a velocidade de regime e o escorregamento.
"""

# Questão 9 - Simulação da variação da frequência da rede em motor de indução trifásico

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Dados do motor (valores típicos de catálogo)
# -------------------------
R1 = 0.641      # Resistência do estator [ohm]
R2 = 0.332      # Resistência do rotor referida ao estator [ohm]
X1 = 1.106      # Reatância do estator [ohm]
X2 = 0.464      # Reatância do rotor referida ao estator [ohm]
Xm = 26.3       # Reatância de magnetização [ohm]
polos = 4       # Número de polos
T_mec = 80      # Torque mecânico aplicado [Nm]
V_nom = 220     # Tensão nominal RMS fase-fase [V]
f_nom = 50      # Frequência nominal [Hz]

# Frequências de teste
frequencias = [50, 40]

# Função para calcular torque em função do escorregamento
def torque_motor(V, f, s):
    ws = 2 * np.pi * f / (polos/2)  # Velocidade síncrona elétrica [rad/s]
    Z1 = R1 + 1j*X1
    Zm = 1j*Xm
    Z2 = R2/s + 1j*X2
    Z_th = Z1 + (Zm*Z2)/(Zm+Z2)
    I2 = V / Z_th
    P_conv = 3 * (abs(I2)**2) * (R2/s) * (1-s)  # Potência convertida mecânica [W]
    Te = P_conv / (ws*(1-s))
    return Te

# Loop para simular e exibir resultados
for f in frequencias:
    print(f"\n=== Resultados para f = {f} Hz ===")
    V = V_nom * (f / f_nom)  # Ajuste de tensão para manter V/f constante
    ns = 120 * f / polos     # Velocidade síncrona [rpm]

    s_vals = np.linspace(0.001, 0.1, 200)
    T_vals = [torque_motor(V, f, s) for s in s_vals]

    # Encontrar velocidade de regime (torque = T_mec)
    idx_regime = np.argmin(np.abs(np.array(T_vals) - T_mec))
    s_regime = s_vals[idx_regime]
    n_regime = ns * (1 - s_regime)
    ws_regime = 2 * np.pi * n_regime / 60

    # Corrente no regime
    Z1 = R1 + 1j*X1
    Zm = 1j*Xm
    Z2 = R2/s_regime + 1j*X2
    Z_th = Z1 + (Zm*Z2)/(Zm+Z2)
    I_regime = V / Z_th

    # Potência mecânica e eficiência
    P_mec = (2*np.pi*n_regime/60) * T_mec
    P_entrada = 3 * abs(I_regime)**2 * R1 + P_mec/(1 - s_regime)
    eficiencia = P_mec / P_entrada * 100

    # Resultados numéricos
    print(f"Velocidade síncrona: {ns:.2f} rpm")
    print(f"Velocidade de regime: {n_regime:.2f} rpm")
    print(f"Escorregamento: {s_regime*100:.2f} %")
    print(f"Corrente de regime: {abs(I_regime):.2f} A")
    print(f"Potência mecânica: {P_mec:.2f} W")
    print(f"Eficiência: {eficiencia:.2f} %")

    # Gráfico Torque x Velocidade
    n_vals = ns * (1 - s_vals)
    plt.plot(n_vals, T_vals, label=f"{f} Hz")

plt.axhline(T_mec, color='k', linestyle='--', label='Torque Mecânico')
plt.xlabel("Velocidade (rpm)")
plt.ylabel("Torque Eletromagnético (Nm)")
plt.title("Torque x Velocidade para diferentes frequências")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# Escorregamento x Torque com pontos de regime
# ===============================
plt.figure(figsize=(8,6))

for f in frequencias:
    V = V_nom * (f / f_nom)
    s_vals = np.linspace(0.001, 1.0, 300)  # até 100% de escorregamento
    T_vals = [torque_motor(V, f, s) for s in s_vals]

    plt.plot(T_vals, s_vals*100, label=f"{f} Hz")

    # Calcular ponto de regime
    idx_regime = np.argmin(np.abs(np.array(T_vals) - T_mec))
    plt.plot(T_vals[idx_regime], s_vals[idx_regime]*100, 'o', label=f"Regime {f} Hz")

    # Anotar valor numérico
    plt.text(T_vals[idx_regime], s_vals[idx_regime]*100 + 1,
             f"{s_vals[idx_regime]*100:.2f}%", ha='center')

plt.xlabel("Torque eletromagnético (Nm)")
plt.ylabel("Escorregamento (%)")
plt.title("Escorregamento x Torque para diferentes frequências")
plt.grid(True)
plt.legend()
plt.show()

print("""
Análise:
- A 50 Hz, o motor apresenta velocidade síncrona de 1500 rpm e de regime ~1457 rpm (escorregamento ≈ 2,84%).
- A 40 Hz, a velocidade síncrona reduz para 1200 rpm e o regime ~1151 rpm (escorregamento ≈ 4,08%).
- A redução de frequência aumenta o escorregamento necessário para gerar o mesmo torque.
""")
