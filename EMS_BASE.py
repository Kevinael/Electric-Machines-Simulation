# -*- coding: utf-8 -*-
"""
EMS_BASE.py — Simulador Unificado de Máquina de Indução Trifásica (Gaiola de Esquilo)
Modelo matemático 0dq de Krause, integração numérica via RK4 (scipy.odeint).

==============================================================================
DICIONÁRIO DE VARIÁVEIS
==============================================================================

--- Parâmetros Elétricos da Máquina ---
Vl      : Tensão RMS de linha [V]
f       : Frequência da rede [Hz]
Rs      : Resistência do estator [Ω]
Rr      : Resistência do rotor (referida ao estator) [Ω]
Xm      : Reatância de magnetização [Ω]
Xls     : Reatância de dispersão do estator [Ω]
Xlr     : Reatância de dispersão do rotor (referida ao estator) [Ω]
Xml     : Reatância mútua equivalente (paralelo de Xm, Xls, Xlr) [Ω]
wb      : Velocidade angular base (2·π·f) [rad/s]

--- Parâmetros Mecânicos ---
p       : Número de polos
J       : Momento de inércia do conjunto rotor-carga [kg·m²]
B       : Coeficiente de atrito viscoso [N·m·s/rad]

--- Variáveis de Estado (fluxos e velocidade) ---
PSIqs   : Fluxo concatenado do estator no eixo q [Wb]
PSIds   : Fluxo concatenado do estator no eixo d [Wb]
PSIqr   : Fluxo concatenado do rotor no eixo q [Wb]
PSIdr   : Fluxo concatenado do rotor no eixo d [Wb]
wr      : Velocidade angular mecânica do rotor [rad/s]

--- Fluxos e Correntes Derivados ---
PSImq   : Fluxo de magnetização no eixo q [Wb]
PSImd   : Fluxo de magnetização no eixo d [Wb]
ids     : Corrente do estator no eixo d [A]
iqs     : Corrente do estator no eixo q [A]
idr     : Corrente do rotor no eixo d [A]
iqr     : Corrente do rotor no eixo q [A]

--- Grandezas Trifásicas (abc) ---
Va, Vb, Vc  : Tensões de fase no domínio do tempo [V] (pico)
ias, ibs, ics : Correntes de fase do estator [A]
iar, ibr, icr : Correntes de fase do rotor [A]

--- Transformada de Park ---
tetae   : Ângulo elétrico de referência (ωe·t) [rad]
tetar   : Ângulo mecânico do rotor (∫wr·dt) [rad]
w_ref   : Velocidade angular do referencial dq [rad/s]
           ref=1 → síncrona (w_ref = we)
           ref=2 → rotórica (w_ref = wr)
           ref=3 → estacionária (w_ref = 0)
Vds, Vqs : Tensões do estator nos eixos d e q [V]
Vaf, Vbt : Componentes α e β (Clarke) da tensão de fase [V]

--- Grandezas do Sistema ---
Te      : Torque eletromagnético [N·m]
Tl      : Torque mecânico de carga (resistente) [N·m]
n       : Velocidade de rotação [rpm]
n_sync  : Velocidade síncrona [rpm]  (= 120·f / p)
s       : Escorregamento (slip)  (= (wb - wr) / wb) [adimensional]
P       : Potência ativa trifásica instantânea [W]

--- Parâmetros de Simulação ---
tmax    : Tempo total de simulação [s]
h       : Passo de integração [s]
t_carga : Instante de aplicação / alteração do torque de carga [s]
t_2     : Instante de alteração de tensão (partidas Y-Δ, compensadora, gerador) [s]
t_pico  : Tempo até atingir tensão nominal na rampa do soft-starter [s]

==============================================================================
GLOSSÁRIO DE SIGLAS
==============================================================================

--- Métodos de Partida ---
DOL         : Direct On-Line — Partida direta: motor ligado à rede em tensão plena.
              Simples, mas gera corrente de partida muito alta (~6–8× nominal).
Y-Δ         : Estrela-Triângulo (do alemão Stern-Dreieck) — Motor parte com bobinas
              em estrela (tensão reduzida a 1/√3) e comuta para triângulo (tensão
              plena). Reduz a corrente de partida a ~1/3 da DOL.
Soft-Starter: Controle eletrônico que aumenta a tensão gradualmente (rampa),
              limitando corrente e torque de choque durante a partida.
Compensadora: Partida com autotransformador que aplica tensão reduzida ao motor
              durante a partida e comuta para tensão plena após aceleração inicial.

--- Modelo Matemático ---
0dq / dq0   : Referencial de Park — transforma as 3 fases em dois eixos ortogonais:
              d (direto) e q (em quadratura), mais componente zero.
              Simplifica as equações diferenciais do motor a coeficientes constantes.
RK4         : Runge-Kutta de 4ª ordem — método numérico de integração de EDOs
              (Equações Diferenciais Ordinárias) com boa estabilidade e precisão.
Clarke      : Transformada αβ — passo intermediário da transformada de Park,
              projeta as 3 fases em dois eixos estacionários (α, β).
EDO         : Equação Diferencial Ordinária.

--- Grandezas Elétricas ---
RMS         : Root Mean Square — valor eficaz (raiz quadrática média).
PSI (Ψ)     : Fluxo concatenado (linkage flux) [Wb].
ss          : Steady State — regime permanente (estado estacionário).

==============================================================================
"""

# =============================================================================
# SEÇÃO 1 — IMPORTAÇÕES
# =============================================================================
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dataclasses import dataclass, field
from typing import Callable


# =============================================================================
# SEÇÃO 2 — PARÂMETROS DA MÁQUINA (dataclass com derivações automáticas)
# =============================================================================

@dataclass
class MachineParams:
    """
    Agrupa todos os parâmetros físicos da máquina de indução.
    Os campos Xml e wb são calculados automaticamente a partir dos dados de placa.
    """
    # Dados elétricos nominais
    Vl:  float = 220.0   # tensão de linha RMS [V]
    f:   float = 60.0    # frequência [Hz]
    Rs:  float = 0.435   # resistência do estator [Ω]
    Rr:  float = 0.816   # resistência do rotor [Ω]
    Xm:  float = 26.13   # reatância de magnetização [Ω]
    Xls: float = 0.754   # reatância de dispersão do estator [Ω]
    Xlr: float = 0.754   # reatância de dispersão do rotor [Ω]
    # Dados mecânicos
    p:   int   = 4       # número de polos
    J:   float = 0.089   # inércia [kg·m²]
    B:   float = 0.0     # atrito viscoso [N·m·s/rad]
    # Calculados automaticamente
    Xml: float = field(init=False)  # reatância mútua equivalente
    wb:  float = field(init=False)  # velocidade angular base [rad/s]

    def __post_init__(self) -> None:
        self.Xml = 1.0 / (1.0/self.Xm + 1.0/self.Xls + 1.0/self.Xlr)
        self.wb  = 2.0 * np.pi * self.f

    @property
    def n_sync(self) -> float:
        """Velocidade síncrona em rpm."""
        return 120.0 * self.f / self.p


# Instância padrão utilizada por todos os experimentos
DEFAULT_MACHINE = MachineParams()


# =============================================================================
# SEÇÃO 3 — MOTOR ODE (núcleo físico — definido uma única vez)
# =============================================================================

def induction_motor_ode(
    states: list[float],
    t: float,
    Vqs: float,
    Vds: float,
    Tl:  float,
    w_ref: float,
    mp: MachineParams,
) -> list[float]:
    """
    Equações de estado da máquina de indução no referencial arbitrário dq0
    (modelo de Krause, rotor em gaiola de esquilo — Vqr = Vdr = 0).

    Derivadas dos fluxos concatenados do estator e rotor, mais a equação
    de movimento do eixo (segunda lei de Newton rotacional).

    Parameters
    ----------
    states : [PSIqs, PSIds, PSIqr, PSIdr, wr]
    t      : tempo atual [s] (requerido pela assinatura do odeint)
    Vqs    : tensão do estator no eixo q [V]
    Vds    : tensão do estator no eixo d [V]
    Tl     : torque mecânico de carga [N·m]
    w_ref  : velocidade angular do referencial [rad/s]
    mp     : parâmetros da máquina

    Returns
    -------
    [dPSIqs/dt, dPSIds/dt, dPSIqr/dt, dPSIdr/dt, dwr/dt]
    """
    PSIqs, PSIds, PSIqr, PSIdr, wr = states

    # Fluxos de magnetização (nó magnético)
    PSImq = mp.Xml * (PSIqs / mp.Xls + PSIqr / mp.Xlr)
    PSImd = mp.Xml * (PSIds / mp.Xls + PSIdr / mp.Xlr)

    # Correntes do estator (deduzidas dos fluxos)
    iqs = (1.0 / mp.Xls) * (PSIqs - PSImq)
    ids = (1.0 / mp.Xls) * (PSIds - PSImd)

    # Equações dos fluxos do estator
    dPSIqs = mp.wb * (Vqs - (w_ref / mp.wb) * PSIds + (mp.Rs / mp.Xls) * (PSImq - PSIqs))
    dPSIds = mp.wb * (Vds + (w_ref / mp.wb) * PSIqs + (mp.Rs / mp.Xls) * (PSImd - PSIds))

    # Equações dos fluxos do rotor (Vqr = Vdr = 0 → gaiola de esquilo)
    slip_ref = (w_ref - wr) / mp.wb
    dPSIqr = mp.wb * (-slip_ref * PSIdr + (mp.Rr / mp.Xlr) * (PSImq - PSIqr))
    dPSIdr = mp.wb * ( slip_ref * PSIqr + (mp.Rr / mp.Xlr) * (PSImd - PSIdr))

    # Torque eletromagnético e equação mecânica
    Te   = (3.0 / 2.0) * (mp.p / 2.0) * (1.0 / mp.wb) * (PSIds * iqs - PSIqs * ids)
    dwr  = (mp.p / (2.0 * mp.J)) * (Te - Tl) - (mp.B / mp.J) * wr

    return [dPSIqs, dPSIds, dPSIqr, dPSIdr, dwr]


# =============================================================================
# SEÇÃO 4 — FUNÇÕES AUXILIARES DE TRANSFORMADAS E SINAIS
# =============================================================================

def abc_voltages(t: float, Vl: float, f: float) -> tuple[float, float, float]:
    """Gera as tensões trifásicas de pico no instante t."""
    tetae = 2.0 * np.pi * f * t
    Va = np.sqrt(2.0) * Vl * np.sin(tetae)
    Vb = np.sqrt(2.0) * Vl * np.sin(tetae - 2.0 * np.pi / 3.0)
    Vc = np.sqrt(2.0) * Vl * np.sin(tetae + 2.0 * np.pi / 3.0)
    return Va, Vb, Vc


def clarke_park_transform(Va: float, Vb: float, Vc: float, tetae: float) -> tuple[float, float]:
    """
    Transformada de Clarke (abc → αβ) seguida de Park (αβ → dq).
    Convenção amplitude-invariante com fator √(3/2).

    Returns
    -------
    Vds, Vqs : componentes d e q da tensão do estator
    """
    # Clarke: projeção nas componentes α e β
    Valpha = np.sqrt(3.0 / 2.0) * (Va - 0.5 * Vb - 0.5 * Vc)
    Vbeta  = np.sqrt(3.0 / 2.0) * ((np.sqrt(3.0) / 2.0) * Vb - (np.sqrt(3.0) / 2.0) * Vc)
    # Park: rotação pelo ângulo do referencial
    Vds = np.cos(tetae) * Valpha + np.sin(tetae) * Vbeta
    Vqs = -np.sin(tetae) * Valpha + np.cos(tetae) * Vbeta
    return Vds, Vqs


def direct_park_transform(Va: float, Vb: float, Vc: float, teta_ref: float) -> tuple[float, float]:
    """
    Transformada de Park direta (abc → dq sem etapa Clarke explícita).
    Convenção de amplitude com fator 2/3, usada pelo Experimento 12.

    Nota: matematicamente equivalente à clarke_park_transform para sistemas
    trifásicos equilibrados, mas com escalonamento e definição de eixo distintos.
    """
    Vqs = (2.0 / 3.0) * (
        Va * np.cos(teta_ref)
        + Vb * np.cos(teta_ref - 2.0 * np.pi / 3.0)
        + Vc * np.cos(teta_ref + 2.0 * np.pi / 3.0)
    )
    Vds = -(2.0 / 3.0) * (
        Va * np.sin(teta_ref)
        + Vb * np.sin(teta_ref - 2.0 * np.pi / 3.0)
        + Vc * np.sin(teta_ref + 2.0 * np.pi / 3.0)
    )
    return Vds, Vqs


def resolve_w_ref(ref_code: int, we: float, last_wr: float) -> float:
    """
    Retorna a velocidade angular do referencial dq conforme a escolha.

    ref_code = 1 → síncrono  (w_ref = we)
    ref_code = 2 → rotórico  (w_ref = wr atual)
    ref_code = 3 → estacionário (w_ref = 0)
    """
    if ref_code == 1:
        return we
    elif ref_code == 2:
        return last_wr
    return 0.0


def reconstruct_abc_currents(
    PSIqs: float, PSIds: float, PSIqr: float, PSIdr: float,
    tetae: float, tetar: float, mp: MachineParams,
) -> tuple:
    """
    Reconstrói as correntes trifásicas do estator e do rotor a partir
    dos estados dq, via Park inverso e Clarke inverso.

    Returns
    -------
    ids, iqs, idr, iqr, ias, ibs, ics, iar, ibr, icr
    """
    PSImq = mp.Xml * (PSIqs / mp.Xls + PSIqr / mp.Xlr)
    PSImd = mp.Xml * (PSIds / mp.Xls + PSIdr / mp.Xlr)

    ids = (1.0 / mp.Xls) * (PSIds - PSImd)
    iqs = (1.0 / mp.Xls) * (PSIqs - PSImq)
    idr = (1.0 / mp.Xlr) * (PSIdr - PSImd)
    iqr = (1.0 / mp.Xlr) * (PSIqr - PSImq)

    # Park inverso do estator (dq → αβ)
    ias_alpha = np.cos(tetae) * ids - np.sin(tetae) * iqs
    ias_beta  = np.sin(tetae) * ids + np.cos(tetae) * iqs

    # Park inverso do rotor
    iar_alpha = np.cos(tetar) * idr - np.sin(tetar) * iqr
    iar_beta  = np.sin(tetar) * idr + np.cos(tetar) * iqr

    # Clarke inverso (αβ → abc), fator √(3/2)
    k = np.sqrt(3.0 / 2.0)
    ias = k * ias_alpha
    ibs = k * (-0.5 * ias_alpha + (np.sqrt(3.0) / 2.0) * ias_beta)
    ics = k * (-0.5 * ias_alpha - (np.sqrt(3.0) / 2.0) * ias_beta)

    iar = k * iar_alpha
    ibr = k * (-0.5 * iar_alpha + (np.sqrt(3.0) / 2.0) * iar_beta)
    icr = k * (-0.5 * iar_alpha - (np.sqrt(3.0) / 2.0) * iar_beta)

    return ids, iqs, idr, iqr, ias, ibs, ics, iar, ibr, icr


# =============================================================================
# SEÇÃO 5 — FUNÇÕES DE AGENDAMENTO DE TORQUE E TENSÃO
# =============================================================================

def torque_step(t: float, Tl_before: float, Tl_after: float, t_switch: float) -> float:
    """Torque de carga em degrau: Tl_before antes de t_switch, Tl_after após."""
    return Tl_after if t >= t_switch else Tl_before


def torque_ramp(t: float, Tl_initial: float, Tl_final: float, t_ramp: float) -> float:
    """
    Rampa linear de torque: cresce de Tl_initial a Tl_final até t_ramp,
    permanece constante em Tl_final após t_ramp.
    """
    if t >= t_ramp:
        return Tl_final
    return Tl_initial + (Tl_final - Tl_initial) * (t / t_ramp)


def voltage_reduced_start(t: float, Vl_nominal: float, Vl_reduced: float, t_switch: float) -> float:
    """
    Tensão reduzida antes de t_switch (partidas Y-Δ e compensadora),
    nominal após t_switch.
    """
    return Vl_nominal if t >= t_switch else Vl_reduced


def voltage_soft_starter(
    t: float, Vl_nominal: float, Vl_initial: float, t_start_ramp: float, t_full: float
) -> float:
    """
    Tensão constante em Vl_initial até t_start_ramp, rampa linear até
    Vl_nominal em t_full, constante após t_full (soft-starter).
    """
    if t < t_start_ramp:
        return Vl_initial
    elif t < t_full:
        return Vl_initial + (Vl_nominal - Vl_initial) * (t - t_start_ramp) / (t_full - t_start_ramp)
    return Vl_nominal


# =============================================================================
# SEÇÃO 6 — ESTRUTURA DE RESULTADOS
# =============================================================================

@dataclass
class SimResults:
    """Armazena todos os vetores de resultado de uma simulação no tempo."""
    t:   np.ndarray
    wr:  np.ndarray
    n:   np.ndarray
    Te:  np.ndarray
    ids: np.ndarray
    iqs: np.ndarray
    idr: np.ndarray
    iqr: np.ndarray
    ias: np.ndarray
    ibs: np.ndarray
    ics: np.ndarray
    iar: np.ndarray
    ibr: np.ndarray
    icr: np.ndarray
    Va:  np.ndarray
    Vb:  np.ndarray
    Vc:  np.ndarray
    Vds: np.ndarray
    Vqs: np.ndarray

    def steady_state(self) -> dict:
        """Retorna um dicionário com os valores de regime permanente (último passo)."""
        mp_wb = 2.0 * np.pi * (self.t[1] - self.t[0])  # aproximação de wb via h
        # Velocidade angular base recalculada a partir de n e wr
        return {
            "wr_ss":  float(self.wr[-1]),
            "n_ss":   float(self.n[-1]),
            "Te_ss":  float(self.Te[-1]),
            "ids_ss": float(self.ids[-1]),
            "iqs_ss": float(self.iqs[-1]),
        }


# =============================================================================
# SEÇÃO 7 — MOTOR DE SIMULAÇÃO UNIFICADO
# =============================================================================

def run_simulation(
    mp:         MachineParams,
    tmax:       float,
    h:          float,
    voltage_fn: Callable[[float], float],
    torque_fn:  Callable[[float], float],
    ref_code:   int = 1,
    wr_initial: float = 0.0,
    use_direct_park: bool = False,
) -> SimResults:
    """
    Executa a simulação no domínio do tempo da máquina de indução.

    A cada passo de tempo:
      1. Determina tensão aplicada (voltage_fn) e torque de carga (torque_fn).
      2. Calcula as tensões dq via transformada de Clarke-Park.
      3. Integra as equações de estado por um passo (odeint/RK45).
      4. Reconstrói as correntes trifásicas.

    Parameters
    ----------
    mp              : parâmetros da máquina
    tmax            : tempo total de simulação [s]
    h               : passo de integração [s]
    voltage_fn      : callable(t) → Vl_aplicada [V RMS de linha]
    torque_fn       : callable(t) → Tl [N·m]
    ref_code        : 1=síncrono, 2=rotórico, 3=estacionário
    wr_initial      : velocidade inicial do rotor [rad/s]
    use_direct_park : se True, usa a convenção direta de Park (Experimento 12)

    Returns
    -------
    SimResults com todos os vetores de resultado indexados no tempo.
    """
    t_values = np.arange(0.0, tmax, h)
    N = len(t_values)

    print(f"  [Simulação] Iniciando integração numérica...")
    print(f"  [Simulação] t_max={tmax}s | h={h}s | passos={N} | ref={'síncrono' if ref_code==1 else 'rotórico' if ref_code==2 else 'estacionário'}")

    # Pré-alocação dos arrays de resultado
    wr_r  = np.empty(N)
    n_r   = np.empty(N)
    Te_r  = np.empty(N)
    ids_r = np.empty(N); iqs_r = np.empty(N)
    idr_r = np.empty(N); iqr_r = np.empty(N)
    ias_r = np.empty(N); ibs_r = np.empty(N); ics_r = np.empty(N)
    iar_r = np.empty(N); ibr_r = np.empty(N); icr_r = np.empty(N)
    Va_r  = np.empty(N); Vb_r  = np.empty(N); Vc_r  = np.empty(N)
    Vds_r = np.empty(N); Vqs_r = np.empty(N)

    # Condições iniciais e estado do integrador
    states   = [0.0, 0.0, 0.0, 0.0, wr_initial]
    last_wr  = wr_initial
    tetar    = 0.0   # integral acumulada de wr (usado no ref rotórico do Exp 12)

    we = mp.wb  # frequência angular elétrica (constante para barramento infinito)

    _milestones = {int(N * p / 100) for p in (25, 50, 75)}

    for i, t_val in enumerate(t_values):
        if i in _milestones:
            pct = round(i / N * 100)
            print(f"  [Simulação] {pct}% concluído (t={t_val:.3f}s | n={n_r[i-1]:.1f} RPM)" if i > 0 else "")
        Vl_apli    = voltage_fn(t_val)
        current_Tl = torque_fn(t_val)

        # Ângulo elétrico do referencial síncrono
        tetae = we * t_val

        # Velocidade e ângulo do referencial escolhido
        w_ref = resolve_w_ref(ref_code, we, last_wr)

        if use_direct_park:
            # Referencial com acumulação de integral (Experimento 12, ref rotórico)
            if ref_code == 2:
                tetar  += last_wr * h
                teta_ref = tetar
            elif ref_code == 1:
                teta_ref = we * t_val
            else:
                teta_ref = 0.0
            V_amp = np.sqrt(2.0 / 3.0) * Vl_apli
            Va = V_amp * np.sin(we * t_val)
            Vb = V_amp * np.sin(we * t_val - 2.0 * np.pi / 3.0)
            Vc = V_amp * np.sin(we * t_val + 2.0 * np.pi / 3.0)
            Vds, Vqs = direct_park_transform(Va, Vb, Vc, teta_ref)
        else:
            Va, Vb, Vc = abc_voltages(t_val, Vl_apli, mp.f)
            Vds, Vqs   = clarke_park_transform(Va, Vb, Vc, tetae)

        # Integração numérica: um passo de [t_val, t_val + h]
        sol    = odeint(induction_motor_ode, states, [t_val, t_val + h],
                        args=(Vqs, Vds, current_Tl, w_ref, mp))
        states = sol[1]
        PSIqs, PSIds, PSIqr, PSIdr, wr = states
        last_wr = wr

        # Ângulo mecânico do rotor para a reconstrução das correntes abc
        tetar_abc = wr * t_val  # convenção consistente com o código original

        # Reconstrução das correntes trifásicas
        ids, iqs, idr, iqr, ias, ibs, ics, iar, ibr, icr = reconstruct_abc_currents(
            PSIqs, PSIds, PSIqr, PSIdr, tetae, tetar_abc, mp
        )

        # Torque eletromagnético instantâneo
        Te = (3.0 / 2.0) * (mp.p / 2.0) * (1.0 / mp.wb) * (PSIds * iqs - PSIqs * ids)

        # Armazenamento
        wr_r[i]  = wr
        n_r[i]   = (120.0 / mp.p) * (wr / (2.0 * np.pi))
        Te_r[i]  = Te
        ids_r[i] = ids; iqs_r[i] = iqs
        idr_r[i] = idr; iqr_r[i] = iqr
        ias_r[i] = ias; ibs_r[i] = ibs; ics_r[i] = ics
        iar_r[i] = iar; ibr_r[i] = ibr; icr_r[i] = icr
        Va_r[i]  = Va;  Vb_r[i]  = Vb;  Vc_r[i]  = Vc
        Vds_r[i] = Vds; Vqs_r[i] = Vqs

    print(f"  [Simulação] 100% concluído (t={t_values[-1]:.3f}s | n={n_r[-1]:.1f} RPM | Te={Te_r[-1]:.2f} N·m)")

    return SimResults(
        t=t_values,   wr=wr_r,   n=n_r,   Te=Te_r,
        ids=ids_r,    iqs=iqs_r, idr=idr_r, iqr=iqr_r,
        ias=ias_r,    ibs=ibs_r, ics=ics_r,
        iar=iar_r,    ibr=ibr_r, icr=icr_r,
        Va=Va_r,      Vb=Vb_r,   Vc=Vc_r,
        Vds=Vds_r,   Vqs=Vqs_r,
    )


# =============================================================================
# SEÇÃO 8 — MODELOS ANALÍTICOS EM REGIME PERMANENTE (sem integração no tempo)
# =============================================================================

def thevenin_torque_speed(
    Rr: float,
    mp: MachineParams,
    T_load: float = 80.0,
    n_pts:  int   = 4_000,
) -> dict:
    """
    Curva de Torque × Velocidade em regime permanente via circuito equivalente
    de Thevenin (modelo clássico de circuito monofásico equivalente).

    Usado pelo Experimento 7 (varredura de Rr).

    Parameters
    ----------
    Rr     : resistência do rotor a ser avaliada [Ω]
    mp     : parâmetros da máquina (Rr de mp é ignorado; usa o argumento Rr)
    T_load : torque de carga para identificar o ponto de operação [N·m]
    n_pts  : pontos na varredura de escorregamento

    Returns
    -------
    dict com arrays s, Te, n_rpm e métricas escalares de partida e regime.
    """
    from math import sqrt

    we    = 2.0 * np.pi * mp.f
    w_syn = 4.0 * np.pi * mp.f / mp.p  # velocidade síncrona mecânica [rad/s]
    ns_rpm = mp.n_sync

    # Tensão de fase
    Vph = mp.Vl / sqrt(3.0)

    # Reatâncias (conversão de Xls, Xlr a partir de Ω com base em wb)
    X1 = mp.Xls   # reatância de dispersão do estator já em [Ω]
    X2 = mp.Xlr   # reatância de dispersão do rotor
    Xm_val = mp.Xm

    Z1 = complex(mp.Rs, X1)
    Zm = complex(0.0, Xm_val)

    # Thevenin visto dos terminais do rotor
    Vth = Vph * (Zm / (Z1 + Zm))
    Zth = (Zm * Z1) / (Z1 + Zm)
    Rth, Xth = Zth.real, Zth.imag

    # Varredura de escorregamento (evita singularidade em s=0)
    s = np.linspace(1e-4, 1.0, n_pts)[::-1]

    R2 = Rr
    Te = (3.0 * abs(Vth)**2 * (R2 / s)) / (
        w_syn * ((Rth + R2 / s)**2 + (Xth + X2)**2)
    )
    n_rpm = (1.0 - s) * ns_rpm

    # Torque de partida (s = 1)
    Te_start = (3.0 * abs(Vth)**2 * R2) / (w_syn * ((Rth + R2)**2 + (Xth + X2)**2))

    # Escorregamento e torque máximo
    s_Tmax = float(np.clip(R2 / np.sqrt(Rth**2 + (Xth + X2)**2), 1e-4, 1.0))
    Te_max = (3.0 * abs(Vth)**2 * (R2 / s_Tmax)) / (
        w_syn * ((Rth + R2 / s_Tmax)**2 + (Xth + X2)**2)
    )
    n_Tmax = (1.0 - s_Tmax) * ns_rpm

    # Ponto de operação para a carga especificada
    idx_op  = int(np.argmin(np.abs(Te - T_load)))
    s_oper  = float(s[idx_op])
    n_oper  = float(n_rpm[idx_op])

    return {
        "s": s, "Te": Te, "n_rpm": n_rpm,
        "Te_start":   float(Te_start),
        "s_Tmax":     s_Tmax,
        "Te_max":     float(Te_max),
        "n_Tmax_rpm": float(n_Tmax),
        "s_oper":     s_oper,
        "n_oper_rpm": n_oper,
        "ns_rpm":     float(ns_rpm),
    } # type: ignore


def phasor_torque_speed(
    f_test: float,
    V_nom:  float,
    f_nom:  float,
    R1: float, R2: float, X1: float, X2: float, Xm_val: float,
    polos:  int,
    T_mec:  float,
    n_pts:  int = 200,
) -> dict:
    """
    Análise fasorial em regime permanente para uma dada frequência de rede.
    Utiliza controle V/f constante (Vph ∝ f) para manter fluxo constante.

    Usado pelo Experimento 9 (variação de frequência).

    Returns
    -------
    dict com arrays s_vals, T_vals, n_vals e grandezas escalares de regime.
    """
    V = V_nom * (f_test / f_nom)
    ns = 120.0 * f_test / polos

    s_vals = np.linspace(1e-3, 0.1, n_pts)
    T_vals = np.empty(n_pts)

    for k, s in enumerate(s_vals):
        ws = 2.0 * np.pi * f_test / (polos / 2.0)
        Z1_c = complex(R1, X1)
        Zm_c = complex(0.0, Xm_val)
        Z2_c = complex(R2 / s, X2)
        Z_th = Z1_c + (Zm_c * Z2_c) / (Zm_c + Z2_c)
        I2   = V / Z_th
        P_conv = 3.0 * abs(I2)**2 * (R2 / s) * (1.0 - s)
        T_vals[k] = P_conv / (ws * (1.0 - s))

    n_vals = ns * (1.0 - s_vals)

    # Ponto de regime (torque = T_mec)
    idx_r  = int(np.argmin(np.abs(T_vals - T_mec)))
    s_r    = float(s_vals[idx_r])
    n_r    = float(n_vals[idx_r])
    ws_r   = 2.0 * np.pi * f_test / (polos / 2.0)

    Z1_c   = complex(R1, X1)
    Zm_c   = complex(0.0, Xm_val)
    Z2_c   = complex(R2 / s_r, X2)
    Z_th_r = Z1_c + (Zm_c * Z2_c) / (Zm_c + Z2_c)
    I_r    = V / Z_th_r

    P_mec  = (2.0 * np.pi * n_r / 60.0) * T_mec
    P_in   = 3.0 * abs(I_r)**2 * R1 + P_mec / (1.0 - s_r)
    eta    = P_mec / P_in * 100.0

    return {
        "s_vals": s_vals, "T_vals": T_vals, "n_vals": n_vals,
        "ns_rpm": float(ns), "n_regime": n_r, "s_regime": s_r,
        "I_regime": abs(I_r), "P_mec": P_mec, "eficiencia": eta,
        "V": V,
    }


# =============================================================================
# SEÇÃO 9 — AUXILIARES DE PLOTAGEM
# =============================================================================

def _label(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True)


def plot_torque_speed(res: SimResults, title: str = "", t_events: list = None) -> None:
    """Gráfico padrão: torque eletromagnético e velocidade em função do tempo."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(title, fontsize=14)
    ax1.plot(res.t, res.Te, label="$T_e$")
    _label(ax1, "", "Torque (N·m)", "$T_e$")
    ax2.plot(res.t, res.n, label="n (RPM)")
    if t_events:
        for te in t_events:
            ax2.axvline(x=te, color="gray", linestyle="--", alpha=0.7)
    _label(ax2, "Tempo (s)", "Velocidade (RPM)", "Velocidade")
    plt.tight_layout()
    plt.show()


def plot_stator(res: SimResults, title: str = "") -> None:
    """Painel 2×2: tensões e correntes trifásicas + componentes dq do estator."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    ax = axes[0, 0]
    ax.plot(res.t, res.Va, label="$V_a$")
    ax.plot(res.t, res.Vb, label="$V_b$")
    ax.plot(res.t, res.Vc, label="$V_c$")
    _label(ax, "Tempo (s)", "Tensão (V)", "$V_a$ × $V_b$ × $V_c$")

    ax = axes[0, 1]
    ax.plot(res.t, res.ias, label="$i_{as}$")
    ax.plot(res.t, res.ibs, label="$i_{bs}$")
    ax.plot(res.t, res.ics, label="$i_{cs}$")
    _label(ax, "Tempo (s)", "Corrente (A)", "$i_{as}$ × $i_{bs}$ × $i_{cs}$")

    ax = axes[1, 0]
    ax.plot(res.t, res.Vds, label="$V_d$")
    ax.plot(res.t, res.Vqs, label="$V_q$")
    _label(ax, "Tempo (s)", "Tensão (V)", "$V_d$ × $V_q$")

    ax = axes[1, 1]
    ax.plot(res.t, res.ids, label="$i_{ds}$")
    ax.plot(res.t, res.iqs, label="$i_{qs}$")
    _label(ax, "Tempo (s)", "Corrente (A)", "$i_{ds}$ × $i_{qs}$")

    plt.tight_layout()
    plt.show()


def plot_rotor(res: SimResults, title: str = "") -> None:
    """Painel 2×1: correntes abc e dq do rotor."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(title, fontsize=14)

    ax1.plot(res.t, res.iar, label="$i_{ar}$")
    ax1.plot(res.t, res.ibr, label="$i_{br}$")
    ax1.plot(res.t, res.icr, label="$i_{cr}$")
    _label(ax1, "", "Corrente (A)", "$i_{ar}$ × $i_{br}$ × $i_{cr}$")

    ax2.plot(res.t, res.idr, label="$i_{dr}$")
    ax2.plot(res.t, res.iqr, label="$i_{qr}$")
    _label(ax2, "Tempo (s)", "Corrente (A)", "$i_{dr}$ × $i_{qr}$")

    plt.tight_layout()
    plt.show()


def plot_generator_transition(
    res: SimResults, Tl_ref: float, n_sync: float, t_2: float, title: str = ""
) -> None:
    """
    Gráficos específicos para operação em modo gerador:
      1. Torque com região de geração sombreada e linha de torque da turbina.
      2. Velocidade com linha síncrona destacada.
    """
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(res.t, res.Te, label="Torque Elétrico")
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax1.axhline(y=Tl_ref, color="r", linestyle="-.", label=f"Torque Turbina ({Tl_ref:.0f} N·m)")
    ax1.fill_between(res.t, res.Te, where=(res.Te < 0),
                     color="green", alpha=0.3, label="Região Gerador")
    ax1.set_title(title or "Transição Motor–Gerador", fontsize=14)
    ax1.legend(); ax1.grid(True)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(res.t, res.n, label="Velocidade do Rotor")
    ax2.axhline(y=n_sync, color="r", linestyle="--",
                label=f"Velocidade Síncrona ({n_sync:.0f} RPM)")
    ax2.axvline(x=t_2, color="g", linestyle=":", label="Aplicação do Torque")
    ax2.set_title("Superação da Velocidade Síncrona", fontsize=14)
    ax2.set_xlabel("Tempo (s)"); ax2.set_ylabel("Velocidade (RPM)")
    ax2.legend(); ax2.grid(True)
    plt.show()


def plot_comparison_references(results_dict: dict) -> None:
    """
    Experimento 12: sobreposição das grandezas físicas (Te, n) e
    dos componentes dq para os três referenciais de Park.
    """
    # Grandezas físicas invariantes
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig1.suptitle("Comparação de Torque e Velocidade nos 3 Referenciais", fontsize=16)
    for name, res in results_dict.items():
        ax1.plot(res["t"], res["Te"], label=name, alpha=0.85)
        ax2.plot(res["t"], res["n"],  label=name, alpha=0.85)
    _label(ax1, "", "Torque (N·m)", "Torque Eletromagnético")
    _label(ax2, "Tempo (s)", "Velocidade (RPM)", "Velocidade do Rotor")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Componentes dq por referencial
    ref_names = list(results_dict.keys())
    fig2, axes = plt.subplots(3, 2, figsize=(12, 15), sharex=True)
    fig2.suptitle("Componentes d-q por Referencial", fontsize=16)
    for i, name in enumerate(ref_names):
        res = results_dict[name]
        axes[i, 0].plot(res["t"], res["Vds"], label="$V_{ds}$")
        axes[i, 0].plot(res["t"], res["Vqs"], label="$V_{qs}$")
        axes[i, 0].set_title(f"{name}: Tensões d-q")
        axes[i, 0].set_ylabel("Tensão (V)"); axes[i, 0].grid(True); axes[i, 0].legend()

        axes[i, 1].plot(res["t"], res["ids"], label="$i_{ds}$")
        axes[i, 1].plot(res["t"], res["iqs"], label="$i_{qs}$")
        axes[i, 1].set_title(f"{name}: Correntes d-q")
        axes[i, 1].set_ylabel("Corrente (A)"); axes[i, 1].grid(True); axes[i, 1].legend()

    axes[2, 0].set_xlabel("Tempo (s)")
    axes[2, 1].set_xlabel("Tempo (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def print_max_values(label: str, res: SimResults) -> dict:
    """Imprime e retorna dicionário com os valores máximos de corrente e torque."""
    values = {
        "Corrente máx. fase A Estator (A)": float(np.max(np.abs(res.ias))),
        "Corrente máx. fase B Estator (A)": float(np.max(np.abs(res.ibs))),
        "Corrente máx. fase C Estator (A)": float(np.max(np.abs(res.ics))),
        "Corrente máx. fase A Rotor (A)":   float(np.max(np.abs(res.iar))),
        "Corrente máx. fase B Rotor (A)":   float(np.max(np.abs(res.ibr))),
        "Corrente máx. fase C Rotor (A)":   float(np.max(np.abs(res.icr))),
        "Torque máx. no eixo (N·m)":        float(np.max(res.Te)),
    }
    print(f"\n{'='*60}")
    print(f"Valores máximos — {label}")
    print(f"{'='*60}")
    for k, v in values.items():
        print(f"  {k}: {v:.3f}")
    return values


# =============================================================================
# SEÇÃO 10 — EXPERIMENTOS (1 a 12)
# =============================================================================

# ---------------------------------------------------------------------------
# Experimento 1 — Partida Direta (DOL — Direct On Line)
# ---------------------------------------------------------------------------
def experiment_dol(
    mp:       MachineParams = DEFAULT_MACHINE,
    tmax:     float = 2.0,
    h:        float = 0.001,
    t_carga:  float = 0.1,
    Tl_final: float = 80.0,
    ref:      int   = 1,
) -> SimResults:
    """
    Partida direta: tensão nominal aplicada desde o instante zero.
    O torque de carga é aplicado em t_carga.

    Esperado: picos elevados de corrente e torque no transitório inicial.
    """
    print("\n" + "="*60)
    print("Experimento 1 — Partida Direta (DOL)")
    print("="*60)
    print(f"  Vl={mp.Vl:.1f} V | Tl={Tl_final:.1f} N·m aplicado em t={t_carga}s")

    res = run_simulation(
        mp=mp, tmax=tmax, h=h,
        voltage_fn=lambda t: mp.Vl,
        torque_fn =lambda t: torque_step(t, 0.0, Tl_final, t_carga),
        ref_code  =ref,
    )

    print("  Gerando gráficos do Experimento 1...")
    plot_torque_speed(res, "Exp. 1 — Partida Direta (DOL)")
    plot_stator(res, "Exp. 1 — Variáveis do Estator (DOL)")
    plot_rotor(res, "Exp. 1 — Correntes do Rotor (DOL)")
    print("  Experimento 1 concluído.")
    return res


# ---------------------------------------------------------------------------
# Experimento 2 — Partida Y-Δ (Estrela-Triângulo)
# ---------------------------------------------------------------------------
def experiment_yd(
    mp:      MachineParams = DEFAULT_MACHINE,
    tmax:    float = 2.0,
    h:       float = 0.001,
    t_2:     float = 0.5,
    t_carga: float = 0.1,
    ref:     int   = 1,
) -> SimResults:
    """
    Partida Y-Δ: tensão reduzida a 1/√3 durante t < t_2, depois nominal.
    Reduz corrente e torque de partida (aprox. 1/3 do valor DOL).

    Esperado: picos de corrente menores que DOL até a comutação em t_2.
    """
    print("\n" + "="*60)
    print("Experimento 2 — Partida Y-Δ")
    print("="*60)

    Vl_Y = mp.Vl / np.sqrt(3.0)
    print(f"  Tensão em Y: {Vl_Y:.2f} V  →  Nominal: {mp.Vl:.1f} V  (comutação em t={t_2} s)")

    res = run_simulation(
        mp=mp, tmax=tmax, h=h,
        voltage_fn=lambda t: voltage_reduced_start(t, mp.Vl, Vl_Y, t_2),
        torque_fn =lambda t: torque_step(t, 0.0, 80.0, t_carga),
        ref_code  =ref,
    )

    print("  Gerando gráficos do Experimento 2...")
    plot_torque_speed(res, "Exp. 2 — Partida Y-Δ", t_events=[t_2])
    plot_stator(res, "Exp. 2 — Variáveis do Estator (Y-Δ)")
    plot_rotor(res, "Exp. 2 — Correntes do Rotor (Y-Δ)")
    print("  Experimento 2 concluído.")
    return res


# ---------------------------------------------------------------------------
# Experimento 3 — Partida com Autotransformador (Compensadora)
# ---------------------------------------------------------------------------
def experiment_compensadora(
    mp:             MachineParams = DEFAULT_MACHINE,
    tmax:           float = 2.0,
    h:              float = 0.001,
    t_2:            float = 0.5,
    t_carga:        float = 0.1,
    voltage_ratio:  float = 0.5,
    ref:            int   = 1,
) -> SimResults:
    """
    Partida compensadora: tensão inicial = voltage_ratio × Vl_nominal.
    Simula autotransformador com tap fixo durante a partida.

    Esperado: perfil de corrente entre DOL e Y-Δ, dependente do tap.
    """
    print("\n" + "="*60)
    print(f"Experimento 3 — Partida Compensadora (tap = {voltage_ratio*100:.0f}%)")
    print("="*60)

    Vl_red = mp.Vl * voltage_ratio
    print(f"  Tensão reduzida: {Vl_red:.1f} V  →  Nominal: {mp.Vl:.1f} V  (comutação em t={t_2} s)")

    res = run_simulation(
        mp=mp, tmax=tmax, h=h,
        voltage_fn=lambda t: voltage_reduced_start(t, mp.Vl, Vl_red, t_2),
        torque_fn =lambda t: torque_step(t, 0.0, 80.0, t_carga),
        ref_code  =ref,
    )

    print("  Gerando gráficos do Experimento 3...")
    plot_torque_speed(res, "Exp. 3 — Partida Compensadora", t_events=[t_2])
    plot_stator(res, "Exp. 3 — Variáveis do Estator (Compensadora)")
    plot_rotor(res, "Exp. 3 — Correntes do Rotor (Compensadora)")
    print("  Experimento 3 concluído.")
    return res


# ---------------------------------------------------------------------------
# Experimento Soft-Starter — Partida com Rampa de Tensão
# ---------------------------------------------------------------------------
def experiment_soft_starter(
    mp:             MachineParams = DEFAULT_MACHINE,
    tmax:           float = 10.0,
    h:              float = 0.001,
    t_2:            float = 0.9,
    t_pico:         float = 5.0,
    t_carga:        float = 0.1,
    voltage_ratio:  float = 0.5,
    Tl_final:       float = 80.0,
    ref:            int   = 1,
) -> SimResults:
    """
    Partida soft-starter: tensão cresce linearmente de voltage_ratio × Vl
    até Vl_nominal entre t_2 e t_pico, permanecendo nominal após t_pico.

    Esperado: correntes de partida mais suaves que Y-Δ ou compensadora;
    transitório prolongado até regime.
    """
    print("\n" + "="*60)
    print(f"Experimento Soft-Starter (rampa de tensão: t={t_2}s → t={t_pico}s)")
    print("="*60)

    Vl_init = mp.Vl * voltage_ratio
    print(f"  Tensão inicial: {Vl_init:.1f} V ({voltage_ratio*100:.0f}%) → {mp.Vl:.1f} V nominal")
    print(f"  Carga de {Tl_final:.1f} N·m aplicada em t={t_carga}s")

    res = run_simulation(
        mp=mp, tmax=tmax, h=h,
        voltage_fn=lambda t: voltage_soft_starter(t, mp.Vl, Vl_init, t_2, t_pico),
        torque_fn =lambda t: torque_step(t, 0.0, Tl_final, t_carga),
        ref_code  =ref,
    )

    ss = res.steady_state()
    print(f"  Regime permanente: n = {ss['n_ss']:.1f} RPM | Te = {ss['Te_ss']:.2f} N·m")
    print("  Gerando gráficos do Soft-Starter...")
    plot_torque_speed(res, "Soft-Starter — Torque e Velocidade", t_events=[t_2, t_pico])
    plot_stator(res, "Soft-Starter — Variáveis do Estator")
    plot_rotor(res, "Soft-Starter — Correntes do Rotor")
    print("  Experimento Soft-Starter concluído.")
    return res


# ---------------------------------------------------------------------------
# Experimento 4 — Partida em Vazio e Aplicação de Carga Nominal
# ---------------------------------------------------------------------------
def experiment_carga_nominal(
    mp:       MachineParams = DEFAULT_MACHINE,
    tmax:     float = 2.0,
    h:        float = 0.001,
    t_carga:  float = 1.0,
    Tl_final: float = 80.0,
    ref:      int   = 1,
) -> SimResults:
    """
    Motor parte em vazio e recebe carga nominal (80 N·m) em t_carga.

    Esperado: queda de velocidade e aumento de corrente na aplicação de carga.
    """
    print("\n" + "="*60)
    print("Experimento 4 — Partida em Vazio + Carga Nominal")
    print("="*60)
    print(f"  Carga de {Tl_final:.1f} N·m aplicada em t={t_carga}s")

    res = run_simulation(
        mp=mp, tmax=tmax, h=h,
        voltage_fn=lambda t: mp.Vl,
        torque_fn =lambda t: torque_step(t, 0.0, Tl_final, t_carga),
        ref_code  =ref,
    )

    ss = res.steady_state()
    print(f"  Regime permanente: n = {ss['n_ss']:.1f} RPM | Te = {ss['Te_ss']:.2f} N·m")
    print("  Gerando gráficos do Experimento 4...")
    plot_torque_speed(res, "Exp. 4 — Carga Nominal", t_events=[t_carga])
    plot_stator(res, "Exp. 4 — Variáveis do Estator")
    plot_rotor(res, "Exp. 4 — Correntes do Rotor")
    print_max_values("Partida em Vazio + Carga Nominal", res)
    print("  Experimento 4 concluído.")
    return res


# ---------------------------------------------------------------------------
# Experimento 5 — Variação de Carga (50% do Nominal)
# ---------------------------------------------------------------------------
def experiment_carga_50pct(
    mp:      MachineParams = DEFAULT_MACHINE,
    tmax:    float = 2.0,
    h:       float = 0.001,
    t_carga: float = 1.0,
    ref:     int   = 1,
) -> SimResults:
    """
    Motor parte em vazio e recebe 50% da carga nominal (40 N·m) em t_carga.

    Esperado: velocidade de regime maior e corrente menor que na carga nominal.
    """
    print("\n" + "="*60)
    print("Experimento 5 — Carga 50% do Nominal")
    print("="*60)

    Tl_50 = 80.0 * 0.5  # 40 N·m
    print(f"  Carga de {Tl_50:.1f} N·m (50% nominal) aplicada em t={t_carga}s")
    res = run_simulation(
        mp=mp, tmax=tmax, h=h,
        voltage_fn=lambda t: mp.Vl,
        torque_fn =lambda t: torque_step(t, 0.0, Tl_50, t_carga),
        ref_code  =ref,
    )

    ss = res.steady_state()
    print(f"  Regime permanente: n = {ss['n_ss']:.1f} RPM | Te = {ss['Te_ss']:.2f} N·m")
    print("  Gerando gráficos do Experimento 5...")
    plot_torque_speed(res, "Exp. 5 — Carga 50%", t_events=[t_carga])
    plot_stator(res, "Exp. 5 — Variáveis do Estator")
    print("  Experimento 5 concluído.")
    return res


# ---------------------------------------------------------------------------
# Experimento 6 — Sobrecarga Temporária (120% do Nominal)
# ---------------------------------------------------------------------------
def experiment_sobrecarga(
    mp:      MachineParams = DEFAULT_MACHINE,
    tmax:    float = 2.0,
    h:       float = 0.001,
    t_carga: float = 1.0,
    ref:     int   = 1,
) -> SimResults:
    """
    Motor parte em vazio e recebe 120% da carga nominal (96 N·m).
    Testa a capacidade de resposta a sobrecargas.

    Esperado: queda de velocidade e pico de corrente maiores que na carga nominal.
    """
    print("\n" + "="*60)
    print("Experimento 6 — Sobrecarga 120% do Nominal")
    print("="*60)

    Tl_120 = 80.0 * 1.2  # 96 N·m
    print(f"  Carga de {Tl_120:.1f} N·m (120% nominal) aplicada em t={t_carga}s")
    res = run_simulation(
        mp=mp, tmax=tmax, h=h,
        voltage_fn=lambda t: mp.Vl,
        torque_fn =lambda t: torque_step(t, 0.0, Tl_120, t_carga),
        ref_code  =ref,
    )

    ss = res.steady_state()
    print(f"  Regime permanente: n = {ss['n_ss']:.1f} RPM | Te = {ss['Te_ss']:.2f} N·m")
    print("  Gerando gráficos do Experimento 6...")
    plot_torque_speed(res, "Exp. 6 — Sobrecarga 120%", t_events=[t_carga])
    plot_stator(res, "Exp. 6 — Variáveis do Estator")
    print("  Experimento 6 concluído.")
    return res


# ---------------------------------------------------------------------------
# Experimento 7 — Impacto de Rr no Torque de Partida e Escorregamento
# ---------------------------------------------------------------------------
def experiment_rr_sweep(
    mp:      MachineParams  = DEFAULT_MACHINE,
    Rr_list: list[float]    = None,
    T_load:  float          = 80.0,
) -> None:
    """
    Análise em regime permanente para diferentes valores de Rr.
    Utiliza o modelo de circuito equivalente de Thevenin (não integração).

    Esperado: maior Rr → maior torque de partida, maior escorregamento em regime.
    """
    if Rr_list is None:
        Rr_list = [0.30, 1.00, 1.20]

    print("\n" + "="*60)
    print("Experimento 7 — Varredura de Rr")
    print("="*60)
    print(f"  Calculando circuito equivalente de Thevenin para {len(Rr_list)} valores de Rr...")

    results = {}
    for Rr in Rr_list:
        print(f"  Calculando Rr = {Rr:.2f} Ω ...", end=" ", flush=True)
        results[Rr] = thevenin_torque_speed(Rr, mp, T_load)
        print("concluído.")

    # Tabela de resultados
    print(f"\n{'Rr (Ω)':>8} {'T_part.(N·m)':>13} {'s_Tmax':>8} {'Tmax(N·m)':>10} "
          f"{'n_Tmax(rpm)':>12} {'s_oper':>8} {'n_oper(rpm)':>12}")
    print("-" * 80)
    for Rr, r in sorted(results.items()):
        print(f"{Rr:>8.2f} {r['Te_start']:>13.2f} {r['s_Tmax']:>8.4f} "
              f"{r['Te_max']:>10.2f} {r['n_Tmax_rpm']:>12.1f} "
              f"{r['s_oper']:>8.4f} {r['n_oper_rpm']:>12.1f}")

    # Gráfico curvas Torque × Velocidade
    plt.figure(figsize=(10, 5))
    for Rr, r in sorted(results.items()):
        plt.plot(r["n_rpm"], r["Te"], label=f"Rr = {Rr:.2f} Ω")
    plt.axhline(T_load, linestyle="--", color="k", label=f"Carga = {T_load:.0f} N·m")
    plt.xlabel("Velocidade (rpm)"); plt.ylabel("Torque (N·m)")
    plt.title("Exp. 7 — Torque × Velocidade para Diferentes Rr")
    plt.legend(); plt.grid(True); plt.show()

    # Torque de partida e escorregamento em função de Rr
    Rr_sorted = sorted(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(Rr_sorted, [results[Rr]["Te_start"] for Rr in Rr_sorted], marker="o")
    ax1.set_xlabel("Rr (Ω)"); ax1.set_ylabel("Torque de partida (N·m)")
    ax1.set_title("Torque de Partida vs Rr"); ax1.grid(True)

    ax2.plot(Rr_sorted, [results[Rr]["s_oper"] for Rr in Rr_sorted], marker="o")
    ax2.set_xlabel("Rr (Ω)"); ax2.set_ylabel("Escorregamento em regime")
    ax2.set_title(f"Escorregamento em Regime (T = {T_load:.0f} N·m)"); ax2.grid(True)
    plt.tight_layout(); plt.show()
    print("  Experimento 7 concluído.")


# ---------------------------------------------------------------------------
# Experimento 8 — Impacto do Momento de Inércia (J) no Tempo de Aceleração
# ---------------------------------------------------------------------------
def experiment_inertia_J(
    mp:       MachineParams = DEFAULT_MACHINE,
    J_values: list[float]   = None,
    tmax:     float         = 5.0,
    h:        float         = 0.0005,
    Tl_final: float         = 80.0,
    t_ramp:   float         = 2.0,
) -> None:
    """
    Compara o tempo de aceleração para diferentes valores de J.
    O torque de carga cresce linearmente até t_ramp.

    Esperado: maior J → aceleração mais lenta, comportamento mais amortecido.
    """
    if J_values is None:
        J_values = [0.04, 0.089, 0.15]

    print("\n" + "="*60)
    print("Experimento 8 — Impacto de J no Tempo de Aceleração")
    print("="*60)
    print(f"  Simulando {len(J_values)} valores de inércia: {J_values}")

    n_target = 0.95 * mp.n_sync
    plt.figure(figsize=(12, 5))

    for J in J_values:
        print(f"  Simulando J = {J:.3f} kg·m² ...")
        mp_J = MachineParams(
            Vl=mp.Vl, f=mp.f, Rs=mp.Rs, Rr=mp.Rr, Xm=mp.Xm,
            Xls=mp.Xls, Xlr=mp.Xlr, p=mp.p, J=J, B=mp.B
        )
        res = run_simulation(
            mp=mp_J, tmax=tmax, h=h,
            voltage_fn=lambda t: mp.Vl,
            torque_fn =lambda t: torque_ramp(t, 0.0, Tl_final, t_ramp),
            ref_code  =1,
        )

        # Tempo para atingir 95% da velocidade síncrona
        idxs = np.where(res.n >= n_target)[0]
        t95  = float(res.t[idxs[0]]) if len(idxs) > 0 else float("nan")
        print(f"  J = {J:.3f} kg·m²  →  t(95% ns) = {t95:.3f} s")

        plt.plot(res.t, res.n, label=f"J = {J:.3f} kg·m²")

    print("  Gerando gráfico do Experimento 8...")
    plt.axhline(n_target, color="g", linestyle=":", label="95% da Vel. Síncrona")
    plt.xlabel("Tempo (s)"); plt.ylabel("Velocidade (RPM)")
    plt.title("Exp. 8 — Velocidade vs Tempo para Diferentes J")
    plt.legend(); plt.grid(True); plt.show()
    print("  Experimento 8 concluído.")


# ---------------------------------------------------------------------------
# Experimento 9 — Variação da Frequência da Rede
# ---------------------------------------------------------------------------
def experiment_frequency(
    mp:        MachineParams = DEFAULT_MACHINE,
    freq_list: list[float]   = None,
    T_mec:     float         = 80.0,
) -> None:
    """
    Análise da influência da frequência de alimentação com V/f constante.
    Usa modelo fasorial em regime (não integração).

    Esperado: redução de f → menor velocidade síncrona, maior escorregamento.
    """
    if freq_list is None:
        freq_list = [50.0, 40.0]

    print("\n" + "="*60)
    print("Experimento 9 — Variação da Frequência da Rede")
    print("="*60)
    print(f"  Análise fasorial (regime permanente) para: {freq_list} Hz")

    # Parâmetros do circuito equivalente em valores ômicos para 60 Hz
    R1, R2  = 0.641, 0.332
    X1_60, X2_60, Xm_60 = 1.106, 0.464, 26.3
    f_nom   = 60.0

    plt.figure(figsize=(10, 5))
    for f_test in freq_list:
        print(f"  Calculando f = {f_test:.0f} Hz ...", end=" ", flush=True)
        res = phasor_torque_speed(
            f_test=f_test, V_nom=220.0, f_nom=f_nom,
            R1=R1, R2=R2, X1=X1_60, X2=X2_60, Xm_val=Xm_60,
            polos=mp.p, T_mec=T_mec,
        )
        print("concluído.")
        print(f"    ns = {res['ns_rpm']:.1f} rpm | n_regime = {res['n_regime']:.1f} rpm")
        print(f"    s = {res['s_regime']*100:.2f}% | I = {res['I_regime']:.2f} A")
        print(f"    P_mec = {res['P_mec']:.1f} W | η = {res['eficiencia']:.2f}%")
        plt.plot(res["n_vals"], res["T_vals"], label=f"{f_test:.0f} Hz")

    print("  Gerando gráfico do Experimento 9...")
    plt.axhline(T_mec, color="k", linestyle="--", label=f"Carga = {T_mec:.0f} N·m")
    plt.xlabel("Velocidade (rpm)"); plt.ylabel("Torque (N·m)")
    plt.title("Exp. 9 — Torque × Velocidade para Diferentes Frequências")
    plt.legend(); plt.grid(True); plt.show()
    print("  Experimento 9 concluído.")


# ---------------------------------------------------------------------------
# Experimento 10 — Operação como Gerador (Carga Negativa)
# ---------------------------------------------------------------------------
def experiment_generator(
    mp:      MachineParams = DEFAULT_MACHINE,
    tmax:    float = 2.0,
    h:       float = 0.0001,
    t_2:     float = 1.0,
    Tl_mec:  float = 80.0,
    ref:     int   = 1,
) -> SimResults:
    """
    Máquina conectada ao barramento infinito com torque negativo (gerador).
    O torque da turbina (-Tl_mec) é aplicado em t_2.

    Esperado: rotor supera velocidade síncrona e Te torna-se negativo.
    """
    print("\n" + "="*60)
    print("Experimento 10 — Gerador Conectado ao Barramento Infinito")
    print("="*60)
    print(f"  Torque da turbina: {Tl_mec:.1f} N·m (negativo) desde t=0s")

    # Sinal negativo: torque impulsionado externamente (turbina)
    Tl_neg = -Tl_mec
    res = run_simulation(
        mp=mp, tmax=tmax, h=h,
        voltage_fn=lambda t: mp.Vl,
        torque_fn =lambda t: torque_step(t, Tl_neg, Tl_neg, t_2),
        ref_code  =ref,
    )

    print("  Gerando gráficos do Experimento 10...")
    plot_generator_transition(res, Tl_neg, mp.n_sync, t_2,
                              "Exp. 10 — Transição Motor–Gerador")
    plot_stator(res, "Exp. 10 — Variáveis do Estator (Gerador)")
    plot_rotor(res, "Exp. 10 — Correntes do Rotor (Gerador)")
    print("  Experimento 10 concluído.")
    return res


# ---------------------------------------------------------------------------
# Experimento 11 — Variação da Carga no Modo Gerador
# ---------------------------------------------------------------------------
def experiment_generator_load_variation(
    mp:          MachineParams = DEFAULT_MACHINE,
    tmax:        float = 2.0,
    h:           float = 0.0001,
    t_2:         float = 1.0,
    Tl_initial:  float = 40.0,
    Tl_final:    float = 120.0,
    ref:         int   = 1,
) -> SimResults:
    """
    Gerador com aumento de torque mecânico: de -Tl_initial para -Tl_final em t_2.
    Analisa resposta dinâmica a variações de potência gerada.

    Esperado: aumento de corrente e variação de velocidade com o incremento de carga.
    """
    print("\n" + "="*60)
    print("Experimento 11 — Variação de Carga no Modo Gerador")
    print("="*60)
    print(f"  Torque: -{Tl_initial:.1f} N·m → -{Tl_final:.1f} N·m em t={t_2}s")

    res = run_simulation(
        mp=mp, tmax=tmax, h=h,
        voltage_fn=lambda t: mp.Vl,
        torque_fn =lambda t: torque_step(t, -Tl_initial, -Tl_final, t_2),
        ref_code  =ref,
    )

    # Análise de regime antes e depois da variação de carga
    idx_antes  = (res.t >= t_2 - 0.05) & (res.t <= t_2)
    idx_depois = (res.t >= t_2 + 0.5)  & (res.t <= tmax)

    n_antes  = float(np.mean(res.n[idx_antes]))
    n_depois = float(np.mean(res.n[idx_depois]))
    P_gerada = 1.5 * (res.Vqs * res.iqs + res.Vds * res.ids)
    P_antes  = float(np.mean(P_gerada[idx_antes]))
    P_depois = float(np.mean(P_gerada[idx_depois]))
    I_antes  = float(np.sqrt(np.mean(
        res.ias[idx_antes]**2 + res.ibs[idx_antes]**2 + res.ics[idx_antes]**2)))
    I_depois = float(np.sqrt(np.mean(
        res.ias[idx_depois]**2 + res.ibs[idx_depois]**2 + res.ics[idx_depois]**2)))

    print(f"  Velocidade:  {n_antes:.1f} → {n_depois:.1f} RPM  (Δ = {n_depois-n_antes:.1f})")
    print(f"  Potência:    {P_antes:.1f} → {P_depois:.1f} W  (Δ = {P_depois-P_antes:.1f})")
    print(f"  Corrente ef: {I_antes:.2f} → {I_depois:.2f} A")

    print("  Gerando gráficos do Experimento 11...")
    plot_generator_transition(res, -Tl_final, mp.n_sync, t_2,
                              "Exp. 11 — Variação de Carga (Gerador)")
    plot_stator(res, "Exp. 11 — Variáveis do Estator")

    # Potência ativa gerada
    plt.figure(figsize=(10, 4))
    plt.plot(res.t, P_gerada, "m-", linewidth=2)
    plt.axvline(x=t_2, color="k", linestyle="--")
    plt.title("Exp. 11 — Potência Ativa Gerada")
    plt.xlabel("Tempo (s)"); plt.ylabel("Potência (W)")
    plt.grid(True); plt.show()
    print("  Experimento 11 concluído.")

    return res


# ---------------------------------------------------------------------------
# Experimento 12 — Comparação dos Referenciais da Transformada de Park
# ---------------------------------------------------------------------------
def experiment_reference_comparison(
    mp:      MachineParams = DEFAULT_MACHINE,
    tmax:    float = 2.5,
    h:       float = 0.0001,
    t_carga: float = 1.0,
    Tl_final: float = 80.0,
) -> None:
    """
    Executa três simulações idênticas usando os referenciais síncrono, rotórico
    e estacionário. Compara grandezas físicas (invariantes) e componentes dq
    (dependentes do referencial).

    Nota: usa a convenção de Park direta (fator 2/3) preservando a consistência
    numérica com a implementação original deste experimento.

    Esperado:
      - Te e n idênticos nos três referenciais (invariância física).
      - Vds/Vqs/ids/iqs com frequência de oscilação distinta em cada referencial.
    """
    print("\n" + "="*60)
    print("Experimento 12 — Comparação dos Referenciais de Park")
    print("="*60)

    ref_names = {1: "Síncrono", 2: "Rotórico", 3: "Estacionário"}
    results_dict = {}

    for ref_code, name in ref_names.items():
        print(f"  Simulando referencial: {name} ...", end=" ", flush=True)
        res = run_simulation(
            mp=mp, tmax=tmax, h=h,
            voltage_fn=lambda t: mp.Vl,
            torque_fn =lambda t: torque_step(t, 0.0, Tl_final, t_carga),
            ref_code  =ref_code,
            use_direct_park=True,
        )
        results_dict[name] = {
            "t": res.t, "Te": res.Te, "n": res.n,
            "Vds": res.Vds, "Vqs": res.Vqs,
            "ids": res.ids, "iqs": res.iqs,
        }
        print("concluído.")

    print("  Gerando gráficos comparativos do Experimento 12...")
    plot_comparison_references(results_dict)
    print("  Experimento 12 concluído.")


# =============================================================================
# SEÇÃO 11 — PONTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EMS_BASE — Simulador de Máquina de Indução Trifásica")
    print("Modelo 0dq de Krause | Integração RK4 (scipy.odeint)")
    print("=" * 60)

    mp = DEFAULT_MACHINE

    # --- Experimentos de Partida ---
    res_dol  = experiment_dol(mp)
    res_yd   = experiment_yd(mp)
    res_comp = experiment_compensadora(mp)
    res_soft = experiment_soft_starter(mp)

    # Comparativo de valores máximos: DOL × Y-Δ × Compensadora × Soft-Starter
    v_dol  = print_max_values("DOL",          res_dol)
    v_yd   = print_max_values("Y-Δ",          res_yd)
    v_comp = print_max_values("Compensadora", res_comp)
    v_soft = print_max_values("Soft-Starter", res_soft)

    print("\n--- Razões em relação à DOL ---")
    for k in v_dol:
        if v_dol[k] > 0:
            print(f"  Y-Δ  / DOL  [{k[:30]}]: {v_yd[k]/v_dol[k]*100:.1f}%")
            print(f"  Comp / DOL  [{k[:30]}]: {v_comp[k]/v_dol[k]*100:.1f}%")
            print(f"  Soft / DOL  [{k[:30]}]: {v_soft[k]/v_dol[k]*100:.1f}%")

    # --- Experimentos de Carga ---
    experiment_carga_nominal(mp)
    experiment_carga_50pct(mp)
    experiment_sobrecarga(mp)

    # --- Experimentos de Parâmetros da Máquina ---
    experiment_rr_sweep(mp)
    experiment_inertia_J(mp)
    experiment_frequency(mp)

    # --- Experimentos de Regime de Gerador ---
    experiment_generator(mp)
    experiment_generator_load_variation(mp)

    # --- Experimento de Referencial ---
    experiment_reference_comparison(mp)
