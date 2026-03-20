"""
EMS_v3.py — Simulador de Maquinas de Inducao Trifasica
Streamlit Cloud compatible | No emojis | 4-step flow | Learning tab
"""

import os, base64, copy
from typing import Any, Dict, List, Tuple
import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from io import BytesIO
from math import pi, sqrt

# ---------------------------------------------------------------------------
# Matplotlib global config
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

st.set_page_config(
    page_title="EMS — Simulador de Maquinas",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 16px !important; }
h1 { font-size: 2rem !important; font-weight: 700; }
h2 { font-size: 1.5rem !important; font-weight: 600; }
h3 { font-size: 1.25rem !important; font-weight: 600; }

.machine-card {
    border: 1.5px solid #dde3ec;
    border-radius: 10px;
    padding: 20px 16px 16px 16px;
    text-align: center;
    background: #f8fafd;
    min-height: 200px;
}
.machine-card.disabled {
    background: #f0f0f0;
    color: #aaa;
}
.machine-card .card-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 4px;
}
.machine-card .card-desc {
    font-size: 0.92rem;
    color: #555;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 10px;
    margin: 12px 0;
}
.kpi-cell {
    background: #eef4ff;
    border-radius: 8px;
    padding: 12px 10px;
    text-align: center;
}
.kpi-cell .kpi-label {
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 4px;
}
.kpi-cell .kpi-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a3a6e;
}

.param-card {
    border-left: 4px solid #4a90d9;
    background: #f0f5ff;
    border-radius: 0 8px 8px 0;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.param-card .pc-title { font-weight: 700; font-size: 1rem; margin-bottom: 2px; }
.param-card .pc-role  { font-size: 0.9rem; color: #333; }
.param-card .pc-up    { color: #1a7a3a; font-size: 0.88rem; }
.param-card .pc-down  { color: #7a1a1a; font-size: 0.88rem; }
.param-card .pc-issue { color: #7a5a00; font-size: 0.88rem; font-style: italic; }

.alert-error {
    background: #fdecea;
    border-left: 5px solid #c0392b;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.95rem;
    color: #7b1c13;
}
.alert-warning {
    background: #fef9e7;
    border-left: 5px solid #d4ac0d;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.95rem;
    color: #7d6608;
}

.step-badge {
    display: inline-block;
    background: #1a3a6e;
    color: white;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    line-height: 28px;
    text-align: center;
    font-weight: 700;
    margin-right: 8px;
    font-size: 0.95rem;
}

.solver-banner-ok {
    background: linear-gradient(90deg,#1a7a3a,#239a4e);
    color: white;
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 10px;
}
.solver-banner-wait {
    background: #e8ecf5;
    color: #555;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.95rem;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "page": "home",
    "machine": None,
    "params_confirmed": False,
    "machine_params": None,
    "disturbances": [],
    "selected_vars": {"Te", "n", "ias"},
    "sim_results": None,
    "stored_curves": [],
    "plot_dark_bg": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COLORS = [
    "#4a90d9", "#e74c3c", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    "#c0392b", "#2980b9",
]

DISTURB_TYPES = {
    "Degrau de Carga": {
        "params": ["t_inicio", "Tl_novo"],
        "labels": ["Inicio (s)", "Torque de carga (N.m)"],
        "defaults": [0.5, 5.0],
        "needs_end": False,
        "desc": "Aplica um degrau no torque de carga no instante t_inicio.",
    },
    "Remocao de Carga": {
        "params": ["t_inicio"],
        "labels": ["Inicio (s)"],
        "defaults": [1.0],
        "needs_end": False,
        "desc": "Remove o torque de carga (Tl = 0) no instante t_inicio.",
    },
    "Sobrecarga Temporaria": {
        "params": ["t_inicio", "t_fim", "Tl_sobre"],
        "labels": ["Inicio (s)", "Fim (s)", "Torque sobrecarregado (N.m)"],
        "defaults": [0.5, 1.5, 15.0],
        "needs_end": True,
        "desc": "Aplica sobrecarga entre t_inicio e t_fim, retornando ao Tl original.",
    },
    "Queda de Tensao": {
        "params": ["t_inicio", "fator"],
        "labels": ["Inicio (s)", "Fator (0-1, ex: 0.8 = 80%)"],
        "defaults": [0.5, 0.8],
        "needs_end": False,
        "desc": "Reduz a tensao de alimentacao pelo fator especificado.",
    },
    "Aumento de Tensao": {
        "params": ["t_inicio", "fator"],
        "labels": ["Inicio (s)", "Fator (>1, ex: 1.1 = 110%)"],
        "defaults": [0.5, 1.1],
        "needs_end": False,
        "desc": "Aumenta a tensao de alimentacao pelo fator especificado.",
    },
    "Rampa de Tensao": {
        "params": ["t_inicio", "t_fim", "Vl_final"],
        "labels": ["Inicio (s)", "Fim da rampa (s)", "Tensao final (V)"],
        "defaults": [0.0, 1.0, 220.0],
        "needs_end": True,
        "desc": "Eleva a tensao linearmente de 0 ate Vl_final (modelo soft-starter).",
    },
}

VAR_OPTIONS = {
    "Te":  "Torque Eletromagnetico (N.m)",
    "n":   "Velocidade (RPM)",
    "ias": "Corrente Fase A - Estator (A)",
    "ibs": "Corrente Fase B - Estator (A)",
    "ics": "Corrente Fase C - Estator (A)",
    "iar": "Corrente Fase A - Rotor (A)",
    "ibr": "Corrente Fase B - Rotor (A)",
    "icr": "Corrente Fase C - Rotor (A)",
}

LEARNING_DATA = [
    {
        "param": "Tensao de Alimentacao (Vl)",
        "role": "Define o campo magnetico girante e, por consequencia, o torque disponivel. A tensao nominal e a referencia de operacao.",
        "up":   "Maior fluxo, maior torque de pico, maior corrente de partida, maior aquecimento.",
        "down": "Menor torque disponivel, risco de nao partida com carga elevada, menor corrente.",
        "issues": "Operar fora de +-10% da nominal degrada isolamento e reduz vida util.",
    },
    {
        "param": "Frequencia (f)",
        "role": "Determina a velocidade sincrona e, portanto, a velocidade de operacao em vazio.",
        "up":   "Velocidade sincrona maior, menor escorregamento relativo, menor corrente magnetizante.",
        "down": "Velocidade sincrona menor, maior corrente magnetizante, possivel saturacao do nucleo.",
        "issues": "Variacao de frequencia altera diretamente a velocidade — usado em inversores de frequencia.",
    },
    {
        "param": "Resistencia do Estator (Rs)",
        "role": "Causa perda Joule no estator. Afeta o torque maximo e a corrente nominal.",
        "up":   "Maior queda de tensao interna, menor torque maximo, maior aquecimento do estator.",
        "down": "Menor perda Joule, maior torque disponivel, maquina mais eficiente.",
        "issues": "Rs elevado indica enrolamento envelhecido ou danificado.",
    },
    {
        "param": "Resistencia do Rotor (Rr)",
        "role": "Determina o escorregamento de maximo torque. Rr alto desloca o torque maximo para baixa velocidade.",
        "up":   "Torque de partida maior, escorregamento maior, menor eficiencia em regime.",
        "down": "Menor escorregamento em regime, maior eficiencia, partida mais suave.",
        "issues": "Rr muito alto indica barra de rotor rompida — analise espectral de corrente recomendada.",
    },
    {
        "param": "Reatancia Magnetizante (Xm)",
        "role": "Representa o acoplamento entre estator e rotor. Xm alto significa nucleo bem aproveitado.",
        "up":   "Melhor fator de potencia, menor corrente magnetizante, maior torque maximo.",
        "down": "Maior corrente magnetizante, menor fator de potencia, maquina subaproveitada.",
        "issues": "Xm baixo pode indicar nucleo com saturacao precoce ou entreferro excessivo.",
    },
    {
        "param": "Reatancias de Dispersao (Xls / Xlr)",
        "role": "Representam fluxo que nao cruza o entreferro. Limitam a corrente de partida e o torque maximo.",
        "up":   "Menor corrente de partida, menor torque maximo, transitorio mais lento.",
        "down": "Maior torque maximo, corrente de partida elevada, maior estresse mecanico.",
        "issues": "Dispersao alta e comum em motores de rotor bobinado ou com ranhuras profundas.",
    },
    {
        "param": "Numero de Pares de Polos (p)",
        "role": "Define a velocidade sincrona: ns = 60f/p. Maquinas com mais polos giram mais devagar.",
        "up":   "Velocidade sincrona menor, maior torque por unidade de corrente, resposta mais lenta.",
        "down": "Velocidade sincrona maior, menor torque, resposta mais rapida.",
        "issues": "Escolha de p e fixa construtivamente — nao e ajustavel em operacao.",
    },
    {
        "param": "Momento de Inercia (J)",
        "role": "Determina a constante de tempo mecanica. J alto suaviza variacoes de velocidade.",
        "up":   "Partida mais lenta, menor sobrevalor de velocidade, menor sensibilidade a perturbacoes.",
        "down": "Partida mais rapida, maior sobrevalor de velocidade, mais sensivel a degraus de carga.",
        "issues": "J do sistema inclui carga acoplada — sempre considere o J total no eixo.",
    },
    {
        "param": "Torque de Carga (Tl)",
        "role": "Torque resistente aplicado ao eixo. Define o ponto de operacao em regime.",
        "up":   "Maior escorregamento, maior corrente, maior aquecimento, risco de parada se Tl > Te_max.",
        "down": "Menor escorregamento, menor corrente, operacao mais eficiente.",
        "issues": "Se Tl > torque de pico da maquina, o motor nao parte ou para em operacao.",
    },
]

# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------
def _machine_image_html(machine_id: str, fallback_svg: str) -> str:
    path = os.path.join("images", f"{machine_id}.png")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-height:100px;max-width:100%;object-fit:contain;" />'
    if fallback_svg:
        return fallback_svg
    return '<p style="color:#aaa;font-size:0.85rem;">Imagem nao disponivel</p>'

_MIT_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 80" width="120" height="80">
  <rect x="10" y="20" width="100" height="40" rx="8" fill="#d0e4f7" stroke="#4a90d9" stroke-width="2"/>
  <circle cx="35" cy="40" r="12" fill="#4a90d9" opacity="0.7"/>
  <circle cx="60" cy="40" r="12" fill="#4a90d9" opacity="0.7"/>
  <circle cx="85" cy="40" r="12" fill="#4a90d9" opacity="0.7"/>
  <rect x="5" y="35" width="8" height="10" rx="2" fill="#1a3a6e"/>
  <rect x="107" y="35" width="8" height="10" rx="2" fill="#1a3a6e"/>
</svg>
"""

# ---------------------------------------------------------------------------
# Motor ODE
# ---------------------------------------------------------------------------
def _motor_odes(states, t, Vqs, Vds, Tl, w_ref, Rs, Rr, Xls, Xlr, Xml, wb, p, J, B):
    iqs, ids, iqr, idr, wr = states
    Lls = Xls / wb; Llr = Xlr / wb; Lm = Xml / wb
    Ls = Lls + Lm; Lr = Llr + Lm
    D = Ls * Lr - Lm ** 2
    we = w_ref

    diqs = (Lr * Vqs - Lm * (Rr * iqr - (we - wr) * (Llr * idr + Lm * ids) + Lm * (we * ids))
            - Rs * iqs * Lr + Lm * Rr * iqr - Lm * (we - wr) * Lr * idr) / D
    dids = (Lr * Vds + Lm * Rr * idr + Lm * (we - wr) * (Llr * iqr + Lm * iqs)
            - Rs * ids * Lr - Lm * we * Lr * iqs) / D
    diqr = (Lm * (Rs * iqs - Vqs) + Ls * (Rr * iqr - (we - wr) * (Llr * idr + Lm * ids))
            + Ls * (we - wr) * Lm * ids) / D
    didr = (Lm * (Rs * ids - Vds) + Ls * Rr * idr + Ls * (we - wr) * (Llr * iqr + Lm * iqs)
            - Ls * we * Lm * iqs) / D

    Te = (3 / 2) * (p / 2) * Lm * (iqs * idr - ids * iqr)
    dwr = ((p / 2) * (Te - Tl) - B * wr) / J

    return [diqs, dids, diqr, didr, dwr]


def _simular(V_func, Tl_func, tmax, h, mp):
    wb = 2 * pi * mp["f"]
    w_ref = wb
    Vl = mp["Vl"]
    Vm = sqrt(2) * Vl / sqrt(3)
    Rs, Rr = mp["Rs"], mp["Rr"]
    Xls, Xlr, Xml = mp["Xls"], mp["Xlr"], mp["Xm"]
    p, J, B = int(mp["p"]), mp["J"], mp["B"]
    Tl0 = mp["Tl"]

    t_arr = np.arange(0, tmax + h, h)
    states = np.zeros((len(t_arr), 5))
    y = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i, t in enumerate(t_arr):
        Vl_t = V_func(t)
        Tl_t = Tl_func(t)
        Vm_t = sqrt(2) * Vl_t / sqrt(3)
        Vqs = Vm_t * np.cos(w_ref * t)
        Vds = -Vm_t * np.sin(w_ref * t)
        sol = odeint(_motor_odes, y, [t, t + h],
                     args=(Vqs, Vds, Tl_t, w_ref, Rs, Rr, Xls, Xlr, Xml, wb, p, J, B),
                     full_output=False)
        y = sol[-1].tolist()
        states[i] = y

    iqs, ids, iqr, idr, wr = (states[:, k] for k in range(5))
    Lm = Xml / wb
    Te = (3 / 2) * (p / 2) * Lm * (iqs * idr - ids * iqr)
    n_rpm = wr * 60 / (2 * pi) * (2 / p)

    # Phase currents (inverse Park)
    theta = w_ref * t_arr
    ias = iqs * np.cos(theta) - ids * np.sin(theta)
    ibs = iqs * np.cos(theta - 2 * pi / 3) - ids * np.sin(theta - 2 * pi / 3)
    ics = iqs * np.cos(theta + 2 * pi / 3) - ids * np.sin(theta + 2 * pi / 3)
    iar = iqr * np.cos(theta) - idr * np.sin(theta)
    ibr = iqr * np.cos(theta - 2 * pi / 3) - idr * np.sin(theta - 2 * pi / 3)
    icr = iqr * np.cos(theta + 2 * pi / 3) - idr * np.sin(theta + 2 * pi / 3)

    return dict(t=t_arr, Te=Te, n=n_rpm, wr=wr,
                ias=ias, ibs=ibs, ics=ics,
                iar=iar, ibr=ibr, icr=icr)


def _build_sim_funcs(mp, disturbances, tmax):
    Vl0 = mp["Vl"]
    Tl0 = mp["Tl"]

    # Separate disturbances by type
    v_events = [d for d in disturbances if d["tipo"] in ("Queda de Tensao", "Aumento de Tensao", "Rampa de Tensao")]
    t_events = [d for d in disturbances if d["tipo"] in ("Degrau de Carga", "Remocao de Carga", "Sobrecarga Temporaria")]

    def V_func(t):
        v = Vl0
        for ev in v_events:
            tp = ev["tipo"]
            if tp == "Queda de Tensao" and t >= ev["t_inicio"]:
                v = Vl0 * ev["fator"]
            elif tp == "Aumento de Tensao" and t >= ev["t_inicio"]:
                v = Vl0 * ev["fator"]
            elif tp == "Rampa de Tensao":
                t0, t1, vf = ev["t_inicio"], ev["t_fim"], ev["Vl_final"]
                if t < t0:
                    v = 0.0
                elif t <= t1:
                    v = vf * (t - t0) / (t1 - t0)
                else:
                    v = vf
        return v

    def Tl_func(t):
        tl = Tl0
        for ev in t_events:
            tp = ev["tipo"]
            if tp == "Degrau de Carga" and t >= ev["t_inicio"]:
                tl = ev["Tl_novo"]
            elif tp == "Remocao de Carga" and t >= ev["t_inicio"]:
                tl = 0.0
            elif tp == "Sobrecarga Temporaria":
                if ev["t_inicio"] <= t < ev["t_fim"]:
                    tl = ev["Tl_sobre"]
        return tl

    return V_func, Tl_func


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
def _calc_kpis(res, mp):
    t, Te, n, ias = res["t"], res["Te"], res["n"], res["ias"]
    p = int(mp["p"])
    f = mp["f"]
    ns = 120 * f / p  # synchronous speed RPM

    # Peak torque
    idx_Te = int(np.argmax(np.abs(Te)))
    Te_pico = Te[idx_Te]
    t_Te_pico = t[idx_Te]

    # Peak current (RMS approximation: peak/sqrt2)
    I_inst = np.abs(ias)
    idx_I = int(np.argmax(I_inst))
    I_pico = I_inst[idx_I] / sqrt(2)
    t_I_pico = t[idx_I]

    # Regime values (last 10%)
    idx_reg = int(0.9 * len(t))
    Te_regime = float(np.mean(Te[idx_reg:]))
    n_regime = float(np.mean(n[idx_reg:]))
    s_regime = (ns - n_regime) / ns if ns > 0 else 0.0

    # Settling time (2% of n_regime)
    band = 0.02 * abs(n_regime) if abs(n_regime) > 1 else 1.0
    t_acomodacao = t[-1]
    for i in range(len(t) - 1, -1, -1):
        if abs(n[i] - n_regime) > band:
            t_acomodacao = t[i + 1] if i + 1 < len(t) else t[-1]
            break

    # Speed overshoot
    n_pico_pos = float(np.max(n))
    sobrevalor_n = ((n_pico_pos - n_regime) / abs(n_regime) * 100) if abs(n_regime) > 1 else 0.0

    # Mechanical power
    wb = 2 * pi * f
    wr_regime = float(np.mean(res["wr"][idx_reg:]))
    P_mec = Te_regime * wr_regime

    return {
        "Te_pico_Nm": round(Te_pico, 2),
        "t_Te_pico_s": round(t_Te_pico, 4),
        "I_pico_A": round(I_pico, 2),
        "t_I_pico_s": round(t_I_pico, 4),
        "Te_regime_Nm": round(Te_regime, 2),
        "n_regime_RPM": round(n_regime, 1),
        "s_regime_pct": round(s_regime * 100, 2),
        "t_acomodacao_s": round(t_acomodacao, 4),
        "sobrevalor_n_pct": round(max(sobrevalor_n, 0.0), 2),
        "P_mecanica_W": round(P_mec, 1),
    }


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def _check_alerts(mp: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    erros: List[str] = []
    avisos: List[str] = []
    Vl:  float = float(mp["Vl"])
    f:   float = float(mp["f"])
    Rs:  float = float(mp["Rs"])
    Rr:  float = float(mp["Rr"])
    Xls: float = float(mp["Xls"])
    Xlr: float = float(mp["Xlr"])
    Xm:  float = float(mp["Xm"])
    p:   int   = int(mp["p"])
    J:   float = float(mp["J"])
    Tl:  float = float(mp["Tl"])

    wb: float = 2.0 * pi * f
    Vm: float = sqrt(2.0) * Vl / sqrt(3.0)

    # Thevenin equivalent
    Zth_num = complex(0, Xm) * complex(Rs, Xls)
    Zth_den = complex(Rs, Xls + Xm)
    Zth = Zth_num / Zth_den if abs(Zth_den) > 1e-12 else complex(0, Xm)
    Vth = Vm * abs(complex(0, Xm)) / abs(Zth_den) if abs(Zth_den) > 1e-12 else Vm
    Rth: float = float(Zth.real)
    Xth: float = float(Zth.imag)
    Vth_f: float = float(Vth)

    s_Temax = Rr / sqrt(Rth ** 2 + (Xth + Xlr) ** 2)
    _denom_Te: float = 2.0 * wb / p * (Rth + sqrt(Rth ** 2 + (Xth + Xlr) ** 2))
    Te_max: float = float((3.0 / 2.0) * (p / 2) * Vth_f ** 2 / _denom_Te) if (wb > 0 and _denom_Te > 1e-12) else 0.0

    # 1 — Torque check
    if Tl > Te_max:
        erros.append(f"Torque de carga ({Tl:.1f} N.m) supera o torque maximo ({Te_max:.1f} N.m). Motor nao parte.")
    elif Tl > 0.85 * Te_max:
        avisos.append(f"Torque de carga proximo ao limite ({Tl:.1f} / {Te_max:.1f} N.m). Margem de seguranca baixa.")

    # 2 — Start torque at s=1
    denom_s1 = (Rth + Rr) ** 2 + (Xth + Xlr) ** 2
    Te_partida = (3 / 2) * (p / 2) * Vth ** 2 * Rr / (
        wb / p * denom_s1
    ) if denom_s1 > 1e-12 else 0
    if Te_partida < Tl:
        erros.append(f"Torque de partida ({Te_partida:.1f} N.m) insuficiente para vencer a carga ({Tl:.1f} N.m).")

    # 3 — Magnetizing current
    Im = Vm / (Xm + Xls) if (Xm + Xls) > 0 else 0
    In_approx = Tl / (Vm * 0.9 + 1e-9)
    if Im > 0.5 * (Im + In_approx) and (Im + In_approx) > 0:
        avisos.append("Corrente magnetizante representa mais de 50% da corrente total. Fator de potencia ruim.")

    # 4 — Rs / Rr ratio
    if Rs > 0 and Rr > 0:
        ratio = Rs / Rr
        if ratio > 5:
            avisos.append(f"Rs/Rr = {ratio:.1f} — resistencia do estator muito alta relativa ao rotor. Eficiencia comprometida.")

    # 5 — Xm vs dispersions
    sigma_tot = Xls + Xlr
    if sigma_tot > 0 and Xm / sigma_tot < 3:
        avisos.append("Xm pouco dominante em relacao as dispersoes. Acoplamento magnetico fraco.")

    # 6 — Mechanical time constant: tau = J * ws / Te_max
    ws = 2 * pi * f / (p / 2)
    tau_mec: float = (J * ws / float(Te_max)) if Te_max > 1e-6 else float("inf")
    if tau_mec > 5:
        avisos.append(f"Constante de tempo mecanica elevada ({tau_mec:.2f} s). Partida lenta — considere reduzir J ou aumentar Te_max.")

    # 7 — Frequency range
    if f < 45 or f > 65:
        avisos.append(f"Frequencia {f} Hz fora da faixa nominal (45-65 Hz). Verifique se e operacao por inversor.")

    return erros, avisos


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def _apply_fig_style(fig, axes_list, dark_bg: bool):
    bg = "#12192a" if dark_bg else "white"
    txt = "#e0e8f8" if dark_bg else "#222"
    grid_c = "#2a3a5a" if dark_bg else "#cccccc"
    fig.patch.set_facecolor(bg)
    for ax in axes_list:
        ax.set_facecolor(bg)
        ax.tick_params(colors=txt)
        ax.xaxis.label.set_color(txt)
        ax.yaxis.label.set_color(txt)
        ax.title.set_color(txt)
        ax.spines["bottom"].set_color(txt)
        ax.spines["left"].set_color(txt)
        for gl in ax.get_xgridlines() + ax.get_ygridlines():
            gl.set_color(grid_c)


def _render_plots(res, selected_vars, dark_bg, stored_curves, mp):
    if not selected_vars:
        st.info("Selecione ao menos uma variavel para visualizar.")
        return None

    var_list = [v for v in VAR_OPTIONS if v in selected_vars]
    n_plots = len(var_list)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.8 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    y_labels = {
        "Te": "Torque (N.m)", "n": "Velocidade (RPM)",
        "ias": "Ias (A)", "ibs": "Ibs (A)", "ics": "Ics (A)",
        "iar": "Iar (A)", "ibr": "Ibr (A)", "icr": "Icr (A)",
    }

    for ax, var in zip(axes, var_list):
        # Stored curves (dashed)
        for ci, sc in enumerate(stored_curves):
            if var in sc:
                lbl = sc.get("label", f"Curva {ci + 1}")
                ax.plot(sc["t"], sc[var], "--", color=_COLORS[(ci + 1) % len(_COLORS)],
                        alpha=0.65, linewidth=1.3, label=lbl)
        # Current curve (solid)
        ax.plot(res["t"], res[var], color=_COLORS[0], linewidth=1.8, label="Atual")
        ax.set_ylabel(y_labels.get(var, var))
        if len(stored_curves) > 0 or len(var_list) == 1:
            ax.legend(fontsize=9, framealpha=0.5)

    axes[-1].set_xlabel("Tempo (s)")
    fig.tight_layout()
    _apply_fig_style(fig, axes, dark_bg)
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def _export_csv(res, selected_vars):
    cols = {"t": res["t"]}
    for v in VAR_OPTIONS:
        if v in selected_vars and v in res:
            cols[v] = res[v]
    df = pd.DataFrame(cols)
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def _export_fig(fig, fmt: str):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Screen: Home
# ---------------------------------------------------------------------------
def screen_home():
    st.markdown("# Simulador de Maquinas Eletricas")
    st.markdown("Selecione o tipo de maquina para iniciar a simulacao.")
    st.markdown("---")

    MACHINES = [
        {
            "id": "mit",
            "name": "Maquina de Inducao Trifasica",
            "desc": "Motor de gaiola de esquilo | Modelo dq0 | Dinamico",
            "enabled": True,
            "svg": _MIT_SVG,
        },
        {
            "id": "sync",
            "name": "Maquina Sincrona",
            "desc": "Em breve",
            "enabled": False,
            "svg": "",
        },
        {
            "id": "dc",
            "name": "Maquina de Corrente Continua",
            "desc": "Em breve",
            "enabled": False,
            "svg": "",
        },
    ]

    cols = st.columns(len(MACHINES), gap="medium")
    for col, m in zip(cols, MACHINES):
        with col:
            img_html = _machine_image_html(m["id"], m["svg"])
            disabled_cls = "" if m["enabled"] else " disabled"
            st.markdown(
                f"""<div class="machine-card{disabled_cls}">
                    {img_html}
                    <div class="card-title">{m["name"]}</div>
                    <div class="card-desc">{m["desc"]}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.write("")
            if m["enabled"]:
                if st.button("Selecionar", key=f"sel_{m['id']}", use_container_width=True):
                    st.session_state.machine = m["id"]
                    st.session_state.page = "params"
                    st.session_state.params_confirmed = False
                    st.session_state.sim_results = None
                    st.rerun()
            else:
                st.button("Em breve", key=f"sel_{m['id']}", disabled=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Screen: Params
# ---------------------------------------------------------------------------
def screen_params():
    # Header
    col_hd, col_back = st.columns([6, 1])
    with col_hd:
        st.markdown("## Maquina de Inducao Trifasica — Configuracao e Simulacao")
    with col_back:
        if st.button("Voltar", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.params_confirmed = False
            st.rerun()

    tab_conf, tab_learn = st.tabs(["Configuracao e Simulacao", "Aprendizado"])

    # ------------------------------------------------------------------ TAB 1
    with tab_conf:
        col_main, col_solver = st.columns([1.2, 1], gap="large")

        # ---- LEFT COLUMN: Steps 1-3 ----
        with col_main:
            # Step 1 badge
            st.markdown(
                '<span class="step-badge">1</span><strong>Parametros da Maquina</strong>',
                unsafe_allow_html=True,
            )

            with st.expander("Parametros eletricos e mecanicos", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    Vl = st.number_input("Tensao de linha (V)", 100.0, 13800.0,
                                         st.session_state.get("p_Vl", 220.0),
                                         step=10.0, key="p_Vl")
                    f  = st.number_input("Frequencia (Hz)", 45.0, 65.0,
                                         st.session_state.get("p_f", 60.0),
                                         step=0.5, key="p_f")
                    Rs = st.number_input("Rs — Resist. Estator (Ohm)", 0.001, 100.0,
                                         st.session_state.get("p_Rs", 0.641),
                                         step=0.01, format="%.4f", key="p_Rs")
                    Rr = st.number_input("Rr — Resist. Rotor (Ohm)", 0.001, 100.0,
                                         st.session_state.get("p_Rr", 0.332),
                                         step=0.01, format="%.4f", key="p_Rr")
                    Xm = st.number_input("Xm — Reat. Mag. (Ohm)", 0.1, 1000.0,
                                         st.session_state.get("p_Xm", 36.0),
                                         step=0.5, format="%.3f", key="p_Xm")
                with c2:
                    Xls = st.number_input("Xls — Disp. Estator (Ohm)", 0.001, 100.0,
                                          st.session_state.get("p_Xls", 1.106),
                                          step=0.01, format="%.4f", key="p_Xls")
                    Xlr = st.number_input("Xlr — Disp. Rotor (Ohm)", 0.001, 100.0,
                                          st.session_state.get("p_Xlr", 0.464),
                                          step=0.01, format="%.4f", key="p_Xlr")
                    p   = st.number_input("Pares de polos (p)", 1, 12,
                                          int(st.session_state.get("p_p", 2)),
                                          step=1, key="p_p")
                    J   = st.number_input("Inercia J (kg.m2)", 0.0001, 100.0,
                                          st.session_state.get("p_J", 0.089),
                                          step=0.001, format="%.4f", key="p_J")
                    B   = st.number_input("Amortecimento B (N.m.s/rad)", 0.0, 10.0,
                                          st.session_state.get("p_B", 0.01),
                                          step=0.001, format="%.4f", key="p_B")
                    Tl  = st.number_input("Torque de carga Tl (N.m)", 0.0, 10000.0,
                                          st.session_state.get("p_Tl", 0.0),
                                          step=0.5, format="%.2f", key="p_Tl")

            # Real-time alerts
            mp_live = dict(Vl=st.session_state.get("p_Vl", 220.0),
                           f=st.session_state.get("p_f", 60.0),
                           Rs=st.session_state.get("p_Rs", 0.641),
                           Rr=st.session_state.get("p_Rr", 0.332),
                           Xm=st.session_state.get("p_Xm", 36.0),
                           Xls=st.session_state.get("p_Xls", 1.106),
                           Xlr=st.session_state.get("p_Xlr", 0.464),
                           p=int(st.session_state.get("p_p", 2)),
                           J=st.session_state.get("p_J", 0.089),
                           B=st.session_state.get("p_B", 0.01),
                           Tl=st.session_state.get("p_Tl", 0.0))
            erros, avisos = _check_alerts(mp_live)
            if erros:
                for e in erros:
                    st.markdown(f'<div class="alert-error"><strong>Erro:</strong> {e}</div>', unsafe_allow_html=True)
            if avisos:
                for a in avisos:
                    st.markdown(f'<div class="alert-warning"><strong>Aviso:</strong> {a}</div>', unsafe_allow_html=True)

            if st.button("Confirmar Parametros", type="primary", use_container_width=True):
                st.session_state.machine_params = mp_live.copy()
                st.session_state.params_confirmed = True
                st.session_state.sim_results = None
                st.session_state.disturbances = []
                st.rerun()

            # Steps 2 & 3 — only after confirm
            if st.session_state.params_confirmed:
                st.markdown("---")

                # Step 2
                st.markdown(
                    '<span class="step-badge">2</span><strong>Eventos e Perturbacoes</strong>',
                    unsafe_allow_html=True,
                )
                disturbances = st.session_state.disturbances

                with st.expander("Adicionar perturbacao", expanded=False):
                    tipo_novo = st.selectbox("Tipo de perturbacao", list(DISTURB_TYPES.keys()), key="new_disturb_tipo")
                    cfg = DISTURB_TYPES[tipo_novo]
                    st.caption(cfg["desc"])
                    vals_novo = {}
                    d_cols = st.columns(len(cfg["params"]))
                    for ci, (pn, lbl, dv) in enumerate(zip(cfg["params"], cfg["labels"], cfg["defaults"])):
                        with d_cols[ci]:
                            vals_novo[pn] = st.number_input(lbl, value=float(dv), key=f"new_d_{pn}")
                    if st.button("Adicionar perturbacao", key="btn_add_disturb"):
                        entry = {"tipo": tipo_novo}
                        entry.update(vals_novo)
                        disturbances.append(entry)
                        st.session_state.disturbances = disturbances
                        st.rerun()

                if disturbances:
                    st.markdown("**Perturbacoes configuradas:**")
                    for di, dv in enumerate(disturbances):
                        dc1, dc2 = st.columns([4, 1])
                        with dc1:
                            detail = ", ".join(f"{k}={v}" for k, v in dv.items() if k != "tipo")
                            st.markdown(f"- **{dv['tipo']}** — {detail}")
                        with dc2:
                            if st.button("Remover", key=f"rem_d_{di}"):
                                disturbances.pop(di)
                                st.session_state.disturbances = disturbances
                                st.rerun()
                else:
                    st.caption("Nenhuma perturbacao adicionada. A simulacao usa os parametros nominais durante todo o periodo.")

                st.markdown("---")

                # Step 3
                st.markdown(
                    '<span class="step-badge">3</span><strong>Variaveis para Visualizacao</strong>',
                    unsafe_allow_html=True,
                )
                sel = st.session_state.selected_vars
                new_sel = set()
                v_cols = st.columns(4)
                for vi, (vk, vname) in enumerate(VAR_OPTIONS.items()):
                    with v_cols[vi % 4]:
                        if st.checkbox(vname, value=(vk in sel), key=f"var_{vk}"):
                            new_sel.add(vk)
                st.session_state.selected_vars = new_sel

        # ---- RIGHT COLUMN: Solver ----
        with col_solver:
            if st.session_state.params_confirmed:
                st.markdown('<div class="solver-banner-ok">Parametros confirmados. Pronto para simular.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="solver-banner-wait">Confirme os parametros para habilitar a simulacao.</div>',
                            unsafe_allow_html=True)

            st.markdown(
                '<span class="step-badge">4</span><strong>Tempo e Integracao</strong>',
                unsafe_allow_html=True,
            )
            st.caption(
                "O metodo de integracao utilizado e o ODEINT (LSODA), adequado para sistemas rigidos "
                "como o modelo dq0 do motor de inducao. Reduza o passo h para maior precisao nos transitories "
                "rapidos; aumente tmax para capturar o regime permanente em sistemas com grande inercia."
            )

            tmax = st.number_input("Tempo total de simulacao (s)", 0.5, 60.0, 3.0, step=0.5, key="s_tmax")
            h    = st.number_input("Passo de integracao h (s)", 1e-5, 0.01,
                                   0.0005, step=0.0001, format="%.5f", key="s_h")

            dark_bg = st.toggle("Fundo escuro nos graficos", value=st.session_state.plot_dark_bg, key="s_dark")
            st.session_state.plot_dark_bg = dark_bg

            disabled_run = not st.session_state.params_confirmed
            if st.button("Executar Simulacao", type="primary", use_container_width=True, disabled=disabled_run):
                mp = st.session_state.machine_params
                dist = st.session_state.disturbances
                with st.spinner("Simulando..."):
                    V_func, Tl_func = _build_sim_funcs(mp, dist, tmax)
                    res = _simular(V_func, Tl_func, tmax, h, mp)
                st.session_state.sim_results = res
                st.rerun()

            # Results
            if st.session_state.sim_results is not None:
                res = st.session_state.sim_results
                mp  = st.session_state.machine_params
                kpis = _calc_kpis(res, mp)

                st.markdown("---")
                st.markdown("**Indicadores de Desempenho**")

                kpi_labels = {
                    "Te_pico_Nm": "Torque Pico (N.m)",
                    "t_Te_pico_s": "t Torque Pico (s)",
                    "I_pico_A": "Corrente Pico (A)",
                    "t_I_pico_s": "t Corrente Pico (s)",
                    "Te_regime_Nm": "Torque Regime (N.m)",
                    "n_regime_RPM": "Velocidade Regime (RPM)",
                    "s_regime_pct": "Escorregamento (%)",
                    "t_acomodacao_s": "Acomodacao (s)",
                    "sobrevalor_n_pct": "Sobrevalor Veloc. (%)",
                    "P_mecanica_W": "Pot. Mecanica (W)",
                }
                cells_html = ""
                for k, lbl in kpi_labels.items():
                    val = kpis.get(k, "-")
                    cells_html += (
                        f'<div class="kpi-cell">'
                        f'<div class="kpi-label">{lbl}</div>'
                        f'<div class="kpi-value">{val}</div>'
                        f'</div>'
                    )
                st.markdown(f'<div class="kpi-grid">{cells_html}</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("**Graficos**")
                fig = _render_plots(
                    res,
                    st.session_state.selected_vars,
                    st.session_state.plot_dark_bg,
                    st.session_state.stored_curves,
                    mp,
                )
                if fig is not None:
                    st.pyplot(fig)

                    # Export buttons
                    exp_c1, exp_c2, exp_c3 = st.columns(3)
                    with exp_c1:
                        csv_buf = _export_csv(res, st.session_state.selected_vars)
                        st.download_button("Exportar CSV", csv_buf, "simulacao.csv", "text/csv",
                                           use_container_width=True)
                    with exp_c2:
                        pdf_buf = _export_fig(fig, "pdf")
                        st.download_button("Exportar PDF", pdf_buf, "simulacao.pdf", "application/pdf",
                                           use_container_width=True)
                    with exp_c3:
                        svg_buf = _export_fig(fig, "svg")
                        st.download_button("Exportar SVG", svg_buf, "simulacao.svg", "image/svg+xml",
                                           use_container_width=True)

                # Curve comparison
                st.markdown("---")
                save_lbl = st.text_input("Rotulo para salvar esta curva", value="Curva A", key="save_lbl")
                sc1, sc2 = st.columns(2)
                with sc1:
                    if st.button("Salvar curva para comparacao", use_container_width=True):
                        entry = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                                 for k, v in res.items()}
                        entry["label"] = save_lbl
                        st.session_state.stored_curves.append(entry)
                        st.success(f"Curva '{save_lbl}' salva.")
                with sc2:
                    if st.button("Limpar curvas salvas", use_container_width=True):
                        st.session_state.stored_curves = []
                        st.rerun()

                if st.session_state.stored_curves:
                    st.caption(f"{len(st.session_state.stored_curves)} curva(s) salva(s): "
                               + ", ".join(c.get("label", f"#{i+1}")
                                           for i, c in enumerate(st.session_state.stored_curves)))

    # ------------------------------------------------------------------ TAB 2
    with tab_learn:
        st.markdown("## Aba de Aprendizado")
        st.caption(
            "Esta aba apresenta exclusivamente informacoes qualitativas sobre os parametros da maquina. "
            "Os diagnosticos abaixo refletem os valores atualmente inseridos nos campos de parametros."
        )

        # Real-time diagnostics from current widget values
        mp_live2 = dict(Vl=st.session_state.get("p_Vl", 220.0),
                        f=st.session_state.get("p_f", 60.0),
                        Rs=st.session_state.get("p_Rs", 0.641),
                        Rr=st.session_state.get("p_Rr", 0.332),
                        Xm=st.session_state.get("p_Xm", 36.0),
                        Xls=st.session_state.get("p_Xls", 1.106),
                        Xlr=st.session_state.get("p_Xlr", 0.464),
                        p=int(st.session_state.get("p_p", 2)),
                        J=st.session_state.get("p_J", 0.089),
                        B=st.session_state.get("p_B", 0.01),
                        Tl=st.session_state.get("p_Tl", 0.0))
        erros2, avisos2 = _check_alerts(mp_live2)

        st.markdown("### Diagnostico em Tempo Real")
        if not erros2 and not avisos2:
            st.success("Parametros dentro de faixas operacionais aceitaveis.")
        for e in erros2:
            st.markdown(f'<div class="alert-error"><strong>Problema critico:</strong> {e}</div>',
                        unsafe_allow_html=True)
        for a in avisos2:
            st.markdown(f'<div class="alert-warning"><strong>Atencao:</strong> {a}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Influencia dos Parametros")

        for entry in LEARNING_DATA:
            with st.expander(entry["param"]):
                st.markdown(
                    f'<div class="param-card">'
                    f'<div class="pc-title">{entry["param"]}</div>'
                    f'<div class="pc-role">{entry["role"]}</div>'
                    f'<div class="pc-up"><strong>Aumentar:</strong> {entry["up"]}</div>'
                    f'<div class="pc-down"><strong>Reduzir:</strong> {entry["down"]}</div>'
                    f'<div class="pc-issue"><strong>Problemas tipicos:</strong> {entry["issues"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("### Guia de Cenarios")
        st.markdown("""
| Cenario | Configuracao recomendada |
|---------|--------------------------|
| Partida direta sem carga | Tl = 0, sem perturbacoes |
| Partida direta com carga | Tl > 0 nominal, sem perturbacoes |
| Degrau de carga em regime | Tl inicial baixo, adicionar Degrau de Carga apos ~1 s |
| Soft-starter | Adicionar Rampa de Tensao de 0 a Vl nominal |
| Queda de tensao em rede | Adicionar Queda de Tensao apos partida estabilizada |
| Sobrecarga momentanea | Adicionar Sobrecarga Temporaria com Tl > nominal por intervalo curto |
| Variacao de carga multipla | Combinar Degrau de Carga e Remocao de Carga em instantes distintos |
        """)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if st.session_state.page == "home":
    screen_home()
elif st.session_state.page == "params":
    screen_params()
