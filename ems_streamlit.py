# -*- coding: utf-8 -*-
"""
ems_streamlit.py — Simulador de Maquinas Eletricas
Modelo 0dq de Krause — Integracao RK4 (scipy.odeint)

Estrutura de modulos:
  BLOCO A — Modelo matematico (nucleo fisico, inalterado)
  BLOCO B — Tema e CSS
  BLOCO C — Selecao de maquina (tela inicial)
  BLOCO D — Aba Simulacao
  BLOCO E — Aba Circuito Equivalente
  BLOCO F — Aba Resultados
  BLOCO G — Aba Teoria
  BLOCO H — Orquestrador principal
"""

from __future__ import annotations
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACAO DA PAGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador de Maquinas Eletricas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═════════════════════════════════════════════════════════════════════════════
# BLOCO A — MODELO MATEMATICO (logica original preservada integralmente)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MachineParams:
    Vl:  float = 220.0
    f:   float = 60.0
    Rs:  float = 0.435
    Rr:  float = 0.816
    Xm:  float = 26.13
    Xls: float = 0.754
    Xlr: float = 0.754
    p:   int   = 4
    J:   float = 0.089
    B:   float = 0.0
    Xml: float = field(init=False)
    wb:  float = field(init=False)

    def __post_init__(self) -> None:
        self.Xml = 1.0 / (1.0/self.Xm + 1.0/self.Xls + 1.0/self.Xlr)
        self.wb  = 2.0 * np.pi * self.f

    @property
    def n_sync(self) -> float:
        return 120.0 * self.f / self.p


def induction_motor_ode(states, t, Vqs, Vds, Tl, w_ref, mp):
    PSIqs, PSIds, PSIqr, PSIdr, wr = states
    PSImq = mp.Xml * (PSIqs / mp.Xls + PSIqr / mp.Xlr)
    PSImd = mp.Xml * (PSIds / mp.Xls + PSIdr / mp.Xlr)
    iqs = (1.0 / mp.Xls) * (PSIqs - PSImq)
    ids = (1.0 / mp.Xls) * (PSIds - PSImd)
    dPSIqs = mp.wb * (Vqs - (w_ref / mp.wb) * PSIds + (mp.Rs / mp.Xls) * (PSImq - PSIqs))
    dPSIds = mp.wb * (Vds + (w_ref / mp.wb) * PSIqs + (mp.Rs / mp.Xls) * (PSImd - PSIds))
    slip_ref = (w_ref - wr) / mp.wb
    dPSIqr = mp.wb * (-slip_ref * PSIdr + (mp.Rr / mp.Xlr) * (PSImq - PSIqr))
    dPSIdr = mp.wb * ( slip_ref * PSIqr + (mp.Rr / mp.Xlr) * (PSImd - PSIdr))
    Te  = (3.0/2.0) * (mp.p/2.0) * (1.0/mp.wb) * (PSIds * iqs - PSIqs * ids)
    dwr = (mp.p / (2.0*mp.J)) * (Te - Tl) - (mp.B/mp.J) * wr
    return [dPSIqs, dPSIds, dPSIqr, dPSIdr, dwr]


def abc_voltages(t, Vl, f):
    tetae = 2.0 * np.pi * f * t
    Va = np.sqrt(2.0) * Vl * np.sin(tetae)
    Vb = np.sqrt(2.0) * Vl * np.sin(tetae - 2.0*np.pi/3.0)
    Vc = np.sqrt(2.0) * Vl * np.sin(tetae + 2.0*np.pi/3.0)
    return Va, Vb, Vc


def clarke_park_transform(Va, Vb, Vc, tetae):
    k = np.sqrt(3.0/2.0)
    Valpha = k * (Va - 0.5*Vb - 0.5*Vc)
    Vbeta  = k * ((np.sqrt(3.0)/2.0)*Vb - (np.sqrt(3.0)/2.0)*Vc)
    Vds = np.cos(tetae)*Valpha + np.sin(tetae)*Vbeta
    Vqs = -np.sin(tetae)*Valpha + np.cos(tetae)*Vbeta
    return Vds, Vqs


def reconstruct_abc_currents(PSIqs, PSIds, PSIqr, PSIdr, tetae, tetar, mp):
    PSImq = mp.Xml*(PSIqs/mp.Xls + PSIqr/mp.Xlr)
    PSImd = mp.Xml*(PSIds/mp.Xls + PSIdr/mp.Xlr)
    ids = (1.0/mp.Xls)*(PSIds - PSImd)
    iqs = (1.0/mp.Xls)*(PSIqs - PSImq)
    idr = (1.0/mp.Xlr)*(PSIdr - PSImd)
    iqr = (1.0/mp.Xlr)*(PSIqr - PSImq)
    ias_alpha = np.cos(tetae)*ids - np.sin(tetae)*iqs
    ias_beta  = np.sin(tetae)*ids + np.cos(tetae)*iqs
    iar_alpha = np.cos(tetar)*idr - np.sin(tetar)*iqr
    iar_beta  = np.sin(tetar)*idr + np.cos(tetar)*iqr
    k = np.sqrt(3.0/2.0)
    ias = k*ias_alpha
    ibs = k*(-0.5*ias_alpha + (np.sqrt(3.0)/2.0)*ias_beta)
    ics = k*(-0.5*ias_alpha - (np.sqrt(3.0)/2.0)*ias_beta)
    iar = k*iar_alpha
    ibr = k*(-0.5*iar_alpha + (np.sqrt(3.0)/2.0)*iar_beta)
    icr = k*(-0.5*iar_alpha - (np.sqrt(3.0)/2.0)*iar_beta)
    return ids, iqs, idr, iqr, ias, ibs, ics, iar, ibr, icr


def voltage_reduced_start(t, Vl_nominal, Vl_reduced, t_switch):
    return Vl_nominal if t >= t_switch else Vl_reduced


def voltage_soft_starter(t, Vl_nominal, Vl_initial, t_start_ramp, t_full):
    if t < t_start_ramp:
        return Vl_initial
    elif t < t_full:
        return Vl_initial + (Vl_nominal - Vl_initial)*(t - t_start_ramp)/(t_full - t_start_ramp)
    return Vl_nominal


def torque_step(t, Tl_before, Tl_after, t_switch):
    return Tl_after if t >= t_switch else Tl_before


def run_simulation(mp, tmax, h, voltage_fn, torque_fn, ref_code=1):
    t_values = np.arange(0.0, tmax, h)
    N = len(t_values)
    arrays = {k: np.empty(N) for k in [
        "wr","n","Te","ids","iqs","idr","iqr",
        "ias","ibs","ics","iar","ibr","icr","Va","Vb","Vc","Vds","Vqs"
    ]}
    states, last_wr, we = [0.0]*5, 0.0, mp.wb
    for i, t_val in enumerate(t_values):
        Vl_apli    = voltage_fn(t_val)
        current_Tl = torque_fn(t_val)
        tetae = we * t_val
        w_ref = we if ref_code == 1 else (last_wr if ref_code == 2 else 0.0)
        Va, Vb, Vc = abc_voltages(t_val, Vl_apli, mp.f)
        Vds, Vqs   = clarke_park_transform(Va, Vb, Vc, tetae)
        sol    = odeint(induction_motor_ode, states, [t_val, t_val+h],
                        args=(Vqs, Vds, current_Tl, w_ref, mp))
        states = list(sol[1])
        PSIqs, PSIds, PSIqr, PSIdr, wr = states
        last_wr  = wr
        tetar_abc = wr * t_val
        ids, iqs, idr, iqr, ias, ibs, ics, iar, ibr, icr = reconstruct_abc_currents(
            PSIqs, PSIds, PSIqr, PSIdr, tetae, tetar_abc, mp)
        Te = (3.0/2.0)*(mp.p/2.0)*(1.0/mp.wb)*(PSIds*iqs - PSIqs*ids)
        arrays["wr"][i]  = wr
        arrays["n"][i]   = (120.0/mp.p)*(wr/(2.0*np.pi))
        arrays["Te"][i]  = Te
        arrays["ids"][i] = ids; arrays["iqs"][i] = iqs
        arrays["idr"][i] = idr; arrays["iqr"][i] = iqr
        arrays["ias"][i] = ias; arrays["ibs"][i] = ibs; arrays["ics"][i] = ics
        arrays["iar"][i] = iar; arrays["ibr"][i] = ibr; arrays["icr"][i] = icr
        arrays["Va"][i]  = Va;  arrays["Vb"][i]  = Vb;  arrays["Vc"][i]  = Vc
        arrays["Vds"][i] = Vds; arrays["Vqs"][i] = Vqs
    arrays["t"] = t_values
    return arrays


def build_voltage_and_torque_fns(config: dict, mp: MachineParams):
    exp = config["exp_type"]
    t_events = []
    if exp == "dol":
        Tl = config["Tl_final"]
        tc = config["t_carga"]
        vfn = lambda t: mp.Vl
        tfn = lambda t: torque_step(t, 0.0, Tl, tc)
        t_events = [tc]
    elif exp == "yd":
        Vl_Y = mp.Vl / np.sqrt(3.0)
        Tl   = config["Tl_final"]
        t2   = config["t_2"]
        tc   = config["t_carga"]
        vfn  = lambda t: voltage_reduced_start(t, mp.Vl, Vl_Y, t2)
        tfn  = lambda t: torque_step(t, 0.0, Tl, tc)
        t_events = [t2, tc]
    elif exp == "comp":
        Vl_red = mp.Vl * config["voltage_ratio"]
        Tl     = config["Tl_final"]
        t2     = config["t_2"]
        tc     = config["t_carga"]
        vfn    = lambda t: voltage_reduced_start(t, mp.Vl, Vl_red, t2)
        tfn    = lambda t: torque_step(t, 0.0, Tl, tc)
        t_events = [t2, tc]
    elif exp == "soft":
        Vl_init = mp.Vl * config["voltage_ratio"]
        t2      = config["t_2"]
        tp      = config["t_pico"]
        Tl      = config["Tl_final"]
        tc      = config["t_carga"]
        vfn = lambda t: voltage_soft_starter(t, mp.Vl, Vl_init, t2, tp)
        tfn = lambda t: torque_step(t, 0.0, Tl, tc)
        t_events = [t2, tp, tc]
    elif exp == "carga":
        Tl = config["Tl_final"]
        tc = config["t_carga"]
        vfn = lambda t: mp.Vl
        tfn = lambda t: torque_step(t, 0.0, Tl, tc)
        t_events = [tc]
    elif exp == "gerador":
        Tl_neg = -config["Tl_mec"]
        vfn = lambda t: mp.Vl
        tfn = lambda t: Tl_neg
        t_events = [config["t_2"]]
    else:
        vfn = lambda t: mp.Vl
        tfn = lambda t: 0.0
    return vfn, tfn, t_events


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO B — TEMA E CSS
# ═════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "dark": {
        "bg":         "#0d1117",
        "surface":    "#161b27",
        "surface2":   "#1e2538",
        "border":     "#2a3150",
        "accent":     "#4f8ef7",
        "accent2":    "#7c6af7",
        "text":       "#e4e8f5",
        "text_muted": "#8892b0",
        "success":    "#22c55e",
        "danger":     "#ef4444",
        "warn_bg":    "rgba(239,68,68,0.08)",
        "input_bg":   "#1a2035",
        "tag_bg":     "#1e2d4a",
    },
    "light": {
        "bg":         "#f0f4ff",
        "surface":    "#ffffff",
        "surface2":   "#eef2ff",
        "border":     "#cbd5f0",
        "accent":     "#2563eb",
        "accent2":    "#7c3aed",
        "text":       "#111827",
        "text_muted": "#4b5563",
        "success":    "#16a34a",
        "danger":     "#dc2626",
        "warn_bg":    "rgba(220,38,38,0.06)",
        "input_bg":   "#eef2ff",
        "tag_bg":     "#dbeafe",
    },
}


def get_palette() -> dict:
    return PALETTE["dark"] if st.session_state.get("dark_mode", True) else PALETTE["light"]


def apply_css() -> None:
    c = get_palette()
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

      html, body,
      [data-testid="stAppViewContainer"],
      [data-testid="stMain"],
      [data-testid="block-container"] {{
        background-color: {c["bg"]} !important;
        color: {c["text"]};
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
      }}

      /* oculta sidebar completamente */
      [data-testid="stSidebar"],
      [data-testid="collapsedControl"] {{
        display: none !important;
      }}

      /* cabecalho da pagina */
      .app-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.2rem 0 1rem 0;
        border-bottom: 1px solid {c["border"]};
        margin-bottom: 2rem;
      }}
      .app-title {{
        font-size: 1.35rem;
        font-weight: 700;
        color: {c["text"]};
        letter-spacing: -0.02em;
      }}
      .app-subtitle {{
        font-size: 0.82rem;
        color: {c["text_muted"]};
        margin-top: 0.1rem;
      }}

      /* cartoes de maquina */
      .machine-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
      }}
      .machine-card {{
        background: {c["surface"]};
        border: 2px solid {c["border"]};
        border-radius: 14px;
        padding: 1.8rem 1.2rem;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.2s, box-shadow 0.2s;
      }}
      .machine-card:hover {{
        border-color: {c["accent"]};
        box-shadow: 0 0 0 3px {c["accent"]}22;
      }}
      .machine-card.active {{
        border-color: {c["accent"]};
        background: {c["tag_bg"]};
        box-shadow: 0 0 0 3px {c["accent"]}33;
      }}
      .machine-card.disabled {{
        opacity: 0.38;
        cursor: not-allowed;
        pointer-events: none;
      }}
      .machine-card .card-icon {{
        font-size: 2.4rem;
        margin-bottom: 0.6rem;
        display: block;
        filter: grayscale(0.2);
      }}
      .machine-card .card-name {{
        font-size: 1rem;
        font-weight: 700;
        color: {c["text"]};
      }}
      .machine-card .card-tag {{
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: {c["tag_bg"]};
        color: {c["accent"]};
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        margin-top: 0.5rem;
      }}
      .machine-card .card-tag.soon {{
        background: {c["surface2"]};
        color: {c["text_muted"]};
      }}

      /* secao de titulo */
      .section-label {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {c["accent"]};
        margin-bottom: 0.5rem;
      }}
      .section-divider {{
        border: none;
        border-top: 1px solid {c["border"]};
        margin: 1.4rem 0;
      }}

      /* card de parametros */
      .param-group {{
        background: {c["surface"]};
        border: 1px solid {c["border"]};
        border-radius: 12px;
        padding: 1.2rem 1.4rem 1rem 1.4rem;
        margin-bottom: 1rem;
      }}
      .param-group-title {{
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {c["accent"]};
        margin-bottom: 0.9rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {c["border"]};
      }}

      /* info box */
      .info-box {{
        background: {c["input_bg"]};
        border-left: 3px solid {c["accent"]};
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        color: {c["text_muted"]};
        line-height: 1.6;
        margin: 0.6rem 0;
      }}

      /* KPI metrics override */
      [data-testid="stMetric"] {{
        background: {c["surface"]};
        border: 1px solid {c["border"]};
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
      }}
      [data-testid="stMetricLabel"] {{
        font-size: 0.78rem !important;
        color: {c["text_muted"]} !important;
        font-weight: 500 !important;
      }}
      [data-testid="stMetricValue"] {{
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: {c["text"]} !important;
      }}

      /* abas */
      [data-baseweb="tab-list"] {{
        background: {c["surface"]} !important;
        border-radius: 10px !important;
        padding: 0.25rem !important;
        border: 1px solid {c["border"]} !important;
        gap: 0.2rem !important;
      }}
      [data-baseweb="tab"] {{
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: {c["text_muted"]} !important;
        border-radius: 7px !important;
        padding: 0.45rem 1.1rem !important;
      }}
      [aria-selected="true"][data-baseweb="tab"] {{
        background: {c["accent"]} !important;
        color: #ffffff !important;
      }}

      /* inputs */
      input[type="number"], select, textarea {{
        background: {c["input_bg"]} !important;
        color: {c["text"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 7px !important;
        font-size: 0.95rem !important;
      }}
      label {{
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: {c["text_muted"]} !important;
      }}

      /* botao principal */
      .run-btn-wrap {{
        display: flex;
        justify-content: center;
        margin: 1.5rem 0 0.5rem 0;
      }}
      .stButton > button {{
        background: {c["accent"]} !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.65rem 2.5rem !important;
        letter-spacing: 0.02em;
        transition: opacity 0.15s;
      }}
      .stButton > button:hover {{ opacity: 0.87 !important; }}

      /* teoria cards */
      .theory-card {{
        background: {c["surface"]};
        border: 1px solid {c["border"]};
        border-left: 4px solid {c["accent2"]};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
      }}
      .theory-card h4 {{
        font-size: 1rem;
        font-weight: 700;
        color: {c["text"]};
        margin: 0 0 0.5rem 0;
      }}
      .theory-card p {{
        font-size: 0.92rem;
        line-height: 1.75;
        color: {c["text_muted"]};
        margin: 0.3rem 0;
      }}
      .tc-up   {{ color: {c["success"]}; font-weight: 600; }}
      .tc-down {{ color: {c["danger"]};  font-weight: 600; }}
      .tc-warn {{
        background: {c["warn_bg"]};
        border-left: 3px solid {c["danger"]};
        border-radius: 5px;
        padding: 0.5rem 0.8rem;
        margin-top: 0.7rem;
        font-size: 0.85rem;
        color: {c["danger"]};
        line-height: 1.5;
      }}

      /* circuito placeholder */
      .circuit-placeholder {{
        background: {c["surface2"]};
        border: 2px dashed {c["border"]};
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        color: {c["text_muted"]};
        font-size: 0.92rem;
        line-height: 1.7;
      }}

      /* headings */
      h1 {{ font-size: 1.9rem; font-weight: 700; color: {c["text"]}; }}
      h2 {{ font-size: 1.35rem; font-weight: 600; color: {c["text"]}; margin-top: 1.5rem; }}
      h3 {{ font-size: 1.1rem; font-weight: 600; color: {c["text"]}; }}
    </style>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO C — TELA DE SELECAO DE MAQUINA
# ═════════════════════════════════════════════════════════════════════════════

MACHINES = [
    {"key": "mit",  "name": "Motor de Inducao Trifasico", "icon": "M", "tag": "Disponivel", "disabled": False},
    {"key": "sync", "name": "Gerador Sincrono",            "icon": "G", "tag": "Em breve",   "disabled": True},
    {"key": "dc",   "name": "Motor de Corrente Continua", "icon": "D", "tag": "Em breve",   "disabled": True},
    {"key": "tr",   "name": "Transformador",               "icon": "T", "tag": "Em breve",   "disabled": True},
]


def render_machine_selector() -> None:
    c = get_palette()

    st.markdown(
        '<p class="section-label">Selecao de Equipamento</p>'
        '<h2 style="margin-top:0">Escolha a maquina para simular</h2>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4, gap="medium")
    for i, m in enumerate(MACHINES):
        with cols[i]:
            is_active  = st.session_state.get("selected_machine") == m["key"]
            is_disabled = m["disabled"]
            card_class = "machine-card"
            if is_active:    card_class += " active"
            if is_disabled:  card_class += " disabled"

            tag_class = "card-tag" + (" soon" if is_disabled else "")

            st.markdown(
                f'<div class="{card_class}">'
                f'<span class="card-icon" style="font-family:monospace;font-weight:900;'
                f'font-size:2rem;color:{c["accent"]}">{m["icon"]}</span>'
                f'<div class="card-name">{m["name"]}</div>'
                f'<span class="{tag_class}">{m["tag"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if not is_disabled:
                if st.button("Selecionar", key=f"btn_{m['key']}", use_container_width=True):
                    st.session_state["selected_machine"] = m["key"]
                    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO D — ABA SIMULACAO
# ═════════════════════════════════════════════════════════════════════════════

VARIABLE_CATALOG = {
    "Torque eletromagnetico  Te  (N.m)":       "Te",
    "Velocidade do rotor  n  (RPM)":            "n",
    "Velocidade angular  wr  (rad/s)":          "wr",
    "Corrente de fase A — estator  ias  (A)":   "ias",
    "Corrente de fase B — estator  ibs  (A)":   "ibs",
    "Corrente de fase C — estator  ics  (A)":   "ics",
    "Corrente de fase A — rotor  iar  (A)":     "iar",
    "Corrente de fase B — rotor  ibr  (A)":     "ibr",
    "Corrente de fase C — rotor  icr  (A)":     "icr",
    "Componente  ids  (A)":                     "ids",
    "Componente  iqs  (A)":                     "iqs",
    "Componente  idr  (A)":                     "idr",
    "Componente  iqr  (A)":                     "iqr",
    "Tensao de fase  Va  (V)":                  "Va",
    "Tensao de fase  Vb  (V)":                  "Vb",
    "Tensao de fase  Vc  (V)":                  "Vc",
    "Tensao dq  Vds  (V)":                      "Vds",
    "Tensao dq  Vqs  (V)":                      "Vqs",
}


def _param_group(title: str) -> None:
    st.markdown(f'<div class="param-group-title">{title}</div>', unsafe_allow_html=True)


def _info(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def render_tab_simulacao() -> tuple:
    """
    Renderiza a aba de Simulacao.
    Retorna (mp, ref_code, exp_config, var_keys, var_labels, tmax, h, dark_plot).
    """
    # ── controles globais no topo ────────────────────────────────────────────
    col_toggle, col_plot, _ = st.columns([1, 1, 3])
    with col_toggle:
        st.toggle("Modo Escuro", value=True, key="dark_mode")
    with col_plot:
        dark_plot = st.toggle("Fundo escuro no grafico", value=True, key="plot_dark")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── PARAMETROS DA MAQUINA ────────────────────────────────────────────────
    st.markdown('<p class="section-label">Parametros Fisicos da Maquina</p>', unsafe_allow_html=True)

    col_e, col_m = st.columns(2, gap="large")

    with col_e:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        _param_group("Dados Eletricos")
        Vl  = st.number_input("Tensao de linha RMS — Vl (V)",       min_value=50.0,  max_value=15000.0, value=220.0,  step=1.0)
        f   = st.number_input("Frequencia da rede — f (Hz)",         min_value=1.0,   max_value=400.0,   value=60.0,   step=1.0)
        Rs  = st.number_input("Resistencia do estator — Rs (Ohm)",   min_value=0.001, max_value=100.0,   value=0.435,  step=0.001, format="%.3f")
        Rr  = st.number_input("Resistencia do rotor — Rr (Ohm)",     min_value=0.001, max_value=100.0,   value=0.816,  step=0.001, format="%.3f")
        Xm  = st.number_input("Reatancia de magnetizacao — Xm (Ohm)",min_value=0.1,   max_value=500.0,   value=26.13,  step=0.01,  format="%.2f")
        Xls = st.number_input("Reatancia de dispersao estator — Xls (Ohm)", min_value=0.001, max_value=50.0, value=0.754, step=0.001, format="%.3f")
        Xlr = st.number_input("Reatancia de dispersao rotor — Xlr (Ohm)",   min_value=0.001, max_value=50.0, value=0.754, step=0.001, format="%.3f")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        _param_group("Dados Mecanicos e Referencial")
        p   = st.selectbox("Numero de polos — p", options=[2, 4, 6, 8, 10, 12], index=1)
        J   = st.number_input("Momento de inercia — J (kg.m2)", min_value=0.001, max_value=100.0, value=0.089, step=0.001, format="%.3f")
        B   = st.number_input("Atrito viscoso — B (N.m.s/rad)",  min_value=0.0,   max_value=10.0,  value=0.0,   step=0.001, format="%.3f")
        ref_label = st.selectbox(
            "Referencial da Transformada de Park",
            ["Sincrono  (w_ref = we)", "Rotorico  (w_ref = wr)", "Estacionario  (w_ref = 0)"],
        )
        ref_code = {"Sincrono  (w_ref = we)": 1, "Rotorico  (w_ref = wr)": 2, "Estacionario  (w_ref = 0)": 3}[ref_label]
        st.markdown('</div>', unsafe_allow_html=True)

    mp = MachineParams(Vl=Vl, f=f, Rs=Rs, Rr=Rr, Xm=Xm, Xls=Xls, Xlr=Xlr, p=p, J=J, B=B)

    # grandezas derivadas
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Velocidade Sincrona", f"{mp.n_sync:.1f} RPM")
    mc2.metric("Velocidade Angular Base", f"{mp.wb:.2f} rad/s")
    mc3.metric("Reatancia Mutua Xml", f"{mp.Xml:.4f} Ohm")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── CONFIGURACAO DO EXPERIMENTO ──────────────────────────────────────────
    st.markdown('<p class="section-label">Configuracao do Experimento</p>', unsafe_allow_html=True)

    exp_options = {
        "Partida Direta (DOL)":                   "dol",
        "Partida Estrela-Triangulo (Y-D)":        "yd",
        "Partida com Autotransformador":           "comp",
        "Soft-Starter (Rampa de Tensao)":          "soft",
        "Aplicacao de Carga (partida em vazio)":  "carga",
        "Operacao como Gerador":                   "gerador",
    }
    exp_label = st.selectbox("Tipo de Experimento", list(exp_options.keys()), key="exp_select")
    exp_type  = exp_options[exp_label]
    config    = {"exp_type": exp_type}

    col_exp, col_exp2 = st.columns(2, gap="large")

    with col_exp:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        _param_group("Parametros de Carga e Tensao")

        if exp_type == "dol":
            config["Tl_final"] = st.number_input("Torque de carga (N.m)", value=80.0, min_value=0.0)
            config["t_carga"]  = st.number_input("Instante de aplicacao da carga (s)", value=0.1, min_value=0.0)

        elif exp_type == "yd":
            config["Tl_final"] = st.number_input("Torque de carga (N.m)", value=80.0, min_value=0.0)
            config["t_2"]      = st.number_input("Instante de comutacao Y para D (s)", value=0.5, min_value=0.01)
            config["t_carga"]  = st.number_input("Instante de aplicacao da carga (s)", value=0.1, min_value=0.0)
            _info("A tensao em estrela e reduzida a Vl / raiz(3). A comutacao para triangulo ocorre no instante t_2.")

        elif exp_type == "comp":
            config["Tl_final"]      = st.number_input("Torque de carga (N.m)", value=80.0, min_value=0.0)
            config["voltage_ratio"] = st.slider("Tap do autotransformador (%)", 10, 95, 50) / 100.0
            config["t_2"]           = st.number_input("Instante de comutacao (s)", value=0.5, min_value=0.01)
            config["t_carga"]       = st.number_input("Instante de aplicacao da carga (s)", value=0.1, min_value=0.0)
            _info(f"Tensao inicial = {config['voltage_ratio']*100:.0f}% de Vl nominal.")

        elif exp_type == "soft":
            config["voltage_ratio"] = st.slider("Tensao inicial do soft-starter (%)", 10, 90, 50) / 100.0
            config["t_2"]           = st.number_input("Inicio da rampa de tensao (s)", value=0.9, min_value=0.0)
            config["t_pico"]        = st.number_input("Tempo para atingir tensao nominal (s)", value=5.0, min_value=0.1)
            config["Tl_final"]      = st.number_input("Torque de carga (N.m)", value=80.0, min_value=0.0)
            config["t_carga"]       = st.number_input("Instante de aplicacao da carga (s)", value=0.1, min_value=0.0)

        elif exp_type == "carga":
            Tl_nom = st.number_input("Torque nominal de referencia (N.m)", value=80.0, min_value=0.1)
            pct    = st.slider("Percentual da carga (%)", min_value=1, max_value=300, value=100)
            config["Tl_final"] = Tl_nom * pct / 100.0
            config["t_carga"]  = st.number_input("Instante de aplicacao da carga (s)", value=1.0, min_value=0.0)
            regime = "nominal" if pct == 100 else ("sobrecarga" if pct > 100 else "carga parcial")
            _info(f"Torque aplicado: <strong>{config['Tl_final']:.2f} N.m</strong> ({pct}% de {Tl_nom:.1f} N.m) &mdash; {regime}")

        elif exp_type == "gerador":
            config["Tl_mec"] = st.number_input("Torque mecanico da turbina (N.m)", value=80.0, min_value=1.0)
            config["t_2"]    = st.number_input("Instante de aplicacao do torque (s)", value=1.0, min_value=0.0)
            _info("O torque negativo impulsiona o rotor acima da velocidade sincrona, colocando a maquina em modo gerador.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_exp2:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        _param_group("Grandezas para Visualizacao")
        selected_labels = st.multiselect(
            "Selecione as grandezas a plotar",
            options=list(VARIABLE_CATALOG.keys()),
            default=[
                "Torque eletromagnetico  Te  (N.m)",
                "Velocidade do rotor  n  (RPM)",
                "Corrente de fase A — estator  ias  (A)",
            ],
        )
        var_keys   = [VARIABLE_CATALOG[v] for v in selected_labels]
        var_labels = selected_labels
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── TEMPO E PASSO ────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Parametros Numericos da Simulacao</p>', unsafe_allow_html=True)

    col_t, col_ti = st.columns([1, 1], gap="large")
    with col_t:
        st.markdown('<div class="param-group">', unsafe_allow_html=True)
        _param_group("Tempo e Passo de Integracao")
        tmax = st.number_input("Tempo total de simulacao — tmax (s)", min_value=0.1, max_value=60.0, value=2.0, step=0.1, format="%.1f")
        h    = st.number_input("Passo de integracao — h (s)", min_value=0.00001, max_value=0.01, value=0.001, step=0.0001, format="%.5f")
        n_steps = int(tmax / h)
        st.caption(f"Total de passos: {n_steps:,}")
        if n_steps > 100_000:
            st.warning("Volume elevado de passos. A simulacao pode levar varios segundos.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ti:
        _info(
            "<strong>Trade-off tempo x precisao:</strong><br><br>"
            "<strong>tmax:</strong> quanto maior, mais do transitorio e capturado, mas maior o custo "
            "computacional.<br><br>"
            "<strong>h (passo):</strong> valores acima de 0,005 s podem causar instabilidade numerica e "
            "resultados incorretos. Valores abaixo de 0,0001 s raramente trazem ganho de precisao perceptivel "
            "e aumentam muito o tempo de processamento. O padrao de 0,001 s equilibra estabilidade e velocidade "
            "para a maioria dos experimentos com a MIT."
        )

    # ── BOTAO ────────────────────────────────────────────────────────────────
    st.markdown('<div class="run-btn-wrap">', unsafe_allow_html=True)
    run = st.button("Executar Simulacao", key="btn_run", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    return mp, ref_code, config, var_keys, var_labels, tmax, h, dark_plot, run


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO E — ABA CIRCUITO EQUIVALENTE
# ═════════════════════════════════════════════════════════════════════════════

def _resistor_pts(x0: float, y: float, w: float, n: int = 6):
    """Gera pontos do simbolo de resistor em ziguezague."""
    seg = w / (2 * n + 2)
    xs, ys = [x0, x0 + seg], [y, y]
    for i in range(n):
        xs += [x0 + seg + (2*i+1)*seg, x0 + seg + (2*i+2)*seg]
        ys += [y + 0.16, y - 0.16]
    xs += [x0 + w - seg, x0 + w]
    ys += [y, y]
    return xs, ys


def _inductor_pts(x0: float, y: float, w: float, n: int = 4):
    """Gera pontos do simbolo de indutor em arcos."""
    arc_w = w / n
    t = np.linspace(np.pi, 0, 22)
    xs_all, ys_all = [], []
    for i in range(n):
        cx = x0 + (i + 0.5) * arc_w
        xs_all += list(cx + (arc_w/2) * np.cos(t))
        ys_all += list(y    + (arc_w/2) * 0.65 * np.sin(t))
        if i < n - 1:
            xs_all.append(None); ys_all.append(None)
    return xs_all, ys_all


def _inductor_vert_pts(x: float, y0: float, h: float, n: int = 3):
    """Gera pontos do simbolo de indutor vertical."""
    arc_h = h / n
    t = np.linspace(np.pi, 0, 22)
    xs_all, ys_all = [], []
    for i in range(n):
        cy = y0 - (i + 0.5) * arc_h
        xs_all += list(x + (arc_h/2) * 0.65 * np.sin(t))
        ys_all += list(cy + (arc_h/2) * np.cos(t))
        if i < n - 1:
            xs_all.append(None); ys_all.append(None)
    return xs_all, ys_all


def render_tab_circuito(mp: MachineParams) -> None:
    """
    Aba Circuito Equivalente.
    Desenha o circuito monofasico equivalente em T da MIT usando Plotly.
    Topologia:
        Vs  --[Rs]--[jXls]--+--[jXlr]--[Rr/s]--
                             |
                           [jXm]
                             |
                            GND
    """
    c   = get_palette()
    dark = st.session_state.get("dark_mode", True)

    bg_plot  = "#0d1117" if dark else "#ffffff"
    fg       = c["text"]
    wire_col = c["accent"]
    comp_col = "#f97316"
    src_col  = "#a78bfa"
    lw = 2.2

    fig = go.Figure()

    def wire(x0, y0, x1, y1):
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=wire_col, width=lw))

    def ann(x, y, txt, size=11, color=fg, anchor="center"):
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(size=size, color=color,
                                     family="Inter, 'Courier New', monospace"),
                           xanchor=anchor, yanchor="middle", bgcolor="rgba(0,0,0,0)")

    def comp_trace(xs, ys):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(color=comp_col, width=lw),
                                 showlegend=False, hoverinfo="skip"))

    # ── coordenadas ────────────────────────────────────────────────────────
    Y  = 2.0      # fio superior
    G  = 0.0      # GND
    Ym = 1.0      # meio

    # posicoes horizontais
    X0  = 0.3    # centro da fonte
    X1  = 0.75   # saida da fonte
    XR0 = 0.75   # inicio Rs
    XR1 = 2.0    # fim Rs
    XL0 = 2.0    # inicio Xls
    XL1 = 3.3    # fim Xls
    XN  = 3.7    # no
    XLR0= 4.1    # inicio Xlr
    XLR1= 5.4    # fim Xlr
    XRR0= 5.4    # inicio Rr/s
    XRR1= 6.8    # fim Rr/s
    XE  = 7.1    # extremo direito

    # ── fios horizontais ────────────────────────────────────────────────────
    wire(X1, Y, XR0, Y)         # saida fonte -> Rs
    wire(XL1, Y, XN, Y)         # Xls -> no
    wire(XN, Y, XLR0, Y)        # no -> Xlr
    wire(XLR1, Y, XRR0, Y)      # Xlr -> Rr/s
    wire(XRR1, Y, XE, Y)        # Rr/s -> fim
    wire(X0 - 0.2, G, XE, G)    # GND completo

    # ── fechamentos verticais ───────────────────────────────────────────────
    wire(XE, Y, XE, G)          # lado direito

    # ── fonte de tensao ─────────────────────────────────────────────────────
    fig.add_shape(type="circle",
                  x0=X0-0.22, y0=G+0.06, x1=X0+0.22, y1=Y-0.06,
                  line=dict(color=src_col, width=lw+0.5),
                  fillcolor=bg_plot)
    wire(X0-0.22, Ym, X0-0.22, Ym)  # ponto ancoragem esq
    wire(X0+0.22, Y,  X1, Y)         # fio saida superior
    wire(X0-0.22, G+0.06, X0-0.22, G)  # fio base
    ann(X0, Ym,       "Vs",  12, src_col)
    ann(X0-0.5, Ym+0.3, "+", 14, src_col)
    ann(X0-0.5, Ym-0.3, "\u2212", 14, src_col)

    # ── Rs ──────────────────────────────────────────────────────────────────
    xs, ys = _resistor_pts(XR0, Y, XR1-XR0)
    comp_trace(xs, ys)
    ann((XR0+XR1)/2, Y+0.33, "Rs",                 10, comp_col)
    ann((XR0+XR1)/2, Y-0.33, f"{mp.Rs:.3f} \u03a9", 9,  fg)

    # ── jXls ────────────────────────────────────────────────────────────────
    xs, ys = _inductor_pts(XL0, Y, XL1-XL0)
    comp_trace(xs, ys)
    ann((XL0+XL1)/2, Y+0.37, "jX<sub>ls</sub>",   10, comp_col)
    ann((XL0+XL1)/2, Y-0.33, f"{mp.Xls:.3f} \u03a9", 9, fg)

    # ── ramo jXm (vertical) ─────────────────────────────────────────────────
    wire(XN, Y,    XN, Y-0.2)
    wire(XN, G+0.2, XN, G)
    xsv, ysv = _inductor_vert_pts(XN, Y-0.2, 1.2)
    comp_trace(xsv, ysv)
    ann(XN+0.42, Ym+0.15, "jX<sub>m</sub>",    10, comp_col, "left")
    ann(XN+0.42, Ym-0.18, f"{mp.Xm:.2f} \u03a9", 9, fg, "left")

    # ── jXlr ────────────────────────────────────────────────────────────────
    xs, ys = _inductor_pts(XLR0, Y, XLR1-XLR0)
    comp_trace(xs, ys)
    ann((XLR0+XLR1)/2, Y+0.37, "jX<sub>lr</sub>",    10, comp_col)
    ann((XLR0+XLR1)/2, Y-0.33, f"{mp.Xlr:.3f} \u03a9", 9, fg)

    # ── Rr/s ────────────────────────────────────────────────────────────────
    xs, ys = _resistor_pts(XRR0, Y, XRR1-XRR0)
    comp_trace(xs, ys)
    ann((XRR0+XRR1)/2, Y+0.33, "R<sub>r</sub> / s",   10, comp_col)
    ann((XRR0+XRR1)/2, Y-0.33, f"{mp.Rr:.3f} \u03a9 / s", 9, fg)

    # ── nota de escorregamento ───────────────────────────────────────────────
    ann((XRR0+XRR1)/2, G-0.45,
        "s = (n<sub>s</sub> \u2212 n) / n<sub>s</sub>  (escorregamento)", 9, c["text_muted"])

    fig.update_layout(
        height=300,
        margin=dict(l=30, r=20, t=20, b=50),
        paper_bgcolor=bg_plot,
        plot_bgcolor=bg_plot,
        xaxis=dict(visible=False, range=[-0.2, 7.6]),
        yaxis=dict(visible=False, range=[-0.7, 2.65]),
        showlegend=False,
        hovermode=False,
    )

    st.markdown('<p class="section-label">Circuito Equivalente Monofasico em T — MIT (Gaiola de Esquilo)</p>',
                unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── tabela de impedancias ────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Impedancias e Grandezas Derivadas</p>', unsafe_allow_html=True)

    Zs  = complex(mp.Rs, mp.Xls)
    Zm  = complex(0, mp.Xm)
    Zr  = complex(mp.Rr, mp.Xlr)

    ci1, ci2, ci3, ci4 = st.columns(4)
    ci1.metric("Impedancia do Estator |Zs|", f"{abs(Zs):.4f} Ohm")
    ci2.metric("Reatancia de Magnetizacao Xm", f"{mp.Xm:.2f} Ohm")
    ci3.metric("Impedancia do Rotor |Zr| (s=1)", f"{abs(Zr):.4f} Ohm")
    ci4.metric("Reatancia Mutua Xml", f"{mp.Xml:.4f} Ohm")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-label">Legenda dos Elementos</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Rs** — Resistencia do enrolamento do estator (perdas Joule no cobre do estator).

        **jXls** — Reatancia de dispersao do estator (fluxo que nao atravessa o entreferro).

        **jXm** — Reatancia de magnetizacao (representa o caminho do fluxo principal pelo nucleo).
        """)
    with c2:
        st.markdown("""
        **jXlr** — Reatancia de dispersao do rotor referida ao estator.

        **Rr/s** — Resistencia do rotor dividida pelo escorregamento. Quando s = 1 (partida), Rr/s = Rr.
        Quando s se aproxima de zero (velocidade sincrona), Rr/s tende ao infinito (circuito aberto).

        **Vs** — Tensao de fase aplicada ao estator (Vl / raiz(3)).
        """)


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO F — ABA RESULTADOS
# ═════════════════════════════════════════════════════════════════════════════

def _line_colors(dark: bool) -> list:
    return [
        "#4f8ef7","#f97316","#22c55e","#a78bfa",
        "#ec4899","#14b8a6","#f59e0b","#6366f1",
        "#84cc16","#ef4444","#06b6d4","#d946ef",
    ] if dark else [
        "#1d4ed8","#ea580c","#16a34a","#7c3aed",
        "#db2777","#0d9488","#d97706","#4f46e5",
        "#65a30d","#dc2626","#0891b2","#c026d3",
    ]


def build_plotly_figure(res: dict, var_keys: list, var_labels: list,
                         dark: bool, t_events: list = None) -> go.Figure:
    n_vars = len(var_keys)
    if n_vars == 0:
        return go.Figure()

    plot_bg  = "#0d1117" if dark else "#ffffff"
    paper_bg = "#161b27" if dark else "#f8faff"
    fg       = "#e4e8f5" if dark else "#111827"
    grid_col = "#2a3150" if dark else "#e5eaf5"
    colors   = _line_colors(dark)

    fig = make_subplots(
        rows=n_vars, cols=1,
        shared_xaxes=True,
        subplot_titles=var_labels,
        vertical_spacing=max(0.03, 0.08 / n_vars),
    )
    t = res["t"]
    for i, (key, lbl) in enumerate(zip(var_keys, var_labels), 1):
        col = colors[(i-1) % len(colors)]
        fig.add_trace(go.Scatter(
            x=t, y=res[key], mode="lines", name=lbl,
            line=dict(color=col, width=1.9),
            hovertemplate=f"<b>{lbl}</b><br>t = %{{x:.4f}} s<br>valor = %{{y:.4f}}<extra></extra>",
        ), row=i, col=1)
        if t_events:
            for te in t_events:
                fig.add_vline(x=te, line_dash="dot", line_color="#64748b",
                              line_width=1.2, row=i, col=1)
        fig.update_yaxes(row=i, col=1,
                         showgrid=True, gridcolor=grid_col, gridwidth=0.4,
                         zeroline=True, zerolinecolor=grid_col,
                         tickfont=dict(size=10, color=fg))
    fig.update_xaxes(row=n_vars, col=1,
                     showgrid=True, gridcolor=grid_col, gridwidth=0.4,
                     tickfont=dict(size=10, color=fg), title_text="Tempo (s)")
    for ann in fig.layout.annotations:
        ann.font.color = fg
        ann.font.size  = 11

    fig.update_layout(
        height=max(280, 210 * n_vars),
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family="Inter, system-ui", size=11, color=fg),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=55, r=20, t=55, b=40),
        hovermode="x unified",
    )
    return fig


def render_tab_resultados(res: dict, var_keys: list, var_labels: list,
                           dark: bool, t_events: list, exp_config: dict) -> None:
    c = get_palette()

    # ── KPIs ────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Indicadores de Regime Permanente</p>',
                unsafe_allow_html=True)

    n_ss   = res["n"][-1]
    Te_ss  = res["Te"][-1]
    ias_ss = res["ias"][-1]
    wr_ss  = res["wr"][-1]
    Te_max = float(np.max(res["Te"]))
    ias_pk = float(np.max(np.abs(res["ias"])))

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Velocidade de Regime (RPM)",   f"{n_ss:.1f}")
    k2.metric("Torque de Regime  Te  (N.m)",  f"{Te_ss:.2f}")
    k3.metric("Torque Maximo  Te_max  (N.m)", f"{Te_max:.2f}")
    k4.metric("Corrente de Pico  ias  (A)",   f"{ias_pk:.2f}")
    k5.metric("Corrente Regime  ias  (A)",    f"{abs(ias_ss):.2f}")
    k6.metric("Velocidade Angular  wr  (rad/s)", f"{wr_ss:.3f}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── controle do fundo do grafico ─────────────────────────────────────────
    col_tog, _ = st.columns([1, 4])
    with col_tog:
        dark_plot = st.toggle("Fundo escuro no grafico", value=dark, key="res_plot_dark")

    # ── graficos ─────────────────────────────────────────────────────────────
    if var_keys:
        fig = build_plotly_figure(res, var_keys, var_labels, dark_plot, t_events)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhuma grandeza selecionada. Volte a aba Simulacao e escolha variaveis para plotar.")


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO G — ABA TEORIA
# ═════════════════════════════════════════════════════════════════════════════

THEORY_PARAMS = [
    {
        "group": "Parametros Eletricos",
        "items": [
            {
                "nome": "Vl — Tensao de Linha RMS",
                "desc": "Define a amplitude do campo magnetico girante no estator. "
                        "E a grandeza que estabelece o ponto de operacao magnetico da maquina.",
                "up":   "O torque maximo cresce com o quadrado da tensao. "
                        "A corrente de partida tambem aumenta proporcionalmente.",
                "down": "O torque de partida cai. Pode ser insuficiente para vencer a carga estatica, "
                        "impedindo o motor de partir.",
                "warn": "Sobretensao causa saturacao do nucleo e aquecimento do isolamento. "
                        "Subtensao severa pode causar parada brusca (stall) com o motor em carga.",
            },
            {
                "nome": "f — Frequencia da Rede",
                "desc": "Determina a velocidade sincrona do campo girante (ns = 120f/p). "
                        "Todas as reatancias do circuito equivalente sao proporcionais a f.",
                "up":   "Velocidade sincrona maior e reatancias maiores (Xm, Xls, Xlr crescem). "
                        "O torque disponivel se redistribui ao longo da curva T x n.",
                "down": "Velocidade menor. Com tensao constante, o fluxo aumenta (V/f sobe), "
                        "podendo saturar o nucleo.",
                "warn": "Operacao fora da frequencia nominal sem controle V/f proporcional "
                        "compromete o fluxo de entreferro e a eficiencia.",
            },
            {
                "nome": "Rs — Resistencia do Estator",
                "desc": "Representa as perdas Joule no cobre do enrolamento estatorico. "
                        "Provoca queda de tensao interna, reduzindo a tensao efetiva no entreferro.",
                "up":   "Maior dissipacao de energia, menor tensao no entreferro e reducao do torque. "
                        "O motor aquece mais em regime.",
                "down": "Menor queda interna e melhor eficiencia. "
                        "O modelo se aproxima de um transformador ideal.",
                "warn": "Rs muito elevado (enrolamento danificado) pode causar sobreaquecimento fatal. "
                        "Rs proximo de zero gera instabilidade numerica nas EDOs com passos grandes.",
            },
            {
                "nome": "Rr — Resistencia do Rotor",
                "desc": "Parametro central da curva de torque. Controla o escorregamento de regime "
                        "e o torque de partida. Nas barras da gaiola, e inversamente proporcional "
                        "a condutividade do material.",
                "up":   "O escorregamento de regime aumenta (o rotor gira mais devagar em relacao ao campo). "
                        "O torque de partida aumenta ate um ponto otimo e depois decresce. "
                        "A curva T x n se torna mais plana e larga.",
                "down": "Menor escorregamento em regime e melhor eficiencia. "
                        "Menor torque de partida. A curva T x n fica mais estreita e alta proximo a ns.",
                "warn": "Rr muito alto (barras da gaiola fraturadas) provoca escorregamento excessivo "
                        "e sobreaquecimento do rotor. "
                        "Rr proximo de zero causa instabilidade numerica (singularidade nas equacoes do rotor).",
            },
            {
                "nome": "Xm — Reatancia de Magnetizacao",
                "desc": "Representa o ramo shunt do circuito equivalente: o caminho do fluxo magnetico "
                        "pelo nucleo de ferro. Quanto maior Xm, menos corrente e desviada para magnetizar o nucleo.",
                "up":   "Menor corrente de magnetizacao, melhor fator de potencia e maior eficiencia. "
                        "O motor se comporta de forma mais proxima ao ideal.",
                "down": "Maior corrente de magnetizacao (circula mesmo em vazio), "
                        "pior fator de potencia e maior dissipacao.",
                "warn": "Xm muito baixo representa um nucleo de ma qualidade magnetica ou saturado. "
                        "Na simulacao, Xm proximo de zero torna Xml desprezivel, "
                        "causando divisao por valores muito pequenos e divergencia numerica.",
            },
            {
                "nome": "Xls e Xlr — Reatancias de Dispersao",
                "desc": "Representam os fluxos que nao cruzam o entreferro (fluxos de dispersao "
                        "do estator e do rotor). Limitam a corrente de curto-circuito e a capacidade "
                        "de transferencia de torque.",
                "up":   "Maior impedancia total. A corrente de partida e reduzida, "
                        "mas o torque maximo tambem cai.",
                "down": "Correntes de partida mais elevadas e torque maximo maior, "
                        "porem o motor fica mais sensivel a transitorios e variacoes de carga.",
                "warn": "Dispersao muito baixa leva a correntes de partida muito altas, "
                        "podendo danificar o isolamento. "
                        "Dispersao muito alta limita o torque a ponto de impedir a partida sob carga.",
            },
        ],
    },
    {
        "group": "Parametros Mecanicos",
        "items": [
            {
                "nome": "p — Numero de Polos",
                "desc": "Define a velocidade sincrona (ns = 120f/p) e, portanto, "
                        "a faixa de velocidade de operacao da maquina.",
                "up":   "Velocidade sincrona menor — a maquina opera em rotacoes mais baixas. "
                        "Para a mesma potencia, o torque necessario e maior.",
                "down": "Velocidade sincrona maior. O torque nominal e menor para a mesma potencia.",
                "warn": "O numero de polos e sempre par e discreto. "
                        "Valores impares ou muito altos com frequencia baixa podem gerar "
                        "velocidades de operacao fisicamente irrealistas no modelo.",
            },
            {
                "nome": "J — Momento de Inercia",
                "desc": "Representa a resistencia do conjunto rotor-carga a variacoes de velocidade "
                        "(segunda lei de Newton rotacional: Te - Tl = J . d(wr)/dt).",
                "up":   "Aceleracao mais lenta e transitorio mais prolongado. "
                        "O sistema absorve e libera energia cinetica de forma mais gradual.",
                "down": "Aceleracao muito rapida. O rotor responde quase instantaneamente "
                        "a qualquer variacao de torque.",
                "warn": "J muito baixo pode causar oscilacoes de velocidade em sistemas com variacao "
                        "de carga. J muito alto pode fazer o transitorio ultrapassar o tempo de simulacao "
                        "sem atingir o regime permanente.",
            },
            {
                "nome": "B — Coeficiente de Atrito Viscoso",
                "desc": "Modela as perdas mecanicas proporcionais a velocidade: "
                        "mancais, ventilacao forcada, resistencia do fluido. "
                        "Produz um torque de freio igual a B.wr.",
                "up":   "Maior dissipacao mecanica e velocidade de regime ligeiramente menor. "
                        "O sistema amorte naturalmente transitorios de velocidade.",
                "down": "Menor atrito. Com B = 0 (padrao), toda resistencia ao movimento "
                        "vem exclusivamente da carga mecanica.",
                "warn": "B muito alto pode paralisar o motor mesmo sem carga nominal, "
                        "pois as perdas mecanicas superam o torque eletromagnetico disponivel. "
                        "Na pratica, atrito excessivo indica falha em mancais ou rolamentos.",
            },
        ],
    },
    {
        "group": "Parametros de Simulacao",
        "items": [
            {
                "nome": "tmax — Tempo Total de Simulacao",
                "desc": "Horizonte temporal da integracao numerica. "
                        "Deve ser longo o suficiente para capturar o regime permanente de interesse.",
                "up":   "Mais do transitorio e do regime sao observados, "
                        "mas o tempo de processamento cresce proporcionalmente.",
                "down": "Simulacao mais rapida, mas pode encerrar antes de o sistema atingir o regime.",
                "warn": "tmax muito grande combinado com passo pequeno pode consumir muita memoria "
                        "e travar o navegador.",
            },
            {
                "nome": "h — Passo de Integracao",
                "desc": "Controla a discretizacao temporal do metodo Runge-Kutta. "
                        "E o parametro mais critico para estabilidade e precisao numerica.",
                "up":   "Simulacao mais rapida, mas menos precisa. "
                        "Passos muito grandes causam instabilidade numerica "
                        "(oscilacoes artificiais ou divergencia).",
                "down": "Maior precisao e estabilidade, mas custo computacional muito maior. "
                        "Abaixo de certo limiar, o ganho de precisao e desprezivel.",
                "warn": "h acima de 0,005 s pode gerar resultados fisicamente incorretos "
                        "para os parametros tipicos da MIT. "
                        "A regra pratica e h menor ou igual a 1/(10.wb) para garantir estabilidade.",
            },
        ],
    },
]


def render_tab_teoria() -> None:
    st.markdown(
        "Nesta aba os parametros sao descritos em termos de seu significado fisico e do "
        "impacto qualitativo que provocam no comportamento da maquina. "
        "Nenhum valor numerico especifico e apresentado: o objetivo e construir intuicao "
        "sobre o sistema eletrico.",
    )
    for group in THEORY_PARAMS:
        st.markdown(f"## {group['group']}")
        for item in group["items"]:
            st.markdown(
                f'<div class="theory-card">'
                f'<h4>{item["nome"]}</h4>'
                f'<p>{item["desc"]}</p>'
                f'<p><span class="tc-up">Se aumentar:</span> {item["up"]}</p>'
                f'<p><span class="tc-down">Se diminuir:</span> {item["down"]}</p>'
                f'<div class="tc-warn">Atencao — calibracoes extremas: {item["warn"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# BLOCO H — ORQUESTRADOR PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # inicializa estado
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True
    if "sim_result" not in st.session_state:
        st.session_state["sim_result"] = None

    # CSS sempre primeiro
    apply_css()
    c = get_palette()

    # ── cabecalho ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="app-header">'
        '<div>'
        '<div class="app-title">Simulador de Maquinas Eletricas</div>'
        '<div class="app-subtitle">Modelo 0dq de Krause &mdash; Integracao Numerica RK4 (scipy.odeint)</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── selecao de maquina ────────────────────────────────────────────────────
    selected = st.session_state.get("selected_machine")

    if not selected:
        render_machine_selector()
        return

    # ── breadcrumb / voltar ───────────────────────────────────────────────────
    col_back, col_title = st.columns([1, 8])
    with col_back:
        if st.button("Voltar", key="btn_back"):
            st.session_state["selected_machine"] = None
            st.session_state["sim_result"]        = None
            st.rerun()
    with col_title:
        machine_name = next(m["name"] for m in MACHINES if m["key"] == selected)
        st.markdown(f"### {machine_name}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── abas da MIT ───────────────────────────────────────────────────────────
    tab_sim, tab_circ, tab_res, tab_teoria = st.tabs(
        ["Simulacao", "Circuito Equivalente", "Resultados", "Teoria"]
    )

    with tab_sim:
        (mp, ref_code, exp_config,
         var_keys, var_labels,
         tmax, h, dark_plot, run_clicked) = render_tab_simulacao()

        if run_clicked:
            if not var_keys:
                st.warning("Selecione ao menos uma grandeza para plotar antes de executar.")
            else:
                vfn, tfn, t_events = build_voltage_and_torque_fns(exp_config, mp)
                with st.spinner("Executando integracao numerica..."):
                    try:
                        res = run_simulation(mp=mp, tmax=tmax, h=h,
                                             voltage_fn=vfn, torque_fn=tfn,
                                             ref_code=ref_code)
                        st.session_state["sim_result"] = {
                            "res": res, "var_keys": var_keys, "var_labels": var_labels,
                            "t_events": t_events, "dark": dark_plot, "mp": mp,
                            "exp_config": exp_config,
                        }
                        st.success(
                            f"Simulacao concluida. "
                            f"n = {res['n'][-1]:.1f} RPM | "
                            f"Te = {res['Te'][-1]:.2f} N.m"
                        )
                    except Exception as e:
                        st.error(f"Erro na simulacao: {e}")
                        st.info(
                            "Verifique os parametros. Passos de integracao muito grandes "
                            "ou parametros fisicamente invalidos podem causar divergencia numerica."
                        )

    with tab_circ:
        # usa o mp do ultimo resultado se disponivel, senao cria com defaults
        if st.session_state["sim_result"] is not None:
            mp_circ = st.session_state["sim_result"]["mp"]
        else:
            mp_circ = MachineParams()
        render_tab_circuito(mp_circ)

    with tab_res:
        sr = st.session_state.get("sim_result")
        if sr is None:
            st.info("Nenhuma simulacao executada ainda. Configure os parametros na aba Simulacao e clique em Executar.")
        else:
            render_tab_resultados(
                res=sr["res"],
                var_keys=sr["var_keys"],
                var_labels=sr["var_labels"],
                dark=sr["dark"],
                t_events=sr["t_events"],
                exp_config=sr["exp_config"],
            )

    with tab_teoria:
        render_tab_teoria()


if __name__ == "__main__":
    main()
