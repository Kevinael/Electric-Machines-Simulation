# -*- coding: utf-8 -*-
"""
ems_streamlit.py — Simulador de Máquinas Elétricas
Modelo 0dq de Krause — Integração RK4 (scipy.odeint)

Blocos:
  A — Modelo matemático (núcleo físico, inalterado)
  B — Tema e CSS dinâmico
  C — Tela inicial (seleção de máquina)
  D — Layout principal do MIT (parâmetros + circuito + experimento)
  E — Circuito equivalente (Plotly)
  F — Resultados (KPIs + gráficos Plotly)
  G — Aba Teoria
  H — Orquestrador principal
"""

from __future__ import annotations
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from dataclasses import dataclass, field
import schemdraw
import schemdraw.elements as elm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador de Máquinas Elétricas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
# BLOCO A — MODELO MATEMÁTICO (lógica original preservada integralmente)
# ═══════════════════════════════════════════════════════════════════════════

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
    dPSIqs = mp.wb * (Vqs - (w_ref/mp.wb)*PSIds + (mp.Rs/mp.Xls)*(PSImq - PSIqs))
    dPSIds = mp.wb * (Vds + (w_ref/mp.wb)*PSIqs + (mp.Rs/mp.Xls)*(PSImd - PSIds))
    slip_ref = (w_ref - wr) / mp.wb
    dPSIqr = mp.wb * (-slip_ref*PSIdr + (mp.Rr/mp.Xlr)*(PSImq - PSIqr))
    dPSIdr = mp.wb * ( slip_ref*PSIqr + (mp.Rr/mp.Xlr)*(PSImd - PSIdr))
    Te  = (3.0/2.0)*(mp.p/2.0)*(1.0/mp.wb)*(PSIds*iqs - PSIqs*ids)
    dwr = (mp.p/(2.0*mp.J))*(Te - Tl) - (mp.B/mp.J)*wr
    return [dPSIqs, dPSIds, dPSIqr, dPSIdr, dwr]


def abc_voltages(t, Vl, f):
    tetae = 2.0*np.pi*f*t
    Va = np.sqrt(2.0)*Vl*np.sin(tetae)
    Vb = np.sqrt(2.0)*Vl*np.sin(tetae - 2.0*np.pi/3.0)
    Vc = np.sqrt(2.0)*Vl*np.sin(tetae + 2.0*np.pi/3.0)
    return Va, Vb, Vc


def clarke_park_transform(Va, Vb, Vc, tetae):
    k = np.sqrt(3.0/2.0)
    Va_ = k*(Va - 0.5*Vb - 0.5*Vc)
    Vb_ = k*((np.sqrt(3.0)/2.0)*Vb - (np.sqrt(3.0)/2.0)*Vc)
    Vds = np.cos(tetae)*Va_ + np.sin(tetae)*Vb_
    Vqs = -np.sin(tetae)*Va_ + np.cos(tetae)*Vb_
    return Vds, Vqs


def reconstruct_abc_currents(PSIqs, PSIds, PSIqr, PSIdr, tetae, tetar, mp):
    PSImq = mp.Xml*(PSIqs/mp.Xls + PSIqr/mp.Xlr)
    PSImd = mp.Xml*(PSIds/mp.Xls + PSIdr/mp.Xlr)
    ids = (1.0/mp.Xls)*(PSIds - PSImd)
    iqs = (1.0/mp.Xls)*(PSIqs - PSImq)
    idr = (1.0/mp.Xlr)*(PSIdr - PSImd)
    iqr = (1.0/mp.Xlr)*(PSIqr - PSImq)
    ia  = np.cos(tetae)*ids - np.sin(tetae)*iqs
    ib  = np.sin(tetae)*ids + np.cos(tetae)*iqs
    iar_a = np.cos(tetar)*idr - np.sin(tetar)*iqr
    iar_b = np.sin(tetar)*idr + np.cos(tetar)*iqr
    k = np.sqrt(3.0/2.0)
    ias = k*ia
    ibs = k*(-0.5*ia + (np.sqrt(3.0)/2.0)*ib)
    ics = k*(-0.5*ia - (np.sqrt(3.0)/2.0)*ib)
    iar = k*iar_a
    ibr = k*(-0.5*iar_a + (np.sqrt(3.0)/2.0)*iar_b)
    icr = k*(-0.5*iar_a - (np.sqrt(3.0)/2.0)*iar_b)
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
    keys = ["wr","n","Te","ids","iqs","idr","iqr",
            "ias","ibs","ics","iar","ibr","icr","Va","Vb","Vc","Vds","Vqs"]
    arr = {k: np.empty(N) for k in keys}
    states, last_wr, we = [0.0]*5, 0.0, mp.wb
    for i, tv in enumerate(t_values):
        Vl_a   = voltage_fn(tv)
        Tl_a   = torque_fn(tv)
        tetae  = we * tv
        w_ref  = we if ref_code == 1 else (last_wr if ref_code == 2 else 0.0)
        Va, Vb, Vc = abc_voltages(tv, Vl_a, mp.f)
        Vds, Vqs   = clarke_park_transform(Va, Vb, Vc, tetae)
        sol    = odeint(induction_motor_ode, states, [tv, tv+h],
                        args=(Vqs, Vds, Tl_a, w_ref, mp))
        states = list(sol[1])
        PSIqs, PSIds, PSIqr, PSIdr, wr = states
        last_wr  = wr
        ids, iqs, idr, iqr, ias, ibs, ics, iar, ibr, icr = reconstruct_abc_currents(
            PSIqs, PSIds, PSIqr, PSIdr, tetae, wr*tv, mp)
        Te = (3.0/2.0)*(mp.p/2.0)*(1.0/mp.wb)*(PSIds*iqs - PSIqs*ids)
        arr["wr"][i]=wr;  arr["n"][i]=(120.0/mp.p)*(wr/(2.0*np.pi))
        arr["Te"][i]=Te;  arr["ids"][i]=ids; arr["iqs"][i]=iqs
        arr["idr"][i]=idr; arr["iqr"][i]=iqr
        arr["ias"][i]=ias; arr["ibs"][i]=ibs; arr["ics"][i]=ics
        arr["iar"][i]=iar; arr["ibr"][i]=ibr; arr["icr"][i]=icr
        arr["Va"][i]=Va;  arr["Vb"][i]=Vb;  arr["Vc"][i]=Vc
        arr["Vds"][i]=Vds; arr["Vqs"][i]=Vqs
    arr["t"] = t_values
    return arr


def build_fns(config: dict, mp: MachineParams):
    """Constrói as funções de tensão e torque para o experimento selecionado."""
    exp = config["exp_type"]
    t_ev = []
    if exp == "dol":
        Tl, tc = config["Tl_final"], config["t_carga"]
        vfn = lambda t: mp.Vl
        tfn = lambda t: torque_step(t, 0.0, Tl, tc)
        t_ev = [tc]
    elif exp == "yd":
        Vy = mp.Vl/np.sqrt(3.0); Tl=config["Tl_final"]; t2=config["t_2"]; tc=config["t_carga"]
        vfn = lambda t: voltage_reduced_start(t, mp.Vl, Vy, t2)
        tfn = lambda t: torque_step(t, 0.0, Tl, tc)
        t_ev = [t2, tc]
    elif exp == "comp":
        Vr=mp.Vl*config["voltage_ratio"]; Tl=config["Tl_final"]; t2=config["t_2"]; tc=config["t_carga"]
        vfn = lambda t: voltage_reduced_start(t, mp.Vl, Vr, t2)
        tfn = lambda t: torque_step(t, 0.0, Tl, tc)
        t_ev = [t2, tc]
    elif exp == "soft":
        Vi=mp.Vl*config["voltage_ratio"]; t2=config["t_2"]; tp=config["t_pico"]
        Tl=config["Tl_final"]; tc=config["t_carga"]
        vfn = lambda t: voltage_soft_starter(t, mp.Vl, Vi, t2, tp)
        tfn = lambda t: torque_step(t, 0.0, Tl, tc)
        t_ev = [t2, tp, tc]
    elif exp == "carga":
        Tl=config["Tl_final"]; tc=config["t_carga"]
        vfn = lambda t: mp.Vl
        tfn = lambda t: torque_step(t, 0.0, Tl, tc)
        t_ev = [tc]
    elif exp == "gerador":
        Tn=-config["Tl_mec"]
        vfn = lambda t: mp.Vl
        tfn = lambda t: Tn
        t_ev = [config["t_2"]]
    else:
        vfn = lambda t: mp.Vl
        tfn = lambda t: 0.0
    return vfn, tfn, t_ev


# ═══════════════════════════════════════════════════════════════════════════
# BLOCO B — TEMA E CSS DINÂMICO
# ═══════════════════════════════════════════════════════════════════════════

def _palette(dark: bool) -> dict:
    if dark:
        return dict(
            bg="#0d1117", surface="#161b27", surface2="#1e2538",
            border="#2a3150", accent="#4f8ef7", accent2="#7c6af7",
            text="#e4e8f5", muted="#8892b0",
            success="#22c55e", danger="#ef4444",
            warn_bg="rgba(239,68,68,0.08)", input_bg="#1a2035",
            tag="#1e2d4a",
        )
    return dict(
        bg="#f4f7ff", surface="#ffffff", surface2="#eef2ff",
        border="#d0d8f0", accent="#2563eb", accent2="#7c3aed",
        text="#111827", muted="#4b5563",
        success="#16a34a", danger="#dc2626",
        warn_bg="rgba(220,38,38,0.06)", input_bg="#eef2ff",
        tag="#dbeafe",
    )


def apply_css(dark: bool) -> None:
    c = _palette(dark)
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="block-container"] {{
        background-color: {c["bg"]} !important;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    }}
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {{ display: none !important; }}

    /* ── cabecalho da aplicação ── */
    .app-header {{
        padding: 1.4rem 0 1rem 0;
        border-bottom: 1px solid {c["border"]};
        margin-bottom: 1.8rem;
    }}
    .app-title {{
        font-size: 1.5rem; font-weight: 700;
        color: {c["text"]}; letter-spacing: -0.02em;
    }}

    /* ── cartões de seleção de máquina ── */
    .machine-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
        margin: 1.6rem 0;
    }}
    .mcard {{
        background: {c["surface"]};
        border: 2px solid {c["border"]};
        border-radius: 14px;
        padding: 2rem 1.2rem 1.4rem 1.2rem;
        text-align: center;
        transition: border-color .18s, box-shadow .18s;
    }}
    .mcard.active {{
        border-color: {c["accent"]};
        background: {c["tag"]};
        box-shadow: 0 0 0 3px {c["accent"]}33;
    }}
    .mcard.disabled {{ opacity: .35; }}
    .mcard-icon {{
        font-family: 'Courier New', monospace;
        font-size: 2.2rem; font-weight: 900;
        color: {c["accent"]}; display: block;
        margin-bottom: .7rem;
    }}
    .mcard-name {{
        font-size: .98rem; font-weight: 700;
        color: {c["text"]}; margin-bottom: .5rem;
    }}
    .mcard-tag {{
        display: inline-block; font-size: .7rem;
        font-weight: 700; letter-spacing: .06em;
        text-transform: uppercase;
        background: {c["tag"]};
        color: {c["accent"]};
        border-radius: 4px; padding: .15rem .55rem;
    }}
    .mcard-tag.soon {{ background:{c["surface2"]}; color:{c["muted"]}; }}

    /* ── rótulo de seção ── */
    .slabel {{
        font-size: .72rem; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase;
        color: {c["accent"]}; margin-bottom: .4rem;
    }}

    /* ── grupo de parâmetros ── */
    .pgroup {{
        background: {c["surface"]};
        border: 1px solid {c["border"]};
        border-radius: 12px;
        padding: 1.2rem 1.4rem 1rem;
        margin-bottom: 1rem;
    }}
    .pgroup-title {{
        font-size: .76rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
        color: {c["accent"]};
        padding-bottom: .5rem;
        border-bottom: 1px solid {c["border"]};
        margin-bottom: .85rem;
    }}

    /* ── caixa informativa ── */
    .ibox {{
        background: {c["input_bg"]};
        border-left: 3px solid {c["accent"]};
        border-radius: 6px;
        padding: .75rem 1rem;
        font-size: .875rem;
        color: {c["muted"]};
        line-height: 1.65;
        margin: .6rem 0 .4rem 0;
    }}

    /* ── métricas ── */
    [data-testid="stMetric"] {{
        background: {c["surface"]};
        border: 1px solid {c["border"]};
        border-radius: 10px;
        padding: .85rem 1.1rem;
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: .78rem !important; font-weight: 500 !important;
        color: {c["muted"]} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.45rem !important; font-weight: 700 !important;
        color: {c["text"]} !important;
    }}

    /* ── botão principal ── */
    .stButton > button {{
        background: {c["accent"]} !important;
        color: #fff !important; border: none !important;
        border-radius: 8px !important; font-weight: 700 !important;
        font-size: 1rem !important; padding: .65rem 2.4rem !important;
        transition: opacity .15s;
    }}
    .stButton > button:hover {{ opacity: .86 !important; }}

    /* ── inputs ── */
    input[type="number"], select, textarea {{
        background: {c["input_bg"]} !important;
        color: {c["text"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 7px !important;
        font-size: .93rem !important;
    }}
    label, [data-testid="stWidgetLabel"] p {{
        font-size: .86rem !important;
        font-weight: 500 !important;
        color: {c["muted"]} !important;
    }}

    /* ── abas ── */
    [data-baseweb="tab-list"] {{
        background: {c["surface"]} !important;
        border-radius: 10px !important;
        padding: .22rem !important;
        border: 1px solid {c["border"]} !important;
    }}
    [data-baseweb="tab"] {{
        font-size: .88rem !important; font-weight: 600 !important;
        border-radius: 7px !important;
        padding: .4rem 1rem !important;
        color: {c["muted"]} !important;
    }}
    [aria-selected="true"][data-baseweb="tab"] {{
        background: {c["accent"]} !important;
        color: #fff !important;
    }}

    /* ── cartões de teoria ── */
    .tcard {{
        background: {c["surface"]};
        border: 1px solid {c["border"]};
        border-left: 4px solid {c["accent2"]};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .tcard h4 {{
        font-size: 1rem; font-weight: 700;
        color: {c["text"]}; margin: 0 0 .55rem 0;
    }}
    .tcard p {{
        font-size: .91rem; line-height: 1.75;
        color: {c["muted"]}; margin: .3rem 0;
    }}
    .tc-up   {{ color: {c["success"]}; font-weight: 600; }}
    .tc-down {{ color: {c["danger"]};  font-weight: 600; }}
    .tc-warn {{
        background: {c["warn_bg"]};
        border-left: 3px solid {c["danger"]};
        border-radius: 5px;
        padding: .5rem .8rem; margin-top: .7rem;
        font-size: .84rem; color: {c["danger"]}; line-height: 1.55;
    }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCO C — TELA INICIAL
# ═══════════════════════════════════════════════════════════════════════════

MACHINES = [
    {"key": "mit",  "name": "Motor de Indução Trifásico",  "icon": "MIT", "tag": "Disponível",  "disabled": False},
    {"key": "sync", "name": "Gerador Síncrono",             "icon": "GS",  "tag": "Em breve",    "disabled": True},
    {"key": "dc",   "name": "Motor de Corrente Contínua",  "icon": "MCC", "tag": "Em breve",    "disabled": True},
    {"key": "tr",   "name": "Transformador",                "icon": "TR",  "tag": "Em breve",    "disabled": True},
]


def render_machine_selector(dark: bool) -> None:
    c = _palette(dark)
    st.markdown('<p class="slabel">Seleção de Equipamento</p>', unsafe_allow_html=True)
    st.markdown("### Escolha o equipamento para simular")
    st.write("")

    cols = st.columns(4, gap="medium")
    for i, m in enumerate(MACHINES):
        with cols[i]:
            active   = st.session_state.get("selected_machine") == m["key"]
            disabled = m["disabled"]
            cls = "mcard" + (" active" if active else "") + (" disabled" if disabled else "")
            tag_cls = "mcard-tag" + (" soon" if disabled else "")
            st.markdown(
                f'<div class="{cls}">'
                f'  <span class="mcard-icon">{m["icon"]}</span>'
                f'  <div class="mcard-name">{m["name"]}</div>'
                f'  <span class="{tag_cls}">{m["tag"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            if not disabled:
                if st.button("Selecionar", key=f"sel_{m['key']}", use_container_width=True):
                    st.session_state["selected_machine"] = m["key"]
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# BLOCO D — LAYOUT PRINCIPAL DO MIT
# ═══════════════════════════════════════════════════════════════════════════

VARIABLE_CATALOG = {
    "Torque eletromagnético  Te  (N·m)":         "Te",
    "Velocidade do rotor  n  (RPM)":              "n",
    "Velocidade angular  wr  (rad/s)":            "wr",
    "Corrente de fase A — estator  ias  (A)":     "ias",
    "Corrente de fase B — estator  ibs  (A)":     "ibs",
    "Corrente de fase C — estator  ics  (A)":     "ics",
    "Corrente de fase A — rotor  iar  (A)":       "iar",
    "Corrente de fase B — rotor  ibr  (A)":       "ibr",
    "Corrente de fase C — rotor  icr  (A)":       "icr",
    "Componente  ids  (A)":                       "ids",
    "Componente  iqs  (A)":                       "iqs",
    "Componente  idr  (A)":                       "idr",
    "Componente  iqr  (A)":                       "iqr",
    "Tensão de fase  Va  (V)":                    "Va",
    "Tensão de fase  Vb  (V)":                    "Vb",
    "Tensão de fase  Vc  (V)":                    "Vc",
    "Tensão dq  Vds  (V)":                        "Vds",
    "Tensão dq  Vqs  (V)":                        "Vqs",
}


def _pgroup(title: str) -> None:
    st.markdown(f'<div class="pgroup-title">{title}</div>', unsafe_allow_html=True)


def _ibox(html: str) -> None:
    st.markdown(f'<div class="ibox">{html}</div>', unsafe_allow_html=True)


def render_machine_params(dark: bool) -> tuple[MachineParams, int]:
    """Coluna esquerda: todos os campos de parâmetros. Retorna (mp, ref_code)."""
    st.markdown('<p class="slabel">Parâmetros Físicos da Máquina</p>', unsafe_allow_html=True)

    # ── Elétricos ─────────────────────────────────────────────────────────
    st.markdown('<div class="pgroup">', unsafe_allow_html=True)
    _pgroup("Dados Elétricos")
    Vl  = st.number_input("Tensão de linha RMS — Vl (V)",            min_value=50.0,  max_value=15000.0, value=220.0,  step=1.0)
    f   = st.number_input("Frequência da rede — f (Hz)",              min_value=1.0,   max_value=400.0,   value=60.0,   step=1.0)
    Rs  = st.number_input("Resistência do estator — Rs (Ω)",          min_value=0.001, max_value=100.0,   value=0.435,  step=0.001, format="%.3f")
    Rr  = st.number_input("Resistência do rotor — Rr (Ω)",            min_value=0.001, max_value=100.0,   value=0.816,  step=0.001, format="%.3f")
    Xm  = st.number_input("Reatância de magnetização — Xm (Ω)",       min_value=0.1,   max_value=500.0,   value=26.13,  step=0.01,  format="%.2f")
    Xls = st.number_input("Reatância de dispersão do estator — Xls (Ω)", min_value=0.001, max_value=50.0, value=0.754, step=0.001, format="%.3f")
    Xlr = st.number_input("Reatância de dispersão do rotor — Xlr (Ω)",   min_value=0.001, max_value=50.0, value=0.754, step=0.001, format="%.3f")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Mecânicos ─────────────────────────────────────────────────────────
    st.markdown('<div class="pgroup">', unsafe_allow_html=True)
    _pgroup("Dados Mecânicos e Referencial")
    p   = st.selectbox("Número de polos — p", options=[2, 4, 6, 8, 10, 12], index=1)
    J   = st.number_input("Momento de inércia — J (kg·m²)",   min_value=0.001, max_value=100.0, value=0.089, step=0.001, format="%.3f")
    B   = st.number_input("Coeficiente de atrito viscoso — B (N·m·s/rad)", min_value=0.0, max_value=10.0, value=0.0, step=0.001, format="%.3f")
    ref_label = st.selectbox(
        "Referencial da Transformada de Park",
        ["Síncrono  (w_ref = we)", "Rotórico  (w_ref = wr)", "Estacionário  (w_ref = 0)"],
    )
    ref_code = {"Síncrono  (w_ref = we)": 1,
                "Rotórico  (w_ref = wr)": 2,
                "Estacionário  (w_ref = 0)": 3}[ref_label]
    st.markdown('</div>', unsafe_allow_html=True)

    mp = MachineParams(Vl=Vl, f=f, Rs=Rs, Rr=Rr, Xm=Xm, Xls=Xls, Xlr=Xlr, p=p, J=J, B=B)

    # grandezas derivadas
    st.write("")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Velocidade Síncrona", f"{mp.n_sync:.1f} RPM")
    mc2.metric("Velocidade Angular Base", f"{mp.wb:.2f} rad/s")
    mc3.metric("Reatância Mútua Xml", f"{mp.Xml:.4f} Ω")

    return mp, ref_code


def render_experiment_config(mp: MachineParams) -> dict:
    """Abaixo do circuito: configuração do experimento."""
    st.write("")
    st.markdown('<p class="slabel">Experimento</p>', unsafe_allow_html=True)

    exp_options = {
        "Partida Direta (DOL)":                  "dol",
        "Partida Estrela-Triângulo (Y-Δ)":       "yd",
        "Partida com Autotransformador":          "comp",
        "Soft-Starter (Rampa de Tensão)":         "soft",
        "Aplicação de Carga (partida em vazio)": "carga",
        "Operação como Gerador":                  "gerador",
    }
    exp_label = st.selectbox("Tipo de Experimento", list(exp_options.keys()), key="exp_select")
    exp_type  = exp_options[exp_label]
    config    = {"exp_type": exp_type}

    st.markdown('<div class="pgroup">', unsafe_allow_html=True)
    _pgroup("Parâmetros de Carga e Tensão")

    if exp_type == "dol":
        config["Tl_final"] = st.number_input("Torque de carga (N·m)", value=80.0, min_value=0.0)
        config["t_carga"]  = st.number_input("Instante de aplicação da carga (s)", value=0.1, min_value=0.0)

    elif exp_type == "yd":
        config["Tl_final"] = st.number_input("Torque de carga (N·m)", value=80.0, min_value=0.0)
        config["t_2"]      = st.number_input("Instante de comutação Y → Δ (s)", value=0.5, min_value=0.01)
        config["t_carga"]  = st.number_input("Instante de aplicação da carga (s)", value=0.1, min_value=0.0)
        _ibox("A tensão em estrela é reduzida a Vl / √3. A comutação para triângulo ocorre no instante t₂.")

    elif exp_type == "comp":
        config["Tl_final"]      = st.number_input("Torque de carga (N·m)", value=80.0, min_value=0.0)
        config["voltage_ratio"] = st.slider("Tap do autotransformador (%)", 10, 95, 50) / 100.0
        config["t_2"]           = st.number_input("Instante de comutação (s)", value=0.5, min_value=0.01)
        config["t_carga"]       = st.number_input("Instante de aplicação da carga (s)", value=0.1, min_value=0.0)
        _ibox(f"Tensão inicial = {config['voltage_ratio']*100:.0f}% de Vl nominal.")

    elif exp_type == "soft":
        config["voltage_ratio"] = st.slider("Tensão inicial do soft-starter (%)", 10, 90, 50) / 100.0
        config["t_2"]           = st.number_input("Início da rampa de tensão (s)", value=0.9, min_value=0.0)
        config["t_pico"]        = st.number_input("Tempo para atingir tensão nominal (s)", value=5.0, min_value=0.1)
        config["Tl_final"]      = st.number_input("Torque de carga (N·m)", value=80.0, min_value=0.0)
        config["t_carga"]       = st.number_input("Instante de aplicação da carga (s)", value=0.1, min_value=0.0)

    elif exp_type == "carga":
        Tl_nom = st.number_input("Torque nominal de referência (N·m)", value=80.0, min_value=0.1)
        pct    = st.slider("Percentual da carga (%)", min_value=1, max_value=300, value=100)
        config["Tl_final"] = Tl_nom * pct / 100.0
        config["t_carga"]  = st.number_input("Instante de aplicação da carga (s)", value=1.0, min_value=0.0)
        regime = "nominal" if pct == 100 else ("sobrecarga" if pct > 100 else "carga parcial")
        _ibox(f"Torque aplicado: <strong>{config['Tl_final']:.2f} N·m</strong> ({pct}% de {Tl_nom:.1f} N·m) — {regime}.")

    elif exp_type == "gerador":
        config["Tl_mec"] = st.number_input("Torque mecânico da turbina (N·m)", value=80.0, min_value=1.0)
        config["t_2"]    = st.number_input("Instante de aplicação do torque (s)", value=1.0, min_value=0.0)
        _ibox("O torque negativo impulsiona o rotor acima da velocidade síncrona, colocando a máquina em modo gerador.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── seleção de variáveis ──────────────────────────────────────────────
    st.write("")
    st.markdown('<p class="slabel">Variáveis para Visualização</p>', unsafe_allow_html=True)
    st.markdown('<div class="pgroup">', unsafe_allow_html=True)
    _pgroup("Grandezas a Plotar")
    selected_labels = st.multiselect(
        "Selecione as grandezas",
        options=list(VARIABLE_CATALOG.keys()),
        default=[
            "Torque eletromagnético  Te  (N·m)",
            "Velocidade do rotor  n  (RPM)",
            "Corrente de fase A — estator  ias  (A)",
        ],
    )
    var_keys   = [VARIABLE_CATALOG[v] for v in selected_labels]
    var_labels = list(selected_labels)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── tempo e passo ─────────────────────────────────────────────────────
    st.write("")
    st.markdown('<p class="slabel">Parâmetros Numéricos da Simulação</p>', unsafe_allow_html=True)
    st.markdown('<div class="pgroup">', unsafe_allow_html=True)
    _pgroup("Tempo Total e Passo de Integração")
    tc1, tc2 = st.columns(2)
    with tc1:
        tmax = st.number_input("Tempo total — tmax (s)", min_value=0.1, max_value=60.0, value=2.0, step=0.1, format="%.1f")
        h    = st.number_input("Passo de integração — h (s)", min_value=0.00001, max_value=0.01, value=0.001, step=0.0001, format="%.5f")
        n_steps = int(tmax / h)
        st.caption(f"Total de passos: {n_steps:,}")
        if n_steps > 100_000:
            st.warning("Número elevado de passos. A simulação pode demorar vários segundos.")
    with tc2:
        _ibox(
            "<strong>tmax:</strong> quanto maior, mais do transitório é capturado, porém maior o custo "
            "computacional. Valores excessivos combinados com passos pequenos podem ser proibitivos.<br><br>"
            "<strong>h (passo):</strong> valores acima de 0,005 s podem causar instabilidade numérica. "
            "Abaixo de 0,0001 s raramente trazem ganho perceptível de precisão. "
            "O padrão de 0,001 s equilibra estabilidade e desempenho para a maioria dos experimentos."
        )
    st.markdown('</div>', unsafe_allow_html=True)

    return config, var_keys, var_labels, tmax, h


# ═══════════════════════════════════════════════════════════════════════════
# BLOCO E — CIRCUITO EQUIVALENTE (schemdraw)
# ═══════════════════════════════════════════════════════════════════════════

def render_circuit(mp: MachineParams, dark: bool) -> None:
    """Desenha o circuito equivalente monofásico em T da MIT com schemdraw."""
    c      = _palette(dark)
    bg_hex = "#0d1117" if dark else "#ffffff"
    wire   = "#e4e8f5" if dark else "#111827"   # monocromático — fios e componentes
    muted  = c["muted"]

    OFST   = 0.22   # offset uniforme acima/abaixo/esq/dir para todas as labels
    FS     = 9      # tamanho fonte — nome do componente
    FS_VAL = 8      # tamanho fonte — valor numérico

    fig_mpl, ax = plt.subplots(figsize=(10, 3.8))
    fig_mpl.patch.set_facecolor(bg_hex)
    ax.set_facecolor(bg_hex)
    ax.set_axis_off()   # remove eixos, ticks e labels dos eixos

    with schemdraw.Drawing(canvas=ax) as d:
        d.config(fontsize=10, color=wire)

        # ── fonte de tensão Vs ──────────────────────────────────────────
        src = d.add(
            elm.SourceSin()
            .up()
            .color(wire)
            .label(r"$V_s$", loc="right", color=wire)
            .length(d.unit)
        )

        # ── fio superior saindo da fonte ────────────────────────────────
        d.add(elm.Line().right().length(0.4))

        # ── Rs (horizontal: nome em cima, valor embaixo, offset igual) ──
        d.add(
            elm.Resistor().right().color(wire)
            .label(r"$R_s$",           loc="top", ofst=OFST, fontsize=FS,     color=wire)
            .label(f"{mp.Rs:.3f} Ω",   loc="bot", ofst=OFST, fontsize=FS_VAL, color=wire)
        )

        # ── jXls (horizontal) ───────────────────────────────────────────
        d.add(
            elm.Inductor2().right().color(wire)
            .label(r"$jX_{ls}$",        loc="top", ofst=OFST, fontsize=FS,     color=wire)
            .label(f"{mp.Xls:.3f} Ω",   loc="bot", ofst=OFST, fontsize=FS_VAL, color=wire)
        )

        # ── nó T ────────────────────────────────────────────────────────
        T_node = d.add(elm.Dot(open=True).color(wire))

        # ── jXlr (horizontal) ───────────────────────────────────────────
        d.add(
            elm.Inductor2().right().color(wire)
            .label(r"$jX_{lr}$",        loc="top", ofst=OFST, fontsize=FS,     color=wire)
            .label(f"{mp.Xlr:.3f} Ω",   loc="bot", ofst=OFST, fontsize=FS_VAL, color=wire)
        )

        # ── Rr/s (horizontal) ───────────────────────────────────────────
        d.add(
            elm.Resistor().right().color(wire)
            .label(r"$R_r/s$",           loc="top", ofst=OFST, fontsize=FS,     color=wire)
            .label(f"{mp.Rr:.3f} Ω/s",  loc="bot", ofst=OFST, fontsize=FS_VAL, color=wire)
        )

        # ── fio de retorno ──────────────────────────────────────────────
        d.add(elm.Line().down().length(d.unit))
        bot_right = d.here
        d.add(elm.Line().left().tox(src.start))
        d.add(elm.Line().up().toy(src.start))

        # ── ramo shunt jXm (vertical: nome à esquerda, valor à direita) ─
        d.add(elm.Line().at(T_node.end).down().length(0.3))
        d.add(
            elm.Inductor2().down().color(wire)
            .label(r"$jX_m$",          loc="left",  ofst=OFST, fontsize=FS,     color=wire)
            .label(f"{mp.Xm:.2f} Ω",  loc="right", ofst=OFST, fontsize=FS_VAL, color=wire)
        )
        d.add(elm.Line().down().toy(bot_right))
        d.add(elm.Ground().color(wire))

    # nota de escorregamento
    ax.text(
        0.5, 0.01,
        "s = (ns − n) / ns  (escorregamento)",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=8, color=muted,
        fontfamily="monospace",
    )

    fig_mpl.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.08)

    buf = io.BytesIO()
    fig_mpl.savefig(buf, format="png", dpi=150, facecolor=bg_hex, bbox_inches="tight")
    plt.close(fig_mpl)
    buf.seek(0)

    st.image(buf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCO F — RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════

_LINE_COLORS_DARK  = ["#4f8ef7","#f97316","#22c55e","#a78bfa",
                      "#ec4899","#14b8a6","#f59e0b","#6366f1",
                      "#84cc16","#ef4444","#06b6d4","#d946ef"]
_LINE_COLORS_LIGHT = ["#1d4ed8","#ea580c","#16a34a","#7c3aed",
                      "#db2777","#0d9488","#d97706","#4f46e5",
                      "#65a30d","#dc2626","#0891b2","#c026d3"]


def _colors(dark: bool) -> list:
    return _LINE_COLORS_DARK if dark else _LINE_COLORS_LIGHT


def _plot_theme(dark: bool) -> dict:
    return dict(
        plot_bg  = "#0d1117" if dark else "#ffffff",
        paper_bg = "#161b27" if dark else "#f4f7ff",
        fg       = "#e4e8f5" if dark else "#111827",
        grid     = "#2a3150" if dark else "#dde4f5",
    )


def build_fig_stacked(res, var_keys, var_labels, dark, t_events) -> go.Figure:
    """Gráficos empilhados — cada variável em seu próprio subplot."""
    n = len(var_keys)
    pt = _plot_theme(dark)
    cols = _colors(dark)

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        subplot_titles=var_labels,
        vertical_spacing=max(0.03, 0.07/max(n,1)),
    )
    t = res["t"]
    for i, (key, lbl) in enumerate(zip(var_keys, var_labels), 1):
        col = cols[(i-1) % len(cols)]
        fig.add_trace(go.Scatter(
            x=t, y=res[key], mode="lines", name=lbl,
            line=dict(color=col, width=1.9),
            hovertemplate=f"<b>{lbl}</b><br>t = %{{x:.4f}} s<br>valor = %{{y:.4f}}<extra></extra>",
        ), row=i, col=1)
        for te in (t_events or []):
            fig.add_vline(x=te, line_dash="dot", line_color="#64748b",
                          line_width=1.1, row=i, col=1)
        fig.update_yaxes(row=i, col=1,
                         showgrid=True, gridcolor=pt["grid"], gridwidth=0.4,
                         zeroline=True, zerolinecolor=pt["grid"],
                         tickfont=dict(size=10, color=pt["fg"]))

    # centraliza títulos dos subplots
    for ann in fig.layout.annotations:
        ann.update(xanchor="center", x=0.5, font_color=pt["fg"], font_size=11)

    fig.update_xaxes(row=n, col=1, title_text="Tempo (s)",
                     showgrid=True, gridcolor=pt["grid"], gridwidth=0.4,
                     tickfont=dict(size=10, color=pt["fg"]))
    fig.update_layout(
        height=max(260, 200*n),
        paper_bgcolor=pt["paper_bg"], plot_bgcolor=pt["plot_bg"],
        font=dict(family="Inter, system-ui", size=11, color=pt["fg"]),
        margin=dict(l=55, r=20, t=45, b=40),
        hovermode="x unified", showlegend=False,
    )
    return fig


def build_fig_sidebyside(res, var_keys, var_labels, dark, t_events) -> list[go.Figure]:
    """Retorna uma lista de figuras pequenas — renderizadas lado a lado via colunas."""
    pt   = _palette(dark)
    cols = _colors(dark)
    figs = []
    t    = res["t"]
    th   = _plot_theme(dark)
    for i, (key, lbl) in enumerate(zip(var_keys, var_labels)):
        col = cols[i % len(cols)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=res[key], mode="lines", name=lbl,
            line=dict(color=col, width=1.8),
            hovertemplate=f"<b>{lbl}</b><br>t = %{{x:.4f}} s<br>valor = %{{y:.4f}}<extra></extra>",
        ))
        for te in (t_events or []):
            fig.add_vline(x=te, line_dash="dot", line_color="#64748b", line_width=1.1)
        fig.update_layout(
            title=dict(text=lbl, x=0.5, xanchor="center",
                       font=dict(size=11, color=th["fg"])),
            height=230,
            paper_bgcolor=th["paper_bg"], plot_bgcolor=th["plot_bg"],
            font=dict(family="Inter, system-ui", size=10, color=th["fg"]),
            margin=dict(l=45, r=12, t=36, b=36),
            xaxis=dict(title="Tempo (s)", showgrid=True,
                       gridcolor=th["grid"], tickfont=dict(size=9, color=th["fg"])),
            yaxis=dict(showgrid=True, gridcolor=th["grid"],
                       zeroline=True, zerolinecolor=th["grid"],
                       tickfont=dict(size=9, color=th["fg"])),
            hovermode="x unified", showlegend=False,
        )
        figs.append(fig)
    return figs


def build_fig_overlay(res, var_keys, var_labels, dark, t_events) -> go.Figure:
    """
    Curvas sobrepostas no mesmo gráfico.
    Usa eixo Y secundário quando há mais de um grupo de unidades.
    Heurística simples: variáveis com 'RPM' ou 'rad/s' → eixo direito.
    """
    pt   = _plot_theme(dark)
    cols = _colors(dark)
    t    = res["t"]

    right_units = {"n", "wr"}   # variáveis que vão ao eixo Y direito
    has_right   = any(k in right_units for k in var_keys)

    fig = go.Figure()
    for i, (key, lbl) in enumerate(zip(var_keys, var_labels)):
        col   = cols[i % len(cols)]
        yaxis = "y2" if (key in right_units and has_right) else "y"
        fig.add_trace(go.Scatter(
            x=t, y=res[key], mode="lines", name=lbl,
            line=dict(color=col, width=1.9), yaxis=yaxis,
            hovertemplate=f"<b>{lbl}</b><br>t = %{{x:.4f}} s<br>valor = %{{y:.4f}}<extra></extra>",
        ))
    for te in (t_events or []):
        fig.add_vline(x=te, line_dash="dot", line_color="#64748b", line_width=1.1)

    y2_cfg = dict(
        overlaying="y", side="right",
        showgrid=False, zeroline=False,
        tickfont=dict(size=10, color=pt["fg"]),
    ) if has_right else {}

    fig.update_layout(
        height=380,
        title=dict(text="Curvas Sobrepostas", x=0.5, xanchor="center",
                   font=dict(size=12, color=pt["fg"])),
        paper_bgcolor=pt["paper_bg"], plot_bgcolor=pt["plot_bg"],
        font=dict(family="Inter, system-ui", size=11, color=pt["fg"]),
        margin=dict(l=55, r=65 if has_right else 20, t=48, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Tempo (s)", showgrid=True,
                   gridcolor=pt["grid"], gridwidth=0.4,
                   tickfont=dict(size=10, color=pt["fg"])),
        yaxis=dict(showgrid=True, gridcolor=pt["grid"],
                   zeroline=True, zerolinecolor=pt["grid"],
                   tickfont=dict(size=10, color=pt["fg"])),
        yaxis2=y2_cfg if has_right else {},
    )
    return fig


def render_results(res: dict, var_keys: list, var_labels: list,
                   dark: bool, t_events: list) -> None:
    """KPIs + controle de modo de visualização + gráficos."""
    st.divider()
    st.markdown('<p class="slabel">Indicadores de Regime Permanente</p>',
                unsafe_allow_html=True)

    # KPIs
    n_ss   = res["n"][-1]
    Te_ss  = res["Te"][-1]
    wr_ss  = res["wr"][-1]
    Te_max = float(np.max(res["Te"]))
    ias_pk = float(np.max(np.abs(res["ias"])))
    ias_ss = float(np.abs(res["ias"][-1]))

    k = st.columns(6)
    k[0].metric("Velocidade de Regime (RPM)",    f"{n_ss:.1f}")
    k[1].metric("Torque de Regime Te (N·m)",     f"{Te_ss:.2f}")
    k[2].metric("Torque Máximo Te_max (N·m)",    f"{Te_max:.2f}")
    k[3].metric("Corrente de Pico  ias  (A)",    f"{ias_pk:.2f}")
    k[4].metric("Corrente de Regime  ias  (A)",  f"{ias_ss:.2f}")
    k[5].metric("Vel. Angular  wr  (rad/s)",     f"{wr_ss:.3f}")

    st.write("")

    if not var_keys:
        st.info("Nenhuma grandeza selecionada. Retorne à configuração e escolha variáveis para plotar.")
        return

    # controles de visualização
    cv1, cv2, _ = st.columns([1.6, 1, 4])
    with cv1:
        modo = st.radio(
            "Modo de visualização",
            ["Empilhados", "Lado a lado", "Sobrepostos"],
            horizontal=True,
            key="plot_mode",
        )
    with cv2:
        dark_plot = st.toggle("Fundo escuro", value=dark, key="plot_dark_toggle")

    st.write("")

    if modo == "Empilhados":
        fig = build_fig_stacked(res, var_keys, var_labels, dark_plot, t_events)
        st.plotly_chart(fig, use_container_width=True)

    elif modo == "Lado a lado":
        figs = build_fig_sidebyside(res, var_keys, var_labels, dark_plot, t_events)
        n_cols = min(len(figs), 3)
        rows   = [figs[i:i+n_cols] for i in range(0, len(figs), n_cols)]
        for row in rows:
            cols = st.columns(len(row), gap="small")
            for col, fig in zip(cols, row):
                with col:
                    st.plotly_chart(fig, use_container_width=True)

    else:  # Sobrepostos
        fig = build_fig_overlay(res, var_keys, var_labels, dark_plot, t_events)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCO G — ABA TEORIA
# ═══════════════════════════════════════════════════════════════════════════

THEORY_DATA = [
    {
        "group": "Parâmetros Elétricos",
        "items": [
            {
                "nome": "Vl — Tensão de Linha RMS",
                "desc": (
                    "Define a amplitude do campo magnético girante no estator. "
                    "É a grandeza que estabelece o ponto de operação magnético da máquina "
                    "e determina o fluxo de entreferro."
                ),
                "up": (
                    "O torque máximo cresce proporcionalmente ao quadrado da tensão. "
                    "A corrente de partida também aumenta de forma significativa."
                ),
                "down": (
                    "O torque de partida cai e pode tornar-se insuficiente para vencer a carga estática, "
                    "impedindo que o motor entre em operação."
                ),
                "warn": (
                    "Sobretensão provoca saturação do núcleo e aquecimento excessivo do isolamento. "
                    "Subtensão severa pode causar parada brusca (stall) com o motor em carga."
                ),
            },
            {
                "nome": "f — Frequência da Rede",
                "desc": (
                    "Determina a velocidade síncrona do campo girante (ns = 120·f/p). "
                    "Todas as reatâncias do circuito equivalente são proporcionais à frequência."
                ),
                "up": (
                    "A velocidade síncrona sobe proporcionalmente e as reatâncias aumentam "
                    "(Xm, Xls, Xlr crescem), redistribuindo o torque ao longo da curva T × n."
                ),
                "down": (
                    "A velocidade de operação diminui. Com tensão constante, o fluxo aumenta "
                    "(relação V/f cresce), podendo saturar o núcleo."
                ),
                "warn": (
                    "Operar fora da frequência nominal sem controle V/f proporcional compromete "
                    "o fluxo de entreferro, a eficiência e a vida útil do isolamento."
                ),
            },
            {
                "nome": "Rs — Resistência do Estator",
                "desc": (
                    "Representa as perdas Joule no cobre do enrolamento estatórico. "
                    "Provoca queda de tensão interna, reduzindo a tensão efetiva disponível no entreferro."
                ),
                "up": (
                    "Maior dissipação de energia, menor tensão efetiva no entreferro e redução do torque. "
                    "O motor aquece mais em regime permanente."
                ),
                "down": (
                    "Menor queda interna e melhor eficiência. "
                    "O modelo se aproxima de um transformador ideal sem perdas no primário."
                ),
                "warn": (
                    "Rs muito elevado — causado por enrolamento danificado — pode provocar sobreaquecimento fatal. "
                    "Rs próximo de zero gera instabilidade numérica nas equações diferenciais com passos de integração grandes."
                ),
            },
            {
                "nome": "Rr — Resistência do Rotor",
                "desc": (
                    "Parâmetro central da curva de torque. Controla o escorregamento de regime "
                    "e o torque de partida. Nas barras da gaiola de esquilo, é inversamente "
                    "proporcional à condutividade elétrica do material."
                ),
                "up": (
                    "O escorregamento de regime aumenta (o rotor gira mais devagar em relação ao campo). "
                    "O torque de partida cresce até um ponto ótimo e depois decresce. "
                    "A curva T × n se torna mais plana e larga."
                ),
                "down": (
                    "Menor escorregamento em regime permanente e melhor eficiência. "
                    "O torque de partida é reduzido e a curva T × n torna-se mais estreita e alta "
                    "próxima à velocidade síncrona."
                ),
                "warn": (
                    "Rr muito alto — típico de barras da gaiola fraturadas — provoca escorregamento excessivo "
                    "e sobreaquecimento do rotor. "
                    "Rr próximo de zero causa instabilidade numérica por singularidade nas equações do rotor."
                ),
            },
            {
                "nome": "Xm — Reatância de Magnetização",
                "desc": (
                    "Representa o ramo paralelo (shunt) do circuito equivalente: "
                    "o caminho do fluxo magnético principal pelo núcleo de ferro. "
                    "Quanto maior Xm, menor a corrente desviada para magnetizar o núcleo."
                ),
                "up": (
                    "Menor corrente de magnetização, melhor fator de potência e maior eficiência. "
                    "O motor opera de forma mais próxima ao comportamento ideal."
                ),
                "down": (
                    "Maior corrente de magnetização circula mesmo em vazio, "
                    "resultando em pior fator de potência e maior dissipação de energia."
                ),
                "warn": (
                    "Xm muito baixo representa um núcleo de má qualidade magnética ou saturado. "
                    "Na simulação, Xm próximo de zero torna Xml desprezível, "
                    "causando divisão por valores muito pequenos e divergência numérica."
                ),
            },
            {
                "nome": "Xls e Xlr — Reatâncias de Dispersão",
                "desc": (
                    "Representam os fluxos que não atravessam o entreferro — os chamados fluxos de dispersão "
                    "do estator e do rotor. Limitam a corrente de curto-circuito e a capacidade de "
                    "transferência de energia entre estator e rotor."
                ),
                "up": (
                    "Maior impedância total. A corrente de partida é reduzida, "
                    "mas o torque máximo também diminui."
                ),
                "down": (
                    "Correntes de partida mais elevadas e torque máximo maior, "
                    "porém o motor torna-se mais sensível a transitórios e variações de carga."
                ),
                "warn": (
                    "Dispersão muito baixa leva a correntes de partida excessivamente altas, "
                    "podendo danificar o isolamento. "
                    "Dispersão muito alta limita o torque a ponto de impedir a partida sob carga."
                ),
            },
        ],
    },
    {
        "group": "Parâmetros Mecânicos",
        "items": [
            {
                "nome": "p — Número de Polos",
                "desc": (
                    "Define a velocidade síncrona (ns = 120·f/p) e, portanto, "
                    "a faixa de velocidade de operação da máquina. "
                    "Máquinas com maior número de polos giram mais devagar para uma mesma frequência."
                ),
                "up": (
                    "A velocidade síncrona diminui. A máquina opera em rotações menores. "
                    "Para a mesma potência mecânica, o torque necessário é maior."
                ),
                "down": (
                    "A velocidade síncrona aumenta. "
                    "O torque nominal é menor para a mesma potência."
                ),
                "warn": (
                    "O número de polos é sempre par e discreto. "
                    "Valores ímpares ou muito altos combinados com baixa frequência "
                    "podem gerar velocidades de operação fisicamente irrealistas no modelo."
                ),
            },
            {
                "nome": "J — Momento de Inércia",
                "desc": (
                    "Representa a resistência do conjunto rotor-carga a variações de velocidade "
                    "(segunda lei de Newton rotacional: Te − Tl = J · d(wr)/dt). "
                    "Inclui a inércia tanto do rotor quanto da carga acoplada."
                ),
                "up": (
                    "Aceleração mais lenta e transitório mais prolongado. "
                    "O sistema absorve e libera energia cinética de forma mais gradual, "
                    "sendo naturalmente mais amortecido."
                ),
                "down": (
                    "Aceleração muito rápida. O rotor responde quase instantaneamente "
                    "a qualquer variação de torque."
                ),
                "warn": (
                    "J muito baixo pode causar oscilações de velocidade de difícil controle "
                    "em sistemas com variação de carga. "
                    "J muito alto pode fazer o transitório ultrapassar o tempo de simulação "
                    "sem que o regime permanente seja atingido."
                ),
            },
            {
                "nome": "B — Coeficiente de Atrito Viscoso",
                "desc": (
                    "Modela as perdas mecânicas proporcionais à velocidade: "
                    "mancais, ventilação forçada e resistência do fluido ao redor do rotor. "
                    "Produz um torque de frenagem igual a B·wr."
                ),
                "up": (
                    "Maior dissipação mecânica e velocidade de regime ligeiramente menor. "
                    "O sistema amorte transitórios de velocidade de forma mais eficiente."
                ),
                "down": (
                    "Menor atrito mecânico. Com B = 0 (valor padrão), toda a resistência ao "
                    "movimento vem exclusivamente da carga mecânica aplicada."
                ),
                "warn": (
                    "B muito elevado pode paralisar o motor mesmo sem carga nominal, "
                    "pois as perdas mecânicas superam o torque eletromagnético disponível. "
                    "Na prática, atrito excessivo indica falha em mancais ou rolamentos."
                ),
            },
        ],
    },
    {
        "group": "Parâmetros de Simulação",
        "items": [
            {
                "nome": "tmax — Tempo Total de Simulação",
                "desc": (
                    "Define o horizonte temporal da integração numérica. "
                    "Deve ser longo o suficiente para capturar o transitório de interesse "
                    "e o eventual regime permanente."
                ),
                "up": (
                    "Mais do transitório e do regime permanente são observados, "
                    "mas o tempo de processamento cresce proporcionalmente."
                ),
                "down": (
                    "A simulação é concluída mais rapidamente, porém pode encerrar "
                    "antes que o sistema atinja o regime permanente."
                ),
                "warn": (
                    "tmax muito grande combinado com passo de integração pequeno "
                    "pode consumir memória excessiva e bloquear o navegador."
                ),
            },
            {
                "nome": "h — Passo de Integração",
                "desc": (
                    "Controla a discretização temporal do método Runge-Kutta de 4ª ordem. "
                    "É o parâmetro mais crítico para a estabilidade e a precisão numérica da simulação."
                ),
                "up": (
                    "A simulação é executada com maior rapidez, porém com menor precisão. "
                    "Passos muito grandes causam instabilidade numérica: "
                    "oscilações artificiais ou divergência completa dos resultados."
                ),
                "down": (
                    "Maior precisão e estabilidade numérica, porém custo computacional muito maior. "
                    "Abaixo de certo limiar, o ganho de precisão torna-se imperceptível."
                ),
                "warn": (
                    "Passos acima de 0,005 s podem gerar resultados fisicamente incorretos "
                    "para os parâmetros típicos da MIT. "
                    "A regra prática é h ≤ 1/(10·wb) para garantir estabilidade no referencial síncrono."
                ),
            },
        ],
    },
]


def render_theory_tab() -> None:
    st.markdown(
        "Nesta aba, cada parâmetro é descrito em termos de seu significado físico e do impacto "
        "qualitativo que provoca no comportamento da máquina. Nenhum valor numérico específico "
        "é apresentado: o objetivo é construir intuição sólida sobre o sistema elétrico antes "
        "ou após a execução das simulações."
    )
    for group in THEORY_DATA:
        st.write("")
        st.markdown(f"## {group['group']}")
        for item in group["items"]:
            st.markdown(
                f'<div class="tcard">'
                f'<h4>{item["nome"]}</h4>'
                f'<p>{item["desc"]}</p>'
                f'<p><span class="tc-up">Se aumentar:</span> {item["up"]}</p>'
                f'<p><span class="tc-down">Se diminuir:</span> {item["down"]}</p>'
                f'<div class="tc-warn">Atenção — calibrações extremas: {item["warn"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# BLOCO H — ORQUESTRADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # inicializa estado de sessão
    if "dark_mode"         not in st.session_state: st.session_state["dark_mode"]         = True
    if "selected_machine"  not in st.session_state: st.session_state["selected_machine"]  = None
    if "sim_result"        not in st.session_state: st.session_state["sim_result"]         = None

    dark = st.session_state.get("dark_mode", True)
    apply_css(dark)

    # ── cabeçalho ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="app-header">'
        '<div class="app-title">Simulador de Máquinas Elétricas</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── tela de seleção ───────────────────────────────────────────────────
    if not st.session_state["selected_machine"]:
        render_machine_selector(dark)
        return

    # ── navegação: voltar ─────────────────────────────────────────────────
    col_back, col_title = st.columns([1, 9])
    with col_back:
        if st.button("Voltar", key="btn_back"):
            st.session_state["selected_machine"] = None
            st.session_state["sim_result"]        = None
            st.rerun()
    with col_title:
        machine_name = next(m["name"] for m in MACHINES
                            if m["key"] == st.session_state["selected_machine"])
        st.markdown(f"### {machine_name}")

    st.divider()

    # ── abas ──────────────────────────────────────────────────────────────
    tab_sim, tab_teoria = st.tabs(["Simulação", "Teoria"])

    # ─── ABA SIMULAÇÃO ────────────────────────────────────────────────────
    with tab_sim:
        # toggle de tema no topo
        ct1, _ = st.columns([1, 6])
        with ct1:
            st.toggle("Modo Escuro", value=True, key="dark_mode")

        st.write("")

        # Layout superior: parâmetros (esq) | circuito equivalente (dir)
        col_params, col_circuit = st.columns([1, 1], gap="large")

        with col_params:
            mp, ref_code = render_machine_params(dark)

        with col_circuit:
            st.markdown('<p class="slabel">Circuito Equivalente Monofásico</p>',
                        unsafe_allow_html=True)
            render_circuit(mp, dark)

            st.write("")

            # Experimento na coluna direita, abaixo do circuito
            exp_config, var_keys, var_labels, tmax, h = render_experiment_config(mp)

        st.write("")

        # Botão centralizado
        bc1, bc2, bc3 = st.columns([2, 1, 2])
        with bc2:
            run_clicked = st.button("Executar Simulação", key="btn_run", use_container_width=True)

        # ── execução ──────────────────────────────────────────────────────
        if run_clicked:
            if not var_keys:
                st.warning("Selecione ao menos uma grandeza para plotar antes de executar.")
            else:
                vfn, tfn, t_events = build_fns(exp_config, mp)
                with st.spinner("Executando integração numérica..."):
                    try:
                        res = run_simulation(
                            mp=mp, tmax=tmax, h=h,
                            voltage_fn=vfn, torque_fn=tfn,
                            ref_code=ref_code,
                        )
                        st.session_state["sim_result"] = dict(
                            res=res, var_keys=var_keys, var_labels=var_labels,
                            t_events=t_events, dark=dark, mp=mp,
                        )
                        st.success(
                            f"Simulação concluída — "
                            f"n = {res['n'][-1]:.1f} RPM | "
                            f"Te = {res['Te'][-1]:.2f} N·m"
                        )
                    except Exception as e:
                        st.error(f"Erro na simulação: {e}")
                        st.info(
                            "Verifique os parâmetros. Passos de integração muito grandes "
                            "ou parâmetros fisicamente inválidos podem causar divergência numérica."
                        )

        # ── resultados (mesma aba, abaixo do botão) ───────────────────────
        sr = st.session_state.get("sim_result")
        if sr is not None:
            render_results(
                res=sr["res"],
                var_keys=sr["var_keys"],
                var_labels=sr["var_labels"],
                dark=sr["dark"],
                t_events=sr["t_events"],
            )

    # ─── ABA TEORIA ───────────────────────────────────────────────────────
    with tab_teoria:
        render_theory_tab()


if __name__ == "__main__":
    main()
