# -*- coding: utf-8 -*-
"""
ems_streamlit.py — Interface Streamlit para o Simulador de Maquina de Inducao Trifasica
Baseado no modelo 0dq de Krause (EMS_BASE.py).

Estrutura modular:
  - render_sidebar()          : toggle de tema e navegacao
  - render_machine_params()   : formulario de parametros da maquina
  - render_experiment_config(): configuracao do experimento e torque
  - render_time_config()      : tempo de simulacao e passo de integracao
  - render_variable_selector(): escolha das variaveis a plotar
  - run_and_plot()            : executa simulacao e exibe graficos Plotly
  - render_learning_tab()     : aba educacional sem valores numericos
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from dataclasses import dataclass, field
from typing import Callable

# =============================================================================
# CONFIGURACAO DA PAGINA (deve ser a primeira chamada Streamlit)
# =============================================================================

st.set_page_config(
    page_title="Simulador MIT — Modelo dq de Krause",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SECAO 1 — MODELO MATEMATICO (nucleo fisico, identico ao EMS_BASE.py)
# =============================================================================

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
    Te  = (3.0 / 2.0) * (mp.p / 2.0) * (1.0 / mp.wb) * (PSIds * iqs - PSIqs * ids)
    dwr = (mp.p / (2.0 * mp.J)) * (Te - Tl) - (mp.B / mp.J) * wr
    return [dPSIqs, dPSIds, dPSIqr, dPSIdr, dwr]


def abc_voltages(t, Vl, f):
    tetae = 2.0 * np.pi * f * t
    Va = np.sqrt(2.0) * Vl * np.sin(tetae)
    Vb = np.sqrt(2.0) * Vl * np.sin(tetae - 2.0 * np.pi / 3.0)
    Vc = np.sqrt(2.0) * Vl * np.sin(tetae + 2.0 * np.pi / 3.0)
    return Va, Vb, Vc


def clarke_park_transform(Va, Vb, Vc, tetae):
    Valpha = np.sqrt(3.0 / 2.0) * (Va - 0.5 * Vb - 0.5 * Vc)
    Vbeta  = np.sqrt(3.0 / 2.0) * ((np.sqrt(3.0) / 2.0) * Vb - (np.sqrt(3.0) / 2.0) * Vc)
    Vds = np.cos(tetae) * Valpha + np.sin(tetae) * Vbeta
    Vqs = -np.sin(tetae) * Valpha + np.cos(tetae) * Vbeta
    return Vds, Vqs


def reconstruct_abc_currents(PSIqs, PSIds, PSIqr, PSIdr, tetae, tetar, mp):
    PSImq = mp.Xml * (PSIqs / mp.Xls + PSIqr / mp.Xlr)
    PSImd = mp.Xml * (PSIds / mp.Xls + PSIdr / mp.Xlr)
    ids = (1.0 / mp.Xls) * (PSIds - PSImd)
    iqs = (1.0 / mp.Xls) * (PSIqs - PSImq)
    idr = (1.0 / mp.Xlr) * (PSIdr - PSImd)
    iqr = (1.0 / mp.Xlr) * (PSIqr - PSImq)
    ias_alpha = np.cos(tetae) * ids - np.sin(tetae) * iqs
    ias_beta  = np.sin(tetae) * ids + np.cos(tetae) * iqs
    iar_alpha = np.cos(tetar) * idr - np.sin(tetar) * iqr
    iar_beta  = np.sin(tetar) * idr + np.cos(tetar) * iqr
    k = np.sqrt(3.0 / 2.0)
    ias = k * ias_alpha
    ibs = k * (-0.5 * ias_alpha + (np.sqrt(3.0) / 2.0) * ias_beta)
    ics = k * (-0.5 * ias_alpha - (np.sqrt(3.0) / 2.0) * ias_beta)
    iar = k * iar_alpha
    ibr = k * (-0.5 * iar_alpha + (np.sqrt(3.0) / 2.0) * iar_beta)
    icr = k * (-0.5 * iar_alpha - (np.sqrt(3.0) / 2.0) * iar_beta)
    return ids, iqs, idr, iqr, ias, ibs, ics, iar, ibr, icr


def voltage_reduced_start(t, Vl_nominal, Vl_reduced, t_switch):
    return Vl_nominal if t >= t_switch else Vl_reduced


def voltage_soft_starter(t, Vl_nominal, Vl_initial, t_start_ramp, t_full):
    if t < t_start_ramp:
        return Vl_initial
    elif t < t_full:
        return Vl_initial + (Vl_nominal - Vl_initial) * (t - t_start_ramp) / (t_full - t_start_ramp)
    return Vl_nominal


def torque_step(t, Tl_before, Tl_after, t_switch):
    return Tl_after if t >= t_switch else Tl_before


def run_simulation(mp, tmax, h, voltage_fn, torque_fn, ref_code=1):
    t_values = np.arange(0.0, tmax, h)
    N = len(t_values)
    wr_r  = np.empty(N); n_r   = np.empty(N); Te_r  = np.empty(N)
    ids_r = np.empty(N); iqs_r = np.empty(N)
    idr_r = np.empty(N); iqr_r = np.empty(N)
    ias_r = np.empty(N); ibs_r = np.empty(N); ics_r = np.empty(N)
    iar_r = np.empty(N); ibr_r = np.empty(N); icr_r = np.empty(N)
    Va_r  = np.empty(N); Vb_r  = np.empty(N); Vc_r  = np.empty(N)
    Vds_r = np.empty(N); Vqs_r = np.empty(N)

    states  = [0.0, 0.0, 0.0, 0.0, 0.0]
    last_wr = 0.0
    we = mp.wb

    for i, t_val in enumerate(t_values):
        Vl_apli    = voltage_fn(t_val)
        current_Tl = torque_fn(t_val)
        tetae = we * t_val
        if ref_code == 1:
            w_ref = we
        elif ref_code == 2:
            w_ref = last_wr
        else:
            w_ref = 0.0

        Va, Vb, Vc = abc_voltages(t_val, Vl_apli, mp.f)
        Vds, Vqs   = clarke_park_transform(Va, Vb, Vc, tetae)

        sol    = odeint(induction_motor_ode, states, [t_val, t_val + h],
                        args=(Vqs, Vds, current_Tl, w_ref, mp))
        states = list(sol[1])
        PSIqs, PSIds, PSIqr, PSIdr, wr = states
        last_wr = wr
        tetar_abc = wr * t_val

        ids, iqs, idr, iqr, ias, ibs, ics, iar, ibr, icr = reconstruct_abc_currents(
            PSIqs, PSIds, PSIqr, PSIdr, tetae, tetar_abc, mp)

        Te = (3.0 / 2.0) * (mp.p / 2.0) * (1.0 / mp.wb) * (PSIds * iqs - PSIqs * ids)

        wr_r[i]  = wr;  n_r[i]   = (120.0 / mp.p) * (wr / (2.0 * np.pi))
        Te_r[i]  = Te
        ids_r[i] = ids; iqs_r[i] = iqs; idr_r[i] = idr; iqr_r[i] = iqr
        ias_r[i] = ias; ibs_r[i] = ibs; ics_r[i] = ics
        iar_r[i] = iar; ibr_r[i] = ibr; icr_r[i] = icr
        Va_r[i]  = Va;  Vb_r[i]  = Vb;  Vc_r[i]  = Vc
        Vds_r[i] = Vds; Vqs_r[i] = Vqs

    return {
        "t": t_values, "wr": wr_r, "n": n_r, "Te": Te_r,
        "ids": ids_r, "iqs": iqs_r, "idr": idr_r, "iqr": iqr_r,
        "ias": ias_r, "ibs": ibs_r, "ics": ics_r,
        "iar": iar_r, "ibr": ibr_r, "icr": icr_r,
        "Va": Va_r, "Vb": Vb_r, "Vc": Vc_r,
        "Vds": Vds_r, "Vqs": Vqs_r,
    }


# =============================================================================
# SECAO 2 — TEMA E ESTILOS CSS
# =============================================================================

def apply_theme(dark_mode: bool) -> None:
    """Injeta CSS para alternar entre tema claro e escuro."""
    if dark_mode:
        bg        = "#0e1117"
        bg2       = "#1a1d27"
        text      = "#e8eaf0"
        text_muted= "#9aa0b4"
        border    = "#2e3250"
        accent    = "#4f8ef7"
        card_bg   = "#161b2e"
        input_bg  = "#1e2333"
        label_col = "#c5cae9"
    else:
        bg        = "#f4f6fb"
        bg2       = "#ffffff"
        text      = "#1a1d2e"
        text_muted= "#5c6380"
        border    = "#d0d5e8"
        accent    = "#1a56db"
        card_bg   = "#ffffff"
        input_bg  = "#f0f4ff"
        label_col = "#334155"

    st.markdown(f"""
    <style>
      /* ---- base ---- */
      html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
        background-color: {bg};
        color: {text};
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
      }}
      /* sidebar */
      [data-testid="stSidebar"] {{
        background-color: {bg2};
        border-right: 1px solid {border};
      }}
      [data-testid="stSidebar"] * {{
        color: {text} !important;
      }}
      /* inputs */
      input, textarea, select {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
        border-radius: 6px !important;
      }}
      /* slider track */
      [data-testid="stSlider"] > div > div > div {{
        background: {accent} !important;
      }}
      /* headings */
      h1 {{ font-size: 2rem; font-weight: 700; color: {text}; margin-bottom: 0.2rem; }}
      h2 {{ font-size: 1.5rem; font-weight: 600; color: {text}; }}
      h3 {{ font-size: 1.2rem; font-weight: 600; color: {text}; }}
      /* labels */
      label, .stTextInput label, .stNumberInput label,
      .stSelectbox label, .stSlider label {{
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: {label_col} !important;
      }}
      /* section card */
      .param-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
      }}
      .section-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {accent};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid {border};
        padding-bottom: 0.4rem;
      }}
      /* info box */
      .info-box {{
        background: {input_bg};
        border-left: 4px solid {accent};
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: {text_muted};
        margin-top: 0.5rem;
      }}
      /* placeholder image */
      .img-placeholder {{
        background: {input_bg};
        border: 2px dashed {border};
        border-radius: 10px;
        padding: 2.5rem 1rem;
        text-align: center;
        color: {text_muted};
        font-size: 0.95rem;
        margin-bottom: 1rem;
      }}
      /* steady state table */
      .ss-table td, .ss-table th {{
        padding: 0.4rem 0.8rem;
        border-bottom: 1px solid {border};
        font-size: 0.95rem;
      }}
      .ss-table th {{ color: {accent}; font-weight: 600; }}
      /* learning card */
      .learn-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
      }}
      .learn-card h4 {{
        color: {accent};
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
      }}
      .learn-card p {{ font-size: 0.95rem; line-height: 1.7; color: {text}; }}
      .learn-card .up {{ color: #22c55e; font-weight: 600; }}
      .learn-card .down {{ color: #ef4444; font-weight: 600; }}
      .learn-card .warn {{
        background: rgba(239,68,68,0.1);
        border-left: 3px solid #ef4444;
        padding: 0.5rem 0.8rem;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #ef4444;
        margin-top: 0.6rem;
      }}
      /* tab styling */
      [data-baseweb="tab"] {{
        font-size: 1rem;
        font-weight: 600;
      }}
      /* number input arrows */
      [data-testid="stNumberInput"] input {{ font-size: 1rem !important; }}
      /* button */
      .stButton > button {{
        background: {accent};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.55rem 1.6rem;
        cursor: pointer;
        transition: opacity 0.2s;
      }}
      .stButton > button:hover {{ opacity: 0.88; }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SECAO 3 — COMPONENTES DE LAYOUT
# =============================================================================

def card(title: str, content_fn):
    """Envolve um bloco de conteudo num card visual."""
    st.markdown(f'<div class="param-card"><div class="section-title">{title}</div>', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def image_placeholder(label: str):
    st.markdown(
        f'<div class="img-placeholder">'
        f'Espaco reservado para imagem<br><strong>{label}</strong><br>'
        f'Faca o upload do arquivo abaixo</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(f"Upload — {label}", type=["png", "jpg", "jpeg", "svg"],
                                 key=f"upload_{label[:20]}", label_visibility="collapsed")
    if uploaded:
        st.image(uploaded, use_container_width=True)


# =============================================================================
# SECAO 4 — SIDEBAR
# =============================================================================

def render_sidebar() -> bool:
    """Renderiza a sidebar e retorna True se modo escuro estiver ativo."""
    with st.sidebar:
        st.markdown("## Simulador MIT")
        st.markdown("Modelo 0dq de Krause — Integracao RK4")
        st.divider()

        dark_mode = st.toggle("Modo Escuro", value=True, key="dark_mode")
        st.divider()

        st.markdown("### Sobre")
        st.markdown(
            "Simulador didatico da **Maquina de Inducao Trifasica** "
            "(gaiola de esquilo). Baseado no modelo de Krause no "
            "referencial arbitrario dq, com integracao numerica por "
            "Runge-Kutta de 4a ordem (scipy.odeint)."
        )

    return dark_mode


# =============================================================================
# SECAO 5 — PARAMETROS DA MAQUINA
# =============================================================================

def render_machine_params() -> MachineParams:
    """Formulario completo de parametros fisicos da maquina."""

    st.markdown("## Parametros da Maquina")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Parametros Eletricos</div>', unsafe_allow_html=True)

        Vl  = st.number_input("Tensao de linha RMS — Vl (V)",
                               min_value=50.0, max_value=15000.0, value=220.0, step=1.0)
        f   = st.number_input("Frequencia da rede — f (Hz)",
                               min_value=1.0, max_value=400.0, value=60.0, step=1.0)
        Rs  = st.number_input("Resistencia do estator — Rs (Ohm)",
                               min_value=0.001, max_value=100.0, value=0.435, step=0.001, format="%.3f")
        Rr  = st.number_input("Resistencia do rotor (ref. estator) — Rr (Ohm)",
                               min_value=0.001, max_value=100.0, value=0.816, step=0.001, format="%.3f")
        Xm  = st.number_input("Reatancia de magnetizacao — Xm (Ohm)",
                               min_value=0.1, max_value=500.0, value=26.13, step=0.01, format="%.2f")
        Xls = st.number_input("Reatancia de dispersao do estator — Xls (Ohm)",
                               min_value=0.001, max_value=50.0, value=0.754, step=0.001, format="%.3f")
        Xlr = st.number_input("Reatancia de dispersao do rotor — Xlr (Ohm)",
                               min_value=0.001, max_value=50.0, value=0.754, step=0.001, format="%.3f")

    with col_r:
        st.markdown('<div class="section-title">Parametros Mecanicos e Referencial</div>', unsafe_allow_html=True)

        p   = st.selectbox("Numero de polos — p", options=[2, 4, 6, 8, 10, 12], index=1)
        J   = st.number_input("Momento de inercia — J (kg.m2)",
                               min_value=0.001, max_value=100.0, value=0.089, step=0.001, format="%.3f")
        B   = st.number_input("Atrito viscoso — B (N.m.s/rad)",
                               min_value=0.0, max_value=10.0, value=0.0, step=0.001, format="%.3f")

        ref_code = st.selectbox(
            "Referencial da Transformada de Park",
            options=["Sincrono (w_ref = we)", "Rotorico (w_ref = wr)", "Estacionario (w_ref = 0)"],
            index=0,
        )
        ref_map = {"Sincrono (w_ref = we)": 1, "Rotorico (w_ref = wr)": 2, "Estacionario (w_ref = 0)": 3}

        st.markdown("")
        st.markdown('<div class="section-title">Diagrama do Circuito Equivalente</div>', unsafe_allow_html=True)
        image_placeholder("Circuito Equivalente Monofasico da MIT")

    mp = MachineParams(Vl=Vl, f=f, Rs=Rs, Rr=Rr, Xm=Xm, Xls=Xls, Xlr=Xlr, p=p, J=J, B=B)

    # Grandezas derivadas como informacao
    col1, col2, col3 = st.columns(3)
    col1.metric("Velocidade Sincrona", f"{mp.n_sync:.1f} RPM")
    col2.metric("Velocidade Angular Base", f"{mp.wb:.2f} rad/s")
    col3.metric("Reatancia Mutua Equivalente (Xml)", f"{mp.Xml:.4f} Ohm")

    return mp, ref_map[ref_code]


# =============================================================================
# SECAO 6 — CONFIGURACAO DO EXPERIMENTO
# =============================================================================

def render_experiment_config(mp: MachineParams):
    """Selecao do tipo de experimento e parametros de torque/tensao."""

    st.markdown("## Configuracao do Experimento")

    exp_options = {
        "Partida Direta (DOL)": "dol",
        "Partida Estrela-Triangulo (Y-D)": "yd",
        "Partida com Autotransformador (Compensadora)": "comp",
        "Soft-Starter (Rampa de Tensao)": "soft",
        "Aplicacao de Carga Nominal em Vazio": "carga_nom",
        "Aplicacao de 50% da Carga Nominal": "carga_50",
        "Sobrecarga (120% do Nominal)": "sobrecarga",
        "Operacao como Gerador": "gerador",
    }

    exp_label = st.selectbox("Tipo de Experimento", options=list(exp_options.keys()))
    exp_type  = exp_options[exp_label]

    config = {"exp_type": exp_type}

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Parametros de Carga e Tensao</div>', unsafe_allow_html=True)

        if exp_type == "dol":
            config["Tl_final"] = st.number_input("Torque de carga nominal (N.m)", value=80.0, min_value=0.0)
            config["t_carga"]  = st.number_input("Instante de aplicacao da carga (s)", value=0.1, min_value=0.0)

        elif exp_type == "yd":
            config["t_2"]     = st.number_input("Instante de comutacao Y -> D (s)", value=0.5, min_value=0.01)
            config["t_carga"] = st.number_input("Instante de aplicacao da carga (s)", value=0.1, min_value=0.0)
            info_box("A tensao em estrela e reduzida a Vl / raiz(3). A comutacao para triangulo ocorre em t_2.")

        elif exp_type == "comp":
            config["voltage_ratio"] = st.slider("Tap do autotransformador (%)", 30, 90, 50) / 100.0
            config["t_2"]           = st.number_input("Instante de comutacao (s)", value=0.5, min_value=0.01)
            config["t_carga"]       = st.number_input("Instante de aplicacao da carga (s)", value=0.1, min_value=0.0)
            info_box(f"Tensao inicial = {config['voltage_ratio']*100:.0f}% de Vl nominal.")

        elif exp_type == "soft":
            config["voltage_ratio"] = st.slider("Tensao inicial do soft-starter (%)", 10, 90, 50) / 100.0
            config["t_2"]           = st.number_input("Inicio da rampa de tensao (s)", value=0.9, min_value=0.0)
            config["t_pico"]        = st.number_input("Tempo para atingir tensao nominal (s)", value=5.0, min_value=0.1)
            config["Tl_final"]      = st.number_input("Torque de carga (N.m)", value=80.0, min_value=0.0)
            config["t_carga"]       = st.number_input("Instante de aplicacao da carga (s)", value=0.1, min_value=0.0)

        elif exp_type in ("carga_nom", "carga_50", "sobrecarga"):
            labels = {"carga_nom": 80.0, "carga_50": 40.0, "sobrecarga": 96.0}
            config["Tl_final"] = labels[exp_type]
            config["t_carga"]  = st.number_input("Instante de aplicacao da carga (s)", value=1.0, min_value=0.0)
            info_box(f"Torque de carga fixo: {config['Tl_final']:.1f} N.m")

        elif exp_type == "gerador":
            config["Tl_mec"] = st.number_input("Torque mecanico da turbina (N.m)", value=80.0, min_value=1.0)
            config["t_2"]    = st.number_input("Instante de aplicacao do torque (s)", value=1.0, min_value=0.0)
            info_box("O torque negativo impulsiona o rotor acima da velocidade sincrona, entrando em modo gerador.")

    with col_b:
        st.markdown('<div class="section-title">Esquema do Experimento</div>', unsafe_allow_html=True)
        image_placeholder(f"Diagrama — {exp_label}")

    return config


# =============================================================================
# SECAO 7 — SELECAO DE VARIAVEIS PARA PLOTAGEM
# =============================================================================

VARIABLE_CATALOG = {
    "Torque Eletromagnetico Te (N.m)": "Te",
    "Velocidade do Rotor n (RPM)": "n",
    "Velocidade Angular wr (rad/s)": "wr",
    "Corrente de Fase A — Estator ias (A)": "ias",
    "Corrente de Fase B — Estator ibs (A)": "ibs",
    "Corrente de Fase C — Estator ics (A)": "ics",
    "Corrente de Fase A — Rotor iar (A)": "iar",
    "Corrente de Fase B — Rotor ibr (A)": "ibr",
    "Corrente de Fase C — Rotor icr (A)": "icr",
    "Componente dq — ids (A)": "ids",
    "Componente dq — iqs (A)": "iqs",
    "Componente dq — idr (A)": "idr",
    "Componente dq — iqr (A)": "iqr",
    "Tensao de Fase Va (V)": "Va",
    "Tensao de Fase Vb (V)": "Vb",
    "Tensao de Fase Vc (V)": "Vc",
    "Tensao dq — Vds (V)": "Vds",
    "Tensao dq — Vqs (V)": "Vqs",
}

def render_variable_selector():
    """Permite ao usuario escolher quais grandezas serao plotadas."""
    st.markdown("## Variaveis para Visualizacao")
    selected = st.multiselect(
        "Selecione as grandezas que deseja plotar",
        options=list(VARIABLE_CATALOG.keys()),
        default=["Torque Eletromagnetico Te (N.m)", "Velocidade do Rotor n (RPM)",
                 "Corrente de Fase A — Estator ias (A)"],
    )
    return [VARIABLE_CATALOG[v] for v in selected], selected


# =============================================================================
# SECAO 8 — CONFIGURACAO DE TEMPO E PASSO
# =============================================================================

def render_time_config():
    """Tempo de simulacao e passo de integracao com explicacao didatica."""
    st.markdown("## Parametros de Simulacao")

    col_params, col_info = st.columns([1, 1])

    with col_params:
        tmax = st.number_input(
            "Tempo total de simulacao — tmax (s)",
            min_value=0.1, max_value=60.0, value=2.0, step=0.1, format="%.1f"
        )
        h = st.number_input(
            "Passo de integracao — h (s)",
            min_value=0.00001, max_value=0.01, value=0.001,
            step=0.0001, format="%.5f"
        )

    with col_info:
        info_box(
            "<strong>Como esses parametros afetam a simulacao:</strong><br><br>"
            "<strong>Tempo total (tmax):</strong> determina por quanto tempo o transitorio e observado. "
            "Valores muito curtos podem nao mostrar o regime permanente; valores excessivos aumentam "
            "drasticamente o tempo de processamento.<br><br>"
            "<strong>Passo de integracao (h):</strong> controla a precisao numerica e a estabilidade. "
            "Passos muito grandes (h &gt; 0.005 s) podem gerar instabilidade numerica "
            "e resultados fisicamente incorretos. "
            "Passos muito pequenos (h &lt; 0.0001 s) aumentam o tempo de computo "
            "sem ganho pratico de precisao. "
            "O valor padrao (h = 0.001 s) oferece bom equilibrio para a maioria dos experimentos."
        )

    n_steps = int(tmax / h)
    st.caption(f"Total de passos de integracao: {n_steps:,}")
    if n_steps > 100_000:
        st.warning("Numero elevado de passos. A simulacao pode demorar varios segundos.")

    return tmax, h


# =============================================================================
# SECAO 9 — PLOTAGEM COM PLOTLY
# =============================================================================

def build_plotly_figure(
    res: dict,
    var_keys: list[str],
    var_labels: list[str],
    dark_plot: bool,
    t_events: list[float] = None,
) -> go.Figure:
    """
    Constroi um grafico Plotly interativo com as variaveis selecionadas.
    Cada variavel ocupa seu proprio subgrafico para leitura clara.
    """
    n_vars = len(var_keys)
    if n_vars == 0:
        return go.Figure()

    plot_bg   = "#0e1117" if dark_plot else "#ffffff"
    paper_bg  = "#161b2e" if dark_plot else "#f8faff"
    font_col  = "#e8eaf0" if dark_plot else "#1a1d2e"
    grid_col  = "#2e3250" if dark_plot else "#e0e6f0"
    line_colors = [
        "#4f8ef7", "#f97316", "#22c55e", "#a78bfa",
        "#ec4899", "#14b8a6", "#f59e0b", "#6366f1",
        "#84cc16", "#ef4444", "#06b6d4", "#d946ef",
    ]

    fig = make_subplots(
        rows=n_vars, cols=1,
        shared_xaxes=True,
        subplot_titles=var_labels,
        vertical_spacing=0.06 / max(n_vars, 1),
    )

    t = res["t"]

    for i, (key, label) in enumerate(zip(var_keys, var_labels), start=1):
        y = res[key]
        col = line_colors[(i - 1) % len(line_colors)]
        fig.add_trace(
            go.Scatter(
                x=t, y=y,
                mode="lines",
                name=label,
                line=dict(color=col, width=1.8),
                hovertemplate=f"<b>{label}</b><br>t = %{{x:.4f}} s<br>y = %{{y:.4f}}<extra></extra>",
            ),
            row=i, col=1,
        )
        if t_events:
            for te in t_events:
                fig.add_vline(
                    x=te, line_dash="dash", line_color="#94a3b8", line_width=1.2,
                    row=i, col=1,
                )
        fig.update_yaxes(
            row=i, col=1,
            showgrid=True, gridcolor=grid_col, gridwidth=0.5,
            zeroline=True, zerolinecolor=grid_col, zerolinewidth=1,
            tickfont=dict(size=11, color=font_col),
            title_font=dict(size=11, color=font_col),
        )

    fig.update_xaxes(
        showgrid=True, gridcolor=grid_col, gridwidth=0.5,
        tickfont=dict(size=11, color=font_col),
        title_text="Tempo (s)",
        row=n_vars, col=1,
    )

    total_height = max(320, 220 * n_vars)
    fig.update_layout(
        height=total_height,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color=font_col),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode="x unified",
    )

    # Anotacoes dos subplot titles com cor correta
    for annotation in fig.layout.annotations:
        annotation.font.color = font_col
        annotation.font.size  = 12

    return fig


def render_results(res: dict, var_keys: list, var_labels: list, dark_plot: bool, t_events=None):
    """Exibe graficos e tabela de regime permanente."""
    st.markdown("### Resultados da Simulacao")

    # Controle do fundo do grafico (independente do tema geral)
    dark_plot = st.toggle("Fundo escuro na area de plotagem", value=dark_plot, key="plot_bg_toggle")

    fig = build_plotly_figure(res, var_keys, var_labels, dark_plot, t_events)
    st.plotly_chart(fig, use_container_width=True)

    # Tabela de regime permanente
    st.markdown("### Valores de Regime Permanente (ultimo instante)")
    ss = {
        "Velocidade (RPM)":         f"{res['n'][-1]:.2f}",
        "Velocidade angular (rad/s)":f"{res['wr'][-1]:.4f}",
        "Torque Te (N.m)":          f"{res['Te'][-1]:.4f}",
        "ids (A)":                  f"{res['ids'][-1]:.4f}",
        "iqs (A)":                  f"{res['iqs'][-1]:.4f}",
        "ias (A)":                  f"{res['ias'][-1]:.4f}",
    }
    col1, col2, col3 = st.columns(3)
    items = list(ss.items())
    for idx, (k, v) in enumerate(items):
        [col1, col2, col3][idx % 3].metric(k, v)


# =============================================================================
# SECAO 10 — EXECUCAO DA SIMULACAO
# =============================================================================

def build_voltage_and_torque_fns(config: dict, mp: MachineParams):
    """Constroi as funcoes de tensao e torque conforme o experimento."""
    exp = config["exp_type"]
    t_events = []

    if exp == "dol":
        vfn = lambda t: mp.Vl
        tfn = lambda t: torque_step(t, 0.0, config["Tl_final"], config["t_carga"])
        t_events = [config["t_carga"]]

    elif exp == "yd":
        Vl_Y = mp.Vl / np.sqrt(3.0)
        vfn  = lambda t: voltage_reduced_start(t, mp.Vl, Vl_Y, config["t_2"])
        tfn  = lambda t: torque_step(t, 0.0, 80.0, config["t_carga"])
        t_events = [config["t_2"], config["t_carga"]]

    elif exp == "comp":
        Vl_red = mp.Vl * config["voltage_ratio"]
        vfn    = lambda t: voltage_reduced_start(t, mp.Vl, Vl_red, config["t_2"])
        tfn    = lambda t: torque_step(t, 0.0, 80.0, config["t_carga"])
        t_events = [config["t_2"], config["t_carga"]]

    elif exp == "soft":
        Vl_init = mp.Vl * config["voltage_ratio"]
        vfn = lambda t: voltage_soft_starter(t, mp.Vl, Vl_init, config["t_2"], config["t_pico"])
        tfn = lambda t: torque_step(t, 0.0, config["Tl_final"], config["t_carga"])
        t_events = [config["t_2"], config["t_pico"], config["t_carga"]]

    elif exp in ("carga_nom", "carga_50", "sobrecarga"):
        vfn = lambda t: mp.Vl
        tfn = lambda t: torque_step(t, 0.0, config["Tl_final"], config["t_carga"])
        t_events = [config["t_carga"]]

    elif exp == "gerador":
        Tl_neg = -config["Tl_mec"]
        vfn = lambda t: mp.Vl
        tfn = lambda t: Tl_neg
        t_events = [config["t_2"]]

    else:
        vfn = lambda t: mp.Vl
        tfn = lambda t: 0.0

    return vfn, tfn, t_events


# =============================================================================
# SECAO 11 — ABA DE APRENDIZADO
# =============================================================================

def render_learning_tab():
    """Aba educacional: parametros, efeitos fisicos, instabilidades."""
    st.markdown("# Aba de Aprendizado")
    st.markdown(
        "Esta aba apresenta a influencia fisica de cada parametro no comportamento "
        "da maquina de inducao trifasica, sem exibir valores numericos especificos. "
        "O objetivo e construir intuicao sobre o sistema antes ou apos a simulacao."
    )

    # ---- PARAMETROS ELETRICOS ----
    st.markdown("## Parametros Eletricos")

    params_eletricos = [
        {
            "nome": "Vl — Tensao de Linha RMS",
            "desc": "A tensao aplicada ao estator define o fluxo de magnetizacao e, portanto, "
                    "o torque disponivel. E a fonte de energia que aciona a maquina.",
            "aumenta": "O torque maximo cresce proporcionalmente ao quadrado da tensao. "
                        "Correntes de partida tambem aumentam significativamente.",
            "diminui": "O torque de partida cai, e pode ser insuficiente para vencer a carga. "
                        "O motor pode nao conseguir partir (nao sai do lugar).",
            "risco": "Tensao muito elevada causa saturacao magnetica do nucleo, "
                     "aquecimento excessivo e possivel dano ao isolamento. "
                     "Tensao muito baixa pode provocar parada brusca sob carga (stall).",
        },
        {
            "nome": "f — Frequencia da Rede",
            "desc": "A frequencia determina a velocidade sincrona do campo girante. "
                    "Toda a dinamica da maquina e referenciada a ela.",
            "aumenta": "A velocidade sincrona sobe proporcionalmente. "
                        "As reatancias Xm, Xls, Xlr tambem crescem (X = 2pi.f.L), "
                        "alterando o circuito equivalente.",
            "diminui": "Menor velocidade sincrona e menor velocidade de operacao. "
                        "Com tensao constante, o fluxo aumenta (V/f aumenta), "
                        "podendo causar saturacao.",
            "risco": "Operar fora da frequencia nominal sem ajuste de tensao (controle V/f) "
                     "resulta em saturacao magnetica ou fluxo insuficiente, "
                     "comprometendo eficiencia e vida util.",
        },
        {
            "nome": "Rs — Resistencia do Estator",
            "desc": "Representa as perdas ohmica nas bobinas do estator. "
                    "Afeta a tensao efetiva que chega ao circuito magnetico.",
            "aumenta": "Maior queda de tensao no estator, menor tensao efetiva nos terminais magneticos. "
                        "Reducao do torque disponivel e aumento das perdas por calor.",
            "diminui": "Melhor eficiencia e menor queda de tensao interna. "
                        "O modelo fica mais proximo de um transformador ideal.",
            "risco": "Rs muito alto (resistencia elevada por desgaste ou falha de bobina) "
                     "provoca superaquecimento e queda de desempenho. "
                     "Rs proximo de zero pode gerar oscilacoes numericas na simulacao "
                     "em passos de integracao grandes.",
        },
        {
            "nome": "Rr — Resistencia do Rotor",
            "desc": "Representa as perdas ohmicas nas barras do rotor (gaiola de esquilo). "
                    "E o parametro que mais influencia o escorregamento e o torque de partida.",
            "aumenta": "O escorregamento de regime aumenta (rotor gira mais devagar). "
                        "O torque de partida aumenta ate o ponto maximo de Rr. "
                        "A curva T x n se alarga.",
            "diminui": "Menor escorregamento em regime (mais eficiente). "
                        "Menor torque de partida. "
                        "A curva T x n se estreita proximo a velocidade sincrona.",
            "risco": "Rr muito alto (gaiola com barras fraturadas) gera superaquecimento do rotor "
                     "e escorregamento excessivo. "
                     "Rr muito proximo de zero causa instabilidade numerica (denominadores pequenos nas EDOs).",
        },
        {
            "nome": "Xm — Reatancia de Magnetizacao",
            "desc": "Representa o ramo magnetizante do circuito equivalente. "
                    "Controla a corrente de excitacao e o fluxo de entreferro.",
            "aumenta": "Menor corrente de magnetizacao (motor mais eficiente, fator de potencia melhor). "
                        "O nucleo e menos saturo.",
            "diminui": "Maior corrente de magnetizacao necessaria para manter o fluxo. "
                        "Fator de potencia pior e aquecimento maior.",
            "risco": "Xm muito baixo implica maquina com alta corrente de excitacao "
                     "(motor de baixa qualidade magnetica ou nucleo saturado). "
                     "Na simulacao, Xm proximo de zero produz Xml tendendo a zero, "
                     "causando divisao por numeros muito pequenos.",
        },
        {
            "nome": "Xls / Xlr — Reatancias de Dispersao",
            "desc": "Representam os fluxos que nao se acoplam entre estator e rotor "
                    "(fluxos de dispersao). Limitam a corrente de curto-circuito e de partida.",
            "aumenta": "Maior impedancia de curto-circuito. "
                        "Corrente de partida reduzida, mas tambem menor torque maximo.",
            "diminui": "Corrente de partida maior, torque maximo maior, "
                        "mas o motor fica mais sensivel a transitories.",
            "risco": "Dispersao muito baixa resulta em correntes de partida extremamente elevadas, "
                     "potencialmente danosas ao enrolamento. "
                     "Dispersao muito alta limita o torque disponivel e pode impedir a partida.",
        },
    ]

    for p in params_eletricos:
        st.markdown(
            f'<div class="learn-card">'
            f'<h4>{p["nome"]}</h4>'
            f'<p>{p["desc"]}</p>'
            f'<p><span class="up">Se aumentar:</span> {p["aumenta"]}</p>'
            f'<p><span class="down">Se diminuir:</span> {p["diminui"]}</p>'
            f'<div class="warn">Atencao — calibracoes extremas: {p["risco"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ---- PARAMETROS MECANICOS ----
    st.markdown("## Parametros Mecanicos")

    params_mecanicos = [
        {
            "nome": "p — Numero de Polos",
            "desc": "Define a velocidade sincrona da maquina (ns = 120.f / p). "
                    "Maquinas com mais polos giram mais devagar para uma mesma frequencia.",
            "aumenta": "Menor velocidade sincrona e, portanto, menor velocidade de operacao. "
                        "Util em aplicacoes de baixa rotacao (bombas, ventiladores grandes).",
            "diminui": "Maior velocidade sincrona. "
                        "Menor torque para a mesma potencia (T = P / w).",
            "risco": "O numero de polos e um parametro discreto (sempre par). "
                     "Inserir valor impar ou zero causa erro fisico. "
                     "Simular p alto com frequencia baixa pode gerar velocidades irrealistas.",
        },
        {
            "nome": "J — Momento de Inercia",
            "desc": "Representa a resistencia do conjunto rotor-carga a variacoes de velocidade "
                    "(segunda lei de Newton rotacional: T = J . d_wr/dt).",
            "aumenta": "Aceleracao mais lenta — o transitorio se prolonga. "
                        "O sistema e mais amortecido naturalmente.",
            "diminui": "Aceleracao muito rapida. "
                        "O rotor responde quase instantaneamente aos transitorios de torque.",
            "risco": "J muito baixo pode causar oscilacoes de velocidade dificeis de controlar "
                     "em sistemas com variacao de carga. "
                     "J muito alto prolonga o transitorio a ponto de ultrapassar o tempo de simulacao.",
        },
        {
            "nome": "B — Coeficiente de Atrito Viscoso",
            "desc": "Representa perdas mecanicas proporcionais a velocidade "
                    "(rolamentos, ventilacao). Produz torque de freio B.wr.",
            "aumenta": "Maior dissipacao mecanica. "
                        "A velocidade de regime permanente e ligeiramente menor.",
            "diminui": "Menor dissipacao. "
                        "Com B = 0 (padrao), toda a resistencia ao movimento vem da carga.",
            "risco": "B muito alto pode causar parada do motor sem carga nominal "
                     "(perdas mecanicas superando o torque eletromagnetico). "
                     "Na pratica, atrito excessivo indica falha nos mancais.",
        },
    ]

    for p in params_mecanicos:
        st.markdown(
            f'<div class="learn-card">'
            f'<h4>{p["nome"]}</h4>'
            f'<p>{p["desc"]}</p>'
            f'<p><span class="up">Se aumentar:</span> {p["aumenta"]}</p>'
            f'<p><span class="down">Se diminuir:</span> {p["diminui"]}</p>'
            f'<div class="warn">Atencao — calibracoes extremas: {p["risco"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ---- PARAMETROS DE SIMULACAO ----
    st.markdown("## Parametros de Simulacao")

    sim_params = [
        {
            "nome": "tmax — Tempo Total de Simulacao",
            "desc": "Define o horizonte temporal da integracao numerica.",
            "aumenta": "Permite observar o regime permanente e transitorios mais longos. "
                        "Aumenta proporcionalmente o tempo de processamento.",
            "diminui": "Simulacao mais rapida, mas pode nao capturar o regime permanente.",
            "risco": "tmax excessivamente grande com passo pequeno pode travar o navegador "
                     "por consumo de memoria e tempo de CPU.",
        },
        {
            "nome": "h — Passo de Integracao",
            "desc": "Controla a discretizacao temporal do metodo Runge-Kutta. "
                    "E o parametro mais critico para estabilidade numerica.",
            "aumenta": "Simulacao mais rapida, mas menos precisa. "
                        "Passos grandes podem causar instabilidade numerica "
                        "(oscilacoes artificiais ou divergencia).",
            "diminui": "Maior precisao e estabilidade, mas tempo de computo muito maior. "
                        "Para a MIT com os parametros padrao, h < 0.0001 s rara vez e necessario.",
            "risco": "h > 0.005 s pode gerar resultados fisicamente invalidos (torques divergentes). "
                     "A regra pratica e h <= 1 / (10 . wb) para garantir estabilidade "
                     "no referencial sincrono.",
        },
    ]

    for p in sim_params:
        st.markdown(
            f'<div class="learn-card">'
            f'<h4>{p["nome"]}</h4>'
            f'<p>{p["desc"]}</p>'
            f'<p><span class="up">Se aumentar:</span> {p["aumenta"]}</p>'
            f'<p><span class="down">Se diminuir:</span> {p["diminui"]}</p>'
            f'<div class="warn">Atencao — calibracoes extremas: {p["risco"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ---- CONCEITOS DO MODELO ----
    st.markdown("## Conceitos do Modelo Matematico")

    st.markdown(
        '<div class="learn-card">'
        '<h4>Transformada de Park e Referencial dq</h4>'
        '<p>A transformada de Park converte as tres fases da maquina em dois eixos ortogonais '
        '(d e q), eliminando a dependencia temporal dos coeficientes das equacoes diferenciais. '
        'Isso simplifica drasticamente a analise e permite integracao numerica eficiente.</p>'
        '<p><span class="up">Referencial Sincrono (w_ref = we):</span> '
        'em regime permanente, Vds e Vqs sao constantes (DC). '
        'Ideal para analise de regime permanente e controle vetorial.</p>'
        '<p><span class="up">Referencial Rotorico (w_ref = wr):</span> '
        'as variaveis do rotor tornam-se DC em regime. '
        'Util para controle de campo orientado pelo rotor.</p>'
        '<p><span class="up">Referencial Estacionario (w_ref = 0):</span> '
        'as variaveis variam na frequencia de escorregamento. '
        'Mais intuitivo fisicamente para analise de transitorio.</p>'
        '<div class="warn">Os tres referenciais produzem o mesmo torque Te e velocidade wr '
        '(grandezas fisicas invariantes). Apenas as formas de onda das correntes e tensoes dq diferem.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="learn-card">'
        '<h4>Escorregamento (Slip)</h4>'
        '<p>O escorregamento e a diferenca relativa entre a velocidade sincrona e a velocidade do rotor. '
        'Em motores, o rotor sempre gira abaixo da velocidade sincrona para que haja inducao e, '
        'portanto, torque eletromagnetico.</p>'
        '<p><span class="up">Escorregamento baixo:</span> maquina proxima ao regime nominal, '
        'alta eficiencia, baixas perdas no rotor.</p>'
        '<p><span class="down">Escorregamento alto (partida):</span> correntes elevadas, '
        'maior dissipacao termica no rotor, torque varia significativamente.</p>'
        '<div class="warn">Escorregamento negativo significa que o rotor gira acima da velocidade sincrona: '
        'a maquina opera como gerador (modo gerador de inducao).</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="learn-card">'
        '<h4>Metodos de Partida</h4>'
        '<p>A corrente de partida direta (DOL) pode atingir 6 a 8 vezes a corrente nominal, '
        'causando quedas de tensao na rede e tensoes mecanicas no acoplamento.</p>'
        '<p><span class="up">Y-Delta:</span> reduz a corrente de partida a aproximadamente 1/3 do valor DOL, '
        'mas tambem reduz o torque de partida na mesma proporcao.</p>'
        '<p><span class="up">Compensadora (autotransformador):</span> permite escolher o tap '
        'de reducao de tensao, oferecendo um compromisso entre corrente e torque de partida.</p>'
        '<p><span class="up">Soft-Starter:</span> rampa de tensao controlada eletronicamente. '
        'Oferece a transicao mais suave, mas prolonga o transitorio.</p>'
        '<div class="warn">Qualquer metodo de reducao de tensao na partida tambem reduz o torque disponivel. '
        'Se o torque de partida for insuficiente para vencer a carga, o motor nao parte '
        'e superaquece rapidamente.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# SECAO 12 — PONTO DE ENTRADA PRINCIPAL
# =============================================================================

def main():
    # 1. Sidebar e tema
    dark_mode = render_sidebar()
    apply_theme(dark_mode)

    # 2. Cabecalho
    st.markdown("# Simulador de Maquina de Inducao Trifasica")
    st.markdown(
        "Modelo matematico **0dq de Krause** — Gaiola de Esquilo — "
        "Integracao numerica via **RK4** (scipy.odeint)"
    )
    st.divider()

    # 3. Abas principais
    tab_sim, tab_learn = st.tabs(["Simulacao", "Aprendizado"])

    with tab_sim:
        # 3a. Parametros da maquina
        mp, ref_code = render_machine_params()
        st.divider()

        # 3b. Configuracao do experimento
        exp_config = render_experiment_config(mp)
        st.divider()

        # 3c. Selecao de variaveis
        var_keys, var_labels = render_variable_selector()
        st.divider()

        # 3d. Tempo e passo
        tmax, h = render_time_config()
        st.divider()

        # 3e. Botao de execucao
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_btn = st.button("Executar Simulacao", use_container_width=True)

        if run_btn:
            if not var_keys:
                st.warning("Selecione ao menos uma variavel para plotar.")
            else:
                vfn, tfn, t_events = build_voltage_and_torque_fns(exp_config, mp)
                with st.spinner("Executando integracao numerica..."):
                    try:
                        res = run_simulation(
                            mp=mp, tmax=tmax, h=h,
                            voltage_fn=vfn, torque_fn=tfn,
                            ref_code=ref_code,
                        )
                        st.success(
                            f"Simulacao concluida. "
                            f"n_final = {res['n'][-1]:.1f} RPM | "
                            f"Te_final = {res['Te'][-1]:.2f} N.m"
                        )
                        render_results(res, var_keys, var_labels, dark_mode, t_events)
                    except Exception as e:
                        st.error(f"Erro na simulacao: {e}")
                        st.info(
                            "Verifique os parametros. Passos de integracao muito grandes "
                            "ou parametros fisicamente invalidos podem causar divergencia numerica."
                        )

    with tab_learn:
        render_learning_tab()


if __name__ == "__main__":
    main()
