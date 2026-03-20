# -*- coding: utf-8 -*-
"""EMS v2 — Simulador de Máquinas Elétricas
Tela 1: Seleção da máquina  |  Tela 2: Parâmetros + Solver
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from math import pi, sqrt

# ══════════════════════════════════════════════════════════════════════════════
# Configuração da página e CSS global
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EMS — Simulador de Máquinas Elétricas",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }

.machine-card {
    background: linear-gradient(145deg, #0d2137 0%, #1a3a5c 100%);
    border: 2px solid #2d5a8e;
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    min-height: 260px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    transition: border-color .2s, box-shadow .2s;
}
.machine-card:hover {
    border-color: #4a90d9;
    box-shadow: 0 6px 24px rgba(74,144,217,.3);
}
.machine-card h3 { color: #e0e8f0; margin: 0; font-size: 15px; font-weight: 600; }
.machine-card p  { color: #7a9ab8; margin: 0; font-size: 12px; }

.machine-card-locked {
    background: #161b22;
    border: 2px solid #2a2a2a;
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    min-height: 260px;
    opacity: .42;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
}
.machine-card-locked h3 { color: #555; margin: 0; font-size: 15px; font-weight: 600; }
.machine-card-locked p  { color: #444; margin: 0; font-size: 12px; }

.section-header {
    background: linear-gradient(90deg, #0d2137 0%, transparent 100%);
    border-left: 4px solid #2d78c8;
    padding: 7px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 14px;
    color: #b0ccec;
    font-weight: 600;
    font-size: 13px;
}

.param-info {
    background: #08111c;
    border-left: 3px solid #2d78c8;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: -4px 0 14px 0;
    font-size: 12px;
    color: #7aaac8;
    line-height: 1.55;
}

.solver-banner-wait {
    background: #14191f;
    border: 1px solid #2d3a50;
    border-radius: 10px;
    padding: 6px 12px;
    text-align: center;
    color: #4a5a70;
    font-size: 12px;
    margin-bottom: 8px;
}
.solver-banner-ok {
    background: linear-gradient(135deg, #0a2a0a, #163a16);
    border: 2px solid #2ecc71;
    border-radius: 10px;
    padding: 6px 12px;
    text-align: center;
    color: #2ecc71;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
    box-shadow: 0 0 16px rgba(46,204,113,.25);
}

.derived-box {
    background: #0a1929;
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 12px;
    color: #5a8ab0;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "page":             "home",
    "machine":          None,
    "learning_mode":    False,
    "params_confirmed": False,
    "machine_params":   {},
    "sim_results":      None,
    "sim_experiment":   "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# SVG — Máquina de Indução Trifásica
# ══════════════════════════════════════════════════════════════════════════════
MIT_SVG = """
<svg viewBox="0 0 200 160" width="155" height="124" xmlns="http://www.w3.org/2000/svg">
  <rect x="28" y="44" width="135" height="82" rx="13" fill="#162d45"/>
  <rect x="34" y="50" width="123" height="70" rx="8"  fill="#1e4070"/>
  <rect x="46" y="34" width="9"  height="13" rx="2" fill="#122338"/>
  <rect x="64" y="31" width="9"  height="16" rx="2" fill="#122338"/>
  <rect x="82" y="29" width="9"  height="18" rx="2" fill="#122338"/>
  <rect x="100" y="29" width="9" height="18" rx="2" fill="#122338"/>
  <rect x="118" y="31" width="9" height="16" rx="2" fill="#122338"/>
  <rect x="136" y="34" width="9" height="13" rx="2" fill="#122338"/>
  <line x1="57"  y1="55" x2="57"  y2="115" stroke="#3a80c0" stroke-width="2.5" stroke-dasharray="5,3"/>
  <line x1="77"  y1="52" x2="77"  y2="118" stroke="#3a80c0" stroke-width="2.5" stroke-dasharray="5,3"/>
  <line x1="97"  y1="51" x2="97"  y2="119" stroke="#64b5f6" stroke-width="3"/>
  <line x1="117" y1="52" x2="117" y2="118" stroke="#3a80c0" stroke-width="2.5" stroke-dasharray="5,3"/>
  <line x1="137" y1="55" x2="137" y2="115" stroke="#3a80c0" stroke-width="2.5" stroke-dasharray="5,3"/>
  <rect x="0"   y="78" width="30" height="14" rx="4" fill="#546e7a"/>
  <rect x="160" y="78" width="30" height="14" rx="4" fill="#546e7a"/>
  <circle cx="97" cy="85" r="22" fill="#122338" stroke="#2d6cb8" stroke-width="2"/>
  <circle cx="97" cy="85" r="13" fill="#0a1929" stroke="#1a4a7a" stroke-width="1.5"/>
  <text x="87" y="90" font-family="Arial" font-size="14" fill="#ffd54f" font-weight="bold">3~</text>
  <rect x="68" y="126" width="58" height="16" rx="4" fill="#122338" stroke="#2d5a8e" stroke-width="1"/>
  <text x="97" y="138" font-family="Arial" font-size="10" fill="#7aaac8" text-anchor="middle">MIT</text>
</svg>
"""

# ══════════════════════════════════════════════════════════════════════════════
# Descrições — Modo de Aprendizado
# ══════════════════════════════════════════════════════════════════════════════
PARAM_DESC = {
    "Vl":  "Tensão de linha RMS. O torque máximo escala com Vl² — reduzi-la à metade "
           "cai para ¼ do torque máximo. Métodos de partida suave (Y-Δ, soft-starter) "
           "exploram exatamente esse comportamento.",
    "f":   "Frequência da rede. Determina a velocidade síncrona (nₛ = 120·f/p) e escala "
           "todas as reatâncias (X = ω·L). Em controle V/f constante, Vl acompanha f "
           "para manter o fluxo de entreferro estável.",
    "Rs":  "Resistência do estator por fase. Causa queda de tensão interna e perdas Joule "
           "(P = 3·Rs·Is²). Valores elevados reduzem a tensão disponível para o entreferro "
           "e, consequentemente, o torque máximo.",
    "Rr":  "Resistência do rotor referida ao estator. Controla o escorregamento em regime "
           "(s ≈ Rr/Xₜₒₜ) e o torque de partida. Rr maior → mais torque de partida, "
           "porém maior escorregamento e menor rendimento.",
    "Xm":  "Reatância de magnetização. Representa o fluxo útil que atravessa o entreferro. "
           "Xm maior → menor corrente de magnetização em vazio e fator de potência melhor.",
    "Xls": "Reatância de dispersão do estator. Fluxo que não cruza o entreferro. "
           "Junto com Xlr define s_Tmax = Rr / √(Rs² + (Xls+Xlr)²) e limita Te_max.",
    "Xlr": "Reatância de dispersão do rotor. Mesmo papel do Xls, mas no lado do rotor. "
           "A soma Xls+Xlr aparece diretamente no denominador do torque máximo.",
    "p":   "Número de polos do enrolamento. Velocidade síncrona: nₛ = 120·f/p. "
           "4 polos a 60 Hz → nₛ = 1800 rpm; 6 polos → 1200 rpm.",
    "J":   "Momento de inércia (rotor + carga). Determina a constante de tempo mecânica "
           "τ ≈ J·ωₛ/(Te−Tl). J maior → aceleração mais lenta e oscilações mais amortecidas.",
    "B":   "Atrito viscoso: torque_atrito = B·ωr. Em geral muito pequeno ou zero. "
           "Use somente se a carga apresentar amortecimento significativo.",
    "Tl_initial": "Torque de carga já presente no eixo antes de t_carga. "
                  "Representa cargas acopladas que dificultam a partida.",
    "Tl_final":   "Torque de carga nominal aplicado a partir de t_carga. "
                  "Define escorregamento e corrente em regime permanente.",
}

# ══════════════════════════════════════════════════════════════════════════════
# Física — ODE única do motor + simulador
# ══════════════════════════════════════════════════════════════════════════════
def _motor_odes(states, t, Vqs, Vds, Tl, w_ref, Rs, Rr, Xls, Xlr, Xml, wb, p, J, B):
    PSIqs, PSIds, PSIqr, PSIdr, wr = states
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    iqs_v = (1.0 / Xls) * (PSIqs - PSImq)
    ids_v = (1.0 / Xls) * (PSIds - PSImd)
    dPSIqs = wb * ( Vqs - (w_ref / wb) * PSIds  + (Rs / Xls) * (PSImq - PSIqs))
    dPSIds = wb * ( Vds + (w_ref / wb) * PSIqs  + (Rs / Xls) * (PSImd - PSIds))
    dPSIqr = wb * ( 0.0 - ((w_ref - wr) / wb) * PSIdr + (Rr / Xlr) * (PSImq - PSIqr))
    dPSIdr = wb * ( 0.0 + ((w_ref - wr) / wb) * PSIqr + (Rr / Xlr) * (PSImd - PSIdr))
    Tem    = (3.0 / 2.0) * (p / 2.0) * (1.0 / wb) * (PSIds * iqs_v - PSIqs * ids_v)
    dwr    = (p / (2.0 * J)) * (Tem - Tl) - (B / J) * wr
    return [dPSIqs, dPSIds, dPSIqr, dPSIdr, dwr]


def _simular(V_func, Tl_func, tmax, h, mp, J_override=None):
    f_m  = mp["f"]; Rs = mp["Rs"]; Rr = mp["Rr"]
    Xm   = mp["Xm"]; Xls = mp["Xls"]; Xlr = mp["Xlr"]
    p    = mp["p"];  J = J_override if J_override is not None else mp["J"]; B = mp["B"]
    wb   = 2.0 * pi * f_m
    Xml  = 1.0 / (1.0 / Xm + 1.0 / Xls + 1.0 / Xlr)
    we   = wb
    t_vals = np.arange(0.0, tmax, h)
    states = [0.0] * 5
    wr_r = []; Te_r = []; n_r = []
    ias_r = []; ibs_r = []; ics_r = []
    iar_r = []; ibr_r = []; icr_r = []

    for tv in t_vals:
        Va_v  = V_func(tv); Tl_v = Tl_func(tv)
        tetae = we * tv
        Va = sqrt(2) * Va_v * np.sin(tetae)
        Vb = sqrt(2) * Va_v * np.sin(tetae - 2.0 * pi / 3.0)
        Vc = sqrt(2) * Va_v * np.sin(tetae + 2.0 * pi / 3.0)
        Vaf = sqrt(1.5) * (Va - 0.5 * Vb - 0.5 * Vc)
        Vbt = sqrt(1.5) * ((sqrt(3.0) / 2.0) * Vb - (sqrt(3.0) / 2.0) * Vc)
        Vds = np.cos(tetae) * Vaf + np.sin(tetae) * Vbt
        Vqs = -np.sin(tetae) * Vaf + np.cos(tetae) * Vbt
        sol    = odeint(_motor_odes, states, [tv, tv + h],
                        args=(Vqs, Vds, Tl_v, we, Rs, Rr, Xls, Xlr, Xml, wb, p, J, B))
        states = sol[1]
        PSIqs, PSIds, PSIqr, PSIdr, wr = states
        PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
        PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
        ids = (1.0 / Xls) * (PSIds - PSImd); iqs = (1.0 / Xls) * (PSIqs - PSImq)
        idr = (1.0 / Xlr) * (PSIdr - PSImd); iqr = (1.0 / Xlr) * (PSIqr - PSImq)
        Te    = (3.0 / 2.0) * (p / 2.0) * (1.0 / wb) * (PSIds * iqs - PSIqs * ids)
        n_rpm = (120.0 / p) * (wr / (2.0 * pi))
        tetar = wr * tv
        iafs = np.cos(tetae) * ids - np.sin(tetae) * iqs
        ibts = np.sin(tetae) * ids + np.cos(tetae) * iqs
        iafr = np.cos(tetar) * idr - np.sin(tetar) * iqr
        ibtr = np.sin(tetar) * idr + np.cos(tetar) * iqr
        sq32 = sqrt(1.5)
        ias_r.append(sq32 * iafs)
        ibs_r.append(sq32 * ((-0.5) * iafs + (sqrt(3.0) / 2.0) * ibts))
        ics_r.append(sq32 * ((-0.5) * iafs - (sqrt(3.0) / 2.0) * ibts))
        iar_r.append(sq32 * iafr)
        ibr_r.append(sq32 * ((-0.5) * iafr + (sqrt(3.0) / 2.0) * ibtr))
        icr_r.append(sq32 * ((-0.5) * iafr - (sqrt(3.0) / 2.0) * ibtr))
        wr_r.append(wr); Te_r.append(Te); n_r.append(n_rpm)

    return {
        "t":   t_vals,
        "wr":  np.array(wr_r),  "Te":  np.array(Te_r),  "n":   np.array(n_r),
        "ias": np.array(ias_r), "ibs": np.array(ibs_r), "ics": np.array(ics_r),
        "iar": np.array(iar_r), "ibr": np.array(ibr_r), "icr": np.array(icr_r),
    }


def _mostrar_dinamico(res, titulo, mp):
    n_sync = 120.0 * mp["f"] / mp["p"]
    t = res["t"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    if titulo:
        fig.suptitle(titulo, fontsize=13, fontweight="bold")
    axes[0, 0].plot(t, res["Te"])
    axes[0, 0].set_title("Torque Eletromagnético (Nm)"); axes[0, 0].set_xlabel("t (s)"); axes[0, 0].grid(True)
    axes[0, 1].plot(t, res["n"])
    axes[0, 1].axhline(n_sync, ls="--", color="gray", label=f"nₛ = {n_sync:.0f} rpm")
    axes[0, 1].set_title("Velocidade (RPM)"); axes[0, 1].set_xlabel("t (s)"); axes[0, 1].legend(); axes[0, 1].grid(True)
    for lb, arr in [("$i_{as}$", res["ias"]), ("$i_{bs}$", res["ibs"]), ("$i_{cs}$", res["ics"])]:
        axes[1, 0].plot(t, arr, label=lb)
    axes[1, 0].set_title("Correntes do Estator (A)"); axes[1, 0].set_xlabel("t (s)"); axes[1, 0].legend(); axes[1, 0].grid(True)
    for lb, arr in [("$i_{ar}$", res["iar"]), ("$i_{br}$", res["ibr"]), ("$i_{cr}$", res["icr"])]:
        axes[1, 1].plot(t, arr, label=lb)
    axes[1, 1].set_title("Correntes do Rotor (A)"); axes[1, 1].set_xlabel("t (s)"); axes[1, 1].legend(); axes[1, 1].grid(True)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)
    wb_l = 2.0 * pi * mp["f"]
    s_ss = (wb_l - float(res["wr"][-1])) / wb_l
    c1, c2, c3 = st.columns(3)
    c1.metric("Escorregamento em regime", f"{s_ss:.4f}")
    c2.metric("Velocidade final (RPM)",   f"{float(res['n'][-1]):.1f}")
    c3.metric("Torque final (Nm)",        f"{float(res['Te'][-1]):.2f}")
    st.subheader("Valores máximos (|pico|)")
    st.table(pd.DataFrame({
        "Grandeza": ["I_as (A)", "I_ar (A)", "Te (Nm)"],
        "Máximo": [
            round(float(np.max(np.abs(res["ias"]))), 3),
            round(float(np.max(np.abs(res["iar"]))), 3),
            round(float(np.max(res["Te"])),          3),
        ],
    }).set_index("Grandeza"))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers estáticos (exp 8 e 10)
# ══════════════════════════════════════════════════════════════════════════════
def _torque_curve_static(mp, Rr_val, T_load, conn):
    f_m = mp["f"]; Xm = mp["Xm"]; Xls = mp["Xls"]; Xlr = mp["Xlr"]
    Rs = mp["Rs"]; p = mp["p"]
    w_e    = 2.0 * pi * f_m
    w_sync = 4.0 * pi * f_m / p
    Vph    = mp["Vl"] / sqrt(3.0) if conn == "Y" else mp["Vl"]
    Lm  = Xm  / w_e
    Lls = max(Xls / w_e - Lm, 0.0)
    Llr = max(Xlr / w_e - Lm, 0.0)
    X1 = w_e * Lls; X2 = w_e * Llr; Xm_v = w_e * Lm
    Z1 = complex(Rs, X1); Zm = complex(0.0, Xm_v)
    Vth = Vph * (Zm / (Z1 + Zm)); Zth = (Zm * Z1) / (Z1 + Zm)
    Rth, Xth = Zth.real, Zth.imag
    s   = np.linspace(1e-4, 1.0, 5000)[::-1]
    Te  = (3.0 * abs(Vth) ** 2 * (Rr_val / s)) / (w_sync * ((Rth + Rr_val / s) ** 2 + (Xth + X2) ** 2))
    ns  = 120.0 * f_m / p
    Te_s  = (3.0 * abs(Vth) ** 2 * Rr_val) / (w_sync * ((Rth + Rr_val) ** 2 + (Xth + X2) ** 2))
    s_Tm  = float(np.clip(Rr_val / sqrt(Rth ** 2 + (Xth + X2) ** 2), 1e-4, 1.0))
    Te_m  = (3.0 * abs(Vth) ** 2 * (Rr_val / s_Tm)) / (w_sync * ((Rth + Rr_val / s_Tm) ** 2 + (Xth + X2) ** 2))
    idx   = int(np.argmin(np.abs(Te - T_load)))
    return {
        "s": s, "Te": Te, "n_rpm": (1.0 - s) * ns, "ns": ns,
        "Te_start": float(Te_s), "s_Tmax": s_Tm, "Te_max": float(Te_m),
        "n_Tmax_rpm": float((1.0 - s_Tm) * ns),
        "s_oper": float(s[idx]), "n_oper_rpm": float((1.0 - s[idx]) * ns),
    }


def _Te_freq(V, freq, s, R1, R2, X1, X2, Xm_v, p):
    ws  = 2.0 * pi * freq / (p / 2.0)
    Z1v = R1 + 1j * X1; Zmv = 1j * Xm_v; Z2v = R2 / s + 1j * X2
    Zth = Z1v + (Zmv * Z2v) / (Zmv + Z2v)
    I2  = V / Zth
    Pc  = 3.0 * abs(I2) ** 2 * (R2 / s) * (1.0 - s)
    return Pc / (ws * (1.0 - s))


# ══════════════════════════════════════════════════════════════════════════════
# TELA 1 — Seleção da máquina
# ══════════════════════════════════════════════════════════════════════════════
def screen_home():
    st.markdown("## ⚡ EMS — Simulador de Máquinas Elétricas")
    st.markdown("Selecione a máquina elétrica para configurar e iniciar a simulação.")
    st.markdown("---")

    # Catálogo — adicione novos dicionários para escalar
    MACHINES = [
        {
            "id":          "mit",
            "name":        "Máquina de Indução Trifásica",
            "description": "Simulação dinâmica via transformada de Park",
            "svg":         MIT_SVG,
            "enabled":     True,
        },
        {
            "id":          "sync",
            "name":        "Máquina Síncrona",
            "description": "Em desenvolvimento",
            "svg":         '<div style="font-size:54px;opacity:.22">🔌</div>',
            "enabled":     False,
        },
        {
            "id":          "dc",
            "name":        "Máquina de Corrente Contínua",
            "description": "Em desenvolvimento",
            "svg":         '<div style="font-size:54px;opacity:.22">⚙️</div>',
            "enabled":     False,
        },
    ]

    cols = st.columns(len(MACHINES), gap="large")
    for col, m in zip(cols, MACHINES):
        with col:
            if m["enabled"]:
                st.markdown(
                    f'<div class="machine-card">{m["svg"]}'
                    f'<h3>{m["name"]}</h3><p>{m["description"]}</p></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                if st.button("Selecionar →", key=f"sel_{m['id']}", use_container_width=True, type="primary"):
                    st.session_state.machine          = m["id"]
                    st.session_state.params_confirmed = False
                    st.session_state.sim_results      = None
                    st.session_state.page             = "params"
                    st.rerun()
            else:
                st.markdown(
                    f'<div class="machine-card-locked">{m["svg"]}'
                    f'<h3>{m["name"]}</h3><p>{m["description"]}</p>'
                    '<span style="font-size:11px;background:#252525;padding:3px 9px;'
                    'border-radius:4px;color:#555">Em breve</span></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TELA 2 — Parâmetros da máquina + painel do solver
# ══════════════════════════════════════════════════════════════════════════════
def screen_params():
    confirmed = st.session_state.params_confirmed

    # ── Cabeçalho + botão voltar ───────────────────────────────────────────────
    cb, ct = st.columns([1, 9])
    with cb:
        if st.button("← Voltar"):
            st.session_state.page             = "home"
            st.session_state.params_confirmed = False
            st.session_state.sim_results      = None
            st.rerun()
    with ct:
        st.markdown("## Máquina de Indução Trifásica")
    st.markdown("---")

    # ── Toggle modo de aprendizado ─────────────────────────────────────────────
    learning = st.toggle(
        "🎓  Modo de Aprendizado",
        value=st.session_state.learning_mode,
        help="Exibe a influência física de cada parâmetro na simulação.",
    )
    st.session_state.learning_mode = learning
    if learning:
        st.info("**Modo de aprendizado ativo** — cada parâmetro exibirá sua influência "
                "no comportamento físico da máquina.")
    st.markdown("")

    # ── Layout: parâmetros (esq) | solver (dir) ────────────────────────────────
    col_main, col_solver = st.columns([11, 5], gap="large")

    # ════════════════════════════════════════════════════════════════════════════
    # Coluna esquerda — Parâmetros da máquina
    # ════════════════════════════════════════════════════════════════════════════
    with col_main:
        st.markdown('<div class="section-header">⚡ Parâmetros Elétricos</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        Vl = c1.number_input("Vl — Tensão de linha RMS (V)", value=220.0, step=10.0,  key="p_Vl")
        f  = c2.number_input("f — Frequência (Hz)",           value=60.0,  step=1.0,   key="p_f")
        if learning:
            st.markdown(f'<div class="param-info"><b>Vl —</b> {PARAM_DESC["Vl"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="param-info"><b>f —</b> {PARAM_DESC["f"]}</div>',   unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        Rs = c1.number_input("Rs — Resist. estator (Ω)", value=0.435, step=0.001, format="%.3f", key="p_Rs")
        Rr = c2.number_input("Rr — Resist. rotor (Ω)",   value=0.816, step=0.001, format="%.3f", key="p_Rr")
        if learning:
            st.markdown(f'<div class="param-info"><b>Rs —</b> {PARAM_DESC["Rs"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="param-info"><b>Rr —</b> {PARAM_DESC["Rr"]}</div>', unsafe_allow_html=True)

        Xm = st.number_input("Xm — Reatância de magnetização (Ω)", value=26.13, step=0.1, key="p_Xm")
        if learning:
            st.markdown(f'<div class="param-info"><b>Xm —</b> {PARAM_DESC["Xm"]}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        Xls = c1.number_input("Xls — Dispersão estator (Ω)", value=0.754, step=0.001, format="%.3f", key="p_Xls")
        Xlr = c2.number_input("Xlr — Dispersão rotor (Ω)",   value=0.754, step=0.001, format="%.3f", key="p_Xlr")
        if learning:
            st.markdown(f'<div class="param-info"><b>Xls —</b> {PARAM_DESC["Xls"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="param-info"><b>Xlr —</b> {PARAM_DESC["Xlr"]}</div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-header">⚙️ Parâmetros Mecânicos</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        p = c1.selectbox("p — Número de polos", [2, 4, 6, 8], index=1, key="p_p")
        J = c2.number_input("J — Inércia (kg·m²)", value=0.089, step=0.001, format="%.3f", key="p_J")
        if learning:
            st.markdown(f'<div class="param-info"><b>p —</b> {PARAM_DESC["p"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="param-info"><b>J —</b> {PARAM_DESC["J"]}</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        B          = c1.number_input("B — Atrito (Nm·s/rad)",        value=0.0,  step=0.001, format="%.3f", key="p_B")
        Tl_initial = c2.number_input("Tl_initial — Torque ini. (Nm)", value=0.0,  step=1.0,   key="p_Tli")
        Tl_final   = c3.number_input("Tl_final — Torque nom. (Nm)",   value=80.0, step=1.0,   key="p_Tlf")
        if learning:
            st.markdown(f'<div class="param-info"><b>B —</b> {PARAM_DESC["B"]}</div>',                       unsafe_allow_html=True)
            st.markdown(f'<div class="param-info"><b>Tl_initial —</b> {PARAM_DESC["Tl_initial"]}</div>',     unsafe_allow_html=True)
            st.markdown(f'<div class="param-info"><b>Tl_final —</b> {PARAM_DESC["Tl_final"]}</div>',         unsafe_allow_html=True)

        st.markdown("")
        # Info derivados
        n_sync = 120.0 * f / p
        we     = 2.0 * pi * f
        Xml    = 1.0 / (1.0 / Xm + 1.0 / Xls + 1.0 / Xlr)
        st.markdown(
            f'<div class="derived-box">📐 '
            f'nₛ = <b>{n_sync:.0f} rpm</b> &nbsp;·&nbsp; '
            f'ωₑ = <b>{we:.2f} rad/s</b> &nbsp;·&nbsp; '
            f'Xm‖Xls‖Xlr = <b>{Xml:.3f} Ω</b>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Botão confirmar
        if st.button("✅  Confirmar Parâmetros", type="primary", use_container_width=True, key="confirm_btn"):
            st.session_state.machine_params = {
                "Vl": Vl, "f": f, "Rs": Rs, "Rr": Rr, "Xm": Xm,
                "Xls": Xls, "Xlr": Xlr, "p": p, "J": J, "B": B,
                "Tl_initial": Tl_initial, "Tl_final": Tl_final,
            }
            st.session_state.params_confirmed = True
            st.session_state.sim_results      = None
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════════
    # Coluna direita — Painel do solver (destaque após confirmar)
    # ════════════════════════════════════════════════════════════════════════════
    with col_solver:
        if confirmed:
            st.markdown(
                '<div class="solver-banner-ok">✅ Parâmetros confirmados — configure o solver</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="solver-banner-wait">⏳ Aguardando confirmação dos parâmetros</div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown("### ⚙️ Parâmetros do Solver")
            tmax = st.slider("Tempo de simulação (s)",   0.5, 15.0, 2.0, 0.1,
                             key="sol_tmax", disabled=not confirmed)
            h    = st.select_slider("Passo de integração (s)", [0.0005, 0.001, 0.002],
                                    value=0.001, key="sol_h", disabled=not confirmed)
            st.markdown("---")
            experimento = st.selectbox(
                "Experimento",
                [
                    "1. Partida Direta (DOL)",
                    "2. Partida Y-Δ",
                    "3. Partida com Autotransformador",
                    "4. Partida Soft-Starter",
                    "5. Partida em Vazio e Carga Nominal",
                    "6. Variação da Carga (50% do Nominal)",
                    "7. Sobrecarga Temporária (120% do Nominal)",
                    "8. Impacto de Rr no Torque de Partida",
                    "9. Impacto do Momento de Inércia (J)",
                    "10. Variação da Frequência da Rede",
                ],
                key="sol_exp",
                disabled=not confirmed,
            )
            st.markdown("")
            run_btn = st.button(
                "▶  Iniciar Simulação",
                type="primary",
                use_container_width=True,
                disabled=not confirmed,
                key="run_btn",
            )

    # ════════════════════════════════════════════════════════════════════════════
    # Parâmetros específicos do experimento + execução
    # ════════════════════════════════════════════════════════════════════════════
    if confirmed:
        st.markdown("---")
        mp = st.session_state.machine_params

        with st.expander(f"⚙️  Opções do experimento selecionado", expanded=True):
            extra = _coletar_exp(experimento, mp, tmax)

        if run_btn:
            with st.spinner("Simulando…"):
                resultado = _executar_exp(experimento, mp, tmax, h, extra)
            st.session_state.sim_results    = resultado
            st.session_state.sim_experiment = experimento
            st.rerun()

        if (
            st.session_state.sim_results is not None
            and st.session_state.sim_experiment == experimento
        ):
            st.markdown("---")
            st.markdown(f"## Resultados — {experimento}")
            _mostrar_exp(experimento, st.session_state.sim_results, mp)


# ──────────────────────────────────────────────────────────────────────────────
# Coleta de parâmetros específicos por experimento
# ──────────────────────────────────────────────────────────────────────────────
def _coletar_exp(exp, mp, tmax):
    extra = {}
    Vl = mp["Vl"]

    if exp.startswith("1."):
        extra["t_carga"] = st.slider(
            "t_carga — Momento de aplicação do torque (s)", 0.0, float(tmax), 0.1, 0.05, key="e1_tc")

    elif exp.startswith("2."):
        c1, c2 = st.columns(2)
        extra["t_2"]     = c1.slider("t_2 — Comutação Y→Δ (s)",           0.1, 5.0, 0.5, 0.1,           key="e2_t2")
        extra["t_carga"] = c2.slider("t_carga — Aplicação do torque (s)",  0.0, float(tmax), 0.1, 0.05,  key="e2_tc")
        extra["Vl_Y"]    = Vl / sqrt(3.0)
        st.info(f"Tensão Y (partida): **{extra['Vl_Y']:.1f} V** → Δ (regime): **{Vl:.1f} V**")

    elif exp.startswith("3."):
        c1, c2 = st.columns(2)
        extra["t_2"]     = c1.slider("t_2 — Duração compensação (s)",      0.1, 5.0, 0.5, 0.1,           key="e3_t2")
        extra["t_carga"] = c1.slider("t_carga — Aplicação do torque (s)",  0.0, float(tmax), 0.1, 0.05,  key="e3_tc")
        reg              = c2.slider("Regulação de tensão (%)", 10, 90, 50, 5, key="e3_reg")
        extra["Vl_comp"] = Vl * reg / 100.0
        st.info(f"Tensão de partida: **{extra['Vl_comp']:.1f} V** ({reg}%) → **{Vl:.1f} V**")

    elif exp.startswith("4."):
        c1, c2 = st.columns(2)
        extra["t_carga"] = c1.slider("t_carga — Aplicação do torque (s)", 0.0, 5.0, 0.1, 0.05, key="e4_tc")
        extra["t_2"]     = c1.slider("t_2 — Início da rampa (s)",         0.0, 5.0, 0.9, 0.1,  key="e4_t2")
        tp_min           = float(extra["t_2"]) + 0.1
        extra["t_pico"]  = c2.slider("t_pico — Fim da rampa (s)",         tp_min, min(float(tmax), 12.0), max(5.0, tp_min), 0.1, key="e4_tp")
        reg              = c2.slider("Tensão inicial (%)", 10, 90, 50, 5, key="e4_reg")
        extra["Vl_ini"]  = Vl * reg / 100.0

    elif exp.startswith("5."):
        extra["t_carga"] = st.slider(
            "t_carga — Aplicação da carga nominal (s)", 0.0, float(tmax), 1.0, 0.1, key="e5_tc")

    elif exp.startswith("6."):
        extra["t_carga"] = st.slider(
            "t_carga — Aplicação da carga 50% (s)", 0.0, float(tmax), 1.0, 0.1, key="e6_tc")
        st.info(f"Carga aplicada: **{mp['Tl_final'] * 0.5:.0f} Nm** (50% de {mp['Tl_final']:.0f} Nm)")

    elif exp.startswith("7."):
        extra["t_carga"] = st.slider(
            "t_carga — Aplicação da sobrecarga 120% (s)", 0.0, float(tmax), 1.0, 0.1, key="e7_tc")
        st.warning(f"Sobrecarga: **{mp['Tl_final'] * 1.2:.0f} Nm** (120% de {mp['Tl_final']:.0f} Nm)")

    elif exp.startswith("8."):
        c1, c2 = st.columns(2)
        extra["Rr_input"] = c1.text_input("Valores de Rr (Ω) separados por vírgula", "0.30, 1.00, 1.20", key="e8_rr")
        extra["T_load"]   = c2.number_input("Torque de carga (Nm)", value=float(mp["Tl_final"]), step=1.0, key="e8_Tl")
        extra["conn"]     = c2.selectbox("Conexão", ["Y", "D"], key="e8_conn")

    elif exp.startswith("9."):
        c1, c2 = st.columns(2)
        extra["J_sim"]   = c1.slider("J (kg·m²)", 0.01, 0.40, float(mp["J"]), 0.005, key="e9_J")
        extra["Tl_fin9"] = c2.slider("Torque final (Nm)", 0.0, 150.0, float(mp["Tl_final"]), 1.0, key="e9_Tf")
        extra["t2_ramp"] = c2.slider("Duração da rampa de torque (s)", 0.1, 5.0, 2.0, 0.1, key="e9_ramp")
        ns_val = 120.0 * mp["f"] / mp["p"]
        st.info(f"Velocidade síncrona: **{ns_val:.0f} rpm** — mede o tempo até 95% ({ns_val * 0.95:.0f} rpm).")

    elif exp.startswith("10."):
        st.markdown("**Parâmetros do circuito equivalente (à frequência nominal):**")
        c1, c2 = st.columns(2)
        extra["R1"]     = c1.number_input("R1 — Resist. estator (Ω)", value=0.641, step=0.001, format="%.3f", key="e10_R1")
        extra["R2"]     = c1.number_input("R2 — Resist. rotor (Ω)",   value=0.332, step=0.001, format="%.3f", key="e10_R2")
        extra["X1"]     = c1.number_input("X1 — Reat. estator (Ω)",   value=1.106, step=0.001, format="%.3f", key="e10_X1")
        extra["X2"]     = c2.number_input("X2 — Reat. rotor (Ω)",     value=0.464, step=0.001, format="%.3f", key="e10_X2")
        extra["Xm_q"]   = c2.number_input("Xm — Reat. magn. (Ω)",     value=26.3,  step=0.1,   key="e10_Xm")
        extra["f_nom"]  = c2.number_input("Frequência nominal (Hz)",   value=50.0,  step=1.0,   key="e10_fn")
        extra["V_nom"]  = c1.number_input("Tensão nominal L-L (V)",    value=220.0, step=10.0,  key="e10_Vn")
        extra["T_mec"]  = c1.number_input("Torque mecânico (Nm)",      value=float(mp["Tl_final"]), step=1.0, key="e10_Tm")
        extra["freqs_s"] = st.text_input("Frequências a simular (Hz)", "50, 40", key="e10_fr")

    return extra


# ──────────────────────────────────────────────────────────────────────────────
# Execução dos experimentos
# ──────────────────────────────────────────────────────────────────────────────
def _executar_exp(exp, mp, tmax, h, extra):
    Vl = mp["Vl"]; Tl_ini = mp["Tl_initial"]; Tl_fin = mp["Tl_final"]

    if exp.startswith("1."):
        tc = extra["t_carga"]
        return _simular(lambda t: Vl, lambda t: Tl_fin if t >= tc else Tl_ini, tmax, h, mp)

    elif exp.startswith("2."):
        t2 = extra["t_2"]; tc = extra["t_carga"]; Vly = extra["Vl_Y"]
        return _simular(lambda t: Vl if t >= t2 else Vly,
                        lambda t: Tl_fin if t >= tc else Tl_ini, tmax, h, mp)

    elif exp.startswith("3."):
        t2 = extra["t_2"]; tc = extra["t_carga"]; Vc = extra["Vl_comp"]
        return _simular(lambda t: Vl if t >= t2 else Vc,
                        lambda t: Tl_fin if t >= tc else Tl_ini, tmax, h, mp)

    elif exp.startswith("4."):
        tc = extra["t_carga"]; t2 = extra["t_2"]; tp = extra["t_pico"]; Vi = extra["Vl_ini"]
        def _V(t):
            if t < t2:  return Vi
            if t < tp:  return Vi + (Vl - Vi) * (t - t2) / (tp - t2)
            return Vl
        return _simular(_V, lambda t: Tl_fin if t >= tc else Tl_ini, tmax, h, mp)

    elif exp.startswith("5."):
        tc = extra["t_carga"]
        return _simular(lambda t: Vl, lambda t: Tl_fin if t >= tc else 0.0, tmax, h, mp)

    elif exp.startswith("6."):
        tc = extra["t_carga"]
        return _simular(lambda t: Vl, lambda t: Tl_fin * 0.5 if t >= tc else 0.0, tmax, h, mp)

    elif exp.startswith("7."):
        tc = extra["t_carga"]
        return _simular(lambda t: Vl, lambda t: Tl_fin * 1.2 if t >= tc else 0.0, tmax, h, mp)

    elif exp.startswith("8."):
        try:
            Rr_list = [float(v.strip()) for v in extra["Rr_input"].split(",") if v.strip()]
        except ValueError:
            st.error("Valores de Rr inválidos."); st.stop()
        return {rv: _torque_curve_static(mp, rv, extra["T_load"], extra["conn"]) for rv in Rr_list}

    elif exp.startswith("9."):
        J_sim = extra["J_sim"]; Tf9 = extra["Tl_fin9"]; t2r = extra["t2_ramp"]
        _Tl9 = lambda t: min(Tf9, Tf9 * t / t2r) if t2r > 0 else Tf9
        return _simular(lambda t: Vl, _Tl9, tmax, h, mp, J_override=J_sim)

    elif exp.startswith("10."):
        try:
            freq_list = [float(v.strip()) for v in extra["freqs_s"].split(",") if v.strip()]
        except ValueError:
            st.error("Frequências inválidas."); st.stop()
        return {"freq_list": freq_list, "extra": extra, "p": mp["p"]}

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Exibição de resultados
# ──────────────────────────────────────────────────────────────────────────────
def _mostrar_exp(exp, resultado, mp):
    if exp.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.")):
        _mostrar_dinamico(resultado, exp, mp)

    elif exp.startswith("8."):
        rows = [
            {
                "Rr (Ω)":         rv,
                "T_partida (Nm)": round(r["Te_start"], 2),
                "s_Tmax":         round(r["s_Tmax"], 4),
                "Tmax (Nm)":      round(r["Te_max"], 2),
                "n_Tmax (rpm)":   round(r["n_Tmax_rpm"], 1),
                "s_oper":         round(r["s_oper"], 4),
                "n_oper (rpm)":   round(r["n_oper_rpm"], 1),
            }
            for rv, r in resultado.items()
        ]
        st.dataframe(pd.DataFrame(rows))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for rv, r in resultado.items():
            axes[0].plot(r["n_rpm"], r["Te"], label=f"Rr={rv:.2f} Ω")
        axes[0].set_xlabel("Velocidade (rpm)"); axes[0].set_ylabel("Torque (Nm)")
        axes[0].set_title("Torque × Velocidade"); axes[0].legend(); axes[0].grid(True)
        Rr_s = sorted(resultado.keys())
        axes[1].plot(Rr_s, [resultado[rv]["Te_start"] for rv in Rr_s], "o-")
        axes[1].set_xlabel("Rr (Ω)"); axes[1].set_ylabel("T_partida (Nm)")
        axes[1].set_title("Torque de Partida vs Rr"); axes[1].grid(True)
        axes[2].plot(Rr_s, [resultado[rv]["s_oper"] for rv in Rr_s], "o-")
        axes[2].set_xlabel("Rr (Ω)"); axes[2].set_ylabel("Escorregamento em regime")
        axes[2].set_title("Escorregamento vs Rr"); axes[2].grid(True)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    elif exp.startswith("9."):
        n_sync = 120.0 * mp["f"] / mp["p"]
        n_target = 0.95 * n_sync
        t_a = resultado["t"]; n_a = resultado["n"]; Te_a = resultado["Te"]
        idxs = np.where(n_a >= n_target)[0]
        t95  = float(t_a[idxs[0]]) if len(idxs) > 0 else float("nan")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(t_a, n_a, color="royalblue")
        axes[0].axhline(n_target, ls="--", color="green", label=f"95% — {n_target:.0f} rpm")
        if not np.isnan(t95):
            axes[0].axvline(t95, ls=":", color="red", label=f"t₉₅ = {t95:.3f} s")
        axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("Velocidade (RPM)")
        axes[0].set_title("Velocidade"); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(t_a, Te_a, color="firebrick")
        axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("Torque (Nm)")
        axes[1].set_title("Torque Eletromagnético"); axes[1].grid(True)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.metric("Tempo para 95% da vel. síncrona",
                  f"{t95:.3f} s" if not np.isnan(t95) else "Não atingido no intervalo")

    elif exp.startswith("10."):
        data = resultado
        freq_list = data["freq_list"]; ex = data["extra"]; p_pol = data["p"]
        R1 = ex["R1"]; R2 = ex["R2"]; X1 = ex["X1"]; X2 = ex["X2"]
        Xm_v = ex["Xm_q"]; f_nom = ex["f_nom"]; V_nom = ex["V_nom"]; T_mec = ex["T_mec"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        rows = []
        for freq in freq_list:
            V      = V_nom * (freq / f_nom)
            ns_rpm = 120.0 * freq / p_pol
            s_v    = np.linspace(0.001, 0.1, 300)
            T_v    = np.array([_Te_freq(V, freq, s, R1, R2, X1, X2, Xm_v, p_pol) for s in s_v])
            idx_r  = int(np.argmin(np.abs(T_v - T_mec)))
            s_r    = float(s_v[idx_r]); n_r = ns_rpm * (1.0 - s_r)
            Z1v = R1 + 1j * X1; Zmv = 1j * Xm_v; Z2v = R2 / s_r + 1j * X2
            I_r = V / (Z1v + (Zmv * Z2v) / (Zmv + Z2v))
            P_m = (2.0 * pi * n_r / 60.0) * T_mec
            P_e = 3.0 * abs(I_r) ** 2 * R1 + P_m / (1.0 - s_r)
            eff = P_m / P_e * 100.0 if P_e > 0 else 0.0
            rows.append({
                "f (Hz)": freq, "nₛ (rpm)": round(ns_rpm, 1),
                "n_regime (rpm)": round(n_r, 1), "s (%)": round(s_r * 100, 3),
                "I_regime (A)": round(abs(I_r), 3),
                "P_mec (W)": round(P_m, 1), "η (%)": round(eff, 2),
            })
            axes[0].plot(ns_rpm * (1.0 - s_v), T_v, label=f"{freq} Hz")
            s_full = np.linspace(0.001, 1.0, 400)
            T_full = np.array([_Te_freq(V, freq, s, R1, R2, X1, X2, Xm_v, p_pol) for s in s_full])
            idx2   = int(np.argmin(np.abs(T_full - T_mec)))
            axes[1].plot(T_full, s_full * 100.0, label=f"{freq} Hz")
            axes[1].plot(T_full[idx2], s_full[idx2] * 100.0, "o")
        axes[0].axhline(T_mec, ls="--", color="k", label="T_mec")
        axes[0].set_xlabel("Velocidade (rpm)"); axes[0].set_ylabel("Torque (Nm)")
        axes[0].set_title("Torque × Velocidade"); axes[0].legend(); axes[0].grid(True)
        axes[1].set_xlabel("Torque (Nm)"); axes[1].set_ylabel("Escorregamento (%)")
        axes[1].set_title("Escorregamento × Torque"); axes[1].legend(); axes[1].grid(True)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.dataframe(pd.DataFrame(rows))


# ══════════════════════════════════════════════════════════════════════════════
# Roteador principal
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    screen_home()
elif st.session_state.page == "params":
    screen_params()
