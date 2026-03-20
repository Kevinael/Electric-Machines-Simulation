# -*- coding: utf-8 -*-
"""Simulação de Máquina de Indução Trifásica — Streamlit App
Experimentos 1-10 consolidados sem redundâncias.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from math import pi, sqrt

st.set_page_config(page_title="MIT — Simulação", layout="wide")
st.title("Simulação de Máquina de Indução Trifásica")

# ─── Parâmetros comuns da máquina (sidebar) ───────────────────────────────────
st.sidebar.header("Parâmetros da Máquina")
Vl  = st.sidebar.number_input("Vl — Tensão RMS de linha (V)",   value=220.0,  step=10.0)
f   = st.sidebar.number_input("f — Frequência (Hz)",             value=60.0,   step=1.0)
Rs  = st.sidebar.number_input("Rs — Resist. estator (Ω)",        value=0.435,  step=0.001, format="%.3f")
Rr  = st.sidebar.number_input("Rr — Resist. rotor (Ω)",          value=0.816,  step=0.001, format="%.3f")
Xm  = st.sidebar.number_input("Xm — Reat. magnetizante (Ω)",     value=26.13,  step=0.1)
Xls = st.sidebar.number_input("Xls — Reat. dispersão estator (Ω)", value=0.754, step=0.001, format="%.3f")
Xlr = st.sidebar.number_input("Xlr — Reat. dispersão rotor (Ω)",   value=0.754, step=0.001, format="%.3f")
p   = st.sidebar.selectbox("p — Número de polos", [2, 4, 6, 8], index=1)
J   = st.sidebar.number_input("J — Inércia do rotor (kg·m²)",    value=0.089,  step=0.001, format="%.3f")
B   = st.sidebar.number_input("B — Atrito viscoso (Nm·s/rad)",   value=0.0,    step=0.001, format="%.3f")
Tl_initial = st.sidebar.number_input("Tl_initial — Torque inicial (Nm)", value=0.0,  step=1.0)
Tl_final   = st.sidebar.number_input("Tl_final — Torque final (Nm)",     value=80.0, step=1.0)

# Parâmetros derivados
wb        = 2 * np.pi * f
Xml       = 1.0 / (1.0/Xm + 1.0/Xls + 1.0/Xlr)
n_sync    = 120.0 * f / p   # rpm

st.sidebar.markdown("---")
experimento = st.sidebar.selectbox(
    "Experimento",
    [
        "1. Partida Direta (DOL)",
        "2. Partida Y-Δ (Estrela-Triângulo)",
        "3. Partida com Autotransformador",
        "4. Partida Soft-Starter",
        "5. Partida em Vazio e Carga Nominal",
        "6. Variação da Carga (50% do Nominal)",
        "7. Sobrecarga Temporária (120% do Nominal)",
        "8. Impacto de Rr no Torque de Partida",
        "9. Impacto do Momento de Inércia (J)",
        "10. Variação da Frequência da Rede",
    ],
)

# ─── Função única de ODE ──────────────────────────────────────────────────────
def motor_odes(states, t, Vqs, Vds, Tl, w_ref, Rs, Rr, Xls, Xlr, Xml, wb, p, J, B):
    PSIqs, PSIds, PSIqr, PSIdr, wr = states
    PSImq = Xml * (PSIqs / Xls + PSIqr / Xlr)
    PSImd = Xml * (PSIds / Xls + PSIdr / Xlr)
    iqs_v = (1.0 / Xls) * (PSIqs - PSImq)
    ids_v = (1.0 / Xls) * (PSIds - PSImd)
    dPSIqs = wb * (Vqs  - (w_ref / wb) * PSIds        + (Rs / Xls) * (PSImq - PSIqs))
    dPSIds = wb * (Vds  + (w_ref / wb) * PSIqs        + (Rs / Xls) * (PSImd - PSIds))
    dPSIqr = wb * (0.0  - ((w_ref - wr) / wb) * PSIdr + (Rr / Xlr) * (PSImq - PSIqr))
    dPSIdr = wb * (0.0  + ((w_ref - wr) / wb) * PSIqr + (Rr / Xlr) * (PSImd - PSIdr))
    Tem    = (3.0 / 2.0) * (p / 2.0) * (1.0 / wb) * (PSIds * iqs_v - PSIqs * ids_v)
    dwr    = (p / (2.0 * J)) * (Tem - Tl) - (B / J) * wr
    return [dPSIqs, dPSIds, dPSIqr, dPSIdr, dwr]


# ─── Função única de simulação ────────────────────────────────────────────────
def simular(V_func, Tl_func, tmax, h, J_sim=None):
    """
    V_func(t)  → tensão de linha RMS em t
    Tl_func(t) → torque de carga em t
    J_sim      → sobrescreve J do sidebar (para exp. 9)
    """
    J_use  = J_sim if J_sim is not None else J
    Xml_use = 1.0 / (1.0/Xm + 1.0/Xls + 1.0/Xlr)
    wb_use  = 2.0 * np.pi * f
    we      = wb_use

    t_vals  = np.arange(0.0, tmax, h)
    states  = [0.0] * 5
    last_wr = 0.0

    wr_r, Te_r, n_r = [], [], []
    ias_r, ibs_r, ics_r = [], [], []
    iar_r, ibr_r, icr_r = [], [], []
    Va_r,  Vb_r,  Vc_r  = [], [], []

    for t_val in t_vals:
        V_apli   = V_func(t_val)
        curr_Tl  = Tl_func(t_val)
        tetae    = we * t_val
        w_ref    = we   # referência síncrona

        Va = sqrt(2) * V_apli * np.sin(tetae)
        Vb = sqrt(2) * V_apli * np.sin(tetae - 2.0 * pi / 3.0)
        Vc = sqrt(2) * V_apli * np.sin(tetae + 2.0 * pi / 3.0)

        Vaf = sqrt(3.0 / 2.0) * (Va - 0.5 * Vb - 0.5 * Vc)
        Vbt = sqrt(3.0 / 2.0) * (0.0 * Va + (sqrt(3.0) / 2.0) * Vb - (sqrt(3.0) / 2.0) * Vc)
        Vds_v = np.cos(tetae) * Vaf + np.sin(tetae) * Vbt
        Vqs_v = -np.sin(tetae) * Vaf + np.cos(tetae) * Vbt

        sol    = odeint(motor_odes, states, [t_val, t_val + h],
                        args=(Vqs_v, Vds_v, curr_Tl, w_ref,
                              Rs, Rr, Xls, Xlr, Xml_use, wb_use, p, J_use, B))
        states  = sol[1]
        PSIqs, PSIds, PSIqr, PSIdr, wr = states
        last_wr = wr

        PSImd = Xml_use * (PSIds / Xls + PSIdr / Xlr)
        PSImq = Xml_use * (PSIqs / Xls + PSIqr / Xlr)
        ids   = (1.0 / Xls) * (PSIds - PSImd)
        iqs   = (1.0 / Xls) * (PSIqs - PSImq)
        idr   = (1.0 / Xlr) * (PSIdr - PSImd)
        iqr   = (1.0 / Xlr) * (PSIqr - PSImq)
        Te    = (3.0 / 2.0) * (p / 2.0) * (1.0 / wb_use) * (PSIds * iqs - PSIqs * ids)
        n_rpm = (120.0 / p) * (wr / (2.0 * pi))
        tetar = wr * t_val

        iafs = np.cos(tetae) * ids - np.sin(tetae) * iqs
        ibts = np.sin(tetae) * ids + np.cos(tetae) * iqs
        iafr = np.cos(tetar) * idr - np.sin(tetar) * iqr
        ibtr = np.sin(tetar) * idr + np.cos(tetar) * iqr
        sq32 = sqrt(3.0 / 2.0)

        ias_r.append(sq32 * iafs)
        ibs_r.append(sq32 * ((-0.5) * iafs + (sqrt(3.0) / 2.0) * ibts))
        ics_r.append(sq32 * ((-0.5) * iafs - (sqrt(3.0) / 2.0) * ibts))
        iar_r.append(sq32 * iafr)
        ibr_r.append(sq32 * ((-0.5) * iafr + (sqrt(3.0) / 2.0) * ibtr))
        icr_r.append(sq32 * ((-0.5) * iafr - (sqrt(3.0) / 2.0) * ibtr))

        wr_r.append(wr); Te_r.append(Te); n_r.append(n_rpm)
        Va_r.append(Va); Vb_r.append(Vb); Vc_r.append(Vc)

    return {
        "t":   t_vals,
        "wr":  np.array(wr_r),  "Te":  np.array(Te_r),  "n":   np.array(n_r),
        "ias": np.array(ias_r), "ibs": np.array(ibs_r), "ics": np.array(ics_r),
        "iar": np.array(iar_r), "ibr": np.array(ibr_r), "icr": np.array(icr_r),
        "Va":  np.array(Va_r),  "Vb":  np.array(Vb_r),  "Vc":  np.array(Vc_r),
    }


# ─── Função de plotagem dos resultados dinâmicos ──────────────────────────────
def mostrar_resultados(res, titulo=""):
    t = res["t"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    if titulo:
        fig.suptitle(titulo, fontsize=13, fontweight="bold")

    axes[0, 0].plot(t, res["Te"])
    axes[0, 0].set_title("Torque Eletromagnético (Nm)"); axes[0, 0].set_xlabel("t (s)"); axes[0, 0].grid(True)

    axes[0, 1].plot(t, res["n"])
    axes[0, 1].axhline(n_sync, ls="--", color="gray", label=f"n_sinc={n_sync:.0f} rpm")
    axes[0, 1].set_title("Velocidade (RPM)"); axes[0, 1].set_xlabel("t (s)"); axes[0, 1].legend(); axes[0, 1].grid(True)

    for lbl, arr in [("$i_{as}$", res["ias"]), ("$i_{bs}$", res["ibs"]), ("$i_{cs}$", res["ics"])]:
        axes[1, 0].plot(t, arr, label=lbl)
    axes[1, 0].set_title("Correntes do Estator (A)"); axes[1, 0].set_xlabel("t (s)"); axes[1, 0].legend(); axes[1, 0].grid(True)

    for lbl, arr in [("$i_{ar}$", res["iar"]), ("$i_{br}$", res["ibr"]), ("$i_{cr}$", res["icr"])]:
        axes[1, 1].plot(t, arr, label=lbl)
    axes[1, 1].set_title("Correntes do Rotor (A)"); axes[1, 1].set_xlabel("t (s)"); axes[1, 1].legend(); axes[1, 1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    wb_loc = 2.0 * pi * f
    s_ss   = (wb_loc - float(res["wr"][-1])) / wb_loc
    col1, col2, col3 = st.columns(3)
    col1.metric("Escorregamento em regime", f"{s_ss:.4f}")
    col2.metric("Velocidade final (RPM)", f"{float(res['n'][-1]):.1f}")
    col3.metric("Torque final (Nm)", f"{float(res['Te'][-1]):.2f}")

    st.subheader("Valores máximos (|pico|)")
    rows = {
        "Corrente": ["I_as (A)", "I_ar (A)", "Te (Nm)"],
        "Máximo":   [
            round(float(np.max(np.abs(res["ias"]))), 3),
            round(float(np.max(np.abs(res["iar"]))), 3),
            round(float(np.max(res["Te"])), 3),
        ],
    }
    st.table(pd.DataFrame(rows).set_index("Corrente"))
    return s_ss


# ─── Parâmetros de simulação reutilizáveis ────────────────────────────────────
def param_sim(key_prefix, tmax_def=2.0, tmax_max=10.0, h_def=0.001):
    col1, col2 = st.columns(2)
    tmax = col1.slider("Tempo de simulação (s)", 0.5, tmax_max, tmax_def, 0.1, key=f"{key_prefix}_tmax")
    h    = col1.select_slider("Passo de integração (s)", [0.0005, 0.001, 0.002],
                               value=h_def, key=f"{key_prefix}_h")
    return col2, tmax, h


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTOS DE PARTIDA
# ══════════════════════════════════════════════════════════════════════════════

if experimento.startswith("1."):
    st.header("1. Partida Direta (DOL — Direct On Line)")
    st.markdown("Alimentação nominal aplicada diretamente desde t=0. Observar picos de corrente e torque.")
    col2, tmax, h = param_sim("e1")
    t_carga = col2.slider("t_carga — Aplicação do torque (s)", 0.0, tmax, 0.1, 0.05, key="e1_tc")

    if st.button("Simular", key="btn1"):
        with st.spinner("Simulando…"):
            res = simular(
                V_func=lambda t: Vl,
                Tl_func=lambda t: Tl_final if t >= t_carga else Tl_initial,
                tmax=tmax, h=h,
            )
        mostrar_resultados(res, "Partida Direta (DOL)")

elif experimento.startswith("2."):
    st.header("2. Partida Y-Δ (Estrela-Triângulo)")
    st.markdown(
        "Tensão reduzida a $V_l/\\sqrt{3}$ durante a partida, depois comuta para tensão nominal."
    )
    col2, tmax, h = param_sim("e2")
    t_2     = col2.slider("t_2 — Comutação Y→Δ (s)",          0.1, 5.0, 0.5, 0.1, key="e2_t2")
    t_carga = col2.slider("t_carga — Aplicação do torque (s)", 0.0, tmax, 0.1, 0.05, key="e2_tc")
    Vl_Y    = Vl / sqrt(3.0)
    st.info(f"Tensão Y (partida): **{Vl_Y:.1f} V** → Δ (regime): **{Vl:.1f} V**")

    if st.button("Simular", key="btn2"):
        with st.spinner("Simulando…"):
            res = simular(
                V_func=lambda t: Vl if t >= t_2 else Vl_Y,
                Tl_func=lambda t: Tl_final if t >= t_carga else Tl_initial,
                tmax=tmax, h=h,
            )
        mostrar_resultados(res, "Partida Y-Δ")

elif experimento.startswith("3."):
    st.header("3. Partida com Autotransformador (Compensadora)")
    st.markdown("Tensão reduzida por um percentual configúrável, depois restaurada ao valor nominal.")
    col2, tmax, h = param_sim("e3")
    t_2      = col2.slider("t_2 — Duração da compensação (s)",   0.1, 5.0, 0.5, 0.1, key="e3_t2")
    t_carga  = col2.slider("t_carga — Aplicação do torque (s)", 0.0, tmax, 0.1, 0.05, key="e3_tc")
    reg      = col2.slider("Regulação de tensão (%)", 10, 90, 50, 5, key="e3_reg") / 100.0
    Vl_comp  = Vl * reg
    st.info(f"Tensão de partida: **{Vl_comp:.1f} V** ({reg*100:.0f}%) → **{Vl:.1f} V**")

    if st.button("Simular", key="btn3"):
        with st.spinner("Simulando…"):
            res = simular(
                V_func=lambda t: Vl if t >= t_2 else Vl_comp,
                Tl_func=lambda t: Tl_final if t >= t_carga else Tl_initial,
                tmax=tmax, h=h,
            )
        mostrar_resultados(res, "Partida com Autotransformador")

elif experimento.startswith("4."):
    st.header("4. Partida Soft-Starter")
    st.markdown(
        "A tensão cresce linearmente de $V_{ini}$ até $V_{nominal}$ "
        "no intervalo $[t_2,\\; t_{pico}]$."
    )
    col2, tmax, h = param_sim("e4", tmax_def=10.0, tmax_max=15.0)
    t_carga = col2.slider("t_carga — Aplicação do torque (s)", 0.0, 5.0, 0.1, 0.05, key="e4_tc")
    t_2     = col2.slider("t_2 — Início da rampa (s)",         0.0, 5.0, 0.9, 0.1,  key="e4_t2")
    t_pico  = col2.slider("t_pico — Fim da rampa (s)",         t_2 + 0.1, min(tmax, 10.0), 5.0, 0.1, key="e4_tp")
    reg     = col2.slider("Tensão inicial (%)", 10, 90, 50, 5, key="e4_reg") / 100.0
    Vl_ini  = Vl * reg

    def _V_soft(t):
        if t < t_2:
            return Vl_ini
        elif t < t_pico:
            return Vl_ini + (Vl - Vl_ini) * (t - t_2) / (t_pico - t_2)
        return Vl

    if st.button("Simular", key="btn4"):
        with st.spinner("Simulando…"):
            res = simular(
                V_func=_V_soft,
                Tl_func=lambda t: Tl_final if t >= t_carga else Tl_initial,
                tmax=tmax, h=h,
            )
        mostrar_resultados(res, "Partida Soft-Starter")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTOS DE CARGA
# ══════════════════════════════════════════════════════════════════════════════

elif experimento.startswith("5."):
    st.header("5. Partida em Vazio e Aplicação de Carga Nominal")
    st.markdown(
        "Motor parte sem carga até $t_{carga}$, onde o torque nominal é aplicado abruptamente."
    )
    col2, tmax, h = param_sim("e5")
    t_carga = col2.slider("t_carga — Aplicação da carga (s)", 0.0, tmax, 1.0, 0.1, key="e5_tc")

    if st.button("Simular", key="btn5"):
        with st.spinner("Simulando…"):
            res = simular(
                V_func=lambda t: Vl,
                Tl_func=lambda t: Tl_final if t >= t_carga else 0.0,
                tmax=tmax, h=h,
            )
        mostrar_resultados(res, "Partida em Vazio + Carga Nominal")

elif experimento.startswith("6."):
    st.header("6. Variação da Carga — 50% do Nominal")
    st.markdown(
        f"Aplica **{Tl_final * 0.5:.0f} Nm** (50% de {Tl_final:.0f} Nm) em $t_{{carga}}$."
    )
    col2, tmax, h = param_sim("e6")
    t_carga = col2.slider("t_carga — Aplicação da carga (s)", 0.0, tmax, 1.0, 0.1, key="e6_tc")

    if st.button("Simular", key="btn6"):
        with st.spinner("Simulando…"):
            res = simular(
                V_func=lambda t: Vl,
                Tl_func=lambda t: Tl_final * 0.5 if t >= t_carga else 0.0,
                tmax=tmax, h=h,
            )
        mostrar_resultados(res, f"Carga 50% — {Tl_final * 0.5:.0f} Nm")

elif experimento.startswith("7."):
    st.header("7. Sobrecarga Temporária — 120% do Nominal")
    st.markdown(
        f"Aplica **{Tl_final * 1.2:.0f} Nm** (120% de {Tl_final:.0f} Nm) em $t_{{carga}}$."
    )
    col2, tmax, h = param_sim("e7")
    t_carga = col2.slider("t_carga — Aplicação da sobrecarga (s)", 0.0, tmax, 1.0, 0.1, key="e7_tc")

    if st.button("Simular", key="btn7"):
        with st.spinner("Simulando…"):
            res = simular(
                V_func=lambda t: Vl,
                Tl_func=lambda t: Tl_final * 1.2 if t >= t_carga else 0.0,
                tmax=tmax, h=h,
            )
        mostrar_resultados(res, f"Sobrecarga 120% — {Tl_final * 1.2:.0f} Nm")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTOS DE PARÂMETROS DA MÁQUINA
# ══════════════════════════════════════════════════════════════════════════════

elif experimento.startswith("8."):
    st.header("8. Impacto de Rr no Torque de Partida e Escorregamento")
    st.markdown(
        "Análise em regime estático (circuito equivalente + Thévenin). "
        "Compara curvas torque-velocidade para diferentes valores de $R_r$."
    )

    col1, col2 = st.columns(2)
    Rr_input = col1.text_input("Valores de Rr (Ω), separados por vírgula", "0.30, 1.00, 1.20")
    T_load   = col2.number_input("Torque de carga (Nm)", value=80.0, step=1.0)
    conn     = col2.selectbox("Conexão", ["Y", "D"])

    def _torque_curve(Rr_val):
        w_sync = 4.0 * pi * f / p
        w_e    = 2.0 * pi * f
        Vph    = Vl / sqrt(3.0) if conn == "Y" else Vl
        # Indutâncias a partir das reatâncias (à frequência nominal)
        Lm_h   = Xm  / w_e
        Lls_h  = max(Xls / w_e - Lm_h, 0.0)
        Llr_h  = max(Xlr / w_e - Lm_h, 0.0)
        X1     = w_e * Lls_h
        X2     = w_e * Llr_h
        Xm_v   = w_e * Lm_h
        Z1     = complex(Rs, X1)
        Zm     = complex(0.0, Xm_v)
        Vth    = Vph * (Zm / (Z1 + Zm))
        Zth    = (Zm * Z1) / (Z1 + Zm)
        Rth, Xth = Zth.real, Zth.imag
        s      = np.linspace(1e-4, 1.0, 4000)[::-1]
        R2     = Rr_val
        Te     = (3.0 * abs(Vth) ** 2 * (R2 / s)) / (w_sync * ((Rth + R2 / s) ** 2 + (Xth + X2) ** 2))
        ns_rpm = 120.0 * f / p
        n_rpm  = (1.0 - s) * ns_rpm
        Te_s   = (3.0 * abs(Vth) ** 2 * R2) / (w_sync * ((Rth + R2) ** 2 + (Xth + X2) ** 2))
        s_Tm   = float(np.clip(R2 / sqrt(Rth ** 2 + (Xth + X2) ** 2), 1e-4, 1.0))
        Te_m   = (3.0 * abs(Vth) ** 2 * (R2 / s_Tm)) / (w_sync * ((Rth + R2 / s_Tm) ** 2 + (Xth + X2) ** 2))
        idx    = int(np.argmin(np.abs(Te - T_load)))
        return {
            "s": s, "Te": Te, "n_rpm": n_rpm, "ns_rpm": ns_rpm,
            "Te_start": float(Te_s), "s_Tmax": s_Tm, "Te_max": float(Te_m),
            "n_Tmax_rpm": float((1.0 - s_Tm) * ns_rpm),
            "s_oper": float(s[idx]), "n_oper_rpm": float(n_rpm[idx]),
        }

    if st.button("Calcular", key="btn8"):
        try:
            Rr_list = [float(v.strip()) for v in Rr_input.split(",") if v.strip()]
            if not Rr_list:
                raise ValueError
        except ValueError:
            st.error("Insira valores válidos de Rr separados por vírgula.")
            st.stop()

        results = {rv: _torque_curve(rv) for rv in Rr_list}

        rows = [
            {
                "Rr (Ω)": rv,
                "T_partida (Nm)": round(r["Te_start"], 2),
                "s_Tmax": round(r["s_Tmax"], 4),
                "Tmax (Nm)": round(r["Te_max"], 2),
                "n_Tmax (rpm)": round(r["n_Tmax_rpm"], 1),
                "s_oper": round(r["s_oper"], 4),
                "n_oper (rpm)": round(r["n_oper_rpm"], 1),
            }
            for rv, r in results.items()
        ]
        st.dataframe(pd.DataFrame(rows))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        Rr_sorted = sorted(results.keys())

        for rv, r in results.items():
            axes[0].plot(r["n_rpm"], r["Te"], label=f"Rr={rv:.2f} Ω")
        axes[0].axhline(T_load, ls="--", color="k", label="T_carga")
        axes[0].set_xlabel("Velocidade (rpm)"); axes[0].set_ylabel("Torque (Nm)")
        axes[0].set_title("Torque × Velocidade"); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(Rr_sorted, [results[rv]["Te_start"] for rv in Rr_sorted], "o-")
        axes[1].set_xlabel("Rr (Ω)"); axes[1].set_ylabel("T_partida (Nm)")
        axes[1].set_title("Torque de Partida vs Rr"); axes[1].grid(True)

        axes[2].plot(Rr_sorted, [results[rv]["s_oper"] for rv in Rr_sorted], "o-")
        axes[2].set_xlabel("Rr (Ω)"); axes[2].set_ylabel("Escorregamento em regime")
        axes[2].set_title("Escorregamento vs Rr"); axes[2].grid(True)

        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

elif experimento.startswith("9."):
    st.header("9. Impacto do Momento de Inércia (J) no Tempo de Aceleração")
    st.markdown("Mede o tempo para atingir 95% da velocidade síncrona para diferentes valores de J.")

    col1, col2 = st.columns(2)
    J_sim   = col1.slider("J (kg·m²)", 0.01, 0.40, float(J), 0.005, key="e9_J")
    tmax    = col1.slider("Tempo de simulação (s)", 1.0, 10.0, 5.0, 0.5, key="e9_tmax")
    h       = col1.select_slider("Passo (s)", [0.0005, 0.001], value=0.0005, key="e9_h")
    Tl_fin9 = col2.slider("Torque final (Nm)", 0.0, 150.0, float(Tl_final), 1.0, key="e9_Tf")
    t2_ramp = col2.slider("Duração da rampa de torque (s)", 0.1, 5.0, 2.0, 0.1, key="e9_t2")

    def _Tl9(t):
        return min(Tl_fin9, Tl_fin9 * t / t2_ramp) if t2_ramp > 0 else Tl_fin9

    if st.button("Simular", key="btn9"):
        with st.spinner("Simulando…"):
            res9 = simular(
                V_func=lambda t: Vl,
                Tl_func=_Tl9,
                tmax=tmax, h=h,
                J_sim=J_sim,
            )

        n_target = 0.95 * n_sync
        idxs     = np.where(res9["n"] >= n_target)[0]
        t95      = float(res9["t"][idxs[0]]) if len(idxs) > 0 else float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(res9["t"], res9["n"], color="royalblue")
        axes[0].axhline(n_target, ls="--", color="green", label=f"95% — {n_target:.0f} rpm")
        if not np.isnan(t95):
            axes[0].axvline(t95, ls=":", color="red", label=f"t95 = {t95:.3f} s")
        axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("Velocidade (RPM)")
        axes[0].set_title(f"Velocidade — J = {J_sim:.3f} kg·m²"); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(res9["t"], res9["Te"], color="firebrick")
        axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("Torque (Nm)")
        axes[1].set_title("Torque Eletromagnético"); axes[1].grid(True)

        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.metric("Tempo para 95% da vel. síncrona",
                  f"{t95:.3f} s" if not np.isnan(t95) else "Não atingido no intervalo")

elif experimento.startswith("10."):
    st.header("10. Variação da Frequência da Rede")
    st.markdown(
        "Análise estática (circuito equivalente). "
        "Tensão ajustada para manter $V/f$ constante."
    )

    col1, col2 = st.columns(2)
    st.markdown("#### Parâmetros específicos (circuito equivalente a 50 Hz)")
    R1_q10  = col1.number_input("R1 — Resist. estator (Ω)",  value=0.641, step=0.001, format="%.3f", key="q10_R1")
    R2_q10  = col1.number_input("R2 — Resist. rotor (Ω)",    value=0.332, step=0.001, format="%.3f", key="q10_R2")
    X1_q10  = col1.number_input("X1 — Reat. estator (Ω)",    value=1.106, step=0.001, format="%.3f", key="q10_X1")
    X2_q10  = col1.number_input("X2 — Reat. rotor (Ω)",      value=0.464, step=0.001, format="%.3f", key="q10_X2")
    Xm_q10  = col1.number_input("Xm — Reat. magnetiz. (Ω)",  value=26.3,  step=0.1,   key="q10_Xm")
    f_nom   = col2.number_input("Frequência nominal (Hz)",    value=50.0,  step=1.0,   key="q10_fnom")
    V_nom   = col2.number_input("Tensão nominal RMS L-L (V)", value=220.0, step=10.0,  key="q10_Vnom")
    T_mec   = col2.number_input("Torque mecânico (Nm)",       value=80.0,  step=1.0,   key="q10_Tmec")
    freqs_s = col2.text_input("Frequências a simular (Hz)", "50, 40", key="q10_freqs")

    def _Te_q10(V, freq, s):
        ws   = 2.0 * pi * freq / (p / 2.0)
        Z1v  = R1_q10 + 1j * X1_q10
        Zmv  = 1j * Xm_q10
        Z2v  = R2_q10 / s + 1j * X2_q10
        Zth  = Z1v + (Zmv * Z2v) / (Zmv + Z2v)
        I2   = V / Zth
        Pconv = 3.0 * abs(I2) ** 2 * (R2_q10 / s) * (1.0 - s)
        return Pconv / (ws * (1.0 - s))

    if st.button("Calcular", key="btn10"):
        try:
            freq_list = [float(v.strip()) for v in freqs_s.split(",") if v.strip()]
            if not freq_list:
                raise ValueError
        except ValueError:
            st.error("Insira frequências válidas separadas por vírgula.")
            st.stop()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        rows = []

        for freq in freq_list:
            V      = V_nom * (freq / f_nom)
            ns_rpm = 120.0 * freq / p
            s_vals = np.linspace(0.001, 0.1, 300)
            T_vals = np.array([_Te_q10(V, freq, s) for s in s_vals])

            idx_r  = int(np.argmin(np.abs(T_vals - T_mec)))
            s_r    = float(s_vals[idx_r])
            n_r    = ns_rpm * (1.0 - s_r)

            Z1v = R1_q10 + 1j * X1_q10
            Zmv = 1j * Xm_q10
            Z2v = R2_q10 / s_r + 1j * X2_q10
            I_r = V / (Z1v + (Zmv * Z2v) / (Zmv + Z2v))
            P_m = (2.0 * pi * n_r / 60.0) * T_mec
            P_e = 3.0 * abs(I_r) ** 2 * R1_q10 + P_m / (1.0 - s_r)
            eff = P_m / P_e * 100.0 if P_e > 0 else 0.0

            rows.append({
                "f (Hz)": freq, "ns (rpm)": round(ns_rpm, 1),
                "n_regime (rpm)": round(n_r, 1), "s (%)": round(s_r * 100, 2),
                "I_regime (A)": round(abs(I_r), 3),
                "P_mec (W)": round(P_m, 1), "η (%)": round(eff, 2),
            })

            axes[0].plot(ns_rpm * (1.0 - s_vals), T_vals, label=f"{freq} Hz")

            s_full = np.linspace(0.001, 1.0, 400)
            T_full = np.array([_Te_q10(V, freq, s) for s in s_full])
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
