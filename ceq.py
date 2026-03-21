# -*- coding: utf-8 -*-
"""
ceq.py — Circuito Equivalente Monofásico em T (MIT)
Desenha o circuito com schemdraw.

Uso como módulo (Streamlit):
    from ceq import render_circuit
    render_circuit(mp, dark, _palette)

Uso standalone (visualização direta):
    python ceq.py           # fundo escuro (padrão)
    python ceq.py --light   # fundo claro
"""

from __future__ import annotations
import io
import sys
from typing import Any, Callable
import matplotlib
import matplotlib.figure
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm


def build_figure(mp: Any, dark: bool, palette_fn: Callable[[bool], dict[str, str]]) -> matplotlib.figure.Figure:
    """
    Constrói e retorna a figura matplotlib com o circuito equivalente.
    Não depende de Streamlit — pode ser usada standalone ou em testes.

    Parâmetros
    ----------
    mp          : MachineParams — parâmetros da máquina (Rs, Rr, Xm, Xls, Xlr)
    dark        : bool — True = fundo escuro, False = fundo claro
    palette_fn  : callable — função _palette(dark) → dict com chaves 'muted', etc.
    """
    c      = palette_fn(dark)
    bg_hex = "#0d1117" if dark else "#ffffff"
    wire   = "#e4e8f5" if dark else "#111827"
    muted  = c["muted"]

    OFST   = 0.20
    OFSTV  = 0.30
    FS     = 12
    FS_VAL = 12

    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor(bg_hex)
    ax.set_facecolor(bg_hex)
    ax.set_axis_off()

    with schemdraw.Drawing(canvas=ax) as d:
        d.config(fontsize=10, color=wire)

        # ── fonte de tensão Vs ──────────────────────────────────────────
        src = d.add(
            elm.SourceSin()
            .up()
            .color(wire)
            .label(r"$V_s$", loc="top",ofst = OFSTV, fontsize=FS, color=wire)
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
            .label(r"$jX_m$",          loc="top",  ofst=OFSTV, fontsize=FS,     color=wire)
            .label(f"{mp.Xm:.2f} Ω",  loc="bot", ofst=OFSTV, fontsize=FS_VAL, color=wire)
        )
        d.add(elm.Line().down().toy(bot_right))
        d.add(elm.Ground().color(wire))


    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.08)
    return fig


def render_circuit(mp: Any, dark: bool, palette_fn: Callable[[bool], dict[str, str]]) -> None:
    """Gera o circuito e exibe via st.image (uso Streamlit)."""
    import streamlit as st

    bg_hex = "#0d1117" if dark else "#ffffff"
    fig = build_figure(mp, dark, palette_fn)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=bg_hex, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    st.image(buf, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Execução standalone: python ceq.py [--light]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataclasses import dataclass, field
    import numpy as np

    @dataclass
    class _MP:
        Vl: float = 220.0
        f:  float = 60.0
        Rs: float = 0.435
        Rr: float = 0.816
        Xm: float = 26.13
        Xls: float = 0.754
        Xlr: float = 0.754
        p:  int   = 4
        J:  float = 0.089
        B:  float = 0.0
        Xml: float = field(init=False)
        wb:  float = field(init=False)
        def __post_init__(self):
            self.Xml = 1.0 / (1.0/self.Xm + 1.0/self.Xls + 1.0/self.Xlr)
            self.wb  = 2.0 * np.pi * self.f

    def _palette(dark: bool) -> dict[str, str]:
        if dark:
            return dict(muted="#8892b0", text="#e4e8f5", accent="#4f8ef7",
                        border="#2a3150", surface="#161b27")
        return dict(muted="#4b5563", text="#111827", accent="#2563eb",
                    border="#d0d8f0", surface="#ffffff")

    dark = "--light" not in sys.argv
    mp   = _MP()

    matplotlib.use("TkAgg")   # backend interativo para exibir janela
    fig  = build_figure(mp, dark, _palette)

    plt.show()
