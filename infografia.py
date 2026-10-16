"""Infografía estática del Laboratorio 9: síntesis ejecutiva del Laboratorio 1 (series de tiempo).

Lee las series mensuales y los resultados del Laboratorio 1 (carpeta datos/) y genera:
  salidas/infografia.png  2200 x 3400 px (11 x 17 in a 200 dpi)
  salidas/infografia.pdf  vectorial, 11 x 17 in, lista para imprimir

Uso:  python infografia.py
"""
import json
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')
RAIZ = Path(__file__).parent
SAL = RAIZ / 'salidas'
SAL.mkdir(exist_ok=True)

# ---------------------------------------------------------------- paleta y tipografía
INK = '#1B2733'         # texto principal
MUTED = '#55606C'       # texto secundario
AZUL = '#1F4E79'        # color temático 1: dato histórico, protagonista
AZUL_CL = '#9DBBD9'     # color temático 2: series de apoyo
GRIS = '#B9C0C8'        # contexto, referencia, temporada baja
NARANJA = '#D9661F'     # acento: anomalías, picos y estimación
NARANJA_TX = '#A84A12'  # acento oscurecido para texto pequeño (contraste AA)
PANEL = '#F2F5F8'       # fondo de tarjetas
REJILLA = '#E3E7EC'

plt.rcParams.update({
    'font.family': ['Segoe UI', 'Arial', 'DejaVu Sans'],
    'font.size': 10, 'text.color': INK, 'axes.labelcolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.edgecolor': GRIS,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42,
})

# ---------------------------------------------------------------- datos del Laboratorio 1
s = pd.read_csv(RAIZ / 'datos' / 'series_mensuales.csv', index_col=0, parse_dates=True).asfreq('MS')
s.columns = ['T', 'SV', 'GT', 'US', 'AIR', 'TER', 'MAR']
s['H'] = s['T'] - s['GT']   # definición homogénea: sin residentes de Guatemala
R = json.load(open(RAIZ / 'datos' / 'resultados.json', encoding='utf-8'))
C = R['comparativo']

# Índice estacional 2009-2019 sobre Turista + Excursionista (Lab 1, cuaderno 01, celda 51).
INDICE = [98.4, 81.2, 107.8, 105.7, 87.6, 95.5, 107.4, 104.4, 82.7, 92.0, 103.5, 133.8]
MESES = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

# ---------------------------------------------------------------- pronóstico del modelo ganador
# Partición de control del Lab 1: entrena jul-2021..dic-2024 y prueba los últimos 18 meses.
# Para anticipar el futuro se reajusta la misma especificación sobre 2023-2026, la única
# ventana con definición de medida constante, con la misma longitud (42 meses).
ORD, ORD_E = (1, 0, 1), (1, 0, 1, 12)


def ajusta(y):
    return SARIMAX(np.log1p(y), order=ORD, seasonal_order=ORD_E,
                   enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=500)


def bandas(fc):
    out = {'media': np.expm1(fc.predicted_mean)}
    for nivel in (80, 95):
        ci = np.expm1(fc.conf_int(alpha=1 - nivel / 100))
        out[nivel] = (ci.iloc[:, 0].clip(lower=0), ci.iloc[:, 1])
    return out


post = s['T'].loc['2021-07':]
tr, te = post.iloc[:-18], post.iloc[-18:]
pr_test = bandas(ajusta(tr).get_forecast(18))
pr_fut = bandas(ajusta(s['T'].loc['2023-01':]).get_forecast(18))

ctrl = R['control_postcovid_detalle']['Total de viajeros']
mape_ref = ctrl['SARIMA(1,0,1)(1,0,1,12)']['MAPE']
mape_rep = float(np.mean(np.abs(te - pr_test['media']) / te) * 100)
# El reajuste debe reproducir la métrica del Lab 1 (difiere solo por el optimizador).
assert abs(mape_rep - mape_ref) < 1, (mape_rep, mape_ref)
total_2027 = pr_fut['media'].loc['2027'].sum()

# ---------------------------------------------------------------- lienzo y utilidades
fig = plt.figure(figsize=(11, 17), facecolor='white')


def T(x, y, txt, **kw):
    return fig.text(x, y, txt, **kw)


def seccion(y, num, titulo, bajada):
    fig.add_artist(Line2D([0.05, 0.95], [y + 0.010, y + 0.010], color=GRIS, lw=0.8))
    T(0.05, y, num, fontsize=16, fontweight='bold', color=NARANJA, va='center')
    T(0.075, y, titulo, fontsize=16, fontweight='bold', color=INK, va='center')
    T(0.075, y - 0.0165, bajada, fontsize=10.5, color=MUTED, va='center')


def limpia(ax, grid='y'):
    ax.grid(axis=grid, color=REJILLA, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ================================================================ TITULAR Y KPIs
T(0.05, 0.979, 'TURISMO RECEPTOR EN GUATEMALA  ·  FLUJO MIGRATORIO 2009–2026',
  fontsize=10, fontweight='bold', color=NARANJA_TX, va='center')
T(0.05, 0.960, 'El turismo de Guatemala ya superó la pandemia', fontsize=32, fontweight='bold', va='center')
T(0.05, 0.935, 'Radiografía de 17 años de datos migratorios: un flujo que crece por tierra desde el vecindario,\n'
               'que sigue un calendario estable y que hoy se puede pronosticar con un error cercano al 7 %.',
  fontsize=12.5, color=MUTED, va='center', linespacing=1.35)

KPIS = [
    ('124 %', AZUL, 'visitantes extranjeros de 2025\nfrente a 2019, con la misma\ndefinición de medida'),
    ('8.7 %', AZUL, 'crecimiento anual compuesto\ndel ingreso de viajeros\nentre 2009 y 2019'),
    ('−97 %', NARANJA, 'caída interanual en abril de\n2020: el cierre de fronteras\ndejó ~10 mil ingresos al mes'),
    ('7 %', AZUL, 'error promedio (MAPE) del\nmejor modelo al pronosticar\n18 meses hacia adelante'),
]
w, g = 0.2125, 0.0125
for i, (num, col, lab) in enumerate(KPIS):
    x = 0.05 + i * (w + g)
    fig.add_artist(FancyBboxPatch((x, 0.848), w, 0.062, boxstyle='round,pad=0,rounding_size=0.005',
                                  transform=fig.transFigure, fc=PANEL, ec='none'))
    fig.add_artist(Line2D([x, x], [0.848, 0.910], color=col, lw=5, solid_capstyle='butt'))
    T(x + 0.017, 0.8915, num, fontsize=30, fontweight='bold', color=col, va='center')
    T(x + 0.017, 0.8615, lab, fontsize=9.3, color=INK, va='center', linespacing=1.25)

# ================================================================ 1. PANORAMA HISTÓRICO
seccion(0.825, '1', 'Diez años de crecimiento, un colapso y una recuperación completa',
        'Viajeros que ingresan al país cada mes. La línea azul excluye a los guatemaltecos residentes, '
        'que dejaron de contarse en 2023.')
ax = fig.add_axes([0.085, 0.668, 0.865, 0.125])
limpia(ax)
ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-30'), color=NARANJA, alpha=0.09, lw=0)
ax.plot(s.index, s['T'] / 1e3, color=GRIS, lw=1.2)
ax.plot(s.index, s['H'] / 1e3, color=AZUL, lw=1.9)
ax.set_ylim(0, 560)
ax.set_yticks([0, 100, 200, 300, 400, 500])
ax.set_xlim(pd.Timestamp('2008-12-01'), pd.Timestamp('2026-09-01'))
ax.set_ylabel('miles de viajeros / mes', fontsize=8.5)
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(axis='x', labelsize=8.3)

ax.text(pd.Timestamp('2015-06-01'), 365, 'Total registrado\n(incluye guatemaltecos residentes)',
        color=MUTED, fontsize=8.5, ha='right', va='center')
ax.text(pd.Timestamp('2016-09-01'), 105, 'Visitantes no residentes', color=AZUL, fontsize=9.5,
        fontweight='bold', ha='left', va='center')
ax.text(pd.Timestamp('2010-03-01'), 240, '+8.7 % anual\n2009–2019', color=AZUL, fontsize=10.5,
        fontweight='bold', ha='left', va='center')
ax.annotate('', xy=(pd.Timestamp('2019-03-01'), 325), xytext=(pd.Timestamp('2012-09-01'), 250),
            arrowprops=dict(arrowstyle='->', color=AZUL, lw=1.1))

fondo = pd.Timestamp('2020-05-01')
ax.scatter([fondo], [s.loc[fondo, 'T'] / 1e3], s=36, color=NARANJA, zorder=5)
ax.text(pd.Timestamp('2020-02-01'), 28, 'Abr–may 2020: ~10 mil al mes (−97 % interanual)',
        color=NARANJA_TX, fontsize=8.5, fontweight='bold', va='center', ha='right')
ax.text(pd.Timestamp('2020-05-01'), 548, 'Pandemia: fronteras\ncerradas desde mar-2020',
        color=NARANJA_TX, fontsize=8.3, va='top', ha='left')
ax.axvline(pd.Timestamp('2020-09-18'), color=NARANJA, lw=0.9, ls=':')
ax.text(pd.Timestamp('2020-10-20'), 455, '18-sep-2020:\nreapertura con\nprueba COVID',
        color=NARANJA_TX, fontsize=8, va='top')
ax.axvline(pd.Timestamp('2023-01-01'), color=INK, lw=0.9, ls='--')
ax.text(pd.Timestamp('2023-02-15'), 548, 'Ene-2023: cambio de medición.\nSe dejan de contar residentes\n'
        'guatemaltecos y cruceristas.', color=INK, fontsize=8.3, va='top')
ax.text(pd.Timestamp('2025-03-01'), 415, '2025: 3.6 M = 124 % de 2019', color=AZUL, fontsize=9.5,
        fontweight='bold', ha='center', va='center')

# ================================================================ 2. ESTACIONALIDAD Y COMPARATIVO
seccion(0.636, '2', 'Un calendario que se repite cada año y seis series con personalidad propia',
        'Izquierda: cuánto se aparta cada mes del mes promedio (2009–2019). '
        'Derecha: las cuatro preguntas del laboratorio, con evidencia.')

ax = fig.add_axes([0.07, 0.478, 0.425, 0.118])
dev = np.array(INDICE) - 100
col = [NARANJA if m == 'Dic' else AZUL if d > 3 else GRIS if d < -10 else AZUL_CL for m, d in zip(MESES, dev)]
ax.bar(MESES, dev, color=col, width=0.72)
ax.axhline(0, color=INK, lw=0.8)
ax.set_ylim(-30, 54)
ax.set_yticks([])
for lado in ('left', 'bottom'):
    ax.spines[lado].set_visible(False)
ax.tick_params(length=0, labelsize=9)
for i, d in enumerate(dev):
    ax.text(i, d + (1.5 if d >= 0 else -1.5), f'{d:+.0f}%', ha='center', va='bottom' if d >= 0 else 'top',
            fontsize=8.2, color=NARANJA_TX if MESES[i] == 'Dic' else INK,
            fontweight='bold' if abs(d) > 10 else 'normal')
for x, lab, c in [(2.5, 'Semana\nSanta', AZUL), (6.5, 'Vacaciones · Bono 14\nFiestas Agostinas (SV)', AZUL),
                  (10.6, 'Aguinaldo,\nfiestas y diáspora', NARANJA_TX)]:
    ax.text(x, 47, lab, ha='center', va='center', fontsize=8.2, color=c, fontweight='bold', linespacing=1.15)
T(0.07, 0.452, 'Diciembre recibe 34 % más viajeros que el mes promedio; febrero y\n'
               'septiembre (temporada baja), ~18 % menos. El patrón sobrevivió intacto\n'
               'a la pandemia: cambió el volumen, no el calendario.',
  fontsize=8.8, color=MUTED, va='top', linespacing=1.3)

NOMBRES = ['El Salvador', 'Guatemala', 'EE. UU.', 'Aérea', 'Terrestre', 'Marítima']
CLAVES = ['País: El Salvador', 'País: Guatemala', 'País: Estados Unidos de América',
          'Vía: Aérea', 'Vía: Terrestre', 'Vía: Marítima']
YPOS = [5.5, 4.5, 3.5, 2.0, 1.0, 0.0]
PANELES = [  # pregunta, respuesta (países · vías), métrica, eje, máximo, ganadores, formato
    ('¿Mayor estacionalidad?', 'EE. UU.  ·  Aérea', 'F_estacional', 'fuerza estacional (0 a 1)', 1.0,
     ('País: Estados Unidos de América', 'Vía: Aérea'), '{:.2f}'),
    ('¿Mayor crecimiento?', 'El Salvador  ·  Terrestre', 'TCAC_09_19_%', '% anual compuesto 2009–2019', 14,
     ('País: El Salvador', 'Vía: Terrestre'), '{:.1f}%'),
    ('¿Mayor volatilidad?', 'El Salvador  ·  Marítima', 'volatilidad_yoy', 'desv. est. del cambio interanual',
     0.26, ('País: El Salvador', 'Vía: Marítima'), '{:.2f}'),
    ('¿Más golpeada por la pandemia?', 'El Salvador  ·  Marítima', 'rec_homog_%', 'nivel 2025 vs. 2019 (%)',
     175, ('País: El Salvador', 'Vía: Marítima'), '{:.0f}%'),
]
pw, ph = 0.120, 0.052
for k, (preg, resp, met, xlab, xmax, ganan, fmt) in enumerate(PANELES):
    px = 0.620 + (k % 2) * (pw + 0.090)
    py = 0.527 - (k // 2) * (ph + 0.040)
    a = fig.add_axes([px, py, pw, ph])
    corta = met == 'volatilidad_yoy'   # único panel con un valor fuera de escala
    a.set_xlim(0, xmax * (1.0 if corta else 1.28))
    a.set_xticks([t for t in a.get_xticks() if t <= xmax])
    a.set_ylim(-0.6, 6.1)
    a.set_yticks(YPOS)
    a.set_yticklabels(NOMBRES, fontsize=8)
    a.tick_params(length=0, labelsize=7.3, pad=2)
    a.spines['left'].set_visible(False)
    a.grid(axis='x', color=REJILLA, lw=0.6)
    a.set_axisbelow(True)
    a.set_xlabel(xlab, fontsize=7.6, labelpad=1)
    for y, c in zip(YPOS, CLAVES):
        v = C[c][met]
        acento = c == 'Vía: Marítima' and met in ('volatilidad_yoy', 'rec_homog_%')
        colr = (NARANJA if acento else AZUL) if c in ganan else AZUL_CL
        if v is None or np.isnan(v):
            a.text(xmax * 0.02, y, 'n/d: dejó de medirse', va='center', fontsize=7.2, color=MUTED, style='italic')
            continue
        largo = min(v, xmax * 0.80) if corta else v
        a.barh(y, largo, height=0.75, color=colr)
        if largo < v:  # fuera de escala: la barra se corta y se marca; el eje no se comprime
            a.text(largo + xmax * 0.015, y, f'{fmt.format(v)} ≫', va='center', fontsize=7.5,
                   color=NARANJA_TX, fontweight='bold')
        else:
            a.text(largo + xmax * 0.015, y, fmt.format(v), va='center', fontsize=7.5,
                   color=INK if c in ganan else MUTED, fontweight='bold' if c in ganan else 'normal')
    if met == 'rec_homog_%':
        a.axvline(100, color=INK, lw=0.8, ls='--')
    a.text(-0.55, 1.36, preg, transform=a.transAxes, fontsize=9.3, fontweight='bold', color=INK)
    a.text(-0.55, 1.12, resp, transform=a.transAxes, fontsize=9.3, fontweight='bold',
           color=NARANJA_TX if met in ('volatilidad_yoy', 'rec_homog_%') else AZUL)
T(0.565, 0.417, 'Línea punteada = nivel 2019. Volatilidad marítima fuera de escala: 12 veces la terrestre.',
  fontsize=7.8, color=MUTED, va='top')

# ================================================================ 3. MODELOS Y PRONÓSTICO
seccion(0.392, '3', 'Con datos post-pandemia el turismo es predecible: error de 7 %',
        'Izquierda: error de cada algoritmo en las dos pruebas del laboratorio. '
        'Derecha: el modelo ganador frente a lo que realmente ocurrió.')

tot = R['series']['Total de viajeros']['comparacion']
MODELOS = [('SARIMA(1,0,1)(1,0,1,12)', 'SARIMA (1,0,1)(1,0,1)₁₂'), ('Seasonal naive', 'Ingenuo estacional'),
           ('Holt-Winters', 'Holt-Winters'), ('Suav. exponencial simple', 'Suaviz. exponencial'),
           ('Prophet', 'Prophet'), ('SARIMA(0,0,1)(0,0,1,12)', 'SARIMA (0,0,1)(0,0,1)₁₂')]
ax = fig.add_axes([0.205, 0.252, 0.205, 0.086])
limpia(ax, 'x')
for i, (clave, nombre) in enumerate(MODELOS):
    y = len(MODELOS) - 1 - i
    a_, b_ = tot[clave]['MAPE'], ctrl[clave]['MAPE']
    gana = i == 0
    ax.plot([b_, a_], [y, y], color=GRIS, lw=2, zorder=1)
    ax.scatter(a_, y, s=34, facecolor='white', edgecolor=MUTED, lw=1.2, zorder=2)
    ax.scatter(b_, y, s=44, color=NARANJA if gana else AZUL, zorder=3)
    ax.text(b_ - 3, y + 0.02, f'{b_:.1f}%', ha='right', va='center', fontsize=7.8,
            color=NARANJA_TX if gana else AZUL, fontweight='bold')
    ax.text(1.06, y, f'{ctrl[clave]["RMSE"] / 1e3:,.1f}', transform=ax.get_yaxis_transform(), ha='left',
            va='center', fontsize=7.8, color=INK if gana else MUTED, fontweight='bold' if gana else 'normal')
    ax.text(1.32, y, f'{ctrl[clave]["MAE"] / 1e3:,.1f}', transform=ax.get_yaxis_transform(), ha='left',
            va='center', fontsize=7.8, color=INK if gana else MUTED, fontweight='bold' if gana else 'normal')
ax.set_yticks(range(len(MODELOS)))
ax.set_yticklabels([n for _, n in MODELOS][::-1], fontsize=8.3)
ax.get_yticklabels()[-1].set_color(NARANJA_TX)
ax.get_yticklabels()[-1].set_fontweight('bold')
ax.set_xlim(-12, 105)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels([f'{v}%' for v in (0, 25, 50, 75, 100)])
ax.set_ylim(-0.6, 5.9)
ax.spines['left'].set_visible(False)
ax.set_xlabel('MAPE: error porcentual promedio', fontsize=7.8, labelpad=2)
ax.text(1.06, 5.75, 'RMSE', transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED, fontweight='bold')
ax.text(1.32, 5.75, 'MAE', transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED, fontweight='bold')
T(0.05, 0.360, '●', fontsize=9, color=AZUL, va='center')
T(0.062, 0.360, 'prueba post-pandemia (ene-2025 a jun-2026)', fontsize=8.2, color=INK, va='center')
T(0.300, 0.360, '○', fontsize=9, color=MUTED, va='center')
T(0.312, 0.360, 'prueba 70/30 con la pandemia', fontsize=8.2, color=INK, va='center')
T(0.05, 0.349, 'RMSE y MAE en miles de viajeros por mes, prueba post-pandemia.', fontsize=7.8, color=MUTED,
  va='center')
T(0.05, 0.231,'En la prueba 70/30 todos fallan (58–100 %): se entrenaron con la pandemia y no podían prever\n'
               'la reapertura. Sin ese choque el error cae a 6–12 %. Gana el SARIMA por menor RMSE.',
  fontsize=8.3, color=MUTED, va='top', linespacing=1.3)

ax = fig.add_axes([0.555, 0.240, 0.395, 0.115])
limpia(ax)
ax.axvspan(te.index[0], te.index[-1] + pd.offsets.MonthEnd(0), color=PANEL, lw=0)
ax.text(te.index[0] + pd.Timedelta(days=12), 640, 'periodo de prueba', fontsize=7.8, color=MUTED, va='top')
# La banda de 95 % de la prueba llega a ~1.3 M: se corta en el eje y se declara, para no aplastar la serie.
ax.text(te.index[0] - pd.Timedelta(days=20), 640, f'IC 95 % llega a {pr_test[95][1].max() / 1e6:.1f} M ▲',
        fontsize=7.6, color=NARANJA_TX, va='top', ha='right')
for pr in (pr_test, pr_fut):
    idx = pr['media'].index
    ax.fill_between(idx, pr[95][0] / 1e3, pr[95][1] / 1e3, color=NARANJA, alpha=0.14, lw=0)
    ax.fill_between(idx, pr[80][0] / 1e3, pr[80][1] / 1e3, color=NARANJA, alpha=0.26, lw=0)
ax.plot(post.index, post / 1e3, color=AZUL, lw=1.8)
ax.plot(pr_test['media'].index, pr_test['media'] / 1e3, color=NARANJA, lw=1.6, ls='--')
ax.plot(pr_fut['media'].index, pr_fut['media'] / 1e3, color=NARANJA, lw=1.6, ls='--')
ax.axvline(pd.Timestamp('2023-01-01'), color=INK, lw=0.8, ls='--')
ax.set_ylim(0, 650)
ax.set_xlim(pd.Timestamp('2021-06-01'), pd.Timestamp('2028-01-01'))
ax.set_ylabel('miles de viajeros / mes', fontsize=8.5)
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.text(pd.Timestamp('2023-05-01'), 430, 'Real', color=AZUL, fontsize=9.5, fontweight='bold')
ax.text(pd.Timestamp('2025-01-15'), 35, f'Estimación (prueba)\nMAPE {mape_ref:.1f} %',
        color=NARANJA_TX, fontsize=8.2, fontweight='bold', linespacing=1.15)
ax.text(pd.Timestamp('2027-12-20'), 590, f'Estimación jul-2026 a dic-2027\n2027 ≈ {total_2027 / 1e6:.1f} M viajeros',
        color=NARANJA_TX, fontsize=8.2, fontweight='bold', linespacing=1.15, ha='right', va='top')
ax.text(te.index[0] - pd.Timedelta(days=20), 590, 'banda oscura: IC 80 %\nbanda clara: IC 95 %',
        color=NARANJA_TX, fontsize=7.6, ha='right', va='top', linespacing=1.15)

# ================================================================ 4. RECOMENDACIONES
seccion(0.197, '4', 'Qué debe hacer el INGUAT', 'Cinco decisiones que se desprenden directamente de la evidencia.')
RECS = [
    ('Invertir donde está el crecimiento: el vecindario y la frontera terrestre.',
     'El Salvador (+12.1 % anual) y la vía terrestre (+11.9 %) crecen más del doble que EE. UU. y la aérea '
     '(+4.7 %). Promoción regional primero.'),
    ('Planificar personal y capacidad con el calendario, que no cambió.',
     'Más personal y capacidad en diciembre (+34 %), Semana Santa y julio–agosto; febrero y septiembre (−18 %) '
     'para ofertas de temporada baja.'),
    ('Tratar a EE. UU. como un mercado planificable y de temporada.',
     'Es la serie más estacional (0.93) y la de mejor recuperación (146 % de 2019): campañas anticipadas '
     'a verano y Navidad rinden más que esfuerzos de volumen todo el año.'),
    ('Decidir explícitamente el futuro de los cruceros.',
     'La vía marítima está en 5 % de su nivel de 2019 y es la más volátil del estudio. O se invierte en '
     'puertos y acuerdos con navieras, o se reasigna su presupuesto de promoción.'),
    ('Presupuestar con pronóstico y corregir la medición.',
     'Un modelo simple anticipa 18 meses con ~7 % de error. Publicar una serie con definición constante y '
     'marcar los cierres como “no medido”.'),
]
y = 0.163
for i, (lead, body) in enumerate(RECS, 1):
    # círculo real en un lienzo de 11x17: el alto en fracción de figura se escala por 11/17
    fig.add_artist(Ellipse((0.062, y - 0.0005), 0.018, 0.018 * 11 / 17, transform=fig.transFigure,
                           color=AZUL, zorder=3))
    T(0.062, y - 0.0005, str(i), fontsize=10, fontweight='bold', color='white', ha='center', va='center',
      zorder=4)
    T(0.082, y + 0.0045, lead, fontsize=10.5, fontweight='bold', color=INK, va='center')
    T(0.082, y - 0.0080, body, fontsize=8.9, color=INK, va='center', linespacing=1.25, wrap=True)
    y -= 0.0255

# ================================================================ FICHA TÉCNICA
fig.add_artist(Line2D([0.05, 0.95], [0.040, 0.040], color=GRIS, lw=0.8))
T(0.05, 0.031, 'FUENTE', fontsize=7.6, fontweight='bold', color=MUTED, va='center')
T(0.05, 0.020, 'Base de Migración 2009–2026 (jun), INGUAT / IGM.\n161,036 registros mensuales. Datos de uso '
               'académico.', fontsize=7.6, color=MUTED, va='center', linespacing=1.25)
T(0.37, 0.031, 'MÉTODO', fontsize=7.6, fontweight='bold', color=MUTED, va='center')
T(0.37, 0.020, '10 modelos por serie (SARIMA, Prophet, Holt-Winters, suavizamiento\nexponencial, ingenuo '
               'estacional). Partición 70/30 y control post-pandemia.', fontsize=7.6, color=MUTED,
  va='center', linespacing=1.25)
T(0.70, 0.031, 'AUTORES', fontsize=7.6, fontweight='bold', color=MUTED, va='center')
T(0.70, 0.020, 'Rodrigo Ajmac 22279 · Andrés Mazariegos 21749 ·\nJune Herrera 231038 · CC3084 Data Science, '
               'UVG 2026', fontsize=7.6, color=MUTED, va='center', linespacing=1.25)

fig.savefig(SAL / 'infografia.png', dpi=200, facecolor='white')
fig.savefig(SAL / 'infografia.pdf', facecolor='white')
print(f'MAPE reproducido {mape_rep:.2f} % (Lab 1: {mape_ref:.2f} %) · 2027 estimado {total_2027:,.0f}')
print('Generado:', SAL / 'infografia.png', '·', SAL / 'infografia.pdf')
