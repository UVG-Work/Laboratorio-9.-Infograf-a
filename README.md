# Laboratorio 9 — Infografía estática

Síntesis ejecutiva para el INGUAT de los resultados del Laboratorio 1 (series de tiempo del
ingreso de viajeros a Guatemala, 2009–2026).

| Archivo | Contenido |
|---|---|
| `infografia.py` | Genera la infografía: `salidas/infografia.png` (2200×3400 px) y `salidas/infografia.pdf` (11×17 in) |
| `documento/explicacion.tex` | Documento de explicación de las decisiones de diseño |
| `datos/` | Series mensuales y resultados del Laboratorio 1 (`series_mensuales.csv`, `resultados.json`) |

```bash
pip install pandas numpy matplotlib statsmodels
python infografia.py
cd documento && xelatex explicacion.tex
```

Las salidas renderizadas (`salidas/` y `documento/explicacion.pdf`) se versionan junto al código.
