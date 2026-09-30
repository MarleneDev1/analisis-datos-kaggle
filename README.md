# Análisis de datos Kaggle: Language Learning Fluency

EDA (análisis exploratorio de datos) de un dataset de 45.000 personas que aprenden idiomas. El objetivo es entender qué factores explican que alguien **alcance la fluidez** (`reached_fluency`, nivel ≥ B2) o **abandone** (`dropped_out`).

## Estructura

```
data/
  language_learning_fluency.csv   # 45.000 filas x 18 columnas
  data_dictionary_language.csv    # descripción de cada columna
eda_language_learning.py          # script del EDA
index.html                        # portafolio web (Tailwind CSS vía CDN)
outputs/
  figures/                        # 13 gráficos PNG
  tables/                         # tablas de resultados en CSV
```

## Ejecución

```bash
pip install -r requirements.txt
python eda_language_learning.py
```

El script imprime el análisis en consola y regenera `outputs/`. Tarda unos 20 s.

## Portafolio web

`index.html` es mi portafolio en inglés: una página responsiva de tema oscuro (negro, gris, violeta y rosa) hecha con Tailwind CSS vía CDN. Incluye una presentación, este proyecto contado como caso de estudio (métricas, gráfica interactiva, hallazgos y figuras de `outputs/figures/`), mis habilidades y enlaces a LinkedIn y GitHub. Para verla, abre el archivo en el navegador o publícala con GitHub Pages desde la raíz de `main`.

## Contenido del EDA

1. Carga de los datos y validación contra el diccionario (columnas y tipos).
2. Clasificación de variables: identificador, objetivos, binarias, categóricas y numéricas.
3. Calidad de datos: valores faltantes, duplicados, rangos válidos y coherencia entre columnas.
4. Estadísticas descriptivas: percentiles, asimetría, curtosis, CV, % de ceros y outliers (IQR).
5. Análisis univariado: histogramas, boxplots y conteos.
6. Variables objetivo: balance de clases y resultado combinado (fluido / en progreso / abandono).
7. Correlaciones de Spearman y detección de multicolinealidad.
8. Análisis bivariado: horas × dificultad, horas × uso activo, abandono × motivación, edad × acento.
9. Importancia de variables con tres métodos: información mutua, Gini de Random Forest y permutación (ROC-AUC).

## Hallazgos principales

**Calidad de datos**
- No hay valores faltantes, filas duplicadas ni IDs repetidos.
- Todos los valores están dentro de los rangos del diccionario.
- `reached_fluency` coincide al 100 % con `cefr_level ∈ {B2, C1, C2}`. Es decir, `cefr_level` es otra forma de la variable objetivo y **no debe usarse como predictor** porque filtraría la respuesta al modelo.
- Nadie tiene `reached_fluency = 1` y `dropped_out = 1` a la vez, así que hay tres resultados: fluido (35,8 %), abandono (37,1 %) y en progreso (27,2 %).
- `total_study_hours` y `comprehensible_input_hours` están muy correlacionadas (ρ = 0,94). Conviene tenerlo en cuenta en modelos lineales.
- `total_study_hours`, `comprehensible_input_hours` e `immersion_months` tienen asimetría positiva (el 60 % tiene 0 meses de inmersión). Una transformación logarítmica ayudaría.

**Variables más relevantes**

| Rango | Para `reached_fluency` | Para `dropped_out` |
|---|---|---|
| 1 | `total_study_hours` (ρ = 0,58) | `fsi_category` (ρ = 0,27) |
| 2 | `fsi_category` (ρ = −0,35) | `total_study_hours` (ρ = −0,30) |
| 3 | `comprehensible_input_hours` (ρ = 0,54) | `active_use_share` (ρ = −0,20) |
| 4 | `active_use_share` (ρ = 0,29) | `comprehensible_input_hours` |
| 5 | `immersion_months` (ρ = 0,12) | `motivation` (ρ = −0,14) y `consistency` (ρ = −0,14) |

Un Random Forest con estas variables obtiene un accuracy de **0,89** para fluidez (baseline 0,64) y de **0,73** para abandono (baseline 0,63).

**Conclusiones**
- **Horas × dificultad**: con 580–700 h llega a B2 el 80 % de quienes estudian un idioma FSI 1, pero solo el 7 % de quienes estudian uno FSI 4. En FSI 4 ni siquiera con más de 1.150 h se supera el 50 %.
- **La calidad importa**: a igualdad de horas, un mayor `active_use_share` (práctica activa frente a repaso pasivo) aumenta claramente la tasa de fluidez.
- **Motivación y consistencia** no predicen la fluidez (ρ ≈ 0), pero sí el **abandono**.
- **Edad de inicio**: no influye en la fluidez (≈ 36 % en todos los rangos de edad), pero sí en el acento. `accent_nativelike` baja de 8,0 (inicio antes de los 12 años) a 3,1 (inicio después de los 45).
- `uses_srs`, `prior_languages` y `related_language` tienen un efecto casi nulo.
