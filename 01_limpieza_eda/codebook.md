# Codebook — Base de Migración (2009–jun 2026)

Ingreso mensual de viajeros internacionales a Guatemala. 161,015 registros tras limpieza; 210 meses.

| Variable | Tipo | Descripción |
|---|---|---|
| `Año` | entero | Año de ingreso al país (2009–2026). |
| `Mes cod` | entero | Código del mes (1–12). |
| `Mes` | texto | Nombre corto del mes (Ene…Dic). |
| `Vía` | categórica | Vía de entrada: Aérea, Terrestre, Marítima. |
| `Frontera` | categórica | Frontera de ingreso (22 categorías; p. ej. `01 La Aurora`). |
| `País` | categórica | Hasta 2023: país de procedencia; desde 2023: agrupación de mercado. |
| `Región` | categórica | Clasificación para reportes nacionales. |
| `Región dos` | categórica | Agrupación en continentes/grandes áreas (11 categorías válidas). |
| `Regiones OMT` | categórica | Subregión de la Organización Mundial del Turismo. |
| `MCEO` | categórica | Mercado/agrupación comercial estratégica. |
| `Agrupación Residencia` | categórica | Región donde reside el viajero. |
| `Tipo de Viajero` | categórica | Turista, Excursionista, Viajero, Cruceristas. |
| `Viajero` | numérica | Cantidad de viajeros (puede ser fraccionaria por estimación). |
| `fecha` | fecha | **Derivada:** primer día del mes (Año + Mes cod). |

## Advertencias metodológicas
- **`Viajero` (categoría):** entre 2022 y 2023 se excluyeron viajeros no turísticos de alta
  frecuencia (tránsito/comercio fronterizo), generando una caída artificial desde 2023.
  Para comparaciones de largo plazo usar **Turista + Excursionista**.
- **`País`:** El Salvador y Guatemala dominan el acumulado por tránsito terrestre, no por turismo aéreo.
- **Fuentes por tramo:** 2009–2020 respaldos históricos; 2021–2022 entrega del IGM; 2023–jun 2026 sistema INGUAT.
