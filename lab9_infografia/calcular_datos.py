"""
Prepara los datos de la infografía a partir del repo del Lab 1 (CC3084).
Uso (desde la RAÍZ de una copia del repo con data/processed/migracion_limpia.csv generado):
    python lab9_infografia/calcular_datos.py
Genera tot.csv (serie mensual Total), te_ex.csv (Turista + Excursionista) y prophet_fc.csv
(pronóstico de Prophet sobre el conjunto de prueba, intervalo de confianza al 80%).
Luego:  python lab9_infografia/graficas_lab9.py
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'01_limpieza_eda/src'); sys.path.insert(0,'02_series_tiempo/src'); sys.path.insert(0,'03_modelado_prediccion/src')
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path(__file__).resolve().parent   # los CSV se guardan junto a este script
import series, modelado as mod
df=series.cargar_limpio()
tot=series.serie_mensual(df)
tr,te=series.split_temporal(tot)
print('n',len(tot),'train',len(tr),tr.index.min().date(),tr.index.max().date(),'test',len(te),te.index.min().date(),te.index.max().date())
print('Total acumulado',tot.sum(), 'registros',len(df))
print('Pico',tot.idxmax().date(),tot.max()); 
print('Minimo',tot.idxmin().date(),tot.min())
m19=tot['2019-04-01':'2019-12-01'].sum(); m20=tot['2020-04-01':'2020-12-01'].sum(); print('caida pandemia total abr-dic',(m20-m19)/m19*100)
print('anual'); print(tot.groupby(tot.index.year).sum().astype(int).to_string())
te_ex=series.serie_mensual(df,'Tipo de Viajero','Turista')+series.serie_mensual(df,'Tipo de Viajero','Excursionista')
print('Turista+Excursionista anual'); print(te_ex.groupby(te_ex.index.year).sum().astype(int).to_string())
# prophet reproduction
pred,m=mod.modelo_prophet(tr,len(te),te.index)
print(mod.metricas(te,pred))
fc=m.predict(pd.DataFrame({'ds':te.index}))
out=pd.DataFrame({'yhat':fc['yhat'].values,'lo':fc['yhat_lower'].values,'hi':fc['yhat_upper'].values},index=te.index)
print(out.head(3)); print('interval width',m.interval_width)
tot.to_csv(HERE/'tot.csv'); out.to_csv(HERE/'prophet_fc.csv'); te_ex.to_csv(HERE/'te_ex.csv')
