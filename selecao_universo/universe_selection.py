# -*- coding: utf-8 -*-
# Seleção de universo MecaQuant — correlações de retornos mensais (log), fonte Yahoo Finance
# Pareamento: apenas meses em comum entre cada par; mínimo 60 meses de overlap.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

TICKERS = ['^BVSP','BOVA11.SA','SMAL11.SA','SPY','QQQ','IWM','EFA','EEM','EWZ','IEF','TLT',
           'LQD','HYG','BRL=X','EURUSD=X','JPY=X','GBPUSD=X','AUDUSD=X','GLD','SLV','DBC',
           'USO','DBA','BTC-USD','ETH-USD','VNQ']

LOWER = {
 'BOVA11.SA':[100],
 'SMAL11.SA':[88,89],
 'SPY':[59,51,50],
 'QQQ':[53,42,43,85],
 'IWM':[58,50,51,88,75],
 'EFA':[65,58,55,86,71,79],
 'EEM':[73,65,59,75,69,71,86],
 'EWZ':[93,93,83,59,48,56,66,77],
 'IEF':[-12,-13,-8,-10,-10,-15,-2,-3,-9],
 'TLT':[-15,-17,-10,-9,-9,-14,-4,-6,-13,92],
 'LQD':[28,35,39,40,33,33,46,41,32,67,65],
 'HYG':[56,57,58,75,68,73,77,71,59,8,4,69],
 'BRL=X':[-36,-36,-37,-18,-14,-18,-24,-26,-36,9,8,-15,-22],
 'EURUSD=X':[20,15,23,12,12,11,26,28,20,2,5,23,24,-43],
 'JPY=X':[5,3,-1,6,7,5,3,2,9,-21,-20,-12,0,14,-30],
 'GBPUSD=X':[28,25,23,13,13,11,21,26,23,-14,-13,6,13,-34,66,-16],
 'AUDUSD=X':[27,19,24,19,17,20,30,35,25,5,10,28,29,-59,74,-21,57],
 'GLD':[32,26,23,9,5,5,22,34,35,31,23,27,17,-3,7,-14,4,3],
 'SLV':[36,35,30,27,23,22,36,44,41,13,6,30,29,-11,9,0,9,10,80],
 'DBC':[51,44,35,46,35,44,53,56,58,-26,-31,10,44,-21,8,15,12,14,32,44],
 'USO':[41,37,30,35,26,37,39,39,46,-32,-34,-1,31,-25,2,13,12,9,5,22,83],
 'DBA':[35,33,30,30,19,25,38,38,45,-5,-13,19,36,-15,15,5,10,19,30,36,63,38],
 'BTC-USD':[18,18,24,37,35,30,27,16,17,0,3,20,32,-5,14,7,0,15,3,9,16,9,23],
 'ETH-USD':[20,20,23,36,36,28,26,13,18,-1,4,20,32,-4,-1,14,-9,5,2,9,12,5,33,76],
 'VNQ':[45,43,49,75,62,75,72,62,46,12,14,50,76,-22,13,-10,6,22,15,22,33,23,26,23,23],
}

STATS = {  # ticker: (inicio, vol anualizada %)
 '^BVSP':('2000-01',24.0),'BOVA11.SA':('2009-01',21.6),'SMAL11.SA':('2009-01',24.8),
 'SPY':('2000-01',15.3),'QQQ':('2000-01',23.3),'IWM':('2000-07',20.2),
 'EFA':('2001-10',16.8),'EEM':('2003-06',20.9),'EWZ':('2000-09',35.1),
 'IEF':('2002-09',6.6),'TLT':('2002-09',13.4),'LQD':('2002-09',7.8),'HYG':('2007-06',10.2),
 'BRL=X':('2004-01',16.0),'EURUSD=X':('2004-01',9.4),'JPY=X':('2000-01',9.4),
 'GBPUSD=X':('2004-01',8.9),'AUDUSD=X':('2006-06',12.8),
 'GLD':('2005-01',17.1),'SLV':('2006-06',32.4),'DBC':('2006-04',19.1),
 'USO':('2006-06',40.4),'DBA':('2007-03',15.4),
 'BTC-USD':('2014-11',67.2),'ETH-USD':('2017-12',92.3),'VNQ':('2004-11',22.1),
}

n = len(TICKERS)
C = np.eye(n)
for i in range(1, n):
    row = LOWER[TICKERS[i]]
    for j in range(i):
        C[i, j] = C[j, i] = row[j] / 100.0
C = np.clip(C, -1, 1)
np.fill_diagonal(C, 1.0)
corr = pd.DataFrame(C, index=TICKERS, columns=TICKERS)
corr.to_csv("correlation_matrix_candidatos.csv")

# --- Clustering hierárquico (average linkage) sobre distância 1 - rho ---
D = 1 - C
np.fill_diagonal(D, 0.0)
Z = linkage(squareform(D, checks=False), method="average")
CUT = 0.35  # junta ativos com correlação média > 0.65
labels = fcluster(Z, t=CUT, criterion="distance")

clusters = {}
for t, l in zip(TICKERS, labels):
    clusters.setdefault(l, []).append(t)

print("=== CLUSTERS (corr média intra-cluster > 0.65) ===")
for l, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
    print(f"Cluster {l}: {members}")

# --- Elegibilidade: histórico >= 15 anos (início <= 2011-07) ---
eligible = {t for t, (start, _) in STATS.items() if start <= '2011-07'}
print("\nInelegíveis por histórico curto (<15 anos):", sorted(set(TICKERS) - eligible))

# --- Heatmap ordenado pelo dendrograma ---
order = dendrogram(Z, no_plot=True)["leaves"]
ordered = [TICKERS[i] for i in order]
Co = corr.loc[ordered, ordered]

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(Co.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(ordered, rotation=90, fontsize=9)
ax.set_yticklabels(ordered, fontsize=9)
for i in range(n):
    for j in range(n):
        v = Co.values[i, j]
        ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(v) > 0.55 else "black")
ax.set_title("Correlação de retornos mensais (2000–2026, pareamento por overlap ≥ 60 meses)\n"
             "Ordenado por clustering hierárquico — fonte: Yahoo Finance", fontsize=11)
fig.colorbar(im, shrink=0.7, label="correlação")
fig.tight_layout()
fig.savefig("heatmap_correlacao_candidatos.png", dpi=160)
print("\nHeatmap salvo. Ordem dendrograma:", ordered)
