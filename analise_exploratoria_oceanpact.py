import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ANALISE EXPLORATORIA DE DADOS - OCEANPACT
# Fonte: Etapa2_Proposta_Analitica_AED_Grupo19.pdf
# ============================================================

dados = pd.DataFrame({
    "ano": [2023, 2024, 2025],
    "escopo1_tco2e": [54446.1, 47206.2, 56184.7],
    "escopo2_tco2e": [65.6, 42.8, 87.8],
    "escopo3_tco2e": [133607.5, 196308.8, 201264.9],
    "energia_gj": [380302.5, 441973.3, 452605.0],
    "ocupacao_frota_pct": [87, 79, 82],
    "colaboradores": [2089, 2162, 2662]
})

# Valores totais publicados no documento.
dados["emissoes_total_tco2e"] = [188096.4, 243602.8, 257515.2]

# Intensidade recalculada apresentada no documento.
dados["intensidade_calc_tco2e_por_mi"] = [114.1, 141.5, 120.7]

print("=" * 70)
print("ANALISE EXPLORATORIA DE DADOS - OCEANPACT")
print("=" * 70)

# ------------------------------------------------------------
# 1. Dimensoes e tipos
# ------------------------------------------------------------
print("\n1. DIMENSOES DA BASE")
print("Linhas:", dados.shape[0])
print("Colunas:", dados.shape[1])

print("\nTipos de dados:")
print(dados.dtypes)

# ------------------------------------------------------------
# 2. Valores ausentes
# ------------------------------------------------------------
print("\n2. VALORES AUSENTES")
print(dados.isnull().sum())

# ------------------------------------------------------------
# 3. Estatistica descritiva
# ------------------------------------------------------------
print("\n3. ESTATISTICA DESCRITIVA")
print(dados.describe().round(2))

# Coeficiente de variacao
variaveis_cv = [
    "escopo1_tco2e",
    "escopo2_tco2e",
    "escopo3_tco2e",
    "emissoes_total_tco2e",
    "intensidade_calc_tco2e_por_mi",
    "energia_gj"
]

print("\nCoeficiente de variacao (%):")
for coluna in variaveis_cv:
    media = dados[coluna].mean()
    desvio = dados[coluna].std()
    cv = desvio / media * 100
    print(f"{coluna}: {cv:.2f}%")

# ------------------------------------------------------------
# 4. Participacao dos escopos
# ------------------------------------------------------------
for escopo in ["escopo1_tco2e", "escopo2_tco2e", "escopo3_tco2e"]:
    dados[escopo + "_pct"] = dados[escopo] / dados["emissoes_total_tco2e"] * 100

print("\n4. PARTICIPACAO DOS ESCOPOS (%)")
print(
    dados[
        [
            "ano",
            "escopo1_tco2e_pct",
            "escopo2_tco2e_pct",
            "escopo3_tco2e_pct"
        ]
    ].round(2).to_string(index=False)
)

# ------------------------------------------------------------
# 5. Variacao anual
# ------------------------------------------------------------
print("\n5. VARIACAO ANUAL (%)")
colunas_variacao = [
    "escopo1_tco2e",
    "escopo2_tco2e",
    "escopo3_tco2e",
    "emissoes_total_tco2e",
    "energia_gj",
    "intensidade_calc_tco2e_por_mi"
]

variacao = dados[colunas_variacao].pct_change() * 100
variacao.insert(0, "ano", dados["ano"])

print(variacao.round(2).to_string(index=False))

# ------------------------------------------------------------
# 6. Correlacao de Pearson
# ------------------------------------------------------------
print("\n6. CORRELACAO DE PEARSON")
correlacao = dados[
    [
        "escopo1_tco2e",
        "escopo3_tco2e",
        "emissoes_total_tco2e",
        "energia_gj",
        "colaboradores",
        "ocupacao_frota_pct"
    ]
].corr()

print(correlacao.round(2))
# ------------------------------------------------------------
# 7. Deteccao simples de outliers por IQR
# ------------------------------------------------------------
print("\n7. DETECCAO DE OUTLIERS PELO IQR")

for coluna in variaveis_cv:
    q1 = dados[coluna].quantile(0.25)
    q3 = dados[coluna].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    outliers = dados[
        (dados[coluna] < limite_inferior) |
        (dados[coluna] > limite_superior)
    ]

    print(f"\n{coluna}")
    print("Limite inferior:", round(limite_inferior, 2))
    print("Limite superior:", round(limite_superior, 2))

    if outliers.empty:
        print("Nenhum outlier estatistico identificado.")
    else:
        print("Outliers:")
        print(outliers[["ano", coluna]].to_string(index=False))

# ------------------------------------------------------------
# 8. Grafico: emissões por escopo
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))

x = np.arange(len(dados))
largura = 0.25

plt.bar(x - largura, dados["escopo1_tco2e"] / 1000,
        width=largura, label="Escopo 1")
plt.bar(x, dados["escopo2_tco2e"] / 1000,
        width=largura, label="Escopo 2")
plt.bar(x + largura, dados["escopo3_tco2e"] / 1000,
        width=largura, label="Escopo 3")

plt.xticks(x, dados["ano"])
plt.xlabel("Ano")
plt.ylabel("Emissões (mil tCO2e)")
plt.title("Emissões de GEE por escopo - OceanPact")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. Grafico: emissões totais
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    dados["ano"],
    dados["emissoes_total_tco2e"] / 1000,
    marker="o"
)

plt.xlabel("Ano")
plt.ylabel("Emissões totais (mil tCO2e)")
plt.title("Evolucao das emissões totais - OceanPact")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. Grafico: Escopo 1 x ocupacao
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.scatter(
    dados["ocupacao_frota_pct"],
    dados["escopo1_tco2e"] / 1000,
    s=70
)

for _, linha in dados.iterrows():
    plt.annotate(
        str(int(linha["ano"])),
        (
            linha["ocupacao_frota_pct"],
            linha["escopo1_tco2e"] / 1000
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel("Taxa de ocupação da frota (%)")
plt.ylabel("Escopo 1 (mil tCO2e)")
plt.title("Taxa de ocupação da frota x Escopo 1")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 11. Grafico: intensidade
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    dados["ano"],
    dados["intensidade_calc_tco2e_por_mi"],
    marker="o"
)

plt.xlabel("Ano")
plt.ylabel("tCO2e por R$ milhão")
plt.title("Intensidade de emissões recalculada")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 12. Conclusoes automaticas
# ------------------------------------------------------------
crescimento_escopo1 = (
    dados.loc[dados["ano"] == 2025, "escopo1_tco2e"].iloc[0]
    / dados.loc[dados["ano"] == 2023, "escopo1_tco2e"].iloc[0]
    - 1
) * 100

crescimento_total = (
    dados.loc[dados["ano"] == 2025, "emissoes_total_tco2e"].iloc[0]
    / dados.loc[dados["ano"] == 2023, "emissoes_total_tco2e"].iloc[0]
    - 1
) * 100

variacao_intensidade = (
    dados.loc[dados["ano"] == 2025, "intensidade_calc_tco2e_por_mi"].iloc[0]
    / dados.loc[dados["ano"] == 2023, "intensidade_calc_tco2e_por_mi"].iloc[0]
    - 1
) * 100

print("\n" + "=" * 70)
print("CONCLUSOES")
print("=" * 70)

print(
    f"1. O Escopo 1 variou {crescimento_escopo1:.1f}% entre 2023 e 2025."
)

print(
    f"2. As emissões totais variaram {crescimento_total:.1f}% entre 2023 e 2025."
)

print(
    f"3. A intensidade recalculada variou {variacao_intensidade:.1f}% "
    "entre 2023 e 2025."
)

print(
    "4. O Escopo 3 concentra a maior parte das emissões totais, "
    "representando aproximadamente 71% a 81% no período."
)

print(
    "5. O Escopo 1 apresenta forte relação com o consumo de energia, "
    "mas a série possui apenas três observações."
)

print(
    "6. As correlações devem ser interpretadas apenas como indicativas, "
    "pois n = 3 é insuficiente para conclusões estatísticas robustas."
)

print(
    "7. O salto do Escopo 3 em 2024 está associado a uma mudança metodológica "
    "pela inclusão de ativos arrendados, segundo o documento."
)
