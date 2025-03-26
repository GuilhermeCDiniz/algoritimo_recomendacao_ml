import os
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# ========= CONFIG LAYOUT =========
st.set_page_config(page_title="🍔 Recomendador Fast Food", layout="wide")
st.markdown("""
    <style>
    .title-style {
        font-size: 36px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .section-header {
        font-size: 24px;
        margin-top: 1.5em;
        color: #ff914d;
    }
    .small-icon {
        width: 24px;
        margin-right: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ========= CARREGAR DADOS =========
@st.cache_data
def carregar_dados(arquivo: str):
    if not os.path.exists(arquivo):
        st.error(f"❌ O arquivo {arquivo} não foi encontrado.")
        return None
    try:
        df = pd.read_excel(arquivo, sheet_name="sheet")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

# ========= PRÉ-PROCESSAMENTO =========
def preprocessar_dados(df):
    df["Data_Compra"] = pd.to_datetime(df["Data_Compra"], errors='coerce')
    df = df.dropna(subset=["Cliente_ID", "Produto_ID", "Quantidade", "Valor_Total"])
    return df

# ========= MATRIZ CLIENTE x PRODUTO =========
def criar_matriz_cliente_produto(df):
    return df.pivot_table(index="Cliente_ID", columns="Produto_ID", values="Quantidade", fill_value=0)

# ========= MODELO KNN =========
@st.cache_resource
def treinar_knn(pivot):
    pivot_binario = (pivot > 0).astype(int)
    knn = NearestNeighbors(n_neighbors=3, metric="jaccard", algorithm="brute")
    knn.fit(pivot_binario)
    return knn, pivot_binario

# ========= RECOMENDAR =========
def recomendar_produtos(cliente_id, knn, pivot_binario, df):
    cliente_id = int(cliente_id)
    if cliente_id not in pivot_binario.index:
        return None, [], []

    with st.spinner("⏳ Treinando Modelo..."):
        time.sleep(1.5)
    distances, indices = knn.kneighbors([pivot_binario.loc[cliente_id]], n_neighbors=3)

    with st.spinner("🔍 Encontrando clientes semelhantes..."):
        time.sleep(1.5)

    cliente_analisado = df.loc[df["Cliente_ID"] == cliente_id, ["Cliente_ID", "Nome_Cliente"]].drop_duplicates().values
    cliente_analisado = (cliente_analisado[0][0], cliente_analisado[0][1]) if len(cliente_analisado) > 0 else (cliente_id, "Nome Desconhecido")

    lista_clientes_similares = pivot_binario.index[indices[0][1:]].tolist()
    clientes_similares = df[df["Cliente_ID"].isin(lista_clientes_similares)][["Cliente_ID", "Nome_Cliente"]].drop_duplicates().values.tolist()

    with st.spinner("📦 Recomendando produtos..."):
        time.sleep(1.5)

    produtos_recomendados = pivot_binario.iloc[indices[0][1:]].sum(axis=0)
    produtos_cliente = pivot_binario.loc[cliente_id]
    produtos_recomendados = produtos_recomendados[produtos_cliente == 0].sort_values(ascending=False).head(5)

    lista_produtos = []
    for produto_id in produtos_recomendados.index:
        nome_produto = df.loc[df["Produto_ID"] == produto_id, "Nome_Produto"].drop_duplicates()
        if not nome_produto.empty:
            lista_produtos.append((produto_id, nome_produto.values[0]))

    return cliente_analisado, clientes_similares, lista_produtos

# ========= GRÁFICO =========
def gerar_grafico(produtos_recomendados):
    if not produtos_recomendados:
        st.warning("⚠️ Nenhuma recomendação disponível.")
        return
    codigos, nomes = zip(*produtos_recomendados)
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.barh(nomes, range(len(nomes), 0, -1), color="#ff914d")
    ax.set_xlabel("Ranking de Recomendação")
    ax.set_title("Top 5 Produtos Recomendados")
    ax.invert_yaxis()
    st.pyplot(fig)

# ========= INTERFACE =========
st.markdown('<h1 class="title-style">🍟 Recomendador de Fast Food</h1>', unsafe_allow_html=True)

arquivo_dados = "base_vendas_fastfood_knn.xlsx"
df = carregar_dados(arquivo_dados)

if df is not None:
    df = preprocessar_dados(df)
    pivot = criar_matriz_cliente_produto(df)

    with st.spinner("⏳ Treinando Modelo..."):
        knn, pivot_binario = treinar_knn(pivot)

    st.markdown("<div class='section-header'>🔎 Buscar Cliente</div>", unsafe_allow_html=True)
    cliente_id = st.text_input("Digite o código do cliente:", "")

    if st.button("🍔 Analisar Cliente") and cliente_id.strip():
        if not cliente_id.isdigit():
            st.error("❌ Código do cliente inválido! Digite apenas números.")
        else:
            cliente_analisado, clientes_similares, produtos_recomendados = recomendar_produtos(
                int(cliente_id), knn, pivot_binario, df)

            if cliente_analisado:
                st.markdown("<div class='section-header'>👤 Cliente Analisado</div>", unsafe_allow_html=True)
                st.write(f"**{cliente_analisado[0]} - {cliente_analisado[1]}**")

                compras_cliente = df[df["Cliente_ID"] == cliente_analisado[0]]
                if not compras_cliente.empty:
                    freq = compras_cliente.shape[0]
                    total_gasto = compras_cliente["Valor_Total"].sum()
                    st.markdown(f"**🕒 Frequência de Compras:** {freq} vezes")
                    st.markdown(f"**💰 Valor Total Gasto:** R$ {total_gasto:.2f}")

                    st.markdown("**🧾 Produtos comprados:**")
                    compras_formatadas = compras_cliente[["Nome_Produto", "Categoria_Produto", "Valor_Total"]]
                    st.dataframe(compras_formatadas.sort_values(by="Valor_Total", ascending=False))

                    st.markdown("**📊 Gráfico de gastos por produto:**")
                    grafico_dados = compras_cliente.groupby("Nome_Produto")["Valor_Total"].sum().sort_values(ascending=True)

                    fig, ax = plt.subplots(figsize=(7, 2))
                    ax.barh(grafico_dados.index, grafico_dados.values, color="green")
                    ax.set_xlabel("Valor Total (R$)")
                    ax.set_title("Gasto total por produto")
                    st.pyplot(fig)
                else:
                    st.info("Este cliente ainda não possui compras registradas.")

            if clientes_similares:
                st.markdown("<div class='section-header'>👥 Clientes Semelhantes</div>", unsafe_allow_html=True)
                for cliente in clientes_similares:
                    st.write(f"- {cliente[0]} - {cliente[1]}")

            if produtos_recomendados:
                st.markdown("<div class='section-header'>🍔 Produtos Recomendados</div>", unsafe_allow_html=True)
                for produto in produtos_recomendados:
                    st.write(f"- {produto[0]} - {produto[1]}")
                gerar_grafico(produtos_recomendados)
