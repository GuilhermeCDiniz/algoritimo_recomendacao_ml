# 🚀 Algoritmo de Recomendação de Produtos Fast Food com Python e Machine Learning

Projeto desenvolvido para aprimorar meus estudos e aplicar conceitos de análise de dados, machine learning e recomendação personalizada.

## 💡 Sobre o Projeto

A ideia surgiu de uma necessidade real: ajudar estabelecimentos de fast food, times de marketing e vendas a identificar produtos que um cliente provavelmente compraria, com base nas compras de outros clientes semelhantes.

> 🔎 Em vez de fazer isso no achismo, essa é uma solução baseada em dados reais de vendas, com potencial para aumentar o ticket médio e melhorar a experiência de compra.

---

## ⚙️ Como Funciona

Utilizei o algoritmo **KNN (K-Nearest Neighbors)** com a métrica de **Jaccard**, que compara os produtos comprados por diferentes clientes.

> “Se clientes com perfis de compra parecidos compraram determinado produto, talvez esse cliente também vá querer.”

### O sistema retorna:
- ✅ Os clientes mais semelhantes
- ✅ Uma lista de produtos recomendados que esse cliente ainda não comprou
- ✅ Um gráfico com os produtos mais indicados, de forma visual

---

## 🧩 Dados Utilizados

A base de dados contém:
- +200 clientes
- +50 produtos diferentes

Campos utilizados:
- `Cliente_ID`
- `Produto_ID`
- `Quantidade`
- `Valor_Total`
- `Nome_Produto`

---

## 🛠️ Etapas da Solução

### 📦 1. Carregar a Base de Dados

```python
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
```

---

### 🧹 2. Pré-processamento dos Dados

```python
def preprocessar_dados(df):
    df["Data_Compra"] = pd.to_datetime(df["Data_Compra"], errors='coerce')
    df = df.dropna(subset=["Cliente_ID", "Produto_ID", "Quantidade", "Valor_Total"])
    return df
```

---

### 📊 3. Matriz Cliente x Produto

```python
def criar_matriz_cliente_produto(df):
    return df.pivot_table(index="Cliente_ID", columns="Produto_ID", values="Quantidade", fill_value=0)
```

---

### 🧠 4. Treinar o Modelo KNN

```python
@st.cache_resource
def treinar_knn(pivot):
    pivot_binario = (pivot > 0).astype(int)
    knn = NearestNeighbors(n_neighbors=3, metric="jaccard", algorithm="brute")
    knn.fit(pivot_binario)
    return knn, pivot_binario
```

---

### 💡 5. Recomendação de Produtos

```python
def recomendar_produtos(cliente_id, knn, pivot_binario, df):
    cliente_id = int(cliente_id)
    if cliente_id not in pivot_binario.index:
        return None, [], []

    distances, indices = knn.kneighbors([pivot_binario.loc[cliente_id]], n_neighbors=3)

    cliente_analisado = df.loc[df["Cliente_ID"] == cliente_id, ["Cliente_ID", "Nome_Cliente"]].drop_duplicates().values
    cliente_analisado = (cliente_analisado[0][0], cliente_analisado[0][1]) if len(cliente_analisado) > 0 else (cliente_id, "Nome Desconhecido")

    lista_clientes_similares = pivot_binario.index[indices[0][1:]].tolist()
    clientes_similares = df[df["Cliente_ID"].isin(lista_clientes_similares)][["Cliente_ID", "Nome_Cliente"]].drop_duplicates().values.tolist()

    produtos_recomendados = pivot_binario.iloc[indices[0][1:]].sum(axis=0)
    produtos_cliente = pivot_binario.loc[cliente_id]
    produtos_recomendados = produtos_recomendados[produtos_cliente == 0].sort_values(ascending=False).head(5)

    lista_produtos = []
    for produto_id in produtos_recomendados.index:
        nome_produto = df.loc[df["Produto_ID"] == produto_id, "Nome_Produto"].drop_duplicates()
        if not nome_produto.empty:
            lista_produtos.append((produto_id, nome_produto.values[0]))

    return cliente_analisado, clientes_similares, lista_produtos
```

---

### 🧾 6. Exibir Informações do Cliente

```python
freq = compras_cliente.shape[0]
total_gasto = compras_cliente["Valor_Total"].sum()
st.markdown(f"**🕒 Frequência de Compras:** {freq} vezes")
st.markdown(f"**💰 Valor Total Gasto:** R$ {total_gasto:.2f}")
```

---

### 📋 7. Tabela e Gráfico de Compras

```python
compras_formatadas = compras_cliente[["Nome_Produto", "Categoria_Produto", "Valor_Total"]]
grafico_dados = compras_cliente.groupby("Nome_Produto")["Valor_Total"].sum().sort_values(ascending=True)
```

---

### 📦 8. Mostrar Recomendações

```python
for produto in produtos_recomendados:
    st.write(f"- {produto[0]} - {produto[1]}")
```

---

## ✅ Conclusão

Este projeto mostra como dados do dia a dia podem se transformar em **valor real para o negócio**, com uma aplicação simples de **Machine Learning** usando **Python**.

---
