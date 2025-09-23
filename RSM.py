import numpy as np
import pandas as pd
from pyDOE3 import ccdesign, bbdesign
from itertools import combinations_with_replacement
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Antes de iniciar o código, por favor coloque esse comando no cmd para importar as bibliotecas!!
# pip install pyDOE3 numpy pandas scikit-learn matplotlib openpyxl

# Definir fatores
fatores = ["Umidade", "Peso da Amostra", "pH"]

# Intervalos = valores que iremos usar para os melhores exprimentos
intervalos = {}

# Função de input para receber os valores do usuário
def get_levels(fator):
    print(f"\nConfiguração do fator: {fator}")
    escolha = input("Deseja digitar os níveis manualmente? (s/n): ").lower()
    if escolha == "s":
        valores_str = input(f"Digite os níveis de {fator} separados por vírgula: ")
        valores = sorted([float(v.strip()) for v in valores_str.split(",")])
        minimo, maximo = min(valores), max(valores)
        return minimo, maximo, np.array(valores)
    else:
        minimo = float(input(f"Digite o valor MÍNIMO para {fator}: "))
        maximo = float(input(f"Digite o valor MÁXIMO para {fator}: "))
        qtd = int(input(f"Digite a QUANTIDADE de níveis para {fator}: "))
        valores = np.linspace(minimo, maximo, qtd)
        return minimo, maximo, valores

# Define se queremos usar por CCD ou Box-Behnken, uma abordagem mais clássica e outra mais recente
def escolher_planejamento():
    print("\nEscolha o tipo de planejamento experimental:")
    print("1 - CCD (Central Composite Design) Padrão")
    print("2 - Box-Behnken Design")
    opcao = int(input("Digite 1 ou 2: "))
    if opcao == 1:
        return "CCD"
    elif opcao == 2:
        return "Box-Behnken"
    else:
        print("Nenhuma opção foi escolhida, fazendo por CCD.")
        return "CCD"

# Recebe os inputs do usuário e utiiza a biblioteca PyDOE3 para fazer os métodos e retornar ao usuário
def gerar_planejamento(metodo, intervalos):
    k = len(fatores)
    if metodo == "CCD":
        design = ccdesign(k, center=(4, 4))  
    else:
        design = bbdesign(k, center=1)
    df_real = pd.DataFrame()
    for i, fator in enumerate(fatores):
        minimo, maximo, _ = intervalos[fator]
        media = (maximo + minimo) / 2
        amplitude = (maximo - minimo) / 2
        df_real[fator] = design[:, i] * amplitude + media
    return df_real

def gerar_termos_quadraticos(X):
    n, k = X.shape
    X_quad = np.ones((n,1))  # intercepto
    X_quad = np.hstack([X_quad, X])
    X_quad = np.hstack([X_quad, X**2])
    for i, j in combinations_with_replacement(range(k), 2):
        if i < j:
            X_quad = np.hstack([X_quad, (X[:,i]*X[:,j]).reshape(-1,1)])
    return X_quad

# Quando o usuário realiza os testes e coloca um input, ele reajusta o modelo para obter respostas mais precisas.
def ajustar_modelo(df):
    print("\nVocê pode fornecer resultados experimentais ou simular resultados.")
    simular = input("Deseja simular os resultados? (s/n): ").lower()

    respostas = []
    if simular == "s":
        print("\nEscolha o método de simulação:")
        print("1 - Quadrática sem ruído")
        print("2 - Quadrática + ruído gaussiano")
        print("3 - Quadrática + ruído proporcional")
        print("4 - Função não-linear (senoidal)")
        metodo = int(input("Digite o número do método desejado: "))

        X = df.values

        if metodo == 1:
            print("\n>>> Simulação: Superfície quadrática sem ruído <<<")
            Y = (
                2*X[:,0] + 0.5*X[:,1] + 1.2*X[:,2]
                - 0.05*X[:,0]**2 - 0.02*X[:,1]**2
                + 0.03*X[:,0]*X[:,2]
            )

        elif metodo == 2:
            print("\n>>> Simulação: Quadrática + ruído gaussiano (σ=1.0) <<<")
            Y = (
                2*X[:,0] + 0.5*X[:,1] + 1.2*X[:,2]
                - 0.05*X[:,0]**2 - 0.02*X[:,1]**2
                + 0.03*X[:,0]*X[:,2]
                + np.random.normal(0, 1.0, size=len(X))
            )

        elif metodo == 3:
            print("\n>>> Simulação: Quadrática + ruído proporcional (10%) <<<")
            base = (
                2*X[:,0] + 0.5*X[:,1] + 1.2*X[:,2]
                - 0.05*X[:,0]**2 - 0.02*X[:,1]**2
                + 0.03*X[:,0]*X[:,2]
            )
            Y = base + np.random.normal(0, 0.1*np.abs(base), size=len(X))

        elif metodo == 4:
            print("\n>>> Simulação: Função não-linear (senoidal) <<<")
            Y = (
                np.sin(0.1*X[:,0]) * 10
                + np.cos(0.2*X[:,1]) * 5
                + (X[:,2]**0.5)*2
                + np.random.normal(0, 0.5, size=len(X))
            )

        else:
            print("Opção inválida, usando quadrática sem ruído por padrão.")
            Y = (
                2*X[:,0] + 0.5*X[:,1] + 1.2*X[:,2]
                - 0.05*X[:,0]**2 - 0.02*X[:,1]**2
                + 0.03*X[:,0]*X[:,2]
            )

        respostas = Y.tolist()

    else:
        print("\nForneça os valores de resposta para cada experimento prático:")
        for idx, row in df.iterrows():
            y = float(input(f"Experimento {idx+1} ({row.to_dict()}): "))
            respostas.append(y)
        Y = np.array(respostas)

    X = df.values
    X_quad = gerar_termos_quadraticos(X)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_quad, Y)

    # Previsão em todos os pontos do planejamento
    Y_pred = model.predict(X_quad)
    df_resultados = df.copy()
    df_resultados["Resposta experimental"] = Y
    df_resultados["Resposta prevista"] = Y_pred

    # Perguntar critério de escolha
    criterio = input("\nDeseja salvar os experimentos com MAIOR ou MENOR resposta prevista? (maior/menor): ").lower()
    n_melhores = int(input("Quantos experimentos deseja salvar no Excel? "))

    if criterio == "maior":
        df_melhores = df_resultados.nlargest(n_melhores, "Resposta prevista")
    else:
        df_melhores = df_resultados.nsmallest(n_melhores, "Resposta prevista")

    # Salvar somente os melhores
    df_melhores.to_excel("experimentos_recomendados.xlsx", index=False)
    print(f"Arquivo 'experimentos_recomendados.xlsx' salvo com os {n_melhores} experimentos mais {criterio} resposta prevista!")

    return model

def mostrar_equacao(model):
    coef = model.coef_
    termos = ["1"] + fatores
    for f in fatores:
        termos.append(f"{f}²")
    for i, j in combinations_with_replacement(fatores, 2):
        if i != j:
            termos.append(f"{i}*{j}")
    eq = "y = " + " + ".join([f"({c:.4f})*{t}" for c,t in zip(coef, termos)])
    print("\n=== Equação da Superfície de Resposta Ajustada ===")
    print(eq)


def construir_grafico(model):
    # Opção de gráfico 3D + contorno 2D (fixando um fator)
    print("\nEscolha qual fator deseja FIXAR (para variar os outros dois):")
    for i, f in enumerate(fatores):
        print(f"{i+1} - {f}")
    fixo_idx = int(input("Digite o número do fator fixado: ")) - 1
    fixo_val = float(input(f"Digite o valor fixo para {fatores[fixo_idx]}: "))

    variaveis = [i for i in range(len(fatores)) if i != fixo_idx]
    v1, v2 = variaveis

    # Criar grid para os dois fatores variáveis
    n_grid = 40
    x1 = np.linspace(intervalos[fatores[v1]][0], intervalos[fatores[v1]][1], n_grid)
    x2 = np.linspace(intervalos[fatores[v2]][0], intervalos[fatores[v2]][1], n_grid)
    X1, X2 = np.meshgrid(x1, x2)

    pontos = []
    for i in range(n_grid):
        for j in range(n_grid):
            p = [0]*len(fatores)
            p[fixo_idx] = fixo_val
            p[v1] = X1[i,j]
            p[v2] = X2[i,j]
            pontos.append(p)
    pontos = np.array(pontos)

    X_quad = gerar_termos_quadraticos(pontos)
    Y_pred = model.predict(X_quad).reshape(n_grid, n_grid)

    # Gráfico 3D
    fig = plt.figure(figsize=(12,5))

    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(X1, X2, Y_pred, cmap='viridis', alpha=0.8)
    ax.set_xlabel(fatores[v1])
    ax.set_ylabel(fatores[v2])
    ax.set_zlabel("Resposta prevista")
    ax.set_title(f"Superfície 3D (fixando {fatores[fixo_idx]} = {fixo_val})")

    # Gráfico de contorno 2D
    ax2 = fig.add_subplot(1,2,2)
    cont = ax2.contourf(X1, X2, Y_pred, cmap='viridis')
    plt.colorbar(cont, ax=ax2, label="Resposta prevista")
    ax2.set_xlabel(fatores[v1])
    ax2.set_ylabel(fatores[v2])
    ax2.set_title(f"Mapa de contorno (fixando {fatores[fixo_idx]} = {fixo_val})")

    plt.tight_layout()
    plt.show()

# Com os dados reais em mãos o modelo prevê as respostas ao usuário, usando somatório
def prever_resposta_multiplos(model):
    print("\nAgora você pode prever a resposta para múltiplos pontos.")
    n_pontos = int(input("Quantos pontos deseja prever? "))
    novos_pontos = []
    for i in range(n_pontos):
        ponto = []
        print(f"\nPonto {i+1}:")
        for fator in fatores:
            val = float(input(f"  {fator}: "))
            ponto.append(val)
        novos_pontos.append(ponto)
    X_novo = np.array(novos_pontos)
    X_novo_quad = gerar_termos_quadraticos(X_novo)
    y_pred = model.predict(X_novo_quad)

    # Mostrar no terminal
    resultados = []
    for i, y in enumerate(y_pred):
        resultado = dict(zip(fatores, X_novo[i]))
        resultado["Previsão"] = y
        resultados.append(resultado)
        print(f"Ponto {i+1} {dict(zip(fatores, X_novo[i]))} -> Resposta prevista: {y:.4f}")

    
    # Opção de salvar em Excel
    salvar = input("\nDeseja salvar as previsões em um arquivo Excel? (s/n): ").lower()
    if salvar == "s":
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_excel("previsoes_RSM.xlsx", index=False)
        print("Arquivo 'previsoes_RSM.xlsx' salvo com sucesso!")


def main():
    global intervalos
    print("=== RSM ===")

    while True:
        if intervalos:
            print("\nVocê já definiu valores anteriormente.")
            usar_antigos = input("Deseja manter os mesmos valores? (s/n): ").lower()
            if usar_antigos == "n":
                intervalos.clear()

        if not intervalos:
            for fator in fatores:
                intervalo = get_levels(fator)
                intervalos[fator] = intervalo

        metodo = escolher_planejamento()
        df_real = gerar_planejamento(metodo, intervalos)

        print(f"\n=== RESULTADOS ({metodo}) ===")
        print(f"Número de experimentos sugeridos: {len(df_real)}\n")
        print(df_real.to_string(index=False))

        calc_resposta = input("\nDeseja fornecer os resultados experimentais para calcular a superfície de resposta? (s/n): ").lower()
        if calc_resposta == "s":
            model = ajustar_modelo(df_real)
            mostrar_equacao(model)

            grafico = input("\nDeseja ver o gráfico").lower()
            if(grafico == "s"):
                construir_grafico(model)

            prever = input("\nDeseja prever a resposta para múltiplos pontos? (s/n): ").lower()
            if prever == "s":
                prever_resposta_multiplos(model)

        continuar = input("\nDeseja rodar novamente? (s/n): ").lower()
        if continuar == "n":
            break

if __name__ == "__main__":
    main()
