import numpy as np
import pandas as pd
from pyDOE3 import ccdesign, bbdesign
from itertools import combinations_with_replacement
from sklearn.linear_model import LinearRegression

# Antes de iniciar o código, por favor coloque esse comando no cmd para importar as bibliotecas!!
# pip install pyDOE3 numpy pandas scikit-learn

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
    print("\nForneça os valores de resposta para cada experimento prático:")
    respostas = []
    for idx, row in df.iterrows():
        y = float(input(f"Experimento {idx+1} ({row.to_dict()}): "))
        respostas.append(y)
    Y = np.array(respostas)
    X = df.values
    X_quad = gerar_termos_quadraticos(X)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_quad, Y)
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
    for i, y in enumerate(y_pred):
        print(f"Ponto {i+1} {dict(zip(fatores, X_novo[i]))} -> Resposta prevista: {y:.4f}")

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

            prever = input("\nDeseja prever a resposta para múltiplos pontos? (s/n): ").lower()
            if prever == "s":
                prever_resposta_multiplos(model)

        continuar = input("\nDeseja rodar novamente? (s/n): ").lower()
        if continuar == "n":
            break

if __name__ == "__main__":
    main()
