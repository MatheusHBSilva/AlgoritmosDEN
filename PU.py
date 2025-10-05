import math
import cmath

#Todos os cálculos baseados no doc enviado e no doc step by step recebido.

# ----------- Funções principais ------------

def calcular_umidade():
    print("\n=== Cálculo de Umidade Gravimétrica ===")
    Msolo = float(input("Digite a massa de solo seco (g): "))
    w = float(input("Digite a umidade gravimétrica alvo (%) : ")) / 100.0
    
    Magua = w * Msolo
    Mumido = Msolo + Magua
    
    print(f"\nPara {Msolo:.2f} g de solo seco com {w*100:.1f}% de umidade gravimétrica:")
    print(f" - Deve-se adicionar {Magua:.2f} g (~{Magua:.2f} mL) de água")
    print(f" - A massa úmida será {Mumido:.2f} g\n")

def mostrar_procedimento():
    print("\n=== Procedimento de Preparação ===")
    passos = [
        "1. Coloque a massa de solo seco no recipiente adequado.",
        "2. Meça o volume de água necessário com proveta ou pipeta.",
        "3. Adicione a água lentamente ao solo, misturando bem.",
        "4. Misture até que o solo esteja homogêneo (sem grumos).",
        "5. Transfira para recipiente fechado e deixe em repouso por 12-24h."
    ]
    for p in passos:
        print(p)
    print()


# ----------- Cálculo da Permissividade ------------

def db_to_linear(magnitude_db):
    """Converte dB para magnitude linear"""
    return 10 ** (magnitude_db / 20.0)

def calcular_permissividade():
    print("\n=== Cálculo da Permissividade (εr) ===")
    # Entradas do usuário
    S11_db = float(input("Digite o valor de S11 (dB): "))
    S21_db = float(input("Digite o valor de S21 (dB): "))
    f_GHz = float(input("Digite a frequência (GHz): "))
    d = float(input("Digite a espessura da amostra (m): "))
    
    # Constantes
    c = 3e8  # velocidade da luz no vácuo (m/s)
    f = f_GHz * 1e9
    k0 = 2 * math.pi * f / c  # número de onda no espaço livre
    
    # Converter S11 e S21 para forma linear
    S11 = db_to_linear(S11_db)
    S21 = db_to_linear(S21_db)
    
    # --- Fórmulas simplificadas (baseadas no método de Nicolson-Ross-Weir) ---
    # Cálculo de coeficientes de reflexão e transmissão
    V1 = (S11 ** 2 - S21 ** 2 + 1) / (2 * S11)
    Gama = V1 - cmath.sqrt(V1**2 - 1)  # raiz fisicamente passiva escolhida
    
    T = S21 / (1 - S11 * Gama)
    
    # Constante de propagação (complexa)
    gamma = -1/d * cmath.log(1/T)
    
    # Cálculo da permissividade relativa
    epsilon_r = (gamma / (1j * k0))**2
    
    print("\n--- Resultados ---")
    print(f"Frequência: {f_GHz} GHz")
    print(f"S11 (lin): {S11:.4f}")
    print(f"S21 (lin): {S21:.4f}")
    print(f"Coef. reflexão Γ: {Gama:.4f}")
    print(f"Coef. transmissão T: {T:.4f}")
    print(f"Permissividade relativa (εr): {epsilon_r.real:.3f} + j{epsilon_r.imag:.3f}\n")

# ----------- Menu Principal ------------

def main():
    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1. Calcular massa de água e solo úmido")
        print("2. Mostrar procedimento de preparação")
        print("3. Calcular permissividade (dados do usuário)")
        print("0. Sair")
        
        opcao = input("Escolha uma opção: ")
        
        if opcao == "1":
            calcular_umidade()
        elif opcao == "2":
            mostrar_procedimento()
        elif opcao == "3":
            calcular_permissividade()
        elif opcao == "0":
            print("Encerrando o programa...")
            break
        else:
            print("Opção inválida, tente novamente.")

if __name__ == "__main__":
    main()
