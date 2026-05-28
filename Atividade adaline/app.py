import tkinter as tk
from tkinter import messagebox
import string

# ==========================================================
# 1. MOTOR ADALINE COM HISTÓRICO DE ERRO (VETOR FIXADO EM 256)
# ==========================================================
class MultiClassAdaline:
    def __init__(self, input_size=256, num_classes=26, lr=0.01, epochs=150):
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.input_size = input_size
        # Matriz de pesos: +1 para acomodar o Bias ativo fixo
        self.weights = [[0.0] * num_classes for _ in range(input_size + 1)]
        self.historico_erro = [] 
        
    def train(self, X, y_labels):
        num_samples = len(X)
        self.historico_erro = []
        
        for epoch in range(self.epochs):
            erro_acumulado_epoca = 0.0
            
            for i in range(num_samples):
                x = [1.0] + X[i] # Injeção de Bias estrito
                actual_label = y_labels[i]
                
                for c in range(self.num_classes):
                    target = 1.0 if actual_label == c else -1.0
                    
                    net_input = 0.0
                    for j in range(len(x)):
                        net_input += x[j] * self.weights[j][c]
                        
                    error = target - net_input
                    erro_acumulado_epoca += error ** 2
                    
                    for j in range(len(x)):
                        self.weights[j][c] += self.lr * error * x[j] / num_samples
            
            erro_medio = erro_acumulado_epoca / (num_samples * self.num_classes)
            self.historico_erro.append(erro_medio)

    def predict(self, x_features):
        x = [1.0] + x_features
        max_output = float('-inf')
        predicted_class = 0
        
        for c in range(self.num_classes):
            net_input = 0.0
            for j in range(len(x)):
                net_input += x[j] * self.weights[j][c]
            if net_input > max_output:
                max_output = net_input
                predicted_class = c
        return predicted_class

# ==========================================================
# 2. DATASET MATRICIAL ALINHADO (Redimensionamento Estrito 16x16)
# ==========================================================
GRID_SIZE = 16

MAPAS_LETRAS_BASE = {
    0:  [[0,1,1,0],[1,0,0,1],[1,1,1,1],[1,0,0,1]], # A
    1:  [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,1]], # B
    2:  [[0,1,1,1],[1,0,0,0],[1,0,0,0],[0,1,1,1]], # C
    3:  [[1,1,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,0]], # D
    4:  [[1,1,1,1],[1,1,0,0],[1,0,0,0],[1,1,1,1]], # E
    5:  [[1,1,1,1],[1,1,0,0],[1,0,0,0],[1,0,0,0]], # F
    6:  [[0,1,1,1],[1,0,0,0],[1,0,1,1],[0,1,1,1]], # G
    7:  [[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1]], # H
    8:  [[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0]], # I
    9:  [[0,0,1,1],[0,0,0,1],[1,0,0,1],[0,1,1,0]], # J
    10: [[1,0,0,1],[1,0,1,0],[1,1,0,0],[1,0,0,1]], # K
    11: [[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,1,1,1]], # L
    12: [[1,0,0,1],[1,1,1,1],[1,0,0,1],[1,0,0,1]], # M
    13: [[1,0,0,1],[1,1,0,1],[1,0,1,1],[1,0,0,1]], # N
    14: [[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]], # O
    15: [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,0]], # P
    16: [[0,1,1,0],[1,0,0,1],[1,0,1,1],[0,1,1,1]], # Q
    17: [[1,1,1,0],[1,0,0,1],[1,1,1,0],[1,0,0,1]], # R
    18: [[0,1,1,1],[1,1,0,0],[0,0,1,1],[1,1,1,0]], # S
    19: [[1,1,1,1],[0,1,1,0],[0,1,1,0],[0,1,1,0]], # T
    20: [[1,0,0,1],[1,0,0,1],[1,0,0,1],[0,1,1,0]], # U
    21: [[1,0,0,1],[1,0,0,1],[0,1,1,0],[0,1,1,0]], # V
    22: [[1,0,0,1],[1,0,0,1],[1,1,1,1],[1,0,0,1]], # W
    23: [[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]], # X
    24: [[1,0,0,1],[0,1,1,0],[0,0,1,0],[0,0,1,0]], # Y
    25: [[1,1,1,1],[0,0,1,0],[0,1,0,0],[1,1,1,1]], # Z
}

def interpolar_matriz_16x16(padrao_4x4):
    """Interpola de forma limpa a matriz 4x4 conceitual para o vetor de entrada de 256 bits."""
    resultado = []
    for linha in padrao_4x4:
        for _ in range(4):
            sub_linha = []
            for bit in linha:
                sub_linha.extend([1.0 if bit == 1 else -1.0] * 4)
            resultado.extend(sub_linha)
    return resultado

# ==========================================================
# 3. DASHBOARD DE VALIDAÇÃO (DIRETO EM TKINTER NATIVO)
# ==========================================================
class AppSeletorComGraficoNativo:
    def __init__(self, root, model):
        self.model = model
        self.root = root
        self.root.title("Adaline AI Dashboard")
        self.root.configure(bg="#fdf2f8") 
        self.root.geometry("760x460")      
        self.root.resizable(False, False)
        
        self.alphabet = list(string.ascii_uppercase)
        self.canvas_dim = 180
        self.escala = self.canvas_dim / GRID_SIZE

        # Header Superior Estilizado em Rosa Magenta
        header = tk.Frame(root, bg="#db2777", height=50)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header, text="MÓDULO DE VALIDAÇÃO E CURVA DE ERRO NATIVA", font=("Segoe UI", 11, "bold"), fg="white", bg="#db2777").pack(pady=12)

        body_frame = tk.Frame(root, bg="#fdf2f8")
        body_frame.pack(fill="both", expand=True, padx=20, pady=15)

        # --- SEÇÃO ESQUERDA: Seleção e Visor ---
        frame_esquerda = tk.Frame(body_frame, bg="#fdf2f8")
        frame_esquerda.pack(side="left", fill="both", expand=False, padx=10)

        tk.Label(frame_esquerda, text="Escolha a Letra:", font=("Segoe UI", 10, "bold"), fg="#9d174d", bg="#fdf2f8").pack(anchor="w", pady=2)
        
        self.valor_selecionado = tk.StringVar()
        self.valor_selecionado.set("A")
        menu_letras = tk.OptionMenu(frame_esquerda, self.valor_selecionado, *self.alphabet, command=self.atualizar_letra_visual)
        menu_letras.config(bg="#fbcfe8", fg="#9d174d", font=("Segoe UI", 9, "bold"), bd=1, width=6)
        menu_letras.pack(pady=4, anchor="w")

        self.btn_testar = tk.Button(frame_esquerda, text="EXECUTAR ADALINE", font=("Segoe UI", 9, "bold"), bg="#ec4899", fg="white", bd=0, height=2, command=self.testar_letra, cursor="hand2", activebackground="#be185d")
        self.btn_testar.pack(fill="x", pady=8)

        frame_canvas = tk.Frame(frame_esquerda, bg="white", bd=0, highlightbackground="#fbcfe8", highlightthickness=2)
        frame_canvas.pack(pady=2, anchor="center")
        
        self.canvas_letra = tk.Canvas(frame_canvas, width=self.canvas_dim, height=self.canvas_dim, bg="white", highlightthickness=0)
        self.canvas_letra.pack()

        self.lbl_resultado = tk.Label(frame_esquerda, text="Selecione e teste a rede", font=("Segoe UI", 11, "bold", "italic"), fg="#9d174d", bg="#fdf2f8")
        self.lbl_resultado.pack(pady=10)

        # --- SEÇÃO DIREITA: Gráfico Nativo em Tkinter Canvas ---
        frame_direita = tk.Frame(body_frame, bg="#fdf2f8")
        frame_direita.pack(side="right", fill="both", expand=True, padx=10)
        
        tk.Label(frame_direita, text="Evolução do Erro de Treinamento (MSE)", font=("Segoe UI", 10, "bold"), fg="#9d174d", bg="#fdf2f8").pack(anchor="w", pady=2)

        # Canvas Técnico para Desenho de Eixos e Curvas
        self.canvas_grafico = tk.Canvas(frame_direita, width=420, height=260, bg="white", highlightbackground="#fbcfe8", highlightthickness=1)
        self.canvas_grafico.pack(fill="both", expand=True, pady=5)
        
        self.renderizar_grafico_nativo()
        self.atualizar_letra_visual("A")

    def obter_vetor_letra(self, letra):
        idx = self.alphabet.index(letra)
        padrao_4x4 = MAPAS_LETRAS_BASE[idx]
        return interpolar_matriz_16x16(padrao_4x4)

    def atualizar_letra_visual(self, letra):
        self.canvas_letra.delete("all")
        vetor = self.obter_vetor_letra(letra)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                idx_vetor = y * GRID_SIZE + x
                if vetor[idx_vetor] == 1.0:
                    x1 = x * self.escala
                    y1 = y * self.escala
                    self.canvas_letra.create_rectangle(x1, y1, x1 + self.escala, y1 + self.escala, fill="#1e1b4b", outline="")

    def testar_letra(self):
        letra_escolhida = self.valor_selecionado.get()
        vetor_recursos = self.obter_vetor_letra(letra_escolhida)
        
        idx_predito = self.model.predict(vetor_recursos)
        letra_ia = self.alphabet[idx_predito]
        
        if letra_escolhida == letra_ia:
            self.lbl_resultado.config(text=f"Reconhecido como: {letra_ia} ✔", fg="#db2777")
        else:
            self.lbl_resultado.config(text=f"Confundiu com: {letra_ia} ❌", fg="#e11d48")

    def renderizar_grafico_nativo(self):
        """Desenha a estrutura cartesiana inteira e plota a curva usando primitivas do Tkinter."""
        self.canvas_grafico.update()
        w = 420
        h = 240
        
        # Margens internas para os eixos
        m_left, m_right, m_top, m_bottom = 45, 20, 20, 35
        plot_w = w - m_left - m_right
        plot_h = h - m_top - m_bottom
        
        # Desenha os eixos cartesianos secundários
        self.canvas_grafico.create_line(m_left, h - m_bottom, w - m_right, h - m_bottom, fill="#888888", width=1.5) # Eixo X
        self.canvas_grafico.create_line(m_left, m_top, m_left, h - m_bottom, fill="#888888", width=1.5) # Eixo Y
        
        historico = self.model.historico_erro
        total_pontos = len(historico)
        
        if total_pontos < 2: return
        
        max_err = max(historico) if max(historico) > 0 else 1.0
        min_err = min(historico)
        
        # Adiciona marcações dinâmicas de texto no Eixo Y (Valores de Erro)
        self.canvas_grafico.create_text(m_left - 8, m_top, text=f"{max_err:.2f}", anchor="e", font=("Segoe UI", 8), fill="#555555")
        self.canvas_grafico.create_text(m_left - 8, h - m_bottom, text=f"{min_err:.2f}", anchor="e", font=("Segoe UI", 8), fill="#555555")
        
        # Adiciona marcações no Eixo X (Épocas)
        self.canvas_grafico.create_text(m_left, h - m_bottom + 12, text="1", anchor="n", font=("Segoe UI", 8), fill="#555555")
        self.canvas_grafico.create_text(w - m_right, h - m_bottom + 12, text=f"{total_pontos}", anchor="n", font=("Segoe UI", 8), fill="#555555")
        
        # Plotagem dos segmentos de reta calculados ponto a ponto
        pontos_coordenadas = []
        for i, valor in enumerate(historico):
            # Normalização e mapeamento linear de coordenadas para pixels na tela
            cx = m_left + (i / (total_pontos - 1)) * plot_w
            cy = (h - m_bottom) - ((valor - min_err) / (max_err - min_err)) * plot_h
            pontos_coordenadas.append((cx, cy))
            
        for i in range(len(pontos_coordenadas) - 1):
            x_atual, y_atual = pontos_coordenadas[i]
            x_prox, y_prox = pontos_coordenadas[i+1]
            self.canvas_grafico.create_line(x_atual, y_atual, x_prox, y_prox, fill="#db2777", width=2.5, smooth=True)

if __name__ == "__main__":
    X_train, y_train = [], []
    for idx, mapa_base in MAPAS_LETRAS_BASE.items():
        # Transforma o bloco original usando a nova função unificada de 256 posições
        X_train.append(interpolar_matriz_16x16(mapa_base))
        y_train.append(idx)
        
    # Inicializa e treina gerando a redução do gradiente mapeada
    modelo_adaline = MultiClassAdaline(input_size=256)
    modelo_adaline.train(X_train, y_train)
    
    janela = tk.Tk()
    app = AppSeletorComGraficoNativo(janela, modelo_adaline)
    janela.mainloop()