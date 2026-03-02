# Documentação Técnica: Inicialização do Sistema (`initialization.py`)

O módulo `initialization.py` é responsável por definir o **Problema de Valor Inicial (PVI)** da simulação. Ele aloca a memória para os tensores de campo e estabelece o estado termodinâmico e hidrodinâmico inicial ($t=0$).

A função principal, `initialize_fields(mode_m, amplitude)`, constrói uma interface difusa perturbada harmonicamente para induzir a Instabilidade de Saffman-Taylor em um modo de onda específico.

---

## 1. Alocação de Tensores de Estado (Memória)

Estas variáveis representam o estado macroscópico e microscópico do sistema em cada nó da grade $(y, x)$.

| Variável | Shape (Dimensões) | Símbolo Matemático | Descrição Física | Inicialização Padrão |
| :--- | :--- | :--- | :--- | :--- |
| `f` | $(N_y, N_x, 9)$ | $f_i(\mathbf{x}, t)$ | **Funções de Distribuição.** Densidade de probabilidade de encontrar partículas com velocidade $\mathbf{c}_i$. | Equilíbrio ($f^{eq}$) com $\mathbf{u}=0$. |
| `phi` | $(N_y, N_x)$ | $\phi(\mathbf{x}, t)$ | **Parâmetro de Ordem.** Variável de fase que distingue os fluidos ($\phi \approx 1$ invasor, $\phi \approx -1$ residente). | Perfil de $\tanh$ (ver seção 2). |
| `psi` | $(N_y, N_x)$ | $\psi(\mathbf{x})$ | **Potencial Magnético.** Campo escalar cuja derivada define o campo magnético $\mathbf{H} = -\nabla \psi$. | Gradiente Linear (ver seção 3). |
| `rho` | $(N_y, N_x)$ | $\rho(\mathbf{x})$ | **Densidade do Fluido.** Soma dos momentos de ordem zero ($\sum f_i$). | Uniforme $\rho_0 = 1.0$. |
| `u_x, u_y` | $(N_y, N_x)$ | $\mathbf{u}(\mathbf{x})$ | **Velocidade Macroscópica.** Momento de primeira ordem. | Repouso ($\mathbf{u} = 0.0$). |
| `K_field` | $(N_y, N_x)$ | $K(\mathbf{x})$ | **Permeabilidade Local.** Campo escalar que define a resistência do meio poroso. | Homogêneo $K = K_0$. |

---

## 2. Perturbação da Interface (Instabilidade)

Para estudar a estabilidade linear e o crescimento de "dedos" viscosos, a interface inicial não é perfeitamente plana. Ela recebe uma perturbação senoidal controlada.

### 2.1 Equação da Posição da Interface
A posição $x$ da interface ao longo da altura $y$ é definida por:

$$x_{int}(y) = x_{center} + A \cos\left( \frac{2\pi m}{L_y} y \right)$$

| Termo Código | Símbolo Teórico | Significado Físico |
| :--- | :--- | :--- |
| `x_center` | $x_0$ | Posição média da interface inicial (Buffer para evitar efeitos de entrada). Valor fixo: `80.0`. |
| `amplitude` | $A$ ou $\epsilon$ | Magnitude da perturbação inicial. Se $A=0$, a interface é plana. |
| `mode_m` | $m$ | **Número de Modo.** Define a frequência espacial da perturbação ($k = \frac{2\pi m}{L_y}$). Determina quantos "dedos" iniciais são formados. |

### 2.2 Perfil de Fase (Solução de Equilíbrio)
Diferente de métodos de rastreamento de fronteira (VOF), o Campo de Fase possui uma interface de espessura finita. A inicialização impõe a solução analítica de equilíbrio da equação de Cahn-Hilliard através da interface perturbada:

$$\phi(x, y) = -\tanh\left( \frac{x - x_{int}(y)}{\xi / 2} \right)$$

* **No Código:** `phi[y, x] = -np.tanh((x - dist) / (INTERFACE_WIDTH / 2.0))`
* **Interpretação:**
    * $\phi \approx +1$: À esquerda da interface (Fluido Invasor).
    * $\phi \approx -1$: À direita da interface (Fluido Residente/Deslocado).
    * O termo `INTERFACE_WIDTH` ($\xi$) controla a suavidade da transição.

---

## 3. Inicialização Magnetostática (Campo H0)

O sistema é inicializado com um campo magnético externo constante aplicado na direção $x$. Como o problema resolve o potencial $\psi$, impõe-se uma condição inicial que gere esse campo.

$$\mathbf{H} = -\nabla \psi \implies H_x = -\frac{\partial \psi}{\partial x}$$

Para obter um campo $H_x = H_0$ constante e uniforme no tempo $t=0$:

$$\psi(x) = H_0 (L_x - x)$$

* **No Código:** `psi[y, x] = H0 * (NX - x)`
* **Consequência Física:**
    * Em $x=0$ (Entrada), o potencial é máximo ($H_0 \cdot L_x$).
    * Em $x=L_x$ (Saída), o potencial é zero.
    * O gradiente resultante aponta da direita para a esquerda, mas o campo magnético físico $\mathbf{H}$ (negativo do gradiente) aponta na direção positiva de $x$.

---

## 4. Inicialização das Populações LBM ($f_i$)

As funções de distribuição não podem ser iniciadas com zeros, pois isso implicaria vácuo. Elas devem ser iniciadas em um estado de **Equilíbrio Termodinâmico Local**.

$$f_i(\mathbf{x}, 0) \approx f_i^{eq}(\rho=1, \mathbf{u}=0)$$

Como a velocidade inicial é nula, a fórmula de equilíbrio quadrática simplifica-se para:

$$f_i = w_i \rho$$

* **No Código:**
    ```python
    for i in range(9):
        f[y, x, i] = W_LBM[i] * rho[y, x]
    ```
* **Justificativa:** Isso garante que, no primeiro passo de tempo, o fluido tenha massa mas nenhum momento, permitindo que as forças (pressão, capilaridade, magnética) comecem a acelerar o fluido suavemente a partir do repouso.