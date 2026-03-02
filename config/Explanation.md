# Documentação Técnica: Parâmetros de Configuração (`config.py`)

Este documento detalha os parâmetros estáticos definidos em `config.py`, conectando cada variável do código ("hard-coded") à sua origem nas equações diferenciais e na teoria física (LBM, Cahn-Hilliard e Magnetostática).

## 1. Topologia e Discretização Espaço-Temporal

Estes parâmetros definem o domínio computacional $\Omega$ e a discretização temporal.

| Parâmetro | Valor Típico | Símbolo Teórico | Definição e Impacto na Equação |
| :--- | :--- | :--- | :--- |
| `NY`, `NX` | 300, 600 | $L_y, L_x$ | **Dimensões do Reticulado.** Define a resolução espacial $\Delta x = \Delta y = 1$ (Lattice Units). Impacta diretamente o Custo Computacional $O(N_x N_y)$. |
| `MAX_ITER` | 5000 | $T_{fim}$ | **Horizonte de Tempo.** O tempo físico total simulado é $t = \text{MAX\_ITER} \times \Delta t_{LBM}$. |
| `SNAPSHOT_STEPS` | 6 | $N_{save}$ | **Frequência de Amostragem.** Define quantos arquivos de saída (checkpoints) serão gerados para pós-processamento. |

---

## 2. Hidrodinâmica (Lattice Boltzmann e Meios Porosos)

Parâmetros que controlam o escoamento dos fluidos e a interação com a matriz porosa via Lei de Darcy-Brinkman.

| Parâmetro | Símbolo | Origem Teórica (Equação) | Descrição Física |
| :--- | :--- | :--- | :--- |
| `TAU_IN` | $\tau_{in}$ | $\nu = c_s^2 (\tau - 0.5)\Delta t$ | **Tempo de Relaxação (Fluido Invasor).** Controla a viscosidade cinemática. Valores próximos a 0.5 indicam fluidos quase invíscidos (alto Reynolds), mas podem gerar instabilidade numérica. |
| `TAU_OUT` | $\tau_{out}$ | $\nu = c_s^2 (\tau - 0.5)\Delta t$ | **Tempo de Relaxação (Fluido Residente).** Define a viscosidade do fluido deslocado. A razão $\nu_{out}/\nu_{in}$ define a Razão de Mobilidade $M$, crítica para a instabilidade de Saffman-Taylor. |
| `U_INLET` | $U_{in}$ | $f_{inlet}^{eq}(\rho, \mathbf{u}_{in})$ | **Velocidade de Injeção.** Imposta na condição de contorno de entrada (Dirichlet). Determina o Número Capilar ($Ca \propto U_{in}\mu/\sigma$). |
| `K_0` | $K$ | $\mathbf{F}_{drag} = -\frac{\nu}{K}\mathbf{u}$ | **Permeabilidade Absoluta.** Representa a resistência do meio poroso. Quanto menor $K$, maior a força de arrasto (termo de Brinkman) oposta ao fluxo. |

---

## 3. Termodinâmica de Interface (Cahn-Hilliard)

Esta seção define a energia livre do sistema. Diferente do LBM, onde a viscosidade é entrada direta, no Método de Campo de Fase (PF), a tensão superficial ($\sigma$) e a largura da interface ($\epsilon$) são as entradas físicas, e os coeficientes da equação ($\beta, \kappa$) são derivados matematicamente.

### 3.1 Entradas Físicas
| Parâmetro | Símbolo | Significado |
| :--- | :--- | :--- |
| `M_MOBILITY` | $M$ | **Mobilidade de Cahn-Hilliard.** Controla a velocidade de difusão do potencial químico ($\frac{\partial \phi}{\partial t} = \nabla \cdot (M \nabla \mu)$). Se muito alto, a interface "borra" rapidamente; se muito baixo, trava a dinâmica. |
| `SIGMA` | $\sigma$ | **Tensão Superficial.** A energia por unidade de área necessária para criar a interface. Define a força de retração capilar. |
| `INTERFACE_WIDTH` | $\epsilon$ ou $\xi$ | **Largura da Interface Difusa.** O número de células da grade sobre as quais a densidade varia de $\phi=-1$ a $\phi=1$. Geralmente $3 < \epsilon < 5$ para estabilidade numérica. |
| `CH_SUBSTEPS` | $N_{sub}$ | **Sub-ciclos.** Como a difusão é explicita, requer $\Delta t_{CH} \ll \Delta x^2 / M$. O código executa $N_{sub}$ passos de Cahn-Hilliard para cada passo de LBM para garantir estabilidade Courant–Friedrichs–Lewy (CFL). |

### 3.2 Coeficientes Derivados (Energia Livre de Ginzburg-Landau)
A funcional de energia livre é $\mathcal{F}(\phi) = \int_{\Omega} \left( \beta \Psi(\phi) + \frac{1}{2}\kappa |\nabla \phi|^2 \right) d\Omega$.

| Parâmetro Calculado | Fórmula no Código | Função na Equação $\mu = \frac{\delta \mathcal{F}}{\delta \phi}$ |
| :--- | :--- | :--- |
| `BETA` ($\beta$) | $\frac{3}{4} \sigma \epsilon$ | **Coeficiente de Bulk.** Multiplica o termo de poço duplo $\phi(\phi^2-1)$. Controla a separação de fases (segregação). |
| `KAPPA` ($\kappa$) | $\frac{3}{8} \sigma \epsilon$ | **Coeficiente de Gradiente.** Multiplica o Laplaciano $-\nabla^2 \phi$. Penaliza gradientes abruptos, "suavizando" a interface. |

> **Nota:** A relação entre $\sigma$, $\epsilon$, $\beta$ e $\kappa$ deriva da solução de perfil de equilíbrio $\phi(x) = \tanh(x \sqrt{\beta/2\kappa})$. O código utiliza um escalonamento específico para garantir que a integral do perfil de energia resulte exatamente em $\sigma$.

---

## 4. Magnetostática (Solver de Poisson)

Parâmetros para a resolução da equação elíptica $\nabla \cdot ((1+\chi)\nabla \psi) = 0$.

| Parâmetro | Símbolo | Equação Associada | Descrição Técnica |
| :--- | :--- | :--- | :--- |
| `H0` | $H_0$ | $\psi(x=L) = H_0 \cdot L$ | **Campo Magnético Aplicado.** Define a condição de contorno de Dirichlet para o potencial magnético, gerando o gradiente global que impulsiona a instabilidade magnética. |
| `CHI_MAX` | $\chi_{max}$ | $\mathbf{M} = \chi \mathbf{H}$ | **Suscetibilidade Magnética.** Define o contraste magnético entre os fluidos. O fluido 1 tem $\chi \approx \chi_{max}$ (ferrofluido) e o fluido 2 tem $\chi \approx 0$. |
| `SOR_OMEGA` | $\omega$ | $\psi^{new} = (1-\omega)\psi^{old} + \omega \psi^*$ | **Fator de Relaxação (SOR).** Acelera a convergência do solver iterativo. Para redes grandes ($600 \times 300$), um valor próximo de 2 (ex: 1.85) é necessário para evitar convergência lenta (over-relaxation). |

---

## 5. Tensores do Modelo LBM (D2Q9)

Definição da estrutura discreta da rede para o método **D2Q9** (2 Dimensões, 9 Velocidades). Estes valores são constantes matemáticas do método BGK.

### 5.1 Pesos e Vetores Base

| Variável | Definição Matemática | Função no Algoritmo |
| :--- | :--- | :--- |
| `W_LBM` ($w_i$) | Pesos da Quadratura Gaussiana | Usados para calcular a **distribuição de equilíbrio** $f^{eq}$. Garantem que os momentos da distribuição recuperem as equações de Navier-Stokes. <br>• Centro (0): $4/9$ <br>• Ortogonais (1-4): $1/9$ <br>• Diagonais (5-8): $1/36$ |
| `CX`, `CY` ($\mathbf{c}_i$) | Vetores de Velocidade Discreta | Definem a direção do passo de **Streaming** (transporte). Mapeiam para onde a densidade $f_i$ se move em $\Delta t$. |
| `OPP` | Tabela de Inversão | Vetor de índices onde `OPP[i]` retorna a direção oposta a `i`. Essencial para aplicar condições de contorno de **Bounce-Back** (reflexão nas paredes). |