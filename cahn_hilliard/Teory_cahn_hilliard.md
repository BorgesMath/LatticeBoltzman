## 5. Solver Cahn-Hilliard (Interface)

| Símbolo | Definição Matemática | Descrição Física/Cinética |
| :--- | :--- | :--- |
| $\phi(\mathbf{r}, t)$ | Parâmetro de Ordem | Campo escalar conservado de concentração relativa. |
| $\mu$ | $\frac{\delta F}{\delta \phi}$ | Potencial químico (derivada variacional da energia livre). |
| $M$ | Coeficiente de Onsager | Mobilidade fenomenológica do sistema (difusividade). |
| $\kappa$ | $\lambda \epsilon^2$ | Coeficiente de energia de gradiente (tensão interfacial). |
| $f(\phi)$ | $\frac{1}{4\theta}(\phi^2 - 1)^2$ | Densidade de energia livre local (potencial de poço duplo). |
| $f'(\phi)$ | $\frac{df}{d\phi} = \phi^3 - \phi$ | Termo de força não-linear (bulk force). |
| $\nabla^2$ | $\sum_{i=1}^{d} \frac{\partial^2}{\partial x_i^2}$ | Operador Laplaciano (difusão e curvatura). |
| $\mathbf{j}$ | $-M \nabla \mu$ | Vetor densidade de fluxo de massa. |
| $\gamma$ | $\frac{2\sqrt{2}}{3}\sqrt{\kappa}$ | Tensão superficial resultante da interface. |
| $\epsilon$ | $\sqrt{\kappa}$ | Parâmetro de largura da interface difusa. |
| $\Delta t$ | $t_{n+1} - t_n$ | Discretização temporal para integração da PDE. |
| $h$ | $\Delta x, \Delta y$ | Espaçamento da malha na discretização espacial. |


### 5.1 Fundamentação Teórica: Potencial Químico e Equação de Transporte

#### 5.1.1 Potencial Químico de Ginzburg-Landau ($\mu$)
O potencial químico $\mu$ é definido como a derivada variacional da funcional de energia livre total $F$ em relação ao parâmetro de ordem $\phi$. Para um sistema binário, ele é composto por uma contribuição local (bulk) e uma contribuição interfacial (gradiente):

$$\mu = \frac{\delta F}{\delta \phi} = \underbrace{4 \beta \phi (\phi^2 - 1)}_{\text{Termo de Poço Duplo}} - \kappa \nabla^2 \phi$$

* **Termo de Poço Duplo:** Representa a densidade de energia livre local. O termo cúbico $4 \beta (\phi^3 - \phi)$ cria dois estados de equilíbrio estáveis (tipicamente $\phi = \pm 1$). Matematicamente, ele força a segregação das fases.
* **Termo de Gradiente ($\kappa \nabla^2 \phi$):** Atua como um regularizador que penaliza interfaces excessivamente nítidas, garantindo uma transição suave e contínua entre as fases (teoria de interface difusa).

#### 5.1.2 Equação de Evolução e Conservação
A dinâmica do parâmetro de ordem segue uma lei de conservação local, onde a variação temporal de $\phi$ é devida tanto ao transporte advectivo quanto ao fluxo difusivo impulsionado pelo gradiente de potencial químico:

$$\frac{\partial \phi}{\partial t} + \nabla \cdot (\phi \mathbf{u}) = \nabla \cdot (M \nabla \mu)$$

Onde:
* $\nabla \cdot (\phi \mathbf{u})$: Representa o **acoplamento hidrodinâmico**. A interface é carregada pelo campo de velocidade $\mathbf{u}$ (geralmente resolvido via equações de Navier-Stokes).
* $\nabla \cdot (M \nabla \mu)$: Representa o **fluxo de Cahn-Hilliard**. Diferente da difusão de Fick clássica, aqui o fluxo é proporcional ao gradiente de $\mu$, permitindo a "difusão uphill" (contra o gradiente de concentração), essencial para a decomposição espinodal.

### 5.2 Logica do codigo

#### 5.2.1 Descrição das Variáveis de Estado e Dimensões

| Variável | Definição Técnica | Função no Algoritmo |
| :--- | :--- | :--- |
| `phi.shape` | Tupla de Dimensão ($N_y, N_x$) | Representa a topologia e resolução discreta do domínio $\Omega$. |
| `ny, nx` | Escalares de Extensão | Dimensões da grade extraídas para parametrização dos limites de iteração. |
| `phi_next` | Tensor de Estado $t + \Delta t$ | Buffer para armazenar o parâmetro de ordem após a integração temporal. |
| `mu` ($\mu$) | Potencial Químico | Campo escalar derivado da variação da energia livre: $\mu = \frac{\delta F}{\delta \phi}$. |

---

---

#### 5.2.2 Fluxograma do Loop de Potencial Químico
```mermaid
graph TD
    Start([Início: prange y=1 to ny-2]) --> XLoop[Loop x=1 to nx-2]
    XLoop --> Lap[Cálculo do Laplaciano: isotropic_laplacian phi, y, x]
    Lap --> Deriv[Derivada da Energia Local: df_dphi]
    Deriv --> MuCalc[mu y,x = df_dphi - kappa * lap_phi]
    MuCalc --> NextX{Próximo x}
    NextX --> XLoop
    NextX --> NextY{Próximo y}
    NextY --> Start
    NextY --> End([Fim do Loop: mu preenchido])

#### 5.2.3 Condição de contorno no potencial quimico (Neumann)


Estas quatro linhas impõem a condição de **Gradiente Normal Nulo** ($\nabla \mu \cdot \mathbf{n} = 0$) nas fronteiras do domínio.

#### O que cada linha executa na malha:

| Linha de Código | Fronteira Afetada | Operação Matemática Discreta |
| :--- | :--- | :--- |
| `mu[0, :] = mu[1, :]` | **Topo** (Linha 0) | Igualar a borda superior à primeira linha interna. |
| `mu[-1, :] = mu[-2, :]` | **Base** (Última linha) | Igualar a borda inferior à penúltima linha interna. |
| `mu[:, 0] = mu[:, 1]` | **Esquerda** (Coluna 0) | Igualar a borda esquerda à primeira coluna interna. |
| `mu[:, -1] = mu[:, -2]` | **Direita** (Última coluna) | Igualar a borda direita à penúltima coluna interna. |

---

#### A Lógica Técnica: Por que "Copiar" os Valores?

1. **Aproximação da Derivada de Neumann:**
   Em diferenças finitas, a derivada na borda é aproximada por $\frac{\mu_{borda} - \mu_{interno}}{\Delta x}$. Ao forçar `mu[borda] = mu[interno]`, o numerador torna-se zero, garantindo matematicamente que a inclinação (gradiente) na parede seja nula.

2. **Eliminação do "Abismo" Numérico:**
   Como a matriz `mu` é inicializada com zeros, sem essas linhas, as bordas permaneceriam em `0.0`. Se o centro da simulação evoluir para um potencial $\mu = 0.5$, haveria um degrau artificial de $0.5$ para $0.0$ em apenas um pixel.
   
3. **Estabilidade no Próximo Passo:**
   No loop seguinte, o código calcula o Laplaciano de $\mu$ (`lap_mu`). O Laplaciano é extremamente sensível a variações bruscas. Esse "degrau" de potencial nas bordas criaria uma força de difusão gigantesca e irreal, fazendo com que a simulação "explodisse" (divergisse) nas paredes.

#### 5.2.3 Fluxograma da equação de conservação de ϕ
```mermaid
graph TD
    A[Início do Loop Espacial: y, x] --> B["Cálculo do Laplaciano: lap_mu = isotropic_laplacian(mu, y, x)"]
    B --> C["Termo Difusivo: diffusion = mobility * lap_mu"]
    C --> D{Verificar ux > 0?}
    
    D -- Sim --> E1["dphi_dx = phi[y, x] - phi[y, x-1] (Backward)"]
    D -- Não --> E2["dphi_dx = phi[y, x+1] - phi[y, x] (Forward)"]
    
    E1 --> F{Verificar uy > 0?}
    E2 --> F
    
    F -- Sim --> G1["dphi_dy = phi[y, x] - phi[y-1, x] (Backward)"]
    F -- Não --> G2["dphi_dy = phi[y+1, x] - phi[y, x] (Forward)"]
    
    G1 --> H["Advecção: -(ux * dphi_dx + uy * dphi_dy)"]
    G2 --> H
    
    H --> I["Euler Explícito: new_phi = phi[y, x] + dt_ch * (advection + diffusion)"]
    
    I --> J["Condições de Contorno de Phi"]
    J --> K1["Entrada (Inlet): phi_next[:, 0] = 1.0"]
    J --> K2["Saída (Outlet): phi_next[:, -1] = phi_next[:, -2]"]
    
    K2 --> L["Saturação (Clipping): phi_next = clip(phi_next, -1.0, 1.0)"]
    
    L --> M[Fim do Loop Espacial]
    ```