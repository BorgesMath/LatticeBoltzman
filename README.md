# **Simulação Magneto-Hidrodinâmica de Instabilidades em Meios Porosos (LBM-PhaseField)**

Este repositório contém uma implementação numérica estritamente acoplada para a simulação de escoamentos multifásicos em meios porosos sob a influência de campos magnéticos. O escopo físico investiga a **Instabilidade de Saffman-Taylor** (digitação viscosa) e sua modulação (supressão ou amplificação) através da competição entre forças magnéticas de Kelvin e forças capilares de Korteweg.

A arquitetura numérica baseia-se em um esquema segregado contendo:

1.  **Método de Lattice Boltzmann (LBM)** para a hidrodinâmica subjacente.
2.  **Campo de Fase (Cahn-Hilliard)** para a captura implícita da interface difusa.
3.  **Solver Elíptico (Poisson)** para o cálculo do potencial magnetostático escalar.

## **🏗️ Arquitetura do Software e Acoplamento Numérico**

O modelo adota uma abordagem de integração temporal particionada (esquema segregado). Em cada iteração de passo de tempo macroscópico ($\Delta t$), o orquestrador executa o fluxo sequencial abaixo, garantindo a atualização progressiva dos campos de força antes da etapa de colisão hidrodinâmica.

**Ciclo de Integração (main.py):**

1.  **Solver Magnetostático:** A suscetibilidade magnética $\chi(\phi)$ é atualizada. O solver de Poisson relaxa o potencial magnético $\psi$ baseado na nova distribuição de fases.
2.  **Sub-ciclos de Cahn-Hilliard:** O campo de fase $\phi$ é advectado usando o campo de velocidade restrito $\mathbf{u}$. Devido à severa restrição de estabilidade difusiva ($M \Delta t < \Delta x^2$), esta etapa é sub-iterada (CH_SUBSTEPS).
3.  **Solver Hidrodinâmico (LBM):** O potencial químico local e os gradientes magnéticos geram os tensores de estresse (forças de Korteweg e Kelvin). O LBM computa a colisão BGK modificada pelo esquema de Guo e realiza o *streaming* das populações, atualizando $f_i$ e $f_i^{eq}$.

## **📂 Descrição Detalhada dos Módulos**

A estrutura de diretórios separa rigorosamente as configurações, a inicialização e os solvers das EDPs. A biblioteca Numba é injetada nos *kernels* numéricos para compilação Just-In-Time (JIT) via decorador `@njit(parallel=True)`.

* **config.py**: Definição global de escalares físicos e tensores discretos. Contém as restrições da malha, tempos de relaxação ($\tau$), coeficientes de Ginzburg-Landau ($\sigma, \xi$), mobilidade, amplitude magnética e os tensores de topologia do reticulado D2Q9 (pesos e velocidades discretas).
* **initialization.py**: Define o Problema de Valor Inicial (PVI). Aloca os tensores em memória RAM e gera o campo de fase inicial com perturbação harmônica associada ao modo $m$ ($y = y_0 + \epsilon \cos(kx)$), inicializando as populações hidrodinâmicas no equilíbrio local.
* **poisson.py**: Resolve a equação elíptica para $\psi$ utilizando o método iterativo de Sobre-Relaxação Sucessiva (SOR). As condições de contorno de Neumann asseguram campo tangencial nulo nas paredes.
* **cahn_hilliard.py**: Resolve a equação não-linear parabólica/quártica do campo de fase. Utiliza esquema *Upwind* de diferenças finitas de 1ª ordem para o termo advectivo linear, vital para a supressão de oscilações espúrias na descontinuidade difusa.
* **lbm.py**: *Kernel* da equação de transporte de Boltzmann. Acopla o termo de forçamento macroscópico ao operador de colisão BGK. Implementa a recuperação da velocidade macroscópica em meios porosos (modelo de Darcy-Brinkman não-linear) e condições de contorno abertas dinâmicas (*Velocity Inlet* / *Neumann Outlet*).
* **lsa.py**: Rotina analítica para Análise de Estabilidade Linear (LSA). Computa a relação de dispersão $\omega(k)$ através dos adimensionais de controle ($Ca, Bo_m, Pe$), provendo uma predição teórica de estabilidade do modo inserido.
* **post_process.py**: Pipeline de diagnósticos espaciais e temporais. Realiza a computação da métrica de curvatura da interface e exporta os campos estocados em tensores NumPy para imagens matriciais bidimensionais com mapas de contorno/corrente.
* **main.py**: Orquestrador lógico do sistema. Gerencia o loop temporal principal e as chamadas sequenciais aos módulos acima.

## **🧮 Formulação Matemático-Física Aplicada**

### **1. Dinâmica de Boltzmann no Reticulado (LBM-BGK)**

A equação cinética discreta com relaxamento e forçamento iterativo é governada por:

$$f_i(\mathbf{x} + \mathbf{e}_i \Delta t, t + \Delta t) - f_i(\mathbf{x}, t) = -\frac{1}{\tau} [f_i(\mathbf{x}, t) - f_i^{eq}(\mathbf{x}, t)] + \Delta t S_i$$

Onde $S_i$ é o termo fonte derivado do esquema de Guo, acoplando três mecanismos macroscópicos:

1.  **Força Interfacial de Korteweg:** $\mathbf{F}_s = \mu_\phi \nabla \phi$
2.  **Força Ponderomotriz de Kelvin:** $\mathbf{F}_m = -\frac{1}{2} (\nabla \psi)^2 \nabla \chi$ (simplificação computacional via suscetibilidade linear isotrópica)
3.  **Arrasto de Darcy-Brinkman:** $\mathbf{F}_d = - \frac{\nu}{K} \mathbf{u}$

### **2. Termodinâmica de Interface (Cahn-Hilliard)**

O escoamento das duas fases incompressíveis é modelado pela conservação do parâmetro de ordem $\phi$:

$$\frac{\partial \phi}{\partial t} + \nabla \cdot (\mathbf{u} \phi) = \nabla \cdot (M \nabla \mu_\phi)$$

O potencial químico da mistura binária $\mu_\phi$ é a derivada variacional da energia livre:

$$\mu_\phi = \frac{\delta \mathcal{F}}{\delta \phi} = 4\beta \phi (\phi^2 - 1) - \kappa \nabla^2 \phi$$

### **3. Aproximação Magnetostática (Equação de Poisson)**

Na ausência de correntes de condução/deslocamento, o campo magnético é escalar $\psi$. Assumindo lei de Gauss conservativa $\nabla \cdot \mathbf{B} = 0$:

$$\nabla \cdot ((1 + \chi(\phi)) \nabla \psi) = 0$$

## **📋 Requisitos e Execução**

As dependências garantem suporte à manipulação tensorial multidimensional e execução paralela assíncrona compilada em hardware (JIT).

### **Pré-requisitos (Python 3.8+)**

```bash
pip install numpy numba matplotlib tqdm