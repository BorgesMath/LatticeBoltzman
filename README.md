# 🧲 LBM-PhaseField: Instabilidade de Saffman-Taylor Magnética

Simulação numérica acoplada (Multifísica) para escoamentos multifásicos em meios porosos sob influência de campos magnéticos. O código investiga a competição entre **Digitação Viscosa** e **Forças Magnéticas** utilizando uma arquitetura segregada.

---

## ⚡ Workflow Numérico (Ciclo de Integração)

O orquestrador (`main.py`) executa o seguinte acoplamento a cada passo de tempo $\Delta t$:

```mermaid
graph TD
    A[Início t] --> B(1. Solver Poisson)
    B -->|Calcula Potencial Magnético| C(2. Cahn-Hilliard)
    C -->|Advecta Interface - Substeps| D(3. LBM-BGK)
    D -->|Colisão e Streaming| E[Atualiza f, rho, u]
    E --> F{t < Max?}
    F -- Sim --> A
    F -- Não --> G[Fim]
    
    style B fill:#f9f,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#bfb,stroke:#333
 ```   
    
## 📂 Estrutura do Projeto

A organização modular separa configuração, inicialização e kernels numéricos (compilados via `@njit`).

```text
📦 LatticeBoltzman
 ┣ 📜 main.py               # Orquestrador do loop temporal
 ┣ 📂 config
 ┃ ┗ 📜 config.py           # Parâmetros Físicos (Re, Ca, Bom) e Grid (D2Q9)
 ┣ 📂 initialization
 ┃ ┗ 📜 initialization.py   # PVI: Perturbação senoidal da interface
 ┣ 📂 solvers
 ┃ ┣ 📂 poisson             # Magnetostática
 ┃ ┃ ┗ 📜 poisson.py        # Solver SOR para ∇·((1+χ)∇ψ) = 0
 ┃ ┣ 📂 cahn_hilliard       # Interface Difusa
 ┃ ┃ ┗ 📜 cahn_hilliard.py  # Evolução do parâmetro de ordem (ϕ)
 ┃ ┗ 📂 lbm                 # Hidrodinâmica
 ┃   ┗ 📜 lbm.py            # Kernel BGK + Forças (Guo) + Darcy-Brinkman
 ┣ 📂 analytics
 ┃ ┗ 📜 lsa.py              # Análise de Estabilidade Linear (Previsão Teórica)
 ┗ 📂 post_process
   ┗ 📜 post_process.py     # Exportação de VTK/PNG e métricas (Curvatura)
```
    
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


## 🚀 Como Executar

### Pré-requisitos
Python 3.8+ com suporte a compilação JIT.

```bash
pip install numpy numba matplotlib tqdm
```
### Rodando a Simulação
Edite `config/config.py` para ajustar a malha ou física, e execute:

```bash
python main.py
```

### Referências Bibliográficas 

1. **Modelo LBM-BGK e Hidrodinâmica Discreta:**
   * Bhatnagar, P. L., Gross, E. P., & Krook, M. (1954). A model for collision processes in gases. I. Small amplitude processes in charged and neutral one-component systems. *Physical Review*, 94(3), 511.
   * Qian, Y. H., d'Humières, D., & Lallemand, P. (1992). Lattice BGK models for Navier-Stokes equation. *Europhysics Letters*, 17(6), 479.

2. **Esquema de Forçamento (Termo Fonte no LBM):**
   * Guo, Z., Zheng, C., & Shi, B. (2002). Discrete lattice effects on the forcing term in the lattice Boltzmann method. *Physical Review E*, 65(4), 046308.

3. **Modelagem em Meios Porosos (Darcy-Brinkman no LBM):**
   * Guo, Z., & Zhao, T. S. (2002). Lattice Boltzmann model for incompressible flows through porous media. *Physical Review E*, 66(3), 036304.

4. **Campo de Fase e Dinâmica de Interfaces:**
   * Cahn, J. W., & Hilliard, J. E. (1958). Free energy of a nonuniform system. I. Interfacial free energy. *The Journal of Chemical Physics*, 28(2), 258-267.
   * Jacqmin, D. (1999). Calculation of two-phase Navier-Stokes flows using phase-field modeling. *Journal of Computational Physics*, 155(1), 96-127.

5. **Ferro-hidrodinâmica e Potencial Magnetostático:**
   * Rosensweig, R. E. (1985). *Ferrohydrodynamics*. Cambridge University Press. (Fundamentação para as Equações de Maxwell no limite magnetostático e Força de Kelvin).


