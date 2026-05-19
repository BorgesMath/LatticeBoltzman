# FAQ — Perguntas Técnicas Específicas

Documento de referência com respostas detalhadas a perguntas frequentes sobre
o solver LBM–CH–Magnetostática implementado neste repositório.

---

## 1. O que é o Guo?

**Guo** refere-se ao **esquema de forçamento de Guo, Zheng & Shi (2002)**,
publicado em *Phys. Rev. E* 65, 046308. É a forma matematicamente consistente
de introduzir uma força externa $\mathbf{F}$ na equação de Boltzmann discreta
sem introduzir viés de segunda ordem nos momentos.

### 1.1 Por que não basta somar a força à velocidade?

A maneira "ingênua" de adicionar uma força no LBM seria computar
$f_i^{\mathrm{eq}}$ com uma velocidade "shiftada" $\mathbf{u} + \mathbf{F}\tau/\rho$.
Mas isso introduz erros de $\mathcal{O}(\mathbf{F}^{2})$ nos momentos
macroscópicos e viola a recuperação correta de Navier–Stokes.

### 1.2 As três peças do esquema de Guo

**(a) Velocidade física corrigida** — quando há força, a velocidade
medida pelos momentos *não é* a velocidade real:

$$
\rho\,\mathbf{u}_{\mathrm{phys}} \;=\; \sum_i f_i\,\mathbf{c}_i \;+\; \tfrac{1}{2}\,\mathbf{F}.
$$

A correção $+\tfrac{1}{2}\mathbf{F}$ vem da expansão Chapman–Enskog
(consistência de segunda ordem).

**(b) Equilíbrio com a velocidade corrigida:**

$$
f_i^{\mathrm{eq}} \;=\; w_i\,\rho\left[1 + 3(\mathbf{c}_i\!\cdot\!\mathbf{u}_{\mathrm{phys}}) + \tfrac{9}{2}(\mathbf{c}_i\!\cdot\!\mathbf{u}_{\mathrm{phys}})^{2} - \tfrac{3}{2}|\mathbf{u}_{\mathrm{phys}}|^{2}\right].
$$

**(c) Termo fonte $S_i$** — a "marca registrada" de Guo:

$$
\boxed{\;S_i \;=\; w_i\,\bigl(1 - \tfrac{\omega}{2}\bigr)\,\left[\frac{3(\mathbf{c}_i - \mathbf{u})}{c_s^{2}} + \frac{9(\mathbf{c}_i\!\cdot\!\mathbf{u})\,\mathbf{c}_i}{c_s^{4}}\right]\!\cdot\!\mathbf{F}\;}
$$

O fator $(1 - \omega/2)$ é o que distingue Guo de esquemas anteriores
(Shan–Chen, He–Chen–Doolen) e elimina erros espúrios de viscosidade.

### 1.3 No código

Em `lbm/lbm.py`, linhas 78–96:

```python
ux_star = (ux_l + 0.5*Fx[y,x]) / rho_l           # correção 1/2
ux_phys = ux_star / (1.0 + 0.5*sigma_drag)       # arrasto implícito
# ...
feq = W[i]*rho_l*(1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)
term1 = (CX[i] - ux_phys)*Fx_total + (CY[i] - uy_phys)*Fy_total
term2 = cu*(CX[i]*Fx_total + CY[i]*Fy_total)
Si = W[i]*(1.0 - 0.5*omega)*(3.0*term1 + 9.0*term2)   # ← Guo
```

📄 Detalhes em [`lbm/lbm.md`](../lbm/lbm.md) §1.5 e §5.5.

---

## 2. Como é calculada a pressão para validar Young–Laplace?

A validação de Young–Laplace ($\Delta P = \sigma\,\kappa$) é feita pelo
script `post_process/valida_case.py`.

### 2.1 Pressão no LBM

Em unidades de rede, a equação de estado isotermal do LBM é:

$$
\boxed{\;p \;=\; c_s^{2}\,\rho \;=\; \frac{\rho}{3}\;}
$$

```python
p_2d = rho_2d / 3.0          # linha 60
```

### 2.2 Procedimento de extrapolação

O problema é: a interface é **difusa** (espessura $W$ não nula), então
não dá para ler $p$ "exatamente em cima" da interface — haveria
contaminação do termo de gradiente do potencial químico.

**Solução:** ler a pressão em **regiões de bulk** (longe da interface),
ajustar uma reta, e **extrapolar até a interface**.

```
       Bulk 1 (invasor)             interface      Bulk 2 (residente)
   ┌─────────────────────┐         ┊         ┌──────────────────────┐
   │  •  •  •  •  •  •   │         ┊         │  •  •  •  •  •  •  • │
   │           ↘         │         ┊         │       ↗               │
   │    regressão linear │ → P₁_extrap   ←  │  regressão linear     │
   │                     │              P₂_extrap                    │
   └─────────────────────┘                   └──────────────────────┘
                          ΔP_cap = P₁ − P₂
```

### 2.3 Algoritmo (linha por linha em $y$)

Para cada linha $y$ do domínio:

1. **Localizar a interface** $x_{\mathrm{int}}(y)$ pelo zero de $\phi$
   (interpolação linear entre sinais opostos).

2. **Definir zonas de bulk** com margem de 10% do domínio:
   - Bulk 1: $x \in [\text{margin},\,\min(x_{\mathrm{int}}) - \text{margin}]$
   - Bulk 2: $x \in [\max(x_{\mathrm{int}}) + \text{margin},\,N_x - \text{margin}]$

3. **Regressão linear** em cada bulk:
   $$p_1(x) = a_1 x + b_1,\qquad p_2(x) = a_2 x + b_2.$$

4. **Extrapolar até a interface:**
   $$p_1^{\mathrm{ext}} = a_1\,x_{\mathrm{int}}(y) + b_1,\qquad p_2^{\mathrm{ext}} = a_2\,x_{\mathrm{int}}(y) + b_2.$$

5. **Salto capilar:**
   $$\Delta P_{\mathrm{cap}}(y) = p_1^{\mathrm{ext}} - p_2^{\mathrm{ext}}.$$

6. **Curvatura local** $\kappa(y)$ no mesmo $x_{\mathrm{int}}(y)$.

### 2.4 Recuperação de $\sigma$

Plotando $\Delta P_{\mathrm{cap}}$ vs $\kappa$ e ajustando uma reta:

$$
\boxed{\;\Delta P_{\mathrm{cap}} \;=\; \sigma_{\mathrm{rec}}\,\kappa \;+\; \text{const}\;}
$$

O **declive** é a tensão superficial recuperada. Comparar com $\sigma$
imposta dá o erro de validação.

```python
coeffs_sigma = np.polyfit(kappa_arr, dp_cap_arr, 1)
sigma_rec = coeffs_sigma[0]          # ← linha 132
```

### 2.5 Por que usar regressão de extrapolação em vez de medir ponto a ponto?

- **Regime de Darcy:** longe da interface, a pressão varia linearmente
  por causa do arrasto viscoso. Regressão captura essa rampa.
- **Cancelamento de gradiente:** ao subtrair $p_1^{\mathrm{ext}} - p_2^{\mathrm{ext}}$
  no mesmo $x$, as rampas de Darcy se cancelam parcialmente, isolando o
  salto capilar puro.

---

## 3. Procedimento para calcular curvatura e amplitude

### 3.1 Curvatura

A interface é definida implicitamente pelo *level-set* $\phi = 0$. A
curvatura de um *level-set* é:

$$
\kappa \;=\; \nabla\!\cdot\!\left(\frac{\nabla\phi}{|\nabla\phi|}\right)
\;=\;
\frac{\phi_x^{2}\,\phi_{yy} + \phi_y^{2}\,\phi_{xx} - 2\,\phi_x\,\phi_y\,\phi_{xy}}{(\phi_x^{2}+\phi_y^{2})^{3/2}}.
$$

**Implementação** em `post_process/resultado_curvatura_temporal.py` e
`valida_case.py`:

```python
dy, dx        = np.gradient(phi)
d2y, dy_dx    = np.gradient(dy)
dx_dy, d2x    = np.gradient(dx)
num = (dx**2)*d2y + (dy**2)*d2x - 2.0*dx*dy*dx_dy
den = (dx**2 + dy**2 + 1e-12)**1.5
kappa = num / den
```

**Faixa interfacial** (máscara para média):

$$
\overline{|\kappa|} \;=\; \frac{1}{|\mathcal{M}|}\sum_{(i,j)\in\mathcal{M}} |\kappa_{ij}|,\quad
\mathcal{M} = \{(i,j) : |\phi_{ij}| < 0.1\}.
$$

A máscara $|\phi| < 0.1$ isola a "casca" da interface, onde $\kappa$ está
bem definido. Fora disso, $\phi$ é quase constante e $\kappa$ é dominado
por ruído de divisão por zero.

### 3.2 Amplitude — Duas Definições

**(a) Estimador simples** (`resultado_curvatura_temporal.py`):

$$
A_{\mathrm{simples}} \;=\; \frac{X_{\max} - X_{\min}}{2},
$$

onde $X(y) = \arg\min_x\{\phi(y,x) < 0\}$. Rápido, mas sensível a ruído.

**(b) Projeção de Fourier** (`valida_lsa.py::_amplitude` — recomendado):

1. Localizar a interface com **interpolação sub-pixel** entre os nós onde
   $\phi$ muda de sinal:
   $$x_{\mathrm{exact}}(y) \;=\; i_L + \frac{\phi_L}{\phi_L - \phi_R + 10^{-15}}.$$

2. Projetar no modo $m$:
   $$\boxed{\;A_m \;=\; \frac{2}{N_y}\,\left|\sum_{y=0}^{N_y-1} x_{\mathrm{exact}}(y)\,e^{-2\pi i\,m\,y/N_y}\right|\;}$$

**Vantagens da projeção de Fourier:**
- Filtra ruído de outros modos automaticamente.
- Captura apenas o crescimento do modo desejado (compatível com LSA).
- Mais robusta quando há ondas secundárias.

📄 Detalhes em [`Relatorios/resultado_curvatura_temporal.md`](resultado_curvatura_temporal.md)
e [`Relatorios/valida_lsa.md`](valida_lsa.md).

---

## 4. Como é calculada a viscosidade?

### 4.1 Viscosidade dos fluidos puros

Pela teoria do LBM-BGK (Chapman–Enskog):

$$
\boxed{\;\nu \;=\; c_s^{2}\,(\tau - \tfrac{1}{2})\,\Delta t \;=\; \frac{\tau - 1/2}{3}\;}
$$

(usando $c_s^{2} = 1/3$ e $\Delta t = 1$ em unidades de rede).

```python
nu_in  = (TAU_IN  - 0.5) / 3.0
nu_out = (TAU_OUT - 0.5) / 3.0
```

### 4.2 Viscosidade da mistura (interpolação em $\phi$)

Como $\phi \in [-1, +1]$ varia através da interface, calcula-se a fração
volumétrica de cada fase:

$$
S_{\mathrm{inv}} = \frac{\phi+1}{2},\qquad S_{\mathrm{res}} = 1 - S_{\mathrm{inv}}.
$$

Duas opções de mistura, controladas pelo parâmetro `visc_linear`:

**(a) Média aritmética / linear** (`visc_linear = True`):

$$
\nu_{\mathrm{eff}} \;=\; S_{\mathrm{inv}}\,\nu_{\mathrm{in}} \;+\; S_{\mathrm{res}}\,\nu_{\mathrm{out}}.
$$

Simples, mas **superestima** a viscosidade efetiva na interface quando há
grande contraste $M = \nu_{\mathrm{out}}/\nu_{\mathrm{in}}$.

**(b) Média harmônica** (`visc_linear = False`, padrão):

$$
\frac{1}{\nu_{\mathrm{eff}}} \;=\; \frac{S_{\mathrm{inv}}}{\nu_{\mathrm{in}}} \;+\; \frac{S_{\mathrm{res}}}{\nu_{\mathrm{out}}}.
$$

**Por que harmônica?** Em escoamentos cisalhantes paralelos à interface,
a tensão $\tau_{xy} = \mu\,\partial_y u$ é **contínua** através da
interface. Para preservar essa continuidade na transição difusa, a
mobilidade $1/\mu$ deve variar linearmente — exatamente o que dá a média
harmônica.

### 4.3 De $\nu_{\mathrm{eff}}$ para $\tau$

```python
tau = 3.0 * nu_eff + 0.5
omega = 1.0 / tau
```

No código (`lbm/lbm.py`, linhas 57–65):

```python
S_inv = (phi[y,x] + 1.0) * 0.5
S_res = 1.0 - S_inv
if visc_linear:
    nu_eff = S_inv*nu_in + S_res*nu_out
    tau = 3.0*nu_eff + 0.5
else:
    inv_nu_eff = S_inv/nu_in + S_res/nu_out
    tau = 3.0/inv_nu_eff + 0.5
omega = 1.0/tau
```

---

## 5. Cahn–Hilliard: diferenças finitas, especificidades, separação e substeps

### 5.1 Sim, é diferenças finitas

A CH é resolvida por **diferenças finitas explícitas (Euler-forward)** com
**stencil isotrópico de 5 pontos** para o Laplaciano:

$$
\nabla^{2}f \;\approx\; f_{y,xp} + f_{y,xm} + f_{yp,x} + f_{ym,x} - 4\,f_{y,x}.
$$

### 5.2 Especificidades

1. **Equação de 4ª ordem** decomposta em duas de 2ª ordem:
   - Passo A: $\mu_c = 4\beta\phi(\phi^2-1) - \kappa\nabla^2\phi$.
   - Passo B: $\partial_t\phi = M\nabla^2\mu_c - \mathbf{u}\!\cdot\!\nabla\phi$.

2. **Clamp** $\phi \in [-1, +1]$ após cada substep — proteção numérica
   contra overshoots do potencial quártico (que amplifica excursões
   exponencialmente).

3. **Stencil isotrópico** essencial: stencils anisotrópicos introduzem
   anisotropia na tensão superficial, gerando padrões artificiais em
   instabilidades.

4. **Sem solver linear** — Euler explícito é simples e paraleliza bem,
   mas tem CFL restritiva.

### 5.3 Por que é separado do LBM?

**Quatro razões físicas e técnicas:**

(a) **Equação distinta:** LBM resolve Navier–Stokes via cinética; CH
    resolve dinâmica de campo de fase. São EDPs diferentes com discretizações
    diferentes.

(b) **CFL incompatível:** o LBM tem $\Delta t \lesssim \Delta x / c_s$
    (convectiva). A CH é de 4ª ordem, com $\Delta t \lesssim (\Delta x)^4/(4M\kappa)$
    (difusiva quártica). Misturar as duas obrigaria o LBM a usar um $\Delta t$
    desnecessariamente pequeno.

(c) **Modularidade:** trocar o modelo de interface (Allen–Cahn, level-set)
    sem mexer no LBM.

(d) **Stencils diferentes:** LBM usa D2Q9 (9 direções); CH usa 5 pontos
    para o Laplaciano. Acoplar internamente seria pesado.

### 5.4 Os substeps não deixam o código mais devagar?

**Tecnicamente sim, mas o custo é proporcional e justificável.**

Suponha $N_{\mathrm{sub}} = 5$ substeps de CH por passo de LBM:

| Operação | Custo relativo |
|---|---|
| 1 passo LBM | $\sim 130$ FLOPs/nó |
| 1 substep CH | $\sim 50$ FLOPs/nó |
| Total por step com $N_{\mathrm{sub}} = 5$ | $130 + 5 \times 50 = 380$ FLOPs/nó |

**Aumento de ~3× em FLOPs**, mas:

- Sem substepping, o $\Delta t$ global teria que ser **$N_{\mathrm{sub}}$ vezes menor**,
  obrigando o LBM a fazer também $N_{\mathrm{sub}}$ passos extras
  — custo total seria $5 \times 130 = 650$ FLOPs/nó. **Pior!**

- Substepping permite manter o LBM no regime ótimo de CFL e dar atenção
  só à CH, que é mais barata por nó.

- Numba paraleliza bem ambos os kernels.

**Conclusão:** sem substepping, o código seria **muito mais lento**.
O custo dos substeps é menor que o custo evitado de rodar o LBM em
passos menores.

📄 Detalhes em [`cahn_hilliard/cahn_hilliard.md`](../cahn_hilliard/cahn_hilliard.md) §9.

---

## 6. Como é feito o periodismo nas boundary conditions?

### 6.1 Convenções globais do projeto

| Eixo | Comportamento |
|---|---|
| $y$ (altura do canal) | **Sempre periódico** em todos os módulos |
| $x$ (direção do fluxo) | Controlado por `is_periodic`: periódico ou Zou–He |

### 6.2 Implementação técnica do periodismo

**(a) Aritmética modular** — usada nos kernels (LBM, CH, Poisson):

```python
yp = (y + 1) % ny
ym = (y - 1 + ny) % ny
```

O operador `%` (módulo) "dobra" o índice: $-1 \mapsto N_y - 1$, $N_y \mapsto 0$.
Isso **identifica fisicamente** as linhas $y = 0$ e $y = N_y - 1$
(canal infinitamente periódico em $y$, geometria tipo Hele-Shaw).

**(b) No streaming do LBM** (linha 100):

```python
be_y = (y + CY[i] + ny) % ny
```

Quando uma partícula sairia do topo, reaparece pelo fundo, com a mesma
velocidade — sem reflexão, sem perda.

**(c) Quando $x$ é periódico** (`is_periodic = True`):

```python
xp = (x + 1) % nx
xm = (x - 1 + nx) % nx
# E no streaming:
be_x = (x + CX[i] + nx) % nx
```

**(d) Quando $x$ não é periódico** (`is_periodic = False`):

```python
if x == 0 or x == nx - 1: continue   # pula bordas
xp, xm = x + 1, x - 1                # vizinhos não modulares
```

E após o streaming, aplica **Zou–He** (linhas 110–128 de `lbm.py`):
- $x = 0$ (inlet): velocidade prescrita $u_x = u_{\mathrm{inlet}}$.
- $x = N_x - 1$ (outlet): pressão prescrita $\rho = 1$.

### 6.3 Periodismo "implícito" na inicialização

A perturbação inicial usa cosseno:

$$x_{\mathrm{int}}(y) \;=\; x_c + A\cos(2\pi m y/N_y),$$

que tem **período exato** $N_y$. Combinado com $y$ modular nos kernels,
não há descontinuidade entre $y = 0$ e $y = N_y - 1$ — o modo $m$ é
inteiro e fecha sobre si mesmo.

### 6.4 BC em $\tilde\psi$ (Poisson magnetostático)

Diferente do LBM/CH: em $\tilde\psi$ usa-se **Neumann em $x$** e
**periódico em $y$**:

```python
psi_tilde[y, 0]      = psi_tilde[y, 1]      # Neumann em x=0
psi_tilde[y, nx - 1] = psi_tilde[y, nx - 2] # Neumann em x=Nx-1
```

📄 Detalhes em [`Magnetismo/poisson.md`](../Magnetismo/poisson.md) §5.

---

## 7. Como é calculado o magnetismo e por que ele tem variações perto da interface?

### 7.1 Cálculo do magnetismo — Decomposição de Helmholtz

Em vez de resolver para $\psi_{\mathrm{total}}$ (potencial completo), o
código resolve apenas a **parte perturbadora** $\tilde\psi$:

$$
\psi_{\mathrm{total}} \;=\; \underbrace{-\mathbf{H}_0\!\cdot\!\mathbf{r}}_{\text{campo de fundo}} \;+\; \underbrace{\tilde\psi}_{\text{perturbação}}.
$$

Que implica:

$$
\boxed{\;\mathbf{H}_{\mathrm{total}}(\mathbf{x}) \;=\; \mathbf{H}_0 \;-\; \nabla\tilde\psi(\mathbf{x})\;}
$$

A equação resolvida é a **Poisson generalizada com fonte**:

$$
\boxed{\;\nabla\!\cdot\!\bigl[\mu_r(\phi)\,\nabla\tilde\psi\bigr] \;=\; \mathbf{H}_0\!\cdot\!\nabla\mu_r\;}
$$

com $\mu_r(\phi) = 1 + \chi(\phi)$. Resolvida por **SOR Gauss–Seidel** com
volumes finitos, 15 iterações por passo, fator $\omega \approx 1.85$.

### 7.2 Por que variações **apenas perto da interface**?

O **lado direito da equação é o termo fonte** $\mathbf{H}_0\!\cdot\!\nabla\mu_r$.
Examine-o:

- Em regiões de **fluido puro** (longe da interface), $\chi$ é constante,
  então $\nabla\mu_r = 0$ e a fonte é **nula**.
- Em regiões de **interface difusa**, $\chi$ varia rapidamente com $\phi$,
  então $\nabla\mu_r \neq 0$ e a fonte é **não-nula**.

Como $\tilde\psi$ é solução de uma Poisson com fonte localizada, $\tilde\psi$
e seu gradiente $\nabla\tilde\psi$ são **concentrados** na faixa interfacial,
decaindo rapidamente longe dela.

### 7.3 Interpretação física

A descontinuidade de permeabilidade entre os dois fluidos **refrata** as
linhas de campo magnético na interface, exatamente como a luz é refratada
ao mudar de meio:

```
       Fluido invasor (χ>0)  │  Fluido residente (χ=0)
                             │
       H₀  ━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━>
                            ╱│╲
                          ╱  │  ╲
                       ╱     │     ╲      ← linhas de H se concentram
                    ╱        │        ╲      no fluido magnético
                 ╱           │           ╲
       H₀  ━━━━━━━━━━━━━━━━━━│━━━━━━━━━━━━━━━━━━━━━━━>
```

Essa "concentração" gera o **gradiente local de $|\mathbf{H}|^{2}$**, que
por sua vez aparece na **força ponderomotriz** que atua na interface:

$$
\mathbf{F}_{m} \;=\; -\tfrac{1}{2}\,|\mathbf{H}|^{2}\,\nabla\chi.
$$

Como $\nabla\chi$ é não-nulo **somente na interface**, a força magnética
**só atua na interface**, agindo como uma "tensão superficial extra"
(positiva ou negativa, dependendo da orientação do campo).

### 7.4 Refração no diagrama da equação

Reescrevendo a equação como condição de salto na interface (formulação
sharp-interface limite):

$$
\mu_r^{(1)}\,\mathbf{H}^{(1)}\!\cdot\!\hat{\mathbf{n}} \;=\; \mu_r^{(2)}\,\mathbf{H}^{(2)}\!\cdot\!\hat{\mathbf{n}}
\qquad \text{(continuidade de }\mathbf{B}\!\cdot\!\hat{\mathbf{n}}\text{)}
$$

$$
\mathbf{H}^{(1)}\!\cdot\!\hat{\mathbf{t}} \;=\; \mathbf{H}^{(2)}\!\cdot\!\hat{\mathbf{t}}
\qquad \text{(continuidade de }\mathbf{H}\!\cdot\!\hat{\mathbf{t}}\text{)}
$$

São essas duas condições que **forçam** a refração e justificam por que
$\tilde\psi$ tem que existir: ele é a "correção" necessária para
satisfazê-las.

📄 Detalhes em [`Magnetismo/poisson.md`](../Magnetismo/poisson.md) §1.

---

## 8. O que é $k\xi$ e por que tem que ser menor que 0.01?

### 8.1 Definição

$$
\boxed{\;k\xi \;=\; \frac{2\pi\,m\,W}{N_y}\;}
$$

onde:
- $k = 2\pi m/N_y$ é o **número de onda** do modo $m$ perturbado.
- $\xi = W$ é a **espessura da interface difusa** (`INTERFACE_WIDTH`).

É um **número adimensional** que mede o produto $k\xi$ — análogo a um
"número de Strouhal" geométrico entre a escala da onda e a escala da
interface.

### 8.2 Significado físico

O método de campo de fase **aproxima** o limite *sharp-interface* da
dinâmica de Cahn–Hilliard. Essa aproximação é válida quando a interface
é **fina** comparada à escala da perturbação:

$$
\lambda_{\mathrm{onda}} = \frac{2\pi}{k} \;\gg\; \xi
\quad \Longleftrightarrow \quad
k\xi \;\ll\; 1.
$$

### 8.3 Por que o limite de 0.01?

Análises assintóticas matched-asymptotics da Cahn–Hilliard
(Pego, Bates, Fife, Cahn–Elliott–Novick-Cohen) mostram que o **erro de
fase-field** vs a solução *sharp-interface* escala como:

$$
\frac{\|\phi_{\mathrm{CH}} - \phi_{\mathrm{sharp}}\|}{\|\phi_{\mathrm{sharp}}\|} \;\sim\; (k\xi)^{2}.
$$

Para erro $< 5\%$:

$$
(k\xi)^{2} < 0.05 \quad \Longrightarrow \quad k\xi < 0.22.
$$

Mas $k\xi < 0.22$ é o **limite teórico permissivo**. Na prática:

- **Critério mais conservador:** $k\xi < 0.05$ → erro $< 0.25\%$.
- **Critério adotado** (`valida_lsa.py` linha 134): $k\xi < 0.01$ →
  erro $< 0.01\%$, com folga para acoplamento com outras fontes de erro
  (CFL, LBM-Mach, etc.).

### 8.4 No código

Em `post_process/valida_lsa.py`:

```python
kxi = 2.0 * np.pi * mode_m * int_width / ny
if amplitude >= int_width:
    print("[AVISO] amplitude >= W — fora do regime LSA puro")
print(f"[info] kξ = {kxi:.4f}  (ideal < 0.01)")
```

### 8.5 Trade-off prático

Para reduzir $k\xi$, opções:
- **Diminuir $W$** (interface mais fina): mas $W < 3$ viola Nyquist no
  stencil de 5 pontos.
- **Aumentar $N_y$** (domínio maior): custo computacional cresce.
- **Diminuir $m$** (modo de menor número de onda): mas pode sair do
  modo mais instável.

Portanto, $k\xi < 0.01$ é um **alvo**, não uma obrigação absoluta —
desvios moderados são aceitáveis se outras métricas (erro LSA, balanço
de massa) ficarem dentro do esperado.

---

## 9. Como é definida e implementada a espessura da interface?

### 9.1 Definição teórica

A solução de equilíbrio da Cahn–Hilliard para uma interface plana
unidimensional é:

$$
\phi_{\mathrm{eq}}(z) \;=\; \tanh\!\left(\frac{z\sqrt{2}}{W}\right).
$$

A **espessura $W$** caracteriza a transição: $\phi$ vai de $-0.96$ a
$+0.96$ em um intervalo $\Delta z \approx 2W$.

A relação entre $W$ e os parâmetros da energia livre é:

$$
\boxed{\;W \;=\; \sqrt{\frac{\kappa}{2\beta}}\;}
$$

(deriva do balanço entre o termo de gradiente $\tfrac{\kappa}{2}|\nabla\phi|^{2}$
e o poço duplo $\beta(\phi^2-1)^2$ no funcional de Ginzburg–Landau).

### 9.2 Calibração inversa: dado $\sigma$ e $W$, calcular $\beta$ e $\kappa$

A tensão superficial é $\sigma = \tfrac{2\sqrt{2}}{3}\sqrt{\kappa\beta}$.
Resolvendo o sistema:

$$
\boxed{\;
\beta \;=\; \frac{3\sqrt{2}\,\sigma}{4\,W},\qquad
\kappa \;=\; \frac{3\sqrt{2}\,\sigma\,W}{4}
\;}
$$

### 9.3 Implementação

**(a) Inicialização** (`initialization/initialization.py`, linha 20):

```python
phi[y, x] = -np.tanh((x - dist) / interface_width)
```

Onde `interface_width` é o parâmetro $W$ direto (sem fator 2 ou $\sqrt{2}$ adicional).

> **Nota:** o argumento do `tanh` é $(x - x_{\mathrm{int}})/W$, não
> $\sqrt{2}(x - x_{\mathrm{int}})/W$. Isso significa que `INTERFACE_WIDTH`
> aqui é o **fator de escala direto**.

**(b) Manutenção da espessura durante a simulação:**

A espessura é **automaticamente preservada** pela CH: o termo de difusão
$M\,\nabla^{2}\mu_c$ relaxa a interface de volta ao perfil tanh em escala
de tempo:

$$
\tau_{\mathrm{relax}} \;\sim\; \frac{W^{2}}{M\,\beta}.
$$

Se a advecção estirar localmente a interface, a difusão "redobra" ela.

**(c) Parâmetro de controle:** no JSON de configuração, define-se
diretamente `INTERFACE_WIDTH = W`. O código calcula $\beta$ e $\kappa$
automaticamente a partir de $\sigma$ e $W$.

### 9.4 Critérios práticos

| Critério | Valor | Razão |
|---|---|---|
| $W \geq 3$ l.u. | sempre | Nyquist no stencil de 5 pontos |
| $W \leq N_y/20$ | sempre | Manter interface "fina" |
| $k W \cdot 2\pi/N_y < 0.01$ | LSA pura | Limite de fase-field |

---

## 10. O que são as correntes espúrias e como minimizá-las?

### 10.1 Definição

**Correntes espúrias** (do inglês *spurious currents* ou *parasitic currents*)
são velocidades **não-físicas** que aparecem em torno de interfaces curvas
em métodos de campo de fase, mesmo quando o sistema deveria estar em
**equilíbrio estático**.

Em uma gota circular isolada (sem fluxo externo), em equilíbrio termodinâmico,
a velocidade física deveria ser **identicamente zero**. Mas tipicamente
observa-se:

$$
\mathbf{u}_{\mathrm{esp}}(\mathbf{x}) \;\neq\; 0 \quad \text{próximo à interface},
$$

com magnitude pequena mas não desprezível.

### 10.2 Origem

Vêm da **inconsistência discreta** entre:

1. A força capilar discretizada $\mathbf{F}_c = \mu_c\,\nabla\phi$.
2. A pressão de Laplace numericamente recuperada.

No contínuo, essas duas grandezas estão em equilíbrio perfeito. No
discreto, as diferenças finitas introduzem **erros de truncamento**
$O(\Delta x^{2})$ que não se cancelam exatamente, gerando uma força
residual e, consequentemente, um fluxo.

A magnitude escala como:

$$
\boxed{\;|\mathbf{u}_{\mathrm{esp}}| \;\sim\; \frac{\sigma}{\nu}\,\frac{(\Delta x)^{2}}{W^{2}}\;}
$$

### 10.3 Como minimizar

**(a) Interface bem resolvida** ($W \geq 3$ l.u.):
   - Stencils centrados (5 pontos) precisam de pelo menos 3 nós de
     transição para representar $\partial_x \phi$ sem ruído.
   - $W < 3$ viola Nyquist e cria oscilações $2\Delta x$ no $\phi$.

**(b) Stencils isotrópicos:**
   - Stencils anisotrópicos (só os 4 vizinhos cardeais sem pesos) introduzem
     **direção preferencial**, gerando correntes alinhadas com a malha.
   - **Solução melhor:** stencil isotrópico de **9 pontos** (E2):
     $$\nabla^{2}f \approx \tfrac{1}{6}\,[\text{cardeais}] + \tfrac{1}{12}\,[\text{diagonais}] - \tfrac{20}{12}\,f_{y,x}.$$
   - **Implementação atual:** stencil de 5 pontos (mais simples, correntes
     ligeiramente maiores).

**(c) Mobilidade $M$ adequada:**
   - $M$ muito pequena: relaxação termodinâmica lenta, interface se desforma.
   - $M$ muito grande: difusão excessiva, interface se espalha.
   - **Ótimo:** $M$ tal que $\mathrm{Pe} = u_{\max}W/(M\beta) \sim O(1)$.

**(d) Razão de viscosidade $M_{\mathrm{visc}}$ não extrema:**
   - $M_{\mathrm{visc}} = \nu_{\mathrm{out}}/\nu_{\mathrm{in}} \gtrsim 100$:
     amplifica correntes devido a desbalanço de momento.
   - **Mitigação:** média harmônica em vez de linear (já implementada com
     `visc_linear = False`).

**(e) Pressão "balanceada":**
   - Esquemas avançados (Lee & Liu 2010, Connington & Lee 2012) recuperam
     a pressão de Laplace consistente diretamente do potencial químico,
     **eliminando** correntes espúrias até erro de máquina.
   - **Não implementado neste código.**

**(f) Tempo de relaxação não extremo:**
   - $\tau$ muito perto de 0.5 ou muito acima de 2.0 piora correntes.

### 10.4 Diagnóstico no código

Para verificar correntes espúrias em um caso teste, rode uma simulação
**sem fluxo externo** ($u_{\mathrm{inlet}} = 0$) com uma gota circular
inicializada e meça:

$$
u_{\mathrm{esp}}^{\max} \;=\; \max_{(y,x)} \sqrt{u_x^{2} + u_y^{2}}.
$$

Critério aceitável: $u_{\mathrm{esp}}^{\max} \lesssim 10^{-4}\,c_s$.

### 10.5 Impacto na física

As correntes espúrias **contaminam** especialmente:

- Validações de Young–Laplace (introduzem ruído nos perfis de pressão).
- Crescimento de instabilidades em modos muito baixos (interfacial
  noise floor).
- Equilíbrios estáticos prolongados (drift de massa).

Em regimes **dinâmicos** com $u_{\mathrm{inlet}}$ não-nulo (caso de
Saffman–Taylor estudado aqui), as correntes espúrias são **dominadas**
pelo fluxo macroscópico e seu impacto é geralmente desprezível.

---

## Referências cruzadas

| Pergunta | Documento principal |
|---|---|
| 1. Guo | [`lbm/lbm.md`](../lbm/lbm.md) §1.5, §5.5 |
| 2. Young–Laplace | `post_process/valida_case.py` |
| 3. Curvatura/Amplitude | [`Relatorios/valida_lsa.md`](valida_lsa.md), [`Relatorios/resultado_curvatura_temporal.md`](resultado_curvatura_temporal.md) |
| 4. Viscosidade | [`lbm/lbm.md`](../lbm/lbm.md) §5.1 |
| 5. Cahn–Hilliard | [`cahn_hilliard/cahn_hilliard.md`](../cahn_hilliard/cahn_hilliard.md) |
| 6. Periodicidade | Todos os módulos |
| 7. Magnetismo | [`Magnetismo/poisson.md`](../Magnetismo/poisson.md) |
| 8. $k\xi$ | [`Relatorios/valida_lsa.md`](valida_lsa.md) §11 |
| 9. Espessura | [`cahn_hilliard/cahn_hilliard.md`](../cahn_hilliard/cahn_hilliard.md) §11.1 |
| 10. Correntes espúrias | (este documento) |
