# Documentação Técnica: Kernel LBM (`lbm.py`)

Este módulo implementa o **solver hidrodinâmico** baseado no Método de Lattice
Boltzmann (LBM) em rede D2Q9, com forçamento de Guo, viscosidade variável,
arrasto de Darcy–Brinkman e acoplamento bidirecional com os módulos de
Cahn–Hilliard (interface difusa) e magnetismo (tensor de Maxwell).

A função principal `lbm_step` é decorada com `@njit(parallel=True, cache=True)`
para JIT em CPU multi-core via Numba.

---

## 1. Fundamentos teóricos do LBM

### 1.1 Da equação de Boltzmann à hidrodinâmica

O LBM resolve uma **forma discretizada da equação de Boltzmann com aproximação
BGK** (Bhatnagar–Gross–Krook):

$$
\frac{\partial f}{\partial t} + \mathbf{c}\cdot\nabla f
\;=\; -\frac{1}{\tau}\bigl(f - f^{\mathrm{eq}}\bigr) + S,
$$

onde $f(\mathbf{x},\mathbf{c},t)$ é a função de distribuição de partículas,
$\tau$ é o tempo de relaxação ao equilíbrio $f^{\mathrm{eq}}$ e $S$ é o
termo fonte (forças externas).

Por **expansão de Chapman–Enskog**, esta equação recupera as equações de
Navier–Stokes incompressíveis no limite de baixo número de Mach
($\mathrm{Ma} \ll 1$), com viscosidade cinemática:

$$
\boxed{\;\nu \;=\; c_s^{2}\,\bigl(\tau - \tfrac{1}{2}\bigr)\,\Delta t,
\quad c_s^{2} = \tfrac{1}{3}\;}
$$

(unidades de rede: $\Delta x = \Delta t = 1 \Rightarrow \nu = (\tau-1/2)/3$).

### 1.2 Rede D2Q9

Discretização espacial em **9 velocidades** (1 estacionária + 4 cardeais + 4 diagonais):

$$
\mathbf{c}_i = \begin{cases}
(0,0) & i=0 \\
(\pm 1, 0),\,(0, \pm 1) & i=1\ldots 4 \\
(\pm 1, \pm 1) & i=5\ldots 8
\end{cases}
\qquad
w_i = \begin{cases}
4/9 & i=0 \\
1/9 & i=1\ldots 4 \\
1/36 & i=5\ldots 8
\end{cases}
$$

No código (linhas 4–6):

```python
W  = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
CX = [0,   1,   0,  -1,   0,   1,   -1,   -1,    1]
CY = [0,   0,   1,   0,  -1,   1,    1,   -1,   -1]
```

### 1.3 Momentos macroscópicos

$$
\rho \;=\; \sum_{i=0}^{8} f_i,\qquad
\rho\,\mathbf{u} \;=\; \sum_{i=0}^{8} f_i\,\mathbf{c}_i
\;+\; \frac{\Delta t}{2}\,\mathbf{F}.
$$

A correção $\tfrac{\Delta t}{2}\mathbf{F}$ aparece quando há **forças
externas** e é parte fundamental do esquema de Guo (§3.2).

### 1.4 Equilíbrio de Maxwell–Boltzmann discreto

$$
f_i^{\mathrm{eq}} \;=\; w_i\,\rho\,\left[
1 \;+\; 3(\mathbf{c}_i\!\cdot\!\mathbf{u})
\;+\; \tfrac{9}{2}(\mathbf{c}_i\!\cdot\!\mathbf{u})^{2}
\;-\; \tfrac{3}{2}\,|\mathbf{u}|^{2}\right].
$$

### 1.5 Termo fonte de Guo (forçamento)

Para incorporar forças $\mathbf{F}$ sem viés de segunda ordem:

$$
S_i \;=\; w_i\,\bigl(1 - \tfrac{\omega}{2}\bigr)\,
\left[\frac{3(\mathbf{c}_i - \mathbf{u})}{c_s^{2}}
\;+\; \frac{9(\mathbf{c}_i\!\cdot\!\mathbf{u})\,\mathbf{c}_i}{c_s^{4}}\right]
\!\cdot\!\mathbf{F},
$$

implementado como `Si = W[i]*(1 - 0.5*omega) * (3*term1 + 9*term2)`.

### 1.6 Equação discreta resolvida

$$
\boxed{\;f_i(\mathbf{x}+\mathbf{c}_i\Delta t,\,t+\Delta t)
\;=\; (1-\omega)\,f_i(\mathbf{x},t)
\;+\; \omega\,f_i^{\mathrm{eq}}
\;+\; S_i\;}
$$

— **colisão** (lado direito) seguida de **streaming** (lado esquerdo).

---

## 2. Assinatura da função `lbm_step`

```python
lbm_step(f_in, f_out, phi, psi, rho, u_x, u_y,
         chi, K_field, Fx, Fy,
         tau_in, tau_out, u_inlet,
         beta, kappa,
         is_periodic, Hx_fundo, Hy_fundo, visc_linear)
```

| Parâmetro | Shape / Tipo | Significado físico |
|---|---|---|
| `f_in`, `f_out` | $(N_y, N_x, 9)$ | Populações antes e depois (double-buffer). |
| `phi` | $(N_y, N_x)$ | Campo de fase Cahn–Hilliard, $\phi\in[-1,+1]$. |
| `psi` | $(N_y, N_x)$ | Potencial magnético escalar reduzido $\tilde\psi$. |
| `rho` | $(N_y, N_x)$ | Densidade local. |
| `u_x`, `u_y` | $(N_y, N_x)$ | Velocidade física macroscópica. |
| `chi` | $(N_y, N_x)$ | Suscetibilidade magnética local $\chi(\phi)$. |
| `K_field` | $(N_y, N_x)$ | Permeabilidade local do meio poroso. |
| `Fx`, `Fy` | $(N_y, N_x)$ | Buffers de força (capilar + magnética). |
| `tau_in/out` | escalar | Tempos de relaxação dos dois fluidos. |
| `u_inlet` | escalar | Velocidade prescrita na entrada (Dirichlet). |
| `beta`, `kappa` | escalar | Coeficientes do potencial Cahn–Hilliard. |
| `is_periodic` | bool | `True` $\Rightarrow$ periódico em $x$; `False` $\Rightarrow$ Zou–He. |
| `Hx_fundo`, `Hy_fundo` | escalar | Componentes do campo magnético de fundo $\mathbf{H}_0$. |
| `visc_linear` | bool | `True` $\Rightarrow$ média linear de $\nu$; `False` $\Rightarrow$ harmônica. |

> **Importante:** o eixo $y$ é **sempre periódico** no código atual
> (linha 100: `be_y = (y + CY[i] + ny) % ny`, sem condicional). O parâmetro
> `is_periodic` controla apenas o eixo $x$.

---

## 3. Estrutura do kernel — três etapas

```
┌────────────────────────────────────────────────────────────────────┐
│ ETAPA 1 — Cálculo de forças externas (Fx, Fy) em cada nó           │
│   • Korteweg (Cahn–Hilliard)                                       │
│   • Maxwell magnética: F = -½ |H|² ∇χ                              │
├────────────────────────────────────────────────────────────────────┤
│ ETAPA 2 — Colisão BGK + forçamento Guo + streaming                 │
│   • Viscosidade variável τ(φ)                                      │
│   • Arrasto de Darcy–Brinkman σ_drag(K, φ)                         │
│   • Velocidade física (correção de Guo + arrasto)                  │
│   • f_eq, S_i, atualização e propagação                            │
├────────────────────────────────────────────────────────────────────┤
│ ETAPA 3 — Condições de contorno em x (se is_periodic = False)      │
│   • Zou–He no inlet (x=0):   velocidade prescrita                  │
│   • Zou–He no outlet (x=Nx−1): densidade prescrita ρ=1             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. ETAPA 1 — Campo de forças externas (linhas 16–49)

Loop paralelo em $y$ que percorre **todos os nós internos em $x$**
(no caso não periódico, pula bordas).

### 4.1 Derivadas centradas

Para cada nó $(y, x)$ com vizinhos $(yp,ym,xp,xm)$:

$$
\partial_x f \;\approx\; \tfrac{1}{2}(f_{y,xp} - f_{y,xm}),
\qquad
\nabla^{2} f \;\approx\; f_{y,xp} + f_{y,xm} + f_{yp,x} + f_{ym,x} - 4\,f_{y,x}.
$$

> Stencil isotrópico de 5 pontos. Suficiente para Cahn–Hilliard, mas o
> projeto pode usar stencils E2 isotrópicos em outros módulos.

### 4.2 Força capilar (Korteweg / Cahn–Hilliard)

O potencial químico é:

$$
\mu_c \;=\; \underbrace{4\beta\,\phi(\phi^{2}-1)}_{\text{termo de bulk (poço duplo)}}
\;-\; \underbrace{\kappa\,\nabla^{2}\phi}_{\text{termo de gradiente}}.
$$

A força capilar é:

$$
\mathbf{F}_{c} \;=\; \mu_c\,\nabla\phi.
$$

No código (linhas 31–34):

```python
mu_c = 4.0*beta*phi[y,x]*(phi[y,x]**2 - 1.0) - kappa*lap_phi
Fx[y,x] = mu_c * dx_phi
Fy[y,x] = mu_c * dy_phi
```

A tensão superficial efetiva e a espessura de interface valem:

$$
\sigma \;=\; \frac{2\sqrt{2}}{3}\sqrt{\kappa\,\beta},\qquad
W \;\approx\; \sqrt{\kappa / (2\beta)}.
$$

### 4.3 Força magnética de Maxwell (linhas 36–49)

A formulação atual usa a **decomposição de Helmholtz**:

$$
\mathbf{H}_{\mathrm{total}} \;=\; \mathbf{H}_{0} \;-\; \nabla\tilde\psi,
$$

onde $\mathbf{H}_0 = (H_{x,\text{fundo}},\,H_{y,\text{fundo}})$ é o campo de
fundo e $\tilde\psi$ é o **potencial perturbador** resolvido pela equação de
Poisson em outro módulo.

```python
hx = Hx_fundo - 0.5*(psi[y,xp] - psi[y,xm])
hy = Hy_fundo - 0.5*(psi[yp,x] - psi[ym,x])
H_sq = hx**2 + hy**2
```

A **força ponderomotriz** é o divergente do tensor de Maxwell, que, para
suscetibilidade linear $\chi$ variável, se reduz a:

$$
\boxed{\;\mathbf{F}_{m} \;=\; -\,\tfrac{1}{2}\,|\mathbf{H}|^{2}\,\nabla\chi\;}
$$

(forma equivalente a $\chi\,\nabla(\tfrac{1}{2}|\mathbf{H}|^{2})$ a menos de
um gradiente puro que é absorvido pela pressão).

```python
dchi_dx = 0.5*(chi[y,xp] - chi[y,xm])
dchi_dy = 0.5*(chi[yp,x] - chi[ym,x])
Fx[y,x] += -0.5 * H_sq * dchi_dx
Fy[y,x] += -0.5 * H_sq * dchi_dy
```

**Conceito-chave:** a força só é não nula onde **$\nabla\chi \neq 0$**,
isto é, **na interface** (já que $\chi$ é constante em cada fase). Isso
significa que o magnetismo atua como uma **tensão interfacial extra** quando
o campo é normal à interface.

---

## 5. ETAPA 2 — Colisão + Forçamento + Streaming (linhas 55–107)

Loop paralelo em $y$. Cada nó executa, sequencialmente:

### 5.1 Viscosidade variável $\nu(\phi)$ (linhas 57–65)

Define a fração volumétrica de cada fluido:

$$
S_{\mathrm{inv}} = \frac{\phi+1}{2},\qquad S_{\mathrm{res}} = 1 - S_{\mathrm{inv}}.
$$

Duas opções controladas por `visc_linear`:

**(a) Linear** (`visc_linear=True`):
$$
\nu_{\mathrm{eff}} = S_{\mathrm{inv}}\,\nu_{\mathrm{in}} + S_{\mathrm{res}}\,\nu_{\mathrm{out}}.
$$

**(b) Harmônica** (`visc_linear=False`, padrão para grandes razões de viscosidade):
$$
\frac{1}{\nu_{\mathrm{eff}}} \;=\; \frac{S_{\mathrm{inv}}}{\nu_{\mathrm{in}}} + \frac{S_{\mathrm{res}}}{\nu_{\mathrm{out}}}.
$$

A média harmônica preserva melhor a continuidade da tensão de cisalhamento
na interface (importante quando $\nu_{\mathrm{out}}/\nu_{\mathrm{in}} \gg 1$).

Daí:
$$
\tau = 3\,\nu_{\mathrm{eff}} + \tfrac{1}{2},\qquad \omega = 1/\tau.
$$

### 5.2 Arrasto de Darcy–Brinkman (linhas 67–70)

Para escoamento em meio poroso, adiciona-se uma força de arrasto
$\mathbf{F}_{D} = -\sigma_{\mathrm{drag}}\,\rho\,\mathbf{u}$. Em escoamento
bifásico com permeabilidade $K$ e mobilidades relativas $k_r^{(j)}$:

$$
\sigma_{\mathrm{drag}} \;=\; \frac{1}{K\,\lambda_{\mathrm{tot}}},\qquad
\lambda_{\mathrm{tot}} \;=\; \frac{k_r^{\mathrm{inv}}}{\nu_{\mathrm{in}}} \;+\; \frac{k_r^{\mathrm{res}}}{\nu_{\mathrm{out}}},
$$

com $k_r^{(j)} = \max(S^{(j)}, 10^{-6})$ (proteção contra divisão por zero
em regiões puras).

> Quando $K \to \infty$ (sem meio poroso), $\sigma_{\mathrm{drag}} \to 0$ e o
> termo de arrasto desaparece, recuperando Navier–Stokes puro.

### 5.3 Momentos brutos e correção de Guo (linhas 72–82)

Soma das populações:
$$
\rho = \sum_i f_i,\qquad
\rho\,\mathbf{u}^{*} = \sum_i f_i\,\mathbf{c}_i + \tfrac{1}{2}\,\mathbf{F}.
$$

```python
ux_star = (ux_l + 0.5*Fx[y,x]) / rho_l
uy_star = (uy_l + 0.5*Fy[y,x]) / rho_l
```

**Correção do arrasto** (Brinkman implícito) — a velocidade física é
obtida invertendo $\mathbf{u}^{*} = \mathbf{u}_{\mathrm{phys}} + \tfrac{1}{2}\sigma_{\mathrm{drag}}\mathbf{u}_{\mathrm{phys}}$:

$$
\boxed{\;\mathbf{u}_{\mathrm{phys}} \;=\; \frac{\mathbf{u}^{*}}{1 + \tfrac{1}{2}\sigma_{\mathrm{drag}}}\;}
$$

```python
ux_phys = ux_star / (1.0 + 0.5*sigma_drag)
uy_phys = uy_star / (1.0 + 0.5*sigma_drag)
```

Esta é a velocidade **armazenada em `u_x, u_y`** e usada para advectar o
campo de fase no módulo Cahn–Hilliard.

### 5.4 Força total efetiva (linhas 87–88)

A força que entra no termo $S_i$ inclui o arrasto explícito:

$$
\mathbf{F}_{\mathrm{tot}} = \mathbf{F}_{c} + \mathbf{F}_{m} - \sigma_{\mathrm{drag}}\,\rho\,\mathbf{u}_{\mathrm{phys}}.
$$

### 5.5 Equilíbrio e termo fonte (linhas 90–96)

Para cada direção $i = 0\ldots 8$:

$$
c_u = \mathbf{c}_i\!\cdot\!\mathbf{u}_{\mathrm{phys}},\qquad
f_i^{\mathrm{eq}} = w_i\,\rho\,(1 + 3c_u + \tfrac{9}{2}c_u^{2} - \tfrac{3}{2}|\mathbf{u}|^{2}),
$$
$$
S_i = w_i\,(1 - \tfrac{\omega}{2})\,(3\,T_1 + 9\,T_2),
$$
$$
T_1 = (\mathbf{c}_i - \mathbf{u})\!\cdot\!\mathbf{F}_{\mathrm{tot}},\quad
T_2 = (\mathbf{c}_i\!\cdot\!\mathbf{u})(\mathbf{c}_i\!\cdot\!\mathbf{F}_{\mathrm{tot}}).
$$

### 5.6 Colisão + streaming combinados (linhas 98–107)

```python
f_val = f_in[y,x,i]*(1 - omega) + omega*feq + Si

be_y = (y + CY[i] + ny) % ny      # ← Y SEMPRE periódico
if is_periodic:
    be_x = (x + CX[i] + nx) % nx
    f_out[be_y, be_x, i] = f_val
else:
    be_x = x + CX[i]
    if 0 <= be_x < nx:
        f_out[be_y, be_x, i] = f_val
```

> **Detalhes importantes:**
> - O streaming é feito **dentro do mesmo loop da colisão** (sem buffer intermediário entre as duas operações) — escreve direto no destino `f_out[be_y, be_x, i]`.
> - Em $y$, sempre `mod ny` $\Rightarrow$ topo e fundo são identificados (canal infinitamente periódico em $y$, geometria do tipo Hele-Shaw).
> - Em $x$ não periódico, populações que tentariam sair do domínio são simplesmente **descartadas** — o que **falta** é reconstruído pela condição de contorno da Etapa 3.

---

## 6. ETAPA 3 — Condições de contorno em $x$ (linhas 110–128)

Executada **somente se `is_periodic=False`**. Implementa o esquema clássico
de **Zou–He** (1997), que reconstrói as populações desconhecidas impondo
momentos prescritos.

### 6.1 Premissas geométricas

Após o streaming, na coluna $x = N_x - 1$ (parede de saída) faltam as
populações vindas de $x = N_x$, ou seja, as que apontam para $-\hat x$:
**$i \in \{3, 6, 7\}$**. Analogamente, em $x = 0$ faltam $i \in \{1, 5, 8\}$
(direções $+\hat x$).

```
                 (5)  (2)  (6)
                   ↖   ↑   ↗
              (1) ← (0) → (3)
                   ↙   ↓   ↘
                 (8)  (4)  (7)
```

### 6.2 Outlet — pressão prescrita $\rho_{\mathrm{out}} = 1$ (linhas 112–120)

A regra de Zou–He usa a equação de $\rho$ e $\rho u_x$ para isolar $u_x$:

$$
u_{x,\mathrm{out}} \;=\; -1 \;+\; \frac{1}{\rho_{\mathrm{out}}}\bigl[
f_0 + f_2 + f_4 \;+\; 2(f_1 + f_5 + f_8)\bigr],
$$

(usando que $f_3 + f_6 + f_7$ são as desconhecidas e $f_3 = f_1 + \ldots$
ao satisfazer não-equilíbrio espelhado). Em seguida:

$$
\begin{aligned}
f_3 &= f_1 - \tfrac{2}{3}\rho_{\mathrm{out}}\,u_{x,\mathrm{out}},\\
f_6 &= f_8 - \tfrac{1}{2}(f_2 - f_4) - \tfrac{1}{6}\rho_{\mathrm{out}}\,u_{x,\mathrm{out}},\\
f_7 &= f_5 + \tfrac{1}{2}(f_2 - f_4) - \tfrac{1}{6}\rho_{\mathrm{out}}\,u_{x,\mathrm{out}}.
\end{aligned}
$$

> **Sinal de $u_x$:** como $\rho_{\mathrm{out}} = 1$ e o fluido sai pelo
> outlet, $u_{x,\mathrm{out}}$ resulta positivo. A diferença $f_2 - f_4$
> captura assimetria transversal (cisalhamento), atribuindo-a corretamente
> às duas diagonais.

### 6.3 Inlet — velocidade prescrita $u_{\mathrm{inlet}}$ (linhas 122–128)

Conhecendo $u_x = u_{\mathrm{inlet}}$ na borda esquerda, a densidade é
isolada de $\rho = \sum f_i$ usando as populações conhecidas:

$$
\rho_{\mathrm{in}} \;=\; \frac{1}{1 - u_{\mathrm{inlet}}}\bigl[
f_0 + f_2 + f_4 \;+\; 2(f_3 + f_6 + f_7)\bigr].
$$

E as populações desconhecidas:

$$
\begin{aligned}
f_1 &= f_3 + \tfrac{2}{3}\rho_{\mathrm{in}}\,u_{\mathrm{inlet}},\\
f_5 &= f_7 - \tfrac{1}{2}(f_2 - f_4) + \tfrac{1}{6}\rho_{\mathrm{in}}\,u_{\mathrm{inlet}},\\
f_8 &= f_6 + \tfrac{1}{2}(f_2 - f_4) + \tfrac{1}{6}\rho_{\mathrm{in}}\,u_{\mathrm{inlet}}.
\end{aligned}
$$

### 6.4 Resumo comparativo

| Borda | Tipo Zou–He | Quantidade fixada | Quantidade calculada |
|---|---|---|---|
| $x = 0$ (inlet) | Dirichlet em $\mathbf{u}$ | $u_x = u_{\mathrm{inlet}},\,u_y = 0$ | $\rho_{\mathrm{in}}$ |
| $x = N_x - 1$ (outlet) | Dirichlet em $\rho$ | $\rho_{\mathrm{out}} = 1$ | $u_{x,\mathrm{out}}$ |
| $y = 0,\,y = N_y - 1$ | Periódico (intrínseco) | — | — |

### 6.5 Por que Zou–He (e não bounce-back)?

| Esquema | Erro local | Adequado para |
|---|---|---|
| **Bounce-back** (half-way) | $O(\Delta x^{2})$ em paredes sólidas | No-slip em sólidos. |
| **Zou–He** | $O(\Delta x^{2})$ exato em momentos | Inlet/outlet com velocidade ou pressão prescrita. |
| **Equilíbrio puro** ($f = f^{\mathrm{eq}}$) | $O(\Delta x)$ | Aproximação grosseira, descartada aqui. |

O Zou–He garante que **$\rho$ e $\rho\mathbf{u}$ na fronteira sejam exatos**,
o que é crítico para a injeção em problemas de Saffman–Taylor onde a vazão
controla o regime de instabilidade.

---

## 7. Acoplamento com os outros módulos

```
                        ┌─────────────────┐
                        │   lbm_step      │
                        │  (este módulo)  │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        u_x, u_y           Fx, Fy escritos    rho, populações f
              │                  ▲                  
              ▼                  │                  
     ┌─────────────────┐         │           
     │  Cahn–Hilliard  │         │           
     │   (advecta φ)   │         │           
     └────────┬────────┘         │           
              │                  │           
              ▼                  │           
            φ novo               │           
              │                  │           
              ▼                  │           
     ┌─────────────────┐         │           
     │   χ(φ), K(φ)    │─────────┘           
     │  acopladores    │  (entram no próximo passo)
     └─────────────────┘
                ▲
                │
     ┌─────────────────┐
     │   Poisson ψ̃    │  ←  resolvido por outro módulo
     └─────────────────┘
```

### 7.1 Fluxo bidirecional

- **LBM → CH:** velocidade $\mathbf{u}_{\mathrm{phys}}$ advecta $\phi$.
- **CH → LBM:** $\phi$ define $\nu_{\mathrm{eff}}$, $\chi$, $K$ e as forças.
- **Poisson → LBM:** $\tilde\psi$ define a perturbação magnética $\mathbf{H}_{\mathrm{total}}$.

---

## 8. Considerações numéricas

### 8.1 Estabilidade

| Critério | Razão |
|---|---|
| $\tau > 0.5$ (estrito) | $\nu > 0$. |
| $\tau < 2$ (recomendado) | Acima disso, BGK tem dispersão e desvios de $f^{\mathrm{eq}}$. |
| $\mathrm{Ma} = u/c_s \lesssim 0.1$ | Hipótese de Mach baixo para Chapman–Enskog. |
| $|\mathbf{F}|\,\Delta t / \rho \ll c_s$ | Forçamento de Guo válido. |
| $W \geq 3$ l.u. | Resolução da interface difusa (Nyquist em D2Q9). |

### 8.2 Custo computacional

- Memória: $11 \times N_y \times N_x$ doubles principais (`f_in`, `f_out`, mais campos auxiliares).
- Operações por nó: $\sim 130$ FLOPs por timestep.
- Paralelização: `prange` sobre $y$ (Numba) — *embarrassingly parallel*
  graças ao double-buffer (`f_in` $\to$ `f_out`).

### 8.3 Por que `f_in` e `f_out` separados?

O streaming `f_out[be_y, be_x, i] = f_val` **escreve em posições adjacentes**,
o que criaria *race condition* se feito in-place no mesmo array. O
double-buffer evita isso e permite paralelismo trivial. A troca
`f_in, f_out = f_out, f_in` ocorre no driver fora deste módulo.

---

## 9. Resumo executivo

| Componente | Implementação | Local |
|---|---|---|
| Rede | D2Q9, $w_i$ e $\mathbf{c}_i$ explícitos | Linhas 4–6 |
| Equilíbrio | Maxwell expandido até $\mathcal{O}(u^{2})$ | Linha 92 |
| Forçamento | Guo (1 − ω/2) | Linhas 94–96 |
| Força capilar | $\mu_c\,\nabla\phi$, Cahn–Hilliard | Linhas 28–34 |
| Força magnética | $-\tfrac{1}{2}|H|^{2}\,\nabla\chi$, Maxwell | Linhas 36–49 |
| Meio poroso | Darcy–Brinkman com mobilidades relativas | Linhas 67–70 |
| Viscosidade | Linear ou harmônica em $\phi$ | Linhas 59–65 |
| Streaming | Pull combinado com colisão, double-buffer | Linhas 100–107 |
| BC inlet (x=0) | Zou–He com velocidade prescrita | Linhas 122–128 |
| BC outlet (x=Nx−1) | Zou–He com pressão $\rho = 1$ | Linhas 112–120 |
| BC top/bottom | **Periódico** (sempre) | Linha 100 |
