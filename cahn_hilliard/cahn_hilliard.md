# Documentação Técnica: Módulo Cahn–Hilliard (`cahn_hilliard.py`)

Este módulo implementa o **solver de interface difusa** que evolui o campo de
fase $\phi(\mathbf{x}, t)$ segundo a **equação de Cahn–Hilliard** com termo
advectivo. É o módulo responsável pela manutenção da interface entre os
dois fluidos e pela tensão superficial implícita, sendo acoplado ao kernel
LBM via a velocidade física $\mathbf{u}_{\mathrm{phys}}$ e via o potencial
químico $\mu_c$.

A função `cahn_hilliard_substep` é decorada com `@njit(parallel=True, cache=True)`
para execução paralela em CPU multi-core.

---

## 1. Fundamentos teóricos

### 1.1 Modelo de interface difusa

Em vez de tratar a interface como uma superfície geométrica de espessura
zero (modelo *sharp interface*), o método de campo de fase representa a
interface como uma **camada difusa** de espessura $W$, onde o parâmetro de
ordem $\phi$ varia suavemente entre dois valores de bulk:

$$
\phi(\mathbf{x}, t) = \begin{cases}
+1 & \text{fluido invasor (puro)} \\
-1 & \text{fluido residente (puro)} \\
\in (-1, +1) & \text{região interfacial}
\end{cases}
$$

A interface geométrica é definida pelo *level-set* $\phi = 0$.

### 1.2 Funcional de energia livre de Ginzburg–Landau

A termodinâmica do sistema é governada pelo funcional:

$$
\boxed{\;
\mathcal{F}[\phi] \;=\; \int_{\Omega}\!\!\left[
\underbrace{\beta\,(\phi^{2} - 1)^{2}}_{\text{poço duplo (bulk)}}
\;+\;
\underbrace{\tfrac{\kappa}{2}\,|\nabla\phi|^{2}}_{\text{penalidade de gradiente}}
\right] d\Omega \;}
$$

- **Termo de bulk** $\beta(\phi^2 - 1)^2$: poço duplo com mínimos em
  $\phi = \pm 1$, energeticamente desfavorável misturar as fases.
- **Termo de gradiente** $\tfrac{\kappa}{2}|\nabla\phi|^2$: penaliza
  variações abruptas; controla a **espessura de interface**.

### 1.3 Potencial químico

Definido como a derivada funcional da energia livre:

$$
\mu_c \;=\; \frac{\delta\mathcal{F}}{\delta\phi}
\;=\; \underbrace{4\beta\,\phi(\phi^{2} - 1)}_{\partial f_{\text{bulk}}/\partial\phi}
\;-\; \underbrace{\kappa\,\nabla^{2}\phi}_{\text{difusional}}.
$$

A interface plana em equilíbrio tem a solução analítica:

$$
\phi_{\mathrm{eq}}(z) \;=\; \tanh\!\left(\frac{z\sqrt{2}}{W}\right),\qquad
W = \sqrt{\frac{\kappa}{2\beta}}.
$$

E a **tensão superficial efetiva** é:

$$
\sigma \;=\; \int_{-\infty}^{+\infty}\!\!\kappa\,(\partial_z\phi_{\mathrm{eq}})^{2}\,dz
\;=\; \frac{2\sqrt{2}}{3}\sqrt{\kappa\,\beta}.
$$

> **Parâmetros calibráveis:** $\beta$ e $\kappa$ permitem ajustar
> independentemente $\sigma$ e $W$.

### 1.4 Equação de Cahn–Hilliard advectiva

A dinâmica conservativa do campo de fase em um fluido em movimento é:

$$
\boxed{\;
\frac{\partial\phi}{\partial t} \;+\; \mathbf{u}\!\cdot\!\nabla\phi
\;=\; M\,\nabla^{2}\mu_c
\;}
$$

- **Advecção** $\mathbf{u}\!\cdot\!\nabla\phi$: transporte pelo escoamento
  vindo do LBM.
- **Difusão de $\mu_c$** $M\,\nabla^{2}\mu_c$: relaxação termodinâmica para
  o equilíbrio energético, mantendo a forma tanh da interface contra
  estiramento numérico.

O parâmetro $M$ é a **mobilidade** (`m_mobility` no código). Valores
típicos: $M \in [10^{-3}, 10^{-1}]$.

### 1.5 Conservatividade

Note que a CH é da forma $\partial_t \phi = -\nabla\!\cdot\!\mathbf{J}$ com
$\mathbf{J} = \phi\,\mathbf{u} - M\,\nabla\mu_c$. Logo:

$$
\frac{d}{dt}\int_{\Omega}\phi\,d\Omega \;=\; -\oint_{\partial\Omega}\!\mathbf{J}\!\cdot\!\hat{\mathbf{n}}\,dS,
$$

e em domínio fechado a massa de cada fase é **conservada exatamente** (em
contraste com modelos de Allen–Cahn).

### 1.6 Ordem da equação

CH é uma EDP **de quarta ordem** em $\phi$ (Laplaciano de $\mu_c$, que já
contém um Laplaciano de $\phi$). Numericamente isso é tratado decompondo em
duas equações de segunda ordem:

$$
\begin{cases}
\mu_c \;=\; 4\beta\,\phi(\phi^2 - 1) - \kappa\,\nabla^2\phi & \text{(passo 1)}\\
\partial_t \phi \;=\; -\mathbf{u}\!\cdot\!\nabla\phi + M\,\nabla^2\mu_c & \text{(passo 2)}
\end{cases}
$$

Esta é exatamente a estrutura do `cahn_hilliard_substep`.

---

## 2. Assinatura da função

```python
cahn_hilliard_substep(phi_in, phi_out, mu,
                      u_x, u_y,
                      beta, kappa, dt_ch, m_mobility,
                      is_periodic)
```

| Parâmetro | Shape / Tipo | Significado |
|---|---|---|
| `phi_in` | $(N_y, N_x)$ | Campo de fase no início do substep. |
| `phi_out` | $(N_y, N_x)$ | Buffer de saída (double-buffer com `phi_in`). |
| `mu` | $(N_y, N_x)$ | Buffer para o potencial químico $\mu_c$. |
| `u_x`, `u_y` | $(N_y, N_x)$ | Velocidade física vinda do LBM. |
| `beta` | escalar | Coeficiente de bulk do poço duplo. |
| `kappa` | escalar | Coeficiente de gradiente. |
| `dt_ch` | escalar | Passo de tempo do substep CH ($\Delta t_{\mathrm{CH}}$). |
| `m_mobility` | escalar | Mobilidade $M$. |
| `is_periodic` | bool | `True` $\Rightarrow$ periódico em $x$; `False` $\Rightarrow$ injeção. |

> **Convenção de eixos** (igual ao LBM): $y$ é **sempre periódico**.
> O parâmetro `is_periodic` controla apenas o eixo $x$.

---

## 3. Estrutura do substep — 5 fases sequenciais

```
┌─────────────────────────────────────────────────────────────────┐
│ FASE 1 — Aplicar BCs em phi_in (apenas em x, se não periódico)  │
├─────────────────────────────────────────────────────────────────┤
│ FASE 2 — Computar mu = 4β·φ(φ²−1) − κ·∇²φ                       │
├─────────────────────────────────────────────────────────────────┤
│ FASE 3 — Aplicar BCs em mu (Neumann em x)                       │
├─────────────────────────────────────────────────────────────────┤
│ FASE 4 — Evoluir: φ_new = φ + Δt·(M·∇²μ − u·∇φ), clamp [-1,+1]  │
├─────────────────────────────────────────────────────────────────┤
│ FASE 5 — Reaplicar BC em phi_out (consistência da injeção)      │
└─────────────────────────────────────────────────────────────────┘
```

Por que **dois passos** ($\mu$ separado da evolução)? Porque o termo
$M\,\nabla^2\mu_c$ exige que $\mu_c$ esteja **disponível em todos os nós
vizinhos** antes do cálculo do segundo Laplaciano. Sem armazenar $\mu_c$,
seria impossível avaliar o Laplaciano final localmente.

---

## 4. FASE 1 — Condições de contorno em `phi_in` (linhas 10–13)

Executada apenas se `is_periodic = False`:

```python
phi_in[y, 0]    = 1.0          # Dirichlet: fluido invasor injetado
phi_in[y, -1]   = phi_in[y, -2]  # Neumann: saída livre
```

- **Inlet ($x = 0$):** $\phi = +1$ (fluido invasor puro, consistente com
  a injeção de velocidade no LBM).
- **Outlet ($x = N_x-1$):** $\partial_x \phi = 0$ (Neumann), permite que
  a interface saia do domínio sem refletir.

> **Eixo $y$:** sempre periódico, herdado do uso de `(y \pm 1) % ny` na
> Fase 2 (linhas 17–18).

---

## 5. FASE 2 — Potencial químico $\mu_c$ (linhas 16–29)

### 5.1 Stencil isotrópico de 5 pontos

Para o Laplaciano:

$$
\nabla^{2}\phi(y,x) \;\approx\; \phi_{y,xp} + \phi_{y,xm} + \phi_{yp,x} + \phi_{ym,x} - 4\,\phi_{y,x}.
$$

### 5.2 Cálculo de $\mu_c$

$$
\mu_c(y, x) \;=\; 4\beta\,\phi(y,x)\,\bigl[\phi(y,x)^{2} - 1\bigr] \;-\; \kappa\,\nabla^{2}\phi(y,x).
$$

```python
lap_phi = (phi_in[y,xp] + phi_in[y,xm] +
           phi_in[yp,x] + phi_in[ym,x] - 4.0*phi_in[y,x])
mu[y,x] = 4.0*beta*phi_in[y,x]*(phi_in[y,x]**2 - 1.0) - kappa*lap_phi
```

> **Sinal físico de $\mu_c$:**
> - $\mu_c > 0$ em regiões com excesso de fase invasora $\Rightarrow$
>   difusão expulsa $\phi$ dessa região.
> - $\mu_c < 0$ no caso inverso.
> - Na interface plana de equilíbrio, $\mu_c \equiv 0$.

---

## 6. FASE 3 — Condições de contorno em `mu` (linhas 32–35)

```python
mu[y, 0]  = mu[y, 1]
mu[y, -1] = mu[y, -2]
```

**Neumann em ambos os lados** ($\partial_x \mu_c = 0$). Isso significa que
**não há fluxo difusivo de $\phi$ através das fronteiras de $x$** — o que
é fisicamente consistente:

- No inlet, a fase é mantida por Dirichlet em $\phi$; não queremos um fluxo
  difusivo adicional contaminando a injeção.
- No outlet, o fluxo de saída é puramente advectivo
  ($\phi$ é "varrido" pela velocidade $\mathbf{u}$).

---

## 7. FASE 4 — Evolução temporal (linhas 38–61)

### 7.1 Esquema Euler explícito

$$
\phi^{n+1}(y,x) \;=\; \phi^{n}(y,x) \;+\; \Delta t_{\mathrm{CH}}\,
\Bigl[M\,\nabla^{2}\mu_c - \bigl(u_x\,\partial_x\phi + u_y\,\partial_y\phi\bigr)\Bigr].
$$

### 7.2 Discretização espacial

**Laplaciano de $\mu_c$** (mesmo stencil 5-pontos):

```python
lap_mu = (mu[y,xp] + mu[y,xm] + mu[yp,x] + mu[ym,x] - 4.0*mu[y,x])
```

**Advecção** com diferenças centradas:

```python
dphi_dx = 0.5*(phi_in[y,xp] - phi_in[y,xm])
dphi_dy = 0.5*(phi_in[yp,x] - phi_in[ym,x])
advec = u_x[y,x]*dphi_dx + u_y[y,x]*dphi_dy
```

> **Observação:** diferenças centradas são $O(\Delta x^2)$ mas podem
> oscilar em frentes íngremes em alta Péclet. A combinação com a difusão
> de $\mu_c$ tipicamente estabiliza, mas para razões $\mathrm{Pe} \gg 1$
> pode-se considerar esquemas upwind.

### 7.3 Atualização e *clamp*

```python
novo_phi = phi_in[y,x] + dt_ch*(m_mobility*lap_mu - advec)
if novo_phi > 1.0:  novo_phi = 1.0
elif novo_phi < -1.0: novo_phi = -1.0
phi_out[y,x] = novo_phi
```

O **clamp** $\phi \in [-1, +1]$ é uma **proteção numérica contra
overshoots** decorrentes do potencial quártico, que pode amplificar
excursões fora do intervalo físico, gerando instabilidade. Em soluções bem
resolvidas (sigma e mobilidade adequados), o clamp raramente é acionado.

> **Trade-off:** o clamp introduz uma pequena perda local de massa, mas
> evita explosões numéricas que destruiriam a simulação inteira.

---

## 8. FASE 5 — Reaplicação de BC em `phi_out` (linhas 64–67)

```python
phi_out[y, 0]    = 1.0
phi_out[y, -1]   = phi_out[y, -2]
```

Necessária porque o loop da Fase 4 pula as colunas $x=0$ e $x=N_x-1$
(linha 46: `if x == 0 or x == nx-1: continue`). Sem essa reaplicação,
`phi_out` ficaria com **lixo não inicializado** nessas colunas, que seria
propagado no próximo substep.

---

## 9. Sub-stepping: por que `dt_ch ≠ dt_lbm`?

O critério de estabilidade do esquema explícito para CH é
**mais restritivo** que o do LBM:

$$
\Delta t_{\mathrm{CH}} \;\lesssim\; \frac{(\Delta x)^{4}}{4\,M\,\kappa}
$$

(devido ao termo $M\,\nabla^4 \phi$ — a CFL explícita escala com $\Delta x^4$).

Em contraste, o LBM tem CFL convectiva $\Delta t \lesssim \Delta x / c_s$,
muito mais permissiva.

**Solução:** o driver da simulação chama `cahn_hilliard_substep` várias
vezes por passo do LBM:

```python
for _ in range(N_SUBSTEPS_CH):
    cahn_hilliard_substep(phi, phi_new, mu, u_x, u_y,
                          beta, kappa, dt_ch=DT_LBM/N_SUBSTEPS_CH, ...)
    phi, phi_new = phi_new, phi
```

---

## 10. Acoplamento com o resto da simulação

```
                       ┌──────────────────────┐
                       │       LBM step       │
                       │  produz u_x, u_y     │
                       └──────────┬───────────┘
                                  │ u_x, u_y
                                  ▼
                       ┌──────────────────────┐
                       │  cahn_hilliard       │
                       │  substep × N_SUB     │ ← este módulo
                       │  produz φ_new        │
                       └──────────┬───────────┘
                                  │ φ_new
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        χ(φ): magnético      K(φ): poroso       ν(φ): viscosidade
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ▼
                       (entram no próximo LBM step)
```

### 10.1 Quem produz $\mathbf{F}_c$?

> **Atenção:** A **força capilar** $\mathbf{F}_c = \mu_c\,\nabla\phi$ é
> calculada **dentro do `lbm_step`** (linhas 28–34 do `lbm.py`), não aqui.
> O módulo CH só evolui o $\phi$. O $\mu_c$ calculado aqui é descartado
> ao fim do substep — o LBM recalcula seu próprio $\mu_c$ local.

Essa duplicação é deliberada: cada módulo precisa de $\mu_c$ avaliado no
seu próprio campo $\phi$ no instante apropriado.

---

## 11. Considerações numéricas

### 11.1 Calibração de parâmetros

Dado o desejo de uma tensão $\sigma$ e espessura $W$:

$$
\beta \;=\; \frac{3\sqrt{2}\,\sigma}{4\,W},\qquad
\kappa \;=\; \frac{3\sqrt{2}\,\sigma\,W}{4}.
$$

### 11.2 Critérios de qualidade

| Critério | Valor recomendado | Razão |
|---|---|---|
| $W$ | $\geq 3$ l.u. | Nyquist no stencil de 5 pontos. |
| $W$ | $\leq N_y / 20$ | Manter interface "fina" frente ao domínio. |
| $\Delta t_{\mathrm{CH}}\,M\,\kappa/(\Delta x)^4$ | $\leq 0.1$ | CFL difusiva de 4ª ordem. |
| $\mathrm{Pe} = u_{\max}\,W/M\beta$ | $\sim O(1)$ | Equilíbrio entre advecção e relaxação. |
| $k\xi = 2\pi m W/N_y$ | $< 0.01$ | Erro de fase-field $< 5\%$ em modos lineares. |

### 11.3 Paralelização

`prange` sobre $y$ em todas as fases. Como o stencil é local (5 pontos),
não há dependência de dados entre linhas. O custo por nó é $\sim 25$ FLOPs
× 2 (uma vez para $\mu_c$, outra para a evolução).

### 11.4 Por que double-buffer (`phi_in`, `phi_out`)?

Na Fase 4, a atualização de cada $\phi(y,x)$ depende de $\phi$ nos vizinhos
**no mesmo instante**. Sem buffer separado, atualizações em $x-1$ feitas
antes do cálculo em $x$ corromperiam o stencil. A troca
`phi_in, phi_out = phi_out, phi_in` ocorre no driver externo.

---

## 12. Resumo executivo

| Componente | Implementação | Local |
|---|---|---|
| Funcional | Ginzburg–Landau $\beta(\phi^2-1)^2 + \tfrac{\kappa}{2}|\nabla\phi|^2$ | — |
| Potencial químico | $4\beta\phi(\phi^2-1) - \kappa\nabla^2\phi$ | Linha 29 |
| Stencil Laplaciano | 5 pontos isotrópico | Linhas 27–28, 49 |
| Evolução | Euler explícito $\phi^{n+1} = \phi^n + \Delta t(M\nabla^2\mu - \mathbf{u}\!\cdot\!\nabla\phi)$ | Linha 54 |
| Advecção | Diferenças centradas $O(\Delta x^2)$ | Linhas 50–53 |
| Clamp | $\phi \in [-1, +1]$ | Linhas 56–59 |
| BC inlet ($x=0$) | Dirichlet $\phi = +1$ | Linhas 12, 66 |
| BC outlet ($x=N_x-1$) | Neumann $\partial_x \phi = 0$ | Linhas 13, 67 |
| BC em $\mu$ | Neumann em ambos os lados ($x$) | Linhas 34–35 |
| BC top/bottom | **Periódico** (sempre) | Linhas 17–18, 39–40 |
| Sub-stepping | Múltiplas chamadas por step LBM | Externo (driver) |
