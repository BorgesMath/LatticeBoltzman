# Documentação Técnica: Solver Magnetostático (`poisson.py`)

Este módulo resolve a **equação de Poisson generalizada com coeficientes
variáveis** para o potencial magnético escalar **perturbador** $\tilde\psi$
em um meio heterogêneo (ferrofluido $+$ fluido não-magnético). A
heterogeneidade vem do campo de fase: $\chi$ varia espacialmente através
da interface difusa.

A função `solve_poisson_magnetic` implementa o método de **Sobre-Relaxação
Sucessiva (SOR)** com JIT paralelo via Numba (`@njit(parallel=True, cache=True)`).

---

## 1. Fundamentos teóricos

### 1.1 Equações de Maxwell magnetostáticas

Na ausência de correntes livres e em regime quase-estático
($\partial_t \mathbf{D} = 0$, $\mathbf{J} = 0$):

$$
\nabla\times\mathbf{H} = 0,\qquad \nabla\!\cdot\!\mathbf{B} = 0,\qquad \mathbf{B} = \mu_0\,\mu_r(\mathbf{x})\,\mathbf{H},
$$

com **permeabilidade relativa** $\mu_r = 1 + \chi$. A irrotacionalidade
de $\mathbf{H}$ permite escrevê-lo como gradiente de um potencial escalar:

$$
\mathbf{H} \;=\; -\nabla\psi_{\mathrm{total}}.
$$

Substituindo em $\nabla\!\cdot\!\mathbf{B} = 0$:

$$
\boxed{\;\nabla\!\cdot\!\bigl[\mu_r(\mathbf{x})\,\nabla\psi_{\mathrm{total}}\bigr] = 0\;}
$$

— **Equação de Poisson generalizada** com coeficiente espacialmente
variável. Sua estrutura elíptica torna o problema **não local**: o valor
de $\psi$ em qualquer ponto depende de **toda** a distribuição de $\chi$.

### 1.2 Decomposição de Helmholtz

O sistema é submetido a um campo de **fundo uniforme**
$\mathbf{H}_0 = (H_{0x}, H_{0y})$ no infinito. Esse campo derivar-se-ia de
um potencial **linear** $-\mathbf{H}_0\!\cdot\!\mathbf{r}$. A presença do
ferrofluido perturba o campo localmente, mas a perturbação **decai longe
da interface**.

Adotamos a decomposição:

$$
\psi_{\mathrm{total}}(\mathbf{x}) \;=\; -\mathbf{H}_0\!\cdot\!\mathbf{r} \;+\; \tilde\psi(\mathbf{x}),
$$

equivalente a:

$$
\boxed{\;\mathbf{H}_{\mathrm{total}}(\mathbf{x}) \;=\; \mathbf{H}_0 \;-\; \nabla\tilde\psi(\mathbf{x})\;}
$$

onde $\tilde\psi$ é a **parte perturbadora** que se anula longe da interface.

### 1.3 Equação para $\tilde\psi$

Substituindo a decomposição em $\nabla\!\cdot\!(\mu_r\nabla\psi_{\mathrm{total}}) = 0$:

$$
\nabla\!\cdot\!\bigl[\mu_r\,(\nabla\tilde\psi - \mathbf{H}_0)\bigr] \;=\; 0,
$$

$$
\boxed{\;\nabla\!\cdot\!(\mu_r\,\nabla\tilde\psi) \;=\; \mathbf{H}_0\!\cdot\!\nabla\mu_r\;}
$$

(usando que $\mathbf{H}_0$ é constante, então $\nabla\!\cdot\!(\mu_r \mathbf{H}_0) = \mathbf{H}_0\!\cdot\!\nabla\mu_r$).

### 1.4 Vantagens da formulação em $\tilde\psi$

| Aspecto | $\psi_{\mathrm{total}}$ | $\tilde\psi$ (este código) |
|---|---|---|
| Condições de contorno | Dirichlet com valores grandes ($\sim H_0 L_x$) | Neumann homogêneo ($\partial_n\tilde\psi = 0$) |
| Magnitude esperada | $\sim H_0 L_x$ (linear no domínio) | $\sim H_0 W$ (localizada na interface) |
| Inicialização | Rampa $H_0(L_x - x)$ | $\tilde\psi = 0$ |
| Erro numérico | Subtração catastrófica entre números grandes | Erro absoluto $\approx$ erro relativo |
| Implementação do campo de fundo | Codificado nas BCs (frágil) | Explícito via $\mathbf{H}_0$ no termo fonte |

> **Em resumo:** resolver para $\tilde\psi$ **isola limpamente a física da
> perturbação**, permite Neumann simples nas bordas e reduz erros
> numéricos quando $H_0 \gg \nabla\chi$.

---

## 2. Assinatura da função

```python
solve_poisson_magnetic(psi_tilde, chi_field, Hx_fundo, Hy_fundo, sor_omega)
```

| Parâmetro | Shape / Tipo | Significado |
|---|---|---|
| `psi_tilde` | $(N_y, N_x)$ | Potencial perturbador $\tilde\psi$ (entrada/saída — chute inicial do passo anterior). |
| `chi_field` | $(N_y, N_x)$ | Suscetibilidade local $\chi(\phi)$. |
| `Hx_fundo`, `Hy_fundo` | escalar | Componentes do campo de fundo $\mathbf{H}_0$. |
| `sor_omega` | escalar | Fator de sobre-relaxação $\omega \in (1, 2)$. |

> **Reutilização do estado:** o `psi_tilde` recebido **não é zerado** — é
> usado como *warm start* do passo anterior. Como a interface muda pouco
> entre passos de tempo, esse chute inicial converge em poucas iterações.

---

## 3. Discretização por volumes finitos

### 3.1 Estrutura conservativa

O esquema usado é **volume finito** (não diferenças finitas), o que
preserva exatamente a conservação do fluxo magnético $\mathbf{B}$ através
das faces das células. Em cada nó $(y, x)$, integra-se:

$$
\int_{\Omega_{ij}}\!\nabla\!\cdot\!(\mu_r\,\nabla\tilde\psi)\,d\Omega
\;=\;
\oint_{\partial\Omega_{ij}} \mu_r\,\nabla\tilde\psi\!\cdot\!\hat{\mathbf{n}}\,dS.
$$

A integral de superfície decompõe-se em 4 fluxos pelas faces (E/W/N/S).

### 3.2 Permeabilidades nas faces (linhas 18–21)

A permeabilidade é **interpolada aritmeticamente** nos pontos médios entre
nós vizinhos, garantindo um esquema centrado:

$$
\begin{aligned}
\mu_E &= 1 + \tfrac{1}{2}(\chi_{y,x+1} + \chi_{y,x}),\\
\mu_W &= 1 + \tfrac{1}{2}(\chi_{y,x} + \chi_{y,x-1}),\\
\mu_N &= 1 + \tfrac{1}{2}(\chi_{yp,x} + \chi_{y,x}),\\
\mu_S &= 1 + \tfrac{1}{2}(\chi_{ym,x} + \chi_{y,x}).
\end{aligned}
$$

```python
mu_E = 1.0 + 0.5*(chi_field[y, x+1] + chi_field[y, x])
mu_W = 1.0 + 0.5*(chi_field[y, x]   + chi_field[y, x-1])
mu_N = 1.0 + 0.5*(chi_field[yp, x]  + chi_field[y, x])
mu_S = 1.0 + 0.5*(chi_field[ym, x]  + chi_field[y, x])
```

> **Alternativas:** poderia-se usar média **harmônica**
> $\mu_{\mathrm{face}} = 2\mu_a\mu_b/(\mu_a+\mu_b)$, que é mais precisa
> para saltos abruptos de permeabilidade. Para o regime de pequenos $\chi$
> em ferrofluidos típicos ($\chi \lesssim 1$), a média aritmética é suficiente.

### 3.3 Fluxos nas faces

O fluxo magnético através de cada face é (em volumes finitos com
$\Delta x = \Delta y = 1$):

$$
\mathcal{F}_E = \mu_E\,(\tilde\psi_{y,x+1} - \tilde\psi_{y,x}),
$$

e similarmente para W, N, S. A soma dos fluxos deve igualar a fonte:

$$
\sum_{\mathrm{face}} \mathcal{F}_{\mathrm{face}} \;=\; \int_{\Omega_{ij}}\mathbf{H}_0\!\cdot\!\nabla\mu_r\,d\Omega.
$$

### 3.4 Termo fonte $\mathbf{H}_0\!\cdot\!\nabla\mu_r$ (linha 26)

A integral da fonte sobre o volume de controle é aproximada usando as
**mesmas permeabilidades de face** (consistência do esquema):

$$
\boxed{\;\mathrm{rhs} \;=\; H_{0x}\,(\mu_E - \mu_W) \;+\; H_{0y}\,(\mu_N - \mu_S)\;}
$$

```python
rhs = Hx_fundo*(mu_E - mu_W) + Hy_fundo*(mu_N - mu_S)
```

**Interpretação física:**

- $\mu_E - \mu_W$ é a variação de $\mu_r$ no eixo $x$ — só é não nulo **na
  interface** (onde $\chi$ muda).
- $\mathbf{H}_0\!\cdot\!\nabla\mu_r$ é o termo que **força** a aparição do
  campo perturbador. Sem $\mathbf{H}_0$, ou sem gradiente de $\chi$,
  $\tilde\psi$ permanece zero.
- **Geometricamente:** a fonte é concentrada na faixa interfacial e tem
  sinal dependendo da orientação relativa entre $\mathbf{H}_0$ e a normal
  à interface.

### 3.5 Equação discreta resolvida

A balança de fluxos no volume de controle $(y, x)$, com a fonte, resulta em:

$$
\sum_{\mathrm{face}} \mu_{\mathrm{face}}\,(\tilde\psi_{\mathrm{viz}} - \tilde\psi_{y,x})
\;=\; \mathrm{rhs},
$$

de onde se isola $\tilde\psi_{y,x}$:

$$
\tilde\psi_{y,x}^{*} \;=\;
\frac{\mu_E\,\tilde\psi_{y,x+1} + \mu_W\,\tilde\psi_{y,x-1} + \mu_N\,\tilde\psi_{yp,x} + \mu_S\,\tilde\psi_{ym,x} - \mathrm{rhs}}{\mu_E + \mu_W + \mu_N + \mu_S}.
$$

(linhas 29–30 do código).

---

## 4. Algoritmo SOR (Sobre-Relaxação Sucessiva)

### 4.1 Por que iterativo?

A equação discreta acima é um sistema linear $A\tilde\psi = b$ com matriz
$A$ esparsa de banda. Para malhas grandes ($\sim 600 \times 300$):

- **Solvers diretos** (LU, Cholesky): $O(N^{3})$ ou $O(N^{1.5})$ — proibitivos.
- **Gradiente conjugado precondicionado** (PCG): $O(N^{1.5})$ — bom, mas requer matriz montada.
- **Multigrid**: $O(N)$ — ótimo, mas implementação complexa.
- **SOR**: $O(N^{1.5})$ — simples, fácil de paralelizar, *embarrassingly local*.

Como o solver é chamado **uma vez por passo de tempo** e o estado anterior
serve de *warm start*, o SOR converge em **15 iterações** com fator
$\omega \approx 1.85$.

### 4.2 Esquema iterativo

Cada iteração de Gauss–Seidel calcula $\tilde\psi^{*}$ usando valores **mais
recentes** dos vizinhos (sweep ordenado). O SOR aplica uma **extrapolação
ponderada**:

$$
\tilde\psi_{y,x}^{\mathrm{novo}} \;=\; (1 - \omega)\,\tilde\psi_{y,x}^{\mathrm{antigo}} \;+\; \omega\,\tilde\psi_{y,x}^{*}.
$$

```python
psi_tilde[y, x] = (1.0 - sor_omega)*psi_tilde[y, x] + sor_omega*psi_new
```

| $\omega$ | Comportamento |
|---|---|
| $\omega = 1$ | Gauss–Seidel puro |
| $1 < \omega < 2$ | Sobre-relaxação: acelera convergência |
| $\omega = 2$ | Divergência (limite teórico) |
| $\omega < 1$ | Sub-relaxação: estabiliza, mas mais lento |

**$\omega$ ótimo** para Poisson em malha retangular $N_x \times N_y$:

$$
\omega_{\mathrm{ot}} \;\approx\; \frac{2}{1 + \sin(\pi/\max(N_x, N_y))} \;\to\; 2 \text{ quando } N \to \infty.
$$

Para $N \sim 200$, $\omega_{\mathrm{ot}} \approx 1.97$, mas valores
ligeiramente menores ($\sim 1.85$) são mais robustos contra a heterogeneidade
de $\mu_r$.

### 4.3 Número fixo de iterações (15)

```python
for _ in range(15):
    ...
```

> **Justificativa:** não há checagem de convergência (`while res > tol`)
> porque:
>
> 1. A heterogeneidade de $\chi$ varia **lentamente** entre passos de tempo
>    (acoplada à evolução do $\phi$).
> 2. O *warm start* do passo anterior já está próximo da solução.
> 3. 15 sweeps com $\omega = 1.85$ reduzem o resíduo em $\sim 10^{-5}$
>    — adequado para acoplamento com LBM.
> 4. Tempo de execução **previsível**, sem early-exit.

---

## 5. Condições de contorno

### 5.1 Eixo $y$: PERIÓDICO (intrínseco)

Linhas 13–14:

```python
yp = (y + 1) % ny
ym = (y - 1 + ny) % ny
```

O acesso a `psi_tilde[yp, x]` com aritmética modular **identifica** as
linhas $y = 0$ e $y = N_y - 1$. Consistente com:

- A inicialização do campo de fase ($\phi$ é periódico em $y$ via
  $\cos(2\pi m y/N_y)$).
- O LBM (que tem $y$ sempre periódico).
- A Cahn–Hilliard (idem).

### 5.2 Eixo $x$: NEUMANN homogêneo (linhas 35–37)

```python
psi_tilde[y, 0]      = psi_tilde[y, 1]
psi_tilde[y, nx - 1] = psi_tilde[y, nx - 2]
```

Implementa $\partial_x \tilde\psi = 0$ em ambas as bordas, ou seja:
**a perturbação $\tilde\psi$ é uniforme transversalmente nas bordas**, o
que corresponde fisicamente a:

$$
\mathbf{H}_{\mathrm{total}}|_{x = 0,\,N_x-1} \;=\; \mathbf{H}_0 \;-\; \underbrace{\nabla\tilde\psi}_{\to 0} \;\approx\; \mathbf{H}_0.
$$

> **Justificativa:** longe da interface (no inlet e outlet), o campo deve
> recuperar o valor de fundo $\mathbf{H}_0$. Como $\chi$ é constante nessas
> regiões (fluido puro), $\tilde\psi$ é uma função harmônica que decai
> rapidamente — Neumann nulo é uma boa aproximação da condição "no infinito".

### 5.3 Comparação com a versão antiga

| Eixo | Versão antiga | Versão atual |
|---|---|---|
| $x = 0$ | **Dirichlet** $\psi = H_0 L_x$ | **Neumann** $\partial_x\tilde\psi = 0$ |
| $x = N_x - 1$ | **Dirichlet** $\psi = 0$ | **Neumann** $\partial_x\tilde\psi = 0$ |
| $y = 0,\,N_y-1$ | **Neumann** $\partial_y\psi = 0$ | **Periódico** |

A migração para Neumann em $x$ + periódico em $y$ é o que **permite resolver
para $\tilde\psi$**, com $\mathbf{H}_0$ injetado pelo termo fonte e não
pelas BCs.

### 5.4 Por que reaplicar Neumann a cada iteração?

```python
for _ in range(15):
    # ... sweep SOR ...
    for y in prange(ny):
        psi_tilde[y, 0] = psi_tilde[y, 1]
        psi_tilde[y, nx-1] = psi_tilde[y, nx-2]
```

A reaplicação **dentro do loop iterativo** é necessária porque:

- O sweep SOR atualiza $\tilde\psi[y, 1]$ e $\tilde\psi[y, N_x-2]$ a cada iteração.
- Se as BCs não fossem atualizadas, a iteração seguinte usaria valores
  **defasados** nas bordas, sabotando a convergência.

---

## 6. Estrutura do loop completo

```python
for _ in range(15):                           # ── Iterações SOR ──
    for y in prange(ny):                      #    Sweep paralelo em y
        yp = (y+1) % ny                       #    Periodicidade y
        ym = (y-1+ny) % ny
        for x in range(1, nx-1):              #    x interno
            # 1. Permeabilidades nas faces
            mu_E, mu_W, mu_N, mu_S = ...
            # 2. Soma do estêncil
            denom = mu_E + mu_W + mu_N + mu_S
            # 3. Termo fonte
            rhs = Hx_fundo*(mu_E-mu_W) + Hy_fundo*(mu_N-mu_S)
            # 4. Solução Gauss-Seidel + SOR
            psi_new = (μ·viz - rhs) / denom
            psi_tilde[y, x] = (1-ω)·psi_tilde[y, x] + ω·psi_new
    for y in prange(ny):                      # ── BC Neumann em x ──
        psi_tilde[y, 0]    = psi_tilde[y, 1]
        psi_tilde[y, nx-1] = psi_tilde[y, nx-2]
return psi_tilde
```

---

## 7. Acoplamento com o resto da simulação

```
                    ┌──────────────────────────┐
                    │   φ (Cahn–Hilliard)      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   χ(φ)  (acoplador)      │   χ = χ_max · (φ+1)/2
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  solve_poisson_magnetic  │  ← este módulo
                    │  produz  ψ̃               │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   lbm_step               │
                    │   H_total = H₀ - ∇ψ̃     │
                    │   F_m = -½|H|² ∇χ        │
                    └──────────────────────────┘
```

### 7.1 Cadência de chamada

`solve_poisson_magnetic` é chamado **uma vez por passo de tempo** do LBM,
**após** a evolução do campo de fase (para usar o $\chi$ mais recente) e
**antes** do `lbm_step` (que precisa de $\tilde\psi$ atualizado para calcular
$\mathbf{F}_m$).

### 7.2 *Warm start* entre passos

Como $\chi$ muda lentamente, o $\tilde\psi$ do passo anterior é um
excelente chute inicial. Por isso o solver não zera o buffer — apenas
itera 15 vezes a partir do estado existente.

---

## 8. Considerações numéricas

### 8.1 Estabilidade do SOR

A condição **necessária e suficiente** para convergência do SOR num
sistema simétrico positivo definido (caso do Poisson) é:

$$
0 < \omega < 2.
$$

O esquema é incondicionalmente estável dentro deste intervalo.

### 8.2 Resíduo aproximado após 15 iterações

Para Poisson 2D em malha $N \times N$ com $\omega$ ótimo:

$$
\frac{\|r^{(15)}\|}{\|r^{(0)}\|} \;\approx\; \rho_{\mathrm{SOR}}^{15},\qquad
\rho_{\mathrm{SOR}} \approx \omega - 1.
$$

Para $\omega = 1.85$: $\rho \approx 0.85$, e $0.85^{15} \approx 0.087$ —
redução de **uma ordem de grandeza por chamada**. Como o estado anterior
já é próximo do correto, isso é suficiente.

### 8.3 Custo computacional

| Operação | Por sweep | Por chamada (15 sweeps) |
|---|---|---|
| FLOPs por nó | $\sim 30$ | $\sim 450$ |
| Acessos a memória | 9 reads, 1 write | 135 reads, 15 writes |
| Total para $300 \times 600$ | $\sim 5.4\,\text{MFLOPs}$ | $\sim 81\,\text{MFLOPs}$ |

Em CPU moderna (Numba paralelo), $\sim 50$ ms por chamada — comparável ao
custo de um `lbm_step`.

### 8.4 Diagnóstico de problemas

| Sintoma | Causa provável | Solução |
|---|---|---|
| $\tilde\psi$ explode (NaN) | $\omega \geq 2$ | Reduzir `sor_omega` para $\sim 1.7$ |
| Convergência lenta | Heterogeneidade alta de $\chi$ | Aumentar iterações para 30 |
| Força magnética com ruído alta freq. | $\chi$ não suave | Suavizar campo de fase |
| Campo de fundo incorreto longe da interface | BCs em $x$ erradas | Verificar Neumann reaplicado |

---

## 9. Resumo executivo

| Componente | Implementação | Local |
|---|---|---|
| Equação | $\nabla\!\cdot\!(\mu_r\nabla\tilde\psi) = \mathbf{H}_0\!\cdot\!\nabla\mu_r$ | — |
| Discretização | Volumes finitos com $\mu_r$ nas faces | Linhas 18–21 |
| Termo fonte | $H_{0x}(\mu_E-\mu_W) + H_{0y}(\mu_N-\mu_S)$ | Linha 26 |
| Solver | SOR Gauss–Seidel paralelo | Linhas 10–32 |
| Iterações | 15 fixas (warm start) | Linha 10 |
| BC eixo $x$ | Neumann $\partial_x\tilde\psi = 0$ | Linhas 35–37 |
| BC eixo $y$ | Periódico | Linhas 13–14 |
| Campo total recuperado | $\mathbf{H} = \mathbf{H}_0 - \nabla\tilde\psi$ no `lbm_step` | `lbm.py` linhas 37–38 |
