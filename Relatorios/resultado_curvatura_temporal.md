# Metodologia do script `resultado_curvatura_temporal.py`

**Caminho:** `post_process/resultado_curvatura_temporal.py`
**Objetivo:** Extrair, a partir dos snapshots `.vtr` da simulação, a evolução
temporal de duas métricas interfaciais — a **curvatura média absoluta**
$\overline{|\kappa|}(t)$ e a **amplitude da perturbação** $A(t)$ — gerando uma
figura acadêmica e um cache `.npz` reutilizável (consumido, p. ex., por
`valida_lsa.py`).

---

## 1. Entradas e saídas

### 1.1 Entrada

```bash
python resultado_curvatura_temporal.py <vtk_dir>
```

- `<vtk_dir>` — diretório contendo arquivos `dados_macro_NNNNN.vtr`
  exportados pelo módulo de pós-processamento da simulação.

### 1.2 Saídas (criadas no diretório-pai de `<vtk_dir>`)

| Arquivo | Conteúdo |
|---|---|
| `curvatura_temporal.png` | Figura com dois painéis verticais (curvatura e amplitude vs. $t$). |
| `curvatura_temporal.npz` | Arrays `t`, `kappa_mean_abs`, `amplitude` em formato NumPy. |

---

## 2. Pipeline geral

```
vtk_dir ──► glob + ordenação ──► subamostragem (20 snapshots)
                                          │
                                          ▼
                          ┌──────  loop por snapshot ──────┐
                          │                                │
                  load_phi_from_vtr               compute_mean_abs_curvature
                          │                                │
                          └──────► compute_interface_amplitude
                                          │
                                          ▼
                                 plot + savez (npz)
```

---

## 3. Passo 1 — Coleta e ordenação dos `.vtr`

Linhas 103–106. Faz `glob` por `dados_macro_*.vtr` e ordena pela chave
`extract_time_step`, que aplica a regex:

$$
\texttt{dados\_macro\_}(\underbrace{\backslash d+}_{t})\texttt{.vtr}
$$

para converter o nome do arquivo em um inteiro $t$ (passo de tempo).
Arquivos fora do padrão recebem $-1$.

---

## 4. Passo 2 — Subamostragem uniforme

Linhas 112–115. Define $N_{\mathrm{target}} = 20$ snapshots. Se houver mais
de 20 arquivos, seleciona índices uniformemente distribuídos:

$$
\mathrm{idx} \;=\; \mathrm{round}\!\left(\mathrm{linspace}(0,\,N-1,\,20)\right),
$$

reduzindo o custo de leitura sem perder a forma global das curvas.

---

## 5. Passo 3 — Leitura do campo $\phi(x,y)$

**Função:** `load_phi_from_vtr(vtr_path)` (linhas 41–55)

1. Lê o `.vtr` com `vtkXMLRectilinearGridReader`.
2. Extrai o array de **célula** chamado `"fase_phi"`.
3. Determina as dimensões a partir do grid:
   $$
   n_x = \texttt{dims\_pts}[0] - 1,\qquad n_y = \texttt{dims\_pts}[1] - 1.
   $$
   (Subtrai 1 porque VTK reporta nós de ponto, e a fase está em centros de célula.)
4. Devolve $\phi \in \mathbb{R}^{n_y \times n_x}$.

---

## 6. Passo 4 — Curvatura média absoluta

**Função:** `compute_mean_abs_curvature(phi)` (linhas 58–75)

### 6.1 Fórmula da curvatura do *level-set*

Para uma função implícita $\phi(x,y)$ cujo zero define a interface,
a curvatura é:

$$
\kappa \;=\;
\nabla \!\cdot\! \left(\frac{\nabla\phi}{|\nabla\phi|}\right)
\;=\;
\frac{\phi_x^{2}\,\phi_{yy} \;+\; \phi_y^{2}\,\phi_{xx} \;-\; 2\,\phi_x\,\phi_y\,\phi_{xy}}{\bigl(\phi_x^{2}+\phi_y^{2}\bigr)^{3/2}}.
$$

### 6.2 Implementação por diferenças finitas

Usa `np.gradient` em duas etapas:

```
dy, dx        = np.gradient(phi)        # ∂φ/∂y, ∂φ/∂x
d2y, dy_dx    = np.gradient(dy)         # ∂²φ/∂y², ∂²φ/∂x∂y
dx_dy, d2x    = np.gradient(dx)         # ∂²φ/∂y∂x, ∂²φ/∂x²
```

> **Nota técnica:** `np.gradient` em um array 2D retorna `(∂/∂axis0, ∂/∂axis1)`
> $= (\partial/\partial y,\,\partial/\partial x)$. O código usa
> `dy_dx` como aproximação de $\phi_{xy}$ via derivada cruzada.

Em seguida:

$$
\kappa(x,y) \;=\; \frac{\phi_x^{2}\,\phi_{yy} + \phi_y^{2}\,\phi_{xx} - 2\,\phi_x\,\phi_y\,\phi_{xy}}{\bigl(\phi_x^{2}+\phi_y^{2}+10^{-8}\bigr)^{3/2}},
$$

onde $10^{-8}$ é a regularização do denominador para evitar divisão por zero
fora da interface.

### 6.3 Máscara interfacial e média

A média absoluta é tomada apenas sobre a **faixa difusa**:

$$
\overline{|\kappa|} \;=\; \frac{1}{|\mathcal{M}|}\sum_{(i,j)\in\mathcal{M}} |\kappa_{ij}|,
\qquad
\mathcal{M} = \{(i,j) : |\phi_{ij}| < 0{,}1\}.
$$

Se $\mathcal{M} = \emptyset$, retorna $0$.

---

## 7. Passo 5 — Amplitude da perturbação

**Função:** `compute_interface_amplitude(phi)` (linhas 78–84)

Versão **simples** (diferente da projeção de Fourier usada em
`valida_lsa.py`): mede a meia-largura da interface no eixo $x$.

1. Para cada linha $y$, encontra o **primeiro** índice $x$ tal que $\phi < 0$:
   $$
   X(y) \;=\; \arg\min_{x}\bigl\{\phi(y,x) < 0\bigr\}.
   $$
2. Define a amplitude como:
   $$
   A \;=\; \frac{X_{\max} - X_{\min}}{2}.
   $$

> **Limitação:** este estimador é sensível a ruído e a modos secundários.
> Para a análise LSA preferir a projeção de Fourier de `valida_lsa.py::_amplitude`.

---

## 8. Passo 6 — Loop principal

Linhas 119–134. Para cada `.vtr` selecionado:

1. Extrai $t$ via regex.
2. Carrega $\phi$.
3. Calcula $\overline{|\kappa|}$ e $A$.
4. Acumula em listas e imprime linha de progresso.

Ao final, converte para arrays NumPy:

$$
\mathbf{t} \in \mathbb{R}^{N},\quad
\boldsymbol{\kappa} \in \mathbb{R}^{N},\quad
\mathbf{A} \in \mathbb{R}^{N}.
$$

---

## 9. Passo 7 — Figura acadêmica

Linhas 142–164. Cria `fig, (ax1, ax2)` com `sharex=True`:

### 9.1 Painel superior — $\overline{|\kappa|}(t)$

- Marcadores circulares vermelhos (`#9b1d20`), face branca.
- **Escala automática:** se
  $$
  \frac{\kappa_{\max}}{\kappa_{\min}^{(>0)}} > 100
  $$
  então usa escala logarítmica em $y$ (faixa dinâmica grande).
- Rótulo: $\overline{|\kappa|}(t)$.

### 9.2 Painel inferior — $A(t)$

- Marcadores quadrados azuis (`#1a5276`), face branca.
- Eixo $x$: `Time step $t$ (lattice units)`.
- Rótulo $y$: $A(t)$ em l.u.

### 9.3 Estilo

Definido nas linhas 24–38:

- Fonte serif, mathtext Computer Modern.
- `lines.linewidth = 1.6`.
- Grade pontilhada (alpha 0.6).
- Salva em 300 dpi com `bbox_inches='tight'`.

---

## 10. Passo 8 — Cache `.npz`

Linhas 169–171:

```python
np.savez(npz_path, t=times, kappa_mean_abs=curvs, amplitude=amps)
```

Esse arquivo é **consumido por `valida_lsa.py`**, que detecta
`curvatura_temporal.npz` e pula a releitura dos `.vtr` (acelera o ajuste
exponencial em ordem de magnitude).

---

## 11. Resumo do fluxo de dados

$$
\boxed{
\begin{array}{rcl}
\texttt{dados\_macro\_*.vtr} & \longrightarrow & \phi_i(x,y),\ i=1\ldots 20\\[6pt]
\phi_i & \longrightarrow & \kappa_i,\ A_i \\[6pt]
\{(t_i,\,\overline{|\kappa|}_i,\,A_i)\} & \longrightarrow & \texttt{curvatura\_temporal.npz} \\[6pt]
 & \longrightarrow & \texttt{curvatura\_temporal.png}
\end{array}}
$$

---

## 12. Interpretação física dos resultados

| Comportamento de $\overline{|\kappa|}(t)$ | Implicação |
|---|---|
| Constante e baixa | Interface plana, regime estável. |
| Crescimento monotônico | Fingering em desenvolvimento. |
| Saturação após pico | Coalescência ou regime não linear. |
| Oscilação irregular | Possíveis correntes espúrias ou interface mal resolvida. |

| Comportamento de $A(t)$ | Implicação |
|---|---|
| $A \propto e^{s t}$ | Regime linear (LSA aplicável). |
| Crescimento sublinear | Saturação não linear. |
| Decaimento | Modo estável $\Rightarrow \zeta(\alpha) < 0$. |

---

## 13. Exemplo de uso

```bash
python post_process/resultado_curvatura_temporal.py \
    OpcaoAC_W3_amp10_kxi0p018_d12mes05-h10_min52/vtk
```

**Saídas geradas (no diretório-pai `vtk/`):**

- `curvatura_temporal.png`
- `curvatura_temporal.npz`

---

## 14. Relação com `valida_lsa.py`

```
resultado_curvatura_temporal.py            valida_lsa.py
─────────────────────────────              ──────────────────
 lê .vtr, calcula A e κ        ┐           lê .npz se existir
                               ├──►        ajusta A(t) = A₀ e^{s t}
 grava curvatura_temporal.npz  ┘           compara com Eq. 9
```

> **Observação:** as **definições de amplitude** diferem nos dois scripts:
> - aqui: $A = (X_{\max} - X_{\min})/2$ (estimador simples).
> - em `valida_lsa.py::_amplitude`: projeção de Fourier do modo $m$ via
>   $A_m = (2/N_y)\,|\sum_y x_{\mathrm{exact}}(y)\,e^{-2\pi i m y/N_y}|$.
>
> Em regimes lineares limpos as duas concordam; em regimes não lineares ou
> ruidosos a projeção de Fourier é mais confiável.
