# Metodologia do script `valida_lsa.py`

**Caminho:** `post_process/valida_lsa.py`
**Objetivo:** Comparar o crescimento numérico (LBM) da amplitude interfacial $A(t)$
com a previsão analítica da relação de dispersão da instabilidade de
Saffman–Taylor em meio poroso magnetizado (Eq. 9 da formulação adimensional).

---

## 1. Convenções de normalização

O script trabalha em **unidades de rede** (lattice units, l.u.) e adota as
seguintes referências:

| Grandeza | Símbolo | Expressão |
|---|---|---|
| Comprimento de referência | $L_{\mathrm{ref}}$ | $N_y$ (altura do domínio) |
| Velocidade de referência | $U_{\mathrm{ref}}$ | $U_{\mathrm{inlet}}$ |
| Nº de onda adimensional | $\alpha$ | $2\pi\,m$ |
| Taxa adimensional | $\zeta$ | $\dfrac{s\,L_{\mathrm{ref}}}{U_{\mathrm{ref}}}$ |
| Darcy | $\mathrm{Da}$ | $K_0 / N_y^{2}$ |
| Capilar | $\mathrm{Ca}$ | $\nu_{\mathrm{in}}\,U_{\mathrm{inlet}} / \sigma$ |
| Bond magnético | $\mathrm{Ca}_m$ | $\chi\,H_0^{2}\,N_y / \sigma$ |
| Contraste magnético | $\Lambda$ | $\chi / (2 + \chi)$ |
| Comp. normal/tangencial | $H_{0n}^{2},\,H_{0t}^{2}$ | $\cos^{2}\theta,\,\sin^{2}\theta$ |

A taxa dimensional segue diretamente da relação
$s = \zeta\,U_{\mathrm{ref}} / L_{\mathrm{ref}}\ [\mathrm{ts}^{-1}]$.

---

## 2. Relação de dispersão analítica (Eq. 9)

Implementada em `zeta_analitico(...)` (linhas 68–81). A taxa adimensional
$\zeta(\alpha)$ se decompõe em quatro termos físicos:

$$
\zeta(\alpha) \;=\; \underbrace{\alpha\,\frac{M-1}{M+1}}_{\text{viscoso}}
\;+\;
\frac{\mathrm{Da}\,\alpha}{\mathrm{Ca}\,(1+M)}\,
\Big(
\underbrace{\mathrm{Bo}}_{\text{gravitacional}}
\;\underbrace{-\,\alpha^{2}}_{\text{capilar}}
\;+\;\underbrace{\mathrm{Ca}_m\,\Lambda\,\alpha\,(H_{0n}^{2}-H_{0t}^{2})}_{\text{magnético}}
\Big).
$$

Onde:

- **Termo viscoso** — fator de Atwood viscoso $(M-1)/(M+1)$, com $M = \mu_{\mathrm{out}}/\mu_{\mathrm{in}}$.
- **Termo gravitacional** — $\mathrm{Bo} = 0$ (sem gravidade nesta formulação).
- **Termo capilar** — $-\alpha^2$, sempre estabilizante.
- **Termo magnético** — favorece instabilidade quando $H_{0n}^2 > H_{0t}^2$
  (campo predominantemente normal à interface).

---

## 3. Pipeline geral (função `main`)

```
case_dir ──► extrair_parametros ──► carregar_amplitude ──► ajuste_exponencial
                                                                    │
                              ┌─────────────────────────────────────┘
                              ▼
                       zeta_analitico  →  _relatorio  →  plotar
```

### 3.1 Argumentos de entrada

```bash
python post_process/valida_lsa.py <case_dir> [--t0 T0] [--t1 T1] [--no-auto]
```

- `case_dir` — diretório com `relatorio_execucao.json` e `vtk/` (ou cache).
- `--t0`, `--t1` — janela manual de ajuste exponencial (em timesteps).
- `--no-auto` — desliga a detecção automática do platô.

---

## 4. Passo 1 — Extração dos parâmetros adimensionais

**Função:** `extrair_parametros(case_dir)` (linhas 86–166)

1. Lê `relatorio_execucao.json` e extrai $N_y$, $\tau_{\mathrm{in}}$,
   $\tau_{\mathrm{out}}$, $\sigma$, $K_0$, $U_{\mathrm{inlet}}$, $H_0$,
   $\chi_{\max}$, $\theta$ (em graus) e $m$ (modo).

2. Converte tempos de relaxação em viscosidades:
   $$
   \nu_{i} \;=\; \frac{\tau_{i}-\tfrac{1}{2}}{3},\qquad i\in\{\text{in},\text{out}\}.
   $$

3. Valida $m \geq 1$ (caso contrário levanta `ValueError`).

4. Emite **avisos de qualidade**:
   - $W < 3$ → viola Nyquist no D2Q9.
   - Amplitude inicial $\geq W$ → fora do regime LSA puro.
   - Calcula $k\xi = 2\pi\,m\,W/N_y$ (ideal $< 0{,}01$).

5. Monta os grupos adimensionais:
   $$
   M = \frac{\nu_{\mathrm{out}}}{\nu_{\mathrm{in}}},\quad
   \mathrm{Da} = \frac{K_0}{N_y^{2}},\quad
   \mathrm{Ca} = \frac{\nu_{\mathrm{in}}\,U_{\mathrm{inlet}}}{\sigma},\quad
   \mathrm{Ca}_m = \frac{\chi_{\max}\,H_0^{2}\,N_y}{\sigma},
   $$
   $$
   \Lambda = \frac{\chi_{\max}}{2+\chi_{\max}},\qquad
   \alpha_{\mathrm{sim}} = 2\pi\,m.
   $$

---

## 5. Passo 2 — Série temporal da amplitude

**Função:** `carregar_amplitude(case_dir, mode_m)` (linhas 272–308)

### 5.1 Fonte dos dados

- **Preferência 1:** cache `.npz` em `curvatura_temporal.npz`.
- **Fallback:** leitura direta dos arquivos `vtk/dados_macro_*.vtr`
  (requer `python-vtk`), subamostrados para no máximo 80 snapshots.

### 5.2 Extração da amplitude por Fourier — `_amplitude(phi, mode_m)`

Para cada snapshot $\phi(x,y)$:

1. **Localiza a interface** ($\phi = 0$) em cada linha $y$: encontra o
   primeiro $x$ tal que $\phi < 0$, isto é, a borda direita da bolha.

2. **Interpolação linear sub-pixel** entre os pontos $\phi_L$ (esquerda) e
   $\phi_R$ (direita):
   $$
   x_{\mathrm{exact}}(y) \;=\; i_L + \frac{\phi_L}{\phi_L - \phi_R + 10^{-15}}.
   $$

3. **Projeção de Fourier no modo $m$**:
   $$
   A_m \;=\; \frac{2}{N_y}\,\left|\sum_{y=0}^{N_y-1} x_{\mathrm{exact}}(y)\,
            \exp\!\left(-\frac{2\pi i\,m\,y}{N_y}\right)\right|.
   $$

   Essa projeção é robusta a ruído e a modos secundários, sendo muito
   superior à simples diferença max-min.

---

## 6. Passo 3 — Detecção automática do regime linear

**Funções:** `_s_local(t, A)` e `_detecta_plateau(t, A)` (linhas 217–269)

### 6.1 Taxa instantânea $s_{\mathrm{loc}}(t)$

Por regressão linear local em uma janela móvel de $\pm k$ pontos
($k = \max(3,\,n/16)$):
$$
s_{\mathrm{loc}}(t_i) \;=\; \left.\frac{d\ln A}{dt}\right|_{t_i}
\quad\text{via}\quad
\mathrm{polyfit}\bigl(t_{i\pm k},\,\ln A_{i\pm k},\,1\bigr).
$$

### 6.2 Localização do platô

O platô é definido como a janela contígua centrada em
$i_{\mathrm{peak}} = \arg\max s_{\mathrm{loc}}$ onde
$$
\frac{|\,s_{\mathrm{loc}}(t) - s_{\mathrm{peak}}\,|}{s_{\mathrm{peak}}}
\;\leq\; \mathrm{tol}.
$$

O algoritmo tenta tolerâncias progressivas
$\mathrm{tol} \in \{4\%,\,6\%,\,9\%,\,13\%,\,20\%\}$ até obter pelo menos
**6 pontos** dentro da janela. Retorna `(t0, t1, s_loc, s_peak, tol_usada)`.

---

## 7. Passo 4 — Ajuste exponencial

**Função:** `ajuste_exponencial(t, A, t0_user, t1_user, auto=True)`
(linhas 314–373)

Hipótese física: durante o regime linear, $A(t) = A_0\,e^{s\,t}$, de modo que
$$
\ln A(t) \;=\; \ln A_0 + s\,t.
$$

### 7.1 Seleção da janela

| Caso | Janela usada |
|---|---|
| `auto=True` e usuário não passou `--t0/--t1` | Platô detectado por `_detecta_plateau` |
| Platô não detectável | Fallback: $[t_0 + 0{,}30\,\Delta T,\; t_0 + 0{,}75\,\Delta T]$ |
| Usuário forneceu limites | Manual (com fallback para os limites ausentes) |

### 7.2 Estimadores

Dentro da janela $[t_0, t_1]$ com $A > 0$:

- **Regressão polinomial** (estimador principal):
  $$
  s_{\mathrm{polyfit}} \;=\; \mathrm{coef}_1\bigl[\mathrm{polyfit}(t,\ln A,\,1)\bigr].
  $$

- **Mediana de $s_{\mathrm{loc}}$** (estimador robusto paralelo):
  $$
  s_{\mathrm{median}} \;=\; \mathrm{median}\bigl\{s_{\mathrm{loc}}(t_i)\bigr\}_{t_i\in[t_0,t_1]}.
  $$

- **Pico de $s_{\mathrm{loc}}$** (instantâneo):
  $$
  s_{\mathrm{peak}} \;=\; \max_t s_{\mathrm{loc}}(t).
  $$

---

## 8. Passo 5 — Comparação e relatório

**Função:** `_relatorio(params, s_num, s_ana, info)` (linhas 379–426)

### 8.1 Conversão para o domínio adimensional

$$
\zeta_{\mathrm{num}} \;=\; \frac{s_{\mathrm{num}}\,L_{\mathrm{ref}}}{U_{\mathrm{ref}}},
\qquad
\zeta_{\mathrm{ana}} \;=\; \zeta(\alpha_{\mathrm{sim}};\,M,\mathrm{Bo},\mathrm{Ca}_m,\Lambda,H_{0n}^{2},H_{0t}^{2},\mathrm{Da},\mathrm{Ca}).
$$

### 8.2 Erro relativo

$$
\varepsilon \;=\; \frac{|\,s_{\mathrm{num}} - s_{\mathrm{ana}}\,|}{|s_{\mathrm{ana}}| + 10^{-30}} \times 100\,\%.
$$

### 8.3 Saída no terminal

O relatório imprime:

- Parâmetros adimensionais ($m$, $\alpha$, $M$, $\mathrm{Da}$, $\mathrm{Ca}$,
  $\mathrm{Ca}_m$, $\Lambda$, $H_{0n}^{2}$, $H_{0t}^{2}$).
- Modo da janela (automática/manual) e tolerância utilizada.
- Três estimadores: $s_{\mathrm{polyfit}}$, $s_{\mathrm{peak}}$, $s_{\mathrm{median}}$.
- Taxa dimensional $s$ (numérica vs. analítica) e adimensional $\zeta$.

---

## 9. Passo 6 — Visualização

**Função:** `plotar(t, A, s_num, A0_num, s_ana, params, ...)` (linhas 432–513)

Gera `comparacao_lsa.png` (300 dpi) com dois painéis:

### 9.1 Painel esquerdo — Crescimento $A(t)$

Escala semilogarítmica:

- Pontos: dados LBM.
- Linha azul: ajuste $A(t) = A_0\,e^{s_{\mathrm{num}}\,t}$.
- Linha vermelha tracejada: $A_{\mathrm{ana}}(t) = A_{0,\mathrm{ana}}\,e^{s_{\mathrm{ana}}(t-t_{0,\mathrm{ana}})}$.
- Sombreamento: janela de ajuste $[t_0,t_1]$.

### 9.2 Painel direito — Curva de dispersão $\zeta(\alpha)$

- Curva contínua: $\zeta(\alpha)$ analítica para $\alpha \in [0,\,3{,}5\,\alpha_{\mathrm{sim}}]$.
- Linha pontilhada vertical em $\alpha_{\mathrm{sim}} = 2\pi m$.
- Marcador vermelho: $\zeta_{\mathrm{ana}}(\alpha_m)$.
- Marcador verde (losango): $\zeta_{\mathrm{num}}$ (LBM convertido).

Limites verticais ajustados pelo intervalo entre percentis 2 e 98 da curva,
com margem de 15 %.

---

## 10. Resumo do fluxo de dados

$$
\boxed{
\begin{array}{rcl}
\text{JSON} & \longrightarrow & \{M,\,\mathrm{Da},\,\mathrm{Ca},\,\mathrm{Ca}_m,\,\Lambda,\,\alpha\}\\[4pt]
\text{VTKs / cache} & \longrightarrow & \{(t_i,\,A_i)\}_{i=1}^{N} \\[4pt]
\text{Fit em } \ln A & \longrightarrow & s_{\mathrm{num}} \\[4pt]
\text{Eq. 9} & \longrightarrow & s_{\mathrm{ana}} = \zeta_{\mathrm{ana}}\,U_{\mathrm{ref}}/L_{\mathrm{ref}} \\[4pt]
\text{Comparação} & \longrightarrow & \varepsilon = \dfrac{|s_{\mathrm{num}}-s_{\mathrm{ana}}|}{|s_{\mathrm{ana}}|}\times 100\%
\end{array}}
$$

---

## 11. Critérios de qualidade da validação

| Indicador | Valor recomendado | Implicação se violado |
|---|---|---|
| $k\xi = 2\pi m W/N_y$ | $< 0{,}01$ | Erro de fase-field $>$ 5 % |
| $W$ (interface width) | $\geq 3$ l.u. | Viola Nyquist (D2Q9) |
| Amplitude inicial | $\ll W$ (≈ $W/5$) | Sai do regime LSA puro |
| Erro relativo $\varepsilon$ | $< 10\%$ | Validação aceitável |
| Pontos no platô | $\geq 6$ | Polyfit instável |

---

## 12. Exemplos de uso

```bash
# Detecção automática do platô (default)
python post_process/valida_lsa.py OpcaoAC_W3_amp10_kxi0p018_d12mes05-h10_min52

# Janela manual
python post_process/valida_lsa.py <case_dir> --t0 50 --t1 400

# Forçar fallback 30%-75% sem detecção de platô
python post_process/valida_lsa.py <case_dir> --no-auto
```

**Saídas produzidas no diretório do caso:**

- `comparacao_lsa.png` — figura de validação (2 painéis).
- Relatório textual impresso no `stdout`.
