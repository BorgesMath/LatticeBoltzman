# Documentação Técnica: Inicialização (`initialization.py`)

O módulo `initialization.py` é responsável por construir o **Problema de
Valor Inicial (PVI)** da simulação. Ele aloca todos os tensores de campo e
estabelece o estado microscópico e macroscópico em $t = 0$, de forma a
**evitar transientes artificiais** ao iniciar o solver LBM.

A função principal `initialize_fields(params)` constrói:

1. Uma **interface difusa perturbada harmonicamente** (modo $m$).
2. Um **perfil analítico de pressão de Darcy** que sustenta o regime
   permanente desde o passo zero.
3. As populações $f_i$ em **equilíbrio local com velocidade $u_{\mathrm{inlet}}$**,
   não em repouso.

---

## 1. Fundamentos teóricos

### 1.1 Por que inicializar com cuidado?

O LBM, sendo um esquema explícito, é **muito sensível a desequilíbrios
iniciais**. Inicializações ruins produzem:

- **Ondas acústicas espúrias** que reverberam pelo domínio.
- **Transientes longos** antes que o regime físico se estabeleça.
- **Crescimento ruidoso da interface**, mascarando o modo $m$ desejado.

A estratégia adotada é **eliminar simultaneamente** três fontes de
transiente:

| Fonte de transiente | Mitigação |
|---|---|
| Velocidade inicial nula → aceleração brusca | $u_x(\mathbf{x}, 0) = u_{\mathrm{inlet}}$ uniforme |
| Pressão uniforme → fluxo de Darcy não estabelecido | Perfil $\rho(x)$ resolvido analiticamente |
| $f_i$ em equilíbrio de repouso → desbalanço com $\mathbf{u}$ | $f_i^{\mathrm{eq}}(\rho_{\mathrm{base}}, u_{\mathrm{inlet}})$ |

### 1.2 Perfil de Darcy unidimensional

Para um fluxo monofásico uniforme em meio poroso com permeabilidade $K_0$,
a lei de Darcy fornece:

$$
u \;=\; -\frac{K_0}{\mu}\,\frac{dp}{dx}.
$$

Em LBM com pressão $p = c_s^{2}\,\rho = \rho/3$, invertendo para a vazão
prescrita $u = u_{\mathrm{inlet}}$:

$$
\boxed{\;\frac{dp}{dx} \;=\; -\frac{\mu\,u_{\mathrm{inlet}}}{K_0}
\;\Longrightarrow\;
\frac{d\rho}{dx} \;=\; -\frac{3\,\nu\,u_{\mathrm{inlet}}}{K_0}\;}
$$

Como há **dois fluidos com viscosidades diferentes** separados em $x_{\mathrm{center}}$,
o gradiente é constante por partes:

$$
\frac{d\rho}{dx} \;=\;
\begin{cases}
-\,3\,\nu_{\mathrm{in}}\,u_{\mathrm{inlet}}/K_0 & \text{se } x < x_{\mathrm{center}} \\
-\,3\,\nu_{\mathrm{out}}\,u_{\mathrm{inlet}}/K_0 & \text{se } x \geq x_{\mathrm{center}}
\end{cases}
$$

> **Observação de sinal:** o código usa `dpdx_in/out = +3·ν·u/K` (positivo)
> e **integra do outlet para o inlet** subtraindo, o que equivale a usar
> $-d\rho/dx$ ao varrer $x$ em ordem reversa. Resultado físico: $\rho$ é
> **maior no inlet** (pressão mais alta) e $\rho = 1$ no outlet (pressão de
> referência).

### 1.3 Interface difusa perturbada

A posição da interface em função de $y$ é:

$$
x_{\mathrm{int}}(y) \;=\; x_{\mathrm{center}} \;+\; A\,\cos\!\left(\frac{2\pi m\,y}{N_y}\right),
$$

e o **perfil de equilíbrio** de Cahn–Hilliard atravessando essa interface é:

$$
\phi(x, y) \;=\; -\tanh\!\left(\frac{x - x_{\mathrm{int}}(y)}{W}\right),
$$

onde $W$ é a `INTERFACE_WIDTH` (espessura difusa). Convenção:

- $\phi \approx +1$ para $x < x_{\mathrm{int}}$ → **fluido invasor** (esquerda).
- $\phi \approx -1$ para $x > x_{\mathrm{int}}$ → **fluido residente** (direita).

> **Nota:** no código atual, o argumento do `tanh` é
> `(x - dist) / interface_width` (e não `/(W/2)` como em versões antigas).
> Isso significa que `INTERFACE_WIDTH` aqui é o **fator de escala direto**
> da tangente hiperbólica.

### 1.4 Equilíbrio com fluxo prescrito

A inicialização de $f_i$ usa a **distribuição de Maxwell discreta completa**,
não a forma simplificada $f_i = w_i \rho$:

$$
f_i(\mathbf{x}, 0) \;=\; w_i\,\rho_{\mathrm{base}}(x)\,
\bigl[1 + 3\,c_u + \tfrac{9}{2}\,c_u^{2} - \tfrac{3}{2}\,u_{\mathrm{inlet}}^{2}\bigr],
$$

com $c_u = c_{i,x}\,u_{\mathrm{inlet}}$. Isto garante:

- $\sum_i f_i = \rho_{\mathrm{base}}(x)$ ✓
- $\sum_i f_i\,c_{i,x} = \rho_{\mathrm{base}}(x)\,u_{\mathrm{inlet}}$ ✓
- Sem componente transverso espúrio ✓

---

## 2. Assinatura e estrutura

```python
initialize_fields(params)
    └──► _init_kernel(...)   # @njit paralelo
```

### 2.1 Função `initialize_fields(params)`

Recebe um dicionário `params` (vindo da configuração `main.py`) e retorna:

```python
return (f_a, f_b), (phi_a, phi_b), psi, rho, u_x, u_y, K_field, (Fx, Fy, mu_buffer)
```

### 2.2 Significado dos buffers retornados

| Buffer | Shape | Função |
|---|---|---|
| `f_a`, `f_b` | $(N_y, N_x, 9)$ | Populações LBM (double-buffer ping-pong). |
| `phi_a`, `phi_b` | $(N_y, N_x)$ | Campo de fase (double-buffer para Cahn–Hilliard). |
| `psi` | $(N_y, N_x)$ | Potencial magnético escalar **perturbador** $\tilde\psi$. |
| `rho` | $(N_y, N_x)$ | Densidade local (com perfil de Darcy embutido). |
| `u_x`, `u_y` | $(N_y, N_x)$ | Velocidade física. |
| `K_field` | $(N_y, N_x)$ | Permeabilidade local. |
| `Fx`, `Fy` | $(N_y, N_x)$ | Buffers de força (zerados; preenchidos pelo LBM). |
| `mu_buffer` | $(N_y, N_x)$ | Buffer de potencial químico para o módulo CH. |

---

## 3. ETAPA 1 — Perfil analítico de Darcy (linhas 41–53)

### 3.1 Cálculo das viscosidades

```python
nu_in  = (TAU_IN  - 0.5) / 3.0
nu_out = (TAU_OUT - 0.5) / 3.0
```

Vindas diretamente da relação $\nu = c_s^{2}(\tau - 1/2)$ do LBM.

### 3.2 Gradientes de pressão por região

```python
dpdx_in  = 3.0 * (nu_in  / K_0) * u_inlet
dpdx_out = 3.0 * (nu_out / K_0) * u_inlet
```

Em unidades de rede, $p = \rho/3$, então `dpdx` aqui já é o **incremento
de $\rho$ por nó** ao caminhar contra o fluxo.

### 3.3 Integração cumulativa do outlet para o inlet

```python
rho_base = np.ones(nx)              # outlet ancorado em ρ = 1
for x in range(nx - 2, -1, -1):
    if x >= x_center:
        rho_base[x] = rho_base[x + 1] + dpdx_out
    else:
        rho_base[x] = rho_base[x + 1] + dpdx_in
```

**Lógica:**

- $\rho(N_x - 1) = 1$ (condição de referência no outlet, consistente com a
  BC Zou–He de pressão fixa no LBM).
- Caminhando para trás, em cada nó soma-se o incremento de pressão
  apropriado à fase local.
- Para $x \geq x_{\mathrm{center}}$ (região do residente), usa $dpdx_{\mathrm{out}}$.
- Para $x < x_{\mathrm{center}}$ (região do invasor), usa $dpdx_{\mathrm{in}}$.

**Resultado:** um perfil $\rho(x)$ linear por partes, com **inclinação que
muda em $x_{\mathrm{center}}$** refletindo o salto de viscosidade.

> **Por que isso elimina o transiente?** Sem este perfil, o solver LBM
> levaria centenas de timesteps para "construir" a queda de pressão de
> Darcy à medida que o atrito viscoso se manifesta. Inicializando já com
> ela, o regime permanente está estabelecido em $t = 0$.

---

## 4. ETAPA 2 — Alocação de buffers (linhas 56–73)

### 4.1 Double-buffer (ping-pong)

`f_a` e `f_b` são alocados como buffers idênticos. Durante a simulação, o
driver alterna entre eles para evitar race conditions no streaming do LBM.
Idem para `phi_a` e `phi_b` na Cahn–Hilliard.

### 4.2 Buffers zerados

| Buffer | Inicialização | Razão |
|---|---|---|
| `u_y` | `zeros` | Sem velocidade transversal inicial. |
| `Fx`, `Fy` | `zeros` | Forças serão calculadas no primeiro `lbm_step`. |
| `mu_buffer` | `zeros` | Será preenchido no primeiro `cahn_hilliard_substep`. |

### 4.3 Permeabilidade

```python
K_field = np.ones((ny, nx)) * K_0
```

**Homogênea** ($K(\mathbf{x}) = K_0$ em todo o domínio). Para meios
heterogêneos, este campo seria modificado após esta chamada.

### 4.4 Campo magnético de fundo

```python
angle_rad = np.radians(H_ANGLE)
Hx = H0 * np.cos(angle_rad)
Hy = H0 * np.sin(angle_rad)
```

Decompõe o vetor de campo magnético **uniforme** $\mathbf{H}_0$ pelo ângulo
$\theta$ (em graus) entre o campo e a direção $\hat{x}$:

- $\theta = 0°$: campo na direção do fluxo (tangencial ao dedo).
- $\theta = 90°$: campo normal ao fluxo (acelera Saffman–Taylor).

Estes valores são **escalares**, passados ao LBM como `Hx_fundo, Hy_fundo`
da decomposição $\mathbf{H}_{\mathrm{total}} = \mathbf{H}_0 - \nabla\tilde\psi$.

---

## 5. ETAPA 3 — Kernel `_init_kernel` (linhas 11–30)

Loop paralelo em $y$:

### 5.1 Cálculo da posição da interface

```python
dist = x_center + amplitude * np.cos(2.0 * np.pi * mode_m * y / ny)
```

Isto é $x_{\mathrm{int}}(y)$, calculado uma vez por linha. Como $y$ é
**implicitamente periódico** (cosseno com $2\pi m$ no expoente), não há
descontinuidade entre $y = 0$ e $y = N_y - 1$.

### 5.2 Perfil de fase

```python
phi[y, x] = -np.tanh((x - dist) / interface_width)
```

Para cada nó $(y, x)$, distância sinalizada à interface dividida pela
espessura difusa. O sinal negativo coloca o invasor à esquerda
($\phi = +1$ quando $x \ll x_{\mathrm{int}}$).

### 5.3 Potencial magnético perturbador zerado

```python
psi[y, x] = 0.0  # Anulação para isolamento da perturbação
```

> **Mudança crucial em relação a versões antigas:** $\tilde\psi$ é
> inicializado a **zero**, não como uma rampa $H_0(L_x - x)$.
>
> **Razão:** o $\psi$ neste código representa **apenas a perturbação**
> sobre o campo de fundo uniforme $\mathbf{H}_0$ (decomposição de
> Helmholtz). Como em $t = 0$ a interface ainda não distorceu o campo,
> a perturbação é nula. O campo total é simplesmente $\mathbf{H} = \mathbf{H}_0$.
>
> Isso **isola limpamente o efeito da perturbação magnética** sem
> contaminação por gradientes iniciais artificiais.

### 5.4 Velocidade inicial não nula

```python
u_x[y, x] = u_inlet
```

> **Outra mudança crucial:** o fluido começa em **movimento uniforme**, não
> em repouso. Combinado com o perfil de Darcy em $\rho$, isso garante que
> o regime permanente já esteja satisfeito em $t = 0$.

### 5.5 Injeção do perfil $\rho_{\mathrm{base}}$

```python
rho_local = rho_base[x]
rho[y, x] = rho_local
```

Cada linha $y$ recebe o **mesmo** perfil $\rho(x)$ (problema é
quase-1D na base, perturbado apenas na interface).

### 5.6 Equilíbrio completo das populações

```python
for i in range(9):
    cu = CX[i] * u_inlet
    f[y, x, i] = W_LBM[i] * rho_local * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
```

Maxwell discreto truncado em $\mathcal{O}(u^2)$, idêntico ao usado no
operador BGK do `lbm_step`. Como $u_y = 0$, apenas a componente $c_{i,x}$
contribui ao produto $\mathbf{c}_i\!\cdot\!\mathbf{u}$.

---

## 6. ETAPA 4 — Sincronização dos buffers ping-pong (linhas 79–80)

```python
f_b[:]   = f_a[:]
phi_b[:] = phi_a[:]
```

Após o kernel preencher `f_a` e `phi_a`, **cópia explícita** para os buffers
companheiros. Isso garante que o primeiro passo do driver possa ler de
qualquer um dos dois buffers sem encontrar lixo de memória.

---

## 7. Resumo do estado em $t = 0$

| Campo | Valor inicial | Onde |
|---|---|---|
| $\phi(x,y)$ | $-\tanh\!\bigl((x - x_{\mathrm{int}}(y))/W\bigr)$ | Interface difusa perturbada |
| $\tilde\psi(x,y)$ | $0$ | Sem distorção magnética |
| $\mathbf{H}_0$ | $(H_0\cos\theta,\,H_0\sin\theta)$ | Uniforme (passado ao LBM) |
| $u_x(x,y)$ | $u_{\mathrm{inlet}}$ | Movimento permanente |
| $u_y(x,y)$ | $0$ | Sem transversal |
| $\rho(x,y)$ | $\rho_{\mathrm{base}}(x)$ | Perfil de Darcy por partes |
| $K(x,y)$ | $K_0$ | Meio poroso homogêneo |
| $f_i(x,y)$ | $f_i^{\mathrm{eq}}(\rho_{\mathrm{base}}, u_{\mathrm{inlet}})$ | Equilíbrio com fluxo |
| $\mathbf{F}$ | $0$ | Forças calculadas no 1º LBM step |
| $\mu_c$ | $0$ | Calculado no 1º CH substep |

---

## 8. Parâmetros lidos de `params`

| Chave | Significado | Usado em |
|---|---|---|
| `NY`, `NX` | Dimensões do domínio | Alocação |
| `U_INLET` | Velocidade de injeção | Perfil $u_x$, perfil $\rho$, $f_i^{\mathrm{eq}}$ |
| `TAU_IN`, `TAU_OUT` | Tempos de relaxação | $\nu \Rightarrow$ gradiente de Darcy |
| `K_0` | Permeabilidade base | $K_{\mathrm{field}}$ e Darcy |
| `mode_m` | Modo da perturbação | $\cos(2\pi m y/N_y)$ |
| `amplitude` | Amplitude da perturbação | Termo $A$ em $x_{\mathrm{int}}(y)$ |
| `INTERFACE_WIDTH` | Espessura difusa $W$ | Argumento do $\tanh$ |
| `H0` | Magnitude do campo magnético | $\mathbf{H}_0$ |
| `H_ANGLE` | Ângulo do campo (graus) | $\theta \Rightarrow (H_x, H_y)$ |

**Constante fixa:** `x_center = 80.0` (linha 36). Buffer de 80 nós entre
o inlet e a interface inicial — distância suficiente para acomodar
perturbações sem que a interface "entre" pela BC.

---

## 9. Acoplamento com o restante da simulação

```
        ┌──────────────────┐
        │  params (JSON)   │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ initialize_fields│  ← este módulo
        └────────┬─────────┘
                 │ retorna todos os tensores
                 ▼
   ┌─────────────────────────────────┐
   │       Loop temporal principal   │
   │  ┌───────────────────────────┐  │
   │  │ lbm_step                  │  │
   │  │  ├─ usa rho, u, K, Fx,Fy  │  │
   │  │  └─ usa Hx_fundo,Hy_fundo │  │
   │  ├───────────────────────────┤  │
   │  │ poisson_solve (ψ̃)         │  │
   │  ├───────────────────────────┤  │
   │  │ cahn_hilliard_substep × N │  │
   │  └───────────────────────────┘  │
   └─────────────────────────────────┘
```

### 9.1 Por que double-buffer já na inicialização?

Porque os módulos `lbm_step` e `cahn_hilliard_substep` precisam, no
primeiro passo, ler de **um buffer** e escrever em **outro**. Sem buffers
companheiros já alocados e populados, o primeiro passo travaria.

---

## 10. Considerações numéricas

### 10.1 Critérios de qualidade da inicialização

| Critério | Valor recomendado | Razão |
|---|---|---|
| $x_{\mathrm{center}} \geq 3\,W$ | sempre | Interface não toca o inlet. |
| $x_{\mathrm{center}} + A \leq N_x - 3\,W$ | sempre | Interface não toca o outlet. |
| $A \ll W$ | $A \lesssim W/5$ | Regime LSA puro (perturbação infinitesimal). |
| $2\pi m W / N_y$ | $< 0.01$ | Erro de fase-field $< 5\%$. |
| $u_{\mathrm{inlet}} \cdot N_x < $ fluxo conservado | sempre | Sanity check de volume injetado. |

### 10.2 Verificação rápida do perfil de Darcy

Após a inicialização, o salto de densidade do inlet ao outlet deve ser:

$$
\Delta\rho \;=\; \rho(0) - \rho(N_x-1)
\;\approx\; \frac{3\,u_{\mathrm{inlet}}}{K_0}\,
\bigl[\nu_{\mathrm{in}}\,x_{\mathrm{center}} + \nu_{\mathrm{out}}\,(N_x - x_{\mathrm{center}})\bigr].
$$

Se este valor for $\gtrsim 0.1$, o LBM pode sair do regime quasi-incompressível
($\rho \approx 1$). Nesses casos: aumentar $K_0$ ou reduzir $u_{\mathrm{inlet}}$.

### 10.3 Paralelização

`prange` sobre $y$ no kernel. Como cada linha é **independente** (toda a
informação do nó $(y, x)$ vem de `params` e de `rho_base[x]`), a
paralelização é trivial e escala linearmente com núcleos.

---

## 11. Resumo executivo

| Componente | Implementação | Local |
|---|---|---|
| Perfil de Darcy $\rho(x)$ | Integração cumulativa reversa, $dpdx = 3\nu u/K$ | Linhas 41–53 |
| Interface perturbada $\phi$ | $-\tanh\!\bigl((x - x_{\mathrm{int}}(y))/W\bigr)$ | Linha 20 |
| $\tilde\psi$ inicial | Zero (decomposição de Helmholtz) | Linha 21 |
| $u_x$ inicial | $u_{\mathrm{inlet}}$ uniforme | Linha 22 |
| $f_i$ inicial | Equilíbrio completo com $u_{\mathrm{inlet}}$ | Linhas 28–30 |
| $\mathbf{H}_0$ | $(H_0\cos\theta,\,H_0\sin\theta)$ | Linhas 71–73 |
| Double-buffer | $f_a/f_b$, $\phi_a/\phi_b$ | Linhas 56–59, 79–80 |
| Periodicidade em $y$ | Implícita via $\cos(2\pi m y/N_y)$ | Linha 17 |
