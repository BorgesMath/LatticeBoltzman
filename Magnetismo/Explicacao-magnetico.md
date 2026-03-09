# Documentação Técnica: Solver Magnetostático (`poisson.py`)

Este módulo resolve a equação governante do potencial magnético escalar $\psi$ em um meio heterogêneo (ferrofluido + fluido não-magnético). Devido à variação espacial da suscetibilidade magnética $\chi(\phi)$, o problema é elíptico com coeficientes variáveis, exigindo um solver iterativo robusto.

A função `solve_poisson_magnetic` implementa o método de **Sobre-Relaxação Sucessiva (SOR)** otimizado com compilação JIT (`@njit`).

---

## 1. Formulação Matemática

Na ausência de correntes elétricas livres ($\mathbf{J}=0$) e assumindo o regime quase-estático, as equações de Maxwell se reduzem a:

1.  $\nabla \times \mathbf{H} = 0 \implies \mathbf{H} = -\nabla \psi$
2.  $\nabla \cdot \mathbf{B} = 0$
3.  $\mathbf{B} = \mu_0 (1 + \chi) \mathbf{H}$

Combinando estas equações, obtemos a **Equação de Poisson Generalizada** para a permeabilidade magnética variável $\mu_r(\mathbf{x}) = 1 + \chi(\mathbf{x})$:

$$\nabla \cdot (\mu_r(\phi) \nabla \psi) = 0$$

Expandida em 2D cartesiano:
$$\frac{\partial}{\partial x} \left( \mu_r \frac{\partial \psi}{\partial x} \right) + \frac{\partial}{\partial y} \left( \mu_r \frac{\partial \psi}{\partial y} \right) = 0$$

---

## 2. Discretização Numérica (Volume Finito)

Para resolver a EDP numericamente, utiliza-se um esquema conservativo de segunda ordem. A permeabilidade magnética $\mu_r$ não é definida no centro da célula $(y,x)$, mas sim nas **faces** do volume de controle, para garantir a continuidade do fluxo magnético.

### 2.1 Interpolação nas Faces

| Variável Código | Face do Volume | Definição Matemática (Média Aritmética) |
| :--- | :--- | :--- |
| `mu_E` | Leste ($i+1/2$) | $\mu_{i+1/2, j} \approx 1 + 0.5(\chi_{i,j} + \chi_{i+1,j})$ |
| `mu_W` | Oeste ($i-1/2$) | $\mu_{i-1/2, j} \approx 1 + 0.5(\chi_{i,j} + \chi_{i-1,j})$ |
| `mu_N` | Norte ($j+1/2$) | $\mu_{i, j+1/2} \approx 1 + 0.5(\chi_{i,j} + \chi_{i,j+1})$ |
| `mu_S` | Sul ($j-1/2$) | $\mu_{i, j-1/2} \approx 1 + 0.5(\chi_{i,j} + \chi_{i,j-1})$ |

### 2.2 Estêncil de 5 Pontos (Equação Discreta)
A equação discretizada resulta em um sistema linear da forma $A\psi = b$, onde o valor central $\psi_{y,x}$ depende dos seus 4 vizinhos ponderados pelas permeabilidades das faces:

$$\psi_{y,x}^* = \frac{\mu_E \psi_{y,x+1} + \mu_W \psi_{y,x-1} + \mu_N \psi_{y+1,x} + \mu_S \psi_{y-1,x}}{\mu_E + \mu_W + \mu_N + \mu_S}$$

No código:
* **Numerador:** Soma ponderada dos potenciais vizinhos.
* **Denominador (`denom`):** Soma das permeabilidades das faces.

---

## 3. Algoritmo SOR (Sobre-Relaxação Sucessiva)

O método de Gauss-Seidel padrão é lento para malhas grandes ($600 \times 300$). O SOR introduz um fator de aceleração $\omega$ (`SOR_OMEGA`) para "extrapolar" a correção em direção à solução final.

$$\psi_{y,x}^{new} = (1 - \omega) \psi_{y,x}^{old} + \omega \psi_{y,x}^*$$

| Parâmetro | Valor Típico | Comportamento |
| :--- | :--- | :--- |
| `SOR_OMEGA` | $1.85$ | **Sobre-relaxação:** Acelera a convergência em sistemas elípticos. Valores $> 2$ causam divergência; valores $< 1$ (sub-relaxação) são usados apenas para estabilidade extrema. |
| `range(15)` | Iterações Internas | Número fixo de varreduras SOR executadas a cada passo de tempo macroscópico da simulação principal. |

---

## 4. Condições de Contorno

O loop espacial percorre apenas o interior do domínio (`range(1, nx-1)` e `range(1, ny-1)`). As bordas são tratadas da seguinte forma:

### 4.1 Dirichlet Implícito (Entrada/Saída - Eixo X)
As colunas `x=0` e `x=nx-1` **não são atualizadas** dentro do loop. Elas mantêm os valores fixos definidos na inicialização (`initialization.py`):

* **Entrada ($x=0$):** $\psi = H_0 L_x$ (Potencial Alto)
* **Saída ($x=L$):** $\psi = 0$ (Terra)

Isso cria o **Campo Magnético Global $H_0$** que permeia o sistema.

### 4.2 Neumann Explícito (Paredes Laterais - Eixo Y)
Nas paredes superior e inferior, impõe-se que o fluxo magnético não saia do domínio (isolamento magnético ou simetria), o que implica derivada normal nula ($\frac{\partial \psi}{\partial y} = 0$).

```python
psi[0, :] = psi[1, :]       # Topo: Copia a linha de baixo
psi[ny-1, :] = psi[ny-2, :] # Base: Copia a linha de cima