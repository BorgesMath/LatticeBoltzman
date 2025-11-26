# Simulação LBM: Instabilidade de Saffman-Taylor em Meios Porosos

Este repositório contém uma implementação numérica em C++ para simular o deslocamento de fluidos miscíveis em meios porosos, focando na **Instabilidade de Saffman-Taylor** (digitação viscosa).

O código utiliza o **Método de Lattice Boltzmann (LBM)** com uma abordagem de duas populações (Double Distribution Function): uma para a hidrodinâmica (Navier-Stokes) e outra para o transporte de massa (Advecção-Difusão).

## 1. Descrição do Problema Físico

A instabilidade de Saffman-Taylor ocorre quando um fluido menos viscoso (ex: CO2 supercrítico ou água) é injetado para deslocar um fluido mais viscoso (ex: óleo) em um meio poroso. Devido à diferença de mobilidade, a interface entre os fluidos torna-se instável, formando "dedos" (fingering) que reduzem a eficiência da varredura.

O modelo resolve o escoamento de uma mistura binária incompressível e miscível. A viscosidade local da mistura, $\nu(C)$, depende da concentração local $C$ do fluido invasor.

## 2. Método Numérico: Lattice Boltzmann (D2Q9)

O domínio é discretizado em uma rede quadrada (lattice) D2Q9, com 9 velocidades discretas $\mathbf{c}_i$:
* $i=0$: Repouso.
* $i=1..4$: Vizinhos diretos (distância 1).
* $i=5..8$: Vizinhos diagonais (distância $\sqrt{2}$).

### 2.1. Hidrodinâmica (Campo de Velocidade)
Para resolver as equações de Navier-Stokes (continuidade e momentum), utilizamos a função distribuição $f_i(\mathbf{x}, t)$.

A evolução é governada pela equação de Boltzmann discretizada com o operador de colisão **TRT (Two-Relaxation-Time)**. O TRT é superior ao BGK padrão para meios porosos, pois permite ajustar a viscosidade independentemente do erro numérico de fronteira.

$$f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) - f_i(\mathbf{x}, t) = \Omega^{TRT}_i(f)$$

O operador de colisão TRT decompõe a função distribuição em partes simétrica ($f^s$) e assimétrica ($f^a$):

$$f_i^{new} = f_i - \frac{1}{\tau^+}(f_i^s - f_i^{eq,s}) - \frac{1}{\tau^-}(f_i^a - f_i^{eq,a})$$

Onde:
* $\tau^+$: Relacionado à viscosidade física cinemática $\nu$.
    $$\nu = c_s^2 (\tau^+ - 0.5)$$
* $\tau^-$: Controla a estabilidade numérica e a precisão nas paredes (parâmetro mágico $\Lambda$).
    $$\Lambda = (\tau^+ - 0.5)(\tau^- - 0.5)$$

A função de equilíbrio $f_i^{eq}$ é dada por uma expansão de Maxwell-Boltzmann de segunda ordem:

$$f_i^{eq} = w_i \rho \left[ 1 + \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_i \cdot \mathbf{u})^2}{2c_s^4} - \frac{\mathbf{u}^2}{2c_s^2} \right]$$

As variáveis macroscópicas são recuperadas por momentos:
$$\rho = \sum f_i, \quad \rho \mathbf{u} = \sum f_i \mathbf{c}_i$$

### 2.2. Transporte de Espécies (Campo de Concentração)
Para rastrear a concentração do fluido invasor $C$ (onde $C=1$ é o fluido invasor e $C=0$ é o fluido residente), utilizamos uma segunda função de distribuição $g_i(\mathbf{x}, t)$.

Esta segue a **Equação de Advecção-Difusão**:
$$\frac{\partial C}{\partial t} + \nabla \cdot (\mathbf{u}C) = \nabla \cdot (D \nabla C)$$

A evolução de $g_i$ utiliza um operador BGK simples (relaxação única), pois a difusividade numérica é menos crítica que a viscosidade neste contexto:

$$g_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) = g_i(\mathbf{x}, t) - \frac{1}{\tau_g} (g_i - g_i^{eq})$$

O equilíbrio para concentração é simplificado para recuperar a advecção:
$$g_i^{eq} = w_i C \left[ 1 + \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^2} \right]$$

A difusividade molecular $D$ está relacionada ao tempo de relaxação $\tau_g$:
$$D = c_s^2 (\tau_g - 0.5)$$

Onde $C$ é calculado como:
$$C = \sum g_i$$

## 3. Acoplamento Físico (Lei de Mistura)

A característica central da simulação é que a viscosidade não é constante. Ela varia espacialmente dependendo da concentração $C$. O código utiliza uma lei de mistura logarítmica (similar à lei de Arrhenius para misturas ideais), que é fisicamente realista para misturas de fluidos:

$$\ln(\nu_{mix}) = C \ln(\nu_{invasor}) + (1-C) \ln(\nu_{residente})$$
Ou, na forma exponencial implementada:
$$\nu_{loc}(C) = \nu_{residente} \left( \frac{\nu_{invasor}}{\nu_{residente}} \right)^C$$

* Quando $C=0$ (óleo puro), $\nu_{loc} = \nu_{residente}$.
* Quando $C=1$ (CO2 puro), $\nu_{loc} = \nu_{invasor}$.

A **Razão de Viscosidade** ($M$) é definida como:
$$M = \frac{\nu_{residente}}{\nu_{invasor}}$$
Se $M > 1$, o escoamento é instável (Saffman-Taylor). Quanto maior o $M$, mais violentos e ramificados são os dedos.

## 4. Números Adimensionais

A simulação é controlada por:
1.  **Número de Reynolds ($Re$)**: Razão entre forças inerciais e viscosas. Mantido baixo ($Re < 1$) para simular escoamento de Darcy (creeping flow) típico de reservatórios.
2.  **Número de Péclet ($Pe$)**: Razão entre transporte advectivo e difusivo.
    $$Pe = \frac{U L}{D}$$
    Um $Pe$ alto mantém as interfaces nítidas (pouca difusão), favorecendo a instabilidade.
3.  **Razão de Mobilidade/Viscosidade ($M$)**: Define a intensidade da instabilidade.

## 5. Condições de Contorno e Geometria

* **Entrada (Oeste, x=0)**:
    * Velocidade constante $U_{inlet}$ (Perfil plano com perturbação Zou-He).
    * Concentração fixa $C=1$ (com pequena perturbação senoidal para "gatilhar" a instabilidade).
* **Saída (Leste, x=L)**:
    * Condição de Neumann nula (gradiente zero) para velocidade e concentração, permitindo que o fluido saia livremente.
* **Paredes e Obstáculos**:
    * Bounce-back (rebote) nas superfícies sólidas para garantir a condição de não-deslizamento ($\mathbf{u}=0$) e fluxo de massa nulo ($\partial C / \partial n = 0$).

## 6. Geração do Meio Poroso

O meio poroso é gerado estocasticamente:
1.  Distribuição aleatória de obstáculos baseada na porosidade alvo.
2.  Suavização via Autômato Celular para agrupar os obstáculos sólidos, criando "grãos" e canais de fluxo mais realistas, evitando nós sólidos isolados (singletons).