# allee-effect-dirichlet-bvp

Trabajo final de **Métodos Topológicos para Análisis No Lineal** (Pablo Amster,
FCEN–UBA).

Estudio de un problema de contorno de Dirichlet con no linealidad de tipo
**efecto Allee**, combinando métodos topológicos (punto fijo, sub y
supersoluciones, grado de Leray–Schauder, función implícita en espacios de
Banach) con exploración numérica que ilustra cada enunciado.

> **Estado del repositorio: esqueleto.** Todas las funciones de `src/` están
> declaradas con su firma y su docstring pero levantan `NotImplementedError`.
> Los tests de `tests/` **sí** están escritos completos: funcionan como
> *especificación ejecutable*, y por lo tanto fallan hasta que se implemente
> cada función.

---

## El problema

$$
\begin{cases}
-u''(t) = \lambda\, u(t)\,\bigl(u(t)-\alpha\bigr)\bigl(1-u(t)\bigr), & t\in(0,1),\\
u(0)=u(1)=0,
\end{cases}
\qquad \lambda>0,\quad 0<\alpha<\tfrac12 .
$$

**Interpretación.** $u$ es la densidad (normalizada) de una población en un
hábitat unidimensional aislado, en estado estacionario. La reacción

$$
f(u)=u(u-\alpha)(1-u)
$$

modela un **efecto Allee fuerte**: la tasa de crecimiento per cápita
$f(u)/u=(u-\alpha)(1-u)$ es *negativa* por debajo del umbral $\alpha$ (a
densidad baja los individuos no logran reproducirse: falta de pareja,
cooperación insuficiente, dilución) y positiva entre $\alpha$ y la capacidad de
carga $1$. La condición de Dirichlet homogénea es un **borde letal**: todo
individuo que alcanza $t=0$ o $t=1$ se pierde. El parámetro $\lambda$ mide el
tamaño del hábitat: reescalando $t\mapsto t/L$, el intervalo $(0,L)$ con
difusión unitaria es equivalente a $(0,1)$ con $\lambda=L^2$.

La pregunta biológica —*¿cuán grande debe ser una reserva para que una especie
con efecto Allee persista?*— es exactamente la pregunta matemática por la
existencia de soluciones positivas.

### Notación

| Símbolo | Definición |
|---|---|
| $f(u)$ | $u(u-\alpha)(1-u) = -u^3+(1+\alpha)u^2-\alpha u$ |
| $F(u)$ | $\int_0^u f = -\tfrac{u^4}{4}+\tfrac{(1+\alpha)u^3}{3}-\tfrac{\alpha u^2}{2}$ |
| $\theta(\alpha)$ | único cero no trivial de $F$ en $(\alpha,1)$ |
| $M(\alpha)$ | $\max_{0<u\le 1} f(u)/u=(1-\alpha)^2/4$, alcanzado en $u=(1+\alpha)/2$ |
| $G(t,s)$ | función de Green de $-d^2/dt^2$ con Dirichlet en $(0,1)$ |
| $T_\lambda$ | operador $(T_\lambda u)(t)=\lambda\int_0^1 G(t,s)\,\tilde f(u(s))\,ds$ |
| $\varphi_1,\lambda_1$ | $\varphi_1(t)=\sin(\pi t)$, $\lambda_1=\pi^2$ (primer par propio de Dirichlet) |

con

$$
G(t,s)=\begin{cases} s(1-t), & 0\le s\le t\le 1,\\ t(1-s), & 0\le t\le s\le 1,\end{cases}
\qquad G\ge 0,\quad G(t,s)=G(s,t),\quad \max G=\tfrac14 .
$$

Dos identidades que se usan todo el tiempo:

$$
F(1)=\frac{1-2\alpha}{12}, \qquad \int_0^1 G(t,s)\,ds=\frac{t(1-t)}{2}.
$$

La primera explica por qué se pide $\alpha<\tfrac12$: si $\alpha\ge\tfrac12$
entonces $F\le 0$ en todo $(0,1]$, y la identidad de energía prohíbe soluciones
positivas para **todo** $\lambda$ (la población nunca compensa la pérdida por
el borde).

---

## Los cinco resultados

### 1. Cotas a priori y formulación de punto fijo

Sea $\tilde f$ la **truncada** de $f$ fuera de $[0,1]$ ($\tilde f\equiv 0$ en
$u<0$ y en $u>1$), que es acotada y Lipschitz.

> **Teorema.** Toda solución del problema truncado satisface $0\le u\le 1$; en
> consecuencia $\tilde f(u)=f(u)$ y $u$ resuelve el problema original.
> Recíprocamente, $u$ es solución si y sólo si $u=T_\lambda u$ en $C([0,1])$.

*Idea:* principio del máximo. Si $u<0$ en un subintervalo, allí $-u''=0$, luego
$u$ es afín y se anula en los extremos de ese subintervalo, contradicción;
análogamente para $u>1$. Como $\tilde f$ es acotada, $T_\lambda$ tiene imagen
acotada en $C^1$ y es **compacta** por Arzelà–Ascoli, así que Schauder se
aplica en cualquier bola grande. Las cotas a priori son lo que convierte al
problema truncado (dócil) en el original.

**Notebook:** `02_cotas_a_priori`.

### 2. No existencia de solución positiva para $\lambda$ chico

> **Teorema.** Si $\lambda<\dfrac{4\pi^2}{(1-\alpha)^2}$, el problema no tiene
> solución positiva no trivial.

*Idea:* multiplicar por la primera autofunción $\varphi_1=\sin(\pi t)$ e
integrar dos veces por partes:

$$
\pi^2\int_0^1 u\varphi_1
=\int_0^1(-u'')\varphi_1
=\lambda\int_0^1 f(u)\varphi_1
\le \lambda M(\alpha)\int_0^1 u\varphi_1 ,
$$

usando $f(u)\le M(\alpha)\,u$ en $[0,1]$, que vale por la cota a priori del
punto 1. Como $\int u\varphi_1>0$, resulta $\pi^2\le\lambda M(\alpha)$, es
decir $\lambda\ge \pi^2/M(\alpha)=4\pi^2/(1-\alpha)^2$. Biológicamente: **un
hábitat chico no sostiene a la población**, sin importar la densidad inicial.

**Notebook:** `03_umbral_no_existencia`.

### 3. Existencia para $\lambda$ grande vía sub y supersoluciones

> **Teorema.** Existe $\lambda_0(\alpha)$ tal que para todo
> $\lambda\ge\lambda_0$ el problema tiene una solución positiva.

$\psi\equiv 1$ es supersolución ($-\psi''=0=\lambda f(1)$ y $\psi\ge 0$ en el
borde). Se construye una subsolución $\varphi$ de **tres tramos**, empalmados
en $C^1$:

1. **tramo convexo** cerca del borde, donde $\varphi<\alpha$ y por lo tanto
   $f(\varphi)<0$: la desigualdad $-\varphi''\le\lambda f(\varphi)<0$ **obliga**
   a $\varphi''>0$, o sea la subsolución tiene que subir convexamente y rápido;
2. **tramo cóncavo** de transición, con $\alpha<\varphi<\beta$, donde
   $f(\varphi)>0$ deja margen para curvar hacia abajo;
3. **meseta** $\varphi\equiv\beta$ con $\beta\in(\alpha,1)$: allí
   $-\varphi''=0\le\lambda f(\beta)$ trivialmente.

> **Advertencia (el error clásico).** La construcción ingenua —una rampa lineal
> pegada directamente a la meseta— **no es subsolución**. En el empalme la
> derivada salta *hacia abajo* (esquina cóncava), lo que produce una delta de
> Dirac **negativa** en $\varphi''$ y viola $-\varphi''\le\lambda f(\varphi)$ en
> sentido distribucional. Sólo se admiten esquinas **convexas** (saltos de
> derivada hacia arriba). Por eso el empalme con la meseta debe ser $C^1$: la
> derivada tiene que llegar a cero de manera continua. Esto está codificado en
> `src/subsuper.py` y contrastado en `tests/test_subsuper.py`.

Con $\varphi\le\psi$, la **iteración monótona** $u_{n+1}=T_\lambda u_n$ desde
$u_0=\psi$ decrece a la solución **maximal** del intervalo de orden
$[\varphi,\psi]$.

**Notebooks:** `04_construccion_subsolucion`, `05_diagrama_bifurcacion`.

### 4. Tres soluciones vía grado de Leray–Schauder

> **Teorema.** Para $\lambda>\lambda^*(\alpha)$ el problema tiene al menos tres
> soluciones: la trivial $u\equiv0$ y dos soluciones positivas ordenadas.

*Idea:* con $\Phi_\lambda=I-T_\lambda$,

- $\deg(\Phi_\lambda,B_R,0)=1$ para $R$ grande, porque $T_\lambda$ tiene imagen
  acotada y es homotópico a $0$;
- $i(0)=+1$: el linealizado en $u\equiv0$ es $-v''+\lambda\alpha v$ (pues
  $f'(0)=-\alpha<0$), cuyos autovalores $k^2\pi^2+\lambda\alpha$ son todos
  positivos, así que $T'_\lambda(0)$ no tiene autovalores mayores que $1$;
- $\deg(\Phi_\lambda,\Omega,0)=1$ en un entorno abierto acotado $\Omega$ del
  intervalo de orden $[\varphi,1]$, donde vive la solución maximal $u_{\max}$;
- por **excisión y aditividad**, el grado en
  $B_R\setminus(\overline{\Omega}\cup \overline{B_\varepsilon(0)})$ vale
  $1-1-1=-1\ne 0$, y ahí aparece la **tercera** solución, de índice $-1$.

La solución intermedia (índice $-1$) es la **inestable**: es el umbral de
extinción. Por debajo de ella la población colapsa; por encima converge a
$u_{\max}$. Es la traducción matemática exacta del efecto Allee.

**Notebooks:** `06_tres_soluciones`, `07_indices_y_grado`.

### 5. Regularidad $C^1$ y monotonía de la rama maximal

Sean $X=\{u\in C^2([0,1]) : u(0)=u(1)=0\}$, $Y=C([0,1])$ y

$$
\Phi:\mathbb{R}\times X\to Y,\qquad \Phi(\lambda,u)=u''+\lambda f(u).
$$

Como $f$ es polinomial, el operador de Nemytskii asociado es $C^\infty$ y
$D_u\Phi(\lambda,u)v=v''+\lambda f'(u)v$.

> **Teorema.** Si $u_\lambda$ es solución y el primer autovalor $\mu_1$ del
> linealizado $L_\lambda v=-v''-\lambda f'(u_\lambda)v$ con Dirichlet es
> $\ne 0$, entonces $D_u\Phi$ es isomorfismo y el **teorema de la función
> implícita en espacios de Banach** da una rama local $\lambda\mapsto u_\lambda$
> de clase $C^1$ a valores en $X$. Sobre la rama maximal $\mu_1>0$, la rama se
> continúa hacia la derecha y $\|u_\lambda\|_\infty$ es creciente.

Derivando la ecuación respecto de $\lambda$ se obtiene la **ecuación de la
rama**

$$
L_\lambda \dot u = f(u_\lambda), \qquad \dot u(0)=\dot u(1)=0,
$$

y como $\mu_1>0$ el operador $L_\lambda^{-1}$ es positivo. El punto donde
$\mu_1=0$ es exactamente el **pliegue** (*fold*) del diagrama de bifurcación,
donde la función implícita deja de aplicarse y hay que pasar a continuación por
pseudo-longitud de arco.

*Punto delicado:* la monotonía **puntual** global no es inmediata, porque
$f(u_\lambda)$ cambia de signo en las capas límite donde $u_\lambda<\alpha$. En
el caso autónomo se la contrasta con el mapa del tiempo, que da la relación
exacta $\lambda=T(\rho)^2$.

**Notebook:** `05_diagrama_bifurcacion`.

### Herramienta transversal: el mapa del tiempo

Como la ecuación es autónoma, multiplicando por $u'$ se obtiene la conservación
de la energía $\tfrac12(u')^2+\lambda F(u)=\lambda F(\rho)$, con
$\rho=\|u\|_\infty=u(1/2)$. Integrando,

$$
T(\rho)=\sqrt{2}\int_0^{\rho}\frac{du}{\sqrt{F(\rho)-F(u)}},
\qquad\qquad \lambda=T(\rho)^2
$$

para $\rho\in(\theta(\alpha),1)$. La condición $\rho>\theta$ es necesaria para
que $F(\rho)>F(u)$ en $[0,\rho)$. Como $T\to\infty$ en ambos extremos del
intervalo ($\rho\to\theta^+$ y $\rho\to1^-$), $T$ tiene un mínimo interior y

$$
\lambda^*(\alpha)=\min_{\theta<\rho<1}T(\rho)^2 .
$$

Esto da el diagrama de bifurcación completo, en forma de **S** acostada: cero
soluciones positivas para $\lambda<\lambda^*$, una (degenerada) en
$\lambda=\lambda^*$, y exactamente dos para $\lambda>\lambda^*$. La desigualdad
$\lambda^*(\alpha)>4\pi^2/(1-\alpha)^2$ cierra el círculo con el resultado 2.

El integrando tiene una singularidad **integrable** en $u=\rho$ (donde
$F(\rho)-F(u)\sim f(\rho)(\rho-u)$); se la desactiva con la sustitución
$u=\rho-s^2$ o con cuadratura de Gauss–Chebyshev.

---

## Contenido de los notebooks

| Notebook | Qué hace |
|---|---|
| `01_no_linealidad` | Retrato de $f$, $f'$ y $F$; ceros $0,\alpha,1$ y $\theta(\alpha)$; verificación de $F(1)=(1-2\alpha)/12$ y de $M=(1-\alpha)^2/4$; por qué se pide $\alpha<1/2$. |
| `02_cotas_a_priori` | Función de Green y su núcleo; truncamiento $\tilde f$; el operador $T_\lambda$ y su compacidad; comprobación numérica de $0\le u\le1$ sobre soluciones obtenidas por *shooting*. |
| `03_umbral_no_existencia` | El argumento con $\varphi_1=\sin(\pi t)$; barrido en $\lambda$ mostrando que no aparece ninguna solución positiva por debajo de $4\pi^2/(1-\alpha)^2$; comparación de esa cota con el $\lambda^*$ exacto. |
| `04_construccion_subsolucion` | Construcción explícita de la subsolución de tres tramos; verificación tramo por tramo de la desigualdad diferencial y de la continuidad de $\varphi'$ en los empalmes; **contraejemplo** de la meseta con esquina cóncava. |
| `05_diagrama_bifurcacion` | Mapa del tiempo $T(\rho)$ y $\lambda^*$; diagrama $(\lambda,\|u\|_\infty)$ en forma de S por continuación de pseudo-longitud de arco; primer autovalor del linealizado a lo largo de la rama y localización del pliegue. |
| `06_tres_soluciones` | Para $\lambda>\lambda^*$ fijo: las tres soluciones superpuestas; iteración monótona desde $\psi\equiv1$ y desde la subsolución; la solución intermedia como umbral de extinción. |
| `07_indices_y_grado` | Índice de Leray–Schauder de cada solución contando autovalores negativos del linealizado; verificación de que $\sum_u i(u)=\deg=1$. |
| `08_caso_no_autonomo` | $-u''=\lambda\,a(t)f(u)$ con $a>0$ no constante: el mapa del tiempo deja de valer, pero sub/supersoluciones y grado sobreviven. Hábitat heterogéneo. |

---

## Estructura

```
allee-effect-dirichlet-bvp/
├── src/
│   ├── nonlinearity.py    f, f', F, M(α), cota inferior de λ, validación de α
│   ├── green.py           núcleo de Green, operador T_λ, truncamiento
│   ├── shooting.py        método de disparo: PVI, función de disparo, barrido
│   ├── timemap.py         θ(α), mapa del tiempo T(ρ), λ*(α)
│   ├── subsuper.py        subsolución de tres tramos y su verificación
│   ├── monotone.py        iteración monótona
│   ├── continuation.py    pseudo-longitud de arco, primer autovalor
│   ├── indices.py         índice de Leray–Schauder, suma = grado
│   └── plotting.py        estilo de figuras y guardado
├── notebooks/             01 … 08
├── tests/                 especificación ejecutable (pytest)
└── figuras/               salida de los notebooks (ignorada por git)
```

---

## Instalación y uso

```bash
git clone https://github.com/lucapetrarca/AlleeEffect-TopologicalMethods.git
cd AlleeEffect-TopologicalMethods

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1       # Windows (PowerShell)

pip install -r requirements.txt
```

Correr los tests (hoy fallan a propósito: son la especificación de lo que falta
implementar):

```bash
python -m pytest tests/ -v
```

Abrir los notebooks:

```bash
jupyter notebook notebooks/
```

Los notebooks importan desde `src/` agregando la raíz del repositorio al
`sys.path`; conviene lanzarlos desde la raíz del repositorio.

---

## Advertencia sobre las figuras

**Las figuras de este repositorio ilustran los teoremas; no los demuestran.**
Todo cálculo numérico —integración del PVI, cuadratura del mapa del tiempo,
conteo de autovalores, continuación— está sujeto a error de discretización y a
la elección de parámetros, y sólo explora un conjunto finito de valores de
$\lambda$ y $\alpha$. Las demostraciones son las de los cinco resultados
enunciados arriba y son independientes del código. El rol del código es exhibir
los objetos que las pruebas garantizan (la subsolución, la tercera solución, el
pliegue de la rama) y servir de control de sanidad sobre los enunciados.

---

## Referencias

- P. Amster, *Topological Methods in the Study of Boundary Value Problems*,
  Springer Universitext, 2014.
- H. Amann, *Fixed point equations and nonlinear eigenvalue problems in ordered
  Banach spaces*, SIAM Review 18 (1976), 620–709.
- K. Deimling, *Nonlinear Functional Analysis*, Springer, 1985.
- W. C. Allee, *Animal Aggregations: A Study in General Sociology*, University
  of Chicago Press, 1931.
- J. Shi y R. Shivaji, *Persistence in reaction diffusion models with weak Allee
  effect*, J. Math. Biol. 52 (2006), 807–829.
