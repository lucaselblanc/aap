From Paper: https://eprint.iacr.org/2019/266.pdf

# PAPER START

def shortgcd2(f,g):
  delta,f,g = 1,ZZ(f),ZZ(g)
  assert f&1
  m = 4+3*max(f.nbits(),g.nbits())
  for loop in range(m):
    if delta>0 and g&1: delta,f,g = -delta,g,-f
    delta,g = 1+delta,(g+(g&1)*f)//2
  return abs(f)

def divstepsx(n,t,delta,f,g):
  assert t >= n and n >= 0
  f,g = f.truncate(t),g.truncate(t)
  kx = f.parent()
  x = kx.gen()
  u,v,q,r = kx(1),kx(0),kx(0),kx(1)
  while n > 0:
    f = f.truncate(t)
    if delta > 0 and g[0] != 0: delta,f,g,u,v,q,r = -delta,g,f,q,r,u,v
    f0,g0 = f[0],g[0]
    delta,g,q,r = 1+delta,(f0*g-g0*f)/x,(f0*q-g0*u)/x,(f0*r-g0*v)/x
    n,t = n-1,t-1
    g = kx(g).truncate(t)
  M2kx = MatrixSpace(kx.fraction_field(),2)
  return delta,f,g,M2kx((u,v,q,r))

from divstepsx import divstepsx
def jumpdivstepsx(n,t,delta,f,g):
  assert t >= n and n >= 0
  kx = f.truncate(t).parent()
  if n <= 1: return divstepsx(n,t,delta,f,g)
  j = n//2
  delta,f1,g1,P1 = jumpdivstepsx(j,j,delta,f,g)
  f,g = P1*vector((f,g))
  f,g = kx(f).truncate(t-j),kx(g).truncate(t-j)
  delta,f2,g2,P2 = jumpdivstepsx(n-j,n-j,delta,f,g)
  f,g = P2*vector((f,g))
  f,g = kx(f).truncate(t-n+1),kx(g).truncate(t-n)
  return delta,f,g,P2*P1

from divstepsx import divstepsx
def gcddegree(R0,R1):
  d = R0.degree()
  assert d > 0 and d > R1.degree()
  f,g = R0.reverse(d),R1.reverse(d-1)
  delta,f,g,P = divstepsx(2*d-1,2*d-1,1,f,g)
  return delta//2
def gcdx(R0,R1):
  d = R0.degree()
  assert d > 0 and d > R1.degree()
  f,g = R0.reverse(d),R1.reverse(d-1)
  delta,f,g,P = divstepsx(2*d-1,3*d-1,1,f,g)
  return f.reverse(delta//2)/f[0]
def recipx(R0,R1):
  d = R0.degree()
  assert d > 0 and d > R1.degree()
  f,g = R0.reverse(d),R1.reverse(d-1)
  delta,f,g,P = divstepsx(2*d-1,2*d-1,1,f,g)
  if delta != 0: return
  kx = f.parent()
  x = kx.gen()
  return kx(x^(2*d-2)*P[0][1]/f[0]).reverse(d-1)

def truncate(f,t):
  if t == 0: return 0
  twot = 1<<(t-1)
  return ((f+twot)&(2*twot-1))-twot
def divsteps2(n,t,delta,f,g):
  assert t >= n and n >= 0
  f,g = truncate(f,t),truncate(g,t)
  u,v,q,r = 1,0,0,1
  while n > 0:
    f = truncate(f,t)
    if delta > 0 and g&1: delta,f,g,u,v,q,r = -delta,g,-f,q,r,-u,-v
    g0 = g&1
    delta,g,q,r = 1+delta,(g+g0*f)/2,(q+g0*u)/2,(r+g0*v)/2
    n,t = n-1,t-1
    g = truncate(ZZ(g),t)
  M2Q = MatrixSpace(QQ,2)
  return delta,f,g,M2Q((u,v,q,r))

from divsteps2 import divsteps2,truncate
def jumpdivsteps2(n,t,delta,f,g):
  assert t >= n and n >= 0
  if n <= 1: return divsteps2(n,t,delta,f,g)
  j = n//2
  delta,f1,g1,P1 = jumpdivsteps2(j,j,delta,f,g)
  f,g = P1*vector((f,g))
  f,g = truncate(ZZ(f),t-j),truncate(ZZ(g),t-j)
  delta,f2,g2,P2 = jumpdivsteps2(n-j,n-j,delta,f,g)
  f,g = P2*vector((f,g))
  f,g = truncate(ZZ(f),t-n+1),truncate(ZZ(g),t-n)
  return delta,f,g,P2*P1

from divsteps2 import divsteps2
def iterations(d):
  return (49*d+80)//17 if d<46 else (49*d+57)//17
def gcd2(f,g):
  assert f & 1
  d = max(f.nbits(),g.nbits())
  m = iterations(d)
  delta,fm,gm,P = divsteps2(m,m+d,1,f,g)
  return abs(fm)
def recip2(f,g):
  assert f & 1
  d = max(f.nbits(),g.nbits())
  m = iterations(d)
  precomp = Integers(f)((f+1)/2)^(m-1)
  delta,fm,gm,P = divsteps2(m,m+1,1,f,g)
  V = sign(fm)*ZZ(P[0][1]*2^(m-1))
  return ZZ(V*precomp)

earlybounds = { 0:1, 1:1, 2:689491/2^20, 3:779411/2^21,
  4:880833/2^22, 5:165219/2^20, 6:97723/2^20, 7:882313/2^24,
  8:306733/2^23, 9:92045/2^22, 10:439213/2^25, 11:281681/2^25,
  12:689007/2^27, 13:824303/2^28, 14:257817/2^27, 15:634229/2^29,
  16:386245/2^29, 17:942951/2^31, 18:583433/2^31, 19:713653/2^32,
  20:432891/2^32, 21:133569/2^31, 22:328293/2^33, 23:800421/2^35,
  24:489233/2^35, 25:604059/2^36, 26:738889/2^37, 27:112215/2^35,
  28:276775/2^37, 29:84973/2^36, 30:829297/2^40, 31:253443/2^39,
  32:625405/2^41, 33:95625/2^39, 34:465055/2^42, 35:286567/2^42,
  36:175951/2^42, 37:858637/2^45, 38:65647/2^42, 39:40469/2^42,
  40:24751/2^42, 41:240917/2^46, 42:593411/2^48, 43:364337/2^48,
  44:889015/2^50, 45:543791/2^50, 46:41899/2^47, 47:205005/2^50,
  48:997791/2^53, 49:307191/2^52, 50:754423/2^54, 51:57527/2^51,
  52:281515/2^54, 53:694073/2^56, 54:212249/2^55, 55:258273/2^56,
  56:636093/2^58, 57:781081/2^59, 58:952959/2^60, 59:291475/2^59,
  60:718599/2^61, 61:878997/2^62, 62:534821/2^62, 63:329285/2^62,
  64:404341/2^63, 65:986633/2^65, 66:603553/2^65,
}
def alpha(w):
  if w >= 67: return (633/1024)^w
  return earlybounds[w]
assert all(alpha(w)^49<2^(-(34*w-23)) for w in range(31,100))
assert min((633/1024)^w/alpha(w) for w in range(68))==633^5/(2^30*165219)

from alpha import alpha
from memoized import memoized
R = MatrixSpace(ZZ,2)
def scaledM(e,q):
  return R((0,2^e,-2^e,q))
@memoized
def beta(w):
  return min(alpha(w+j)/alpha(j) for j in range(68))
@memoized
def gamma(w,e):
  return min(beta(w+j)*2^j*70/169 for j in range(e,e+68))
def spectralradiusisatmost(PP,N): # assuming PP has form P.transpose()*P
  (a,b),(c,d) = PP
  X = 2*N^2-a-d
  return N >= 0 and X >= 0 and X^2 >= (a-d)^2+4*b^2
def verify(w,P):
  nodes = 1
  PP = P.transpose()*P
  if w>0 and spectralradiusisatmost(PP,4^w*beta(w)): return nodes
  assert spectralradiusisatmost(PP,4^w*alpha(w))
  for e in PositiveIntegers():
    if spectralradiusisatmost(PP,4^w*gamma(w,e)): return nodes
    for q in range(1,2^(e+1),2):
      nodes += verify(e+w,scaledM(e,q)*P)

print verify(0,R(1))

# PAPER END

# --- PYTHON EXAMPLE --- #

from fractions import Fraction

def truncate(f, t):
    if t == 0:
        return 0
    mask = (1 << t) - 1
    f = f & mask
    if f >= (1 << (t-1)):
        f -= (1 << t)
    return f

def sign(x):
    return 1 if x >= 0 else -1

def div2n(x, p, p_inv, m):
    two_m = 1 << m
    correction = (x * p_inv) & (two_m - 1)
    x = x - correction * p
    return x >> m

def divsteps2(n, t, delta, f, g):
    f, g = truncate(f, t), truncate(g, t)
    u, v, q, r = Fraction(1), Fraction(0), Fraction(0), Fraction(1)
    while n > 0:
        f = truncate(f, t)
        if delta > 0 and (g & 1):
            delta, f, g, u, v, q, r = -delta, g, -f, q, r, -u, -v
        g0 = g & 1
        delta = 1 + delta
        g = (g + g0 * f) // 2
        q = (q + Fraction(g0) * u) / 2
        r = (r + Fraction(g0) * v) / 2
        n, t = n - 1, t - 1
        g = truncate(g, t)
    return delta, f, g, ((u, v), (q, r))

def iterations(d):
    return (49*d + 80)//17 if d < 46 else (49*d + 57)//17

def recip2(f, g):
    assert f & 1
    d = max(f.bit_length(), g.bit_length())
    m = iterations(d)
    precomp = pow((f + 1)//2, m - 1, f)
    delta, fm, gm, P = divsteps2(m, m + 1, 1, f, g)
    (u_frac, v_frac), (q_frac, r_frac) = P
    V_int = int(v_frac * (1 << (m - 1))) * sign(fm)
    inv = (V_int * precomp) % f
    return inv

def main():
    f = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    g = 0x33e7665705359f04f28b88cf897c603c9
    print("Calculando inverso de g modulo f...")
    inv = recip2(f, g)
    print("Inverso modular:", hex(inv))
    check = (g * inv) % f
    print("Verificação: g * inv % f =", hex(check))
    assert check == 1, "Inverso modular incorreto!"

if __name__ == "__main__":
    main()

# --- CUDA --- #

#include <cuda_runtime.h>
#include <stdio.h>

// Função device: pode ser chamada dentro de kernels CUDA
__device__ int truncate(int f, int t) {
    if (t == 0) {
        return 0;
    }
    int mask = (1 << t) - 1;
    f = f & mask;
    if (f >= (1 << (t - 1))) {
        f -= (1 << t);
    }
    return f;
}