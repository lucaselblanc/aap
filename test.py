# --- ! --- #

# N almost inverse
N = 17

def div2(p, a):
    """Helper routine to compute a/2 mod p (where p is odd)."""
    assert p & 1
    if a & 1:  # If a is odd, make it even by adding p.
        a += p
    # a must be even now, so clean division by 2 is possible.
    return a // 2

def modinv(p, a):
    """Compute the inverse of a mod p (given that it exists, and p is odd)."""
    assert p & 1
    delta, u, v = 1, p, a
    x1, x2 = 0, 1
    while v != 0:
        # Note that while division by two for u and v is only ever done on even inputs,
        # this is not true for x1 and x2, so we need the div2 helper function.
        if delta > 0 and v & 1:
            delta, u, v = 1 - delta, v, (u - v) // 2
            x1, x2 = x2, div2(p, x1 - x2)
        elif v & 1:
            delta, u, v = 1 + delta, u, (v - u) // 2
            x1, x2 = x1, div2(p, x2 - x1)
        else:
            delta, u, v = 1 + delta, u, v // 2
            x1, x2 = x1, div2(p, x2)
        # Verify that the invariants x1=u/a mod p, x2=v/a mod p are maintained.
        assert u % p == (a * x1) % p
        assert v % p == (a * x2) % p
    assert u == 1 or u == -1  # |u| is the GCD, it must be 1
    # Because of invariant x1 = u/a mod p, 1/a = x1/u mod p. Since |u| = 1, x1/u = x1 * u.
    return (x1 * u) % p

# --- transition_matrix --- #

# Computes the matrix after N divsteps, this also computes u(N),v(N)u(N),v(N):

def transition_matrix(delta, u, v):
    """Compute delta and transition matrix t after N divsteps (multiplied by 2^N)."""
    m00, m01, m10, m11 = 1, 0, 0, 1  # start with identity matrix
    for _ in range(N):
        if delta > 0 and v & 1:
            delta, u, v, m00, m01, m10, m11 = 1 - delta, v, (u - v) // 2, 2*m10, 2*m11, m00 - m10, m01 - m11
        elif v & 1:
            delta, u, v, m00, m01, m10, m11 = 1 + delta, u, (v - u) // 2, 2*m00, 2*m01, m10 - m00, m11 - m01
        else:
            delta, u, v, m00, m01, m10, m11 = 1 + delta, u, v // 2, 2*m00, 2*m01, m10, m11
    return delta, u, v, (m00, m01, m10, m11)

# --- div2n & update_x1x2 --- #

# Applies the matrix to [x1,x2][x1​,x2​] to compute x1(N),x2(N)x1(N)​,x2(N):

def div2n(p, p_inv, x):
    """Compute x/2^N mod p, given p_inv = 1/p mod 2^N."""
    assert (p * p_inv) % 2**N == 1
    m = (x * p_inv) % 2**N
    x -= m * p
    assert x % 2**N == 0
    return (x >> N) % p

def update_x1x2(x1, x2, t, p, p_inv):
    """Multiply matrix t/2^N with [x1, x2], modulo p."""
    m00, m01, m10, m11 = t
    x1n, x2n = m00*x1 + m01*x2, m10*x1 + m11*x2
    return div2n(p, p_inv, x1n), div2n(p, p_inv, x2n)

# --- modinv --- #

# Combines all components to compute the modular inverse:

def modinv(p, p_inv, x):
    """Compute the modular inverse of x mod p, given p_inv=1/p mod 2^N."""
    assert p & 1
    delta, u, v, x1, x2 = 1, p, x, 0, 1
    while v != 0:
        delta, u, v, t = transition_matrix(delta, u, v)
        x1, x2 = update_x1x2(x1, x2, t, p, p_inv)
    assert u == 1 or u == -1  # |u| must be 1
    return (x1 * u) % p

# --- Version 1 --- #

# Allowing Range Expansion, instead of restricting outputs to modulo pp, we can eliminate the mod operation if we ensure that all intermediate values remain within a wider range, such as (−2p,p)(−2p,p):

def update_x1x2_optimized_ver1(x1, x2, t, p, p_inv):
    """Multiply matrix t/2^N with [x1, x2], modulo p, given p_inv = 1/p mod 2^N."""
    m00, m01, m10, m11 = t
    x1n, x2n = m00*x1 + m01*x2, m10*x1 + m11*x2
    # Cancel out bottom N bits of x1n and x2n.
    mx1n = ((x1n * p_inv) % 2**N)
    mx2n = ((x2n * p_inv) % 2**N)
    x1n -= mx1n * p
    x2n -= mx2n * p
    return x1n >> N, x2n >> N

# --- Version 2 --- #

# Pre-Clamping Strategy,to prevent range blow-up, we can add pp to negative values to bring the range back to (−p,p)(−p,p):

def update_x1x2_optimized_ver2(x1, x2, t, p, p_inv):
    """Multiply matrix t/2^N with [x1, x2], modulo p, given p_inv = 1/p mod 2^N."""
    m00, m01, m10, m11 = t
    # x1, x2 in (-2*p, p)
    if x1 < 0:
        x1 += p
    if x2 < 0:
        x2 += p
    # x1, x2 in (-p, p)
    x1n, x2n = m00*x1 + m01*x2, m10*x1 + m11*x2
    mx1n = -((p_inv * x1n) % 2**N)
    mx2n = -((p_inv * x2n) % 2**N)
    x1n += mx1n * p
    x2n += mx2n * p
    return x1n >> N, x2n >> N

# --- normalization --- #

# After the final iteration, the results x1(N),x2(N)x1(N)​,x2(N)​ lie in (−2p,p)(−2p,p). We want to map x1x1​ into the range [0,p)[0,p) using the following normalization:

def normalize(sign, v, p):
    """Compute sign*v mod p, where v is in range (-2*p, p); output in [0, p)."""
    assert sign == 1 or sign == -1
    if v < 0:
        v += p
    if sign == -1:
        v = -v
    if v < 0:
        v += p
    return v

# --- Final Implementation --- #

def modinv(p, p_inv, x):
    """Compute the modular inverse of x mod p, given p_inv=1/p mod 2^N."""
    assert p & 1
    delta, u, v, x1, x2 = 1, p, x, 0, 1
    while v != 0:
        delta, u, v, t = transition_matrix(delta, u, v)
        x1, x2 = update_x1x2_optimized_ver2(x1, x2, t, p, p_inv)
    assert u == 1 or u == -1  
    return normalize(u, x1, p)

def main():
    # --- Priv Key LSB-first (Entropy 130/256 bits) ---
    h_priv = [
        0x28b88cf897c603c9,
        0x3e7665705359f04f,
        0x0000000000000003,
        0x0000000000000000
    ]

    # LSB-first
    a = h_priv[0] + (h_priv[1] << 64) + (h_priv[2] << 128) + (h_priv[3] << 192)

    # --- prime p secp256k1 ---
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

    # p_inv = 1/p mod 2^N
    p_inv = pow(p, -1, 2**N)

    # Call modular inverse
    inv_a = modinv(p, p_inv, a)

    print("Chave privada (decimal):", a)
    print("Módulo p (decimal):", p)
    print("Inverso modular (decimal):", inv_a)
    print("Inverso modular (hex) :", hex(inv_a))

if __name__ == "__main__":
    main()
