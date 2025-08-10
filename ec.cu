/*
SECP256K1
Compilação: nvcc -o ec_cuda_fixed ec_cuda_fixed.cu -arch=sm_50 -O3 -lineinfo
Requisitos: CUDA 10.0+, GPU Compute Capability 5.0+
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// P = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
__constant__ unsigned int P_CONST[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// N = ordem do grupo
__constant__ unsigned int N_CONST[8] = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// Ponto gerador G (valores originais - serão convertidos para Montgomery quando necessário)
__constant__ unsigned int GX_CONST[8] = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};

__constant__ unsigned int GY_CONST[8] = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

// R = 2^256 mod P
__constant__ unsigned int R_MOD_P[8] = {
    0x000003D1, 0x00000001, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// R^2 mod P
__constant__ unsigned int R2_MOD_P[8] = {
    0x000E90A1, 0x000007A2, 0x00000001, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// R^2 mod N (little-endian words)
__constant__ unsigned int R2_MOD_N[8] = {
    0x67D7D140, 0x896CF214, 0x0E7CF878, 0x741496C2,
    0x5BCD07C6, 0xE697F5E4, 0x81C69BC5, 0x9D671CD5
};

// mu = -P^{-1} mod 2^32
__constant__ unsigned int MU_P = 0xD2253531;

// mu para N
__constant__ unsigned int MU_N = 0x5588B13F;

// Constantes pré-computadas para secp256k1
__constant__ unsigned int ZERO[8] = {0, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int ONE[8] = {1, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int TWO[8] = {2, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int THREE[8] = {3, 0, 0, 0, 0, 0, 0, 0};
__constant__ unsigned int SEVEN[8] = {7, 0, 0, 0, 0, 0, 0, 0};

// Constantes em Montgomery form
// ONE_MONT = 1 * R mod P
__constant__ unsigned int ONE_MONT[8] = {
    0x000003D1, 0x00000001, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// 7 * R mod P
__constant__ unsigned int SEVEN_MONT[8] = {
    0x00001AB7, 0x00000007, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// Ponto em coordenadas afins (x, y)
typedef struct {
    unsigned int x[8];
    unsigned int y[8];
    int infinity;
} ECPoint;

// Ponto em coordenadas Jacobianas (X, Y, Z)
typedef struct {
    unsigned int X[8];
    unsigned int Y[8];
    unsigned int Z[8];
    int infinity;
} ECPointJacobian;

__device__ int bignum_cmp(const unsigned int *a, const unsigned int *b) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ int bignum_is_zero(const unsigned int *a) {
    for (int i = 0; i < 8; i++) {
        if (a[i] != 0) return 0;
    }
    return 1;
}

__device__ int bignum_is_odd(const unsigned int *a) {
    return a[0] & 1;
}

__device__ void bignum_copy(unsigned int *dst, const unsigned int *src) {
    for (int i = 0; i < 8; i++) {
        dst[i] = src[i];
    }
}

__device__ void bignum_zero(unsigned int *a) {
    for (int i = 0; i < 8; i++) {
        a[i] = 0;
    }
}

__device__ void bignum_set_ui(unsigned int *a, unsigned int val) {
    bignum_zero(a);
    a[0] = val;
}

// Soma com carry
__device__ unsigned int bignum_add_carry(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned long long carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (unsigned long long)a[i] + b[i];
        result[i] = (unsigned int)carry;
        carry >>= 32;
    }
    return (unsigned int)carry;
}

__device__ unsigned int bignum_sub_borrow(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    long long carry = 0; // signed para detectar negative
    for (int i = 0; i < 8; i++) {
        long long tmp = (long long)a[i] - (long long)b[i] - carry;
        if (tmp < 0) {
            result[i] = (unsigned int)(tmp + (1ULL << 32));
            carry = 1;
        } else {
            result[i] = (unsigned int)tmp;
            carry = 0;
        }
    }
    return (unsigned int)carry;
}

// Shift right por 1 bit
__device__ void bignum_shr1(unsigned int *result, const unsigned int *a) {
    unsigned int carry = 0;
    for (int i = 7; i >= 0; i--) {
        unsigned int new_carry = a[i] & 1;
        result[i] = (a[i] >> 1) | (carry << 31);
        carry = new_carry;
    }
}

// Multiplicação 256x256 -> 512 bits
__device__ void bignum_mul_full(unsigned int *result_high, unsigned int *result_low, 
                                const unsigned int *a, const unsigned int *b) {
    unsigned int temp[16];
    
    // Zerar resultado temporário
    for (int i = 0; i < 16; i++) {
        temp[i] = 0;
    }
    
    // Multiplicação escolar
    for (int i = 0; i < 8; i++) {
        unsigned long long carry = 0;
        for (int j = 0; j < 8; j++) {
            unsigned long long prod = (unsigned long long)a[i] * b[j] + temp[i + j] + carry;
            temp[i + j] = (unsigned int)prod;
            carry = prod >> 32;
        }
        temp[i + 8] = (unsigned int)carry;
    }
    
    // Copiar partes baixa e alta
    for (int i = 0; i < 8; i++) {
        result_low[i] = temp[i];
        result_high[i] = temp[i + 8];
    }
}

// Montgomery reduction especializada para P = 2^256 - 2^32 - 977
__device__ void montgomery_reduce_p(unsigned int *result, const unsigned int *input_high, const unsigned int *input_low) {
    unsigned int temp[16];
    
    // Copiar input (512 bits) para temp
    for (int i = 0; i < 8; i++) {
        temp[i] = input_low[i];
        temp[i + 8] = input_high[i];
    }
    
    // 8 iterações de Montgomery reduction
    for (int i = 0; i < 8; i++) {
        // ui = temp[i] * μ mod 2^32
        unsigned int ui = (temp[i] * MU_P) & 0xFFFFFFFF;
        
        // temp += ui * P * 2^(32*i)
        unsigned long long carry = 0;
        
        // Multiplicar ui * P e somar
        for (int j = 0; j < 8; j++) {
            unsigned long long prod = (unsigned long long)ui * P_CONST[j] + temp[i + j] + carry;
            temp[i + j] = (unsigned int)prod;
            carry = prod >> 32;
        }
        
        // Propagar carry para palavras superiores
        for (int j = 8; j < 16 - i; j++) {
            carry += temp[i + j];
            temp[i + j] = (unsigned int)carry;
            carry >>= 32;
        }
    }
    
    // Resultado está nas 8 palavras superiores
    for (int i = 0; i < 8; i++) {
        result[i] = temp[i + 8];
    }
    
    // Redução final se resultado >= P
    if (bignum_cmp(result, P_CONST) >= 0) {
        bignum_sub_borrow(result, result, P_CONST);
    }
}

// Montgomery reduction para N
__device__ void montgomery_reduce_n(unsigned int *result, const unsigned int *input_high, const unsigned int *input_low) {
    unsigned int temp[16];
    
    for (int i = 0; i < 8; i++) {
        temp[i] = input_low[i];
        temp[i + 8] = input_high[i];
    }
    
    for (int i = 0; i < 8; i++) {
        unsigned int ui = (temp[i] * MU_N) & 0xFFFFFFFF;
        
        unsigned long long carry = 0;
        for (int j = 0; j < 8; j++) {
            unsigned long long prod = (unsigned long long)ui * N_CONST[j] + temp[i + j] + carry;
            temp[i + j] = (unsigned int)prod;
            carry = prod >> 32;
        }
        
        for (int j = 8; j < 16 - i; j++) {
            carry += temp[i + j];
            temp[i + j] = (unsigned int)carry;
            carry >>= 32;
        }
    }
    
    for (int i = 0; i < 8; i++) {
        result[i] = temp[i + 8];
    }
    
    if (bignum_cmp(result, N_CONST) >= 0) {
        bignum_sub_borrow(result, result, N_CONST);
    }
}

// Conversão para Montgomery form: a * R mod P
__device__ void to_montgomery_p(unsigned int *result, const unsigned int *a) {
    unsigned int high[8], low[8];
    bignum_mul_full(high, low, a, R2_MOD_P);
    montgomery_reduce_p(result, high, low);
}

// Conversão de Montgomery form: a / R mod P
__device__ void from_montgomery_p(unsigned int *result, const unsigned int *a) {
    unsigned int zero[8];
    bignum_zero(zero);
    montgomery_reduce_p(result, zero, a);
}

// Conversão para Montgomery form mod N: a * R mod N
__device__ void to_montgomery_n(unsigned int *result, const unsigned int *a) {
    unsigned int high[8], low[8];
    bignum_mul_full(high, low, a, R2_MOD_N);
    montgomery_reduce_n(result, high, low);
}

// Conversão de Montgomery form mod N: a / R mod N
__device__ void from_montgomery_n(unsigned int *result, const unsigned int *a) {
    unsigned int zero[8];
    bignum_zero(zero);
    montgomery_reduce_n(result, zero, a);
}

__device__ void mod_add_p(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned int temp[8];
    unsigned int carry = bignum_add_carry(temp, a, b);
    
    if (carry || bignum_cmp(temp, P_CONST) >= 0) {
        bignum_sub_borrow(result, temp, P_CONST);
    } else {
        bignum_copy(result, temp);
    }
}

__device__ void mod_sub_p(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned int temp[8];
    unsigned int borrow = bignum_sub_borrow(temp, a, b);
    
    if (borrow) {
        bignum_add_carry(result, temp, P_CONST);
    } else {
        bignum_copy(result, temp);
    }
}

// Multiplicação modular Montgomery
__device__ void mod_mul_mont_p(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned int high[8], low[8];
    bignum_mul_full(high, low, a, b);
    montgomery_reduce_p(result, high, low);
}

// Quadrado modular Montgomery
__device__ void mod_sqr_mont_p(unsigned int *result, const unsigned int *a) {
    mod_mul_mont_p(result, a, a);
}

// Multiplicação modular Montgomery mod N
__device__ void mod_mul_mont_n(unsigned int *result, const unsigned int *a, const unsigned int *b) {
    unsigned int high[8], low[8];
    bignum_mul_full(high, low, a, b);
    montgomery_reduce_n(result, high, low);
}

__device__ void mod_inverse_p_fermat(unsigned int *result, const unsigned int *a) {
    // P - 2 para secp256k1
    unsigned int p_minus_2[8] = {
        0xFFFFFC2D, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    };
    
    unsigned int base[8], exp[8], temp_result[8];
    bignum_copy(base, a);
    bignum_copy(exp, p_minus_2);
    bignum_copy(temp_result, ONE_MONT); // 1 em forma Montgomery
    
    // Exponenciação binária: a^exp mod p em Montgomery
    while (!bignum_is_zero(exp)) {
        if (bignum_is_odd(exp)) {
            mod_mul_mont_p(temp_result, temp_result, base);
        }
        
        mod_mul_mont_p(base, base, base);
        bignum_shr1(exp, exp);
    }
    
    bignum_copy(result, temp_result);
}

// Binary Extended GCD
__device__ void mod_inverse_p_binary(unsigned int *result, const unsigned int *a) {
    unsigned int u[8], v[8], A[8], B[8], C[8], D[8];
    unsigned int temp[8];
    
    // Inicialização
    bignum_copy(u, a);
    bignum_copy(v, P_CONST);
    bignum_set_ui(A, 1); bignum_zero(B);
    bignum_zero(C); bignum_set_ui(D, 1);
    
    // Algoritmo Binary Extended GCD
    while (!bignum_is_zero(u)) {
        // Enquanto u for par
        while (!bignum_is_odd(u)) {
            bignum_shr1(u, u);
            
            if (bignum_is_odd(A) || bignum_is_odd(B)) {
                // Normalização: se A ou B for ímpar, adicionar P_CONST para manter paridade
                bignum_add_carry(A, A, P_CONST);
                if (bignum_cmp(B, a) >= 0) {
                    bignum_sub_borrow(B, B, a);
                } else {
                    // B - a seria negativo, então fazer P_CONST + B - a
                    bignum_add_carry(temp, B, P_CONST);
                    bignum_sub_borrow(B, temp, a);
                }
            }
            
            bignum_shr1(A, A);
            bignum_shr1(B, B);
        }
        
        // Enquanto v for par
        while (!bignum_is_odd(v)) {
            bignum_shr1(v, v);
            
            if (bignum_is_odd(C) || bignum_is_odd(D)) {
                // Normalização: se C ou D for ímpar, adicionar P_CONST para manter paridade
                bignum_add_carry(C, C, P_CONST);
                if (bignum_cmp(D, a) >= 0) {
                    bignum_sub_borrow(D, D, a);
                } else {
                    // D - a seria negativo, então fazer P_CONST + D - a
                    bignum_add_carry(temp, D, P_CONST);
                    bignum_sub_borrow(D, temp, a);
                }
            }
            
            bignum_shr1(C, C);
            bignum_shr1(D, D);
        }
        
        // Subtrair o menor do maior
        if (bignum_cmp(u, v) >= 0) {
            bignum_sub_borrow(u, u, v);
            
            // Atualizar A e B com normalização modular
            if (bignum_cmp(A, C) >= 0) {
                bignum_sub_borrow(A, A, C);
            } else {
                // A - C seria negativo, então fazer P_CONST + A - C
                bignum_add_carry(temp, A, P_CONST);
                bignum_sub_borrow(A, temp, C);
            }
            
            if (bignum_cmp(B, D) >= 0) {
                bignum_sub_borrow(B, B, D);
            } else {
                // B - D seria negativo, então fazer P_CONST + B - D
                bignum_add_carry(temp, B, P_CONST);
                bignum_sub_borrow(B, temp, D);
            }
        } else {
            bignum_sub_borrow(v, v, u);
            
            // Atualizar C e D com normalização modular
            if (bignum_cmp(C, A) >= 0) {
                bignum_sub_borrow(C, C, A);
            } else {
                // C - A seria negativo, então fazer P_CONST + C - A
                bignum_add_carry(temp, C, P_CONST);
                bignum_sub_borrow(C, temp, A);
            }
            
            if (bignum_cmp(D, B) >= 0) {
                bignum_sub_borrow(D, D, B);
            } else {
                // D - B seria negativo, então fazer P_CONST + D - B
                bignum_add_carry(temp, D, P_CONST);
                bignum_sub_borrow(D, temp, B);
            }
        }
    }
    
    // v contém o GCD, C contém o inverso de 'a'
    if (bignum_cmp(v, ONE) == 0) {
        // Normalizar C para estar no intervalo [0, P_CONST)
        bignum_copy(result, C);
        while (bignum_cmp(result, P_CONST) >= 0) {
            bignum_sub_borrow(result, result, P_CONST);
        }
    } else {
        bignum_zero(result);  // Não existe inverso
    }
}

// Inversão Modular
__device__ void mod_inverse_p(unsigned int *result, const unsigned int *a) {
    // Verificar casos especiais
    if (bignum_is_zero(a)) {
        bignum_zero(result);
        return;
    }
    
    if (bignum_cmp(a, ONE_MONT) == 0) {
        bignum_copy(result, ONE_MONT);
        return;
    }
    
    // Usar Fermat (mais rápido para primos)
    mod_inverse_p_fermat(result, a);
    
    // Verificação: a * a^(-1) ≡ 1 (mod p) em Montgomery
    unsigned int verification[8];
    mod_mul_mont_p(verification, a, result);
    
    // Deve retornar 1 em Montgomery form
    if (bignum_cmp(verification, ONE_MONT) != 0) {
        // Fallback para algoritmo binário se Fermat falhar
        // Converter para forma normal, fazer inversão, converter de volta
        unsigned int a_normal[8], result_normal[8];
        from_montgomery_p(a_normal, a);
        mod_inverse_p_binary(result_normal, a_normal);
        to_montgomery_p(result, result_normal);
    }
}

__device__ void jacobian_init(ECPointJacobian *point) {
    bignum_zero(point->X);
    bignum_zero(point->Y);
    bignum_copy(point->Z, ONE_MONT); // Z = 1 em Montgomery
    point->infinity = 0;
}

__device__ void jacobian_set_infinity(ECPointJacobian *point) {
    bignum_copy(point->X, ONE_MONT);
    bignum_copy(point->Y, ONE_MONT);
    bignum_zero(point->Z);
    point->infinity = 1;
}

__device__ int jacobian_is_infinity(const ECPointJacobian *point) {
    return point->infinity || bignum_is_zero(point->Z);
}

// Conversão de afim para Jacobiano
__device__ void affine_to_jacobian(ECPointJacobian *jac, const ECPoint *aff) {
    if (aff->infinity) {
        jacobian_set_infinity(jac);
        return;
    }
    
    bignum_copy(jac->X, aff->x);
    bignum_copy(jac->Y, aff->y);
    bignum_copy(jac->Z, ONE_MONT); // Z = 1 em Montgomery
    jac->infinity = 0;
}

__device__ void jacobian_to_affine(ECPoint *aff, const ECPointJacobian *jac) {
    if (jacobian_is_infinity(jac)) {
        bignum_zero(aff->x);
        bignum_zero(aff->y);
        aff->infinity = 1;
        return;
    }
    
    unsigned int z_inv[8], z_inv_sqr[8], z_inv_cube[8];
    
    // z_inv = Z^(-1) em Montgomery
    mod_inverse_p(z_inv, jac->Z);
    
    // z_inv_sqr = Z^(-2) em Montgomery
    mod_mul_mont_p(z_inv_sqr, z_inv, z_inv);
    
    // z_inv_cube = Z^(-3) em Montgomery
    mod_mul_mont_p(z_inv_cube, z_inv_sqr, z_inv);
    
    // x = X * Z^(-2) em Montgomery
    mod_mul_mont_p(aff->x, jac->X, z_inv_sqr);
    
    // y = Y * Z^(-3) em Montgomery
    mod_mul_mont_p(aff->y, jac->Y, z_inv_cube);
    
    aff->infinity = 0;
}

__device__ void jacobian_double(ECPointJacobian *result, const ECPointJacobian *point) {
    if (jacobian_is_infinity(point) || bignum_is_zero(point->Y)) {
        jacobian_set_infinity(result);
        return;
    }
    
    unsigned int A[8], B[8], C[8], D[8], E[8], F[8];
    unsigned int X2[8]; // Temporary para X^2
    
    // A = Y₁²
    mod_sqr_mont_p(A, point->Y);
    
    // B = 4 * X₁ * Y₁²
    mod_mul_mont_p(B, point->X, A);
    mod_add_p(B, B, B);  // 2 * X₁ * Y₁²
    mod_add_p(B, B, B);  // 4 * X₁ * Y₁²
    
    // C = 8 * Y₁⁴
    mod_sqr_mont_p(C, A);
    mod_add_p(C, C, C);  // 2 * Y₁⁴
    mod_add_p(C, C, C);  // 4 * Y₁⁴
    mod_add_p(C, C, C);  // 8 * Y₁⁴
    
    mod_sqr_mont_p(X2, point->X);    // X2 = X₁²
    mod_add_p(D, X2, X2);            // D = 2 * X₁²
    mod_add_p(D, D, X2);             // D = 3 * X₁²
    
    // E = D²
    mod_sqr_mont_p(E, D);
    
    // F = E - 2*B
    mod_sub_p(F, E, B);
    mod_sub_p(F, F, B);
    
    // X₃ = F
    bignum_copy(result->X, F);
    
    // Y₃ = D * (B - F) - C
    mod_sub_p(result->Y, B, F);
    mod_mul_mont_p(result->Y, D, result->Y);
    mod_sub_p(result->Y, result->Y, C);
    
    // Z₃ = 2 * Y₁ * Z₁
    mod_mul_mont_p(result->Z, point->Y, point->Z);
    mod_add_p(result->Z, result->Z, result->Z);
    
    result->infinity = 0;
}

// Adição de pontos em coordenadas Jacobianas
__device__ void jacobian_add(ECPointJacobian *result, const ECPointJacobian *P, const ECPointJacobian *Q) {
    if (jacobian_is_infinity(P)) {
        bignum_copy(result->X, Q->X);
        bignum_copy(result->Y, Q->Y);
        bignum_copy(result->Z, Q->Z);
        result->infinity = Q->infinity;
        return;
    }
    
    if (jacobian_is_infinity(Q)) {
        bignum_copy(result->X, P->X);
        bignum_copy(result->Y, P->Y);
        bignum_copy(result->Z, P->Z);
        result->infinity = P->infinity;
        return;
    }
    
    unsigned int U1[8], U2[8], S1[8], S2[8], H[8], I[8], J[8], r[8], V[8];
    unsigned int Z1Z1[8], Z2Z2[8];
    
    // Z1Z1 = Z₁²
    mod_sqr_mont_p(Z1Z1, P->Z);
    
    // Z2Z2 = Z₂²
    mod_sqr_mont_p(Z2Z2, Q->Z);
    
    // U1 = X₁ * Z2Z2
    mod_mul_mont_p(U1, P->X, Z2Z2);
    
    // U2 = X₂ * Z1Z1
    mod_mul_mont_p(U2, Q->X, Z1Z1);
    
    // S1 = Y₁ * Z₂³ = Y₁ * Z₂ * Z2Z2
    mod_mul_mont_p(S1, Q->Z, Z2Z2);
    mod_mul_mont_p(S1, P->Y, S1);
    
    // S2 = Y₂ * Z₁³ = Y₂ * Z₁ * Z1Z1
    mod_mul_mont_p(S2, P->Z, Z1Z1);
    mod_mul_mont_p(S2, Q->Y, S2);
    
    // Verificar se P = Q (point doubling)
    if (bignum_cmp(U1, U2) == 0) {
        if (bignum_cmp(S1, S2) == 0) {
            // P = Q, fazer doubling
            jacobian_double(result, P);
            return;
        } else {
            // P = -Q, resultado é infinito
            jacobian_set_infinity(result);
            return;
        }
    }
    
    // H = U2 - U1
    mod_sub_p(H, U2, U1);
    
    // I = 4 * H²
    mod_sqr_mont_p(I, H);
    mod_add_p(I, I, I);
    mod_add_p(I, I, I);
    
    // J = H * I
    mod_mul_mont_p(J, H, I);
    
    // r = 2 * (S2 - S1)
    mod_sub_p(r, S2, S1);
    mod_add_p(r, r, r);
    
    // V = U1 * I
    mod_mul_mont_p(V, U1, I);
    
    // X₃ = r² - J - 2*V
    mod_sqr_mont_p(result->X, r);
    mod_sub_p(result->X, result->X, J);
    mod_sub_p(result->X, result->X, V);
    mod_sub_p(result->X, result->X, V);
    
    // Y₃ = r * (V - X₃) - 2 * S1 * J
    mod_sub_p(result->Y, V, result->X);
    mod_mul_mont_p(result->Y, r, result->Y);
    mod_mul_mont_p(S1, S1, J);
    mod_add_p(S1, S1, S1);
    mod_sub_p(result->Y, result->Y, S1);
    
    // Z₃ = 2 * Z₁ * Z₂ * H
    mod_mul_mont_p(result->Z, P->Z, Q->Z);
    mod_mul_mont_p(result->Z, result->Z, H);
    mod_add_p(result->Z, result->Z, result->Z);
    
    result->infinity = 0;
}

// Função para reduzir scalar módulo N de forma eficiente
__device__ void scalar_reduce_n(unsigned int *result, const unsigned int *scalar) {
    bignum_copy(result, scalar);
    
    // Reduzir k módulo N usando subtração repetida otimizada
    while (bignum_cmp(result, N_CONST) >= 0) {
        bignum_sub_borrow(result, result, N_CONST);
    }
}

// Multiplicação escalar em coordenadas Jacobianas otimizada
__device__ void jacobian_scalar_mult(ECPointJacobian *result, const unsigned int *scalar, const ECPointJacobian *point) {
    ECPointJacobian Q, temp;
    jacobian_init(&Q);
    jacobian_set_infinity(result);
    
    // Copiar ponto base
    bignum_copy(Q.X, point->X);
    bignum_copy(Q.Y, point->Y);
    bignum_copy(Q.Z, point->Z);
    Q.infinity = point->infinity;
    
    unsigned int k[8];
    scalar_reduce_n(k, scalar);
    
    // Algoritmo binary (double-and-add)
    while (!bignum_is_zero(k)) {
        if (bignum_is_odd(k)) {
            if (jacobian_is_infinity(result)) {
                bignum_copy(result->X, Q.X);
                bignum_copy(result->Y, Q.Y);
                bignum_copy(result->Z, Q.Z);
                result->infinity = Q.infinity;
            } else {
                jacobian_add(&temp, result, &Q);
                bignum_copy(result->X, temp.X);
                bignum_copy(result->Y, temp.Y);
                bignum_copy(result->Z, temp.Z);
                result->infinity = temp.infinity;
            }
        }
        
        // Q = 2*Q
        jacobian_double(&temp, &Q);
        bignum_copy(Q.X, temp.X);
        bignum_copy(Q.Y, temp.Y);
        bignum_copy(Q.Z, temp.Z);
        Q.infinity = temp.infinity;
        
        bignum_shr1(k, k);
    }
}

__device__ void point_init(ECPoint *point) {
    bignum_zero(point->x);
    bignum_zero(point->y);
    point->infinity = 0;
}

__device__ void point_add(ECPoint *R, const ECPoint *P, const ECPoint *Q) {
    // Converter para Jacobiano, fazer operação, converter de volta
    ECPointJacobian P_jac, Q_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    affine_to_jacobian(&Q_jac, Q);
    
    jacobian_add(&R_jac, &P_jac, &Q_jac);
    
    jacobian_to_affine(R, &R_jac);
}

__device__ void point_double(ECPoint *R, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    jacobian_double(&R_jac, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

__device__ void scalar_mult(ECPoint *R, const unsigned int *k, const ECPoint *P) {
    ECPointJacobian P_jac, R_jac;
    
    affine_to_jacobian(&P_jac, P);
    jacobian_scalar_mult(&R_jac, k, &P_jac);
    jacobian_to_affine(R, &R_jac);
}

__device__ int point_is_valid(const ECPoint *point) {
    if (point->infinity) return 1;

    unsigned int lhs[8], rhs[8], temp[8];
    
    // y² = x³ + 7 (secp256k1) - ambos em Montgomery
    mod_sqr_mont_p(lhs, point->y);           // y² em Montgomery
    mod_sqr_mont_p(rhs, point->x);           // x² em Montgomery
    mod_mul_mont_p(rhs, rhs, point->x);      // x³ em Montgomery
    mod_add_p(rhs, rhs, SEVEN_MONT);         // x³ + 7 em Montgomery

    return (bignum_cmp(lhs, rhs) == 0);
}

// Função para converter constantes do gerador para Montgomery
__device__ void get_generator_montgomery(ECPoint *G) {
    to_montgomery_p(G->x, GX_CONST);
    to_montgomery_p(G->y, GY_CONST);
    G->infinity = 0;
}

// Função para gerar chave pública (k * G) onde k é chave privada
__device__ void generate_public_key(ECPoint *public_key, const unsigned int *private_key) {
    ECPoint G;
    get_generator_montgomery(&G);
    scalar_mult(public_key, private_key, &G);
}

// Função para converter resultado final de Montgomery para forma normal
__device__ void point_from_montgomery(ECPoint *result, const ECPoint *point_mont) {
    if (point_mont->infinity) {
        result->infinity = 1;
        bignum_zero(result->x);
        bignum_zero(result->y);
        return;
    }
    
    from_montgomery_p(result->x, point_mont->x);
    from_montgomery_p(result->y, point_mont->y);
    result->infinity = 0;
}

__device__ void get_compressed_public_key(unsigned char *out, const ECPoint *public_key_mont) {
    // Converter de Montgomery para forma normal antes de serializar
    ECPoint public_key_normal;
    point_from_montgomery(&public_key_normal, public_key_mont);
    
    unsigned char prefix = (public_key_normal.y[0] & 1) ? 0x03 : 0x02;
    out[0] = prefix;
    
    // Converter x para big-endian
    for (int i = 0; i < 8; i++) {
        unsigned int word = public_key_normal.x[7-i];
        out[1 + i*4] = (word >> 24) & 0xFF;
        out[1 + i*4 + 1] = (word >> 16) & 0xFF;
        out[1 + i*4 + 2] = (word >> 8) & 0xFF;
        out[1 + i*4 + 3] = word & 0xFF;
    }
}