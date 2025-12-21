// Power of 10 significand generator for Żmij.
// Copyright (c) 2025 - present, Victor Zverovich

use num_bigint::BigUint as Uint;
use std::f64::consts::LOG2_10;

fn main() {
    // Range of decimal exponents [K_min, K_max] from the paper.
    let dec_exp_min = -324_i32;
    let dec_exp_max = 292_i32;

    let num_bits = 128_i32;

    // Negate dec_pow_min and dec_pow_max because we need negative powers 10^-k.
    for dec_exp in -dec_exp_max..=-dec_exp_min {
        // dec_exp is -k in the paper.
        let bin_exp = (f64::from(dec_exp) * LOG2_10).floor() as i32 - (num_bits - 1);
        let bin_pow = Uint::from(2_u8).pow(bin_exp.unsigned_abs());
        let dec_pow = Uint::from(10_u8).pow(dec_exp.unsigned_abs());
        let result = if dec_exp < 0 {
            bin_pow / dec_pow
        } else if bin_exp < 0 {
            dec_pow * bin_pow
        } else {
            dec_pow / bin_pow
        };
        let hi = &result >> 64;
        let lo = result & (Uint::from(2_u8).pow(64) - Uint::from(1_u8));
        println!("{{{hi:#x}, {lo:#018x}}}, // {dec_exp:4}");
    }
}
