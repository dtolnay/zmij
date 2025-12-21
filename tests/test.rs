#![allow(clippy::float_cmp, clippy::unreadable_literal)]

fn dtoa(value: f64) -> String {
    zmij::Buffer::new().format(value).to_owned()
}

#[test]
fn normal() {
    assert_eq!(dtoa(6.62607015e-34), "6.62607015e-34");
}

#[test]
fn small_int() {
    assert_eq!(dtoa(1.0), "1.0");
}

#[test]
fn zero() {
    assert_eq!(dtoa(0.0), "0.0");
    assert_eq!(dtoa(-0.0), "-0.0");
}

#[test]
fn inf() {
    assert_eq!(dtoa(f64::INFINITY), "inf");
    assert_eq!(dtoa(f64::NEG_INFINITY), "-inf");
}

#[test]
fn nan() {
    assert_eq!(dtoa(f64::NAN.copysign(1.0)), "NaN");
    assert_eq!(dtoa(f64::NAN.copysign(-1.0)), "NaN");
}

#[test]
fn shorter() {
    // A possibly shorter underestimate is picked (u' in Schubfach).
    assert_eq!(dtoa(-4.932096661796888e-226), "-4.932096661796888e-226");

    // A possibly shorter overestimate is picked (w' in Schubfach).
    assert_eq!(dtoa(3.439070283483335e+35), "3.439070283483335e+35");
}

#[test]
fn single_candidate() {
    // Only an underestimate is in the rounding region (u in Schubfach).
    assert_eq!(dtoa(6.606854224493745e-17), "6.606854224493745e-17");

    // Only an overestimate is in the rounding region (w in Schubfach).
    assert_eq!(dtoa(6.079537928711555e+61), "6.079537928711555e+61");
}

#[test]
fn all_exponents() {
    for exp in f64::MIN_EXP..f64::MAX_EXP {
        let expected = f64::exp2(f64::from(exp));
        let actual = dtoa(expected).parse::<f64>().unwrap();
        assert_eq!(actual, expected);
    }
}
