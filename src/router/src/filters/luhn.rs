pub fn luhn_valid(input: &str) -> bool {
    let digits: Vec<u32> = input
        .chars()
        .filter(char::is_ascii_digit)
        .filter_map(|c| c.to_digit(10))
        .collect();
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == 1 {
                let n = d * 2;
                if n > 9 { n - 9 } else { n }
            } else {
                d
            }
        })
        .sum();
    sum % 10 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_visa() {
        assert!(luhn_valid("4111111111111111"));
    }

    #[test]
    fn valid_mastercard() {
        assert!(luhn_valid("5500000000000004"));
    }

    #[test]
    fn valid_amex() {
        assert!(luhn_valid("340000000000009"));
    }

    #[test]
    fn invalid_card() {
        assert!(!luhn_valid("1234567812345678"));
    }

    #[test]
    fn too_short() {
        assert!(!luhn_valid("123"));
    }

    #[test]
    fn all_zeros() {
        assert!(luhn_valid("0000000000000000"));
    }

    #[test]
    fn with_separators() {
        assert!(luhn_valid("4111-1111-1111-1111"));
    }
}
