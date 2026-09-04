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
                if n > 9 {
                    n - 9
                } else {
                    n
                }
            } else {
                d
            }
        })
        .sum();
    sum % 10 == 0
}
#[cfg(test)]
#[path = "../../tests/filters_luhn.rs"]
mod tests;
