# hash-ids

A fast, dependency-free implementation for [hashids](https://hashids.org/).

## Usage

```rust
fn main() {
    let hash_ids = hash_ids::HashIds::builder()
        .with_salt("Arbitrary string")
        .finish(); 
    assert_eq!("neHrCa", hash_ids.encode(&[1, 2, 3]));
    assert_eq!(vec![1, 2, 3], hash_ids.decode("neHrCa"));
}
```