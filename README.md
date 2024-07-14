# whisper-rs


RUSTFLAGS="-C target-feature=+avx,+avx2,+fma,+f16c -C relocation-model=pic -C link-arg=-pthread -C opt-level=3 -Ctarget-cpu=native" cargo run