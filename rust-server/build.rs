// On macOS, ONNX Runtime's prebuilt static library calls `__isPlatformVersionAtLeast`
// (from `@available` checks), which lives in clang's compiler-rt, not in rustc's builtins.
// Link libclang_rt.osx.a explicitly so `cargo build` works out of the box.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let Ok(out) = std::process::Command::new("clang").arg("--print-resource-dir").output() else {
        return;
    };
    let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let lib = std::path::Path::new(&dir).join("lib/darwin/libclang_rt.osx.a");
    if lib.exists() {
        println!("cargo:rustc-link-arg={}", lib.display());
    } else {
        println!("cargo:warning=libclang_rt.osx.a not found under {dir}; link may fail with ___isPlatformVersionAtLeast");
    }
}
