// On macOS, ONNX Runtime's prebuilt static library calls `__isPlatformVersionAtLeast`
// (from `@available` checks), which lives in clang's compiler-rt, not in rustc's builtins.
// Link libclang_rt.osx.a explicitly so `cargo build` works out of the box.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    // `webgpu`: ORT's macOS WebGPU build links libwebgpu_dawn.dylib via @rpath. ort's
    // `copy-dylibs` symlinks it next to the binary; point the rpath there.
    if std::env::var_os("CARGO_FEATURE_WEBGPU").is_some() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
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
