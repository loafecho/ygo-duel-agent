// Links the engine we build ourselves from edo9300/ygopro-core (meson, shared library).
// OCGCORE_DIR points at the meson build dir; default is the layout on the game host.
fn main() {
    let dir = std::env::var("OCGCORE_DIR").unwrap_or_else(|_| "/srv/ygo/core/build".into());
    println!("cargo:rerun-if-env-changed=OCGCORE_DIR");
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=ocgcore");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
}
