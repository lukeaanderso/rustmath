fn main() {
    println!("cargo:rustc-link-arg=-llapack");
    //pkg_config::Config::new().probe("lapack").unwrap();
    //println!("cargo:rerun-if-changed=build.rs");

}