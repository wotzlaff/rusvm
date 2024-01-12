use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(value: &str);
}

#[macro_export]
/// Redefines `println` macro for wasm.
macro_rules! println {
    ($($t:tt)*) => (crate::console::log(&format_args!($($t)*).to_string()))
}
