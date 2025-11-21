#![allow(non_upper_case_globals)] // 允许不使用大写蛇形命名的全局变量/常量
#![allow(non_camel_case_types)]  // 允许不使用驼峰命名的类型
#![allow(non_snake_case)]      // 允许不使用蛇形命名的变量/函数

mod inner {
    // 在这个内部模块里，我们包含了 bindgen 生成的文件
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use inner::*;