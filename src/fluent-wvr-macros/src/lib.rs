#![forbid(unsafe_code)]

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Data, DataStruct, DeriveInput, Fields, Lit};

struct FieldMeta {
    desc: Option<String>,
    min: Option<f64>,
    max: Option<f64>,
    clamp: bool,
    skip: bool,
    required: bool,
    format: Option<String>,
    max_len: Option<usize>,
    sanitize: Option<String>,
    pattern: Option<String>,
    coerce: Option<String>,
    parse: Option<String>,
    empty_is_none: bool,
}

fn parse_field_attrs(field: &syn::Field) -> FieldMeta {
    let mut result = FieldMeta {
        desc: None,
        min: None,
        max: None,
        clamp: false,
        skip: false,
        required: true,
        format: None,
        max_len: None,
        sanitize: None,
        pattern: None,
        coerce: None,
        parse: None,
        empty_is_none: true,
    };

    for attr in &field.attrs {
        if !attr.path().is_ident("field") {
            continue;
        }

        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                result.skip = true;
                return Ok(());
            }
            if meta.path.is_ident("desc") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Str(s) = lit {
                    result.desc = Some(s.value());
                }
                return Ok(());
            }
            if meta.path.is_ident("min") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                match lit {
                    Lit::Int(i) => result.min = Some(i.base10_parse::<f64>().unwrap()),
                    Lit::Float(f) => result.min = Some(f.base10_parse::<f64>().unwrap()),
                    _ => {}
                }
                return Ok(());
            }
            if meta.path.is_ident("max") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                match lit {
                    Lit::Int(i) => result.max = Some(i.base10_parse::<f64>().unwrap()),
                    Lit::Float(f) => result.max = Some(f.base10_parse::<f64>().unwrap()),
                    _ => {}
                }
                return Ok(());
            }
            if meta.path.is_ident("required") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Bool(b) = lit {
                    result.required = b.value();
                }
                return Ok(());
            }
            if meta.path.is_ident("format") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Str(s) = lit {
                    result.format = Some(s.value());
                }
                return Ok(());
            }
            if meta.path.is_ident("max_len") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Int(i) = lit {
                    result.max_len = Some(i.base10_parse::<usize>().unwrap());
                }
                return Ok(());
            }
            if meta.path.is_ident("sanitize") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Str(s) = lit {
                    let mode = s.value();
                    match mode.as_str() {
                        "trim" | "lowercase" | "strip_html" | "slugify" => {
                            result.sanitize = Some(mode);
                        }
                        _ => {
                            return Err(meta.error(
                                "unknown sanitize mode, expected `trim`, `lowercase`, `strip_html`, or `slugify`",
                            ));
                        }
                    }
                }
                return Ok(());
            }
            if meta.path.is_ident("pattern") {
                // Substring pattern, not regex. The value must contain
                // the pattern string. Kept dependency-free (no regex crate).
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Str(s) = lit {
                    result.pattern = Some(s.value());
                }
                return Ok(());
            }
            if meta.path.is_ident("empty_is_none") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Bool(b) = lit {
                    result.empty_is_none = b.value();
                }
                return Ok(());
            }
            if meta.path.is_ident("coerce") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Str(s) = lit {
                    result.coerce = Some(s.value());
                }
                return Ok(());
            }
            if meta.path.is_ident("parse") {
                let value = meta.value()?;
                let lit: Lit = value.parse()?;
                if let Lit::Str(s) = lit {
                    match s.value().as_str() {
                        "number" => result.parse = Some("number".into()),
                        other => {
                            return Err(meta.error(format!(
                                "unknown parse mode, expected `number`, got `{other}`"
                            )));
                        }
                    }
                }
                return Ok(());
            }
            if meta.path.is_ident("clamp") {
                result.clamp = true;
                return Ok(());
            }
            Err(meta.error(
                "unknown field attribute, expected `skip`, `desc`, `min`, `max`, `clamp`, `required`, `format`, `max_len`, `sanitize`, `pattern`, `coerce`, `parse`, or `empty_is_none`",
            ))
        });
    }

    result
}

fn is_numeric_type(ty_str: &str) -> bool {
    matches!(
        ty_str,
        "u8" | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "f32"
            | "f64"
            | "usize"
            | "isize"
    )
}

fn quote_type_string(ty_str: &str) -> TokenStream2 {
    if matches!(
        ty_str,
        "u8" | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "usize"
            | "isize"
    ) {
        quote! { "integer" }
    } else if matches!(ty_str, "f32" | "f64") {
        quote! { "number" }
    } else if ty_str == "bool" {
        quote! { "boolean" }
    } else {
        quote! { "string" }
    }
}

/// Derive macro for `fluent_wvr::FieldAccess`.
///
/// Generates `set_field`, `get_field`, and `field_names` implementations
/// that intern property names via `ArcIntern<str>` for O(1) pointer-sized
/// key matching in routing boundaries.
///
/// Supports optional field attributes:
/// - `#[field(skip)]` — exclude this field from runtime access and schema
/// - `#[field(desc = "...")]` — field description (used by `Describable`)
/// - `#[field(min = N)]` — minimum value constraint (numeric fields only)
/// - `#[field(max = N)]` — maximum value constraint (numeric fields only)
/// - `#[field(format = "...")]` — JSON Schema format hint (informational)
/// - `#[field(max_len = N)]` — maximum string length
/// - `#[field(sanitize = "trim,lowercase")]` — sanitization mode
/// - `#[field(pattern = "...")]` — substring pattern; value must contain this string
#[proc_macro_derive(FieldAccess, attributes(field))]
pub fn derive_field_access(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => &fields.named,
        _ => {
            return syn::Error::new_spanned(
                input,
                "FieldAccess can only be derived for structs with named fields",
            )
            .to_compile_error()
            .into();
        }
    };

    let field_name_strs: Vec<String> = fields
        .iter()
        .filter(|f| !parse_field_attrs(f).skip)
        .map(|f| f.ident.as_ref().unwrap().to_string())
        .collect();

    let mut set_body = TokenStream2::new();
    let mut get_body = TokenStream2::new();

    let mut non_skip_idx = 0usize;
    for f in fields.iter() {
        let field_ident = f.ident.as_ref().unwrap();
        let field_name_str = field_ident.to_string();
        let ty = &f.ty;
        let ty_str = quote!(#ty).to_string();
        let meta = parse_field_attrs(f);

        if meta.skip {
            continue;
        }

        let has_constraints = meta.min.is_some() || meta.max.is_some();

        // Detect Option<T> by checking if the type string contains "Option" with angle brackets
        let normalized_ty = ty_str.replace(' ', "");
        let is_option = normalized_ty.starts_with("Option<")
            || normalized_ty.starts_with("std::option::Option<");
        let inner_ty_str = if is_option {
            normalized_ty
                .strip_prefix("Option<")
                .or_else(|| normalized_ty.strip_prefix("std::option::Option<"))
                .and_then(|s| s.strip_suffix('>'))
                .unwrap_or("String")
                .to_string()
        } else {
            ty_str.clone()
        };
        let inner_is_numeric = is_numeric_type(&inner_ty_str);
        let number_tolerant = meta.parse.as_deref() == Some("number") && inner_is_numeric;
        let inner_ty_tokens: TokenStream2 = if number_tolerant {
            syn::parse_str::<TokenStream2>(&inner_ty_str).unwrap_or_else(|_| {
                syn::parse_str::<TokenStream2>("i64").expect("i64 parses")
            })
        } else {
            TokenStream2::new()
        };
        // Numeric initializer honoring `#[field(parse = "number")]`: tolerant
        // parse_number when requested, strict str::parse otherwise.
        let wide_init_tokens: TokenStream2 = if number_tolerant {
            quote! {
                ::fluent_wvr::coerce::parse_number(&value).ok_or_else(|| fluent_wvr::FieldError::Parse(
                    format!("invalid {} for '{}': {}", #ty_str, #field_name_str, value)
                ))?
            }
        } else {
            quote! {
                value.parse().map_err(|_| fluent_wvr::FieldError::Parse(
                    format!("invalid {} for '{}': {}", #ty_str, #field_name_str, value)
                ))?
            }
        };

        let mut parse_and_set = if is_option {
            let inner_is_string = inner_ty_str == "String"
                || inner_ty_str == "std::string::String"
                || inner_ty_str.ends_with("::String");
            if inner_is_string && meta.empty_is_none {
                quote! {
                    if value.is_empty() {
                        None
                    } else {
                        Some(value.to_string())
                    }
                }
            } else if inner_is_string {
                quote! {
                    Some(value.to_string())
                }
            } else if number_tolerant {
                quote! {
                    if value.is_empty() {
                        None
                    } else {
                        Some(::fluent_wvr::coerce::parse_number(&value).ok_or_else(|| fluent_wvr::FieldError::Parse(
                            format!("invalid Option<{}> for '{}': {}", #inner_ty_str, #field_name_str, value)
                        ))? as #inner_ty_tokens)
                    }
                }
            } else {
                quote! {
                    if value.is_empty() {
                        None
                    } else {
                        Some(value.parse::<#inner_ty_tokens>().map_err(|_| fluent_wvr::FieldError::Parse(
                            format!("invalid Option<{}> for '{}': {}", #inner_ty_str, #field_name_str, value)
                        ))?)
                    }
                }
            }
        } else if ty_str == "String"
            || ty_str == "std::string::String"
            || ty_str.ends_with("::String")
        {
            quote! { value.into() }
        } else if ty_str == "bool" {
            quote! {
                value.parse().map_err(|_| fluent_wvr::FieldError::Parse(
                    format!("invalid bool for '{}': {}", #field_name_str, value)
                ))?
            }
        } else if ty_str.starts_with("ArcIntern") || ty_str.contains("ArcIntern") {
            quote! { fluent_wvr::ArcIntern::from(value) }
        } else if has_constraints && inner_is_numeric {
            quote! {
                {
                    let wide_val: f64 = #wide_init_tokens;
                    wide_val as #ty
                }
            }
        } else if number_tolerant {
            quote! {
                ::fluent_wvr::coerce::parse_number(&value).ok_or_else(|| fluent_wvr::FieldError::Parse(
                    format!("invalid {} for '{}': {}", #ty_str, #field_name_str, value)
                ))? as #ty
            }
        } else {
            quote! {
                value.parse::<#ty>().map_err(|_| fluent_wvr::FieldError::Parse(
                    format!("invalid {} for '{}': {}", #ty_str, #field_name_str, value)
                ))?
            }
        };

        // Apply string sanitization before constraint checks
        if !is_option
            && (ty_str == "String"
                || ty_str == "std::string::String"
                || ty_str.ends_with("::String"))
        {
            if let Some(ref sanitize_mode) = meta.sanitize {
                match sanitize_mode.as_str() {
                    "trim" => {
                        parse_and_set = quote! {
                            {
                                let s: String = #parse_and_set;
                                s.trim().to_string()
                            }
                        };
                    }
                    "lowercase" => {
                        parse_and_set = quote! {
                            {
                                let s: String = #parse_and_set;
                                s.to_lowercase()
                            }
                        };
                    }
                    "strip_html" => {
                        parse_and_set = quote! {
                            {
                                let s: String = #parse_and_set;
                                ::fluent_wvr::strip_html(&s)
                            }
                        };
                    }
                    "slugify" => {
                        parse_and_set = quote! {
                            {
                                let s: String = #parse_and_set;
                                ::fluent_wvr::slugify(&s)
                            }
                        };
                    }
                    _ => {
                        return syn::Error::new_spanned(
                            f,
                            format!("unknown sanitize mode: {}", sanitize_mode),
                        )
                        .to_compile_error()
                        .into();
                    }
                }
            }

            // Apply max_len check
            if let Some(max_len) = meta.max_len {
                let max_len_lit = proc_macro2::Literal::usize_suffixed(max_len);
                let max_len_err = format!(
                    "{}: string length exceeds maximum {}",
                    field_name_str, max_len
                );
                parse_and_set = quote! {
                    {
                        let s: String = #parse_and_set;
                        if s.chars().count() > #max_len_lit {
                            return Err(fluent_wvr::FieldError::Constraint(#max_len_err.into()));
                        }
                        s
                    }
                };
            }

            // Apply pattern check
            if let Some(ref pattern) = meta.pattern {
                let pattern_lit = pattern.clone();
                let pattern_err = format!(
                    "{}: value does not match pattern '{}'",
                    field_name_str, pattern
                );
                parse_and_set = quote! {
                    {
                        let s: String = #parse_and_set;
                        if !s.contains(#pattern_lit) {
                            return Err(fluent_wvr::FieldError::Constraint(#pattern_err.into()));
                        }
                        s
                    }
                };
            }
        }

        if is_numeric_type(&ty_str) {
            let min_val = meta.min;
            let max_val = meta.max;

            if meta.clamp {
                let min_lit = min_val.map(proc_macro2::Literal::f64_suffixed);
                let max_lit = max_val.map(proc_macro2::Literal::f64_suffixed);
                match (min_lit, max_lit) {
                    (Some(min), Some(max)) => {
                        parse_and_set = quote! {
                            {
                                let wide: f64 = #wide_init_tokens;
                                (wide.clamp(#min, #max)) as #ty
                            }
                        };
                    }
                    _ => {
                        return syn::Error::new_spanned(
                            f,
                            "`clamp` requires both `min` and `max` to be specified",
                        )
                        .to_compile_error()
                        .into();
                    }
                }
            } else {
                let min_check = min_val.map(|min_val| {
                    let min_lit = proc_macro2::Literal::f64_suffixed(min_val);
                    let min_err = format!("{}: value below minimum {}", field_name_str, min_val);
                    quote! {
                        if wide < #min_lit {
                            return Err(fluent_wvr::FieldError::Constraint(#min_err.into()));
                        }
                    }
                });
                let max_check = max_val.map(|max_val| {
                    let max_lit = proc_macro2::Literal::f64_suffixed(max_val);
                    let max_err = format!("{}: value above maximum {}", field_name_str, max_val);
                    quote! {
                        if wide > #max_lit {
                            return Err(fluent_wvr::FieldError::Constraint(#max_err.into()));
                        }
                    }
                });

                if min_check.is_some() || max_check.is_some() {
                    parse_and_set = quote! {
                        {
                            let wide: f64 = #wide_init_tokens;
                            #min_check
                            #max_check
                            wide as #ty
                        }
                    };
                }
            }
        }

        // Boundary-string coercion pipeline (`#[field(coerce = "...")]`)
        // applied before parsing/constraints: shapes the untrusted string the
        // same way `fluent_wvr::boundary` does, so both decode paths share one
        // vocabulary.
        let coerce_stmt: TokenStream2 = if let Some(ref modes) = meta.coerce {
            let mut tokens: Vec<TokenStream2> = Vec::new();
            let mut bad: Option<String> = None;
            for m in modes.split(',') {
                let m = m.trim();
                if m.is_empty() {
                    continue;
                }
                match m {
                    "trim" => tokens.push(quote! { ::fluent_wvr::coerce::Coercion::Trim }),
                    "lowercase" => {
                        tokens.push(quote! { ::fluent_wvr::coerce::Coercion::Lowercase })
                    }
                    "strip_quotes" => {
                        tokens.push(quote! { ::fluent_wvr::coerce::Coercion::StripQuotes })
                    }
                    "json_escape" => {
                        tokens.push(quote! { ::fluent_wvr::coerce::Coercion::JsonEscape })
                    }
                    "normalize_literal" => {
                        tokens.push(quote! { ::fluent_wvr::coerce::Coercion::NormalizeLiteral })
                    }
                    other => {
                        bad = Some(other.to_string());
                        break;
                    }
                }
            }
            if let Some(other) = bad {
                syn::Error::new_spanned(f, format!("unknown coerce mode: {other}"))
                    .to_compile_error()
            } else {
                quote! {
                    let value = ::fluent_wvr::coerce::coerce(value, &[#(#tokens),*]);
                }
            }
        } else {
            quote! {}
        };

        let set_expr = quote! {
            #coerce_stmt
            self.#field_ident = #parse_and_set;
            Ok(())
        };

        if non_skip_idx == 0 {
            set_body.extend(quote! {
                if name == #field_name_str {
                    #set_expr
                }
            });
        } else {
            set_body.extend(quote! {
                else if name == #field_name_str {
                    #set_expr
                }
            });
        }

        let to_string_expr = if is_option {
            let inner_is_string = inner_ty_str == "String"
                || inner_ty_str == "std::string::String"
                || inner_ty_str.ends_with("::String");
            if inner_is_string {
                quote! { self.#field_ident.as_deref().unwrap_or("").to_string() }
            } else {
                // Numeric/other Option inner types have no Deref: render the
                // contained value (or empty when None).
                quote! { self.#field_ident.map(|v| v.to_string()).unwrap_or_default() }
            }
        } else if ty_str == "String"
            || ty_str == "std::string::String"
            || ty_str.ends_with("::String")
        {
            quote! { self.#field_ident.clone() }
        } else {
            quote! { self.#field_ident.to_string() }
        };

        if non_skip_idx == 0 {
            get_body.extend(quote! {
                if name == #field_name_str {
                    Ok(#to_string_expr)
                }
            });
        } else {
            get_body.extend(quote! {
                else if name == #field_name_str {
                    Ok(#to_string_expr)
                }
            });
        }

        non_skip_idx += 1;
    }

    let expanded = if fields.is_empty() {
        quote! {
            impl #impl_generics fluent_wvr::FieldAccess for #name #ty_generics #where_clause {
                fn set_field(&mut self, name: &str, _value: &str) -> Result<(), fluent_wvr::FieldError> {
                    Err(fluent_wvr::FieldError::NotFound(name.into()))
                }

                fn get_field(&self, name: &str) -> Result<String, fluent_wvr::FieldError> {
                    Err(fluent_wvr::FieldError::NotFound(name.into()))
                }

                fn field_names(&self) -> &'static [&'static str] {
                    static NAMES: &[&str] = &[];
                    NAMES
                }
            }
        }
    } else {
        quote! {
            impl #impl_generics fluent_wvr::FieldAccess for #name #ty_generics #where_clause {
                fn set_field(&mut self, name: &str, value: &str) -> Result<(), fluent_wvr::FieldError> {
                    #set_body else {
                        Err(fluent_wvr::FieldError::NotFound(name.into()))
                    }
                }

                fn get_field(&self, name: &str) -> Result<String, fluent_wvr::FieldError> {
                    #get_body else {
                        Err(fluent_wvr::FieldError::NotFound(name.into()))
                    }
                }

                fn field_names(&self) -> &'static [&'static str] {
                    static NAMES: &[&str] = &[#(#field_name_strs),*];
                    NAMES
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for `fluent_wvr::Describable`.
///
/// Generates a `describe()` method that returns a JSON Schema representation
/// of the struct, using `#[field(...)]` attributes for descriptions and constraints.
#[proc_macro_derive(Describable, attributes(field))]
pub fn derive_describable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => &fields.named,
        _ => {
            return syn::Error::new_spanned(
                input,
                "Describable can only be derived for structs with named fields",
            )
            .to_compile_error()
            .into();
        }
    };

    let mut properties = Vec::new();
    let mut required = Vec::new();
    let mut schema_fields = Vec::new();

    for f in fields {
        let field_ident = f.ident.as_ref().unwrap();
        let field_name_str = field_ident.to_string();
        let ty = &f.ty;
        let ty_str = quote!(#ty).to_string();
        let meta = parse_field_attrs(f);

        if meta.skip {
            continue;
        }

        let mut schema = Vec::new();

        let type_str = quote_type_string(&ty_str);
        schema.push(quote! { "type": #type_str });

        // Option<T> fields are not required
        let normalized_ty_for_desc = ty_str.replace(' ', "");
        let is_option = normalized_ty_for_desc.starts_with("Option<")
            || normalized_ty_for_desc.starts_with("std::option::Option<");
        let effective_required = if is_option { false } else { meta.required };

        if let Some(ref desc) = meta.desc {
            schema.push(quote! { "description": #desc });
        }

        if is_numeric_type(&ty_str) {
            if let Some(min) = meta.min {
                let min_str = format!("{}", min);
                schema.push(quote! { "minimum": #min_str });
            }
            if let Some(max) = meta.max {
                let max_str = format!("{}", max);
                schema.push(quote! { "maximum": #max_str });
            }
        }

        if let Some(ref fmt) = meta.format {
            schema.push(quote! { "format": #fmt });
        }

        if let Some(max_len) = meta.max_len {
            let max_len_str = max_len.to_string();
            schema.push(quote! { "maxLength": #max_len_str });
        }

        if let Some(ref sanitize) = meta.sanitize {
            schema.push(quote! { "sanitize": #sanitize });
        }

        if let Some(ref pattern) = meta.pattern {
            schema.push(quote! { "pattern": #pattern });
        }

        if let Some(ref coerce) = meta.coerce {
            schema.push(quote! { "coerce": #coerce });
        }

        if let Some(ref parse) = meta.parse {
            schema.push(quote! { "parse": #parse });
        }

        let field_name_lit = field_name_str.clone();
        properties.push(quote! {
            #field_name_lit: {
                #(#schema),*
            }
        });

        let desc_expr = match &meta.desc {
            Some(d) => quote! { Some(#d.into()) },
            None => quote! { None },
        };
        let min_expr = match meta.min {
            Some(v) => quote! { Some(#v) },
            None => quote! { None },
        };
        let max_expr = match meta.max {
            Some(v) => quote! { Some(#v) },
            None => quote! { None },
        };
        let type_name_str = ty_str.clone();

        let required_expr = if effective_required {
            quote! { true }
        } else {
            quote! { false }
        };

        let format_expr = match &meta.format {
            Some(f) => quote! { Some(#f.into()) },
            None => quote! { None },
        };
        let max_len_expr = match meta.max_len {
            Some(v) => quote! { Some(#v) },
            None => quote! { None },
        };
        let sanitize_expr = match &meta.sanitize {
            Some(s) => quote! { Some(#s.into()) },
            None => quote! { None },
        };
        let pattern_expr = match &meta.pattern {
            Some(p) => quote! { Some(#p.into()) },
            None => quote! { None },
        };
        let coerce_expr = match &meta.coerce {
            Some(c) => quote! { Some(#c.into()) },
            None => quote! { None },
        };
        let parse_expr = match &meta.parse {
            Some(p) => quote! { Some(#p.into()) },
            None => quote! { None },
        };

        schema_fields.push(quote! {
            fluent_wvr::FieldSchema {
                name: #field_name_str.into(),
                type_name: #type_name_str.into(),
                description: #desc_expr,
                min: #min_expr,
                max: #max_expr,
                required: #required_expr,
                format: #format_expr,
                max_len: #max_len_expr,
                sanitize: #sanitize_expr,
                pattern: #pattern_expr,
                coerce: #coerce_expr,
                parse: #parse_expr,
            }
        });

        if effective_required {
            required.push(field_name_str);
        }
    }

    let expanded = quote! {
        impl #impl_generics fluent_wvr::Describable for #name #ty_generics #where_clause {
            fn describe(&self) -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        #(#properties),*
                    },
                    "required": [#(#required),*]
                })
            }
        }

        impl #impl_generics fluent_wvr::SchemaProvider for #name #ty_generics #where_clause {
            fn schema(&self) -> Vec<fluent_wvr::FieldSchema> {
                vec![#(#schema_fields),*]
            }
        }
    };

    TokenStream::from(expanded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_field(attrs: &str, ty: &str, name: &str) -> syn::Field {
        let input: syn::DeriveInput = syn::parse_str(&format!(
            "struct Foo {{ #[field({})] {}: {} }}",
            attrs, name, ty
        ))
        .unwrap();
        match input.data {
            syn::Data::Struct(s) => match s.fields {
                syn::Fields::Named(named) => named.named.into_iter().next().unwrap(),
                _ => panic!("expected named fields"),
            },
            _ => panic!("expected struct"),
        }
    }

    #[test]
    fn parse_field_attrs_extracts_desc() {
        let field = make_field(r#"desc = "TCP port""#, "u16", "port");
        let meta = parse_field_attrs(&field);
        assert_eq!(meta.desc.as_deref(), Some("TCP port"));
        assert!(meta.min.is_none());
        assert!(meta.max.is_none());
        assert!(!meta.skip);
        assert!(meta.required);
    }

    #[test]
    fn parse_field_attrs_extracts_min_max() {
        let field = make_field("min = 1, max = 65535", "u16", "port");
        let meta = parse_field_attrs(&field);
        assert_eq!(meta.min, Some(1.0));
        assert_eq!(meta.max, Some(65535.0));
    }

    #[test]
    fn parse_field_attrs_extracts_skip() {
        let field = make_field("skip", "String", "ignored");
        let meta = parse_field_attrs(&field);
        assert!(meta.skip);
    }

    #[test]
    fn parse_field_attrs_extracts_required_false() {
        let field = make_field("required = false", "Option<String>", "nickname");
        let meta = parse_field_attrs(&field);
        assert!(!meta.required);
    }

    #[test]
    fn parse_field_attrs_extracts_format() {
        let field = make_field(r#"format = "url""#, "String", "endpoint");
        let meta = parse_field_attrs(&field);
        assert_eq!(meta.format.as_deref(), Some("url"));
    }

    #[test]
    fn parse_field_attrs_extracts_max_len() {
        let field = make_field("max_len = 100", "String", "name");
        let meta = parse_field_attrs(&field);
        assert_eq!(meta.max_len, Some(100));
    }

    #[test]
    fn parse_field_attrs_extracts_sanitize() {
        let field = make_field(r#"sanitize = "trim""#, "String", "name");
        let meta = parse_field_attrs(&field);
        assert_eq!(meta.sanitize.as_deref(), Some("trim"));
    }

    #[test]
    fn parse_field_attrs_extracts_pattern() {
        let field = make_field(r#"pattern = "https://""#, "String", "url");
        let meta = parse_field_attrs(&field);
        assert_eq!(meta.pattern.as_deref(), Some("https://"));
    }

    #[test]
    fn parse_field_attrs_extracts_empty_is_none() {
        let field = make_field("empty_is_none = false", "Option<String>", "email");
        let meta = parse_field_attrs(&field);
        assert!(!meta.empty_is_none);
    }

    #[test]
    fn parse_field_attrs_default_empty_is_none_true() {
        let field = make_field("required = false", "Option<String>", "nickname");
        let meta = parse_field_attrs(&field);
        assert!(meta.empty_is_none);
    }

    #[test]
    fn is_numeric_type_true_for_integers() {
        for ty in &[
            "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128", "usize", "isize",
        ] {
            assert!(is_numeric_type(ty), "expected {ty} to be numeric");
        }
    }

    #[test]
    fn is_numeric_type_true_for_floats() {
        assert!(is_numeric_type("f32"));
        assert!(is_numeric_type("f64"));
    }

    #[test]
    fn is_numeric_type_false_for_non_numeric() {
        assert!(!is_numeric_type("String"));
        assert!(!is_numeric_type("bool"));
        assert!(!is_numeric_type("Option<u32>"));
        assert!(!is_numeric_type("Vec<u8>"));
    }

    #[test]
    fn quote_type_string_maps_integers_to_integer() {
        for ty in &[
            "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128", "usize", "isize",
        ] {
            let ts = quote_type_string(ty);
            let s = ts.to_string();
            assert_eq!(
                s.trim_matches('"'),
                "integer",
                "expected {ty} -> integer, got {s}"
            );
        }
    }

    #[test]
    fn quote_type_string_maps_floats_to_number() {
        let ts = quote_type_string("f64");
        let s = ts.to_string();
        assert_eq!(s.trim_matches('"'), "number", "got {s}");
    }

    #[test]
    fn quote_type_string_maps_bool_to_boolean() {
        let ts = quote_type_string("bool");
        let s = ts.to_string();
        assert_eq!(s.trim_matches('"'), "boolean", "got {s}");
    }

    #[test]
    fn quote_type_string_maps_string_to_string() {
        let ts = quote_type_string("String");
        let s = ts.to_string();
        assert_eq!(s.trim_matches('"'), "string", "got {s}");
    }

    #[test]
    fn unknown_sanitize_mode_returns_none() {
        // parse_field_attrs returns sanitize: None for unknown modes
        // (the error from parse_nested_meta is swallowed by `let _ =`)
        let field = make_field(r#"sanitize = "invalid""#, "String", "x");
        let meta = parse_field_attrs(&field);
        assert_eq!(
            meta.sanitize, None,
            "unknown sanitize mode should not be set"
        );
    }
}
