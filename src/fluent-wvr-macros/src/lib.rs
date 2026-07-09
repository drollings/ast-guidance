use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, Data, DataStruct, DeriveInput, Fields, Lit};

struct FieldMeta {
    desc: Option<String>,
    min: Option<f64>,
    max: Option<f64>,
    skip: bool,
}

fn parse_field_attrs(field: &syn::Field) -> FieldMeta {
    let mut result = FieldMeta {
        desc: None,
        min: None,
        max: None,
        skip: false,
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
                    Lit::Float(f) => result.max = Some(f.base10_parse::<f64>().unwrap()),
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
            Err(meta.error("unknown field attribute, expected `skip`, `desc`, `min`, or `max`"))
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

        let mut parse_and_set = if ty_str == "String"
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
        } else if has_constraints && is_numeric_type(&ty_str) {
            quote! {
                {
                    let wide_val: f64 = value.parse().map_err(|_| fluent_wvr::FieldError::Parse(
                        format!("invalid {} for '{}': {}", #ty_str, #field_name_str, value)
                    ))?;
                    wide_val as #ty
                }
            }
        } else {
            quote! {
                value.parse::<#ty>().map_err(|_| fluent_wvr::FieldError::Parse(
                    format!("invalid {} for '{}': {}", #ty_str, #field_name_str, value)
                ))?
            }
        };

        if is_numeric_type(&ty_str) {
            let min_check = meta.min.map(|min_val| {
                let min_lit = proc_macro2::Literal::f64_suffixed(min_val);
                let min_err = format!("{}: value below minimum {}", field_name_str, min_val);
                quote! {
                    if wide < #min_lit {
                        return Err(fluent_wvr::FieldError::Constraint(#min_err.into()));
                    }
                }
            });
            let max_check = meta.max.map(|max_val| {
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
                        let wide: f64 = value.parse().map_err(|_| fluent_wvr::FieldError::Parse(
                            format!("invalid {} for '{}': {}", #ty_str, #field_name_str, value)
                        ))?;
                        #min_check
                        #max_check
                        wide as #ty
                    }
                };
            }
        }

        let set_expr = quote! {
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

        let to_string_expr = if ty_str == "String"
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

        schema_fields.push(quote! {
            fluent_wvr::FieldSchema {
                name: #field_name_str.into(),
                type_name: #type_name_str.into(),
                description: #desc_expr,
                min: #min_expr,
                max: #max_expr,
                required: true,
            }
        });

        required.push(field_name_str);
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
