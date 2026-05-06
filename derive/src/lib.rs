use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{
    parse_macro_input, AttrStyle, Attribute, Data, DeriveInput, Expr, Fields, Index, LitStr, Type,
    TypeGenerics, TypePath,
};

/// Derive the `H5Type` trait for structs and enums.
///
/// # Structs
///
/// Structs must have `#[repr(C)]` or `#[repr(packed)]`. Each field must implement `H5Type`.
///
/// ```ignore
/// #[derive(Copy, Clone, H5Type)]
/// #[repr(C)]
/// struct Point {
///     x: f64,
///     y: f64,
///     label: i32,
/// }
/// ```
///
/// Fields can be renamed with `#[hdf5(rename = "name")]`.
///
/// # Enums
///
/// Enums must have an explicit integer repr and all variants must be unit variants
/// with explicit discriminants.
///
/// ```ignore
/// #[derive(Copy, Clone, H5Type)]
/// #[repr(u8)]
/// enum Color {
///     Red = 0,
///     Green = 1,
///     Blue = 2,
/// }
/// ```
#[proc_macro_derive(H5Type, attributes(hdf5))]
pub fn derive_h5type(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let body = impl_trait(name, &input.data, &input.attrs, &ty_generics);
    let expanded = quote! {
        #[automatically_derived]
        #[allow(unused_variables)]
        unsafe impl #impl_generics ::hdf5_pure_rust::H5Type for #name #ty_generics #where_clause {
            #[inline]
            fn type_size() -> usize {
                ::std::mem::size_of::<#name #ty_generics>()
            }

            #body
        }
    };

    proc_macro::TokenStream::from(expanded)
}

fn impl_trait(
    ty: &Ident,
    data: &Data,
    attrs: &[Attribute],
    ty_generics: &TypeGenerics,
) -> TokenStream {
    match *data {
        Data::Struct(ref data) => impl_struct(ty, data, attrs, ty_generics),
        Data::Enum(ref data) => impl_enum(ty, data, attrs),
        Data::Union(_) => {
            panic!("cannot derive `H5Type` for unions");
        }
    }
}

fn impl_struct(
    ty: &Ident,
    data: &syn::DataStruct,
    attrs: &[Attribute],
    ty_generics: &TypeGenerics,
) -> TokenStream {
    match data.fields {
        Fields::Unit => panic!("cannot derive `H5Type` for unit structs"),
        Fields::Named(ref fields) => {
            let fields: Vec<_> = fields
                .named
                .iter()
                .filter(|f| !is_phantom_data(&f.ty))
                .collect();
            if fields.is_empty() {
                panic!("cannot derive `H5Type` for empty structs");
            }

            let repr = find_repr(attrs, &["C", "packed", "transparent"]);
            if repr.is_none() {
                panic!("`H5Type` requires #[repr(C)] or #[repr(packed)] for structs");
            }
            let repr = repr.unwrap();

            if repr == "transparent" {
                if fields.len() != 1 {
                    panic!("#[repr(transparent)] requires exactly one non-PhantomData field");
                }
                let inner_ty = &fields[0].ty;
                return quote! {
                    fn compound_fields() -> Option<Vec<::hdf5_pure_rust::hl::types::FieldDescriptor>> {
                        <#inner_ty as ::hdf5_pure_rust::H5Type>::compound_fields()
                    }
                    fn enum_members() -> Option<Vec<(String, i64)>> {
                        <#inner_ty as ::hdf5_pure_rust::H5Type>::enum_members()
                    }
                };
            }

            let field_names: Vec<String> = fields
                .iter()
                .map(|f| {
                    find_hdf5_rename(&f.attrs)
                        .unwrap_or_else(|| f.ident.as_ref().unwrap().to_string())
                })
                .collect();
            let field_idents: Vec<_> = fields.iter().map(|f| f.ident.clone().unwrap()).collect();
            let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

            quote! {
                fn compound_fields() -> Option<Vec<::hdf5_pure_rust::hl::types::FieldDescriptor>> {
                    let origin = ::std::mem::MaybeUninit::<#ty #ty_generics>::uninit();
                    let origin_ptr = origin.as_ptr();
                    let fields = vec![#(
                        ::hdf5_pure_rust::hl::types::FieldDescriptor {
                            name: #field_names.to_string(),
                            offset: unsafe {
                                ::std::ptr::addr_of!((*origin_ptr).#field_idents).cast::<u8>()
                                    .offset_from(origin_ptr.cast()) as usize
                            },
                            size: <#field_types as ::hdf5_pure_rust::H5Type>::type_size(),
                            type_class: {
                                // Determine type class from size and signedness
                                let size = <#field_types as ::hdf5_pure_rust::H5Type>::type_size();
                                if size == 4 || size == 8 {
                                    // Could be float or integer -- use type_id to distinguish
                                    if ::std::any::TypeId::of::<#field_types>() == ::std::any::TypeId::of::<f32>()
                                        || ::std::any::TypeId::of::<#field_types>() == ::std::any::TypeId::of::<f64>()
                                    {
                                        ::hdf5_pure_rust::hl::types::TypeClass::Float
                                    } else if ::std::any::TypeId::of::<#field_types>() == ::std::any::TypeId::of::<i8>()
                                        || ::std::any::TypeId::of::<#field_types>() == ::std::any::TypeId::of::<i16>()
                                        || ::std::any::TypeId::of::<#field_types>() == ::std::any::TypeId::of::<i32>()
                                        || ::std::any::TypeId::of::<#field_types>() == ::std::any::TypeId::of::<i64>()
                                    {
                                        ::hdf5_pure_rust::hl::types::TypeClass::Integer { signed: true }
                                    } else {
                                        ::hdf5_pure_rust::hl::types::TypeClass::Integer { signed: false }
                                    }
                                } else {
                                    // Small sizes are likely integers
                                    if ::std::any::TypeId::of::<#field_types>() == ::std::any::TypeId::of::<i8>()
                                        || ::std::any::TypeId::of::<#field_types>() == ::std::any::TypeId::of::<i16>()
                                    {
                                        ::hdf5_pure_rust::hl::types::TypeClass::Integer { signed: true }
                                    } else {
                                        ::hdf5_pure_rust::hl::types::TypeClass::Integer { signed: false }
                                    }
                                }
                            },
                        }
                    ),*];
                    Some(fields)
                }
            }
        }
        Fields::Unnamed(ref fields) => {
            let fields: Vec<_> = fields
                .unnamed
                .iter()
                .enumerate()
                .filter(|(_, f)| !is_phantom_data(&f.ty))
                .collect();
            if fields.is_empty() {
                panic!("cannot derive `H5Type` for empty tuple structs");
            }

            let repr = find_repr(attrs, &["C", "packed", "transparent"]);
            if repr.is_none() {
                panic!("`H5Type` requires #[repr(C)] or #[repr(packed)] for tuple structs");
            }

            let field_names: Vec<String> = fields
                .iter()
                .enumerate()
                .map(|(n, (_, f))| find_hdf5_rename(&f.attrs).unwrap_or_else(|| n.to_string()))
                .collect();
            let field_indices: Vec<Index> = fields.iter().map(|(i, _)| Index::from(*i)).collect();
            let field_types: Vec<_> = fields.iter().map(|(_, f)| &f.ty).collect();

            quote! {
                fn compound_fields() -> Option<Vec<::hdf5_pure_rust::hl::types::FieldDescriptor>> {
                    let origin = ::std::mem::MaybeUninit::<#ty #ty_generics>::uninit();
                    let origin_ptr = origin.as_ptr();
                    let fields = vec![#(
                        ::hdf5_pure_rust::hl::types::FieldDescriptor {
                            name: #field_names.to_string(),
                            offset: unsafe {
                                ::std::ptr::addr_of!((*origin_ptr).#field_indices).cast::<u8>()
                                    .offset_from(origin_ptr.cast()) as usize
                            },
                            size: <#field_types as ::hdf5_pure_rust::H5Type>::type_size(),
                            type_class: ::hdf5_pure_rust::hl::types::TypeClass::Integer { signed: false },
                        }
                    ),*];
                    Some(fields)
                }
            }
        }
    }
}

fn impl_enum(_ty: &Ident, data: &syn::DataEnum, attrs: &[Attribute]) -> TokenStream {
    let variants = &data.variants;

    if variants
        .iter()
        .any(|v| v.fields != Fields::Unit || v.discriminant.is_none())
    {
        panic!("`H5Type` can only be derived for enums with scalar discriminants");
    }
    if variants.is_empty() {
        panic!("cannot derive `H5Type` for empty enums");
    }

    let enum_reprs = &[
        "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "isize", "usize",
    ];
    let repr = find_repr(attrs, enum_reprs);
    if repr.is_none() {
        panic!("`H5Type` requires explicit integer repr for enums");
    }
    let repr = repr.unwrap();

    let names: Vec<String> = variants
        .iter()
        .map(|v| find_hdf5_rename(&v.attrs).unwrap_or_else(|| v.ident.to_string()))
        .collect();
    let values: Vec<&Expr> = variants
        .iter()
        .map(|v| &v.discriminant.as_ref().unwrap().1)
        .collect();
    let repr_iter = std::iter::repeat(&repr);

    quote! {
        fn enum_members() -> Option<Vec<(String, i64)>> {
            Some(vec![#(
                (#names.to_string(), (#values) as #repr_iter as i64)
            ),*])
        }
    }
}

fn is_phantom_data(ty: &Type) -> bool {
    match *ty {
        Type::Path(TypePath {
            qself: None,
            ref path,
        }) => path
            .segments
            .iter()
            .last()
            .is_some_and(|x| x.ident == "PhantomData"),
        _ => false,
    }
}

fn find_repr(attrs: &[Attribute], expected: &[&str]) -> Option<Ident> {
    let mut repr = None;
    for attr in attrs {
        if attr.style != AttrStyle::Outer || !attr.path().is_ident("repr") {
            continue;
        }
        let _ = attr.parse_nested_meta(|meta| {
            if expected.iter().any(|s| meta.path.is_ident(s)) {
                repr = meta.path.get_ident().cloned();
            }
            Ok(())
        });
    }
    repr
}

fn find_hdf5_rename(attrs: &[Attribute]) -> Option<String> {
    let mut rename = None;
    let attr = attrs.iter().find(|a| a.path().is_ident("hdf5"))?;
    let _ = attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("rename") && rename.is_none() {
            rename = Some(meta.value()?.parse::<LitStr>()?.value());
        }
        Ok(())
    });
    rename
}
