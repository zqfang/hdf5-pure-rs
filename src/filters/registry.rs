use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::format::messages::filter_pipeline::{
    FilterDesc, FilterPipelineMessage, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_NBIT,
    FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP,
};

const FILTER_LZF: u16 = 32_000;
const FILTER_BLOSC: u16 = 32_001;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredFilter {
    pub id: u16,
    pub name: Option<String>,
    pub flags: u16,
    pub client_data: Vec<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilterRegistry {
    filters: BTreeMap<u16, RegisteredFilter>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterInfo {
    pub id: u16,
    pub flags: u16,
    pub client_data_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum XformExpr {
    Number(f64),
    Variable,
    UnaryMinus(Box<XformExpr>),
    Binary {
        op: XformOp,
        left: Box<XformExpr>,
        right: Box<XformExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XformOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XformNodeType {
    Number,
    Variable,
    Unary,
    Binary,
}

#[derive(Debug, Clone, PartialEq)]
pub enum XformToken {
    Number(f64),
    Variable,
    Operator(char),
    LParen,
    RParen,
}

impl RegisteredFilter {
    pub fn from_desc(desc: &FilterDesc) -> Self {
        Self {
            id: desc.id,
            name: desc.name.clone(),
            flags: desc.flags,
            client_data: desc.client_data.clone(),
        }
    }

    fn builtin(id: u16, name: &str) -> Self {
        Self {
            id,
            name: Some(name.to_string()),
            flags: 0,
            client_data: Vec::new(),
        }
    }
}

impl FilterRegistry {
    pub fn init_package() -> Self {
        let mut registry = Self::default();
        for filter in [
            RegisteredFilter::builtin(FILTER_DEFLATE, "deflate"),
            RegisteredFilter::builtin(FILTER_SHUFFLE, "shuffle"),
            RegisteredFilter::builtin(FILTER_FLETCHER32, "fletcher32"),
            RegisteredFilter::builtin(FILTER_SZIP, "szip"),
            RegisteredFilter::builtin(FILTER_NBIT, "nbit"),
            RegisteredFilter::builtin(FILTER_SCALEOFFSET, "scaleoffset"),
            RegisteredFilter::builtin(FILTER_LZF, "lzf"),
            RegisteredFilter::builtin(FILTER_BLOSC, "blosc"),
        ] {
            registry.filters.insert(filter.id, filter);
        }
        registry
    }

    pub fn term_package(self) {}

    pub fn register(&mut self, filter: RegisteredFilter) -> Result<()> {
        self.register_internal(filter)
    }

    pub fn register_internal(&mut self, filter: RegisteredFilter) -> Result<()> {
        if filter.id == 0 {
            return Err(Error::InvalidFormat("filter id 0 is reserved".into()));
        }
        self.filters.insert(filter.id, filter);
        Ok(())
    }

    pub fn unregister(&mut self, filter_id: u16) -> Result<RegisteredFilter> {
        self.unregister_internal(filter_id)
    }

    pub fn unregister_internal(&mut self, filter_id: u16) -> Result<RegisteredFilter> {
        self.filters
            .remove(&filter_id)
            .ok_or_else(|| Error::InvalidFormat(format!("filter {filter_id} is not registered")))
    }

    pub fn check_unregister(&self, filter_id: u16, pipeline: &FilterPipelineMessage) -> bool {
        !pipeline.filters.iter().any(|filter| filter.id == filter_id)
    }

    pub fn check_unregister_dset_cb(
        &self,
        filter_id: u16,
        pipeline: &FilterPipelineMessage,
    ) -> bool {
        self.check_unregister(filter_id, pipeline)
    }

    pub fn filter_avail(&self, filter_id: u16) -> bool {
        self.filter_avail_internal(filter_id)
    }

    pub fn filter_avail_internal(&self, filter_id: u16) -> bool {
        self.filters.contains_key(&filter_id)
    }

    pub fn prepare_prelude_callback_dcpl(
        &self,
        pipeline: &FilterPipelineMessage,
    ) -> Vec<FilterInfo> {
        let mut infos = Vec::new();
        self.prepare_prelude_callback_dcpl_into(pipeline, &mut infos);
        infos
    }

    pub fn prepare_prelude_callback_dcpl_into(
        &self,
        pipeline: &FilterPipelineMessage,
        out: &mut Vec<FilterInfo>,
    ) {
        out.clear();
        out.reserve(pipeline.filters.len());
        out.extend(pipeline.filters.iter().map(|filter| FilterInfo {
            id: filter.id,
            flags: filter.flags,
            client_data_count: filter.client_data.len(),
        }));
    }

    pub fn set_local_direct(
        &self,
        pipeline: &mut FilterPipelineMessage,
        filter_id: u16,
        client_data: Vec<u32>,
    ) -> Result<()> {
        let index = Self::find_idx(pipeline, filter_id)
            .ok_or_else(|| Error::InvalidFormat(format!("filter {filter_id} not in pipeline")))?;
        pipeline.filters[index].client_data = client_data;
        Ok(())
    }

    pub fn ignore_filters(pipeline: &mut FilterPipelineMessage) {
        pipeline.filters.clear();
    }

    pub fn modify(
        pipeline: &mut FilterPipelineMessage,
        filter_id: u16,
        flags: u16,
        client_data: Vec<u32>,
    ) -> Result<()> {
        let index = Self::find_idx(pipeline, filter_id)
            .ok_or_else(|| Error::InvalidFormat(format!("filter {filter_id} not in pipeline")))?;
        let filter = &mut pipeline.filters[index];
        filter.flags = flags;
        filter.client_data = client_data;
        Ok(())
    }

    pub fn append(pipeline: &mut FilterPipelineMessage, filter: FilterDesc) -> Result<()> {
        if pipeline.filters.len() >= 32 {
            return Err(Error::InvalidFormat(
                "filter pipeline has too many filters".into(),
            ));
        }
        pipeline.filters.push(filter);
        Ok(())
    }

    pub fn find_idx(pipeline: &FilterPipelineMessage, filter_id: u16) -> Option<usize> {
        pipeline
            .filters
            .iter()
            .position(|filter| filter.id == filter_id)
    }

    pub fn filter_info(pipeline: &FilterPipelineMessage, filter_id: u16) -> Option<FilterInfo> {
        let filter = pipeline
            .filters
            .iter()
            .find(|filter| filter.id == filter_id)?;
        Some(FilterInfo {
            id: filter.id,
            flags: filter.flags,
            client_data_count: filter.client_data.len(),
        })
    }

    pub fn all_filters_avail(&self, pipeline: &FilterPipelineMessage) -> bool {
        pipeline
            .filters
            .iter()
            .all(|filter| self.filter_avail(filter.id))
    }

    pub fn delete(pipeline: &mut FilterPipelineMessage, filter_id: u16) -> Result<FilterDesc> {
        let index = Self::find_idx(pipeline, filter_id)
            .ok_or_else(|| Error::InvalidFormat(format!("filter {filter_id} not in pipeline")))?;
        Ok(pipeline.filters.remove(index))
    }
}

impl XformExpr {
    pub fn xform_destroy_parse_tree(self) {}

    pub fn xform_parse(input: &str) -> Result<Self> {
        let mut parser = Parser::new(input);
        let expr = parser.parse_expression()?;
        parser.skip_ws();
        if parser.peek().is_some() {
            return Err(Error::InvalidFormat(format!(
                "unexpected transform expression suffix at byte {}",
                parser.pos
            )));
        }
        Ok(expr)
    }

    pub fn parse_expression(input: &str) -> Result<Self> {
        Self::xform_parse(input)
    }

    pub fn parse_term(input: &str) -> Result<Self> {
        let mut parser = Parser::new(input);
        parser.parse_term()
    }

    pub fn parse_factor(input: &str) -> Result<Self> {
        let mut parser = Parser::new(input);
        parser.parse_factor()
    }

    pub fn new_node(op: XformOp, left: XformExpr, right: XformExpr) -> Self {
        XformExpr::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    pub fn xform_eval_full(&self, x: f64) -> Result<f64> {
        match self {
            XformExpr::Number(value) => Ok(*value),
            XformExpr::Variable => Ok(x),
            XformExpr::UnaryMinus(expr) => Ok(-expr.xform_eval_full(x)?),
            XformExpr::Binary { op, left, right } => {
                let left = left.xform_eval_full(x)?;
                let right = right.xform_eval_full(x)?;
                match op {
                    XformOp::Add => Ok(left + right),
                    XformOp::Sub => Ok(left - right),
                    XformOp::Mul => Ok(left * right),
                    XformOp::Div => {
                        if right == 0.0 {
                            Err(Error::InvalidFormat(
                                "division by zero in transform expression".into(),
                            ))
                        } else {
                            Ok(left / right)
                        }
                    }
                }
            }
        }
    }

    pub fn xform_find_type(&self) -> XformNodeType {
        match self {
            XformExpr::Number(_) => XformNodeType::Number,
            XformExpr::Variable => XformNodeType::Variable,
            XformExpr::UnaryMinus(_) => XformNodeType::Unary,
            XformExpr::Binary { .. } => XformNodeType::Binary,
        }
    }

    pub fn op_is_numbs(&self) -> bool {
        matches!(self, XformExpr::Number(_))
    }

    pub fn op_is_numbs2(left: &XformExpr, right: &XformExpr) -> bool {
        left.op_is_numbs() && right.op_is_numbs()
    }

    pub fn xform_reduce_tree(self) -> Result<Self> {
        match self {
            XformExpr::UnaryMinus(expr) => {
                let expr = expr.xform_reduce_tree()?;
                if let XformExpr::Number(value) = expr {
                    Ok(XformExpr::Number(-value))
                } else {
                    Ok(XformExpr::UnaryMinus(Box::new(expr)))
                }
            }
            XformExpr::Binary { op, left, right } => {
                let left = left.xform_reduce_tree()?;
                let right = right.xform_reduce_tree()?;
                if Self::op_is_numbs2(&left, &right) {
                    let folded = XformExpr::new_node(op, left, right).xform_eval_full(0.0)?;
                    Ok(XformExpr::Number(folded))
                } else {
                    Ok(XformExpr::new_node(op, left, right))
                }
            }
            expr => Ok(expr),
        }
    }

    pub fn xform_destroy(self) {}

    pub fn xform_copy(&self) -> Self {
        self.clone()
    }

    pub fn xform_noop(&self) -> bool {
        matches!(self, XformExpr::Variable)
    }
}

pub fn get_token(input: &str, pos: &mut usize) -> Result<Option<XformToken>> {
    let bytes = input.as_bytes();
    while matches!(bytes.get(*pos), Some(c) if c.is_ascii_whitespace()) {
        *pos += 1;
    }
    let Some(&c) = bytes.get(*pos) else {
        return Ok(None);
    };
    match c {
        b'x' | b'X' => {
            *pos += 1;
            Ok(Some(XformToken::Variable))
        }
        b'+' | b'-' | b'*' | b'/' => {
            *pos += 1;
            Ok(Some(XformToken::Operator(c as char)))
        }
        b'(' => {
            *pos += 1;
            Ok(Some(XformToken::LParen))
        }
        b')' => {
            *pos += 1;
            Ok(Some(XformToken::RParen))
        }
        c if c.is_ascii_digit() || c == b'.' => {
            let start = *pos;
            let mut parser = Parser {
                input: bytes,
                pos: *pos,
            };
            let token = match parser.parse_number()? {
                XformExpr::Number(value) => XformToken::Number(value),
                _ => unreachable!(),
            };
            if parser.pos == start {
                return Err(Error::InvalidFormat(
                    "empty transform expression number".into(),
                ));
            }
            *pos = parser.pos;
            Ok(Some(token))
        }
        _ => Err(Error::InvalidFormat(format!(
            "unknown transform token at byte {pos}",
            pos = *pos
        ))),
    }
}

pub fn unget_token(pos: &mut usize, previous_pos: usize) {
    *pos = previous_pos;
}

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    fn parse_expression(&mut self) -> Result<XformExpr> {
        let mut expr = self.parse_term()?;
        loop {
            self.skip_ws();
            let Some(op) = self.consume_op(b"+-") else {
                break;
            };
            let right = self.parse_term()?;
            expr = XformExpr::new_node(op, expr, right);
        }
        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<XformExpr> {
        let mut expr = self.parse_factor()?;
        loop {
            self.skip_ws();
            let Some(op) = self.consume_op(b"*/") else {
                break;
            };
            let right = self.parse_factor()?;
            expr = XformExpr::new_node(op, expr, right);
        }
        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<XformExpr> {
        self.skip_ws();
        match self.peek() {
            Some(b'x') | Some(b'X') => {
                self.pos += 1;
                Ok(XformExpr::Variable)
            }
            Some(b'-') => {
                self.pos += 1;
                Ok(XformExpr::UnaryMinus(Box::new(self.parse_factor()?)))
            }
            Some(b'(') => {
                self.pos += 1;
                let expr = self.parse_expression()?;
                self.skip_ws();
                if self.peek() != Some(b')') {
                    return Err(Error::InvalidFormat(
                        "missing closing parenthesis in transform expression".into(),
                    ));
                }
                self.pos += 1;
                Ok(expr)
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => self.parse_number(),
            _ => Err(Error::InvalidFormat(format!(
                "expected transform expression factor at byte {}",
                self.pos
            ))),
        }
    }

    fn parse_number(&mut self) -> Result<XformExpr> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == b'.' || c == b'e' || c == b'E' || c == b'+' || c == b'-')
        {
            if self.pos != start && matches!(self.input[self.pos], b'+' | b'-') {
                let prev = self.input[self.pos - 1];
                if prev != b'e' && prev != b'E' {
                    break;
                }
            }
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.input[start..self.pos])
            .map_err(|e| Error::InvalidFormat(format!("invalid transform number: {e}")))?;
        let value = text
            .parse::<f64>()
            .map_err(|e| Error::InvalidFormat(format!("invalid transform number {text}: {e}")))?;
        Ok(XformExpr::Number(value))
    }

    fn consume_op(&mut self, ops: &[u8]) -> Option<XformOp> {
        let op = self.peek()?;
        if !ops.contains(&op) {
            return None;
        }
        self.pos += 1;
        match op {
            b'+' => Some(XformOp::Add),
            b'-' => Some(XformOp::Sub),
            b'*' => Some(XformOp::Mul),
            b'/' => Some(XformOp::Div),
            _ => None,
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_ascii_whitespace()) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pipeline() -> FilterPipelineMessage {
        FilterPipelineMessage {
            version: 2,
            filters: vec![FilterDesc {
                id: FILTER_DEFLATE,
                name: Some("deflate".into()),
                flags: 0,
                client_data: vec![6],
            }],
        }
    }

    #[test]
    fn registry_reports_builtin_filter_availability() {
        let registry = FilterRegistry::init_package();
        assert!(registry.filter_avail(FILTER_DEFLATE));
        assert!(registry.filter_avail(FILTER_SHUFFLE));
        assert!(!registry.filter_avail(32_099));
    }

    #[test]
    fn registry_prelude_callback_reuses_caller_buffer() {
        let registry = FilterRegistry::init_package();
        let pipeline = pipeline();
        let mut infos = vec![FilterInfo {
            id: 0,
            flags: 0,
            client_data_count: 0,
        }];
        registry.prepare_prelude_callback_dcpl_into(&pipeline, &mut infos);
        assert_eq!(
            infos,
            vec![FilterInfo {
                id: FILTER_DEFLATE,
                flags: 0,
                client_data_count: 1,
            }]
        );
        assert_eq!(registry.prepare_prelude_callback_dcpl(&pipeline), infos);
    }

    #[test]
    fn registry_mutates_pipeline_entries_by_filter_id() {
        let mut pipeline = pipeline();
        FilterRegistry::modify(&mut pipeline, FILTER_DEFLATE, 1, vec![9]).unwrap();
        assert_eq!(pipeline.filters[0].flags, 1);
        assert_eq!(pipeline.filters[0].client_data, vec![9]);
        let removed = FilterRegistry::delete(&mut pipeline, FILTER_DEFLATE).unwrap();
        assert_eq!(removed.id, FILTER_DEFLATE);
        assert!(pipeline.filters.is_empty());
    }

    #[test]
    fn xform_parses_evaluates_and_reduces() {
        let expr = XformExpr::xform_parse("x * 2 + 3").unwrap();
        assert_eq!(expr.xform_eval_full(4.0).unwrap(), 11.0);
        assert!(!expr.xform_noop());

        let reduced = XformExpr::xform_parse("1 + 2 * 3")
            .unwrap()
            .xform_reduce_tree()
            .unwrap();
        assert_eq!(reduced, XformExpr::Number(7.0));
    }
}
