use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Write};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Int64Array, LargeStringArray, RecordBatch, StringArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{ArrowError, DataType, Field, Schema, TimeUnit};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::file::properties::WriterProperties;

const F_TOB: u8 = 64;
const SNAPSHOT: u8 = 32;
pub const UNDEF_PRICE: i64 = 9_223_372_036_854_775_807;
pub const BATCH: usize = 65_536;

#[derive(Clone, Copy)]
struct Order {
    side: u8,
    price: i64,
    size: u32,
}

#[derive(Default, Clone, Copy)]
struct Level {
    size: u64,
    count: u32,
}

#[derive(Default)]
struct Book {
    orders: HashMap<u64, Order>,
    bids: BTreeMap<i64, Level>,
    asks: BTreeMap<i64, Level>,
}

#[derive(Default, Clone)]
pub struct Stats {
    pub rows: u64,
    pub emitted: u64,
    pub adds: u64,
    pub cancels: u64,
    pub modifies: u64,
    pub clears: u64,
    pub trades_or_fills: u64,
    pub missing_cancels: u64,
    pub missing_modifies: u64,
    pub duplicate_adds: u64,
    pub bad_sides: u64,
    pub unknown_actions: u64,
}

#[derive(Clone, Copy)]
pub struct Row {
    pub ts_event: i64,
    pub row_nr: u64,
    pub sequence: u32,
    pub publisher_id: u16,
    pub instrument_id: u32,
    pub channel_id: u8,
    pub action: u8,
    pub side: u8,
    pub price: i64,
    pub size: u32,
    pub order_id: u64,
    pub flags: u8,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct TopLevel {
    price: i64,
    size: u64,
    count: u32,
}

#[derive(Clone, PartialEq, Eq)]
struct Top {
    bids: Vec<TopLevel>,
    asks: Vec<TopLevel>,
}

pub fn depth_schema(levels: usize) -> Arc<Schema> {
    Arc::new(schema(levels))
}

pub fn read_rows_parquet(path: &str) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    const COLS: &[&str] = &[
        "ts_event",
        "sequence",
        "publisher_id",
        "instrument_id",
        "channel_id",
        "action",
        "side",
        "price",
        "size",
        "order_id",
        "flags",
    ];
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let mask = ProjectionMask::columns(builder.parquet_schema(), COLS.iter().copied());
    rows_from_batches(
        builder
            .with_projection(mask)
            .with_batch_size(262_144)
            .build()?,
    )
}

pub fn read_rows_ipc<R: Read>(reader: R) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    rows_from_batches(StreamReader::try_new(reader, None)?)
}

fn rows_from_batches<I>(batches: I) -> Result<Vec<Row>, Box<dyn std::error::Error>>
where
    I: IntoIterator<Item = Result<RecordBatch, ArrowError>>,
{
    let mut rows = Vec::new();
    let mut row_nr = 0_u64;
    for batch in batches {
        let batch = batch?;
        let c = InCols::new(&batch)?;
        for i in 0..batch.num_rows() {
            rows.push(Row {
                ts_event: timestamp_ns_at(c.ts_event, i)?,
                row_nr,
                sequence: c.sequence.value(i),
                publisher_id: c.publisher_id.value(i),
                instrument_id: c.instrument_id.value(i),
                channel_id: c.channel_id.value(i),
                action: byte_at(c.action, i)?,
                side: byte_at(c.side, i)?,
                price: c.price.value(i),
                size: c.size.value(i),
                order_id: c.order_id.value(i),
                flags: c.flags.value(i),
            });
            row_nr += 1;
        }
    }
    Ok(rows)
}

pub fn sort_rows(rows: &mut [Row]) {
    rows.sort_unstable_by_key(|r| {
        let snapshot = r.flags & SNAPSHOT != 0;
        (
            !snapshot,
            if snapshot { 0 } else { r.publisher_id },
            if snapshot { 0 } else { r.instrument_id },
            if snapshot { 0 } else { r.channel_id },
            if snapshot { 0 } else { r.sequence },
            r.row_nr,
        )
    });
}

pub fn write_depth_parquet(
    rows: Vec<Row>,
    levels: usize,
    out: &str,
) -> Result<Stats, Box<dyn std::error::Error>> {
    let mut batches = BookBatchGenerator::new(rows.into_iter(), levels);
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(File::create(out)?, batches.schema(), Some(props))?;
    for batch in batches.by_ref() {
        writer.write(&batch?)?;
    }
    writer.close()?;
    Ok(batches.stats().clone())
}

pub fn write_depth_ipc<W: Write>(
    rows: Vec<Row>,
    levels: usize,
    out: W,
) -> Result<Stats, Box<dyn std::error::Error>> {
    let mut batches = BookBatchGenerator::new(rows.into_iter(), levels);
    let mut writer = StreamWriter::try_new(out, &batches.schema())?;
    for batch in batches.by_ref() {
        writer.write(&batch?)?;
    }
    writer.finish()?;
    Ok(batches.stats().clone())
}

pub struct BookBatchGenerator<I> {
    rows: I,
    levels: usize,
    schema: Arc<Schema>,
    buf: OutBuffer,
    books: HashMap<(u16, u32), Book>,
    stats: Stats,
    done: bool,
}

impl<I: Iterator<Item = Row>> BookBatchGenerator<I> {
    pub fn new(rows: I, levels: usize) -> Self {
        let schema = depth_schema(levels);
        Self {
            rows,
            levels,
            schema: schema.clone(),
            buf: OutBuffer::new(levels, schema),
            books: HashMap::new(),
            stats: Stats::default(),
            done: false,
        }
    }

    pub fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        if self.done {
            return Ok(None);
        }
        while let Some(row) = self.rows.next() {
            self.push_row(row);
            if self.buf.len() >= BATCH {
                return self.buf.take_batch();
            }
        }
        self.done = true;
        self.buf.take_batch()
    }

    fn push_row(&mut self, row: Row) {
        self.stats.rows += 1;
        let is_trade = row.action == b'T';
        let top = {
            let book = self
                .books
                .entry((row.publisher_id, row.instrument_id))
                .or_default();
            let before = if may_change(row.action) {
                Some(book.top(self.levels))
            } else {
                None
            };
            if apply(book, row, &mut self.stats) {
                let after = book.top(self.levels);
                (before.as_ref() != Some(&after)).then_some((after, false))
            } else if is_trade {
                Some((book.top(self.levels), true))
            } else {
                None
            }
        };
        if let Some((top, is_trade)) = top {
            self.buf.push(row, &top, is_trade);
            self.stats.emitted += 1;
        }
    }
}

impl<I: Iterator<Item = Row>> Iterator for BookBatchGenerator<I> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch().transpose()
    }
}

fn apply(book: &mut Book, row: Row, stats: &mut Stats) -> bool {
    match row.action {
        b'T' | b'F' | b'N' => {
            stats.trades_or_fills += 1;
            false
        }
        b'R' => {
            stats.clears += 1;
            book.orders.clear();
            book.bids.clear();
            book.asks.clear();
            true
        }
        b'A' => {
            stats.adds += 1;
            if !valid_side(row.side) {
                stats.bad_sides += 1;
                return false;
            }
            if row.price == UNDEF_PRICE && row.flags & F_TOB != 0 {
                side_levels_mut(book, row.side).clear();
                return true;
            }
            if let Some(old) = book.orders.remove(&row.order_id) {
                stats.duplicate_adds += 1;
                remove_level_qty(book, old.side, old.price, old.size, true);
            }
            add_order(book, row.order_id, row.side, row.price, row.size);
            true
        }
        b'C' => {
            stats.cancels += 1;
            let Some(mut order) = book.orders.get(&row.order_id).copied() else {
                stats.missing_cancels += 1;
                return false;
            };
            let cancel_size = row.size.min(order.size);
            remove_level_qty(book, order.side, order.price, cancel_size, false);
            order.size -= cancel_size;
            if order.size == 0 {
                remove_level_qty(book, order.side, order.price, 0, true);
                book.orders.remove(&row.order_id);
            } else {
                book.orders.insert(row.order_id, order);
            }
            true
        }
        b'M' => {
            stats.modifies += 1;
            if !valid_side(row.side) {
                stats.bad_sides += 1;
                return false;
            }
            if let Some(old) = book.orders.remove(&row.order_id) {
                remove_level_qty(book, old.side, old.price, old.size, true);
            } else {
                stats.missing_modifies += 1;
            }
            add_order(book, row.order_id, row.side, row.price, row.size);
            true
        }
        _ => {
            stats.unknown_actions += 1;
            false
        }
    }
}

impl Book {
    fn top(&self, n: usize) -> Top {
        Top {
            bids: top_side(self.bids.iter().rev(), n),
            asks: top_side(self.asks.iter(), n),
        }
    }
}

fn top_side<'a, I>(iter: I, n: usize) -> Vec<TopLevel>
where
    I: Iterator<Item = (&'a i64, &'a Level)>,
{
    let mut out: Vec<_> = iter
        .take(n)
        .map(|(price, level)| TopLevel {
            price: *price,
            size: level.size,
            count: level.count,
        })
        .collect();
    out.resize(
        n,
        TopLevel {
            price: UNDEF_PRICE,
            size: 0,
            count: 0,
        },
    );
    out
}

fn add_order(book: &mut Book, order_id: u64, side: u8, price: i64, size: u32) {
    if order_id == 0 || price == UNDEF_PRICE {
        return;
    }
    book.orders.insert(order_id, Order { side, price, size });
    let level = side_levels_mut(book, side).entry(price).or_default();
    level.size += size as u64;
    level.count += 1;
}

fn remove_level_qty(book: &mut Book, side: u8, price: i64, size: u32, remove_count: bool) {
    let levels = side_levels_mut(book, side);
    let Some(level) = levels.get_mut(&price) else {
        return;
    };
    level.size = level.size.saturating_sub(size as u64);
    if remove_count {
        level.count = level.count.saturating_sub(1);
    }
    if level.size == 0 || level.count == 0 {
        levels.remove(&price);
    }
}

fn side_levels_mut(book: &mut Book, side: u8) -> &mut BTreeMap<i64, Level> {
    if side == b'B' {
        &mut book.bids
    } else {
        &mut book.asks
    }
}

fn may_change(action: u8) -> bool {
    matches!(action, b'A' | b'C' | b'M' | b'R')
}

fn valid_side(side: u8) -> bool {
    side == b'A' || side == b'B'
}

struct OutBuffer {
    schema: Arc<Schema>,
    levels: usize,
    ts_event: Vec<i64>,
    row_nr: Vec<u64>,
    sequence: Vec<u32>,
    publisher_id: Vec<u16>,
    instrument_id: Vec<u32>,
    trade_px: Vec<Option<i64>>,
    trade_sz: Vec<Option<u32>>,
    trade_side: Vec<Option<u8>>,
    bid_px: Vec<Vec<i64>>,
    bid_sz: Vec<Vec<u64>>,
    bid_ct: Vec<Vec<u32>>,
    ask_px: Vec<Vec<i64>>,
    ask_sz: Vec<Vec<u64>>,
    ask_ct: Vec<Vec<u32>>,
}

impl OutBuffer {
    fn new(levels: usize, schema: Arc<Schema>) -> Self {
        Self {
            schema,
            levels,
            ts_event: Vec::with_capacity(BATCH),
            row_nr: Vec::with_capacity(BATCH),
            sequence: Vec::with_capacity(BATCH),
            publisher_id: Vec::with_capacity(BATCH),
            instrument_id: Vec::with_capacity(BATCH),
            trade_px: Vec::with_capacity(BATCH),
            trade_sz: Vec::with_capacity(BATCH),
            trade_side: Vec::with_capacity(BATCH),
            bid_px: vec_cols(levels),
            bid_sz: vec_cols(levels),
            bid_ct: vec_cols(levels),
            ask_px: vec_cols(levels),
            ask_sz: vec_cols(levels),
            ask_ct: vec_cols(levels),
        }
    }

    fn len(&self) -> usize {
        self.row_nr.len()
    }

    fn push(&mut self, row: Row, top: &Top, is_trade: bool) {
        self.ts_event.push(row.ts_event);
        self.row_nr.push(row.row_nr);
        self.sequence.push(row.sequence);
        self.publisher_id.push(row.publisher_id);
        self.instrument_id.push(row.instrument_id);
        self.trade_px
            .push((is_trade && row.price != UNDEF_PRICE).then_some(row.price));
        self.trade_sz.push(is_trade.then_some(row.size));
        self.trade_side
            .push(if is_trade { side_code(row.side) } else { None });
        for i in 0..self.levels {
            self.bid_px[i].push(top.bids[i].price);
            self.bid_sz[i].push(top.bids[i].size);
            self.bid_ct[i].push(top.bids[i].count);
            self.ask_px[i].push(top.asks[i].price);
            self.ask_sz[i].push(top.asks[i].size);
            self.ask_ct[i].push(top.asks[i].count);
        }
    }

    fn take_batch(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        if self.len() == 0 {
            return Ok(None);
        }
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(
                TimestampNanosecondArray::from(std::mem::take(&mut self.ts_event))
                    .with_timezone("UTC"),
            ),
            Arc::new(UInt64Array::from(std::mem::take(&mut self.row_nr))),
            Arc::new(UInt32Array::from(std::mem::take(&mut self.sequence))),
            Arc::new(UInt16Array::from(std::mem::take(&mut self.publisher_id))),
            Arc::new(UInt32Array::from(std::mem::take(&mut self.instrument_id))),
            Arc::new(Int64Array::from(std::mem::take(&mut self.trade_px))),
            Arc::new(UInt32Array::from(std::mem::take(&mut self.trade_sz))),
            Arc::new(UInt8Array::from(std::mem::take(&mut self.trade_side))),
        ];
        for i in 0..self.levels {
            cols.push(Arc::new(Int64Array::from(std::mem::take(
                &mut self.bid_px[i],
            ))));
            cols.push(Arc::new(UInt64Array::from(std::mem::take(
                &mut self.bid_sz[i],
            ))));
            cols.push(Arc::new(UInt32Array::from(std::mem::take(
                &mut self.bid_ct[i],
            ))));
            cols.push(Arc::new(Int64Array::from(std::mem::take(
                &mut self.ask_px[i],
            ))));
            cols.push(Arc::new(UInt64Array::from(std::mem::take(
                &mut self.ask_sz[i],
            ))));
            cols.push(Arc::new(UInt32Array::from(std::mem::take(
                &mut self.ask_ct[i],
            ))));
        }
        let batch = RecordBatch::try_new(self.schema.clone(), cols)?;
        self.reset_caps();
        Ok(Some(batch))
    }

    fn reset_caps(&mut self) {
        self.ts_event = Vec::with_capacity(BATCH);
        self.row_nr = Vec::with_capacity(BATCH);
        self.sequence = Vec::with_capacity(BATCH);
        self.publisher_id = Vec::with_capacity(BATCH);
        self.instrument_id = Vec::with_capacity(BATCH);
        self.trade_px = Vec::with_capacity(BATCH);
        self.trade_sz = Vec::with_capacity(BATCH);
        self.trade_side = Vec::with_capacity(BATCH);
        reserve_cols(&mut self.bid_px);
        reserve_cols(&mut self.bid_sz);
        reserve_cols(&mut self.bid_ct);
        reserve_cols(&mut self.ask_px);
        reserve_cols(&mut self.ask_sz);
        reserve_cols(&mut self.ask_ct);
    }
}

fn vec_cols<T>(levels: usize) -> Vec<Vec<T>> {
    (0..levels).map(|_| Vec::with_capacity(BATCH)).collect()
}

fn reserve_cols<T>(cols: &mut [Vec<T>]) {
    for col in cols {
        col.reserve(BATCH);
    }
}

fn schema(levels: usize) -> Schema {
    let mut fields = vec![
        Field::new(
            "ts_event",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            false,
        ),
        Field::new("row_nr", DataType::UInt64, false),
        Field::new("sequence", DataType::UInt32, false),
        Field::new("publisher_id", DataType::UInt16, false),
        Field::new("instrument_id", DataType::UInt32, false),
        Field::new("trade_px", DataType::Int64, true),
        Field::new("trade_sz", DataType::UInt32, true),
        Field::new("trade_side", DataType::UInt8, true),
    ];
    for i in 0..levels {
        fields.push(Field::new(format!("bid_px_{i}"), DataType::Int64, false));
        fields.push(Field::new(format!("bid_sz_{i}"), DataType::UInt64, false));
        fields.push(Field::new(format!("bid_ct_{i}"), DataType::UInt32, false));
        fields.push(Field::new(format!("ask_px_{i}"), DataType::Int64, false));
        fields.push(Field::new(format!("ask_sz_{i}"), DataType::UInt64, false));
        fields.push(Field::new(format!("ask_ct_{i}"), DataType::UInt32, false));
    }
    Schema::new(fields)
}

struct InCols<'a> {
    ts_event: &'a dyn Array,
    sequence: &'a UInt32Array,
    publisher_id: &'a UInt16Array,
    instrument_id: &'a UInt32Array,
    channel_id: &'a UInt8Array,
    action: &'a dyn Array,
    side: &'a dyn Array,
    price: &'a Int64Array,
    size: &'a UInt32Array,
    order_id: &'a UInt64Array,
    flags: &'a UInt8Array,
}

impl<'a> InCols<'a> {
    fn new(batch: &'a RecordBatch) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            ts_event: batch.column(batch.schema().index_of("ts_event")?).as_ref(),
            sequence: col(batch, "sequence")?,
            publisher_id: col(batch, "publisher_id")?,
            instrument_id: col(batch, "instrument_id")?,
            channel_id: col(batch, "channel_id")?,
            action: batch.column(batch.schema().index_of("action")?).as_ref(),
            side: batch.column(batch.schema().index_of("side")?).as_ref(),
            price: col(batch, "price")?,
            size: col(batch, "size")?,
            order_id: col(batch, "order_id")?,
            flags: col(batch, "flags")?,
        })
    }
}

fn col<'a, T: 'static>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a T, Box<dyn std::error::Error>> {
    let i = batch.schema().index_of(name)?;
    batch.column(i).as_any().downcast_ref::<T>().ok_or_else(|| {
        format!(
            "column {name} has unexpected type {:?}",
            batch.schema().field(i).data_type()
        )
        .into()
    })
}

fn byte_at(array: &dyn Array, i: usize) -> Result<u8, Box<dyn std::error::Error>> {
    if array.is_null(i) {
        return Err("null action/side".into());
    }
    if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
        return a
            .value(i)
            .as_bytes()
            .first()
            .copied()
            .ok_or_else(|| "empty action/side".into());
    }
    if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
        return a
            .value(i)
            .as_bytes()
            .first()
            .copied()
            .ok_or_else(|| "empty action/side".into());
    }
    Err(format!("expected string action/side, got {:?}", array.data_type()).into())
}

fn timestamp_ns_at(array: &dyn Array, i: usize) -> Result<i64, Box<dyn std::error::Error>> {
    if array.is_null(i) {
        return Err("null ts_event".into());
    }
    if let Some(a) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok(a.value(i));
    }
    if let Some(a) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return Ok(a.value(i) * 1_000);
    }
    if let Some(a) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return Ok(a.value(i) * 1_000_000);
    }
    if let Some(a) = array.as_any().downcast_ref::<TimestampSecondArray>() {
        return Ok(a.value(i) * 1_000_000_000);
    }
    Err(format!("expected timestamp ts_event, got {:?}", array.data_type()).into())
}

fn side_code(side: u8) -> Option<u8> {
    match side {
        b'B' => Some(0),
        b'A' => Some(1),
        _ => None,
    }
}
